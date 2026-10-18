/*
 * Watashinosekaito Hatsune Miku
 * Copyright (c) 2025-2026 ChordDotE. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-ChordDotE-NonCommercial-2.0
 *
 * NON-COMMERCIAL USE ONLY. NO BUSINESS USE. NO ADAPTATION.
 *
 * Business use (any use by or for a company, organization, or sole
 * proprietor, WHETHER OR NOT ANY PAYMENT IS INVOLVED) and publishing or
 * distributing modified versions are PROHIBITED IN ALL CASES; no permission
 * is granted for them. Forks may be used only to propose improvements to the
 * Author (e.g. pull requests). Commercial use, redistribution, and Adaptation
 * are PROHIBITED without prior written permission. "Adaptation" includes
 * manual reimplementation, automated translation, and generation using large
 * language models or other machine learning, of software having equivalent or
 * similar functionality. Permission is granted individually, in recorded form.
 * Contact: https://x.com/ChordDotE
 *
 * 業務利用（対価の有無を問わず、法人・団体・個人事業主による利用と事業のための利用）と、
 * 改変版の公開・配布は、許可の余地なく禁止します。フォークは作者への改善の提案
 * （プルリクエスト等）のためだけに使えます。商用利用・再配布・翻案は、作者の許可が
 * ない限り禁止します。翻案には、再実装・自動変換・大規模言語モデル等を用いた生成に
 * よる、同等または類似の機能を持つソフトウェアの作成を含みます。
 * 許可が必要な場合は、記録の残る形でご連絡ください。
 *
 * See LICENSE for full terms. 詳細は LICENSE を参照してください。
 */

/**
 * miku-chat.js - チャット入力コンポーネント（Web Component ホスト）
 *
 * MikuChat は薄いオーケストレーター。実処理は同ディレクトリの内部クラスに分割:
 *   cameraManager.js           カメラ制御・バッファ管理・プレビュー吹き出し
 *   sttController.js           STTリアルタイム表示・auto-sendカウントダウン
 *   idleAutoSendController.js  STTオフ時の入力アイドル自動送信（SttController と排他）
 *   microphoneUI.js            マイクボタンの状態表示・レベルメーター
 * 各クラスは MikuChat の Shadow DOM を引数に取る内部部品（単独では動かない）。
 *
 * モジュール間通信は CustomEvent（modules/events.js の定数）経由のみ。
 */

import { EVENTS, SOCKET_EVENTS } from '../../modules/events.js';
import { getSessionId }           from '../../modules/session.js';
import { getSocket }              from '../../modules/socket.js';
import { UI_STRINGS }             from '../../modules/uiStrings.js';
import { MicrophoneController, MicState } from '../../modules/microphone.js';
import { appState }               from '../../modules/appState.js';
import { turnState, TurnBelief } from '../../modules/turnState.js';
import { compressVideoFile }      from '../../modules/videoCompressor.js';
import { convertAudioForEnroll }  from '../../modules/audioConverter.js';
import { CameraManager }          from './cameraManager.js';
import { SttController }          from './sttController.js';
import { IdleAutoSendController } from './idleAutoSendController.js';
import { MicrophoneUI }           from './microphoneUI.js';
import { MenuController }         from './menuController.js';
// =============================================================================
// MikuChat - Web Component ホスト（薄いオーケストレーター）
// =============================================================================

class MikuChat extends HTMLElement {
    constructor() {
        super();
        /** @type {MicrophoneController|null} */
        this._mic            = null;
        /** @type {string} 'OFF' | 'LISTEN' | 'AUTO' */
        this._micMode        = 'OFF';
        /** @type {boolean} サーバー設定でauto-send有効か */
        this._autoSendEnabled = false;
        /** @type {boolean} STTモデルがロード中か */
        this._sttModelLoading = false;
        /** @type {boolean} ロード完了後にマイクを自動スタートするか */
        this._micPendingStart = false;
        /** @type {boolean} 接続確認（app:connectionChecking）で自分がボタンを無効化したか */
        this._connCheckDisabled = false;
        // 会話ターン状態（考え中か・音が出ているか等）は modules/turnState.js に集約。
        // ボタン表示・マイクミュートは _applyTurnState() が状態から導出する（フラグは持たない）

        // 内部クラスのインスタンス（Shadow DOM 確立後に生成）
        /** @type {CameraManager|null}          */ this._camera = null;
        /** @type {SttController|null}          */ this._stt    = null;
        /** @type {MicrophoneUI|null}           */ this._micUI  = null;
        /** @type {IdleAutoSendController|null} */ this._idle   = null;
        /** @type {MenuController|null}         */ this._menu   = null;
    }

    async connectedCallback() {
        await this.loadTemplate();
        this.applyUIStrings();
        this.setupEventListeners();
        await this.initMicButton();
        await this.initCameraButton();
        this._menu = new MenuController(this.shadowRoot, {
            onCaptionToggle: (enabled) => this._closeCaptionTurnIfStuck(enabled),
            // 表示メニューのカメラ切替ボタン → CameraManager（カメラ制御はあちらの責務）
            onCameraSwitch: () => this._camera?.switchToNextCamera(),
        });
        console.log('miku-chat コンポーネントが初期化されました');
    }

    disconnectedCallback() {}

    // =========================================================================
    // 初期化
    // =========================================================================

    applyUIStrings() {
        const shadow = this.shadowRoot;
        if (!shadow) return;
        this._setSubmitButton(UI_STRINGS.submitButton.default, false);
        const textarea = shadow.getElementById('input_text');
        if (textarea) textarea.placeholder = UI_STRINGS.placeholder.input;
    }

    /**
     * 送信ボタンの表示状態を変更する（各イベントハンドラに分散していた同一処理の集約）
     * @param {string} label - ボタンに表示する文言
     * @param {boolean} disabled - 無効化するか
     * @param {boolean} [foldReady] - 考え中の畳み込み送信を受け付ける状態にするか。
     *   true にすると .fold-ready（半透明メインカラー・押せる）が付き、
     *   handleFormSubmit がこのクラスの有無で畳み込み分岐を判定する
     */
    _setSubmitButton(label, disabled, foldReady = false) {
        const button = this.shadowRoot?.getElementById('submitButton');
        if (!button) return;
        button.disabled = disabled;
        button.textContent = label;
        button.classList.toggle('fold-ready', foldReady);
        // カウントダウン表示（緑グラデーション背景）が残っていたら外す。
        // counting-down の background は :disabled の background-color より優先されるため、
        // 残ったままだと無効化しても灰色にならない
        button.classList.remove('counting-down');
    }

    /**
     * 会話ターン状態（turnState）からボタン表示とマイクミュートを導出する。
     * 状態×振る舞いの仕様は basic_design §4.6 を正とする。
     *
     * | turnBelief | audible | singing | ボタン               | マイク  |
     * |-----------|---------|---------|----------------------|---------|
     * | OPEN      | 任意    | -       | 考え中（追加OK）・有効 | audibleに従う |
     * | それ以外   | ○      | ○       | ♪歌ってる・**有効**（songMode が sing 以外は ♪流してる） | ミュート |
     * | それ以外   | ○      | ✕       | 返答中...・無効       | ミュート |
     * | それ以外   | ✕      | -       | 話す・有効            | 開      |
     *
     * 歌唱中にボタンを塞がないのは、**話しかけること自体が中断手段**だから
     * （マイクはミュートされるので、テキストで話しかける道を残す）。
     * 歌わずに曲を流しているとき（インスト＝play・作りかけ＝preview）は「♪ 流してる」（2026-10-03）。
     */
    _applyTurnState({ audible, singing, songMode, turnBelief, prev }) {
        // ボタン表示（接続確認中はそちらの表示を優先。復帰時に再適用される）
        if (!this._connCheckDisabled) {
            if (turnBelief === TurnBelief.OPEN) {
                this._setSubmitButton(UI_STRINGS.submitButton.thinking, false, true);
            } else if (singing) {
                this._setSubmitButton(songMode && songMode !== 'sing'
                    ? UI_STRINGS.submitButton.songPlaying : UI_STRINGS.submitButton.singing, false);
            } else if (audible) {
                this._setSubmitButton(UI_STRINGS.submitButton.playing, true);
            } else {
                this._setSubmitButton(UI_STRINGS.submitButton.default, false);
            }
        }

        // マイクボタンの「追加受付中」表示: 考え中（OPEN）の聞き取りは薄い緑にして
        // 「今話すと畳み込みになる」ことを示す（CSS は data-turn-open で分岐）
        const micBtn = this.shadowRoot?.getElementById('micButton');
        if (micBtn) {
            if (turnBelief === TurnBelief.OPEN) {
                micBtn.dataset.turnOpen = 'true';
            } else {
                delete micBtn.dataset.turnOpen;
            }
        }

        // マイクは audible の変化にのみ従う（半二重・回り込み対策。
        // ミュート方式: ストリームは保持し入力だけ捨てる → 解除がほぼ即時）
        if (audible && !prev.audible) {
            if (this._mic && this._mic.getState() !== MicState.OFF) {
                const socket = getSocket();
                if (socket) socket.emit(SOCKET_EVENTS.AUDIO_STREAM_STOP);
                this._mic.mute();
                console.log('[MikuChat] 音声再生開始 → マイクをミュート');
            }
        } else if (!audible && prev.audible) {
            // 再生中にユーザーが手動でマイクを切った場合（MUTED でない）は何もしない
            if (this._mic && this._mic.getState() === MicState.MUTED) {
                this._mic.unmute();
                this._stt?.resetUtterance();
                const socket = getSocket();
                if (socket) socket.emit(SOCKET_EVENTS.AUDIO_STREAM_START);
                console.log('[MikuChat] 音声再生完了 → マイクのミュート解除');
            }
        }
    }

    async loadTemplate() {
        try {
            const response = await fetch('/js/components/miku-chat/miku-chat.html');
            const html     = await response.text();
            const shadow   = this.attachShadow({ mode: 'open' });
            const parser   = new DOMParser();
            const doc      = parser.parseFromString(html, 'text/html');
            const template = doc.querySelector('template');
            if (template) {
                shadow.appendChild(template.content.cloneNode(true));
            } else {
                shadow.innerHTML = html;
            }
        } catch (error) {
            console.error('miku-chat テンプレート読み込みエラー:', error);
        }
    }

    setupEventListeners() {
        const shadow = this.shadowRoot;
        if (!shadow) return;

        // フォーム送信
        const form = shadow.getElementById('generateForm');
        if (form) form.addEventListener('submit', (e) => this.handleFormSubmit(e));

        // Ctrl+Enter で送信・フォーカスイベント
        const textarea = shadow.getElementById('input_text');
        if (textarea) {
            textarea.addEventListener('keydown', (e) => {
                if (e.ctrlKey && e.key === 'Enter') {
                    e.preventDefault();
                    shadow.getElementById('submitButton').click();
                }
            });
            textarea.addEventListener('focus', () =>
                document.dispatchEvent(new CustomEvent(EVENTS.INPUT_FOCUS, { bubbles: true, composed: true }))
            );
            textarea.addEventListener('blur', () =>
                document.dispatchEvent(new CustomEvent(EVENTS.INPUT_BLUR, { bubbles: true, composed: true }))
            );

            // 会話中リアクション（テキスト入力版）:
            // 手止まり1秒でドラフトを text_interim としてサーバーへ送る（判定・頻度制御はサーバー側）
            const sendTypingDraft = () => {
                const socket = getSocket();
                if (!socket) return;
                const text = textarea.value.trim();
                if (text === this._lastTypingDraft) return;
                this._lastTypingDraft = text;
                window.__mikuDebugToast?.(`📝 draft ${text.length}字`);
                socket.emit(SOCKET_EVENTS.TEXT_INTERIM, { text });
            };
            const scheduleTypingDraft = () => {
                clearTimeout(this._typingReactionTimer);
                this._typingReactionTimer = setTimeout(sendTypingDraft, 1000);
            };
            textarea.addEventListener('input', (e) => {
                if (e && e.isComposing) return;  // IME変換中は送らない
                scheduleTypingDraft();
            });
            textarea.addEventListener('compositionend', scheduleTypingDraft);
        }

        // 添付追加ボタン
        const addButton = shadow.getElementById('addAttachmentButton');
        if (addButton) {
            addButton.addEventListener('click', () => {
                this.createAttachmentItem();
                this.setupFileInputs();
            });
        }
        this.setupFileInputs();

        // 最終応答確定: motion・inactivity_timeout を適用。
        // ボタン表示は turnState（RESPONSE_READY → CONFIRMED/NO_TURN）からの導出に任せる
        document.addEventListener(EVENTS.SERVER_RESPONSE_READY, (e) => {
            const { motion, inactivity_timeout, has_audio } = e.detail;

            if (inactivity_timeout !== undefined) {
                document.dispatchEvent(new CustomEvent(EVENTS.APP_INACTIVITY_TIMEOUT, {
                    detail: { seconds: inactivity_timeout }
                }));
            }
            if (motion) {
                document.dispatchEvent(new CustomEvent(EVENTS.APP_MOTION_REQUEST, {
                    detail: { motionCode: motion }
                }));
            }
            // 字幕モード中は音声再生をスキップするため voice:end が来ない
            // → 音声なしの場合と同様にターンを閉じる（アバターの考え中解除）
            if (!has_audio || appState.isCaptionActive()) {
                document.dispatchEvent(new CustomEvent(EVENTS.THINKING_END));
            }
        });

        // 会話ターン状態の変化 → ボタン表示・マイクミュートを導出（basic_design §4.6）
        turnState.init();
        document.addEventListener(EVENTS.TURN_STATE_CHANGE, (e) => this._applyTurnState(e.detail));

        // サーバー疎通確認（main.js が /ping 前後に発火）: ボタンの無効化・復帰
        document.addEventListener(EVENTS.APP_CONNECTION_CHECKING, () => {
            this._connCheckDisabled = true;
            this._setSubmitButton(UI_STRINGS.submitButton.connecting, true);
        });
        document.addEventListener(EVENTS.APP_CONNECTION_RESULT, () => {
            // CHECKING を受けて自分が無効化した場合のみ復帰させる
            // （リスナー登録前に CHECKING を取り逃したケースで、進行中ターンのボタンを誤復帰させない）
            if (!this._connCheckDisabled) return;
            this._connCheckDisabled = false;
            // 現在のターン状態から導出し直す（進行中ターンがあれば「考え中」等に戻る）
            const snap = turnState.snapshot();
            this._applyTurnState({ ...snap, prev: snap });
        });

        // 無応答リマインダー = 新しいターンの開始。字幕モードをこの時点でラッチする
        document.addEventListener(EVENTS.SERVER_INACTIVITY_REMINDER, () => {
            appState.latchTurnCaption();
        });

        // STTモデルロード中: マイクボタンをLOADING状態に
        document.addEventListener(EVENTS.STT_MODEL_LOADING, () => {
            this._sttModelLoading = true;
            if (this._micPendingStart) {
                const btn = this.shadowRoot?.getElementById('micButton');
                if (btn) btn.dataset.micState = 'LOADING';
            }
        });

        // STTモデルロード完了: 待機中だったマイクを自動スタート
        document.addEventListener(EVENTS.STT_MODEL_READY, async () => {
            this._sttModelLoading = false;
            if (this._micPendingStart) {
                this._micPendingStart = false;
                await this.handleMicToggle();
            }
        });

        // generate エラー: アラート（ボタン復帰は THINKING_END → turnState 経由の導出）
        document.addEventListener(EVENTS.SERVER_GENERATE_ERROR, (e) => {
            console.error('generate エラー:', e.detail.message);
            document.dispatchEvent(new CustomEvent(EVENTS.THINKING_END));
            // 受け取った文面をそのまま出す（2026-09-20）。以前は固定文だったので、
            // 残高切れなのかキーの誤りなのか、利用者には区別がつかなかった
            alert(e.detail.message || 'エラーが発生しました');
        });

        // Socket.IO 再接続: マイクが動いていれば audio_stream_start を再送信
        document.addEventListener(EVENTS.SERVER_SOCKET_RECONNECTED, () => {
            if (this._micMode !== 'OFF' && this._mic && this._mic.getState() !== MicState.OFF) {
                const socket = getSocket();
                if (socket) socket.emit(SOCKET_EVENTS.AUDIO_STREAM_START);
                this._stt?.resetUtterance();
                console.log('[MikuChat] Socket再接続 → audio_stream_start を再送信');
            }
        });
    }

    // =========================================================================
    // ファイル添付
    // =========================================================================

    setupFileInputs() {
        const shadow = this.shadowRoot;
        if (!shadow) return;
        shadow.querySelectorAll('.file-input').forEach(input => {
            input.removeEventListener('change', this.handleFileInputChange);
            input.addEventListener('change', this.handleFileInputChange);
        });
        shadow.querySelectorAll('.clear-button').forEach(button => {
            button.removeEventListener('click', this.handleClearButtonClick);
            button.addEventListener('click', this.handleClearButtonClick);
        });
    }

    handleFileInputChange(e) {
        const input     = e.target;
        const container = input.closest('.file-input-container');
        const textSpan  = container.querySelector('.file-input-text');
        if (input.files.length > 0) {
            textSpan.textContent = input.files[0].name;
            container.classList.add('has-file');
        } else {
            textSpan.textContent = 'ファイルを選択';
            container.classList.remove('has-file');
        }
    }

    handleClearButtonClick(e) {
        const button         = e.currentTarget;
        const attachmentItem = button.closest('.attachment-item');
        const attachmentArea = button.closest('.attachment-section').querySelector('#attachmentArea');
        if (attachmentItem && attachmentArea.contains(attachmentItem)) {
            attachmentArea.removeChild(attachmentItem);
        }
    }

    createAttachmentItem() {
        const shadow         = this.shadowRoot;
        if (!shadow) return;
        const attachmentArea = shadow.getElementById('attachmentArea');
        const newItem        = document.createElement('div');
        newItem.className    = 'attachment-item';
        const fileInputId    = 'file-input-' + Date.now();
        newItem.innerHTML = `
            <div class="file-input-container">
                <input type="file" name="file_upload[]" accept=".jpg,.jpeg,.png,.gif,.webp,.svg,.mp3,.m4a,.wav,.amr,.mpga,.mp4,.mov,.mpeg,.webm,image/jpeg,image/png,image/gif,image/webp,image/svg+xml,audio/mpeg,audio/mp4,audio/wav,audio/amr,video/mp4,video/quicktime,video/mpeg,video/webm" class="file-input" id="${fileInputId}">
                <label for="${fileInputId}" class="file-input-label">
                    <i class="fas fa-file file-input-icon"></i>
                    <span class="file-input-text">ファイルを選択</span>
                </label>
            </div>
            <button type="button" class="clear-button" data-input-id="${fileInputId}">
                <i class="fas fa-times"></i>
            </button>
        `;
        attachmentArea.appendChild(newItem);
    }

    clearAllAttachments() {
        const shadow         = this.shadowRoot;
        if (!shadow) return;
        const attachmentArea = shadow.getElementById('attachmentArea');
        while (attachmentArea.firstChild) attachmentArea.removeChild(attachmentArea.firstChild);
    }

    // =========================================================================
    // フォーム送信
    // =========================================================================

    async handleFormSubmit(e) {
        e.preventDefault();
        const shadow   = this.shadowRoot;
        if (!shadow) return;

        const form     = e.target;
        const textarea = shadow.getElementById('input_text');

        // カウントダウン中の手動送信: 自動送信タイマーを止める
        // （止めないと緑のカウントダウン表示が残り、タイマーも裏で発火し続ける）
        this._stt?.cancelAutoSend();
        this._idle?.cancel();

        // 考え中の畳み込み送信（P2）: ターンが OPEN のときはテキストのみを
        // 直接 POST し、通常のターン開始処理（thinking 状態遷移・音声キューリセット等）は行わない。
        // 手動タップ・Ctrl+Enter・AUTO音声送信のすべてがここに合流する
        if (turnState.turnBelief === TurnBelief.OPEN) {
            // 文字起こしマーク〔 〕・区間情報を付けるかは _submitFoldText の中で判定する
            this._stt?.clearSubmittedByAutoSend();
            await this._submitFoldText();
            this._idle?.clearSubmittedByIdle();
            return;
        }

        // このターンの字幕モードを確定（ターン中の判定は全てラッチ値を参照する）
        appState.latchTurnCaption();
        // 再生中の音声の停止は voicePcmPlayer.js が thinking:start を購読して行う。
        // ボタンの「考え中（追加OK）」化も turnState（THINKING_START → OPEN）からの導出
        document.dispatchEvent(new CustomEvent(EVENTS.THINKING_START));

        const rawFormData = new FormData(form);
        const formData    = new FormData();
        for (const [key, value] of rawFormData.entries()) {
            if (key !== 'file_upload[]') formData.append(key, value);
        }
        formData.append('clientSessionId', getSessionId());

        // 音声認識の自動送信: 文字起こしであることを示す（2026-08-03）。
        // 〔 〕の意味（誤認識を含みうる）は base_prompt 3.1.1 に定義済みのため、
        // 以前の毎回約90字の注記は廃止（会話ログも汚さない）
        // 音声由来かの判定は _applyTranscriptMark 側に一本化した（経路ごとの条件漏れを防ぐ）
        this._stt?.clearSubmittedByAutoSend();
        this._applyTranscriptMark(formData);
        this._idle?.clearSubmittedByIdle();

        textarea.value = '';

        // 会話中リアクション（テキスト入力版）: 送信したのでドラフト監視をリセット
        clearTimeout(this._typingReactionTimer);
        this._lastTypingDraft = '';
        try { getSocket()?.emit(SOCKET_EVENTS.TEXT_INTERIM, { text: '' }); } catch (_) { /* 未接続時は無視 */ }

        // 手動添付ファイル（画像・動画は圧縮）
        for (const file of rawFormData.getAll('file_upload[]')) {
            if (file instanceof File && file.size > 0) {
                let processed = file;
                if (file.type.startsWith('video/') || /\.(mp4|mov|webm|mkv|avi)$/i.test(file.name)) {
                    // Android の content:// 由来 File は時間経過・再読込で読めなくなることがあり、
                    // アップロード中の読み取り失敗＝multipart 途中切断（サーバー400）の原因になる。
                    // 先にメモリへスナップショットしてから圧縮・送信に使う
                    let source = file;
                    try {
                        source = new File([await file.arrayBuffer()], file.name, { type: file.type });
                    } catch (e) {
                        console.warn('[Attach] 動画のメモリ読込に失敗（原本参照のまま送信を試行）:', e);
                        window.__mikuDebugToast?.(`📎 動画読込失敗: ${e?.message ?? e}`);
                    }
                    // 動画: WebCodecs で低fps・低解像度の WebM に変換（失敗時はスナップショットを送信）
                    processed = (await compressVideoFile(source)) || source;

                    // ミクが見られる inline 上限（サーバー側18MB）を超える動画は送信しない。
                    // 送ってもミクは「大きすぎて見えない」としか言えず、巨大アップロードは
                    // Cloudflare トンネルの上限（約100MB）でリクエストごと壊れることもある
                    const VIDEO_MAX_BYTES = 18 * 1024 * 1024;
                    if (processed.size > VIDEO_MAX_BYTES) {
                        const mb = (processed.size / 1024 / 1024).toFixed(0);
                        alert(`動画「${file.name}」は圧縮後も約${mb}MBあり、ミクが見られる上限（約18MB）を超えるため送信できません。\n短い動画（目安5分以内）でお試しください。`);
                        textarea.value = rawFormData.get('input_text') || '';  // 消したテキストを復元
                        // ボタン復帰は THINKING_END → turnState 経由の導出
                        document.dispatchEvent(new CustomEvent(EVENTS.THINKING_END));
                        return;
                    }
                } else if (file.type.startsWith('audio/') || /\.(mp3|m4a|wav|aac|amr|ogg|flac|mpga)$/i.test(file.name)) {
                    // 音声: 16kHz mono WAV に変換して送る（声の登録がサーバー側で読める形にする。
                    // サーバーの soundfile は m4a/aac を読めないため。失敗時は原本を送信）
                    processed = (await convertAudioForEnroll(file)) || file;
                } else {
                    processed = await this._compressImageFile(file);
                }
                formData.append('file_upload[]', processed, processed.name);
                if (processed !== file) {
                    console.log(`[Compress] ${file.name}: ${(file.size / 1024).toFixed(0)}KB → ${(processed.size / 1024).toFixed(0)}KB`);
                }
            }
        }

        // カメラバッファのフレームを添付
        if (this._camera?.isOn() && this._camera.getBuffer().length > 0) {
            this._camera.getBuffer().forEach((blob, i) => {
                formData.append('file_upload[]', blob, `camera_${i}.jpg`);
            });
            console.log(`[Camera] バッファから${this._camera.getBuffer().length}枚を添付しました`);
            this._camera.clearBuffer();
        }

        this.clearAllAttachments();

        try {
            // Fire & Forget: ACK のみ受け取る。motion・inactivity_timeout・応答完了は
            // server:responseReady / server:generateError イベント経由で処理する
            const response = await fetch('/generate', { method: 'POST', body: formData });
            const data     = await response.json();

            if (!data.success) throw new Error(data.error || '応答の生成に失敗しました');
            console.log('generate 受付完了（バックグラウンドで処理中）');

        } catch (error) {
            console.error('Error:', error);
            window.__mikuDebugToast?.(`❌ 送信失敗: ${error?.message ?? error}`);
            alert('エラーが発生しました');
            // ボタン復帰は THINKING_END → turnState 経由の導出
            document.dispatchEvent(new CustomEvent(EVENTS.THINKING_END));
        }
    }

    /**
     * 考え中の追加発話を送る畳み込み経路（P2・2026-08-01）。
     *
     * テキストのみを /generate へ直接 POST する。サーバー側は進行中ターンがあれば
     * 時系列注入（`{folded:true}`）、ターンが丁度終わっていれば通常の新規ターンとして
     * 処理する（応答はどちらもソケットイベント経由で届くため、ここでは待たない）。
     * 通常のターン開始処理（thinking 遷移・字幕ラッチ・音声キューリセット）は行わない。
     */
    /**
     * 音声認識由来の入力に印を付ける。
     *
     * 話者照合の区間情報がサーバーから届いていれば **そのまま素通し**する（voice_segments）。
     * 整形（LLM へ渡す形にすること）は agent_main に一本化しているので、ここでは組み立てない。
     * 区間情報が無い（声が未登録・照合失敗・テキストを編集した）ときは、
     * 従来どおり文字起こしマーク〔 〕を付ける。意味は base_prompt 3.1.1 に定義。
     */
    _applyTranscriptMark(formData) {
        const original = (formData.get('input_text') || '').trim();
        if (!original) return;
        // 音声由来かの判定は**ここ1箇所**で行う。呼び出し側で分岐していたときは、
        // 送信経路（通常 / 畳み込みのタイマー / 畳み込みの即時送信）ごとに条件が食い違い、
        // 一部の追加発話が素のテキストのまま保存されていた（2026-08-12）
        if (!this._stt?.isTranscriptText() && !this._idle?.isSubmittedByIdle()) return;
        const segments = this._stt?.getSegmentsFor(original) || null;
        // 送信時に区間が「採用されたか」の通過確認。受信側（sttController の [STT区間]）と
        // 対で見ると、届いていないのか・届いたが捨てたのかを一度で切り分けられる（2026-08-12）
        // console.log(`[送信区間] 採用=${segments?.length ?? 'なし'} `
        //     + `送信text="${original}" 保持text="${this._stt?._segmentsText ?? ''}"`);
        // window.__mikuDebugToast?.(`📤 区間 採用${segments?.length ?? 0}`);
        if (segments) {
            formData.set('voice_segments', JSON.stringify(segments));
        } else {
            formData.set('input_text', `〔${original}〕`);
        }
        this._stt?.clearSegments();
    }

    async _submitFoldText() {
        const shadow = this.shadowRoot;
        if (!shadow) return;
        const textarea = shadow.getElementById('input_text');
        const text = textarea?.value.trim();
        if (!text) return;

        // 添付は畳み込みでは送れない（テキストのみ。添付は返答後に改めて送ってもらう）
        const hasAttachment = [...shadow.querySelectorAll('#attachmentArea .file-input')]
            .some((input) => input.files && input.files.length > 0);
        if (hasAttachment) {
            alert('ミクが考え中のため、いまは添付付きの送信はできません（テキストのみ送れます）。\n返答が終わってから改めて送信してください。');
            return;
        }

        textarea.value = '';

        // 会話中リアクション（テキスト入力版）: 送信したのでドラフト監視をリセット
        clearTimeout(this._typingReactionTimer);
        this._lastTypingDraft = '';
        try { getSocket()?.emit(SOCKET_EVENTS.TEXT_INTERIM, { text: '' }); } catch (_) { /* 未接続時は無視 */ }

        const formData = new FormData();
        formData.append('input_text', text);
        // 音声由来かの判定は _applyTranscriptMark 側に一本化した（かつては呼び出し側で
        // 自動送信フラグを見ていたが、`turn_complete=true → 即時送信` の経路では落ちており、
        // その追加発話が素のテキストのまま保存されていた・2026-08-12）
        this._applyTranscriptMark(formData);
        formData.append('clientSessionId', getSessionId());

        try {
            const response = await fetch('/generate', { method: 'POST', body: formData });
            const data = await response.json();
            if (!data.success) throw new Error(data.error || '追加発話の送信に失敗しました');
            console.log(`[Fold] 考え中の追加発話を送信 (folded=${data.folded === true}): "${text}"`);
        } catch (error) {
            console.error('[Fold] 追加発話の送信に失敗:', error);
            // 失敗時はテキストを復元して手動で再送できるようにする
            if (textarea && !textarea.value) textarea.value = text;
            window.__mikuDebugToast?.(`❌ 追加発話の送信失敗: ${error?.message ?? error}`);
        }
    }

    // =========================================================================
    // マイクボタン初期化
    // =========================================================================

    async initMicButton() {
        const shadow    = this.shadowRoot;
        if (!shadow) return;
        const micButton = shadow.getElementById('micButton');
        if (!micButton) return;

        // /api/config を取得（カメラ設定も含む）
        let sttEnabled = false;
        let autoSendDelay = 5000;
        let cameraEnabledByServer = false;
        let cameraBufferSize = 3;

        try {
            const res = await fetch('/api/config');
            if (res.ok) {
                const config = await res.json();
                sttEnabled            = config.stt_enabled === true;
                this._autoSendEnabled = config.auto_send_enabled === true;
                autoSendDelay         = (config.auto_send_delay || 5) * 1000;
                cameraEnabledByServer = config.camera_enabled === true;
                cameraBufferSize      = config.camera_buffer_size || 3;

                // タイムアウト初期値を通知
                if (config.inactivity_timeout !== undefined) {
                    document.dispatchEvent(new CustomEvent(EVENTS.APP_INACTIVITY_TIMEOUT, {
                        detail: { seconds: config.inactivity_timeout }
                    }));
                }
            }
        } catch (e) {
            console.log('[MicButton] /api/config 取得失敗 → マイクボタン非表示');
        }

        // CameraManager を生成（/api/config の結果が必要）
        this._camera = new CameraManager(shadow, {
            bufferSize: cameraBufferSize,
        });
        this._cameraEnabledByServer = cameraEnabledByServer;

        if (!sttEnabled) {
            this._initIdleAutosend(shadow, micButton, autoSendDelay);
            return;
        }

        micButton.style.display = 'flex';
        micButton.title = UI_STRINGS.micButton.OFF;
        console.log('[MicButton] STT有効 → マイクボタン表示');

        // MicrophoneController 生成
        this._mic = new MicrophoneController();

        // SttController 生成
        this._stt = new SttController(shadow, {
            autoSendDelay: autoSendDelay,
            getMicMode:    () => this._micMode,
            getMic:        () => this._mic,
            isTurnOpen:    () => turnState.turnBelief === TurnBelief.OPEN,
        });

        // MicrophoneUI 生成
        this._micUI = new MicrophoneUI(shadow, {
            getMicMode:    () => this._micMode,
            getMic:        () => this._mic,
            cancelAutoSend: () => this._stt?.cancelAutoSend(),
            isTurnOpen:    () => turnState.turnBelief === TurnBelief.OPEN,
        });

        // マイクボタン: 短押し=ON/OFFトグル、長押し(500ms)=音量メーター調整モード
        this._micUI.setupButtonEvents(micButton, () => this.handleMicToggle());

        // mic イベントをそれぞれのクラスに委譲
        document.addEventListener(EVENTS.MIC_STATE_CHANGE, (e) => {
            this._micUI.updateButtonState(e.detail.state);
            // 録音終了時の自動送信再アーム（final が録音中に届いた場合のレース対策）
            this._stt?.onMicStateChange(e.detail.state, e.detail.prevState);
            // 録音の入口（RECORDING 開始）で送信カウンタを起こし、抜けたところで実績を出す。
            // 「録音できているのに文字起こしが返らない」とき、ブラウザが送ったのか
            // サーバーが捨てたのかを切り分けるために要る（Gateway 側にも受信集計がある）
            if (e.detail.state === MicState.RECORDING) {
                this._sentChunks = 0;
                this._sentBytes  = 0;
                this._sendSkipped = 0;
            } else if (e.detail.prevState === MicState.RECORDING) {
                console.log(`[MikuChat] 音声送信: ${this._sentChunks ?? 0}チャンク / `
                    + `${(this._sentBytes ?? 0).toLocaleString()}バイト`
                    + (this._sendSkipped ? ` （未接続でスキップ ${this._sendSkipped}）` : ''));
            }
        });
        document.addEventListener(EVENTS.MIC_LEVEL,       (e) => this._micUI.updateLevel(e.detail));
        document.addEventListener(EVENTS.MIC_SPEECH_END,   (e) => this.handleSpeechEnd(e.detail));
        document.addEventListener(EVENTS.STT_RESULT,      (e) => this.handleSttResult(e.detail));
        document.addEventListener(EVENTS.STT_ERROR,       (e) => this.handleSttError(e.detail));
        document.addEventListener(EVENTS.MIC_AUDIO_CHUNK,  (e) => this._sendAudioStreamChunk(e.detail.pcmData, e.detail.sampleRate));

        // リアルタイムSTT（socket.on はここで設定）
        const socket = getSocket();
        if (socket) {
            socket.on(SOCKET_EVENTS.STT_INTERIM, (data) => {
                this._stt.handleInterim(data);
                // 「言葉として認識された」＝**本当に喋っている証拠**を配る（音量では雑音と区別できない）。
                // voicePcmPlayer が再生保留の上限タイマーを引き直すのに使う
                if (data?.text) {
                    document.dispatchEvent(new CustomEvent(EVENTS.STT_SPEECH_EVIDENCE));
                }
            });
            socket.on(SOCKET_EVENTS.STT_FINAL,   (data) => {
                this._stt.handleFinal(data);
                // 「マスターが言い終わったか」を turnState へ伝える。
                // 音声再生の保留判断に使う（sttController を直接見に行かせない）
                document.dispatchEvent(new CustomEvent(EVENTS.STT_UTTERANCE_END, {
                    detail: { turnComplete: data.turn_complete },
                }));
            });
        }

        // VAD閾値スライダー
        const slider = shadow.getElementById('micThresholdSlider');
        if (slider) {
            slider.value = Math.round(this._mic.getVadThreshold() * 1000);
            slider.addEventListener('input', (e) => {
                this._mic.setVadThreshold(parseInt(e.target.value, 10) / 1000);
            });
        }
    }

    // =========================================================================
    // カメラボタン初期化
    // =========================================================================

    async initCameraButton() {
        if (!this._camera) return;
        await this._camera.init(this._cameraEnabledByServer || false);
    }

    // =========================================================================
    // 字幕ONへの切替時のターン閉じ（メニュー操作は MenuController、状態依存の後処理のみ MikuChat）
    // =========================================================================

    /**
     * 字幕ONに切り替えた時、voice:end 待ちのまま残ったターンを閉じる（MenuController のコールバック）。
     * 音声の破棄自体は turnState が CAPTION_SET_ENABLED で audible を落とし、ボタンは導出で
     * 復帰する。ここでは応答確定済みのターン（voice:end がもう来ない）を閉じて
     * アバターの考え中状態も解除する。
     */
    _closeCaptionTurnIfStuck(enabled) {
        if (enabled && turnState.turnBelief === TurnBelief.CONFIRMED) {
            document.dispatchEvent(new CustomEvent(EVENTS.THINKING_END));
        }
    }

    // =========================================================================
    // 入力アイドル自動送信（STTオフ時）
    // =========================================================================

    _initIdleAutosend(shadow, micButton, autoSendDelay) {
        this._idle = new IdleAutoSendController(shadow, { getDelay: () => autoSendDelay });
        this._idle.attach();
        micButton.style.display = 'flex';
        micButton.dataset.micState = 'OFF';
        const icon = micButton.querySelector('i');
        if (icon) icon.className = 'fas fa-paper-plane';
        micButton.title = UI_STRINGS.idleAutosend.OFF;
        micButton.addEventListener('click', () => this._toggleIdleAutosend(micButton));
        console.log('[IdleAutosend] 初期化完了 → トグルOFF');
    }

    _toggleIdleAutosend(micButton) {
        if (this._idle.isEnabled()) {
            this._idle.setEnabled(false);
            micButton.dataset.micState = 'OFF';
            micButton.title = UI_STRINGS.idleAutosend.OFF;
            console.log('[IdleAutosend] OFF');
        } else {
            this._idle.setEnabled(true);
            micButton.dataset.micState = 'LISTENING';
            micButton.title = UI_STRINGS.idleAutosend.ON;
            const ta = this.shadowRoot?.getElementById('input_text');
            if (ta) ta.focus();
            console.log('[IdleAutosend] ON');
        }
    }

    // =========================================================================
    // マイクトグル
    // =========================================================================

    async handleMicToggle() {
        if (!this._mic) return;

        // 歌っている最中にマイクボタンを押した＝「話しかけたい」の合図。
        // 歌唱中はマイクをミュートしている（自分の歌を拾わないため）ので、
        // **声で割り込む道はここしかない**。歌をやめさせてから通常のトグルへ進む
        // （止める専用ボタンは置かない。現実で歌っている人に話しかけるのと同じ形にする）。
        if (turnState.singing) {
            document.dispatchEvent(new CustomEvent(EVENTS.SONG_INTERRUPT));
            console.log('[MicButton] 歌唱中に押された → 歌をやめて聞く体勢へ');
        }

        if (this._micMode === 'OFF') {
            // STTモデルがロード中
            if (this._sttModelLoading) {
                if (this._micPendingStart) {
                    // 再タップでキャンセル → OFFに戻す
                    this._micPendingStart = false;
                    this._micUI?.updateButtonState(MicState.OFF);
                    console.log('[MicButton] ロード待機をキャンセル → OFF');
                } else {
                    // 初回タップ → ロード完了後に自動スタート
                    this._micPendingStart = true;
                    const btn = this.shadowRoot?.getElementById('micButton');
                    if (btn) btn.dataset.micState = 'LOADING';
                    console.log('[MicButton] STTモデルロード中 → 完了後に自動スタート');
                }
                return;
            }
            try {
                await this._mic.start();
                this._stt?.resetUtterance();
                const socket = getSocket();
                if (socket) socket.emit(SOCKET_EVENTS.AUDIO_STREAM_START);
                this._micMode = 'LISTEN';
                this._micUI?.updateModeUI(this._micMode);
                console.log('[MicButton] OFF → 音声認識');
            } catch (err) {
                console.error('[MicButton] マイク起動失敗:', err);
                alert(UI_STRINGS.error.micPermission);
            }
        } else if (this._micMode === 'LISTEN' && this._autoSendEnabled) {
            this._micMode = 'AUTO';
            this._micUI?.updateModeUI(this._micMode);
            console.log('[MicButton] 音声認識 → 音声認識+自動送信');
        } else {
            this._stt?.cancelAutoSend();
            const socket = getSocket();
            if (socket) socket.emit(SOCKET_EVENTS.AUDIO_STREAM_STOP);
            this._mic.stop();
            this._micMode = 'OFF';
            this._micUI?.updateModeUI(this._micMode);
            console.log('[MicButton] → OFF');
        }
    }

    // =========================================================================
    // STT 連携
    // =========================================================================

    handleSpeechEnd(_detail) {
        if (this._mic) this._mic.notifySTTComplete();
    }

    handleSttResult(detail) {
        const shadow    = this.shadowRoot;
        if (!shadow) return;
        const textarea  = shadow.getElementById('input_text');
        if (textarea && detail.text) {
            textarea.value = textarea.value.trim()
                ? textarea.value + '\n' + detail.text
                : detail.text;
            console.log(`[MikuChat] STT結果を入力欄に表示: "${detail.text}"`);
        }
        if (this._mic) this._mic.notifySTTComplete();
    }

    handleSttError(detail) {
        console.error(`[MikuChat] STTエラー: ${detail.error}`);
        if (this._mic) this._mic.notifySTTComplete();
    }

    /**
     * Float32 PCM → 16kHz Int16 に変換して Gateway へ送信
     */
    _sendAudioStreamChunk(pcmData, sampleRate) {
        const socket = getSocket();
        if (!socket || !socket.connected) {
            this._sendSkipped = (this._sendSkipped ?? 0) + 1;
            return;
        }

        const TARGET_RATE = 16000;
        let samples = pcmData;
        if (sampleRate !== TARGET_RATE) {
            const ratio  = sampleRate / TARGET_RATE;
            const newLen = Math.floor(pcmData.length / ratio);
            samples = new Float32Array(newLen);
            for (let i = 0; i < newLen; i++) {
                const start = Math.floor(i * ratio);
                const end   = Math.floor((i + 1) * ratio);
                let sum = 0;
                for (let j = start; j < end; j++) sum += pcmData[j];
                samples[i] = sum / (end - start);
            }
        }
        const int16 = new Int16Array(samples.length);
        for (let i = 0; i < samples.length; i++) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        socket.emit(SOCKET_EVENTS.AUDIO_STREAM_CHUNK, { audio: int16.buffer });
        this._sentChunks = (this._sentChunks ?? 0) + 1;
        this._sentBytes  = (this._sentBytes  ?? 0) + int16.byteLength;
    }

    // =========================================================================
    // 画像圧縮ユーティリティ
    // =========================================================================

    async _compressImageFile(file) {
        const IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
        if (!IMAGE_TYPES.includes(file.type)) return file;

        const MAX_PX  = 1536;
        const QUALITY = 0.85;

        const bitmap = await createImageBitmap(file);
        const { width, height } = bitmap;
        let dstW = width, dstH = height;
        if (Math.max(width, height) > MAX_PX) {
            const ratio = MAX_PX / Math.max(width, height);
            dstW = Math.round(width * ratio);
            dstH = Math.round(height * ratio);
        }
        const canvas = document.createElement('canvas');
        canvas.width  = dstW;
        canvas.height = dstH;
        canvas.getContext('2d').drawImage(bitmap, 0, 0, dstW, dstH);
        bitmap.close();

        return new Promise(resolve => {
            canvas.toBlob(blob => {
                resolve(new File([blob], file.name.replace(/\.[^.]+$/, '.jpg'), { type: 'image/jpeg' }));
            }, 'image/jpeg', QUALITY);
        });
    }
}

// カスタム要素として登録
customElements.define('miku-chat', MikuChat);
