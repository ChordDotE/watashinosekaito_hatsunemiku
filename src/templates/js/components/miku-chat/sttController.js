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
 * sttController.js - STTリアルタイム表示・auto-sendカウントダウン
 *
 * <miku-chat> の内部部品（Shadow DOM を引数に取る。単独では動かない）。
 * stt_interim/stt_final の textarea 反映と、AUTOモードの自動送信カウントダウンを担う。
 *
 * ## 終話判定（Smart Turn v3・stt.smart_turn.enabled 時）
 *
 * stt_final に `turn_complete`（bool）が付いてくる場合がある：
 *   - true  … サーバーが「言い終わった話し方」と判定 → カウントダウンなしで即送信
 *   - false … 「続きがある話し方」（言いよどみ等） → 送信せず待つ。
 *             既存のカウントダウンは「続きが来なかった場合の保険」として起動する
 *   - なし  … Smart Turn 無効（従来どおりカウントダウン）
 *
 * ## 考え中（turnBelief=OPEN）の追加発話（2026-08-11）
 *
 * ミクが考え中の送信は「新しいターンの開始」ではなく進行中ターンへの畳み込み
 * （サーバーが時系列注入する）。中途半端に送っても後から続きを足せる一方、
 * **届かないことが致命的**なので、通常時と扱いを分ける:
 *   - 待ち時間を FOLD_SEND_DELAY_MS へ短縮する
 *   - **録音開始でカウントダウンをキャンセルしない**（キャンセルすると話し続ける限り
 *     一度も満了せず、追加発話がまったく届かない）
 *   - 代わりに満了時が発話の途中なら、未確定テキストを送らず次の final を待つ
 */

import { MicState } from '../../modules/microphone.js';

/** 考え中の追加発話を送るまでの待ち時間。通常時の auto_send.delay_seconds（既定5秒）
 *  と違い、畳み込みは細切れでも時系列注入されるため長く待つ意味がない */
const FOLD_SEND_DELAY_MS = 1000;

/** 「直近の STT final からこの時間内の送信は音声由来」とみなす窓。
 *  話し続けている間の追加発話（数秒間隔）を確実に拾いつつ、あとから手で打った文は
 *  拾わない長さにする。手入力には final 自体が来ないので、多少長くても誤検知しない */
const TRANSCRIPT_WINDOW_MS = 30000;

export class SttController {
    /**
     * @param {ShadowRoot} shadow
     * @param {{ autoSendDelay: number, getMicMode: function, getMic: function,
     *           isTurnOpen: function }} options
     */
    constructor(shadow, options = {}) {
        this._shadow          = shadow;
        this._autoSendDelay   = options.autoSendDelay || 5000;
        this._getMicMode      = options.getMicMode || (() => 'OFF');
        this._getMic          = options.getMic     || (() => null);
        // ミクが考え中（turnBelief=OPEN）か。turnState を直接見ずに miku-chat から注入する
        this._isTurnOpen      = options.isTurnOpen || (() => false);

        this._rtConfirmedBase     = '';
        this._rtInUtterance       = false;
        this._autoSendTimer       = null;
        this._submittedByAutoSend = false;
        // カウントダウン満了時に発話の途中だった場合の予約（次の final が来たら即送信）
        this._sendOnNextFinal     = false;
        // 録音中に final が届いた場合の保留（録音終了時に再アームする）
        this._pendingAfterRecording = null;   // { turnComplete: bool } | null
        this._rearmTimer            = null;
        // 話者照合の区間情報（サーバーが付けてきたものを**素通し**する）。
        // ブラウザ側で組み立てず、整形は agent_main に一本化している
        this._segments     = null;   // [{start, end, score, borrowed, scored, text}] | null
        this._segmentsText = '';     // その segments が対応するテキスト（編集検知に使う）
        this._segmentsEnd  = 0;      // 連結時に次の final へ足す時刻オフセット（秒）
        // 最後に STT final が届いた時刻（ms）。区間（_segments）は送信のたびに使い切るが、
        // **音声由来だったという事実は消さない**。これが無いと、続けて送った追加発話が
        // 素のテキストで保存され、後から見て文字起こしだと分からなくなる（2026-08-12）。
        // テキストの一致では判定できない —— 畳み込み中は認識結果の**断片**が送られるため
        // （final "俺の声をちょっと覚えてほしいな。" に対し、送信は "声をちょっと〜" だけ）
        this._lastFinalAt = 0;
    }

    /**
     * 送信テキストに対応する区間情報を返す（一致しなければ null）。
     *
     * textarea はユーザーが編集できるので、**サーバーが認識したままのテキスト**の
     * ときだけ添える。編集が入ったら区間と対応が取れないため捨てる。
     */
    getSegmentsFor(text) {
        if (!this._segments || !this._segmentsText) return null;
        return text.trim() === this._segmentsText.trim() ? this._segments : null;
    }

    /**
     * いま送ろうとしている内容が音声認識由来か（＝文字起こしマークを付けるべきか）。
     *
     * **直近の STT final からの経過時間**で判定する。理由:
     *   - 自動送信フラグ（isSubmittedByAutoSend）は「タイマー経由で送った」ときしか立たず、
     *     `turn_complete=true → 即時送信` の経路では落ちる
     *   - テキストの一致でも判定できない。畳み込み中は認識結果の**断片**が送られるため
     * 手で打った文には final が来ないので、この判定に引っかからない。
     */
    isTranscriptText() {
        return this._lastFinalAt > 0
            && (Date.now() - this._lastFinalAt) <= TRANSCRIPT_WINDOW_MS;
    }

    clearSegments() {
        this._segments     = null;
        this._segmentsText = '';
        this._segmentsEnd  = 0;
    }

    /**
     * STT interim 結果をtextareaにプレビュー表示する。
     * 発話開始時点のtextarea内容をベースとして保持し、上書きを防ぐ。
     */
    handleInterim(data) {
        if (!data.text) return;
        const textarea = this._shadow.getElementById('input_text');
        if (!textarea) return;

        if (!this._rtInUtterance) {
            this._rtInUtterance   = true;
            this._rtConfirmedBase = textarea.value.trimEnd();
            // 話し始めたのでカウントダウンをキャンセル（**通常時のみ**）。
            // 考え中は止めない — 止めると話し続ける限り満了せず追加発話が届かない
            if (!this._isTurnOpen()) this.cancelAutoSend();
        }

        const sep = this._rtConfirmedBase ? ' ' : '';
        textarea.value = this._rtConfirmedBase + sep + data.text;
    }

    /**
     * STT final 結果をtextareaに確定表示する。
     * AUTOモードなら自動送信カウントダウンを開始する。
     */
    handleFinal(data) {
        if (!data.text) {
            // **空文字でも黙って捨てない**。捨てると「サーバーに届いていない」のか
            // 「届いたが認識結果が空だった」のかがログから区別できなくなる
            // （2026-08-16 の iPad 調査では、この違いが分からず経路側を疑って遠回りした）
            console.log('[MikuChat] STT final: 【空文字】'
                + (data.turn_complete !== undefined ? ` (turn_complete=${data.turn_complete})` : ''));
            return;
        }
        const textarea = this._shadow.getElementById('input_text');
        if (!textarea) return;

        if (!this._rtInUtterance) {
            this._rtConfirmedBase = textarea.value.trimEnd();
        }

        const sep = this._rtConfirmedBase ? ' ' : '';
        textarea.value = this._rtConfirmedBase + sep + data.text;

        // 区間情報は**連結しても持ち回る**（2026-08-12）。
        // かつては「連結だと区間と対応が取れない」として捨てていたが、Smart Turn の実会話では
        // 言い直し・言い足しで final が複数回届くのが普通で（実測: 1発話で3回）、
        // その条件だと区間情報がほとんど届かなかった。
        // 各 final の区間は「その final の先頭からの相対時刻」なので、連結時は先行ぶんの
        // 長さを足してから繋ぐ。ズレるのは絶対時刻だけで、一致度の意味は変わらない。
        if (data.segments) {
            const offset = this._segmentsEnd;
            const shifted = data.segments.map(s => ({
                ...s,
                start: (s.start || 0) + offset,
                end:   (s.end   || 0) + offset,
            }));
            this._segments     = (this._rtConfirmedBase && this._segments)
                ? this._segments.concat(shifted)          // 連結: 前の区間の後ろに足す
                : shifted;                                 // 単独: そのまま
            this._segmentsText = textarea.value;           // 連結後の全文と対応づける
            this._segmentsEnd  = shifted.length
                ? Math.max(...shifted.map(s => s.end || 0))
                : offset;
        } else if (!this._rtConfirmedBase) {
            this.clearSegments();      // 区間なしの単独 final → 前の区間は無関係なので捨てる
        }
        // 音声由来だった事実を残す（区間と違い、送信では使い切らない）
        this._lastFinalAt = Date.now();

        // 区間が「サーバーから届いたか」の通過確認。2026-08-12 の調査で、
        // Gateway が stt_final の詰め替えで segments を落としていたのをこれで特定した。
        // 同じ経路を疑うときは下2行を戻す（送信側は _applyTranscriptMark にも同種のログがある）
        // console.log(`[STT区間] 受信=${data.segments?.length ?? 'なし'} `
        //     + `連結base="${this._rtConfirmedBase}" 保持=${this._segments?.length ?? 0}`);
        // window.__mikuDebugToast?.(`🎙 区間 受信${data.segments?.length ?? 0}/保持${this._segments?.length ?? 0}`);

        this._rtInUtterance   = false;
        this._rtConfirmedBase = '';

        console.log(`[MikuChat] STT final: "${data.text}" (confidence=${data.confidence}`
            + (data.turn_complete !== undefined ? `, turn_complete=${data.turn_complete}` : '')
            + ')');

        // 新しい final が届いた → 録音終了時の再アーム予約は破棄（この final が引き継ぐ）
        if (this._rearmTimer) {
            clearTimeout(this._rearmTimer);
            this._rearmTimer = null;
        }
        // ただし「言い終わった」判定は**引き継ぐ**（2026-08-11）。破棄していたため、
        // turn_complete=true の保留中に次の final が届くと即時送信の根拠が消え、
        // 確定テキストが入力欄に取り残されたまま送られなくなっていた
        const carriedTurnComplete = this._pendingAfterRecording?.turnComplete === true;
        this._pendingAfterRecording = null;

        if (this._getMicMode() === 'AUTO') {
            const mic = this._getMic();
            const turnComplete = data.turn_complete === true || carriedTurnComplete;
            if (mic && mic.getState() === MicState.RECORDING) {
                // 既に次の発話を録音中 → 送信せず続きを待つ（turn_complete より優先）。
                // ただし STT の final が VAD の無音猶予より早く出揃うと、録音終了後に
                // final が二度と来ず送信されないままになる（実機で発生したレース）。
                // 録音終了時（onMicStateChange）に最後の判定で再アームする
                this._pendingAfterRecording = { turnComplete };
                console.log('[AutoSend] 録音中のためスキップ（録音終了時に再アーム）');
            } else if (this._sendOnNextFinal) {
                // 未確定テキストを送らないよう待っていた確定が届いた → いま送る
                console.log('[AutoSend] 確定待ちだった送信を実行');
                this.submitNow();
            } else if (turnComplete) {
                // Smart Turn が「言い終わった」と判定 → 固定タイマーなしで即送信
                console.log('[AutoSend] turn_complete=true → 即時送信');
                this.submitNow();
            } else {
                // turn_complete=false（続きがありそう）または Smart Turn 無効
                // → 従来のカウントダウン（false の場合は「続きが来なかった時の保険」）
                this.startAutoSend();
            }
        }
    }

    /**
     * マイクの状態遷移通知（MikuChat の mic:stateChange リスナーから呼ばれる）。
     *
     * 録音中にスキップした final があれば、録音終了時に自動送信を再アームする。
     * 400ms の猶予を置くのは、録音終了直後に新しい final が届くレースがあるため
     * （届けば handleFinal が予約を破棄して自分の判定で送り直す）。
     */
    onMicStateChange(state, prevState) {
        if (state === MicState.MUTED || state === MicState.OFF) {
            // ミク発話開始・マイクOFF: 保留と予約を破棄
            this._pendingAfterRecording = null;
            this._sendOnNextFinal       = false;
            if (this._rearmTimer) {
                clearTimeout(this._rearmTimer);
                this._rearmTimer = null;
            }
            return;
        }
        if (prevState === MicState.MUTED && state === MicState.LISTENING) {
            // ミク発話が終わってマイクが開いた。発話中に取り残された未送信テキスト
            // （カウントダウン発火時にボタンが「返答中」で送れなかった等）があれば再アーム
            const textarea = this._shadow.getElementById('input_text');
            if (this._getMicMode() === 'AUTO' && textarea && textarea.value.trim()
                && !this._autoSendTimer) {
                console.log('[AutoSend] ミク発話終了 → 未送信テキストの自動送信を再アーム');
                this.startAutoSend();
            }
            return;
        }
        if (prevState !== MicState.RECORDING || !this._pendingAfterRecording) return;

        const pending = this._pendingAfterRecording;
        this._pendingAfterRecording = null;
        if (this._getMicMode() !== 'AUTO') return;

        this._rearmTimer = setTimeout(() => {
            this._rearmTimer = null;
            const textarea = this._shadow.getElementById('input_text');
            if (!textarea || !textarea.value.trim()) return;
            // 猶予中に次の発話が始まっていたら送らず確定を待つ（2026-08-11）。
            // ここで送ると入力欄の未確定テキスト（interim）を含む断片を送ってしまい、
            // 直後に確定した final が入力欄に取り残される
            if (this._rtInUtterance) {
                this._pendingAfterRecording = pending;   // turn_complete 判定を保持
                this._sendOnNextFinal       = true;
                console.log('[AutoSend] 再アーム時に次の発話が進行中 → 次の final で即送信');
                return;
            }
            console.log(`[AutoSend] 録音終了 → 保留していた自動送信を再アーム (turn_complete=${pending.turnComplete})`);
            if (pending.turnComplete) {
                this.submitNow();
            } else {
                this.startAutoSend();
            }
        }, 400);
    }

    /**
     * カウントダウンなしで即座に送信する（Smart Turn の COMPLETE 判定時）。
     * 送信経路は startAutoSend と同じ submitButton.click()。
     * ミクが考え中は button が fold-ready（押せる）のため click がそのまま
     * MikuChat.handleFormSubmit の畳み込み分岐（_submitFoldText）に合流する。
     * disabled は音声再生中・接続中のみで、そのときは送らず待つ。
     */
    submitNow() {
        this.cancelAutoSend();
        this._sendOnNextFinal = false;
        const btn = this._shadow.getElementById('submitButton');
        if (!btn || btn.disabled) return;
        this._submittedByAutoSend = true;
        btn.click();
    }

    /** 今の待ち時間（考え中は畳み込み用に短縮する） */
    _currentDelay() {
        return this._isTurnOpen() ? FOLD_SEND_DELAY_MS : this._autoSendDelay;
    }

    startAutoSend() {
        this.cancelAutoSend();
        const btn = this._shadow.getElementById('submitButton');
        if (!btn || btn.disabled) return;

        const delay = this._currentDelay();
        btn.style.setProperty('--countdown-duration', `${delay}ms`);
        btn.classList.add('counting-down');

        this._autoSendTimer = setTimeout(() => {
            btn.classList.remove('counting-down');
            this._autoSendTimer = null;
            if (btn.disabled) {
                // カウントダウン中にミクの音声再生が始まった（返答中）。ここで click しても
                // 握りつぶされるだけなので送らない。ミク発話終了（MUTED→LISTENING）時に
                // onMicStateChange が未送信テキストを見て再アームする
                console.log('[AutoSend] タイムアウト時にボタン無効（返答中）→ 発話終了後に再アーム');
                return;
            }
            if (this._rtInUtterance) {
                // 発話の途中で満了した（考え中は録音開始でキャンセルしないため起こりうる）。
                // 未確定テキストを送らず、確定した瞬間に送る
                this._sendOnNextFinal = true;
                console.log('[AutoSend] 満了時に発話の途中 → 次の final で即送信');
                return;
            }
            console.log('[AutoSend] タイムアウト → 自動送信');
            this._submittedByAutoSend = true;
            btn.click();
        }, delay);

        console.log(`[AutoSend] カウントダウン開始: ${delay}ms`);
    }

    cancelAutoSend() {
        if (this._autoSendTimer) {
            clearTimeout(this._autoSendTimer);
            this._autoSendTimer = null;
        }
        const btn = this._shadow.getElementById('submitButton');
        if (btn) btn.classList.remove('counting-down');
    }

    resetUtterance() {
        this._rtConfirmedBase = '';
        this._rtInUtterance   = false;
    }

    isSubmittedByAutoSend()   { return this._submittedByAutoSend; }
    clearSubmittedByAutoSend(){ this._submittedByAutoSend = false; }
}
