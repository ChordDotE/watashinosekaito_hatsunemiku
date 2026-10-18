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
 * miku-avatar-live2d.js - アバター表示コンポーネント（Live2D iframe版）
 * 
 * 機能:
 * - Live2DをGateway経由でiframeに埋め込み
 * - 状態に応じたクラス切り替え（idle, thinking, speaking）
 * - カスタムイベントの購読（voice:start, voice:end, thinking:start, thinking:end）
 * 
 * リファクタリング後:
 * - リップシンク計算は voiceOutputShared.js に移動
 * - Live2D制御は voiceOutputShared.js から postMessage 経由で行う
 * - このコンポーネントは状態管理とiframe埋め込みに集中
 */

import { EVENTS } from '../../modules/events.js';
import { getSessionId } from '../../modules/session.js';

//: 歌っているあいだ再生する立ち姿（MOTION_MAP の 27）。**歌唱専用**なので LLM には選ばせない。
//  ★`startMotion` にループ指定が無いので、**尺ぶんの間隔で再生し直す**。
//    モーションは正弦波2周期（21.33 秒）で先頭と末尾が同じ値なので、繋ぎ目は滑らか。
const SING_MOTION = '27';
const SING_LOOP_MS = 21330;

/**
 * 曲を流しているあいだの立ち姿を、**何を流しているか**で選ぶ（2026-09-23）。
 *
 * - `sing`    … 実歌声。ミクが歌っている
 * - `preview` … **まだ歌になっていない音**（途中の段階）を一緒に聴いている。口は動かない
 * - `play`    … **歌の入らない曲（インスト）**を一緒に聴いている。口は動かない（2026-10-03）
 *
 * ★**`sing` と `preview` は歌の立ち姿**（本人判断 2026-09-23・「一旦乙で」）。
 *   将来「聴いてる」専用のモーションを作ったら、**ここの `preview` を差し替えるだけ**で移れる
 *   （本人:「**将来的に丙に拡張するかもしれないから、そこだけできるような形にしておいて**」）。
 *   本人が迷っていた点:「**音楽流しながらモーションが激しいとうるさい**」——
 *   もし気になったら、専用モーションを `live2d-motion-lab` で作ってここへ入れる。
 * ★**`play` は 09（うなずき・傾聴）**。完成した演奏を流すときは歌の立ち姿にしない（歌っているように見せない）。
 *   09 は口を一度も開けず（ParamMouthOpenY 0）、頭を小さく縦に振るだけ（1.3 秒）なので音楽の上でうるさくない。
 *   1回きりのモーションなので、再生し直すまでのあいだは通常の待機モーションに戻る。
 */
const PLAYBACK_MOTION = { sing: SING_MOTION, preview: SING_MOTION, play: '09' };

/**
 * MikuAvatarLive2D カスタム要素
 */
class MikuAvatarLive2D extends HTMLElement {
    constructor() {
        super();
        this.state = 'idle'; // idle, thinking, speaking
        this.iframe = null;
        this.live2dReady = false;
        this.singTimer = null;   // 歌っているあいだ、立ち姿を再生し直すタイマー
    }

    /**
     * コンポーネントがDOMに追加された時
     */
    async connectedCallback() {
        // HTMLテンプレートを読み込み
        await this.loadTemplate();

        // iframe要素を取得
        this.iframe = this.shadowRoot?.querySelector('#live2d-iframe');

        if (this.iframe) {
            // セッションIDを取得（session.js に一元化。旧: sessionStorage + 自前UUID生成の複製実装）
            const sessionId = getSessionId();

            // iframeのsrc設定（セッションIDはもはやWebSocket登録には使わないが、URLパラメータとして残す）
            this.iframe.src = `/live2d?sessionId=${encodeURIComponent(sessionId)}`;
            console.log(`📺 Live2D iframe src設定: /live2d`);

            // iframeの読み込み完了を待機
            this.iframe.addEventListener('load', () => {
                console.log('✅ Live2D iframe loaded');
                this.live2dReady = true;

                // ローディングオーバーレイを非表示
                const loadingOverlay = this.shadowRoot?.querySelector('#loading-overlay');
                if (loadingOverlay) {
                    loadingOverlay.classList.add('hidden');
                    setTimeout(() => {
                        loadingOverlay.style.display = 'none';
                    }, 500);
                }
            });

            // iframeのエラー時の処理
            this.iframe.addEventListener('error', () => {
                console.warn('⚠️ Live2D iframe load error');
                this.hideLoadingOverlay();
            });
        }

        // イベントリスナーをセットアップ
        this.setupEventListeners();

        // 初期状態を設定
        this.setState('idle');

        console.log('miku-avatar-live2d コンポーネントが初期化されました');
    }

    /**
     * コンポーネントがDOMから削除された時
     */
    disconnectedCallback() {
        // イベントリスナーを解除
        this.removeEventListeners();
    }

    /**
     * HTMLテンプレートを読み込み
     */
    async loadTemplate() {
        try {
            const response = await fetch('/js/components/miku-avatar/miku-avatar-live2d.html');
            const html = await response.text();

            // Shadow DOMを作成
            const shadow = this.attachShadow({ mode: 'open' });

            // テンプレートをパースして追加
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            const template = doc.querySelector('template');

            if (template) {
                shadow.appendChild(template.content.cloneNode(true));
            } else {
                shadow.innerHTML = html;
            }
        } catch (error) {
            console.error('miku-avatar-live2d テンプレート読み込みエラー:', error);
            this.createFallbackTemplate();
        }
    }

    /**
     * フォールバックテンプレートを作成（テンプレート読み込み失敗時）
     */
    createFallbackTemplate() {
        const shadow = this.attachShadow({ mode: 'open' });
        shadow.innerHTML = `
            <style>
                :host {
                    display: block;
                    width: 100%;
                    height: 100%;
                }
                .character-section {
                    width: 100%;
                    height: 100%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    position: relative;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    overflow: hidden;
                }
                #live2d-iframe {
                    width: 100%;
                    height: 100%;
                    border: none;
                }
            </style>
            <div class="character-section idle">
                <iframe id="live2d-iframe" src="/live2d" allow="autoplay"></iframe>
            </div>
        `;
    }

    /**
     * iframe を読み込めなかったときの後始末。
     * 表示の中身は /live2d 側（未導入なら案内画面）が受け持つので、ここでは
     * 読み込み中のぐるぐるを止めるだけにする（静止画のフォールバックは持たない）
     */
    hideLoadingOverlay() {
        const loadingOverlay = this.shadowRoot?.querySelector('#loading-overlay');
        if (loadingOverlay) {
            loadingOverlay.style.display = 'none';
        }
    }

    /**
     * イベントリスナーをセットアップ
     */
    setupEventListeners() {
        // voice:start - 音声再生開始
        this.handleVoiceStart = () => this.setState('speaking');
        document.addEventListener(EVENTS.VOICE_START, this.handleVoiceStart);

        // voice:end - 音声再生終了
        this.handleVoiceEnd = () => this.setState('idle');
        document.addEventListener(EVENTS.VOICE_END, this.handleVoiceEnd);

        // thinking:start - 考え中開始
        this.handleThinkingStart = () => this.setState('thinking');
        document.addEventListener(EVENTS.THINKING_START, this.handleThinkingStart);

        // thinking:end - 考え中終了
        this.handleThinkingEnd = () => {
            if (this.state !== 'speaking') this.setState('idle');
        };
        document.addEventListener(EVENTS.THINKING_END, this.handleThinkingEnd);

        // app:motionRequest - モーション送信要求（miku-chat.js / その他モジュールから発火）
        this.handleMotionRequest = (e) => {
            const { motionCode } = e.detail;
            // ★**歌っているあいだは受け取らない**（2026-09-22）。
            //   受けると歌の立ち姿（27）が上書きされ、**次の送り直しまで最大 21 秒戻らない**。
            //   2026-09-21 の収録では、歌い出し 7 秒後に追加応答のモーション 16 が、
            //   その 60 秒後に無応答リマインダーのモーション 01 が割り込み、
            //   録画のほとんどの時間で歌の立ち姿になっていなかった。
            //   `singTimer` は song:end で必ず解除されるので、状態が残らない。
            if (this.singTimer) {
                console.log(`🎭 歌っているので見送り: ${motionCode}`);
                return;
            }
            if (motionCode) this.sendMotion(motionCode);
        };
        document.addEventListener(EVENTS.APP_MOTION_REQUEST, this.handleMotionRequest);

        // song:start - 歌い出したら立ち姿（27）へ。以後は尺ぶんの間隔で再生し直す
        this.handleSongStart = (e) => {
            // ★**`force` で送る**（2026-09-22）。`startMotion` は**同じ優先度のものが
            //   流れている間は受け付けない**（`reserveMotion`）。歌い出しの瞬間はミクの
            //   返事のモーションがまだ流れているので、既定の優先度だと 27 が弾かれ、
            //   次の送り直しまで 21.3 秒入らない。実測でも**曲の前半 32 秒だけ
            //   揺れの周期が出ていなかった**（自己相関 10.67 秒で −0.42。後半は +0.26）。
            const motion = PLAYBACK_MOTION[e?.detail?.mode] || SING_MOTION;
            this.sendMotion(motion, 'force');
            if (this.singTimer) clearInterval(this.singTimer);
            this.singTimer = setInterval(() => this.sendMotion(motion, 'force'), SING_LOOP_MS);
        };
        document.addEventListener(EVENTS.SONG_START, this.handleSongStart);

        // song:end - 通常の立ち姿へ戻す（タイマーを必ず止める）
        this.handleSongEnd = () => {
            this.stopSingLoop();
            this.sendMotion('01');
        };
        document.addEventListener(EVENTS.SONG_END, this.handleSongEnd);

        // gaze:target - マスターの顔の方を向く（gazeTracker が発火。0,0 で正面）。
        // Live2D 側ではドラッグ追従と同じ経路に入り、モーションの上に重なる
        this.handleGazeTarget = (e) => {
            const { x, y } = e.detail || {};
            this.sendToIframe({ type: 'gaze', x, y });
        };
        document.addEventListener(EVENTS.GAZE_TARGET, this.handleGazeTarget);
    }

    /** 歌の立ち姿ループを止める（song:end と、要素が外れるときの両方から呼ぶ） */
    stopSingLoop() {
        if (this.singTimer) {
            clearInterval(this.singTimer);
            this.singTimer = null;
        }
    }

    /**
     * イベントリスナーを解除
     */
    removeEventListeners() {
        document.removeEventListener(EVENTS.VOICE_START,       this.handleVoiceStart);
        document.removeEventListener(EVENTS.VOICE_END,         this.handleVoiceEnd);
        document.removeEventListener(EVENTS.THINKING_START,    this.handleThinkingStart);
        document.removeEventListener(EVENTS.THINKING_END,      this.handleThinkingEnd);
        document.removeEventListener(EVENTS.APP_MOTION_REQUEST, this.handleMotionRequest);
        document.removeEventListener(EVENTS.SONG_START,        this.handleSongStart);
        document.removeEventListener(EVENTS.SONG_END,          this.handleSongEnd);
        document.removeEventListener(EVENTS.GAZE_TARGET,       this.handleGazeTarget);
        this.stopSingLoop();   // タイマーを残すと、要素が外れたあとも送り続ける
    }

    /**
     * 状態を設定
     * @param {string} newState - 新しい状態（'idle', 'thinking', 'speaking'）
     */
    setState(newState) {
        const validStates = ['idle', 'thinking', 'speaking'];
        if (!validStates.includes(newState)) {
            console.warn(`無効な状態: ${newState}`);
            return;
        }

        const oldState = this.state;
        this.state = newState;

        // Shadow DOM内の要素を取得
        const section = this.shadowRoot?.querySelector('.character-section');
        if (section) {
            section.classList.remove('idle', 'thinking', 'speaking');
            section.classList.add(newState);
        }

        console.log(`🎭 アバター状態変更: ${oldState} → ${newState}`);

        // 状態変更イベントを発火
        this.dispatchEvent(new CustomEvent('avatar:statechange', {
            detail: { oldState, newState },
            bubbles: true,
            composed: true
        }));
    }

    /**
     * 現在の状態を取得
     * @returns {string} 現在の状態
     */
    getState() {
        return this.state;
    }

    /**
     * iframeにpostMessageを送信（外部から呼び出し可能）
     * @param {Object} data - 送信データ
     */
    sendToIframe(data) {
        if (this.iframe && this.iframe.contentWindow) {
            try {
                this.iframe.contentWindow.postMessage(data, '*');
            } catch (e) {
                console.warn('postMessage送信エラー:', e);
            }
        }
    }

    /**
     * モーションコード（"01"〜"26"）からgroup/indexに変換するマッピング
     */
    static MOTION_MAP = {
        '01': { group: 'Idle', index: 0 },      // 軽く微笑む（通常）
        '02': { group: 'Tap', index: 0 },       // 怒り・不満
        '03': { group: 'Tap', index: 1 },       // 悲しみ
        '04': { group: 'Idle', index: 1 },      // 楽しそうな笑顔
        '05': { group: 'Flick', index: 0 },     // 照れ
        '06': { group: 'FlickUp', index: 0 },   // 驚き
        '07': { group: 'Idle', index: 2 },      // 興奮
        '08': { group: 'Flick', index: 1 },     // ダンス
        '09': { group: 'React', index: 0 },     // うなずき（同意・傾聴）
        '10': { group: 'React', index: 1 },     // 首かしげ（疑問・興味）
        '11': { group: 'React', index: 2 },     // 共感（しょんぼり・心配）
        '12': { group: 'React', index: 3 },     // ふふっ（笑い）
        '13': { group: 'React', index: 4 },     // ウィンク（決めポーズ）
        '14': { group: 'React', index: 5 },     // やったー（弾む喜び）
        '15': { group: 'React', index: 6 },     // いやいや（否定の首振り）
        '16': { group: 'React', index: 7 },     // ごきげんゆらゆら
        '17': { group: 'React', index: 8 },     // ぴょこん（跳ね）
        '18': { group: 'React', index: 9 },     // くるっと見回し
        '19': { group: 'React', index: 10 },    // 目パチパチ（きょとん驚き）
        '20': { group: 'React', index: 11 },    // ジト目（じとー…）
        '21': { group: 'React', index: 12 },    // 考え中（うーん…）
        '22': { group: 'React', index: 13 },    // ため息（はぁ…）
        '23': { group: 'React', index: 14 },    // 苦笑い（あはは…^^;）
        '24': { group: 'React', index: 15 },    // 引き笑い（うっ、あはは…）
        '25': { group: 'React', index: 16 },    // わくわく前のめり
        '26': { group: 'React', index: 17 },    // ドヤ顔（えっへん）
        // ★27 は**歌唱専用**（LLM には選ばせない＝schemas.py の MotionCode に入れない）。
        //   歌っているときの立ち姿。21.3 秒でループし、呼吸 8拍・体の揺れ 32拍で動く
        //   （周期は『加速させて』の実測＝息継ぎが 8拍ごと。details は live2d.md）
        '27': { group: 'React', index: 18 }     // 歌っているときの立ち姿（ループ）
    };

    /**
     * モーションを送信
     * @param {string} motionCode - モーションコード（"01"〜"26"）
     * @param {string} [priority] - 'force' なら**いま流れているものを押しのけて**差し替える
     */
    sendMotion(motionCode, priority) {
        const motion = MikuAvatarLive2D.MOTION_MAP[motionCode];
        if (!motion) {
            console.warn(`[Motion] 不明なモーションコード: ${motionCode}、デフォルト(01)を使用`);
            const defaultMotion = MikuAvatarLive2D.MOTION_MAP['01'];
            this.sendToIframe({
                type: 'motion',
                group: defaultMotion.group,
                index: defaultMotion.index
            });
            return;
        }

        console.log(`🎭 モーション送信: ${motionCode} → ${motion.group}[${motion.index}]`
            + (priority ? ` (${priority})` : ''));
        window.__mikuDebugToast?.(`🎭 ${motionCode} → ${motion.group}[${motion.index}]`);
        this.sendToIframe({
            type: 'motion',
            group: motion.group,
            index: motion.index,
            ...(priority ? { priority } : {}),
        });
    }

}

// カスタム要素として登録
customElements.define('miku-avatar-live2d', MikuAvatarLive2D);