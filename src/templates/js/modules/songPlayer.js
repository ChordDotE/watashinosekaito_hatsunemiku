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
 * songPlayer.js - ミクが作った曲を歌う・流す（Phase 2 歌唱再生）
 *
 * 鳴らすのは **完成品 `master_<形>_v{N}.wav` 1本**（2026-10-03〜。歌もの・アカペラは V6 の書き出しを仕上げたもの、
 * インストは伴奏を仕上げた演奏）。どのファイルかは Gateway が決めて URL で渡すので、ここは名前を知らない。
 * 口の動きは音を解析せず、**事前計算した lipsync.json**（60fps の開き値）を再生位置で引く。
 * 音量ではなく歌詞から作った値なので、伴奏のドラム等に口が反応しない。
 *
 * **mode**（Gateway が決めて送る。ここは受け取って `song:start` に載せるだけ）:
 *   sing     自分の歌声で歌う（口の形のデータが来る）
 *   play     歌の入らない曲（インスト）を流す。**歌わない・口は動かさない**（口の形は来ない・F8①）
 *   preview  まだ歌になっていない作りかけを聴かせる（口の形は来ない）
 *   どれも `song:start`〜`song:end` の間は歌と同じ扱い（マイクを切る・無応答の声かけをしない）。
 *
 * **予約 → voice:end で開始**:
 *   ツールが呼ばれるのは LLM ループの途中で、そのあとにミクの応答音声が流れる。
 *   受信と同時に鳴らすと「聴いてね！」と曲が重なるので、受け取った時点では**保留**し、
 *   ミクが喋り終わった合図（voice:end）で鳴らす。サーバー側で待てないのは、
 *   サーバーは PCM を送り終えた時刻しか知らず、実際に音が鳴り終わった時刻を知らないため。
 *
 * 購読する CustomEvent:
 *   server:songPlay        曲の再生予約（すぐには鳴らさない）
 *   voice:end              ミクが喋り終わった → 予約があれば再生開始
 *   thinking:start         テキストで話しかけられた → フェードアウトして止める
 *   song:interrupt         歌唱中にマイクボタンが押された（＝声で話しかけたい）→ 同上
 *   caption:setEnabled     字幕モード ON → 予約も再生も捨てる（音を出せない場面では歌わない）
 *   server:disconnected    切断 → 停止
 *
 * 発火する CustomEvent:
 *   song:start / song:end  歌っている区間（マイクのミュート制御が購読する）
 */

import { EVENTS } from './events.js';
import { sendToLive2D } from './voiceOutputShared.js';

/** 中断時のフェードアウト時間（秒）。ブツッと切れると「システムが止めた」感が出る */
const FADE_SEC = 0.6;
/** フェードの更新間隔（ミリ秒） */
const FADE_STEP_MS = 40;
/**
 * 口の開きの見栄え補正（`lipsync.json` の値にかける指数）。
 *
 * `lipsync.json` は「音量エンベロープ × 母音の開き係数」という**物理的に正しい開き**で、
 * 実測は平均 0.47 / 中央値 0.40 と控えめ（「い」0.4・「う」0.35 は声を張っても開かない）。
 * 一方**会話の口パクは音量を 128 で割って 1.0 に飽和させている**ため、
 * 並べると歌のときだけ口が小さく見える（2026-08-18 実機で指摘）。
 *
 * 倍率でなく指数なのは、線形に持ち上げると「あ」も「お」も 1.0 に張り付いて
 * **母音の差が消える**から。0 は 0 のままなので m/b/p の唇閉じも保たれる。
 * **小さくするほど大きく開く**（0.6=もっと開く / 0.8=控えめ）。
 */
// ★2026-09-21: 0.7 → 0.5 に強めた（本人が「開き具合が小さい」と指摘）。
//   あわせて Cubism 側の重みも 0.8 → 1.0 にしてある（apply_patches.ps1）——
//   **重みが 0.8 のままだと最大が 0.80 で頭打ち**になり、ガンマだけ下げても上が伸びない。
const LIPSYNC_GAMMA = 0.5;
/**
 * 口を**先読みする**秒数（`lipsync.json` を引くときに足す）。
 *
 * ★**データの時点で口が音より 33ms 遅れている**（2026-09-21 実測）。
 *   `build_lipsync.py` が**フレームごとの RMS＝音量の移動平均**から開き値を作るので、
 *   平均を取る窓のぶん機械的に後ろへずれる。**系統誤差**で、曲にも歌い方にも依らない——
 *   `加速させて` を 10 秒ごと 13 区間で測ったら**全区間ぴったり +2 コマ（標準偏差 0.00）**だった。
 *   だから**一律に先読みして構わない**（ばらつくなら逆に早すぎる箇所が出るが、その心配は無い）。
 *
 * 33ms ＋ 描画の遅れ（`requestAnimationFrame` で最大 16ms ＋ 1〜2 フレーム）を見て **50ms**。
 * 人の知覚は「音が先」に敏感で「口が先」には鈍い（放送規格 ITU-R BT.1359 も
 * 音の遅れ +125ms / 先行 -45ms と非対称）ので、**多めに引くほうが安全**。
 */
const LIPSYNC_LEAD_SEC = 0.05;
/**
 * 口と音の関係をコンソールに出す間隔（ミリ秒）。**0 にすると出さない。**
 *
 * ★本人の要望（2026-09-21）:「リアルタイムのブラウザのコンソールに口の開きと、
 *   曲のどこを再生しているかを出したら比べやすいんじゃないか」。
 *   ずれの感じ方は主観なので、**鳴っている音と、そのとき送っている開き値を並べて出す**。
 *   毎フレーム出すと 60 行/秒でコンソールが溢れるので間引く。
 */
const LIPSYNC_LOG_MS = 500;
/**
 * 画面の隅に**再生位置の目印**を出すか（既定は出さない）。
 *
 * 口が音より遅く見える原因を、**本番の仕組み**と**画面に出るまで**に分けるための道具。
 * 口を動かしているのと同じループが、`currentTime` が整数秒をまたぐたびに
 * 四角の白黒を反転させる。**録画に写ったその反転が、録れた音に対して何ミリ秒ずれるか**が
 * そのまま「描画＋収録で積み上がる遅れ」になる（口の開き方や曲に依らない）。
 *
 * ⚠ **広報用の収録では必ず false に戻す**（画面に四角が写り込む）。
 */
const LIPSYNC_HUD = false;

class SongPlayer {
    constructor() {
        /** @type {{songId: string, title: string, url: string, lipsyncUrl: string}|null} 再生待ちの予約 */
        this._pending = null;
        /** @type {HTMLAudioElement|null} */
        this._audio = null;
        /** @type {number[]} lipsync の開き値（60fps） */
        this._values = [];
        /** @type {number} lipsync の fps */
        this._fps = 60;
        /** @type {number|null} requestAnimationFrame の ID */
        this._rafId = null;
        /** @type {number|null} フェードアウトの setInterval ID */
        this._fadeId = null;
        /** @type {boolean} 字幕モード中か（音を出せない＝歌わない） */
        this._captionOn = false;

        this._subscribe();
    }

    /** イベント購読（import するだけで有効になる＝voicePcmPlayer と同じ作法） */
    _subscribe() {
        document.addEventListener(EVENTS.SERVER_SONG_PLAY, (e) => this._reserve(e.detail));
        document.addEventListener(EVENTS.VOICE_END, () => this._startIfReserved());
        document.addEventListener(EVENTS.THINKING_START, () => this._fadeOutAndStop());
        document.addEventListener(EVENTS.SONG_INTERRUPT, () => this._fadeOutAndStop());
        // 切断したら**予約ごと**捨てる（残すと再接続後、次に喋り終わった瞬間に
        // 古い予約が発火して脈絡なく歌い出す）。字幕モード ON と同じ扱いに揃える
        document.addEventListener(EVENTS.SERVER_DISCONNECTED, () => {
            this._pending = null;
            this._stop();
        });
        document.addEventListener(EVENTS.CAPTION_SET_ENABLED, (e) => {
            this._captionOn = Boolean(e.detail?.enabled);
            if (this._captionOn) {
                // 音を出せない場面では歌わない。消音して進めると、字幕を戻した瞬間に
                // 途中から鳴り出して事故になるため、**予約ごと捨てて止める**
                this._pending = null;
                this._stop();
            }
        });
        console.log('[SongPlayer] 初期化完了');
    }

    /** 曲を予約する（まだ鳴らさない）。音声とリップシンクの先読みまで済ませておく */
    async _reserve(detail) {
        if (!detail?.url) return;
        if (this._captionOn) {
            console.log('[SongPlayer] 字幕モード中のため予約を無視');
            return;
        }
        this._stop();                       // 前の曲が残っていれば片付ける（予約枠は1つ＝上書き）
        // ★**前の曲の口の形を捨てる**（2026-10-03）。残すと、歌のあとに口の形の無い曲（インスト・作りかけ）を
        //   流したとき、前の歌の口の動きでミクの口が動く（`_startIfReserved` は値があれば口を動かすため）
        this._values = [];
        this._fps = 60;
        this._pending = {
            songId: detail.song_id, title: detail.title,
            url: detail.url, lipsyncUrl: detail.lipsync_url,
            // ★**歌うのか、演奏を流すのか、途中の音を聴かせるのか**（2026-09-23・play は 2026-10-03）。
            //   口の形（lipsync）が無ければ Gateway が 'play'（完成品）か 'preview'（作りかけ）にして送ってくる——
            //   **口を動かすかはフラグでなくデータで決まる**ので、ここでは受け取るだけ。
            mode: detail.mode || (detail.lipsync_url ? 'sing' : 'play'),
        };
        console.log(`[SongPlayer] 予約: ${detail.title}`);
        // 音声を先に用意する（lipsync の取得に失敗しても、音の先読みを道連れにしない）
        try {                               // 出だしがガクつかないよう、待っている間に読み込んでおく
            const audio = new Audio(detail.url);
            audio.preload = 'auto';
            audio.load();
            this._audio = audio;
        } catch (err) {
            console.error('[SongPlayer] 音声の先読みに失敗（再生時に作り直す）:', err);
        }
        if (detail.lipsync_url) {
            await this._loadLipSync(detail.lipsync_url);
        }                                   // 口の形が無い＝歌ではない。読みにいかない
    }

    /** 口の動きを読み込む。失敗したら空のままにする（再生開始時にもう一度だけ試す） */
    async _loadLipSync(url) {
        try {
            const res = await fetch(url);
            const data = await res.json();
            this._values = Array.isArray(data?.values) ? data.values : [];
            this._fps = data?.fps || 60;
            return this._values.length > 0;
        } catch (err) {
            console.error('[SongPlayer] 口の動きを読めなかった:', err);
            this._values = [];
            return false;
        }
    }

    /** ミクが喋り終わったら、予約されている曲を鳴らす */
    async _startIfReserved() {
        if (!this._pending || this._captionOn) return;
        const { title, url, lipsyncUrl, mode } = this._pending;
        this._pending = null;

        // 予約時に読めていなければ、ここで読み直す（口が閉じたまま歌うのを避ける）。
        // それでも駄目なら歌は流す（音が出ないより、口が動かない方がまし）
        if (this._values.length === 0 && lipsyncUrl) {
            const ok = await this._loadLipSync(lipsyncUrl);
            if (!ok) console.warn('[SongPlayer] 口の動きが無いまま歌う（音声のみ）');
        }

        const audio = this._audio || new Audio(url);
        this._audio = audio;
        audio.volume = 1.0;
        audio.onended = () => this._stop();
        audio.play().then(() => {
            console.log(`[SongPlayer] ${mode === 'sing' ? '歌い始め' : '再生開始'}（${mode}）: ${title}`);
            document.dispatchEvent(new CustomEvent(EVENTS.SONG_START, {
                detail: { title, mode }, bubbles: true, composed: true,
            }));
            if (this._values.length) this._startLipSync();   // 口の形が無ければ口は動かさない
        }).catch((err) => {                 // 自動再生がブロックされた等
            console.error('[SongPlayer] 再生できなかった:', err);
            this._stop();
        });
    }

    /** 再生位置から lipsync 配列を引いて Live2D へ送る（rAF＝60fps に追随） */
    _startLipSync() {
        this._stopLipSync();
        this._logAudioLatency();
        const hud = this._showHud();
        let nextLog = 0;
        const tick = () => {
            const audio = this._audio;
            if (!audio || audio.paused) return;
            const i = Math.floor((audio.currentTime + LIPSYNC_LEAD_SEC) * this._fps);
            const open = this._values[i] ?? 0;
            const value = open > 0 ? Math.pow(open, LIPSYNC_GAMMA) : 0;
            sendToLive2D({ type: 'lipSync', value });
            if (hud) {
                // 整数秒をまたぐたびに白黒が入れ替わる。**口へ値を送ったのと同じ瞬間**に切り替える
                hud.style.background = (Math.floor(audio.currentTime) % 2 === 0) ? '#fff' : '#000';
            }
            if (LIPSYNC_LOG_MS) {
                const now = performance.now();
                if (now >= nextLog) {
                    nextLog = now + LIPSYNC_LOG_MS;
                    // ★**鳴っている位置**と**そのとき送っている開き**を並べる。
                    //   棒は開き具合の目安（8 段階）。生の値も併記して、
                    //   「口が動いていない時間」が**データが 0 なのか送れていないのか**を分ける。
                    const bar = '█'.repeat(Math.round(value * 8)).padEnd(8, '·');
                    console.log(
                        `[Lip] 再生 ${audio.currentTime.toFixed(2)}s`
                        + ` | 配列 #${i}/${this._values.length}`
                        + ` | 生 ${open.toFixed(2)} → 送信 ${value.toFixed(2)} ${bar}`,
                    );
                }
            }
            this._rafId = requestAnimationFrame(tick);
        };
        this._rafId = requestAnimationFrame(tick);
    }

    /**
     * **音が実際にスピーカーから出るまでの遅れ**をコンソールに出す（歌い始めに1回）。
     *
     * 口は `audio.currentTime` に合わせて動かしているが、`currentTime` は
     * **これから鳴らす位置**であって、そこから OS の音声バッファを通るぶん遅れて耳に届く。
     * この遅れが大きいと、**口だけ先に動いて見える**。ブラウザはこの値を
     * `AudioContext.outputLatency` として持っているので、測らずに読める。
     * 鳴らさない空の AudioContext を1つ作るだけなので、曲の音には触らない。
     */
    _logAudioLatency() {
        if (!LIPSYNC_LOG_MS) return;
        try {
            const Ctx = window.AudioContext || window.webkitAudioContext;
            if (!Ctx) return;
            const ctx = new Ctx();
            const show = () => {
                const base = (ctx.baseLatency ?? 0) * 1000;
                const out = (ctx.outputLatency ?? 0) * 1000;
                console.log(
                    `[Lip] 音声出力の遅れ: 出力まで ${out.toFixed(0)}ms`
                    + `（うち内部処理 ${base.toFixed(0)}ms）/ 口の先読み ${LIPSYNC_LEAD_SEC * 1000}ms`,
                );
                ctx.close();
            };
            // resume するまで outputLatency は 0 のまま。少し待ってから読む
            ctx.resume().then(() => setTimeout(show, 300)).catch(() => ctx.close());
        } catch (err) {
            console.warn('[Lip] 音声出力の遅れを読めなかった:', err);
        }
    }

    /**
     * 再生位置の目印（白黒が反転する四角）を画面の右上に出す。
     * `LIPSYNC_HUD` が false なら何もせず null を返す。
     */
    _showHud() {
        if (!LIPSYNC_HUD) return null;
        let el = document.getElementById('lipsync-hud');
        if (!el) {
            el = document.createElement('div');
            el.id = 'lipsync-hud';
            // ★アバターに重ならない右上へ。録画から明るさを読むので**縁取りで位置を出す**
            el.style.cssText = 'position:fixed;top:0;right:0;width:96px;height:96px;'
                + 'background:#000;border:4px solid #f0f;z-index:2147483647;pointer-events:none';
            document.body.appendChild(el);
        }
        return el;
    }

    /** 目印を片付ける（歌い終わったら消す） */
    _hideHud() {
        const el = document.getElementById('lipsync-hud');
        if (el) el.remove();
    }

    _stopLipSync() {
        if (this._rafId !== null) {
            cancelAnimationFrame(this._rafId);
            this._rafId = null;
        }
        this._hideHud();
    }

    /**
     * 話しかけられたので歌うのをやめる。
     * ブツ切りだと「再生を止めた」に見えるので、少しだけ余韻を残して止める
     * （歌っている人が気づいて歌うのをやめる、に近づける）。
     */
    _fadeOutAndStop() {
        const audio = this._audio;
        if (!audio || audio.paused) return;
        if (this._fadeId !== null) return;                  // 既にフェード中
        const step = audio.volume / (FADE_SEC * 1000 / FADE_STEP_MS);
        this._fadeId = setInterval(() => {
            const next = audio.volume - step;
            if (next <= 0.01) {
                clearInterval(this._fadeId);
                this._fadeId = null;
                this._stop();
                return;
            }
            audio.volume = next;
        }, FADE_STEP_MS);
    }

    /** 停止して後片付け（口も閉じる）。歌っていなければ何もしない */
    _stop() {
        if (this._fadeId !== null) {
            clearInterval(this._fadeId);
            this._fadeId = null;
        }
        this._stopLipSync();
        const audio = this._audio;
        // ⚠ **`ended` のときは `paused` も true になる**（HTML の仕様）。`!paused` だけで見ると
        //   **最後まで歌い切った場合に song:end が飛ばず**、マイクのミュートが解除されない・
        //   口が開いたまま残る・歌の立ち姿から戻らない（2026-09-21 に実機で再現）。
        //   途中で止めたとき（マイクボタン＝song:interrupt）は再生中なので従来どおり通る。
        const wasPlaying = Boolean(audio && (!audio.paused || audio.ended));
        if (audio) {
            audio.pause();
            audio.onended = null;
            this._audio = null;
        }
        if (wasPlaying) {
            sendToLive2D({ type: 'lipSync', value: 0 });    // 開いたままにしない
            document.dispatchEvent(new CustomEvent(EVENTS.SONG_END, {
                bubbles: true, composed: true,
            }));
            console.log('[SongPlayer] 歌い終わり');
        }
    }
}

// シングルトンとしてエクスポート（import した時点で購読が始まる）
export const songPlayer = new SongPlayer();
