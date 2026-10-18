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
 * voicePcmPlayer.js - Fish Audio PCM ストリーミング再生モジュール
 *
 * 設計書: miku/src/plan/fish_audio_pcm_streaming.md
 *
 * 機能:
 * - Socket.IO から届く voice_pcm_start/chunk/end イベントを受信して再生
 * - チャプターキューで複数ループの音声を順番に再生
 * - チャプター間に不足分の無音を挿入して文間ポーズを復元（CHAPTER_GAP_TARGET_MS）
 * - AnalyserNode でリップシンク値を計算して Live2D iframe へ送信
 * - voice:start / voice:end CustomEvent を発火
 *
 * 購読する CustomEvent:
 *   server:voicePcmStart   チャプター開始
 *   server:voicePcmChunk   PCM チャンク受信
 *   server:voicePcmEnd     チャプター終了
 *   server:disconnected    切断 → 強制停止
 *   server:inactivityReminder → 強制停止
 *   thinking:start         ユーザー発話割り込み → 強制停止
 *   caption:setEnabled     字幕モード切替。ON中はチャプターを受け付けず音声再生・
 *                          リップシンクをスキップ（ONにした瞬間は再生中音声を強制停止）
 *   turnstate:change       マスターの発話中は再生を保留する（下記）
 *
 * 発火する CustomEvent:
 *   voice:start  実際に音が出始めた時（チャプター先頭のチャンク到着時）
 *   voice:end    全チャプター再生完了時 / 強制停止時 / データ待ちギャップに入った時
 *                （「実際に音が出ている区間」を表す。ギャップ中はマイクミュートが解除される）
 *
 * ## 知覚できない長さの状態変化は出さない（2026-08-16）
 *
 * データ待ちに入っても **すぐには voice:end を出さない**（`GAP_HOLD_MS` の猶予）。
 * フィラー再生中に本回答の音声が用意できていると、チャプター切替の一瞬だけ手元の
 * バッファが空になり、**2ms 後にはデータが届いている**。この 2ms で voice:end→voice:start
 * を出すと、`audible` が落ちて戻り、マイクの mute/unmute と Gateway の STT セッション
 * 張り直しが連打される。人間には知覚できず（音の途切れと分かるのは概ね 10ms 以上）、
 * VAD の onset 判定（200ms）にも届かないので、開けて得るものは無い。
 *
 * 発火する側が整った信号を出し、状態庫（turnState）は届いた事実を信じる——という
 * 層の分担にしている。規約は basic_design §4.6「状態を使う側の規約」。
 *
 * ## マスターの発話中は音を出さない（2026-08-13）
 *
 * 再生を始めると `voice:start` → マイクがミュートされ、**話している途中の録音が
 * 打ち切られる**（実機で発生。1.7秒で切られ、STT の確定に至らず話者照合も走らなかった）。
 * ミュートを外すのは回り込み（echoCancellation は Bluetooth 出力時のみ ON）のため不可。
 *
 * そこで**音を出す側を待たせる**。`turnState.userTurnOpen` が立っている間は
 * `voice:start` を発火せず、PCM は `_pendingChunks` に溜めておく。
 * マスターが言い終わったら（Smart Turn の終話判定）そこから鳴らす。
 * 判定に `userSpeaking` を使わないのは、VAD の無音 800ms では言いよどみの隙間に
 * 差し込まれてしまうため（turnState.js の説明を参照）。
 */

import { EVENTS } from './events.js';
import { appState } from './appState.js';
import {
    LipSyncLoop,
    dispatchVoiceStart,
    dispatchVoiceEnd,
} from './voiceOutputShared.js';

const WORKLET_URL = '/js/audio-pcm-worklet.js';
const SAMPLE_RATE = 44100;
// プログレッシブフィード: バッファ残量がこの値（秒）を下回る間だけチャンクを供給する。
// リングバッファ（10秒）に対し十分なマージンを取り、溢れを防ぐ。
const TARGET_BUFFER_SECONDS = 4;
const TARGET_SAMPLES = SAMPLE_RATE * TARGET_BUFFER_SECONDS;
// データ待ちを「ギャップ」と認めるまでの猶予。これより短い空白でマイクを開けても、
// **VAD の onset 判定（200ms）に届かず発話を拾えない**ため、開けた側の代償
// （マイクの mute/unmute・Gateway の STT セッション張り直し）だけが残る。
//
// 実測（2026-08-16）: フィラー再生中に本回答の音声が用意できていると、チャプター切替の
// 一瞬だけ手元のバッファが空になり、**2ms 後にはデータが届いていた**。この 2ms のために
// voice:end → voice:start が連続し、audible が落ちて戻る＝マイクが開いて即閉じていた。
// 人間には知覚できない（音の途切れとして分かるのは概ね 10ms 以上）ので、得るものはゼロ。
const GAP_HOLD_MS = 400;
// 再生保留を諦めるまでの時間。**「言葉として認識されなくなってから」**数える
// （stt:speechEvidence が届くたびに引き直す）。VAD が雑音で RECORDING に張り付いたり
// 終話判定が来なかった場合に、ミクが永久に黙るのを防ぐ保険。
//
// 「保留開始から」で数えていた頃は、**長く話し続けるほど発動した**（畳み込みが効いている
// ときこそ発動する）。実害: 15秒話し続けたマスターの声に、2.3秒時点で作られたフィラー
// 「そうだね」が被さって鳴った（2026-08-15）。音量（mic:speechStart）で数え直すのは不可
// ——雑音で張り付いた場合こそリセットが掛かり続け、保険が永久に効かなくなるため。
const HOLD_MAX_MS = 15000;
// フィラー（chap0＝「そうだね」等の相づち）の消費期限。ターン開始直後に作られる短い一言なので、
// 長く保留されると**話題が変わったあとに突然鳴る**。上限満了で解除するときは、
// これを過ぎたフィラーは鳴らさずに捨てる（本応答は捨てない＝黙り込ませない）。
const FILLER_STALE_MS = 10000;
const FILLER_CHAPTER = 0;
// チャプター間に確保する文間ポーズの目安。Fish は1ストリーム内なら文末に
// 650〜950ms のポーズを生成するが、チャプター終端の flush では付かないため、
// 境界（フィラー→本応答・ツール応答間）がゼロ連結になり「間が詰まる」
// （実測: test/tts_pause_check/results.md）。**固定で足すのではなく**、
// 前チャプターの再生終了からの実経過を差し引いた不足分だけ挿入する
// （生成待ちで既に空いた間に重ねて足さない）。
const CHAPTER_GAP_TARGET_MS = 950;

class VoicePcmPlayer {
    constructor() {
        this.audioContext = null;
        this.workletNode  = null;
        this.analyser     = null;
        this._lipSync     = new LipSyncLoop();

        // チャプターキュー: { chapter, chunks: string[], finished: bool }[]
        this._queue = [];
        // 現在受信中のチャプターエントリ
        this._receivingChapter = null;
        // 現在再生中のチャプターエントリ
        this._playingChapter = null;
        // ターン中に一度でも voice:start を発火したか
        this._voiceStarted = false;
        // ギャップ確定待ちのタイマー（GAP_HOLD_MS 内にデータが来たら取り消す）
        this._gapTimer = null;
        // バッファ残量推定用（現チャプターで worklet に送った/再生済みのサンプル数）
        this._fedSamples = 0;
        this._playedSamples = 0;
        // 強制停止の検出用世代番号。_playNext の await 中に _forceStop が入った場合、
        // 復帰後にこの番号の不一致で再生を中断する
        this._generation = 0;
        // 直前のチャプターの再生が終わった時刻（performance.now()）。
        // 次チャプターの音を流し始めるとき、ここからの経過と CHAPTER_GAP_TARGET_MS の
        // 差分だけ無音を挿入する。0 は「挿入しない」（ターン跨ぎ・強制停止後）
        this._lastAudioEndAt = 0;
        // マスターの番が続いているか（turnState の写し）。true の間は音を出さない
        this._userTurnOpen = false;
        // 実際に保留したか。保留していないときに解放処理を走らせないための印
        this._held = false;
        this._holdTimer = null;

        this._initServerEventListeners();
    }

    // -----------------------------------------------------------------------
    // イベントリスナー登録
    // -----------------------------------------------------------------------

    _initServerEventListeners() {
        document.addEventListener(EVENTS.SERVER_VOICE_PCM_START, (e) => {
            this._onStart(e.detail);
        });
        document.addEventListener(EVENTS.SERVER_VOICE_PCM_CHUNK, (e) => {
            this._onChunk(e.detail);
        });
        document.addEventListener(EVENTS.SERVER_VOICE_PCM_END, (e) => {
            this._onEnd(e.detail);
        });
        document.addEventListener(EVENTS.SERVER_DISCONNECTED,       () => this._forceStop());
        document.addEventListener(EVENTS.SERVER_INACTIVITY_REMINDER, () => this._forceStop());
        document.addEventListener(EVENTS.THINKING_START,            () => this._forceStop());
        // 字幕モードON: 再生中の音声を即座に止める（以降のチャプターは _onStart で破棄）
        document.addEventListener(EVENTS.CAPTION_SET_ENABLED, (e) => {
            if (e.detail?.enabled) this._forceStop();
        });
        // 言葉として認識された＝本当に喋っている → 諦めるまでの時間を数え直す
        document.addEventListener(EVENTS.STT_SPEECH_EVIDENCE, () => this._restartHoldTimer());
        // マスターの番が閉じたら、保留していた音を鳴らし始める
        document.addEventListener(EVENTS.TURN_STATE_CHANGE, (e) => {
            const open = Boolean(e.detail?.userTurnOpen);
            const was = this._userTurnOpen;
            this._userTurnOpen = open;
            if (was && !open) this._releaseHold('マスターが言い終わった');
        });
    }

    // -----------------------------------------------------------------------
    // 発話中の保留
    // -----------------------------------------------------------------------

    /** いま音を出してよいか。マスターが話している間は出さない */
    _shouldHold() {
        return this._userTurnOpen;
    }

    /** 保留に入る（多重呼び出しは無視）。言葉が途切れたまま上限に達したら諦めて鳴らす */
    _holdPlayback() {
        if (this._held) return;
        this._held = true;
        console.log('[VoicePcmPlayer] マスターの発話中 → 再生を保留（音は溜めておく）');
        this._restartHoldTimer();
    }

    /**
     * 「諦めるまで」のタイマーを引き直す（保留中のみ）。
     * 言葉が認識されるたびに呼ばれるので、話し続けている限り発動しない。
     */
    _restartHoldTimer() {
        if (!this._held) return;
        if (this._holdTimer) clearTimeout(this._holdTimer);
        this._holdTimer = setTimeout(() => {
            this._holdTimer = null;
            console.warn(`[VoicePcmPlayer] 言葉が ${HOLD_MAX_MS}ms 認識されなかった → 再生を開始する`);
            this._releaseHold('上限');
        }, HOLD_MAX_MS);
    }

    /** 保留を解いて、溜めておいたぶんから鳴らし始める */
    _releaseHold(reason) {
        if (!this._held) return;
        this._held = false;
        if (this._holdTimer) {
            clearTimeout(this._holdTimer);
            this._holdTimer = null;
        }
        const entry = this._playingChapter;
        if (!entry || entry._pendingChunks === null) return;
        // 古くなった相づちは鳴らさない（話題が変わったあとの「そうだね」は会話を壊す）
        const age = Date.now() - (entry.startedAt || 0);
        if (entry.chapter === FILLER_CHAPTER && age > FILLER_STALE_MS) {
            console.warn(`[VoicePcmPlayer] フィラーが古い（${(age / 1000).toFixed(1)}秒）→ 鳴らさずに捨てる`);
            entry._pendingChunks = [];
            this._onChapterEnded();
            return;
        }
        console.log(`[VoicePcmPlayer] 保留を解除（${reason}）→ `
            + `チャプター${entry.chapter} を再生（溜まり ${entry._pendingChunks.length}チャンク）`);
        if (entry._pendingChunks.length > 0 || entry.finished) {
            this._ensureVoiceStarted();
            this._feedPendingChunks(entry);
        }
    }

    /**
     * 「鳴らし始める」の唯一の入口。保留中なら音を出さずに溜めるだけにする。
     *
     * ここを通さずに `_ensureVoiceStarted` / `_feedPendingChunks` を直接呼ぶと、
     * その経路だけ保留を素通りしてマスターの発話を打ち切ってしまう
     * （経路ごとに判定を書かない — 2026-08-12 の話者照合と同じ失敗をしない）。
     */
    _startOrHold(entry) {
        if (this._shouldHold()) {
            this._holdPlayback();
            return;
        }
        this._ensureVoiceStarted();
        this._feedPendingChunks(entry);
    }

    // -----------------------------------------------------------------------
    // サーバーイベントハンドラ
    // -----------------------------------------------------------------------

    _onStart({ chapter }) {
        if (appState.isCaptionActive()) {
            // 字幕モード中は音声再生をスキップ（チャンクはエントリ未登録のため自動破棄）
            console.log(`[VoicePcmPlayer] チャプター${chapter} は字幕モードのためスキップ`);
            return;
        }
        console.log(`[VoicePcmPlayer] チャプター${chapter} 開始`);
        // _pendingChunks: バックログをプログレッシブフィード中の配列、null なら直接書き込みモード
        // _gapFilled: チャプター間の差分無音を挿入済みか（最初の実データ供給時に1回だけ）
        const entry = { chapter, chunks: [], finished: false, _pendingChunks: null, startedAt: Date.now(), _gapFilled: false };
        this._queue.push(entry);
        this._receivingChapter = entry;

        if (!this._playingChapter) {
            this._playNext();
        }
    }

    _onChunk({ chapter, data }) {
        const entry = this._findEntry(chapter);
        if (!entry) return;

        if (this._playingChapter === entry) {
            // データ待ちギャップ中に最初のチャンクが届いた → ここから鳴り始める
            // （マスターが話している間は _startOrHold が溜めるだけにする）
            if (entry._pendingChunks !== null) {
                entry._pendingChunks.push(data);
                // 供給はバッファ残量を見てペーシングされる（溢れない範囲でのみ流す）
                this._startOrHold(entry);
            } else {
                this._ensureVoiceStarted();
                this._pushChunkToWorklet(data);
            }
        } else {
            // 待機中チャプターはバッファに溜める
            entry.chunks.push(data);
        }
    }

    _onEnd({ chapter, no_audio }) {
        if (no_audio) {
            // 音声なし通知（response_text が空だった場合）は無視
            return;
        }
        const entry = this._findEntry(chapter);
        if (!entry) return;

        entry.finished = true;
        console.log(`[VoicePcmPlayer] チャプター${chapter} データ受信完了`);

        if (this._playingChapter === entry && this.workletNode) {
            if (entry._pendingChunks === null) {
                // 既に全供給済み → 即座に finish を送る
                this.workletNode.port.postMessage({ type: 'finish' });
            } else {
                // 残りを供給し、空になれば _feedPendingChunks が finish を送る。
                // pending が空のまま finished になったケースのデッドロックもここで解消する。
                // 保留中は供給しない（解除時に _releaseHold が finish まで面倒を見る）
                this._startOrHold(entry);
            }
        }
    }

    // -----------------------------------------------------------------------
    // チャプター再生
    // -----------------------------------------------------------------------

    async _playNext() {
        if (this._queue.length === 0) {
            this._playingChapter = null;
            return;
        }

        const entry = this._queue.shift();
        // NOTE: _playingChapter は worklet 準備完了後にセットする。
        // await 中に到着したチャンクを entry.chunks へ確実に溜めるため、
        // ここでは意図的にセットしない。
        console.log(`[VoicePcmPlayer] チャプター${entry.chapter} 再生開始`);

        // await 中の強制停止（字幕ON切替など）を検出するため世代番号を記録
        const generation = this._generation;

        try {
            await this._initAudioContext();
        } catch (e) {
            console.error('[VoicePcmPlayer] AudioContext 初期化エラー:', e);
            this._onChapterEnded();
            return;
        }

        if (this._generation !== generation) {
            // await 中に _forceStop が入った → このチャプターは破棄して再生しない
            console.log(`[VoicePcmPlayer] チャプター${entry.chapter} は強制停止済みのためスキップ`);
            return;
        }

        // 前の WorkletNode を切断
        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }

        // AudioWorkletNode を生成
        this.workletNode = new AudioWorkletNode(this.audioContext, 'pcm-ring-buffer');
        this.workletNode.port.postMessage({ type: 'reset' });

        // AnalyserNode（リップシンク用）
        if (this.analyser) this.analyser.disconnect();
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
        this.workletNode.connect(this.analyser);
        this.analyser.connect(this.audioContext.destination);

        // worklet からのメッセージハンドラ
        this.workletNode.port.onmessage = (e) => {
            const t = e.data.type;
            if (t === 'ended') {
                console.log(`[VoicePcmPlayer] チャプター${entry.chapter} 再生完了`);
                this._onChapterEnded();
            } else if (t === 'buffer_low') {
                if (entry._pendingChunks !== null) {
                    this._feedPendingChunks(entry);
                }
            } else if (t === 'progress') {
                // 再生位置更新 → バッファ残量が下がっていれば追加供給
                this._playedSamples = e.data.played;
                if (entry._pendingChunks !== null) {
                    this._feedPendingChunks(entry);
                }
            } else if (t === 'overflow') {
                console.warn(`[VoicePcmPlayer] ⚠️ チャプター${entry.chapter} リングバッファ溢れ! dropped=${e.data.dropped}samples (${(e.data.dropped/SAMPLE_RATE).toFixed(2)}s) chunk=${e.data.chunkSamples}samples available=${e.data.available}samples`);
            }
        };

        this._lipSync.start(this.analyser);

        // worklet 準備完了 → _playingChapter をセット
        // await 中に entry.chunks に溜まったバックログをここで処理する
        this._playingChapter = entry;
        // バッファ残量カウンタをリセット（worklet の reset と整合）
        this._fedSamples = 0;
        this._playedSamples = 0;
        console.log(`[VoicePcmPlayer] チャプター${entry.chapter} worklet 準備完了 backlog=${entry.chunks.length}チャンク finished=${entry.finished}`);

        // 常にプログレッシブフィードモードで開始（バッファ残量を見て供給をペーシング）
        entry._pendingChunks = entry.chunks;
        entry.chunks = [];

        // voice:start/end は「実際に音が出ている区間」を表す（2026-08-03 変更）。
        // データ未着でチャプターを開いて待つ間（フィラー後の考え中ギャップ等）は
        // voice:end 扱いにしてマイクミュートを解除させ、最初のチャンク到着
        // （_onChunk / ここのバックログ）で voice:start し直す。
        // 以前はチャプター開始で無条件に voice:start していたため、考え中の間も
        // マイクが MUTED のままになり、追加発話（畳み込み）の音声入口が塞がっていた
        if (entry._pendingChunks.length > 0 && !this._shouldHold()) {
            this._ensureVoiceStarted();
        } else {
            // データ未着、またはマスターが話している最中 → 鳴らさない＝マイクを開けておく
            this._markVoiceGap(entry.chapter);
        }
        this._startOrHold(entry);
        // pending が空の場合: _onChunk でチャンク到着時に供給
        // progress/buffer_low イベントでも残量に応じて追加供給される
    }

    /** 音が出始める（出続けている）ことを保証する。未発火なら voice:start を発火 */
    _ensureVoiceStarted() {
        if (!this._voiceStarted) {
            this._voiceStarted = true;
            dispatchVoiceStart();
        }
    }

    _cancelGapTimer() {
        if (this._gapTimer) {
            clearTimeout(this._gapTimer);
            this._gapTimer = null;
        }
    }

    /**
     * データ待ちに入った。**すぐには voice:end を出さない**。
     *
     * GAP_HOLD_MS 待って、なお次のデータが来なければ「本当に間が空いた」とみなして
     * 発火する（マイク解放＝考え中の追加発話の入口）。猶予内に届けば
     * `_ensureVoiceStarted` が取り消すので、状態は一度も落ちない。
     *
     * 発火する側の責務として、**知覚できない長さの状態変化を出さない**
     * （basic_design §4.6「状態を使う側の規約」）。
     *
     * ★このチャプターがまだ `finished`（サーバーからの終端）を受け取っていなければ、
     * **同じ発話の続きがまだ来る**（ツール呼び出し待ち等）。`moreExpected: true` を
     * 添えて turnState 側に伝える——でないと turnBelief が CONFIRMED のときに
     * 「ターンが終わった」と誤認され、考え中を示す薄い緑（data-turn-open）が外れて
     * 通常の LISTENING 色に見えてしまう（2026-09-22 実機で確認）。
     */
    _markVoiceGap(chapter) {
        if (!this._voiceStarted || this._gapTimer) return;
        this._gapTimer = setTimeout(() => {
            this._gapTimer = null;
            if (!this._voiceStarted) return;
            this._voiceStarted = false;
            const entry = this._findEntry(chapter);
            const moreExpected = !(entry?.finished ?? true);
            console.log(`[VoicePcmPlayer] チャプター${chapter} データ待ちギャップ → voice:end（マイク解放・moreExpected=${moreExpected}）`);
            dispatchVoiceEnd({ moreExpected });
        }, GAP_HOLD_MS);
    }

    _onChapterEnded() {
        // 次チャプターとの間に挟む無音の起点（実際に音が鳴り止んだ時刻）
        this._lastAudioEndAt = performance.now();
        this._playingChapter = null;

        if (this._queue.length > 0) {
            // 次チャプターへ
            this._playNext();
        } else {
            // 全チャプター完了（データ待ちギャップで既に voice:end 済みなら二重発火しない）。
            // 猶予中のギャップ判定は破棄する（ここで確定して出すので二重に出さない）
            this._cancelGapTimer();
            this._lipSync.stop();
            if (this._voiceStarted) {
                this._voiceStarted = false;
                dispatchVoiceEnd();
            }
        }
    }

    // -----------------------------------------------------------------------
    // 強制停止
    // -----------------------------------------------------------------------

    _forceStop() {
        // 世代番号は先にバンプする: _playNext が await 中（queue から取り出し済みで
        // アイドルに見える瞬間）でも、復帰後の再生を確実に中断させるため
        this._generation++;

        // 保留を残さない。残すと上限タイマーだけが生き延び、破棄済みチャプターに対して
        // 解除処理が走る（_playingChapter は null なので実害はないが、状態を持ち越さない）
        this._held = false;
        if (this._holdTimer) {
            clearTimeout(this._holdTimer);
            this._holdTimer = null;
        }
        // 破棄したチャプターに対して遅れて voice:end が飛ばないよう、ギャップ判定も捨てる
        this._cancelGapTimer();

        if (!this._voiceStarted && this._queue.length === 0 && !this._playingChapter) {
            this._receivingChapter = null;
            return;  // 再生中でなければ以降のクリーンアップは不要
        }
        console.log('[VoicePcmPlayer] 強制停止');

        this._queue = [];
        this._receivingChapter = null;
        this._playingChapter = null;
        this._lastAudioEndAt = 0;

        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }

        this._lipSync.stop();

        if (this._voiceStarted) {
            this._voiceStarted = false;
            dispatchVoiceEnd();
        }
    }

    // -----------------------------------------------------------------------
    // ユーティリティ
    // -----------------------------------------------------------------------

    _findEntry(chapter) {
        if (this._receivingChapter?.chapter === chapter) return this._receivingChapter;
        return this._queue.find(e => e.chapter === chapter) ?? null;
    }

    /**
     * バッファ残量（_fedSamples - _playedSamples）が TARGET_SAMPLES を下回る間だけ
     * entry._pendingChunks を worklet に流す。これによりリングバッファ溢れを防ぐ。
     * 全消化＆finished なら finish を送る。
     */
    _feedPendingChunks(entry) {
        // 最初の実データを流す直前に、前チャプターとの間の不足分の無音を挟む。
        // ここが全供給経路（_playNext バックログ / _onChunk 到着 / _releaseHold）の合流点
        if (!entry._gapFilled && entry._pendingChunks.length > 0) {
            entry._gapFilled = true;
            this._insertChapterGap(entry.chapter);
        }
        let fedBytes = 0;
        let count = 0;
        while (entry._pendingChunks.length > 0) {
            const buffered = this._fedSamples - this._playedSamples;
            if (buffered >= TARGET_SAMPLES) break;  // バッファ十分 → 供給を止める
            const data = entry._pendingChunks.shift();
            const bytes = this._pushChunkToWorklet(data);
            this._fedSamples += bytes / 2;  // Int16 → サンプル数
            fedBytes += bytes;
            count++;
        }

        if (count > 0) {
            const buffered = (this._fedSamples - this._playedSamples) / SAMPLE_RATE;
            console.log(`[VoicePcmPlayer] チャプター${entry.chapter} フィード ${count}チャンク (${(fedBytes/1024).toFixed(1)}KB) 残り${entry._pendingChunks.length}チャンク buffered=${buffered.toFixed(2)}s`);
        }

        if (entry._pendingChunks.length === 0 && entry.finished) {
            // 全データ消化完了 → finish 送信
            entry._pendingChunks = null;
            console.log(`[VoicePcmPlayer] チャプター${entry.chapter} プログレッシブフィード完了 → finish 送信`);
            if (this.workletNode) {
                this.workletNode.port.postMessage({ type: 'finish' });
            }
        }
    }

    /**
     * チャプター間の不足分の無音を worklet に挿入する。
     * 挿入量 = CHAPTER_GAP_TARGET_MS − 前チャプター再生終了からの実経過（負なら挿入なし）。
     * 生成待ち・考え中で既に間が空いていた場合は自然の間が優先され、二重に間延びしない。
     */
    _insertChapterGap(chapter) {
        if (!this._lastAudioEndAt || !this.workletNode) return;
        const elapsed = performance.now() - this._lastAudioEndAt;
        const ms = Math.round(CHAPTER_GAP_TARGET_MS - elapsed);
        if (ms < 20) return;  // 20ms 未満は知覚できず、送るだけ無駄
        const samples = Math.floor(SAMPLE_RATE * ms / 1000);
        const buffer = new ArrayBuffer(samples * 2);  // Int16 のゼロ埋め＝無音
        this.workletNode.port.postMessage({ type: 'chunk', buffer }, [buffer]);
        this._fedSamples += samples;
        console.log(`[VoicePcmPlayer] チャプター${chapter} の前に無音 ${ms}ms を挿入`
            + `（前チャプター終了から ${Math.round(elapsed)}ms 経過）`);
    }

    /** worklet にPCMチャンクを送る。送ったバイト数を返す。 */
    _pushChunkToWorklet(base64Data) {
        if (!this.workletNode) return 0;
        // **実際に音が出た**のでギャップ確定待ちを取り消す。
        // 取り消しを `_ensureVoiceStarted` に置くと、`_startOrHold` が
        // データの有無に関わらずそれを呼ぶため、仕掛けた直後に必ず消えてしまう
        // （2026-08-16 に一度そう実装し、**voice:end が永久に出ずマイクが開かなくなった**）
        this._cancelGapTimer();
        try {
            const binary = atob(base64Data);
            const buffer = new ArrayBuffer(binary.length);
            const view = new Uint8Array(buffer);
            for (let i = 0; i < binary.length; i++) {
                view[i] = binary.charCodeAt(i);
            }
            this.workletNode.port.postMessage({ type: 'chunk', buffer }, [buffer]);
            return binary.length;
        } catch (e) {
            console.error('[VoicePcmPlayer] チャンク書き込みエラー:', e);
            return 0;
        }
    }

    async _initAudioContext() {
        if (!this.audioContext) {
            this.audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
            await this.audioContext.audioWorklet.addModule(WORKLET_URL);
            console.log('[VoicePcmPlayer] AudioContext 初期化完了');
        }
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
    }

}

// シングルトンとしてエクスポート
export const voicePcmPlayer = new VoicePcmPlayer();