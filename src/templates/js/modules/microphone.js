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
 * MicrophoneController - マイク入力・VAD・状態管理モジュール
 *
 * ブラウザ側のリアルタイム音声入力を管理する。
 * マイクからの音声をキャプチャし、VAD（Voice Activity Detection）で
 * 発話の開始・終了を検知し、録音データを蓄積する。
 *
 * ## 状態遷移（plan/realtime_voice_input.md §4.1）
 *
 *   OFF → LISTENING → RECORDING → PROCESSING → LISTENING → ...
 *   （LISTENING/RECORDING/PROCESSING ⇄ MUTED: ミク音声の再生中の一時ミュート）
 *
 *   OFF:        マイク未使用。リソース未取得。
 *   LISTENING:  VAD監視中。getUserMedia + AudioWorklet が動作中。
 *   RECORDING:  PCM蓄積中。VADが音声を検知した状態。
 *   PROCESSING: STT結果待ち。録音データの送信が完了し、結果を待っている状態。
 *   MUTED:      一時ミュート。ストリームは保持したまま入力を捨てる（mute()/unmute()）。
 *               stop()→start() のマイク再取得ラグ（約1秒）なしで即再開できる。
 *
 * ## カスタムイベント（document に dispatch）
 *
 *   mic:stateChange  - 状態遷移時に発火。detail: { state, prevState }
 *   mic:speechStart  - 発話開始検知時に発火。（割り込み判定に使用）
 *   mic:speechEnd    - 発話終了検知時に発火。detail: { audioData: Float32Array, sampleRate: number }
 *   mic:level        - 音声レベル通知（約60fps）。detail: { rms, threshold, isSpeech }
 *
 * ## 使用例（ブラウザコンソールから手動テスト）
 *
 *   const { MicrophoneController } = await import('/js/modules/microphone.js');
 *   const { EVENTS } = await import('/js/modules/events.js');
 *   const mic = new MicrophoneController();
 *   document.addEventListener(EVENTS.MIC_STATE_CHANGE, e => console.log('state:', e.detail));
 *   document.addEventListener(EVENTS.MIC_SPEECH_START, () => console.log('speech start!'));
 *   document.addEventListener(EVENTS.MIC_SPEECH_END, e => console.log('speech end, samples:', e.detail.audioData.length));
 *   await mic.start();
 *   // マイクに向かって話す → ログが出る
 *   mic.stop();
 */

import { EVENTS } from './events.js';

// ============================================================================
// 定数
// ============================================================================

/** マイクコントローラーの状態定義 */
const MicState = Object.freeze({
    OFF: 'OFF',               // マイク未使用
    LISTENING: 'LISTENING',   // VAD監視中
    RECORDING: 'RECORDING',   // PCM蓄積中（発話検知）
    PROCESSING: 'PROCESSING', // STT結果待ち
    MUTED: 'MUTED',           // 一時ミュート（ミク音声の再生中。ストリームは保持し処理だけ止める）
});

/** VADのデフォルト設定 */
const VAD_DEFAULTS = Object.freeze({
    /**
     * RMS閾値。この値を超えると「音声あり」と判定する。
     * 0.01〜0.03 が一般的。環境ノイズが多い場合は上げる。
     */
    threshold: 0.015,

    /**
     * onset判定ウィンドウ（ミリ秒）。
     * この時間内に speechOnsetRatio 以上のチャンクが閾値を超えたら発話開始と認定する。
     * 200ms推奨。キーボード打鍵(〜80ms)は短いため誤検知しにくい。
     */
    speechOnsetDuration: 200,

    /**
     * onset判定の閾値超えチャンク割合（0〜1）。
     * ウィンドウ内でこの割合以上のチャンクが閾値を超えたら発話開始と認定する。
     * 0.8（80%）推奨。「えっと、あの」など途中に無音が入る発話でも検知できる。
     */
    speechOnsetRatio: 0.8,

    /**
     * 無音猶予（ミリ秒）。発話中に一瞬無音になっても、
     * この時間内なら発話継続とみなす。
     * 日本語は助詞の前に間が空くことがあるため、800ms推奨。
     */
    silenceHangover: 800,

    /**
     * 最大録音時間（ミリ秒）。この時間を超えたら強制的に発話終了とする。
     * サーバー側のSTTの処理負荷を抑えるための安全弁。
     */
    maxRecordingDuration: 30000,
});

// ============================================================================
// MicrophoneController クラス
// ============================================================================

export class MicrophoneController {
    /**
     * @param {Object} [options] - VAD設定のオーバーライド
     * @param {number} [options.vadThreshold] - RMS閾値
     * @param {number} [options.silenceHangover] - 無音猶予（ms）
     * @param {number} [options.maxRecordingDuration] - 最大録音時間（ms）
     */
    constructor(options = {}) {
        // --- 状態 ---
        this._state = MicState.OFF;

        // --- VAD設定 ---
        this._vadThreshold       = options.vadThreshold       ?? VAD_DEFAULTS.threshold;
        this._speechOnsetDuration= options.speechOnsetDuration?? VAD_DEFAULTS.speechOnsetDuration;
        this._speechOnsetRatio   = options.speechOnsetRatio   ?? VAD_DEFAULTS.speechOnsetRatio;
        this._silenceHangover    = options.silenceHangover    ?? VAD_DEFAULTS.silenceHangover;
        this._maxRecordingDuration=options.maxRecordingDuration??VAD_DEFAULTS.maxRecordingDuration;

        // --- Web Audio API リソース（start() で初期化、stop() で解放）---
        this._audioContext = null;     // AudioContext
        this._mediaStream = null;      // getUserMedia() の戻り値
        this._sourceNode = null;       // MediaStreamSourceNode
        this._workletNode = null;      // AudioWorkletNode

        // --- 録音バッファ ---
        // RECORDING 中に AudioWorklet から受け取った PCM チャンクを蓄積する。
        // 発話終了時にこれらを結合して1つの Float32Array にする。
        this._recordingChunks = [];

        // --- リアルタイムSTT用チャンクバッファ ---
        // AudioWorklet の ~2.67ms 単位チャンクを 100ms 分溜めて mic:audioChunk を発火する。
        // _rtTargetSamples は start() 後に sampleRate × 0.1 に設定される。
        this._rtAccumBuffer = [];
        this._rtAccumSamples = 0;
        this._rtTargetSamples = 0;

        // --- リアルタイムSTT用プレロールバッファ ---
        // onset判定中（LISTENING）の音声を仮蓄積し、RECORDING遷移時にまとめてSTTに送る。
        // 発話冒頭の欠落を防ぐ。onset後にリセット。
        this._rtPreRollBuffer = [];

        // --- VAD 内部状態 ---
        this._silenceStart = null;         // 無音が始まった時刻（ms）
        this._recordingStartTime = null;   // 録音開始時刻（ms）。タイムアウト判定用
        this._lastLevelTime = 0;           // 最後にmic:levelを発火した時刻（間引き用）
        this._speechOnsetStart = null;     // onset判定ウィンドウ開始時刻（ms）
        this._onsetChunks = [];            // onset期間中のPCMチャンク（batch STT用）
        this._onsetSpeechCount = 0;        // onset期間内で閾値を超えたチャンク数
        this._onsetTotalCount = 0;         // onset期間内の総チャンク数
    }

    // ========================================================================
    // 公開 API
    // ========================================================================

    /** 現在の状態を取得 */
    getState() {
        return this._state;
    }

    /**
     * マイクを起動し、LISTENING 状態に遷移する。
     *
     * 処理内容:
     *   1. getUserMedia() でマイクアクセスを取得
     *   2. AudioContext を作成
     *   3. AudioWorklet にプロセッサを登録
     *   4. マイク → AudioWorkletNode を接続
     *   5. Worklet からの PCM データを onAudioChunk() で処理
     *
     * @throws {Error} マイクアクセスが拒否された場合
     */
    async start() {
        if (this._state !== MicState.OFF) {
            console.warn('[Microphone] すでに起動中です (state=' + this._state + ')');
            return;
        }

        try {
            // 1. マイクアクセスを取得（ユーザーに許可ダイアログが出る）
            //
            // **`echoCancellation` は入力だけでなく出力にも効く**。ここを間違えると、
            // 「マイクが聞こえない」か「ミクの声が劣化する」のどちらかが必ず起きる。
            // 環境によって正解が**真逆**なので、機能検出で分岐する（basic_design §4.7）:
            //
            //   WebKit（iOS/iPadOS）: AGC を独立して制御できず、EC が音声処理ユニットごとの
            //     スイッチを兼ねる。**EC を切ると AGC も死に**、内蔵マイクの入力が
            //     RMS 0.002 未満まで痩せて STT が一切反応しない（2026-08-16 実測）
            //   Blink（Android/PC）: AGC は独立して効く。逆に **EC を入れると OS が
            //     通話モード（MODE_IN_COMMUNICATION）に入り、出力が通話音量・受話口寄りの
            //     経路になってミクの声が劣化する**（2026-08-18 実機で確認）
            //
            // 判定には `autoGainControl` のサポート申告を使う。「AGC を独立して指定できるか」が
            // そのまま両者を分ける性質なので、UA 判定より素性がよく、将来 WebKit が AGC を
            // 実装したら自動的に Blink 側の扱いへ寄る。経緯は decisions.md 2026-08-16 / 2026-08-18。
            const supported = navigator.mediaDevices.getSupportedConstraints?.() || {};
            const agcIndependent = !!supported.autoGainControl;
            this._mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: agcIndependent
                    // AGC が独立して効く環境: EC は切る（通話モードに落とさない）
                    ? { echoCancellation: false, noiseSuppression: true, autoGainControl: true }
                    // AGC を独立制御できない環境（WebKit）: 既定に委ねる＝EC が入り AGC も効く
                    : true,
                video: false,
            });

            // **実際に適用された設定**を残す（要求した制約ではなく結果を見る）。
            // 端末によって効く制約が違い、キーごと存在しないこともある。
            // 2026-08-16 の「iPad だけ文字起こしされない」は、この1行があれば即断できた
            const track = this._mediaStream.getAudioTracks()[0];
            if (track && track.getSettings) {
                console.log('[Microphone] 適用された設定: ' + JSON.stringify(track.getSettings()));
            }

            // 2. AudioContext を作成（サンプルレートはブラウザのデフォルトを使用）
            this._audioContext = new AudioContext();

            // 3. AudioWorklet にプロセッサを登録
            //    audio-processor-worklet.js は別ファイルとして配信される
            await this._audioContext.audioWorklet.addModule('/js/modules/audio-processor-worklet.js');

            // 4. マイク → AudioWorkletNode を接続
            this._sourceNode = this._audioContext.createMediaStreamSource(this._mediaStream);
            this._workletNode = new AudioWorkletNode(this._audioContext, 'audio-capture-processor');

            // 5. Worklet からの PCM データを受信
            this._workletNode.port.onmessage = (event) => {
                if (event.data.type === 'pcmData') {
                    this._onAudioChunk(event.data.data);
                }
            };

            // マイク → Worklet を接続（出力先は不要なので destination には繋がない）
            this._sourceNode.connect(this._workletNode);
            // Worklet の出力を destination に繋がないと、一部ブラウザで
            // process() が呼ばれない問題があるため、念のため接続する。
            // ただし音量0なのでスピーカーからは聞こえない。
            this._workletNode.connect(this._audioContext.destination);

            // 状態遷移: OFF → LISTENING
            this._setState(MicState.LISTENING);

            // リアルタイムSTT用: 100ms 分のサンプル数を確定
            this._rtTargetSamples = Math.round(this._audioContext.sampleRate * 0.1);

            console.log('[Microphone] 起動完了 (sampleRate=' + this._audioContext.sampleRate + ')');

        } catch (err) {
            console.error('[Microphone] 起動エラー:', err);
            // リソースを解放してOFFに戻す
            this._releaseResources();
            throw err;
        }
    }

    /**
     * マイクを停止し、全リソースを解放して OFF に戻る。
     * どの状態からでも呼び出し可能。
     */
    stop() {
        if (this._state === MicState.OFF) {
            return;
        }

        console.log('[Microphone] 停止中...');

        // AudioWorklet に停止コマンドを送信
        if (this._workletNode) {
            this._workletNode.port.postMessage({ command: 'stop' });
        }

        // 全リソースを解放
        this._releaseResources();

        // 録音バッファをクリア
        this._resetTransientBuffers();
        this._rtTargetSamples = 0;

        // 状態遷移: 任意 → OFF
        this._setState(MicState.OFF);
    }

    /**
     * 録音・VAD の一時バッファをクリアする（stop / mute 共用）。
     * _rtTargetSamples はミュート解除後も使うため、ここでは触らない。
     * @private
     */
    _resetTransientBuffers() {
        this._recordingChunks = [];
        this._rtAccumBuffer = [];
        this._rtAccumSamples = 0;
        this._silenceStart = null;
        this._recordingStartTime = null;
        this._speechOnsetStart = null;
        this._onsetChunks = [];
        this._onsetSpeechCount = 0;
        this._onsetTotalCount = 0;
        this._rtPreRollBuffer = [];
    }

    /**
     * 一時ミュートする（ミク音声の再生中に呼ぶ）。
     *
     * stop() と違い getUserMedia のストリームと AudioWorklet は保持したまま、
     * 音声チャンクの処理だけを止める。unmute() での再開はほぼ即時
     * （stop()→start() はマイク再取得と STT セッション再開で約1秒かかり、
     * その間の発話を取りこぼしていた）。
     * 録音途中の断片・プレロールは破棄する（ミク音声の回り込みを確定させないため）。
     */
    mute() {
        if (this._state === MicState.OFF || this._state === MicState.MUTED) {
            return;
        }
        this._resetTransientBuffers();
        this._setState(MicState.MUTED);
    }

    /**
     * ミュートを解除して LISTENING に戻る。MUTED 以外では何もしない。
     */
    unmute() {
        if (this._state !== MicState.MUTED) {
            return;
        }
        this._setState(MicState.LISTENING);
    }

    /**
     * VAD閾値を動的に変更する。
     * レベルメーターUIのスライダーから呼ばれる想定。
     *
     * @param {number} threshold - 新しいRMS閾値（0.0〜1.0）
     */
    setVadThreshold(threshold) {
        this._vadThreshold = Math.max(0, Math.min(1, threshold));
        console.log('[Microphone] VAD閾値変更: ' + this._vadThreshold.toFixed(4));
    }

    /** 現在のVAD閾値を取得 */
    getVadThreshold() {
        return this._vadThreshold;
    }

    /**
     * STT処理完了を通知する。
     *
     * - PROCESSING 中 → LISTENING に遷移（通常フロー）
     * - RECORDING 中 → 何もしない（PROCESSING中に追加発話が始まっているので、
     *   その録音完了後に再び PROCESSING に遷移する。前のSTT結果は受け取り済み）
     *
     * 外部（socket.js など）から STT 結果受信時に呼ばれる想定。
     */
    notifySTTComplete() {
        if (this._state === MicState.RECORDING) {
            // PROCESSING中に追加発話が検知されてRECORDINGに遷移済み。
            // 前のSTTは完了したが、現在の録音はそのまま続行する。
            console.log('[Microphone] notifySTTComplete: 追加発話のRECORDING中のため遷移しない');
            return;
        }

        if (this._state !== MicState.PROCESSING) {
            console.warn('[Microphone] notifySTTComplete: 現在の状態が PROCESSING ではありません (state=' + this._state + ')');
            return;
        }

        this._setState(MicState.LISTENING);
    }

    // ========================================================================
    // 内部メソッド
    // ========================================================================

    /**
     * 状態を変更し、カスタムイベントを発火する。
     *
     * @param {string} newState - 遷移先の状態（MicState の値）
     * @private
     */
    _setState(newState) {
        const prevState = this._state;
        if (prevState === newState) return;

        this._state = newState;
        console.log(`[Microphone] ${prevState} → ${newState}`);

        // mic:stateChange イベントを発火
        // 他のモジュール（UIボタン更新など）がこのイベントをリッスンする
        document.dispatchEvent(new CustomEvent(EVENTS.MIC_STATE_CHANGE, {
            detail: { state: newState, prevState },
            bubbles: true,
        }));
    }

    /**
     * AudioWorklet から受け取った PCM チャンクを処理する。
     * 状態に応じて VAD 判定や録音バッファへの蓄積を行う。
     *
     * @param {Float32Array} pcmData - Float32 PCM データ（-1.0〜1.0）
     * @private
     */
    _onAudioChunk(pcmData) {
        // ミュート中はマイク入力を一切処理しない（ミク音声の回り込み対策。
        // ストリーム自体は生きているのでチャンクは届き続けるが、ここで捨てる）
        if (this._state === MicState.MUTED) {
            return;
        }

        // --- リアルタイムSTT用チャンク送信 ---
        if (this._rtTargetSamples > 0) {
            if (this._state === MicState.RECORDING) {
                // RECORDING中: 100ms分蓄積してmic:audioChunkを発火
                this._rtAccumBuffer.push(new Float32Array(pcmData));
                this._rtAccumSamples += pcmData.length;
                if (this._rtAccumSamples >= this._rtTargetSamples) {
                    const merged = this._mergeChunks(this._rtAccumBuffer);
                    document.dispatchEvent(new CustomEvent(EVENTS.MIC_AUDIO_CHUNK, {
                        detail: { pcmData: merged, sampleRate: this._audioContext.sampleRate },
                    }));
                    this._rtAccumBuffer = [];
                    this._rtAccumSamples = 0;
                }
            } else {
                // LISTENING/PROCESSING中: onset判定に備えてプレロールバッファに仮蓄積
                // onset確定時にまとめてflushする（発話冒頭の欠落防止）
                // 上限: speechOnsetDuration × 2 分のサンプル数（メモリ節約のため古いものを捨てる）
                this._rtPreRollBuffer.push(new Float32Array(pcmData));
                const maxPreRollSamples = this._audioContext.sampleRate * (this._speechOnsetDuration / 1000) * 2;
                let total = this._rtPreRollBuffer.reduce((s, c) => s + c.length, 0);
                while (total > maxPreRollSamples && this._rtPreRollBuffer.length > 0) {
                    total -= this._rtPreRollBuffer.shift().length;
                }
            }
        }

        // RMS（二乗平均平方根）を計算してエネルギーレベルを算出
        const rms = this._calculateRMS(pcmData);
        const isSpeech = rms > this._vadThreshold;

        // mic:level イベントを発火（レベルメーター表示用）
        // AudioWorkletは約2.67msごとに呼ばれるが、UIの更新頻度を抑えるため
        // 約60fps（≒16ms間隔）に間引く
        const now = performance.now();
        if (now - this._lastLevelTime > 16) {
            this._lastLevelTime = now;
            document.dispatchEvent(new CustomEvent(EVENTS.MIC_LEVEL, {
                detail: { rms, threshold: this._vadThreshold, isSpeech },
            }));
        }

        switch (this._state) {
            case MicState.LISTENING:
                this._handleListening(pcmData, isSpeech);
                break;

            case MicState.RECORDING:
                this._handleRecording(pcmData, isSpeech);
                break;

            case MicState.PROCESSING:
                // STT結果待ち中でもVAD監視を継続する。
                // 追加発話が検知されたら RECORDING に遷移し、新たな録音を開始する。
                // これにより「STT処理中に追加で喋った内容」も文字起こしされる。
                this._handleListening(pcmData, isSpeech);
                break;
        }
    }

    /**
     * LISTENING 状態での VAD 処理（duty cycle onset方式）。
     *
     * 判定ウィンドウ（speechOnsetDuration ms）内で閾値を超えたチャンクの割合が
     * speechOnsetRatio 以上なら発話開始と認定する。
     * - 「えっと、あの」など途中に無音が入る発話も検知できる
     * - キーボード打鍵（持続時間が短く全体の8割に届かない）は除外できる
     *
     * @param {Float32Array} pcmData
     * @param {boolean} isSpeech - RMSが閾値を超えているか
     * @private
     */
    _handleListening(pcmData, isSpeech) {
        const now = Date.now();

        if (this._speechOnsetStart === null) {
            // onset判定ウィンドウ外
            if (!isSpeech) return;
            // 初めて閾値を超えた → ウィンドウ開始
            this._speechOnsetStart = now;
            this._onsetChunks = [new Float32Array(pcmData)];
            this._onsetSpeechCount = 1;
            this._onsetTotalCount = 1;
            return;
        }

        // onset判定ウィンドウ中: チャンクを蓄積・カウント
        this._onsetChunks.push(new Float32Array(pcmData));
        this._onsetTotalCount++;
        if (isSpeech) this._onsetSpeechCount++;

        const elapsed = now - this._speechOnsetStart;

        if (elapsed < this._speechOnsetDuration) {
            // ウィンドウ未満 → 蓄積継続
            return;
        }

        // ウィンドウ終了 → duty cycle判定
        const ratio = this._onsetSpeechCount / this._onsetTotalCount;

        if (ratio < this._speechOnsetRatio) {
            // 判定失敗（打鍵音など）→ リセットして再待機
            // _rtPreRollBuffer は消去しない。発話がonset失敗直後に始まった場合に
            // 冒頭音声が失われるのを防ぐ。スライディングウィンドウなので自然に古い音が捨てられる。
            this._speechOnsetStart = null;
            this._onsetChunks = [];
            this._onsetSpeechCount = 0;
            this._onsetTotalCount = 0;
            return;
        }

        // 判定成功 → RECORDING遷移
        this._recordingChunks = this._onsetChunks;
        this._onsetChunks = [];
        this._speechOnsetStart = null;
        this._onsetSpeechCount = 0;
        this._onsetTotalCount = 0;
        this._silenceStart = null;
        this._recordingStartTime = Date.now();

        this._setState(MicState.RECORDING);

        // プレロールバッファをflush（onset前の発話冒頭を補完）
        if (this._rtTargetSamples > 0 && this._rtPreRollBuffer.length > 0) {
            const preRoll = this._mergeChunks(this._rtPreRollBuffer);
            document.dispatchEvent(new CustomEvent(EVENTS.MIC_AUDIO_CHUNK, {
                detail: { pcmData: preRoll, sampleRate: this._audioContext.sampleRate },
            }));
            this._rtPreRollBuffer = [];
        }

        document.dispatchEvent(new CustomEvent(EVENTS.MIC_SPEECH_START, { bubbles: true }));
    }

    /**
     * RECORDING 状態での VAD 処理。
     * PCM データを蓄積しつつ、無音検知やタイムアウトを判定する。
     *
     * @param {Float32Array} pcmData
     * @param {boolean} isSpeech - 閾値を超えているか
     * @private
     */
    _handleRecording(pcmData, isSpeech) {
        // PCM データをバッファに蓄積
        this._recordingChunks.push(new Float32Array(pcmData));

        const now = Date.now();

        // --- タイムアウト判定 ---
        // 最大録音時間を超えたら強制的に発話終了
        if (now - this._recordingStartTime > this._maxRecordingDuration) {
            console.log('[Microphone] 最大録音時間到達 (' + (this._maxRecordingDuration / 1000) + '秒)');
            this._finishRecording();
            return;
        }

        // --- 無音判定 ---
        if (isSpeech) {
            // 音声あり → 無音カウンタをリセット
            this._silenceStart = null;
        } else {
            // 無音検知
            if (this._silenceStart === null) {
                // 無音の開始時刻を記録
                this._silenceStart = now;
            } else if (now - this._silenceStart > this._silenceHangover) {
                // 無音が猶予時間を超えた → 発話終了
                console.log('[Microphone] 無音検知 → 発話終了');
                this._finishRecording();
            }
        }
    }

    /**
     * 録音を終了し、蓄積した PCM データをイベントで通知する。
     * RECORDING → PROCESSING に遷移する。
     *
     * @private
     */
    _finishRecording() {
        // 蓄積した PCM チャンクを1つの Float32Array に結合
        const audioData = this._mergeChunks(this._recordingChunks);

        // サンプルレートを取得（AudioContext から）
        const sampleRate = this._audioContext ? this._audioContext.sampleRate : 48000;

        const durationSec = audioData.length / sampleRate;
        console.log('[Microphone] 録音完了: ' + audioData.length + ' samples (' + durationSec.toFixed(1) + '秒)');

        // バッファをクリア（メモリ解放）
        this._recordingChunks = [];
        this._silenceStart = null;
        this._recordingStartTime = null;

        // 状態遷移: RECORDING → PROCESSING
        this._setState(MicState.PROCESSING);

        // mic:speechEnd イベントを発火
        // 後で Gateway 連携時に、このイベントのリスナーが audio_for_stt を送信する
        document.dispatchEvent(new CustomEvent(EVENTS.MIC_SPEECH_END, {
            detail: { audioData, sampleRate },
            bubbles: true,
        }));
    }

    /**
     * RMS（Root Mean Square / 二乗平均平方根）を計算する。
     * 音声のエネルギーレベルを表す指標。0 = 完全無音、1.0 = 最大音量。
     *
     * @param {Float32Array} pcmData - PCM データ
     * @returns {number} RMS 値（0.0〜1.0）
     * @private
     */
    _calculateRMS(pcmData) {
        let sumSquares = 0;
        for (let i = 0; i < pcmData.length; i++) {
            sumSquares += pcmData[i] * pcmData[i];
        }
        return Math.sqrt(sumSquares / pcmData.length);
    }

    /**
     * 複数の Float32Array チャンクを1つに結合する。
     *
     * @param {Float32Array[]} chunks - PCM チャンクの配列
     * @returns {Float32Array} 結合された PCM データ
     * @private
     */
    _mergeChunks(chunks) {
        // 全チャンクの合計サンプル数を計算
        let totalLength = 0;
        for (const chunk of chunks) {
            totalLength += chunk.length;
        }

        // 1つの配列に結合
        const merged = new Float32Array(totalLength);
        let offset = 0;
        for (const chunk of chunks) {
            merged.set(chunk, offset);
            offset += chunk.length;
        }

        return merged;
    }

    /**
     * Web Audio API / getUserMedia のリソースを解放する。
     * メモリリーク防止のため、stop() から確実に呼ばれる。
     *
     * @private
     */
    _releaseResources() {
        // AudioWorkletNode を切断
        if (this._workletNode) {
            this._workletNode.disconnect();
            this._workletNode = null;
        }

        // MediaStreamSource を切断
        if (this._sourceNode) {
            this._sourceNode.disconnect();
            this._sourceNode = null;
        }

        // AudioContext を閉じる
        if (this._audioContext) {
            this._audioContext.close().catch(() => {});
            this._audioContext = null;
        }

        // マイクのストリームを停止（ブラウザのマイクインジケータが消える）
        if (this._mediaStream) {
            this._mediaStream.getTracks().forEach(track => track.stop());
            this._mediaStream = null;
        }
    }
}

// 状態定数もエクスポート（外部から状態比較に使える）
export { MicState };