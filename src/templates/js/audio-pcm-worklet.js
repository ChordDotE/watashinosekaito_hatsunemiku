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
 * audio-pcm-worklet.js - PCM リングバッファ AudioWorklet プロセッサ
 *
 * AudioWorkletProcessor として登録される。
 * voicePcmPlayer.js が AudioWorkletNode を通じてメッセージを送受信する。
 *
 * 受信メッセージ:
 *   { type: 'chunk',  buffer: ArrayBuffer }  Int16 PCM を追記
 *   { type: 'finish' }                        データ終端（バッファ消費後に停止）
 *   { type: 'reset'  }                        バッファをクリアして再初期化
 *
 * 送信メッセージ:
 *   { type: 'ended' }                       バッファが空 かつ finish 済みになったとき（再生完了）
 *   { type: 'buffer_low', available }        バッファ残量が LOW_WATERMARK を下回ったとき（閾値を超えるまで再送しない）
 *   { type: 'progress', played }             再生位置（readPos）の定期報告。プレイヤーがバッファ残量を把握する
 *   { type: 'overflow', dropped, ... }       リングバッファ溢れで書き込めず捨てたサンプルがあるとき（異常検知）
 */

const RING_BUFFER_SECONDS = 10;  // 10秒分のリングバッファ
const SAMPLE_RATE = 44100;
const RING_SIZE = SAMPLE_RATE * RING_BUFFER_SECONDS;
const LOW_WATERMARK = SAMPLE_RATE * 2;       // 残り2秒を下回ったら buffer_low 通知
const PROGRESS_INTERVAL = SAMPLE_RATE / 4;   // 0.25秒ごとに再生位置を報告

class PcmRingBufferProcessor extends AudioWorkletProcessor {
    constructor() {
        super();

        this._buffer = new Float32Array(RING_SIZE);
        this._writePos = 0;  // 書き込み位置（絶対値）
        this._readPos  = 0;  // 読み取り位置（絶対値）
        this._finished = false;
        this._endedNotified = false;
        this._lowNotified = false;
        this._lastProgressPos = 0;

        this.port.onmessage = (e) => {
            const { type } = e.data;
            if (type === 'chunk') {
                this._writeChunk(e.data.buffer);
            } else if (type === 'finish') {
                this._finished = true;
            } else if (type === 'reset') {
                this._writePos = 0;
                this._readPos  = 0;
                this._finished = false;
                this._endedNotified = false;
                this._lowNotified = false;
                this._lastProgressPos = 0;
                this._buffer.fill(0);
            }
        };
    }

    /**
     * Int16 ArrayBuffer を Float32 に変換してリングバッファへ書き込む
     */
    _writeChunk(arrayBuffer) {
        const int16 = new Int16Array(arrayBuffer);
        const available = this._writePos - this._readPos;
        const space = RING_SIZE - available;
        const toWrite = Math.min(int16.length, space);

        if (toWrite < int16.length) {
            // リングバッファに収まらない → 未再生データの上書きを防ぐため超過分は捨てる
            this.port.postMessage({
                type: 'overflow',
                dropped: int16.length - toWrite,
                chunkSamples: int16.length,
                available,
            });
        }

        for (let i = 0; i < toWrite; i++) {
            this._buffer[this._writePos % RING_SIZE] = int16[i] / 32768.0;
            this._writePos++;
        }
    }

    /**
     * AudioWorklet コールバック（128サンプルごと）
     */
    process(inputs, outputs) {
        const output = outputs[0][0];
        const available = this._writePos - this._readPos;

        // バッファ残量が低下したら通知（finish済みは対象外、再浮上するまで再送しない）
        if (available < LOW_WATERMARK && !this._finished && !this._lowNotified) {
            this._lowNotified = true;
            this.port.postMessage({ type: 'buffer_low', available });
        } else if (available >= LOW_WATERMARK) {
            this._lowNotified = false;
        }

        if (available <= 0) {
            // バッファ空
            output.fill(0);
            if (this._finished && !this._endedNotified) {
                this._endedNotified = true;
                this.port.postMessage({ type: 'ended' });
                return false;  // プロセッサ停止
            }
            return true;
        }

        const toRead = Math.min(output.length, available);
        for (let i = 0; i < toRead; i++) {
            output[i] = this._buffer[this._readPos % RING_SIZE];
            this._readPos++;
        }
        // 読み切れなかった分は 0 埋め（通常は output.length === 128 で toRead と一致）
        for (let i = toRead; i < output.length; i++) {
            output[i] = 0;
        }

        // 再生位置を定期報告（プレイヤーがバッファ残量を把握して供給をペーシングする）
        if (this._readPos - this._lastProgressPos >= PROGRESS_INTERVAL) {
            this._lastProgressPos = this._readPos;
            this.port.postMessage({ type: 'progress', played: this._readPos });
        }

        return true;
    }
}

registerProcessor('pcm-ring-buffer', PcmRingBufferProcessor);
