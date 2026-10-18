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
 * AudioWorkletProcessor - マイク入力のPCMデータ抽出
 *
 * AudioWorkletは音声レンダリングスレッド（メインスレッドとは別）で動作する。
 * マイクから入力された生のPCMデータ（Float32）を、port.postMessage() で
 * メインスレッドの MicrophoneController に転送する。
 *
 * ブラウザの AudioWorklet 仕様:
 *   - process() は約128サンプル（≒2.67ms @48kHz）ごとに呼ばれる
 *   - メインスレッドとは SharedArrayBuffer か postMessage で通信する
 *   - ここでは postMessage を使用（シンプル＆十分な性能）
 *
 * 使用方法（メインスレッド側）:
 *   await audioContext.audioWorklet.addModule('/js/modules/audio-processor-worklet.js');
 *   const workletNode = new AudioWorkletNode(audioContext, 'audio-capture-processor');
 *   workletNode.port.onmessage = (e) => { // e.data.data = Float32Array };
 *
 * 停止方法（メインスレッド側）:
 *   workletNode.port.postMessage({ command: 'stop' });
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();

        // キャプチャ中フラグ。false にすると process() が false を返し、
        // ブラウザがこのプロセッサをガベージコレクションする。
        this.isCapturing = true;

        // メインスレッドからの制御コマンドを受信
        this.port.onmessage = (event) => {
            if (event.data.command === 'stop') {
                this.isCapturing = false;
            }
        };
    }

    /**
     * 音声レンダリングコールバック。
     * ブラウザの音声処理ループから約128サンプルごとに呼ばれる。
     *
     * @param {Float32Array[][]} inputs - 入力チャンネルの配列
     *   inputs[0] = 最初の入力ソース（マイク）
     *   inputs[0][0] = チャンネル0（モノラル）の Float32Array（-1.0〜1.0）
     * @param {Float32Array[][]} outputs - 未使用（マイク入力なので出力しない）
     * @param {Object} parameters - AudioParam の値（未使用）
     * @returns {boolean} true=処理継続、false=プロセッサ破棄
     */
    process(inputs, outputs, parameters) {
        // 停止指示が来ていたらプロセッサを破棄
        if (!this.isCapturing) {
            return false;
        }

        const input = inputs[0]; // 最初の入力ソース
        if (input && input.length > 0) {
            // チャンネル0のPCMデータをコピーしてメインスレッドに送信
            // ※ AudioWorklet内のバッファは再利用されるため、必ずコピーする
            this.port.postMessage({
                type: 'pcmData',
                data: new Float32Array(input[0])
            });
        }

        return true; // 処理を継続
    }
}

// プロセッサを名前付きで登録。メインスレッドから 'audio-capture-processor' で参照する。
registerProcessor('audio-capture-processor', AudioCaptureProcessor);
