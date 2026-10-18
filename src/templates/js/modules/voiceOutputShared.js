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
 * voiceOutputShared.js - 音声出力の共有ロジック
 *
 * もとは旧TTS（VOICEVOX/RVC のファイル再生・voiceQueue.js）と voicePcmPlayer.js（Fish PCM）に
 * 二重実装されていたロジックを集約したもの。旧TTS は 2026-09-25 に削除し、いまの利用者は
 * voicePcmPlayer.js だけ:
 *   - Live2D iframe への postMessage 送信（iframe 参照のキャッシュ込み）
 *   - リップシンク値の計算・送信ループ（AnalyserNode → 60fps）
 *   - voice:start / voice:end の発火
 *
 * 再生方式そのもの（BufferSource スケジューリング / PCM リングバッファ・ペーシング）は
 * プレイヤー固有のため、ここには置かない。
 */

import { EVENTS } from './events.js';

// Live2D iframe の参照キャッシュ（動的に作られるため、無効になったら再取得する）
let _live2dIframe = null;

/**
 * Live2D iframe に postMessage を送信する
 * @param {Object} data - 送信データ（{type:'lipSync', value} / {type:'motion', ...} 等）
 */
export function sendToLive2D(data) {
    if (!_live2dIframe || !_live2dIframe.contentWindow) {
        _live2dIframe =
            document.querySelector('miku-avatar-live2d')?.shadowRoot?.querySelector('iframe')
            ?? document.getElementById('live2d-iframe');
    }
    try {
        _live2dIframe?.contentWindow?.postMessage(data, '*');
    } catch (_) {
        // iframe 未ロード時などは無視
    }
}

/**
 * リップシンク送信ループ。
 * AnalyserNode の低周波帯平均から口の開き（0〜1）を算出し、約60fpsで Live2D へ送る。
 */
export class LipSyncLoop {
    constructor() {
        /** @type {number|null} requestAnimationFrame の ID */
        this._id = null;
    }

    /**
     * ループ開始（既に動いていれば止めてから開始 = 二重ループを防ぐ）
     * @param {AnalyserNode} analyser
     */
    start(analyser) {
        this.stop();
        const data = new Uint8Array(analyser.frequencyBinCount);
        const tick = () => {
            analyser.getByteFrequencyData(data);
            let sum = 0;
            const len = Math.min(data.length, 32);
            for (let i = 0; i < len; i++) sum += data[i];
            sendToLive2D({ type: 'lipSync', value: Math.min(1, (sum / len) / 128) });
            this._id = requestAnimationFrame(tick);
        };
        tick();
    }

    /** ループ停止（口を閉じる lipSync=0 を送ってから止める） */
    stop() {
        if (this._id !== null) {
            cancelAnimationFrame(this._id);
            this._id = null;
        }
        sendToLive2D({ type: 'lipSync', value: 0 });
    }
}

/**
 * voice:start を発火する（音声再生開始。ターン最初の1回だけ呼ぶ規律は呼び出し側が守る）
 * @param {Object} [detail] - 任意の付加情報（旧TTSは {fileName} を載せる）
 */
export function dispatchVoiceStart(detail = {}) {
    document.dispatchEvent(new CustomEvent(EVENTS.VOICE_START, {
        detail,
        bubbles: true,
        composed: true,
    }));
    console.log('🎵 voice:start イベント発火');
}

/**
 * voice:end を発火する（全再生完了 / 強制停止によるターン終了 / データ待ちギャップ）
 * @param {Object} [detail] - `{ moreExpected: true }` を渡すと、まだ同じ発話の続きが
 *   来る見込みのある一時的な間（ツール待ち等のデータ待ちギャップ）であることを示す
 *   （turnState.js がターン終了と誤認しないための情報。voicePcmPlayer.js 参照）
 */
export function dispatchVoiceEnd(detail = {}) {
    document.dispatchEvent(new CustomEvent(EVENTS.VOICE_END, {
        detail,
        bubbles: true,
        composed: true,
    }));
    console.log('🔇 voice:end イベント発火');
}
