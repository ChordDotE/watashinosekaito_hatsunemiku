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
 * videoCompressor.js - 送信前の動画圧縮（WebCodecs + mediabunny）
 *
 * 目的:
 * - アップロード通信量の削減と、LLM への inline 添付上限（合計約20MB）への収まりを両立する
 * - Gemini は動画を毎秒1フレームでサンプリングするため、低fps・低解像度に落としても
 *   LLM から見た品質はほぼ変わらない
 *
 * 方式:
 * - vendor/mediabunny（公式ビルド済みESM・v1.50.8・npm公式とハッシュ一致確認済み）の
 *   Conversion API で demux→decode→縮小/fps間引き→encode→mux を一括実行
 * - 出力は WebM（VP8/VP9 + Opus。ブラウザでエンコード可能かつ Gemini 対応形式）
 * - mediabunny は動画添付時のみ動的 import する（通常ページ読込には影響しない）
 *
 * フォールバック:
 * - WebCodecs 非対応ブラウザ / 入力コーデックをデコードできない端末（例: HEVC 非対応）/
 *   変換失敗 / 圧縮効果なし の場合は null を返し、呼び出し側は原本をそのまま送る
 */

// この閾値未満のファイルは inline 上限に収まるため圧縮しない（変換時間の節約）
const COMPRESS_MIN_BYTES = 5 * 1024 * 1024;
// 出力パラメータ（Gemini の 1fps サンプリングに合わせた設定）
const TARGET_WIDTH = 640;       // 長辺ではなく出力幅。縦動画は高さが自動調整される
const TARGET_FPS = 2;           // Gemini のサンプリング(1fps)に対して余裕をみて2fps
const VIDEO_BITRATE = 400_000;  // bps
const AUDIO_BITRATE = 32_000;   // bps（音声トラックも LLM が聞くため残す）

/**
 * 動画ファイルを圧縮して WebM の File を返す。
 * 圧縮しない/できない場合は null を返す（呼び出し側は原本を使う）。
 * @param {File} file - 入力動画ファイル
 * @returns {Promise<File|null>}
 */
export async function compressVideoFile(file) {
    if (!file || file.size < COMPRESS_MIN_BYTES) return null;
    if (typeof VideoEncoder === 'undefined' || typeof VideoDecoder === 'undefined') {
        console.warn('[VideoCompressor] WebCodecs 非対応ブラウザのため原本を送信します');
        window.__mikuDebugToast?.('🎬 WebCodecs非対応→原本送信');
        return null;
    }
    window.__mikuDebugToast?.(`🎬 変換開始 ${(file.size / 1024 / 1024).toFixed(1)}MB`);

    const t0 = performance.now();
    try {
        const mb = await import('../vendor/mediabunny/mediabunny.mjs');

        const input = new mb.Input({
            source: new mb.BlobSource(file),
            formats: mb.ALL_FORMATS,
        });
        const output = new mb.Output({
            format: new mb.WebMOutputFormat(),
            target: new mb.BufferTarget(),
        });

        const conversion = await mb.Conversion.init({
            input,
            output,
            video: {
                width: TARGET_WIDTH,
                frameRate: TARGET_FPS,
                bitrate: VIDEO_BITRATE,
            },
            audio: {
                numberOfChannels: 1,
                bitrate: AUDIO_BITRATE,
            },
        });

        if (!conversion.isValid) {
            console.warn('[VideoCompressor] この動画は変換できません（原本を送信）:',
                conversion.discardedTracks);
            window.__mikuDebugToast?.('🎬 変換不可→原本送信');
            return null;
        }

        await conversion.execute();

        const buffer = output.target.buffer;
        if (!buffer || buffer.byteLength === 0 || buffer.byteLength >= file.size) {
            console.warn('[VideoCompressor] 圧縮効果がないため原本を送信します');
            return null;
        }

        const name = file.name.replace(/\.[^.]+$/, '') + '.webm';
        const secs = ((performance.now() - t0) / 1000).toFixed(1);
        console.log(`[VideoCompressor] ${file.name}: ` +
            `${(file.size / 1024 / 1024).toFixed(1)}MB → ${(buffer.byteLength / 1024 / 1024).toFixed(1)}MB ` +
            `(${TARGET_WIDTH}px/${TARGET_FPS}fps, ${secs}秒)`);
        window.__mikuDebugToast?.(`🎬 変換完了 →${(buffer.byteLength / 1024 / 1024).toFixed(1)}MB (${secs}s)`);
        return new File([buffer], name, { type: 'video/webm' });

    } catch (e) {
        console.warn('[VideoCompressor] 変換失敗のため原本を送信します:', e);
        window.__mikuDebugToast?.(`🎬 変換失敗→原本送信: ${e?.message ?? e}`);
        return null;
    }
}
