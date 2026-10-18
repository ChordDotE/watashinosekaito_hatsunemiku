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
 * audioConverter.js - 送信前の音声変換（Web Audio API）
 *
 * 目的:
 * - 声の登録（voice_enroll）に使う音声を、サーバー側が確実に読める形へ揃える
 * - サーバーの照合は 16kHz mono を前提にしているので、**ブラウザ側でその形にして送る**
 *
 * なぜブラウザで変換するか:
 * - サーバー側の soundfile（libsndfile）は m4a / aac を読めない。iPhone のボイスメモは m4a なので
 *   「ミクは聴こえているのに覚えられない」状態になる（LLM へは原本が inline で渡るため）
 * - ffmpeg を足せば解決するが、作曲機能のセットアップでしか入らない外部バイナリ（約100MB）に
 *   声の登録が依存してしまう
 * - decodeAudioData はブラウザのコーデックを使うので、m4a / aac / webm / mp3 / wav を
 *   **追加依存ゼロ**で扱える。16kHz mono に落とすぶん送信量も減る（44.1kHz ステレオの約1/6）
 *
 * フォールバック:
 * - デコード失敗（対応していないコーデック等）は null を返し、呼び出し側は原本をそのまま送る
 *   （LLM は原本を聴けるので、会話としては成立する）
 */

const TARGET_SR = 16000;   // 照合モデル（ECAPA-TDNN）の学習条件に合わせる
// この長さを超える音声は先頭だけ使う。登録に必要なのは十数秒で、長いほど変換とアップロードが重くなる
const MAX_SECONDS = 60;

/**
 * 音声ファイルを 16kHz mono の WAV File に変換して返す。
 * 変換できない場合は null（呼び出し側は原本を使う）。
 * @param {File} file - 入力音声ファイル
 * @returns {Promise<File|null>}
 */
export async function convertAudioForEnroll(file) {
    if (!file) return null;
    const AudioCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    if (!AudioCtx || typeof AudioContext === 'undefined') {
        console.warn('[AudioConverter] Web Audio 非対応のため原本を送信します');
        return null;
    }

    let decoded;
    try {
        const buf = await file.arrayBuffer();
        // デコード専用の AudioContext（sampleRate 指定は端末により無視されるため後段でリサンプルする）
        const ctx = new AudioContext();
        try {
            decoded = await ctx.decodeAudioData(buf);
        } finally {
            ctx.close?.();
        }
    } catch (e) {
        console.warn('[AudioConverter] デコードできませんでした（原本を送信）:', e);
        window.__mikuDebugToast?.(`🎤 音声デコード失敗: ${e?.message ?? e}`);
        return null;
    }

    try {
        const seconds = Math.min(decoded.duration, MAX_SECONDS);
        const frames = Math.floor(seconds * TARGET_SR);
        if (frames <= 0) return null;

        // モノラル化 + 16kHz へのリサンプルを OfflineAudioContext に任せる
        const offline = new AudioCtx(1, frames, TARGET_SR);
        const src = offline.createBufferSource();
        src.buffer = decoded;
        src.connect(offline.destination);
        src.start(0);
        const rendered = await offline.startRendering();
        const wav = encodeWav(rendered.getChannelData(0), TARGET_SR);

        const name = file.name.replace(/\.[^.]+$/, '') + '.wav';
        return new File([wav], name, { type: 'audio/wav' });
    } catch (e) {
        console.warn('[AudioConverter] 変換に失敗しました（原本を送信）:', e);
        return null;
    }
}

/**
 * Float32 の波形を 16bit PCM の WAV バイト列にする。
 * @param {Float32Array} samples
 * @param {number} sampleRate
 * @returns {ArrayBuffer}
 */
function encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const writeStr = (offset, s) => {
        for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
    };

    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);          // fmt チャンクのサイズ
    view.setUint16(20, 1, true);           // PCM
    view.setUint16(22, 1, true);           // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);  // バイト/秒（mono・16bit）
    view.setUint16(32, 2, true);           // ブロックサイズ
    view.setUint16(34, 16, true);          // ビット深度
    writeStr(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return buffer;
}
