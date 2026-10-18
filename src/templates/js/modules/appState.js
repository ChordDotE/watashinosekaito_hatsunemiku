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
 * appState.js - アプリケーション共有状態モジュール
 *
 * モジュール間で共有が必要な状態をここに集約する。
 * window.* グローバル変数の代替として使用。
 *
 * 現在の管理対象:
 * - cameraBuffer: カメラのフレームバッファ（base64 data URL 配列）
 *   書き込み: CameraManager (_syncCameraBufferToGlobal)
 *   読み込み: inactivityTimer (requestInactivityReminder)
 * - captionEnabled: 返答テキスト字幕モード（ON中は音声再生・リップシンクをスキップ）
 *   書き込み: miku-chat (initCaptionButton)
 *   読み込み: voicePcmPlayer / inactivityTimer
 */
export const appState = {
    /** @type {string[]} カメラフレームの base64 data URL 配列（最大 N 枚） */
    cameraBuffer: [],

    /** @type {boolean} 字幕モード（生の値・字幕ボタンの現在状態）。既定OFF・永続化なし */
    captionEnabled: false,

    /** @type {boolean} 現在のターンの字幕モード（ターン開始時に captionEnabled をラッチ）。
     *  ターン処理（音声破棄・ボタン復帰・タイマー起点）の判定は必ずラッチ値を使う。
     *  ターン途中のトグルで判定が揺れてデッドロックするのを防ぐ（詳細は decisions.md 2026-07-14） */
    turnCaptionEnabled: false,

    /**
     * 字幕モードを設定する（書き込みは miku-chat の字幕ボタンのみ）
     * @param {boolean} enabled
     */
    setCaptionEnabled(enabled) {
        this.captionEnabled = !!enabled;
    },

    /**
     * 字幕モードの生の値（字幕バンドの表示判定用）
     * @returns {boolean}
     */
    getCaptionEnabled() {
        return this.captionEnabled;
    },

    /**
     * ターン開始時（thinking:start / 無応答リマインダー受信時）に呼び、
     * このターンの字幕モードを確定させる
     */
    latchTurnCaption() {
        this.turnCaptionEnabled = this.captionEnabled;
    },

    /**
     * ターン処理の字幕判定。ラッチ値に加えて生の値も OR で見る:
     * ターン途中で字幕ONに切り替えた場合も以降の音声は破棄してよい
     * （その時点のターン閉じ処理は miku-chat / inactivityTimer の
     *   caption:setEnabled ハンドラが行う）
     * @returns {boolean}
     */
    isCaptionActive() {
        return this.turnCaptionEnabled || this.captionEnabled;
    },

    /**
     * カメラバッファを更新する
     * @param {string[]} frames - base64 data URL の配列
     */
    setCameraBuffer(frames) {
        this.cameraBuffer = frames;
    },

    /**
     * カメラバッファを取得する
     * @returns {string[]}
     */
    getCameraBuffer() {
        return this.cameraBuffer;
    },

    /**
     * カメラバッファをクリアする
     */
    clearCameraBuffer() {
        this.cameraBuffer = [];
    },
};
