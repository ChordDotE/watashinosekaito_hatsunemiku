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
 * miku-caption.js - 返答テキスト字幕コンポーネント
 *
 * 機能:
 * - ミクの返答テキストをアバター領域下部に字幕バンドで表示する
 *   （イヤホン・スピーカーが使えない状況でも会話内容が分かるようにする）
 * - 表示ON/OFFはチャット入力欄の字幕ボタン（miku-chat.js の #captionButton）が制御し、
 *   `caption:setEnabled` イベントで通知される。永続化はせず、ページ読み込みごとに既定OFF
 * - 字幕モードON中は音声再生・リップシンクもスキップされる（担当は
 *   voicePcmPlayer.js。このコンポーネントは表示のみ）
 * - 字幕バンドのタップでその場の表示だけを閉じる（機能はONのまま）
 *
 * 購読するカスタムイベント:
 *   server:responseReady      → detail.response を表示（最終応答確定）
 *   server:inactivityReminder → detail.response を表示（無応答リマインダー）
 *   thinking:start            → 字幕をクリア（次の応答に備える）
 *   caption:setEnabled        → detail.enabled で表示ON/OFF切替（ONなら直近の返答を復元表示）
 *
 * 配置のしくみ:
 * - <miku-caption> は .container 内でアバターとチャットの間に置かれた
 *   「サイズゼロのアンカー要素」。デスクトップ（横並び）では幅0の縦帯、
 *   モバイル（縦並び）では高さ0の横帯になる。
 * - 字幕バンドはこのアンカーから position:absolute でアバター領域側に伸ばす。
 *   チャット欄の高さが変動しても常に「アバター領域の下端」に張り付く
 *   （レイアウト計測用のJSが不要）。
 *
 * テンプレートは fetch せずインラインで持つ:
 * 小規模であることに加え、初回応答がテンプレート取得より先に届いても
 * 取りこぼさないようにするため。
 */

import { EVENTS } from '../../modules/events.js';
import { appState } from '../../modules/appState.js';

class MikuCaption extends HTMLElement {
    constructor() {
        super();
        // 表示ON/OFFの判定は appState.captionEnabled（単一ソース）を参照する。
        // 字幕バンドの表示は即時反映なので生の値を使う（ターンラッチではない）
        this._lastText = '';
    }

    connectedCallback() {
        const shadow = this.attachShadow({ mode: 'open' });
        shadow.innerHTML = `
            <style>
                :host {
                    position: relative;
                    display: block;
                }
                /* 字幕バンド（アンカーからアバター領域側へ伸ばす） */
                #caption {
                    position: absolute;
                    bottom: 14px;
                    right: 16px;
                    width: 560px;
                    max-height: 32vh;
                    overflow-y: auto;
                    z-index: 20;
                    background: rgba(13, 15, 24, 0.72);
                    color: #fff;
                    border-left: 3px solid var(--main-color, #39c5bb);
                    border-radius: 12px;
                    padding: 10px 14px;
                    font-size: 15px;
                    line-height: 1.6;
                    white-space: pre-wrap;
                    word-break: break-word;
                    cursor: pointer;
                    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.25);
                }
                #caption[hidden] {
                    display: none;
                }
                /* モバイル（縦並びレイアウト）: アンカーが高さ0の横帯になる */
                @media (max-width: 768px) {
                    #caption {
                        bottom: 8px;
                        left: 8px;
                        right: 8px;
                        width: auto;
                        max-height: 26vh;
                    }
                }
            </style>
            <div id="caption" hidden></div>
        `;

        this._caption = shadow.getElementById('caption');
        this._caption.addEventListener('click', () => this._hide());

        this.setupEventListeners();
    }

    disconnectedCallback() {
        this.removeEventListeners();
    }

    setupEventListeners() {
        // 最終応答確定: 返答テキストを表示
        this.handleResponseReady = (e) => {
            const text = e.detail?.response;
            if (typeof text === 'string' && text.trim()) this._show(text);
        };
        document.addEventListener(EVENTS.SERVER_RESPONSE_READY, this.handleResponseReady);

        // 無応答リマインダー: 同じく返答テキストを表示
        this.handleReminder = (e) => {
            const text = e.detail?.response;
            if (typeof text === 'string' && text.trim()) this._show(text);
        };
        document.addEventListener(EVENTS.SERVER_INACTIVITY_REMINDER, this.handleReminder);

        // 新しい送信開始: 前の字幕をクリア
        this.handleThinkingStart = () => this._clear();
        document.addEventListener(EVENTS.THINKING_START, this.handleThinkingStart);

        // 字幕ボタン（miku-chat）からのON/OFF切替（状態本体は appState が持つ）
        this.handleSetEnabled = (e) => {
            if (!e.detail?.enabled) {
                this._caption.hidden = true;
            } else if (this._lastText) {
                this._show(this._lastText);  // 直近の返答があれば復元表示
            }
        };
        document.addEventListener(EVENTS.CAPTION_SET_ENABLED, this.handleSetEnabled);
    }

    removeEventListeners() {
        document.removeEventListener(EVENTS.SERVER_RESPONSE_READY, this.handleResponseReady);
        document.removeEventListener(EVENTS.SERVER_INACTIVITY_REMINDER, this.handleReminder);
        document.removeEventListener(EVENTS.THINKING_START, this.handleThinkingStart);
        document.removeEventListener(EVENTS.CAPTION_SET_ENABLED, this.handleSetEnabled);
    }

    _show(text) {
        this._lastText = text;
        if (!appState.getCaptionEnabled()) return;
        this._caption.textContent = text;
        this._caption.hidden = false;
        this._caption.scrollTop = 0;
    }

    /** バンドのタップ: 今の表示だけ閉じる（機能はONのまま） */
    _hide() {
        this._caption.hidden = true;
    }

    _clear() {
        this._lastText = '';
        this._caption.textContent = '';
        this._caption.hidden = true;
    }
}

customElements.define('miku-caption', MikuCaption);