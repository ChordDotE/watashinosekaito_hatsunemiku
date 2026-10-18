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
 * microphoneUI.js - マイクボタンの状態表示・マイクパネル（履歴＋レベルメーター）
 *
 * <miku-chat> の内部部品（Shadow DOM を引数に取る。単独では動かない）。
 * MicState と micMode に応じたボタン表示の更新と、マイクパネルの開閉・描画を担う。
 *
 * ## マイクパネル（2026-08-01・旧「調整モード」を置換）
 *
 * マイクボタン長押し(500ms)で開き、再度長押しで閉じる（カメラプレビューと同じ操作系）。
 * パネルは全マイクモード共通:
 *   - 上段: 直近の会話履歴（GET /api/conversation/recent。会話ログ新形式のみ・話者+時刻つき）
 *   - 下段: 音声レベルメーター＋閾値スライダー（マイクON中のみ表示。OFF時は履歴のみ）
 *
 * ## 音声のみレイアウト（AUTO モード）
 *
 * micMode が AUTO の間は .input-section に .voice-only を付けて入力欄・話すボタンを隠す
 * （即時送信では編集の機会がなく「編集できる見た目」が誤解を生むため。履歴はパネルで確認する）。
 */

import { UI_STRINGS } from '../../modules/uiStrings.js';
import { MicState } from '../../modules/microphone.js';
import { attachPressGesture } from '../../modules/pressGesture.js';

/** パネルに表示する直近発話の件数 */
const PANEL_HISTORY_COUNT = 5;

export class MicrophoneUI {
    /**
     * @param {ShadowRoot} shadow
     * @param {{ getMicMode: function, getMic: function, cancelAutoSend: function,
     *           isTurnOpen: function }} options
     */
    constructor(shadow, options = {}) {
        this._shadow        = shadow;
        this._getMicMode    = options.getMicMode    || (() => 'OFF');
        this._getMic        = options.getMic        || (() => null);
        this._cancelAutoSend = options.cancelAutoSend || (() => {});
        // ミクが考え中（turnBelief=OPEN）か。考え中はカウントダウンを止めない
        this._isTurnOpen    = options.isTurnOpen    || (() => false);
        /** @type {boolean} マイクパネル表示状態（マイクボタン長押しでトグル。既定OFF） */
        this._panelOpen     = false;
        /** @type {number|null} パネル表示中の履歴ポーリングタイマー */
        this._historyTimer  = null;
        /** @type {string|null} 前回描画した履歴の内容キー（変化検出用） */
        this._lastHistoryKey = null;
    }

    /** MicState に応じてボタンの見た目を更新する */
    updateButtonState(state) {
        const btn = this._shadow.getElementById('micButton');
        if (!btn) return;

        btn.dataset.micState = state;

        // 録音開始 → 話しているのでカウントダウンをキャンセル（**通常時のみ**）。
        // 考え中（畳み込み）は止めない — 止めると話し続ける限り一度も満了せず、
        // 追加発話が1文字も届かないまま返事が始まる（2026-08-11 の実バグ）
        if (state === MicState.RECORDING && !this._isTurnOpen()) this._cancelAutoSend();

        // ツールチップ
        const micMode = this._getMicMode();
        btn.title = UI_STRINGS.micButton[this._titleKey(state, micMode)] || '';

        // アイコンは fa-microphone で固定（他ボタンと同様、状態は色のみで表す）

        // マイク状態でレベルメーターの表示可否が変わる（OFF→履歴のみのパネルになる）
        this._applyLevelVisibility();

        console.log(`[MicButton] 状態更新: ${state}`);
    }

    /**
     * 状態・モードに対応するツールチップ文言のキーを返す。
     * MUTED（ミク音声再生中の一時ミュート）はモードによらず専用文言。
     * @private
     */
    _titleKey(state, micMode) {
        if (state === MicState.MUTED) return 'MUTED';
        if (micMode === 'AUTO') {
            return state === MicState.RECORDING ? 'RECORDING_AUTO' : 'LISTENING_AUTO';
        }
        return state;
    }

    /**
     * マイクパネルをトグルする（マイクボタン長押しから呼ばれる）。
     * どのマイクモードでも開ける。開くたびに履歴を取得し直す。
     */
    togglePanel() {
        this._panelOpen = !this._panelOpen;
        const panel = this._shadow.getElementById('micPanel');
        if (panel) panel.classList.toggle('open', this._panelOpen);
        this._applyLevelVisibility();
        if (this._panelOpen) {
            this._lastHistoryKey = null;  // 開くたびに必ず描画し直す
            this._refreshHistory();
            // 開いている間はリアルタイム更新: 会話が進む（ターン完了で KV に追記される）と
            // 自動で反映する。内容が変わらなければ再描画しない
            this._historyTimer = setInterval(() => this._refreshHistory(), 3000);
        } else if (this._historyTimer) {
            clearInterval(this._historyTimer);
            this._historyTimer = null;
        }
        console.log(`[MicButton] マイクパネル: ${this._panelOpen ? 'OPEN' : 'CLOSE'}`);
    }

    /** パネルの履歴セクションを最新の直近発話で描画し直す（内容が変わったときだけ） */
    async _refreshHistory() {
        const container = this._shadow.getElementById('micPanelHistory');
        if (!container) return;
        let utterances = [];
        try {
            const res = await fetch(`/api/conversation/recent?n=${PANEL_HISTORY_COUNT}`);
            if (res.ok) utterances = (await res.json()).utterances || [];
        } catch (e) {
            console.warn('[MicPanel] 履歴取得失敗:', e);
            // 一時的な取得失敗では表示中の履歴を消さない（初回のみ空表示に進む）
            if (this._lastHistoryKey !== null) return;
        }
        const key = JSON.stringify(utterances);
        if (key === this._lastHistoryKey) return;
        this._lastHistoryKey = key;
        container.textContent = '';
        if (utterances.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'mic-panel-empty';
            empty.textContent = '（このセッションの履歴はまだありません）';
            container.appendChild(empty);
            return;
        }
        for (const u of utterances) {
            const row = document.createElement('div');
            row.className = 'mic-panel-utt';
            const time = document.createElement('span');
            time.className = 'utt-time';
            time.textContent = (u.time || '').slice(11, 16);  // "HH:MM"
            const who = document.createElement('span');
            who.className = 'utt-who' + (u.speaker === 'user' ? ' user' : '');
            who.textContent = u.speaker === 'user' ? 'マスター' : 'ミク';
            const text = document.createElement('span');
            text.textContent = u.text || '';
            row.append(time, who, text);
            container.appendChild(row);
        }
        container.scrollTop = container.scrollHeight;  // 最新を見せる（上限の高さを超えたら上へスクロールしてさかのぼる）
    }

    /** レベルメーターセクションの表示を、パネル表示×マイク状態から反映する */
    _applyLevelVisibility() {
        const levelSection = this._shadow.getElementById('micLevelSection');
        if (!levelSection) return;
        const mic     = this._getMic();
        const state   = mic ? mic.getState() : MicState.OFF;
        const visible = this._panelOpen && state !== MicState.OFF;
        levelSection.style.display = visible ? 'flex' : 'none';
        if (!visible) {
            const fill = this._shadow.getElementById('micLevelFill');
            if (fill) fill.style.width = '0%';
        }
    }

    /**
     * マイクボタンの長押し判定を設定する（カメラボタンの長押しプレビューと同じ仕組み）。
     * 短押し = onShortPress()（OFF/認識/AUTO のトグル）、長押し(500ms) = マイクパネルのトグル。
     * @param {HTMLElement} btn
     * @param {function} onShortPress
     */
    setupButtonEvents(btn, onShortPress) {
        attachPressGesture(btn, {
            onShortPress,
            onHoldStart: () => this.togglePanel(),
        });
    }

    /** RMS 値からレベルメーターバーを更新する */
    updateLevel(detail) {
        const { rms } = detail;
        const db      = rms > 0 ? 20 * Math.log10(rms) : -60;
        const percent = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
        const fill    = this._shadow.getElementById('micLevelFill');
        if (fill) fill.style.width = percent + '%';
    }

    /** マイクモードに応じてボタン表示と音声のみレイアウト（AUTO時の入力欄非表示）を更新する */
    updateModeUI(micMode) {
        const btn = this._shadow.getElementById('micButton');
        if (!btn) return;

        btn.dataset.micMode = micMode === 'AUTO' ? 'AUTO' : '';

        const mic   = this._getMic();
        const state = mic ? mic.getState() : MicState.OFF;
        btn.title = UI_STRINGS.micButton[this._titleKey(state, micMode)] || '';

        // AUTO（即時送信）の間は入力欄・話すボタンを隠す（編集の機会がないため）
        const inputSection = this._shadow.querySelector('.input-section');
        if (inputSection) inputSection.classList.toggle('voice-only', micMode === 'AUTO');
    }
}
