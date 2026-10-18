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
 * menuController.js - miku-chat の操作ボタン群（メニュー/トグル）を担当する内部部品
 *
 * MikuChat の Shadow DOM を引数に取り、以下を配線する:
 *   - 字幕ボタン（返答テキスト表示 ON/OFF。表示本体は miku-caption）
 *   - 無応答リマインダーのミュートボタン
 *   - 会話メニュー（字幕・リマインダーをまとめる親）
 *   - 表示メニュー（カメラ切替 → CameraManager へ委譲 / 視線 → GAZE_SET_ENABLED 発火 /
 *     画面ロック → LOCK_ENTER 発火 / 全画面 → Fullscreen API）
 *
 * 状態（captionEnabled / reminderMuted / gazeEnabled）はここが単独で保持する（永続化なし。
 * 既定は字幕・ミュートが OFF、視線が ON——視線はカメラ ON のときだけ効くので、既定 ON でも勝手には動かない）。
 * 字幕 ON 切替時の「voice:end 待ちターンを閉じる」処理だけは MikuChat のターン状態に
 * 依存するため、コールバック（onCaptionToggle）で通知して MikuChat 側に任せる。
 *
 * モジュール間通信は CustomEvent（events.js の定数）経由のみ。
 */

import { EVENTS } from '../../modules/events.js';
import { UI_STRINGS } from '../../modules/uiStrings.js';
import { appState } from '../../modules/appState.js';
import { canTrackFace } from '../../modules/gazeMath.js';

export class MenuController {
    /**
     * @param {ShadowRoot} shadow
     * @param {{ onCaptionToggle?: (enabled: boolean) => void,
     *           onCameraSwitch?: () => Promise<boolean> }} [callbacks]
     */
    constructor(shadow, callbacks = {}) {
        this._shadow = shadow;
        this._cb = callbacks;
        /** @type {boolean} 返答テキスト字幕の表示ON/OFF（既定OFF・永続化なし。判定は appState） */
        this.captionEnabled = false;
        /** @type {boolean} 無応答リマインダーのミュート状態（既定false=ミュートなし・永続化なし） */
        this.reminderMuted = false;
        /** @type {boolean} 視線追従（既定ON・永続化なし。カメラONのときだけ効く） */
        this.gazeEnabled = true;
        /** @type {boolean} いま顔を追える映像があるか（カメラON かつインカメラ）。false の間はボタンを DISABLED 表示 */
        this._gazeUsable = false;

        this._initCaption();
        this._initReminder();
        this._initConversationMenu();
        this._initDisplayMenu();
        this._initGaze();
    }

    // ---- 視線ボタン（表示メニュー内） ---------------------------------------

    _initGaze() {
        const btn = this._shadow?.getElementById('gazeButton');
        if (!btn) return;
        this._dispatchGazeEnabled();
        this._applyGazeState(btn);
        // カメラ OFF（またはアウトカメラ）の間は「今この瞬間に使えない」ので DISABLED 表示にする
        // （カメラ切替ボタンと同じ流儀・browser_conventions §8。2026-09-27 本人指示）
        document.addEventListener(EVENTS.CAMERA_STREAM_CHANGE, (e) => {
            this._gazeUsable = canTrackFace(e.detail?.stream || null);
            this._applyGazeState(btn);
        });
        // 続けて他の項目も触れるよう、メニューは閉じない（字幕・リマインダーと同じ）
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            // 使えない間は切り替えず、理由をトーストで知らせる（disabled にすると click が来ず理由を出せない）
            if (!this._gazeUsable) {
                window.__mikuDebugToast?.(`👀 ${UI_STRINGS.gaze.DISABLED}`);
                return;
            }
            this.gazeEnabled = !this.gazeEnabled;
            this._dispatchGazeEnabled();
            this._applyGazeState(btn);
        });
    }

    _dispatchGazeEnabled() {
        document.dispatchEvent(new CustomEvent(EVENTS.GAZE_SET_ENABLED, {
            detail: { enabled: this.gazeEnabled }
        }));
    }

    _applyGazeState(btn) {
        if (!this._gazeUsable) {
            btn.dataset.gazeState = 'DISABLED';
            btn.title = UI_STRINGS.gaze.DISABLED;
            return;
        }
        btn.dataset.gazeState = this.gazeEnabled ? 'ON' : 'OFF';
        btn.title = this.gazeEnabled ? UI_STRINGS.gaze.ON : UI_STRINGS.gaze.OFF;
    }

    // ---- 字幕ボタン ---------------------------------------------------------

    _initCaption() {
        const btn = this._shadow?.getElementById('captionButton');
        if (!btn) return;
        appState.setCaptionEnabled(this.captionEnabled);
        this._applyCaptionState(btn);
        btn.addEventListener('click', () => {
            this.captionEnabled = !this.captionEnabled;
            appState.setCaptionEnabled(this.captionEnabled);
            this._applyCaptionState(btn);
            this._applyConversationState();
            document.dispatchEvent(new CustomEvent(EVENTS.CAPTION_SET_ENABLED, {
                detail: { enabled: this.captionEnabled }
            }));
            // ターン状態に依存する後処理（voice:end 待ちターンを閉じる）は MikuChat に委ねる
            this._cb.onCaptionToggle?.(this.captionEnabled);
        });
    }

    _applyCaptionState(btn) {
        btn.dataset.captionState = this.captionEnabled ? 'ON' : 'OFF';
        btn.title = this.captionEnabled ? UI_STRINGS.caption.ON : UI_STRINGS.caption.OFF;
    }

    // ---- 無応答リマインダーのミュートボタン ---------------------------------

    _initReminder() {
        const btn = this._shadow?.getElementById('reminderButton');
        if (!btn) return;
        this._dispatchReminderEnabled();
        this._applyReminderState(btn);
        btn.addEventListener('click', () => {
            this.reminderMuted = !this.reminderMuted;
            this._dispatchReminderEnabled();
            this._applyReminderState(btn);
            this._applyConversationState();
        });
    }

    /** inactivityTimer.js へは「有効/無効」で伝える（ミュート状態の反転） */
    _dispatchReminderEnabled() {
        document.dispatchEvent(new CustomEvent(EVENTS.APP_REMINDER_ENABLED, {
            detail: { enabled: !this.reminderMuted }
        }));
    }

    _applyReminderState(btn) {
        // アイコンは fa-bell-slash で固定（他ボタンと同様、状態は色のみで表す）
        btn.dataset.reminderState = this.reminderMuted ? 'ON' : 'OFF';
        btn.title = this.reminderMuted ? UI_STRINGS.reminder.ON : UI_STRINGS.reminder.OFF;
    }

    // ---- 会話メニュー（字幕／リマインダーをまとめる親） ---------------------

    _initConversationMenu() {
        const btn  = this._shadow?.getElementById('conversationButton');
        const menu = this._shadow?.getElementById('conversationMenu');
        if (!btn || !menu) return;
        btn.title = UI_STRINGS.conversation.menu;

        const close = () => { menu.classList.remove('open'); this._applyConversationState(); };
        const open  = () => { menu.classList.add('open'); this._applyConversationState(); };
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            menu.classList.contains('open') ? close() : open();
        });
        // メニュー外のタップで閉じる（メニュー内の字幕/リマインダー操作では閉じない＝続けて切替可）
        document.addEventListener('click', () => {
            if (menu.classList.contains('open')) close();
        });
        menu.addEventListener('click', (e) => e.stopPropagation());

        this._applyConversationState();
    }

    /** 会話メニュー親ボタンの色: メニュー開=OPEN、閉時は子が有効（字幕ON or リマインダーミュートON）なら ON */
    _applyConversationState() {
        const btn  = this._shadow?.getElementById('conversationButton');
        const menu = this._shadow?.getElementById('conversationMenu');
        if (!btn || !menu) return;
        const active = this.captionEnabled || this.reminderMuted;
        btn.dataset.conversationState = menu.classList.contains('open') ? 'OPEN' : (active ? 'ON' : 'OFF');
    }

    // ---- 表示メニュー（画面ロック／全画面） ---------------------------------

    _initDisplayMenu() {
        const dispBtn = this._shadow?.getElementById('displayButton');
        const menu    = this._shadow?.getElementById('displayMenu');
        const lockBtn = this._shadow?.getElementById('lockButton');
        const fsBtn   = this._shadow?.getElementById('fullscreenButton');
        const camBtn  = this._shadow?.getElementById('cameraSwitchButton');
        if (!dispBtn || !menu || !lockBtn || !fsBtn) return;

        dispBtn.title = UI_STRINGS.display.menu;
        lockBtn.title = UI_STRINGS.display.lock;

        const closeMenu = () => { menu.classList.remove('open'); this._applyDisplayState(); };
        const openMenu  = () => { menu.classList.add('open'); this._applyDisplayState(); };

        dispBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            menu.classList.contains('open') ? closeMenu() : openMenu();
        });
        document.addEventListener('click', () => {
            if (menu.classList.contains('open')) closeMenu();
        });

        // 全画面（ON/OFFトグル。自分で抜けた場合も fullscreenchange で同期）
        fsBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            try {
                if (document.fullscreenElement) await document.exitFullscreen();
                else await document.documentElement.requestFullscreen();
            } catch (err) {
                console.warn('[display] フルスクリーン切替に失敗:', err);
            }
            closeMenu();
        });
        document.addEventListener('fullscreenchange', () => this._applyDisplayState());

        // 画面ロック（オーバーレイ表示は miku-lock が担当）
        lockBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            document.dispatchEvent(new CustomEvent(EVENTS.LOCK_ENTER));
            closeMenu();
        });

        // カメラ切替（カメラ制御そのものは CameraManager の責務。ここは配線のみ）。
        // 使えない状況では CameraManager 側が理由をトーストするので、メニューは閉じない
        camBtn?.addEventListener('click', async (e) => {
            e.stopPropagation();
            const switched = await this._cb.onCameraSwitch?.();
            if (switched) closeMenu();
        });

        this._applyDisplayState();
    }

    /** 表示ボタン／全画面ボタンの色を実状態に同期する（背景色のみで表現） */
    _applyDisplayState() {
        const dispBtn = this._shadow?.getElementById('displayButton');
        const menu    = this._shadow?.getElementById('displayMenu');
        const fsBtn   = this._shadow?.getElementById('fullscreenButton');
        const fs = !!document.fullscreenElement;
        if (fsBtn) {
            fsBtn.dataset.fsState = fs ? 'ON' : 'OFF';
            fsBtn.title = fs ? UI_STRINGS.display.fullscreenOn : UI_STRINGS.display.fullscreenOff;
        }
        // メニューを開いている間は OPEN（緑）、閉じているときは全画面状態に従う
        if (dispBtn && menu) {
            dispBtn.dataset.displayState = menu.classList.contains('open') ? 'OPEN' : (fs ? 'ON' : 'OFF');
        }
    }
}
