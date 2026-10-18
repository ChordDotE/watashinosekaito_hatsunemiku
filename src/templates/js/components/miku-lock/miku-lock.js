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
 * miku-lock.js - 画面ロックモード（放置会話用の全画面オーバーレイ）
 *
 * <miku-help> / <miku-caption> と同じくページ直下に独立配置する小規模コンポーネント。
 * miku-chat の「表示」メニューの「ロック」から EVENTS.LOCK_ENTER を受けて起動する。
 *
 * 目的: 画面ONのまま放置して音声会話を続けるとき、画面への誤タッチ・長押しメニューを
 * 遮断しつつ、ミクの表示はスポットライトで見えるようにする。解除は「枠を一周なぞる」。
 *
 * できること / できないこと:
 * - ページ内のタッチ・長押しメニューは遮断できる
 * - OS のハードキー（音量/電源/ホーム/戻る/通知）は Web では遮断できない（設計上の限界）
 * - 画面OFFでの継続は不可（Wake Lock で「消えにくく」はするが、ネイティブではない）
 *
 * ミク枠の位置は <miku-avatar-live2d> の getBoundingClientRect() で測定する
 * （コンポーネント import はせず DOM 測定のみ。レイアウト読み取りで副作用なし）。
 */

import { EVENTS } from '../../modules/events.js';
import { UI_STRINGS } from '../../modules/uiStrings.js';

const FULL_TURN = Math.PI * 2;   // 枠を一周＝解除に必要な累積回転角
const RESET_MS = 5000;           // なぞり途中で放置したらリセットするまでの時間

class MikuLock extends HTMLElement {
    connectedCallback() {
        this._active = false;
        this._wakeLock = null;
        this._resetTimer = null;
        this._tracking = false;
        this._acc = 0;                 // ミク枠中心まわりの累積回転角
        this._lastPt = null;
        this._center = { x: 0, y: 0 };

        const shadow = this.attachShadow({ mode: 'open' });
        shadow.innerHTML = `
            <style>
                #overlay {
                    position: fixed; inset: 0; z-index: 2000;
                    touch-action: none; user-select: none;
                    -webkit-user-select: none; -webkit-touch-callout: none;
                }
                #overlay[hidden] { display: none; }
                /* スポットライト（ミク枠は明るいまま外側を暗転） */
                #hole {
                    position: fixed; border-radius: 14px;
                    box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.6);
                }
                /* 周回ガイド（点線の四角） */
                #ring {
                    position: fixed; pointer-events: none;
                    border: 3px dashed rgba(255, 255, 255, 0.9); border-radius: 14px;
                }
                /* 解除ボタン（起点。なぞり進捗を conic-gradient で表示） */
                #knob {
                    position: fixed; width: 72px; height: 72px; border-radius: 50%;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 30px; color: #fff; border: 3px solid #fff;
                    background: #8a8a90; box-shadow: 0 2px 14px rgba(0, 0, 0, 0.5);
                    cursor: pointer; touch-action: none;
                }
                /* 案内文: 位置は _layout() が解除ボタンの下に動的配置する（ミクに被せない） */
                #hint {
                    position: fixed; text-align: center;
                    color: #fff; font-size: 14px; font-weight: bold;
                    text-shadow: 0 1px 4px #000;
                }
            </style>
            <div id="overlay" hidden>
                <div id="hole"></div>
                <div id="ring"></div>
                <div id="hint">${UI_STRINGS.lock.hint}</div>
                <div id="knob">⟳</div>
            </div>
        `;

        this._overlay = shadow.getElementById('overlay');
        this._hole = shadow.getElementById('hole');
        this._ring = shadow.getElementById('ring');
        this._knob = shadow.getElementById('knob');
        this._hint = shadow.getElementById('hint');

        document.addEventListener(EVENTS.LOCK_ENTER, () => this.enter());

        // なぞり操作（起点は解除ボタン。以降は overlay がポインタをキャプチャ）
        this._knob.addEventListener('pointerdown', (e) => this._onDown(e));
        this._overlay.addEventListener('pointermove', (e) => this._onMove(e));
        this._overlay.addEventListener('pointerup', () => this._onUp());
        this._overlay.addEventListener('pointercancel', () => this._onUp());
        this._overlay.addEventListener('contextmenu', (e) => e.preventDefault());

        // 画面回転/リサイズでミク枠がずれたら再配置
        window.addEventListener('resize', () => { if (this._active) this._layout(); });
        // 背面から戻ったら Wake Lock を取り直す（背面/画面OFFで解放されるため）
        document.addEventListener('visibilitychange', () => {
            if (this._active && document.visibilityState === 'visible') this._acquireWake();
        });
    }

    enter() {
        if (this._active) return;
        this._active = true;
        this._overlay.hidden = false;
        this._resetTrace();
        this._layout();
        this._acquireWake();
    }

    exit() {
        if (!this._active) return;
        this._active = false;
        this._overlay.hidden = true;
        this._resetTrace();
        this._releaseWake();
        document.dispatchEvent(new CustomEvent(EVENTS.LOCK_EXIT));
    }

    /** ミク枠に合わせてスポットライト・周回ガイド・解除ボタンを配置 */
    _layout() {
        const avatar = document.querySelector('miku-avatar-live2d');
        const r = avatar
            ? avatar.getBoundingClientRect()
            : { left: 0, top: 0, width: window.innerWidth, height: window.innerHeight * 0.7 };
        const pad = 8;
        this._place(this._hole, r.left - pad, r.top - pad, r.width + pad * 2, r.height + pad * 2);
        const rx = r.left - pad * 2, ry = r.top - pad * 2;
        const rw = r.width + pad * 4, rh = r.height + pad * 4;
        this._place(this._ring, rx, ry, rw, rh);
        this._center = { x: r.left + r.width / 2, y: r.top + r.height / 2 };
        // 解除ボタンの起点: スマホ=枠の下辺中央 / PC=枠の右下
        const narrow = window.innerWidth <= 768;
        const kx = narrow ? (rx + rw / 2) : (rx + rw);
        const ky = ry + rh;
        const knobTop = Math.min(ky - 36, window.innerHeight - 78);
        this._knob.style.left = (kx - 36) + 'px';
        this._knob.style.top = knobTop + 'px';
        // 案内文: ミク枠に被せない位置に置く。
        //   スマホ = 解除ボタンの真下（下に入らなければ上）
        //   PC     = 解除ボタンの右横（チャット側の暗転域。右に入らなければ左横）
        const hw = Math.min(340, window.innerWidth - 16);
        this._hint.style.width = hw + 'px';
        const hintH = this._hint.offsetHeight || 40;
        if (narrow) {
            this._hint.style.textAlign = 'center';
            this._hint.style.left = Math.max(8, Math.min(kx - hw / 2, window.innerWidth - hw - 8)) + 'px';
            let hy = knobTop + 72 + 10;
            if (hy + hintH > window.innerHeight - 8) hy = knobTop - hintH - 10;
            this._hint.style.top = hy + 'px';
        } else {
            this._hint.style.textAlign = 'left';
            let hx = kx + 36 + 12;
            if (hx + hw > window.innerWidth - 8) hx = kx - 36 - 12 - hw;
            this._hint.style.left = Math.max(8, hx) + 'px';
            this._hint.style.top = Math.max(8, knobTop + 36 - hintH / 2) + 'px';
        }
    }

    _place(el, x, y, w, h) {
        el.style.left = x + 'px'; el.style.top = y + 'px';
        el.style.width = w + 'px'; el.style.height = h + 'px';
    }

    _onDown(e) {
        if (!this._active) return;
        e.preventDefault();
        this._tracking = true;
        this._acc = 0;
        this._lastPt = { x: e.clientX, y: e.clientY };
        this._setProgress(0);
        this._overlay.setPointerCapture?.(e.pointerId);
        clearTimeout(this._resetTimer);
        this._resetTimer = setTimeout(() => this._resetTrace(), RESET_MS);
    }

    _onMove(e) {
        if (!this._tracking || !this._active) return;
        const p = { x: e.clientX, y: e.clientY };
        // ミク枠中心まわりの角度差を積算（枠を一周＝2π で解除）
        const v1 = { x: this._lastPt.x - this._center.x, y: this._lastPt.y - this._center.y };
        const v2 = { x: p.x - this._center.x, y: p.y - this._center.y };
        const cross = v1.x * v2.y - v1.y * v2.x;
        const dot = v1.x * v2.x + v1.y * v2.y;
        this._acc += Math.atan2(cross, dot);
        this._lastPt = p;
        this._setProgress(Math.abs(this._acc) / FULL_TURN);
        if (Math.abs(this._acc) >= FULL_TURN) {
            this._tracking = false;
            this.exit();
        }
    }

    _onUp() {
        if (this._tracking) this._resetTrace();
    }

    _resetTrace() {
        this._tracking = false;
        this._acc = 0;
        this._lastPt = null;
        clearTimeout(this._resetTimer);
        this._setProgress(0);
    }

    /** なぞり進捗を解除ボタンの conic-gradient で表す（一周で全周が緑＝解除） */
    _setProgress(ratio) {
        const deg = Math.max(0, Math.min(360, ratio * 360));
        this._knob.style.background =
            `conic-gradient(var(--main-color, #39c5bb) ${deg}deg, #8a8a90 ${deg}deg)`;
    }

    async _acquireWake() {
        if (!('wakeLock' in navigator) || this._wakeLock) return;
        try {
            this._wakeLock = await navigator.wakeLock.request('screen');
            this._wakeLock.addEventListener('release', () => { this._wakeLock = null; });
        } catch (_) { /* 非対応/失敗は許容（画面が消えるだけ） */ }
    }

    async _releaseWake() {
        if (this._wakeLock) {
            try { await this._wakeLock.release(); } catch (_) { /* noop */ }
            this._wakeLock = null;
        }
    }
}

customElements.define('miku-lock', MikuLock);
