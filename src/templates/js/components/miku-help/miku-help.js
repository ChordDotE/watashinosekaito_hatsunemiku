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
 * miku-help.js - ヘルプボタン（長押し中だけ、隠れた操作のヒント一覧を表示する）
 *
 * <miku-avatar-live2d> 等と同じくページ直下に独立配置する小規模コンポーネント
 * （<miku-caption> と同様、テンプレートは fetch せずインラインで持つ）。
 *
 * スマホでは title 属性のホバーツールチップが機能しないため、
 * マイク・カメラの長押しジェスチャーなど「隠れた操作」を長押し中だけ一覧表示する。
 * ボタン自体はON/OFF状態を持たない（表示は押している間だけ）。
 *
 * ヒント文言は uiStrings.js の既存ツールチップ文言をそのまま参照する（二重管理を避けるため）。
 * 既知の簡略化: マイクのヒントは STT有効時の文言（micButton.LISTENING）を常に使う。
 * STT無効設定（入力アイドル自動送信トグルにマイクボタンを流用するモード）では実際の挙動と
 * 文言が一致しない。このモードは長押しジェスチャー自体を持たないため実害は小さいと判断し、
 * 未対応のまま残している。
 */

import { UI_STRINGS } from '../../modules/uiStrings.js';
import { attachPressGesture } from '../../modules/pressGesture.js';

// 画面上のボタン並び順に合わせる。sub: true はメニュー内のサブボタン（親の直後に
// インデント表示して階層を示す）。操作ボタンを増減したらここも同期する
// （同期漏れは browser_structure_check.py の検査 3 が検出する）
const HINTS = [
    { icon: 'fa-microphone',    text: () => UI_STRINGS.micButton.LISTENING },
    { icon: 'fa-camera',        text: () => UI_STRINGS.camera.ON },
    { icon: 'fa-people-arrows', text: () => UI_STRINGS.conversation.menu },
    { icon: 'fa-comment-dots',  text: () => UI_STRINGS.caption.OFF,           sub: true },
    { icon: 'fa-bell-slash',    text: () => UI_STRINGS.reminder.OFF,          sub: true },
    { icon: 'fa-desktop',       text: () => UI_STRINGS.display.menu },
    { icon: 'fa-camera-rotate', text: () => UI_STRINGS.cameraSwitch.ON,       sub: true },
    { icon: 'fa-eye',           text: () => UI_STRINGS.gaze.OFF,              sub: true },
    { icon: 'fa-lock',          text: () => UI_STRINGS.display.lock,          sub: true },
    { icon: 'fa-expand',        text: () => UI_STRINGS.display.fullscreenOff, sub: true },
];

class MikuHelp extends HTMLElement {
    connectedCallback() {
        const shadow = this.attachShadow({ mode: 'open' });
        shadow.innerHTML = `
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
            <style>
                :host {
                    position: fixed;
                    top: 8px;
                    right: 8px;
                    z-index: 30;
                }
                #helpButton {
                    width: 26px;
                    height: 26px;
                    border-radius: 50%;
                    border: none;
                    background: rgba(153, 153, 153, 0.65);
                    color: #fff;
                    font-size: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    cursor: pointer;
                    padding: 0;
                }
                #hintPopup {
                    display: none;
                    position: absolute;
                    top: 32px;
                    right: 0;
                    width: 260px;
                    max-width: 70vw;
                    background: rgba(13, 15, 24, 0.88);
                    color: #fff;
                    border-radius: 10px;
                    padding: 10px 12px;
                    font-size: 13px;
                    line-height: 1.6;
                    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
                }
                #hintPopup.visible { display: block; }
                #hintPopup ul { list-style: none; margin: 0; padding: 0; }
                #hintPopup li { display: flex; align-items: flex-start; gap: 8px; padding: 4px 0; }
                /* メニュー内サブボタンはインデントして階層を示す */
                #hintPopup li.sub { padding-left: 20px; }
                #hintPopup i { width: 16px; text-align: center; margin-top: 2px; opacity: 0.8; }
            </style>
            <button type="button" id="helpButton" title="長押しでボタンの説明を表示">
                <i class="fas fa-question"></i>
            </button>
            <div id="hintPopup">
                <ul>
                    ${HINTS.map(h => `<li${h.sub ? ' class="sub"' : ''}><i class="fas ${h.icon}"></i><span>${h.text()}</span></li>`).join('')}
                </ul>
            </div>
        `;

        const btn = shadow.getElementById('helpButton');
        const popup = shadow.getElementById('hintPopup');
        attachPressGesture(btn, {
            onHoldStart: () => popup.classList.add('visible'),
            onHoldEnd:   () => popup.classList.remove('visible'),
        });
    }
}

customElements.define('miku-help', MikuHelp);
