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
 * pressGesture.js - 長押し/短押しの判定ヘルパー
 *
 * mousedown/touchstart から thresholdMs 経過したら「長押し」とみなし onHoldStart() を呼ぶ。
 * その後 mouseup/touchend で指を離すと onHoldEnd() を呼ぶ（長押し中の表示を閉じる用途など）。
 * thresholdMs 未満で離した場合は「短押し」とみなし onShortPress() を呼ぶ（onHoldStart は呼ばれない）。
 *
 * 3コールバックはすべて省略可能。用途に応じて必要なものだけ渡す:
 *   - トグル系（マイクパネル・カメラプレビュー）: onShortPress + onHoldStart のみ
 *   - 押している間だけ表示する系（ヘルプボタン）: onHoldStart + onHoldEnd のみ
 */
export function attachPressGesture(el, { thresholdMs = 500, onHoldStart, onHoldEnd, onShortPress } = {}) {
    let timer = null;
    let holding = false;

    const startPress = () => {
        holding = false;
        timer = setTimeout(() => {
            timer = null;
            holding = true;
            onHoldStart?.();
        }, thresholdMs);
    };

    const endPress = () => {
        if (timer !== null) {
            clearTimeout(timer);
            timer = null;
            onShortPress?.();
        } else if (holding) {
            holding = false;
            onHoldEnd?.();
        }
    };

    const cancelPress = () => {
        if (timer !== null) {
            clearTimeout(timer);
            timer = null;
        } else if (holding) {
            holding = false;
            onHoldEnd?.();
        }
    };

    el.addEventListener('mousedown', startPress);
    el.addEventListener('mouseup', endPress);
    el.addEventListener('mouseleave', cancelPress);

    el.addEventListener('touchstart', (e) => { e.preventDefault(); startPress(); }, { passive: false });
    el.addEventListener('touchend', (e) => { e.preventDefault(); endPress(); });
    el.addEventListener('touchcancel', cancelPress);
}
