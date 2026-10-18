# -*- coding: utf-8 -*-
#
# Watashinosekaito Hatsune Miku
# Copyright (c) 2025-2026 ChordDotE. All rights reserved.
# SPDX-License-Identifier: LicenseRef-ChordDotE-NonCommercial-2.0
#
# NON-COMMERCIAL USE ONLY. NO BUSINESS USE. NO ADAPTATION.
#
# Business use (any use by or for a company, organization, or sole
# proprietor, WHETHER OR NOT ANY PAYMENT IS INVOLVED) and publishing or
# distributing modified versions are PROHIBITED IN ALL CASES; no permission
# is granted for them. Forks may be used only to propose improvements to the
# Author (e.g. pull requests). Commercial use, redistribution, and Adaptation
# are PROHIBITED without prior written permission. "Adaptation" includes
# manual reimplementation, automated translation, and generation using large
# language models or other machine learning, of software having equivalent or
# similar functionality. Permission is granted individually, in recorded form.
# Contact: https://x.com/ChordDotE
#
# 業務利用（対価の有無を問わず、法人・団体・個人事業主による利用と事業のための利用）と、
# 改変版の公開・配布は、許可の余地なく禁止します。フォークは作者への改善の提案
# （プルリクエスト等）のためだけに使えます。商用利用・再配布・翻案は、作者の許可が
# ない限り禁止します。翻案には、再実装・自動変換・大規模言語モデル等を用いた生成に
# よる、同等または類似の機能を持つソフトウェアの作成を含みます。
# 許可が必要な場合は、記録の残る形でご連絡ください。
#
# See LICENSE for full terms. 詳細は LICENSE を参照してください。
#

"""スタート画面（コントロールパネル）の起動入口。

**なぜ `-m control_panel` をやめてこのファイルを置くのか**:
配布物の保護版では `control_panel` パッケージが Nuitka で `.pyd` へ機械語化される。
`.pyd` になったパッケージに対して `python -m <パッケージ>` は使えず、
`No code object available for control_panel` で落ちる（2026-09-19 実測）。
**薄いランチャーから `main()` を呼ぶ形なら、素版・保護版のどちらでも動く。**

このファイル自体は保護版でも `.py` のまま残す（呼び出し口なので、中身は 2 行しかない）。
経緯と全体設計は docs/plan/distribution_protection.md。
"""
from __future__ import annotations


def main() -> int:
    from control_panel.__main__ import main as _panel_main
    return _panel_main()


if __name__ == "__main__":
    raise SystemExit(main())
