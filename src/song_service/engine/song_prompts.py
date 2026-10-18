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

"""**作曲工程のプロンプトを `.md` から読む**（規約: コード内ハードコード禁止）。

プロンプトの正は `song_service/prompts/*.md`。コードは名前で引くだけにする——
文章を直すのに Python を触らせない（既に `coordinator.md` / `reviewer.md` が同じ作法）。

読んだ結果は**プロセス内で覚える**（同じ .md を何度も読まない）。
編集して確かめたいときは `reload()` を呼ぶ。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


@lru_cache(maxsize=None)
def prompt(name: str) -> str:
    """`prompts/<name>.md` の中身。**無ければ落とす**（黙って空文字を渡すと沈黙で壊れる）。"""
    p = PROMPT_DIR / f"{name}.md"
    if not p.exists():
        raise FileNotFoundError(f"プロンプトが見つかりません: {p}")
    return p.read_text(encoding="utf-8")


def reload() -> None:
    """`.md` を編集したあとに読み直す（開発中の確認用）。"""
    prompt.cache_clear()
