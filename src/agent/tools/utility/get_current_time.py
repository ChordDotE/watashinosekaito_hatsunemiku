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

"""現在時刻取得ツール。

Dify の「現在時刻」ツールに相当する汎用ユーティリティ。
複数の subagent（web_search_agent / email_compose / miku_note_compose 等）から
再利用できるよう独立ファイルに切り出している。

権限レベル: 0 (utility / 副作用なし)
副作用: なし（ローカル時刻読み取りのみ）
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from agent.schemas import Failure, Success


class GetCurrentTimeInput(BaseModel):
    timezone: str = Field(default="Asia/Tokyo", description="IANA タイムゾーン名（例: Asia/Tokyo, UTC）")
    fmt: str = Field(default="%Y-%m-%d %H:%M:%S %Z", description="strftime フォーマット")


class GetCurrentTimeValue(BaseModel):
    text: str = Field(description="フォーマット済み現在時刻")


GetCurrentTimeResult = Success[GetCurrentTimeValue] | Failure


def run_get_current_time(input_: GetCurrentTimeInput) -> GetCurrentTimeResult:
    """指定タイムゾーンの現在時刻を返す。"""
    try:
        now = datetime.now(ZoneInfo(input_.timezone))
        return Success[GetCurrentTimeValue](value=GetCurrentTimeValue(text=now.strftime(input_.fmt)))
    except Exception as exc:
        return Failure(reason=f"get_current_time_failed: {exc!r}", retryable=False)
