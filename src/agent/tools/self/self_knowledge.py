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

"""自己機能認識ツール。

KV サービス（http://...:8010/kv?key=self_function_recognition）から
ミク自身の機能・能力・思考プロセス・記憶の仕組み等を記載したテキストを取得する。

Dify「自己能力認識取得」HTTP-request ノード相当。

権限レベル: 2 (self)
副作用: なし（KV GET のみ）
"""
from __future__ import annotations

import requests
from pydantic import BaseModel, Field

from agent.config import get_kv_base_url, load_settings
from agent.schemas import Failure, Success

_settings = load_settings()
_KV_BASE_URL = get_kv_base_url(_settings)
_KV_KEY = "self_function_recognition"


class SelfKnowledgeInput(BaseModel):
    """自己知識を照会する際の入力。（現状は引数不要だが拡張用に定義）"""

    query: str = Field(
        default="",
        description="知りたい自己機能の側面（空文字の場合は全文を返す）",
    )


class SelfKnowledgeValue(BaseModel):
    text: str = Field(description="自己機能認識テキスト")


SelfKnowledgeResult = Success[SelfKnowledgeValue] | Failure


def run_self_knowledge(input_: SelfKnowledgeInput) -> SelfKnowledgeResult:
    """KV サービスから self_function_recognition を取得して返す。"""
    try:
        resp = requests.get(f"{_KV_BASE_URL}/kv", params={"key": _KV_KEY}, timeout=10)
        if resp.status_code == 404:
            return Failure(reason=f"self_knowledge_not_found: key={_KV_KEY!r}", retryable=False)
        if resp.status_code != 200:
            return Failure(
                reason=f"kv_get HTTP {resp.status_code}: {resp.text[:200]}",
                retryable=True,
            )
        return Success[SelfKnowledgeValue](value=SelfKnowledgeValue(text=resp.text))
    except requests.exceptions.ConnectionError:
        return Failure(reason="kv_service_unreachable", retryable=True)
    except Exception as exc:
        return Failure(reason=f"unexpected: {exc!r}", retryable=False)
