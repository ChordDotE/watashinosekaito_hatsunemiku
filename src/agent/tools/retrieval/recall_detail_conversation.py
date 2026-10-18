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

"""過去の会話をそのまま思い出すツール（recall_detail_conversation）。

VDB サービスの /vdb/search エンドポイントを呼び出し、
**会話アーカイブ**（parents/children）から関連する発話を取得する（RAG）。

**記憶の2層のうち「下層＝逐語」を引く**。上層（要点としてまとめられた経験）を引くのは
`subagents/recall_summary_experience.py`。同じ出来事が両方にあるが、こちらは
**話した言葉そのもの**が返る（設計: docs/components/vdb.md）。

直前の会話履歴と同じデータソースなので、直近の会話に答えが含まれる場合は使用不要。

※ 2026-08-27 に `memory_search` から改名（`recall_long_memory` と紛らわしく、
   どちらが何を探すのか名前から分からなかったため）。

権限レベル: 1 (retrieval)
副作用: なし
"""
from __future__ import annotations

import json
import requests
from pydantic import BaseModel, Field

from agent.config import get_vdb_base_url, load_settings
from agent.schemas import Failure, KnowledgeSearchHit, Success

_settings = load_settings()
_VDB_BASE_URL = get_vdb_base_url(_settings)

_DEFAULT_TOP_K = _settings.get("knowledge_base", {}).get("top_k", 10)


class RecallDetailConversationInput(BaseModel):
    query: str = Field(description="検索クエリ。過去の会話から思い出したい内容を自然文で")
    top_k: int = Field(default=_DEFAULT_TOP_K, ge=1, le=30, description="取得件数（デフォルト: 設定値）")


class RecallDetailConversationValue(BaseModel):
    hits: list[KnowledgeSearchHit]
    formatted: str = Field(description="LLM に渡す整形済みテキスト（検索結果パース互換 JSON）")


RecallDetailConversationResult = Success[RecallDetailConversationValue] | Failure


def _format_hits(query: str, hits: list[KnowledgeSearchHit]) -> str:
    """ヒット結果を Dify「検索結果パース」ノード互換の JSON 形式に整形する。

    ヒットなし → "" を返す（runner.py が「検索結果なし」プレースホルダを補う）

    出力例（ヒットあり）:
      [{"検索クエリ": "query"}, {"検索結果": 1, "content": "..."}, ...]
    """
    if not hits:
        return ""
    items: list[dict] = [{"検索クエリ": query}]
    for i, h in enumerate(hits, 1):
        items.append({"検索結果": i, "content": h.parent_text})
    return json.dumps(items, ensure_ascii=False, indent=2)


def run_recall_detail_conversation(input_: RecallDetailConversationInput) -> RecallDetailConversationResult:
    """VDB サービスでベクトル + BM25 ハイブリッド検索を実行する。"""
    try:
        resp = requests.post(
            f"{_VDB_BASE_URL}/vdb/search",
            json={"query": input_.query, "top_k": input_.top_k},
            timeout=60,
        )
        if resp.status_code != 200:
            return Failure(
                reason=f"vdb_search HTTP {resp.status_code}: {resp.text[:200]}",
                retryable=True,
            )
        body = resp.json()
        hits = [KnowledgeSearchHit(**h) for h in body.get("hits", [])]
        formatted = _format_hits(input_.query, hits)
        return Success[RecallDetailConversationValue](
            value=RecallDetailConversationValue(hits=hits, formatted=formatted),
            summary=f"{len(hits)} 件ヒット（query={input_.query[:50]!r}）",
        )
    except requests.exceptions.ConnectionError:
        return Failure(reason="vdb_service_unreachable", retryable=True)
    except Exception as exc:
        return Failure(reason=f"unexpected: {exc!r}", retryable=False)
