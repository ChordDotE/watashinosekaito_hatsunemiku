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

"""Web 検索ツール。

Tavily Search API を使い、Web から情報を取得する。
settings.json の api_keys.tavily を参照する。

権限レベル: 1 (retrieval)
副作用: なし（受信のみ）
"""
from __future__ import annotations

import json

import requests
from pydantic import BaseModel, Field

from agent.config import get_tavily_api_key, load_settings
from agent.schemas import Failure, Success

_settings = load_settings()

_TAVILY_URL = "https://api.tavily.com/search"
_TIMEOUT_SEC = 60


class WebSearchInput(BaseModel):
    query: str = Field(description="検索クエリ。日本語・英語どちらでも可")
    max_results: int = Field(default=5, ge=1, le=10, description="取得件数（1〜10）")


class WebSearchResultItem(BaseModel):
    title: str
    url: str
    content: str = Field(description="本文の抜粋")
    score: float = Field(default=0.0)


class WebSearchValue(BaseModel):
    results: list[WebSearchResultItem]
    formatted: str = Field(description="LLM に渡す整形済みテキスト")


WebSearchResult = Success[WebSearchValue] | Failure


def _format_results(items: list[WebSearchResultItem]) -> str:
    """検索結果を JSON 形式に整形する（memory_search の「検索結果パース」互換に揃える）。

    結果なし → "" を返す（web_search_agent が「（結果なし）」を補う）

    出力例:
      [{"検索結果": 1, "title": "...", "url": "...", "content": "..."}, ...]

    生テキストを連結すると、取得したページ本文の改行や見出し記号がそのまま
    プロンプトの構造として読まれる（本文側から枠を閉じたり偽の見出しを作れる）。
    json.dumps を通すと本文は content の中から出られず、url が同じレコードに
    残るので出所が本文から切り離されない。
    """
    if not items:
        return ""
    records = [
        {"検索結果": i, "title": r.title, "url": r.url, "content": r.content}
        for i, r in enumerate(items, 1)
    ]
    return json.dumps(records, ensure_ascii=False, indent=2)


def run_web_search(input_: WebSearchInput) -> WebSearchResult:
    """Tavily API で Web 検索を実行する。"""
    try:
        api_key = get_tavily_api_key(_settings)
    except KeyError:
        return Failure(reason="tavily_api_key_not_configured", retryable=False)

    try:
        resp = requests.post(
            _TAVILY_URL,
            json={
                "api_key": api_key,
                "query": input_.query,
                "search_depth": "basic",
                "max_results": input_.max_results,
            },
            timeout=_TIMEOUT_SEC,
        )
        if resp.status_code != 200:
            return Failure(
                reason=f"tavily HTTP {resp.status_code}: {resp.text[:200]}",
                retryable=(resp.status_code >= 500),
            )
        body = resp.json()
        items = [
            WebSearchResultItem(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                score=float(r.get("score", 0.0)),
            )
            for r in body.get("results", [])
        ]
        formatted = _format_results(items)
        return Success[WebSearchValue](
            value=WebSearchValue(results=items, formatted=formatted),
            summary=f"{len(items)} 件取得（query={input_.query[:50]!r}）",
        )
    except requests.exceptions.Timeout:
        return Failure(reason="tavily_timeout", retryable=True)
    except requests.exceptions.ConnectionError:
        return Failure(reason="tavily_unreachable", retryable=True)
    except Exception as exc:
        return Failure(reason=f"unexpected: {exc!r}", retryable=False)
