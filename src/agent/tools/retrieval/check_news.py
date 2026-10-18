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

"""世の中の話題を眺めるツール。

RSS から見出しだけを取得する。本文もリンク先も取らないので、外部テキストが
LLM に入る量が小さく、インジェクションの余地が桁違いに小さい。
どのカテゴリを浴びるかは settings.json の interest.exposure.categories で決まる。

権限レベル: 1 (retrieval)
副作用: なし（受信のみ）
"""
from __future__ import annotations

import random
import re
import urllib.request
import xml.etree.ElementTree as ET
from urllib.error import URLError

from pydantic import BaseModel, Field

from agent.config import load_settings
from agent.schemas import Failure, Success

_settings = load_settings()

_TIMEOUT_SEC = 5
_TITLE_MAX_CHARS = 120
_FEEDS_PER_CALL = 3       # 全カテゴリを毎回叩くとリクエストが増えるので絞る
_TITLES_PER_FEED = 8
_DEFAULT_LIMIT = 6

# カテゴリ → RSS URL。**自動学習で増やさない**（fetch_allowlist が自動学習で
# 素通りするようになった前例があるため、ここは設定と実装で固定する）
FEEDS: dict[str, tuple[str, ...]] = {
    "technology": ("https://b.hatena.ne.jp/hotentry/it.rss",),
    "learning": ("https://b.hatena.ne.jp/hotentry/knowledge.rss",
                 "https://www.nhk.or.jp/rss/news/cat3.xml"),
    "life": ("https://b.hatena.ne.jp/hotentry/life.rss",),
    "fun": ("https://b.hatena.ne.jp/hotentry/fun.rss",),
    "anime_game": ("https://b.hatena.ne.jp/hotentry/game.rss",),
    "entertainment": ("https://news.google.com/rss/headlines/section/topic/ENTERTAINMENT?hl=ja&gl=JP&ceid=JP:ja",),
    "news": ("https://www.nhk.or.jp/rss/news/cat0.xml",),
}

# Google News の見出しは末尾に「 - 媒体名」が付き、他より 2 倍近く長い
_TRAILING_SOURCE = re.compile(r"\s+-\s+[^-]{2,30}$")


class CheckNewsInput(BaseModel):
    limit: int = Field(default=_DEFAULT_LIMIT, ge=1, le=12,
                       description="眺める見出しの数（1〜12）")


class CheckNewsValue(BaseModel):
    titles: list[str]
    formatted: str = Field(description="LLM に渡す整形済みテキスト")


CheckNewsResult = Success[CheckNewsValue] | Failure


def _enabled_categories() -> list[str]:
    """設定で ON になっているカテゴリ。未設定なら全部 ON とみなす。"""
    conf = (_settings.get("interest") or {}).get("exposure") or {}
    cats = conf.get("categories") or {}
    if not cats:
        return list(FEEDS)
    return [c for c in FEEDS if cats.get(c, True)]


def _clean(title: str) -> str:
    t = _TRAILING_SOURCE.sub("", " ".join(str(title).split()))
    return t[:_TITLE_MAX_CHARS]


def _fetch_titles(url: str) -> list[str]:
    """1 フィードから見出しを取る。RSS 2.0 と RDF(RSS 1.0) が混在するので
    タグのローカル名で判定する（名前空間つきでも拾える）。失敗は空リスト。"""
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_SEC) as resp:
            root = ET.fromstring(resp.read())
    except (URLError, ET.ParseError, ValueError, OSError):
        return []
    titles: list[str] = []
    for elem in root.iter():
        if elem.tag.rsplit("}", 1)[-1] != "item":
            continue
        for child in elem:
            if child.tag.rsplit("}", 1)[-1] == "title" and child.text:
                titles.append(_clean(child.text))
                break
        if len(titles) >= _TITLES_PER_FEED:
            break
    return titles


def _format_titles(titles: list[str]) -> str:
    """見出しに「自分の基準で選んでよい」を添える。

    この一文が無いと、拾う話題がすべてマスター向けになる（実測）。
    """
    lines = "\n".join(f"- {t}" for t in titles)
    return (
        "いま世の中で話題になっていること:\n"
        f"{lines}\n"
        "この中に気になるものがあれば、それについて考えたり調べたりしてよい。"
        "マスターの役に立つかどうかではなく、自分が面白いと思うかで選んでよい。"
        "気になるものが無ければ、無理に拾わなくてよい。"
    )


def run_check_news(input_: CheckNewsInput) -> CheckNewsResult:
    """RSS から見出しを集めて返す。取得に失敗しても会話は止めない。"""
    cats = _enabled_categories()
    if not cats:
        return Failure(reason="exposure_categories_all_disabled", retryable=False)

    urls = [u for c in cats for u in FEEDS[c]]
    picked = random.sample(urls, min(_FEEDS_PER_CALL, len(urls)))
    pool: list[str] = []
    for url in picked:
        pool.extend(_fetch_titles(url))

    # 同じ話題が複数フィードに載ることがあるので、順序を保ったまま重複を落とす
    seen: set[str] = set()
    uniq = [t for t in pool if t and not (t in seen or seen.add(t))]
    if not uniq:
        return Failure(reason="no_titles_fetched", retryable=True)

    titles = random.sample(uniq, min(input_.limit, len(uniq)))
    return Success[CheckNewsValue](
        value=CheckNewsValue(titles=titles, formatted=_format_titles(titles)),
        summary=f"{len(titles)} 件の見出しを取得（{len(picked)} フィード）",
    )
