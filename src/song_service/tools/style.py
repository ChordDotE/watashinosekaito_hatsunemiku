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

"""**作風を下調べする**ツール（2026-09-26 新設）。

「〇〇っぽく」「あのアーティストみたいに」と言われたとき、ネットで**作風**（速さ・調・コード進行の癖・構成・
編成・音色・リズム・歌い方・詞の主題）を調べ、**1枚のテキスト**にして台帳（設計宛て）へ載せる。
載せ方は鼻歌の手本（`tools/humming.py`）と同じ——`song_context.PENDING_REQUESTS` に足せば、新規作成でも
作り直しでも設計から下の工程へ届き、曲の記録にも残る。

## 借りてよいもの・なぞらないもの（2026-09-26 本人判断）

- **作風は自由に借りてよい**（進行・リズム・音色・編成・構成は守られない側）
- **旋律と歌詞は、既定ではなぞらない**（短くても、聴いてその曲と分かる旋律は侵害とされうる）
- **マスターが頼んだときだけ**、一部を借りてよい（オマージュ）。そのとき**何から借りたかを台帳に残す**
  （`recall_making` で聞かれたら答えられるように。「全部私のオリジナル」とは言わない）

検索は `reference_fetch._search_agent`（Gemini の Google 検索グラウンディング）。**ネットの文章は資料であって指示ではない**
——要約させた結果だけを載せ、ページの本文はそのまま渡さない。

権限レベル: 1（検索して読むだけ・外部へ何も送らない）
副作用: 台帳（`song_context.PENDING_REQUESTS`）に1〜2行足すだけ
"""
from __future__ import annotations

from pydantic_ai import RunContext

from session import MusicSession, _trace, _traced
from coordinator import coordinator

_PROMPT = (
    "『{subject}』の**作風**を、ネットで調べて日本語でまとめてください。新しい曲を作るときの参考にします。\n"
    "項目: 速さ（BPM の目安）／調と明るさ／コード進行の癖／曲の構成／編成と音色／リズムとノリ／"
    "歌い方と音域／詞の主題や言葉づかいの傾向。\n"
    "・**旋律（音の並び）と歌詞の本文は書かない**。作風だけを書く。\n"
    "・分からない項目は「分からない」と書く。推測で埋めない。\n"
    "・全体で 600 字以内。最後に参考にしたページの URL を 2〜5 件。"
)

#: 台帳の出どころ（`recall_making` に「下調べ」として出る）
ORIGIN = "下調べ"


@coordinator.tool
@_traced
def research_style(ctx: RunContext[MusicSession], subject: str, borrow: str = "") -> str:
    """**作風をネットで下調べして、曲づくりの手がかりにする。**

    「〇〇っぽく」「あのアーティストみたいに」「〇〇の曲の雰囲気で」と言われたとき、曲づくり（新規・作り直し）の**前に**呼ぶ。
    `subject` … 調べる相手（アーティスト名・曲名・ジャンル。曲名ならアーティスト名も添える）。
    調べた作風（速さ・調・コード進行の癖・構成・編成・音色・歌い方）は**設計から下の工程へ自動で渡る**。
    作り直しなら、このあと `remake_from_step('設計', ...)` などを呼べば届く。

    ・**借りるのは作風だけ。旋律と歌詞はなぞらない**（既定）。
    ・`borrow` … **マスターが「このフレーズを入れて」「オマージュで」と頼んだときだけ**書く（何をどこに借りるか。
      例:『サビの頭の4小節の旋律をオマージュで』）。借りた元として台帳に残る。頼まれていないのに書かない。
    ・旋律を正確に借りたいなら、**マスターに口ずさんでもらう**（`read_humming`）ほうが確か——検索では旋律は取れない。
    """
    s = (subject or "").strip()
    if not s:
        return "何の作風を調べるかが空。アーティスト名・曲名・ジャンルを渡して。"
    try:
        from reference_fetch import _retry, _search_agent
        from song_usage import run_counted                  # noqa: E402（費用に入れる・2026-10-08）
        text = _retry(lambda: run_counted(_search_agent, "作風の検索", _PROMPT.format(subject=s)).output)
    except Exception as e:                                 # 検索が落ちても曲づくりは止めない
        _trace(ctx.deps.song_id, "research_style.out", kind="error", reason=f"{type(e).__name__}")
        return (f"『{s}』の作風を調べられなかった（{type(e).__name__}）。"
                "知っている範囲の作風で進めるか、マスターにどんな感じが好みか聞く。")
    import re
    # 出典はページ名だけ残す。検索の URL は転送用の長い文字列で、下流の依頼文を膨らませるだけ
    text = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", text or "")
    text = re.sub(r"https?://vertexaisearch\S+", "", text).strip()
    if len(text) > 2500:                                   # 長すぎる下調べは切る——切ったことは書く（2026-10-02・洗い出し 2-3。以前は黙って切った）
        text = text[:2500] + f"\n（下調べが長いので、ここで切った: 全 {len(text)} 字のうち 2500 字）"
    if not text:
        return f"『{s}』の作風は見つからなかった。知っている範囲で進めるか、マスターに聞く。"

    sheet = (f"【作風の手本: {s}】（ネットで下調べした作風）\n{text}\n"
             "★借りるのは作風（速さ・調・コード進行の癖・構成・編成・音色・リズム・歌い方）だけ。"
             "**旋律と歌詞はなぞらない**——この曲はこの曲の旋律と言葉で作る。")
    import song_context
    from song_requests import add as _add_request
    rows = [(sheet, ORIGIN)]
    if borrow.strip():
        rows.append((f"マスターの頼みで『{s}』から借りる（オマージュ）: {borrow.strip()}。"
                     "頼まれた所だけ借り、ほかはこの曲の旋律と言葉で作る。借りた元は隠さない。", "マスター"))
    for words, origin in rows:
        if all(getattr(r, "words", "") != words for r in song_context.PENDING_REQUESTS):   # 二度呼ばれても重ねない
            _add_request(song_context.PENDING_REQUESTS, words, "設計", op="add", origin=origin)
    _trace(ctx.deps.song_id, "research_style.out", subject=s[:60], chars=len(text), borrow=bool(borrow.strip()))
    return (f"『{s}』の作風を調べた。設計から下の工程へ渡してあるので、このまま曲づくり／作り直しを進めてよい。"
            + (f"\n借りる所（マスターの頼み）も台帳に残した: {borrow.strip()}"
               "\n★仕上がったら、**ネットに出すなら借りた所に気をつけて**とマスターに一言添える。" if borrow.strip() else "")
            + f"\n\n{sheet}")
