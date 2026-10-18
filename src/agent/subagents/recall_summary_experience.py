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

"""経験の記憶を思い出す Subagent (recall_summary_experience)。

人間の「えーと、いつだっけ…」想起ポーズを再現するツール。
context に常時投入されていない長期記憶領域（episodic.events / semantic.visited /
procedural）から、ユーザーの参照に該当する情報をベクトル検索で取り出す。

**記憶の2層のうち「上層＝要点」を引く**。下層（話した言葉そのもの）を引くのは
`tools/retrieval/recall_detail_conversation.py`。同じ出来事が両方にあるが、こちらは
**後から要点としてまとめられたもの**が返る（設計: docs/components/vdb.md）。

※ 2026-08-27 に `recall_long_memory` から改名（実装名〔long_memory〕ではなく、
   ミクから見た意味で呼ぶため。`memory_search` との区別も付きにくかった）。

内部動作:
  1. LLM Agent が conversation_history + user_input + ai_judgements を読んで、
     検索クエリと category（events/visited/procedural/None）を決定する。
  2. VDB `/vdb/long_memory/search` を叩いてベクトル検索（必要なら category フィルタ）。
  3. 取得した hits を読みやすい形式に整形して文字列で返す。

権限レベル: 1 (retrieval / 副作用なし)
副作用: なし
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import requests
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agent.agent_logger import record_llm_usage, log_subagent_request, log_subagent_response, mark_prompt
from agent.config import (
    configure_env_from_settings,
    get_pydantic_ai_flash_model,
    get_vdb_base_url,
    load_settings,
)

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
_settings = load_settings()
configure_env_from_settings(_settings)
_VDB_BASE_URL = get_vdb_base_url(_settings)


# =============================================================================
# 入出力スキーマ
# =============================================================================


class RecallSummaryExperienceInput(BaseModel):
    """recall_summary_experience の入力（runner ループ層から渡される）。"""
    conversation_history: str = Field(default="", description="ここまでの会話本体")
    user_input: str = Field(default="", description="ユーザーの新たな入力")
    main_chat_output_json: str = Field(
        default="",
        description="メインエージェントの直前の構造化出力（MainChatOutput）を JSON 文字列化したもの。"
                    "purpose / inner_thoughts / requires_tool / tool_name / response 等を含む。",
    )
    tool_results: str = Field(default="", description="同一ターンで蓄積された既存ツール結果")


class _RecallPlan(BaseModel):
    """検索計画。LLM が立てる。"""
    query: str = Field(description="VDB に投げる意味検索クエリ。簡潔に・キーワード中心で")
    category: Optional[str] = Field(
        default=None,
        description="絞り込み category（'events' / 'visited' / 'procedural' / None=全部）",
    )
    top_k: int = Field(default=5, ge=1, le=15, description="取得件数")
    at_random: bool = Field(
        default=False,
        description=(
            "特定のことを探すのではなく、ふと何かを思い出したいときは true にする。"
            "そのときクエリは使われず、無作為に選ばれた記憶が返る"
        ),
    )
    rationale: str = Field(description="なぜこのクエリとカテゴリを選んだかの理由（短文）")


# =============================================================================
# Planner Agent（内部 LLM）
# =============================================================================


_PLANNER_SYSTEM_PROMPT = """あなたは長期記憶検索プランナーです。

メインエージェント（初音ミク）が「ユーザーが過去のことを参照してきたので長期記憶を取り出したい」と判断した状況です。
以下の情報から、最適な検索クエリとカテゴリを決定してください。

# 検索可能なカテゴリ

- events: 過去のエピソード記憶（出来事・体験の要約: いつ・どこで・誰と・何をしたか・どう感じたか）
- visited: 訪問した場所の詳細（名前・日付・印象）
- procedural: 手続き的記憶（行動パターン・癖・習慣・ルール）
- None（指定なし）: 全カテゴリ横断（複数カテゴリにまたがる可能性が高いとき）

# 決定ルール

1. ユーザー入力とメインエージェントの inner_thoughts を読み、何を思い出したいかを把握する
2. もっとも適切な検索クエリを 1 つ作る（できれば 3〜10 語の名詞句中心）
3. category は **ユーザー入力が明らかに 1 カテゴリを指す場合のみ** 指定。曖昧なら None で横断検索
4. top_k は通常 5。1 件だけほしいことが明らかなら 3、幅広く欲しいなら 10
5. **探している対象が特に無く「ふと昔のことを思い出したい」だけのときは at_random を true にする**。
   そのときクエリは使われず、無作為に選ばれた記憶が返る。
   inner_thoughts が「何か思い出してみよう」「昔のことを振り返りたい」のように
   対象を指していないときがこれにあたる
6. rationale には選択理由を 1〜2 文で

# 出力スキーマ

JSON: {"query": str, "category": "events"|"visited"|"procedural"|null, "top_k": int, "at_random": bool, "rationale": str}
"""


_planner_agent: Agent[None, _RecallPlan] = Agent(
    model=get_pydantic_ai_flash_model(_settings),
    output_type=_RecallPlan,
    system_prompt=_PLANNER_SYSTEM_PROMPT,
)


def _build_planner_user_prompt(input_: RecallSummaryExperienceInput) -> str:
    return (
        f"## ここまでの会話内容\n{input_.conversation_history}\n\n"
        f"## ユーザーの新たな入力\n{input_.user_input}\n\n"
        f"## メインエージェントの直前の判断（MainChatOutput JSON 全体）\n{input_.main_chat_output_json}\n\n"
        f"## 既存ツール結果（このターン内で得たもの）\n{input_.tool_results}"
    )


# =============================================================================
# VDB 検索呼び出し
# =============================================================================


def _search_vdb(query: str, category: Optional[str], top_k: int) -> list[dict]:
    """VDB の /vdb/long_memory/search を叩く。失敗時は空リスト。

    **常時 context に入っている記憶は検索対象から外す**（`always_in_context=False`・2026-08-26）。
    想起は「いま見えていないものを思い出す」行為で、既に見えているものを引き直しても
    枠を埋めるだけになる。無作為抽出（`_pick_random`）は元からこの条件で絞っていたが、
    クエリ検索側が漏れていた（実測: category 未指定のとき上位8件すべてが常時投入済みだった）。
    """
    url = f"{_VDB_BASE_URL}/vdb/long_memory/search"
    payload: dict = {"query": query, "top_k": top_k, "always_in_context": False}
    if category:
        payload["category"] = category
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json().get("hits", [])
    except Exception as exc:
        print(f"[WARN] recall_summary_experience VDB 検索失敗: {exc}")
        return []


def _pick_random(category: Optional[str], top_k: int) -> list[dict]:
    """クエリを使わず無作為に引く（「ふと思い出す」用）。失敗時は空リスト。

    VDB に無作為抽出の API は無いので、常時 context に入らないもの
    （events / visited / procedural）を全件取ってから選ぶ。
    """
    url = f"{_VDB_BASE_URL}/vdb/long_memory/by_category"
    params = {"category": category} if category else {"always_in_context": "false"}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except Exception as exc:
        print(f"[WARN] recall_summary_experience 無作為取得失敗: {exc}")
        return []
    return random.sample(items, min(top_k, len(items)))


def _format_hits(plan: _RecallPlan, hits: list[dict]) -> str:
    """検索結果を読みやすいテキストに整形する。

    LLM が次のループで参照しやすいよう、各 hit について category / document / 主要 metadata を表示。
    """
    if not hits:
        if plan.at_random:
            return f"（category={plan.category or '全カテゴリ'}）思い出せる記憶がありませんでした。"
        return (
            f"検索クエリ「{plan.query}」"
            f"（category={plan.category or '全カテゴリ'}）に該当する長期記憶は見つかりませんでした。"
        )

    lines = [
        f"検索クエリ: {'（無作為に思い出した）' if plan.at_random else plan.query}",
        f"カテゴリ: {plan.category or '全カテゴリ'}",
        f"理由: {plan.rationale}",
        f"ヒット件数: {len(hits)}",
        "",
        "--- 検索結果（スコア降順）---",
    ]
    for i, h in enumerate(hits, 1):
        meta = h.get("metadata") or {}
        cat = meta.get("category", "?")
        score = h.get("score", 0.0)
        doc = h.get("document", "")
        extras: list[str] = []
        if meta.get("date"):
            extras.append(f"date={meta['date']}")
        if meta.get("importance"):
            extras.append(f"importance={meta['importance']}")
        if meta.get("record_type"):
            extras.append(f"record_type={meta['record_type']}")
        extra_str = " (" + ", ".join(extras) + ")" if extras else ""
        item_id = h.get("id", "?")
        lines.append(f"[{i}] id={item_id} (score={score:.3f}, category={cat}{extra_str})")
        lines.append(f"    {doc}")
    return "\n".join(lines)


# =============================================================================
# 公開エントリポイント
# =============================================================================


async def run_recall_summary_experience(input_: RecallSummaryExperienceInput) -> str:
    """長期記憶を検索して結果を整形した文字列で返す。

    失敗時は "[エラー] ..." 文字列を返す（例外を raise しない・runner と互換）。
    """
    try:
        user_prompt = _build_planner_user_prompt(input_)
        log_subagent_request(
            "recall_summary_experience",
            mark_prompt("recall_summary_experience.planner.system", _PLANNER_SYSTEM_PROMPT),
            user_prompt,
        )

        # 1. LLM で検索計画を立てる
        plan_result = await _planner_agent.run(user_prompt)
        record_llm_usage(_planner_agent, plan_result, kind="recall_plan")
        plan: _RecallPlan = plan_result.output

        # 2. VDB 検索（ふと思い出したいときはクエリを使わず無作為に引く）
        if plan.at_random:
            hits = _pick_random(plan.category, plan.top_k)
        else:
            hits = _search_vdb(plan.query, plan.category, plan.top_k)

        # 3. 整形
        formatted = _format_hits(plan, hits)

        log_subagent_response(
            "recall_summary_experience",
            formatted,
            meta={
                "query": plan.query,
                "category": plan.category,
                "top_k": plan.top_k,
                "at_random": plan.at_random,
                "hit_count": len(hits),
            },
        )
        return formatted

    except Exception as exc:
        log_subagent_response("recall_summary_experience", None, error=repr(exc))
        return f"[エラー] recall_summary_experience 失敗: {exc!r}"
