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

"""**LLM の使用量と費用を1箇所で数える**（2026-08-23）。

## なぜ切り出したか

集計は `compose_v3._call` の中にあり、**そこを通らない呼び出しが5箇所あった**——
編成・伴奏・ハモリ・選抜・関門。結果:

  ・**報告していた費用が過少だった**（「1曲 $0.92」には上の5工程が入っていない）
  ・**再生モード（LLM を呼ばない検証）で差し替えられなかった**——
    LLM を呼ばないはずが、5工程は実際に呼んでいた

「同じことをする場所が複数ある」型の問題で、今日ずっと直してきたのと同じ。
集計と差し替えの入口を**ここ1つ**に寄せる。

## 使う側

    from song_usage import record
    r = agent.run_sync(prompt)
    record(agent, r, "編成")
    return r.output

## 単価（2026-10-08 に直した）

以前は tier（pro/flash/lite）ごとの単価で数えていたが、**表の数字が実際のモデルと合っていなかった**——
flash の行は gemini-3.5-flash の単価（$1.50/$9.00）で、曲は gemini-3.8-flash で動いている。lite も
$0.50/$3.00 と書いてあったが実際は $0.25/$1.50。キャッシュから読んだ入力の割引も入っていなかった。
いまは**モデル名で引き、日付で切り替わる単価は期日つきで持つ**（`PRICES`）。
過去の記録の `usd` は書き換えない（その時点の表で数えた値のまま）。
"""
from __future__ import annotations

from datetime import date
from typing import Any

# 1曲ぶんの記録（`compose_v3.USAGE` が**同じ実体**を指す）
USAGE: list[dict] = []

# ★**単価の表はここが正**（$/100万トークン・入力／キャッシュから読んだ入力／出力）。
#   モデル名ごとに「この日から」の単価を古い順に並べる（期日の無い単価は `None` から）。
#   出典: https://ai.google.dev/gemini-api/docs/pricing （2026-10-08 取得）。
#   出力の単価は思考のトークンにもかかる（Gemini は思考も出力として課金し、`output_tokens` に含まれる）。
PRICES: dict[str, list[tuple[date | None, tuple[float, float, float]]]] = {
    "gemini-3.8-flash": [
        (None, (0.75, 0.075, 3.75)),                 # 2026-12-31 まで（導入時の価格）
        (date(2027, 1, 1), (1.50, 0.15, 7.50)),
    ],
    "gemini-3.5-flash": [(None, (1.50, 0.15, 9.00))],
    # キャッシュ読みの単価は 2026-10-08 に取っていない——割引なし（入力と同じ）で数える＝多めに出る
    "gemini-3.1-flash-lite": [(None, (0.25, 0.25, 1.50))],
    # 2026-10-08 に取り直していない（以前の表の pro の行）。曲づくりの pro は 3.8 Flash を指している
    "gemini-3.1-pro": [(None, (2.00, 2.00, 12.00))],
}

#: 表に無いモデルはこの単価で数える（いちばん高い Flash＝少なく見積もらない側）。記録の `model` で見分けられる
FALLBACK_MODEL = "gemini-3.5-flash"


def model_of(agent: Any) -> str:
    """agent のモデル名（`gemini-3.8-flash` 等・小文字）。取れなければ空。"""
    try:
        m = getattr(agent, "model", None)
        name = getattr(m, "model_name", "") or (m if isinstance(m, str) else "") or ""
        return str(name).split(":")[-1].lower()
    except Exception:
        return ""


def tier_of(agent: Any) -> str:
    """agent のモデル名から tier を推定する（記録の見出し用。単価は `price_of` がモデル名で引く）。"""
    name = model_of(agent)
    if "lite" in name:
        return "lite"
    if "pro" in name:
        return "pro"
    return "flash"


def price_of(model: str, on: date | None = None) -> tuple[float, float, float]:
    """そのモデルの、その日の単価（入力・キャッシュ読み・出力 $/100万トークン）。

    名前は前方一致で引く（`gemini-3.1-pro-preview` → `gemini-3.1-pro`）。表に無ければ `FALLBACK_MODEL`。
    """
    on = on or date.today()
    name = (model or "").lower()
    key = next((k for k in sorted(PRICES, key=len, reverse=True) if name.startswith(k)), FALLBACK_MODEL)
    price = PRICES[key][0][1]
    for since, p in PRICES[key]:
        if since is None or since <= on:
            price = p
    return price


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0,
             on: date | None = None) -> float:
    """費用（$）。入力はキャッシュから読んだ分を**含む**数で渡す（pydantic_ai の `input_tokens` と同じ）。"""
    pin, pcache, pout = price_of(model, on)
    cache = min(max(int(cache_read_tokens or 0), 0), int(input_tokens or 0))
    return ((int(input_tokens or 0) - cache) * pin + cache * pcache + int(output_tokens or 0) * pout) / 1e6


def usage_of(result: Any) -> Any:
    """実行結果から使用量（`RunUsage`）を取り出す。

    pydantic_ai 1.107 から `usage` は**プロパティ**（括弧つきで呼ぶと非推奨の警告）。古い形（メソッド）や、
    テストの代役（`usage=lambda: ...`）でも取れるようにしておく。取れなければ None。
    """
    u = getattr(result, "usage", None)
    if u is None:
        return None
    if not hasattr(u, "input_tokens") and callable(u):
        u = u()
    return u if hasattr(u, "input_tokens") else None


def record(agent: Any, result: Any, label: str, *, on: date | None = None) -> None:
    """1回ぶんの使用量を積む。**失敗しても曲を落とさない**。"""
    try:
        u = usage_of(result)
        if u is None:
            return
        model = model_of(agent)
        inp = int(getattr(u, "input_tokens", 0) or 0)
        out = int(getattr(u, "output_tokens", 0) or 0)
        cache = int(getattr(u, "cache_read_tokens", 0) or 0)
        details = getattr(u, "details", None) or {}
        USAGE.append({"step": label, "tier": tier_of(agent), "model": model,
                      "in": inp, "out": out, "cache": cache,
                      "thoughts": int(details.get("thoughts_tokens", 0) or 0) if isinstance(details, dict) else 0,
                      "usd": cost_usd(model, inp, out, cache, on)})
    except Exception:
        pass


def run_counted(agent: Any, label: str, *args: Any, **kw: Any) -> Any:
    """`agent.run_sync(*args, **kw)` を呼び、使用量を積んで結果を返す。

    `song_llm.run_sync`（工程の共通の入口）を通らない呼び出し——カバーの区間・歌詞・整列、Web 検索、
    検査官、申し送り、構成の変更——が費用に入っていなかった（2026-10-08）。そういう所はこれで呼ぶ。
    """
    r = agent.run_sync(*args, **kw)
    record(agent, r, label)
    return r


def reset() -> None:
    USAGE.clear()


def summary() -> str:
    if not USAGE:
        return "  （LLM 呼び出しなし）"
    lines = [f"  {u['step']:<12}{u['tier']:<6} 入力 {u['in']:>6,} 出力 {u['out']:>6,} "
             f"${u['usd']:.4f}" for u in USAGE]
    lines.append(f"  {'合計':<18} 入力 {sum(u['in'] for u in USAGE):>6,} "
                 f"出力 {sum(u['out'] for u in USAGE):>6,} "
                 f"${sum(u['usd'] for u in USAGE):.4f}")
    return "\n".join(lines)
