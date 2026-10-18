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

"""相手の話を聞いている間に、返答の方針だけ先に立てておく（2026-08-30）。

## なぜ要るか

現行は「話し終えてから、考えて→話す」を1回でやっている。**出力の 75% は考える部分**で、
そこが喋り始めを遅らせていた。人間も相手の話の要点を掴んだ時点で応答の計画を始めており
（EEG 研究）、それに寄せる。

    現行  : 話し終える → [考える239字 ＋ 話す60字] → 発話
    これから: 話している間 → [考える]          ← 発話中に隠れる
             話し終える  → [話す]             ← 短い

実測（方向転換11ケース）: 喋り始め 6.53s → 5.03s、方向転換への追随 9/9、記憶の具体性 3/3。

## どこで動くか

**`TurnHandle` には持たせない。** あれは「ミクが応答生成中」にだけ存在する器で、
段階1が走る「相手がまだ話している最中」には**まだ生成されていない**。
`context_cache` と同じく、このモジュール内で状態を持ちロックで守る。
LLM 呼び出しは `agent_main` と同じく**永続イベントループ**へ投げる（下の `_run_async`）。

## 費用

プロンプトの固定部（約 52,000 tok）は**段階2・事前作成と同じキャッシュを共有する**。
段階1は1ターンに最大3回走るので、フルで送ると段階2より重くなる（$0.236/ターン →
$0.024/ターン）。共有できるのは、`build_plan_prompt` が会話ターンと同じ手順・同じ境界で
プロンプトを組むため。**ここがずれると互いに作り直し合う**ので、
`[context_cache] stale` がログに出ていないか実機で見ること。

    ユーザーが話し始めた   → begin_utterance() でリセット
    interim が届くたび     → maybe_start() が起動を判断（走行中ならスキップ）
    話し終えた             → take() で方針を取り出す（無ければ None）

## 起動の間隔

interim は 2 秒ごとに来るが、**走行中はスキップ**するので、実際の間隔は段階1の
所要時間（3.4〜4.5 秒）で決まる。上限 3 回なので、

    2秒 起動 → 4秒 スキップ → 6秒 起動 → 8秒 スキップ → 10秒 起動 → 以降スキップ

発話の中央値は 7.1 秒なので、多くのターンで 1〜2 回に収まる。

## 壊れないこと

方針は**あれば使う**もので、無くても会話は成立する（現行と同じ1段階に落ちるだけ）。
起動の失敗・間に合わなかった場合・別ターンの結果は、すべて黙って捨てる。
`_ENABLED = False` で完全に従来動作へ戻る。
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput

from agent import context_cache
from agent.agent_logger import record_llm_usage
from agent.config import configure_env_from_settings, get_pydantic_ai_flash_model, load_settings

_logger = logging.getLogger(__name__)

#: 段階1を使うか。False で従来動作（1段階のみ）に戻る
_ENABLED = True

#: 1回の発話で段階1を起動する上限。長い発話でも際限なく走らせない。
#: 実測では 19字（2.8秒）時点で既に正しい方向を掴んでいたので、多くても3回で足りる
_MAX_STARTS = 3

#: これ未満の入力では起動しない（「あ、」だけでは方針が立たない）
_MIN_CHARS = 8

#: 立ててからこれだけ経った方針は使わない（秒）。
#: **取りこぼしの掃除**——発話したのに送信されなかった（Smart Turn が続きを待った等）とき、
#: 古い方針が次のターンへ紛れ込むのを防ぐ。話し終えてから段階2までは数秒しかかからない
_PLAN_TTL_SEC = 30.0

#: 思考の深さ。**指定しないと 13〜16 秒かかり、発話中に間に合わない**（2026-08-30 実機）。
#: 段階1を決めた実験も LOW で測っている（追随 9/9・所要 3.4〜4.5 秒）。
#: 段階2のループ1も同じ値（`runner._LOOP1_THINKING_LEVEL`）。切り戻すときは None にする
_THINKING_LEVEL: str | None = "LOW"

_settings = load_settings()
configure_env_from_settings(_settings)

#: 明示キャッシュを作るときのモデル名（`google-gla:` などの接頭辞を外す）
_MODEL_ID = get_pydantic_ai_flash_model(_settings).split(":", 1)[-1]


class PlanOutput(BaseModel):
    """相手の話を聞いている途中で立てる方針。**喋る文はここでは作らない。**

    足すものを検討して**却下した**もの（2026-08-30 実測）:
      - response の仮案 … 段階2が 1.82 秒**遅くなる**（引きずられはしないが割に合わない）
      - 次回予定       … 段階1が **2倍遅くなり**、発話中に間に合わなくなる
      - 記録系（mood_updates / interest_updates / acknowledged_task_id）
        … **原理的に不可**。「実際にそうした」ことの記録なので、話す前には決められない
    """

    user_state_inference: str = Field(description="ユーザーの現在の状態についてミクが推定した内容")
    purpose: str = Field(description="今回の発話の社会的会話目的または内的行動目的")
    inner_thoughts: str = Field(
        description="内心の思考・発話の方針。**記憶やメモから使う内容もここに書き出す**")
    requires_tool: bool = Field(description="ツールが必要かを true/false で回答")
    tool_name: str | None = Field(default=None, description="使用するツール名。使わないなら null")


_agent: Agent[None, PlanOutput] = Agent(
    model=get_pydantic_ai_flash_model(_settings),
    output_type=NativeOutput(PlanOutput),
)


# =============================================================================
# 永続イベントループ（このモジュール専用に 1 本）
# =============================================================================
#
# pydantic_ai は httpx AsyncClient を Agent オブジェクトにキャッシュし、**最初に
# run() を呼んだ時の event loop に紐付ける**。段階1はスレッドで走らせるので、
# `asyncio.run()`（使い捨てループ）だと 2 回目以降が
# 「Event loop is closed」で即死する（2026-08-30 実機で検出。
# 同じ罠を 2026-08-05 にフィラーでも踏んでいる）。
#
# 対策は `agent_main` と同じで、デーモンスレッドで 1 本のループを回し続け、
# 全ての起動をそこへ投げる。接続を使い回せるので**所要時間も縮む**
# （初回は TLS 確立を含んで 19.9 秒かかっていた）。

_loop_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None

_T = TypeVar("_T")


def _get_or_create_loop() -> asyncio.AbstractEventLoop:
    """このモジュール専用に 1 本だけ稼働する event loop を返す（必要なら生成する）。"""
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            _loop_thread = threading.Thread(
                target=_loop.run_forever,
                name="plan_stage_loop",
                daemon=True,
            )
            _loop_thread.start()
    return _loop


def _run_async(coro: Awaitable[_T]) -> _T:
    """永続 event loop に coroutine を投げて結果を同期取得する。"""
    fut = asyncio.run_coroutine_threadsafe(coro, _get_or_create_loop())
    return fut.result()


def _model_settings(cached_name: str | None):
    """思考の深さと明示キャッシュ参照をまとめた model_settings（不要なら None）。"""
    if _THINKING_LEVEL is None and cached_name is None:
        return None
    try:
        from pydantic_ai.models.google import GoogleModelSettings
    except ImportError:
        return None      # Google 以外のモデル構成でも動くようにする
    kw: dict = {}
    if _THINKING_LEVEL is not None:
        kw["google_thinking_config"] = {"thinking_level": _THINKING_LEVEL}
    if cached_name:
        kw["google_cached_content"] = cached_name
    return GoogleModelSettings(**kw)


@dataclass
class _Plan:
    """立てた方針1件。"""

    output: PlanOutput
    heard: str                  # **どこまで聞いた時点か**（段階2がこれと確定発話を読み比べる）
    at: float = field(default_factory=time.time)


_lock = threading.Lock()
_running = False                # 走行中フラグ（多重起動を防ぐ）
_starts = 0                     # この発話で起動した回数
_latest: _Plan | None = None    # 最後に立った方針
_utterance_id = 0               # 発話ごとに増やす。古い結果を捨てるために使う


def is_enabled() -> bool:
    return _ENABLED


def begin_utterance() -> None:
    """相手が話し始めた。前の発話の残りを捨てて数え直す。"""
    global _starts, _latest, _utterance_id
    if not _ENABLED:
        return
    with _lock:
        _utterance_id += 1
        _starts = 0
        _latest = None


def reset_starts() -> None:
    """発話が確定した。次の発話に備えて起動回数だけ戻す。

    **方針そのものは消さない**（これから段階2が使う）。
    取りこぼしたぶんは `take()` の TTL で自然に落ちる。
    """
    global _starts
    if not _ENABLED:
        return
    with _lock:
        _starts = 0


def take() -> tuple[PlanOutput, str] | None:
    """立っている方針を取り出す（1回だけ）。無ければ None。

    **走行中でも待たない。** 待つと短縮のぶんが消えるので、
    間に合った方針だけを使い、間に合わなければ従来どおり1段階で作る。

    古すぎる方針（`_PLAN_TTL_SEC` 超）も捨てる —— 発話したのに送信されなかったときに、
    前のターンの方針が紛れ込まないようにするため。
    """
    global _latest
    if not _ENABLED:
        return None
    with _lock:
        p = _latest
        _latest = None
        if p is None:
            return None
        if time.time() - p.at > _PLAN_TTL_SEC:
            _logger.info("[plan_stage] 古い方針のため破棄（%.0f秒前）", time.time() - p.at)
            return None
        return p.output, p.heard


def maybe_start(partial_text: str, build_prompt) -> bool:
    """interim を受けて段階1を起動する。走行中・上限到達なら何もしない。

    Args:
        partial_text: ここまでに聞き取れた発話（部分入力）
        build_prompt: `(partial_text) -> (instructions, 前半, 後半)` を返す関数。
            プロンプトの組み立ては呼び出し元（gateway）に任せる —— このモジュールが
            context の作り方を知る必要はない。前半は明示キャッシュに載せる固定部で、
            **段階2と同じ文字列**であること（違うと互いに作り直し合う）

    Returns:
        起動したら True
    """
    global _running, _starts
    if not _ENABLED or len(partial_text.strip()) < _MIN_CHARS:
        return False

    with _lock:
        if _running or _starts >= _MAX_STARTS:
            return False
        _running = True
        _starts += 1
        my_utterance = _utterance_id
        n = _starts

    def _work() -> None:
        global _running, _latest
        try:
            instructions, cache_head, cache_tail = build_prompt(partial_text)
            # 固定部は段階2・事前作成と同じ文字列なので、**1つのキャッシュを共有する**。
            # 段階1は1ターンに最大3回走るため、フルで送ると段階2より費用が重くなる。
            #
            # **ここでは作らない**（`ensure` ではなく `get_if_matches`）。作りに行くと、
            # 記憶の取得が一時的に失敗しただけで段階2のキャッシュを壊して作り直す。
            # 用意するのは接続時の事前作成と段階2の役目で、無ければ全文を送るだけ
            cached_name = context_cache.get_if_matches(instructions, cache_head) if cache_head else None
            if cached_name:
                user_prompt = cache_tail
                instructions = ""          # キャッシュ側に入っている
            else:
                user_prompt = cache_head + cache_tail
            t0 = time.perf_counter()
            result = _run_async(_agent.run(user_prompt, instructions=instructions,
                                           model_settings=_model_settings(cached_name)))
            sec = time.perf_counter() - t0
            # 段階1も課金される（1ターン最大3回）。会話の記録が開く前に走るので、記録は
            # `LLM使用量_<日付>.log` へ入る（agent_logger.log_llm_usage）
            record_llm_usage(_agent, result, kind="plan_stage", label=f"plan_stage usage（{n}/{_MAX_STARTS}）")
            with _lock:
                # 発話が切り替わっていたら捨てる（古い方針を次のターンへ持ち越さない）
                if my_utterance != _utterance_id:
                    _logger.info("[plan_stage] 発話が切り替わったため破棄")
                    return
                _latest = _Plan(output=result.output, heard=partial_text)
            print(f"[plan_stage] 方針 {n}/{_MAX_STARTS} 完成 {sec:.2f}s "
                  f"（{len(partial_text)}字まで聞いた時点・"
                  f"キャッシュ{'あり' if cached_name else 'なし'}）: {result.output.purpose[:44]}")
        except Exception as exc:
            # 方針が無くても会話は成立する。握り潰さずログには残す
            _logger.warning("[plan_stage] 方針づくりに失敗: %r", exc)
            print(f"[plan_stage] 方針づくりに失敗（従来どおり1段階で作る）: {exc!r}")
        finally:
            with _lock:
                _running = False

    threading.Thread(target=_work, daemon=True, name="Plan-Stage").start()
    return True


def build_plan_block(plan: PlanOutput, heard: str) -> str:
    """段階2へ渡す方針ブロックを組み立てる。

    **「どこまで聞いた時点か」を入れるのが要点**（2026-08-30 実測）。
    判定フィールドを持たせて作り直しを指示するより、この一文と
    「当てはまらないなら気にせず考え直すこと」だけで方向転換への追随が 9/9 になった。
    """
    return (
        "\n\n# 参考: 相手の話を聞いている途中に、あなたが立てた方針\n"
        f"※ これは相手の話を「{heard}」まで聞いた時点で立てたもの。\n"
        "確定した発話に当てはまるなら沿ってよい。**当てはまらないなら気にせず考え直すこと**。\n"
        f"- 相手の状態: {plan.user_state_inference}\n"
        f"- 返答の狙い: {plan.purpose}\n"
        f"- 考えたこと: {plan.inner_thoughts}\n"
        f"- ツールの見込み: {plan.tool_name or '使わない'}\n"
    )


def status() -> dict:
    """いまの状態（デバッグ用）。"""
    with _lock:
        return {"enabled": _ENABLED, "running": _running, "starts": _starts,
                "has_plan": _latest is not None,
                "heard_chars": len(_latest.heard) if _latest else 0}
