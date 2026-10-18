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

"""長期記憶即時パッチツール。

ミク自身の判断で long_memory をセッション終了を待たずに即時更新する。
人間が覚醒中に重要な会話の内容から長期記憶を更新するのと同じ動作を模倣する。

runner.py から asyncio.create_task() でバックグラウンド実行されるため、
LLM（ミク）はツール呼び出し後に結果を待たずに次のループへ進める。

入力として渡すもの:
  - conversation_history: これまでの会話履歴（通常更新と同じデータ）
  - inner_thoughts_hint: ミクが「ここを更新してほしい」と判断した理由・内容

is_transfer=False で run_memory_update を呼ぶ（日次の記憶更新とまったく同じ経路）。
**圧縮率による却下はしない** —— `compression_rate` は更新前後のサイズ比を観測してログに残すだけで、
値がいくつでも結果は採用される（2026-08-11 にコード照合で確認。旧 Dify 由来の
「0.70〜1.00 のチェック」という記述が実装と食い違っていたため訂正した）。

フロー:
  1. パッチ用テキスト（ヒント + 会話履歴）を KV の registration_waiting_conversation に書き込む
     - 既存の待機中会話がある場合はパッチテキストの後ろに連結する
  2. run_memory_update(is_transfer=False) を呼ぶ
     - KV から registration_waiting_conversation を読み込み、通常フローで更新

権限レベル: 2 (memory)
副作用: KV サービスの registration_waiting_conversation を上書きした後 long_memory を更新する
"""
from __future__ import annotations

import logging

import requests

from agent.config import get_kv_base_url, load_settings
from agent.schemas import MemoryUpdateInput, MemoryUpdateResult
from agent.subagents.memory_update_db import run_memory_update

_logger = logging.getLogger(__name__)

_settings = load_settings()
_KV_BASE_URL = get_kv_base_url(_settings)

_KV_WAITING = "registration_waiting_conversation"


def _kv_get(key: str) -> str:
    r = requests.get(f"{_KV_BASE_URL}/kv", params={"key": key}, timeout=10)
    if r.status_code == 404:
        return ""
    r.raise_for_status()
    return r.text


def _kv_set(key: str, value: str) -> None:
    r = requests.post(f"{_KV_BASE_URL}/kv", json={"key": key, "value": value}, timeout=10)
    r.raise_for_status()


def _build_patch_conversation(conversation_history: str, inner_thoughts_hint: str) -> str:
    """会話履歴と inner_thoughts ヒントから memory_update 用テキストを組み立てる。

    ヒントを冒頭に置くことで、memory_update_agent が「何を重点的に更新すべきか」を
    会話を読む前に把握できるようにする。
    """
    parts: list[str] = []
    if inner_thoughts_hint and inner_thoughts_hint.strip():
        parts.append(
            "（ミクが会話の中で気づき、即座に長期記憶を更新したいと判断した内容）\n"
            + inner_thoughts_hint.strip()
        )
    if conversation_history and conversation_history.strip():
        parts.append(conversation_history.strip())
    return "\n\n".join(parts)


async def run_long_memory_patch(
    conversation_history: str,
    inner_thoughts_hint: str = "",
) -> MemoryUpdateResult | None:
    """会話履歴と inner_thoughts ヒントを使って long_memory を即時更新する。

    パッチ用テキストを registration_waiting_conversation に書き込んでから
    is_transfer=False で run_memory_update を起動する（日次と同じ経路・圧縮率による却下はしない）。

    スキップ・失敗になるのは「入力が空」「待機中の会話が空」「例外」のときだけで、
    いずれも呼び出し元は結果を待たない（create_task）ため**ログにしか残らない**。

    Args:
        conversation_history: これまでの会話全文（raw_input["conversation_history"] をそのまま渡す）。
        inner_thoughts_hint: ミクが「ここを更新すべき」と判断した理由・内容
                             （ai_judgements から取り出した inner_thoughts）。

    Returns:
        MemoryUpdateResult（成功時）または None（スキップ・失敗時）
    """
    if not conversation_history and not inner_thoughts_hint:
        _logger.debug("[long_memory_patch] 入力が空のためスキップ")
        return None

    patch_text = _build_patch_conversation(conversation_history, inner_thoughts_hint)

    # 既存の待機中会話がある場合はパッチテキストの後ろに連結する
    # （パッチのヒントを先頭に置き、既存の会話ログが後続する形にする）
    existing = _kv_get(_KV_WAITING)
    combined = patch_text + "\n\n" + existing if existing.strip() else patch_text
    _kv_set(_KV_WAITING, combined)

    _logger.info(
        "[long_memory_patch] 即時パッチ開始 conv_chars=%d hint_chars=%d waiting_chars=%d",
        len(conversation_history),
        len(inner_thoughts_hint),
        len(existing),
    )
    print(
        f"[long_memory_patch] 開始"
        f" conv_chars={len(conversation_history)}"
        f" hint_chars={len(inner_thoughts_hint)}"
        f" existing_waiting_chars={len(existing)}"
    )

    result = await run_memory_update(MemoryUpdateInput(is_transfer=False))

    if result is None or result.skip_reason:
        reason = result.skip_reason if result else "None returned"
        _logger.warning("[long_memory_patch] パッチ失敗: %s", reason)
        print(f"[long_memory_patch] 失敗: {reason}")
    else:
        _logger.info(
            "[long_memory_patch] パッチ完了 compression_rate=%.3f",
            result.compression_rate,
        )
        print(f"[long_memory_patch] 完了 compression_rate={result.compression_rate:.3f}")

    return result
