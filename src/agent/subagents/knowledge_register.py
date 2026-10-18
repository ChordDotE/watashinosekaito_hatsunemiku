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

"""会話テキストの要約を生成する Subagent。

Dify の「会話ナレッジ登録文」WF における以下のノードを Python で再現する:
  - 移行モード判定（is_transfer）
  - 前回会話取得 / 前回会話クリア（KV サービス）
  - LLM: 要約作成 → abstract
  - 会話文構造取得（document_title / document_body 生成）

詳細:
  - .claude/skills/dev-rules/details/agents.md
  - .claude/skills/dev-rules/details/wf_knowledge_register.md
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import requests
from pydantic_ai import Agent

_logger = logging.getLogger(__name__)

from agent.agent_logger import record_llm_usage, log, log_subagent_request, log_subagent_response, mark_prompt
from agent.config import (
    configure_env_from_settings,
    get_kv_base_url,
    get_pydantic_ai_flash_model,
    get_vdb_base_url,
    load_settings,
)
from agent.schemas import KnowledgeRegisterInput, KnowledgeRegisterOutput, KnowledgeRegisterResult
from utils import conversation_log

_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

_settings = load_settings()
configure_env_from_settings(_settings)
_KV_BASE_URL = get_kv_base_url(_settings)
_VDB_BASE_URL = get_vdb_base_url(_settings)
_SYSTEM_PROMPT = (_PROMPTS_DIR / "knowledge_register" / "system.md").read_text(encoding="utf-8")

knowledge_register_agent: Agent[None, KnowledgeRegisterOutput] = Agent(
    model=get_pydantic_ai_flash_model(_settings),
    output_type=KnowledgeRegisterOutput,
    system_prompt=_SYSTEM_PROMPT,
)

_TS_RE = re.compile(r'(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)')


def build_document_title(body: str) -> str:
    """body から最初の日時を検出して session_YYYYMMDD_HHMMSS 形式のタイトルを返す。

    日時が見つからない場合は実行時刻でタイトルを生成する。
    Dify の「会話文構造取得」コードノード相当。
    """
    m = _TS_RE.search(body)
    raw = m.group(1).replace("T", " ").split(".")[0] if m else None
    dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S") if raw else datetime.now()
    return f"session_{dt.strftime('%Y%m%d_%H%M%S')}"


def build_document_body(abstract: str, body: str) -> str:
    """abstract と生本文を結合して document_body を構築する。

    Dify の「会話文構造取得」コードノード相当。
    """
    return f"### 会話の要約\n{abstract}\n### 会話の本文\n{body}"


def _kv_get(base_url: str, key: str) -> str | None:
    """KV サービスから値を取得する。キーが存在しない場合は None。"""
    r = requests.get(f"{base_url}/kv", params={"key": key}, timeout=10)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.text


def _kv_set(base_url: str, key: str, value: str) -> None:
    """KV サービスにキーと値をセットする。"""
    r = requests.post(f"{base_url}/kv", json={"key": key, "value": value}, timeout=10)
    r.raise_for_status()


def _kv_clear(base_url: str, key: str) -> None:
    """KV サービスのキーを空文字で上書きする（Dify の「前回会話クリア」相当）。"""
    _kv_set(base_url, key, "")


async def run_knowledge_register(
    input_: KnowledgeRegisterInput,
) -> KnowledgeRegisterResult | None:
    """Orchestrator / ingest スクリプトから呼ぶ公開関数。

    is_transfer=False（通常モード）:
      KV サービスの this_conversation を取得し、登録後にクリアする。
      テキストが空の場合は None を返す。

    is_transfer=True（移行・直接渡しモード）:
      input_.conversation_text をそのまま使う。

    戻り値 KnowledgeRegisterResult は abstract / document_title / document_body を含み、
    Dify の call_knowledge_workflow() レスポンスと同等の情報を持つ。
    """
    if input_.is_transfer:
        # 移行モード: テキストを直接受け取る（Dify の sys.query 相当）
        text = input_.conversation_text
    else:
        # 通常モード: KV から this_conversation を取得
        text = _kv_get(_KV_BASE_URL, "this_conversation")
        if not text or not text.strip():
            return None  # Dify の「空白文エラーチェック → エラー終了」相当

        # 既存の登録待機会話と新規会話をマージして registration_waiting_conversation に書き込む
        # （Dify の「登録待機会話マージ」コードノード相当）
        existing_waiting = _kv_get(_KV_BASE_URL, "registration_waiting_conversation") or ""
        merged = existing_waiting + "\n" + text if existing_waiting else text
        _kv_set(_KV_BASE_URL, "registration_waiting_conversation", merged)

        # 取得直後にクリア（Dify の「前回会話クリア」相当）
        _kv_clear(_KV_BASE_URL, "this_conversation")

    # 要約役は会話の場に居合わせていないため、レコードの読み方（〔 〕・camera_N.jpg 等）を先に渡す
    prompt = f"{conversation_log.LOG_READING_GUIDE}\n[会話テキスト]\n{text}"
    log_subagent_request("knowledge_register", mark_prompt("knowledge_register.system", _SYSTEM_PROMPT), prompt)
    llm_result = await knowledge_register_agent.run(prompt)
    record_llm_usage(knowledge_register_agent, llm_result, kind="knowledge_register")
    abstract = llm_result.output.abstract
    log_subagent_response("knowledge_register", abstract)
    document_title = build_document_title(text)
    document_body = build_document_body(abstract, text)
    log(
        "knowledge_register_built",
        "knowledge_register 文書整形完了（abstract + 元会話本文 → document_title / document_body）",
        output={
            "document_title": document_title,
            "document_body": document_body,
        },
    )

    # VDB サービスに upsert（HTTP）— マイクロサービス間通信は直 import 禁止
    m = _TS_RE.search(text)
    registered_at = m.group(1) if m else datetime.now().isoformat()
    resp = requests.post(
        f"{_VDB_BASE_URL}/vdb/upsert",
        json={
            "document_title": document_title,
            "document_body": document_body,
            "source": document_title,
            "registered_at": registered_at,
        },
        timeout=60,
    )
    if not resp.ok:
        _logger.error(
            "VDB upsert 失敗: status=%s reason=%s",
            resp.status_code,
            resp.text[:500],
        )
    resp.raise_for_status()

    return KnowledgeRegisterResult(
        abstract=abstract,
        document_title=document_title,
        document_body=document_body,
    )
