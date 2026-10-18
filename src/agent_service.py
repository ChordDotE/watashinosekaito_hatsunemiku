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

"""Agent Service — Pydantic AI WF を HTTP 経由で公開するマイクロサービス。

ポート: settings.json の agent_service.port（デフォルト 5010）

エンドポイント:
  POST /knowledge  — knowledge_register WF
  POST /memory     — memory_update WF
  POST /chat       — main_chat WF（未移植: 501）
  GET  /ping       — 疎通確認

起動方法:
  python agent_service.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from flask import Flask, jsonify
from flask import request as flask_request

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent.agent_logger import (
    WF_KNOWLEDGE_REGISTER,
    WF_MEMORY_UPDATE,
    log as agent_log,
    open_log,
)
from agent.config import load_settings
from agent.schemas import KnowledgeRegisterInput, MemoryUpdateInput
from agent_main import _run_async  # 永続 event loop ヘルパー
import agent.subagents.knowledge_register as kr_module
import agent.subagents.memory_update_db as mu_module

_settings = load_settings()
_agent_cfg = _settings.get("services", {}).get("agent", {})
_DEFAULT_PORT = _agent_cfg.get("port", 5010)
# 待ち受けはこの PC の中だけ（認証が無いので設定で開けない・2026-10-08）
_DEFAULT_HOST = "127.0.0.1"

app = Flask("agent_service")


@app.post("/knowledge")
def knowledge():
    """ナレッジ登録 WF (knowledge_register)。

    Request:  {"is_transfer": false}
    Response: {"status": "ok",   "document_title": "...", "document_body": "...", "abstract": "..."}
           or {"status": "skip", "reason": "..."}
    """
    data = flask_request.get_json(force=True, silent=True) or {}
    is_transfer = bool(data.get("is_transfer", False))

    with open_log(WF_KNOWLEDGE_REGISTER):
        agent_log(
            "request",
            "POST /knowledge の受領入力（gateway/scheduler → agent_service）",
            input={"is_transfer": is_transfer},
        )
        result = _run_async(
            kr_module.run_knowledge_register(KnowledgeRegisterInput(is_transfer=is_transfer))
        )

        if result is None:
            agent_log(
                "response",
                "POST /knowledge の戻り値（skip：this_conversation が空）",
                output={"status": "skip", "reason": "this_conversation が空"},
            )
            return jsonify({"status": "skip", "reason": "this_conversation が空"})

        response = {
            "status": "ok",
            "document_title": result.document_title,
            "document_body": result.document_body,
            "abstract": result.abstract,
        }
        agent_log(
            "response",
            "POST /knowledge の戻り値（agent_service → 呼出元）",
            output=response,
        )
        return jsonify(response)


@app.post("/memory")
def memory():
    """記憶更新 WF (memory_update)。

    Request:  {"is_transfer": false}
    Response: {"status": "ok",   "compression_rate": 1.047, "long_memory": "{...}"}
           or {"status": "skip", "reason": "..."}
    """
    data = flask_request.get_json(force=True, silent=True) or {}
    is_transfer = bool(data.get("is_transfer", False))

    with open_log(WF_MEMORY_UPDATE):
        agent_log(
            "request",
            "POST /memory の受領入力（gateway/scheduler → agent_service）",
            input={"is_transfer": is_transfer},
        )
        result = _run_async(
            mu_module.run_memory_update(MemoryUpdateInput(is_transfer=is_transfer))
        )

        if result is None or result.skip_reason:
            reason = result.skip_reason if result else "unknown"
            agent_log(
                "response",
                f"POST /memory の戻り値（skip：{reason}）",
                output={"status": "skip", "reason": reason},
            )
            return jsonify({"status": "skip", "reason": reason})

        response = {
            "status": "ok",
            "compression_rate": result.compression_rate,
            "long_memory": result.updated_memory_json,
        }
        agent_log(
            "response",
            "POST /memory の戻り値（agent_service → 呼出元）",
            output=response,
        )
        return jsonify(response)


@app.post("/chat")
def chat():
    """main_chat WF（未移植）。"""
    return jsonify({"error": "main_chat WF は未移植です"}), 501


@app.get("/ping")
def ping():
    return jsonify({"status": "ok"})


def run_agent_service(port: int = _DEFAULT_PORT, host: str = _DEFAULT_HOST) -> None:
    app.run(host=host, port=port, threaded=True)


def main() -> None:
    """単体起動の入口。配布物では `.pyd` 化されるため、起動処理は関数に置く。"""
    run_agent_service()


if __name__ == "__main__":
    main()
