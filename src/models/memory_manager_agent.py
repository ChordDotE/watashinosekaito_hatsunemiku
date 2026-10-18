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

"""Agent Service を使った記憶管理モジュール。

memory_manager_dify.py の Agent Service 版。
process_all_conversations() のシグネチャを完全に維持するため、
gateway_service.py 側はインポート先を変えるだけで切り替えられる。
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

from datetime import datetime

from utils.path_config import PathConfig
from models.agent_adapter import create_agent_adapter_from_settings


def save_conversation_backup(document_title: str, document_body: str, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in document_title)[:50]
    out = backup_dir / f"{ts}_{safe_title}.txt"
    out.write_text(f"{document_title}\n\n{document_body}", encoding="utf-8")
    print(f"会話バックアップ保存: {out}")


def save_long_memory_backup(long_memory: str, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = backup_dir / f"{ts}_long_memory.txt"
    out.write_text(long_memory, encoding="utf-8")
    print(f"長期記憶バックアップ保存: {out}")


def process_all_conversations(conversation_backup_dir: Path, long_memory_backup_dir: Path) -> None:
    """Agent Service 経由で記憶を更新する。

    ナレッジ登録 → メモリ更新 の順に実行する（memory_manager_dify.py と同順）。

    Args:
        conversation_backup_dir: 会話バックアップの保存先ディレクトリ
        long_memory_backup_dir:  長期記憶バックアップの保存先ディレクトリ
    """
    print("Agent Service を使った記憶更新を開始します...")

    try:
        path_config = PathConfig.get_instance()
        with open(str(path_config.settings_file), "r", encoding="utf-8-sig") as f:   # BOM 付きでも読む
            settings = json.load(f)

        adapter = create_agent_adapter_from_settings(settings)

        # ── ナレッジ登録 ──────────────────────────────────────────
        print("ナレッジ登録ワークフローを呼び出し中...")
        kr = adapter.call_knowledge_workflow(is_transfer=False)
        print(f"ナレッジ登録ワークフロー完了: status={kr.get('status')}")

        if kr.get("status") == "ok":
            document_title = kr.get("document_title", "")
            document_body = kr.get("document_body", "")
            if document_title and document_body:
                save_conversation_backup(document_title, document_body, conversation_backup_dir)
            else:
                print("バックアップ対象のドキュメントがありません")
        else:
            print(f"ナレッジ登録スキップ: {kr.get('reason')}")

        # ── 記憶更新 ──────────────────────────────────────────────
        print("メモリ更新ワークフローを呼び出し中...")
        mu = adapter.call_memory_workflow(is_transfer=False)
        print(f"メモリ更新ワークフロー完了: status={mu.get('status')}")

        if mu.get("status") == "ok":
            long_memory = mu.get("long_memory", "")
            compression_rate = mu.get("compression_rate", "")
            if long_memory:
                save_long_memory_backup(long_memory, long_memory_backup_dir)
            else:
                print("バックアップ対象の長期記憶がありません")
            print(f"圧縮率: {compression_rate}")
        else:
            print(f"メモリ更新スキップ: {mu.get('reason')}")

        print("全てのワークフロー処理が完了しました。")

    except Exception as e:
        print(f"ワークフロー処理中にエラーが発生しました: {e}")
        traceback.print_exc()
