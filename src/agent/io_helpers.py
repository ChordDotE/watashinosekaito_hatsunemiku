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

"""LLM 入出力に関する共通ヘルパ。

複数の subagent / Orchestrator が共通で使う変換ロジックを集約する。

- 添付画像のローカルパス → pydantic_ai BinaryContent への変換
- 添付ファイル名のリスト整形

Dify の vision 設定（statements: 統合応答 / 高度思考 / ミクメモ更新 / メール内容考案）に
対応するため、これらの subagent から呼び出される。
"""
from __future__ import annotations

from pathlib import Path

from pydantic_ai import BinaryContent

# ファイル拡張子 → media_type マッピング
_IMAGE_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# 動画・音声の拡張子 → media_type マッピング（Gemini 対応形式）
_AV_MEDIA_TYPES: dict[str, str] = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".mp3": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".amr": "audio/amr",
}


def to_binary_contents(image_paths: list[str]) -> list[BinaryContent]:
    """画像ファイルパスのリストを pydantic_ai BinaryContent のリストに変換する。

    media_type は拡張子から自動判定。判定不能な拡張子は image/jpeg として扱う。
    存在しないパスは無視せず例外を伝播させる（呼び出し側で Failure 化する想定）。
    """
    contents: list[BinaryContent] = []
    for p in image_paths:
        path = Path(p)
        media_type = _IMAGE_MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")
        contents.append(BinaryContent(data=path.read_bytes(), media_type=media_type))
    return contents


def to_media_binary_contents(media_paths: list[str]) -> list[BinaryContent]:
    """動画・音声ファイルパスのリストを pydantic_ai BinaryContent のリストに変換する。

    media_type は拡張子から自動判定。判定不能な拡張子は video/mp4 として扱う。
    サイズ制限（inline 合計 約20MB）は呼び出し前に agent_main 側で適用済みの前提。
    """
    contents: list[BinaryContent] = []
    for p in media_paths:
        path = Path(p)
        media_type = _AV_MEDIA_TYPES.get(path.suffix.lower(), "video/mp4")
        contents.append(BinaryContent(data=path.read_bytes(), media_type=media_type))
    return contents


def derive_file_names(image_paths: list[str]) -> str:
    """画像ファイルパスのリストからファイル名のみを抽出して改行区切り文字列にする。

    Dify 統合応答ノード user 2 の `# 今回添付されたファイルの名称` ブロックに
    渡す文字列を組み立てる用途。
    """
    return "\n".join(Path(p).name for p in image_paths)
