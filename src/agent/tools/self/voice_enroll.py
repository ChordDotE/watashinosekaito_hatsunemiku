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

"""マスター声登録ツール。

添付音声（local_path）を base64 エンコードして Recognition サービス（:5007/voice/enroll）へ
POST し、登録結果を返す。

音声の長さはミク自身には数えられないため、**短すぎる場合はサービス側が検証して
理由つきで拒否する**（顔で「複数の顔が検出されました（3人）」を返しているのと同じ設計）。
これによりミクは正確な秒数を知ったうえで「もう少し長い音声がほしい」と言える。

権限レベル: 2 (self。ローカルファイル読み取り + Recognition サービスへの書き込みを伴う)
副作用: master_voice_embedding.pkl / master_voice_sample.wav を更新（Recognition サービス内）
"""
from __future__ import annotations

import base64

import requests
from pydantic import BaseModel, Field

from agent.config import get_recognition_base_url, load_settings
from agent.schemas import Failure, Success

_settings = load_settings()
_RECOGNITION_BASE_URL = get_recognition_base_url(_settings)


class VoiceEnrollInput(BaseModel):
    audio_path: str = Field(description="登録に使う音声のローカルパス（attached_audio の要素）")


class VoiceEnrollValue(BaseModel):
    enrolled: bool = Field(description="登録に成功したか")
    seconds: float = Field(description="登録に使った音声の長さ（秒）")
    short_warning: bool = Field(
        default=False,
        description="登録はできたが推奨より短い。次に機会があればもっと長い音声で覚え直すとよい",
    )


VoiceEnrollResult = Success[VoiceEnrollValue] | Failure


def run_voice_enroll(input_: VoiceEnrollInput) -> VoiceEnrollResult:
    """音声を読み込み、Recognition サービスに声の登録リクエストを送る。"""
    try:
        with open(input_.audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")
    except OSError as e:
        return Failure(reason=f"audio_read_error: {e}", retryable=False)

    try:
        resp = requests.post(
            f"{_RECOGNITION_BASE_URL}/voice/enroll",
            json={"audio_b64": audio_b64},
            timeout=60,
        )
    except requests.exceptions.ConnectionError:
        return Failure(reason="recognition_service_unreachable", retryable=True)
    except Exception as e:
        return Failure(reason=f"http_error: {e!r}", retryable=False)

    if resp.status_code == 400:
        body = resp.json()
        # 短すぎ・読めない形式は理由をそのまま返す（ミクが言葉にできる形になっている）
        reason = body.get("reason") or body.get("error", "")
        return Failure(reason=reason, retryable=True)

    if resp.status_code != 200:
        return Failure(reason=f"enroll HTTP {resp.status_code}: {resp.text[:200]}", retryable=True)

    body = resp.json()
    seconds = float(body.get("seconds", 0.0))
    short = bool(body.get("short_warning", False))
    summary = f"マスターの声を登録しました（{seconds:.1f}秒）。"
    if short:
        summary += "少し短めなので、機会があればもっと長い音声で覚え直すとより正確になります。"
    return Success[VoiceEnrollValue](
        value=VoiceEnrollValue(
            enrolled=body.get("status") == "enrolled",
            seconds=seconds,
            short_warning=short,
        ),
        summary=summary,
    )
