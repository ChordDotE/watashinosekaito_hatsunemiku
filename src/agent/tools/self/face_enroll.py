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

"""マスター顔登録ツール。

添付画像（local_path）を base64 エンコードして Recognition サービス（:5007/face/enroll）へ
POST し、登録結果を返す。

権限レベル: 2 (self。ローカルファイル読み取り + FaceRecognition サービスへの書き込みを伴う)
副作用: master_face_encoding.pkl を更新（FaceRecognition サービス内）
"""
from __future__ import annotations

import base64

import requests
from pydantic import BaseModel, Field

from agent.config import get_recognition_base_url, load_settings
from agent.schemas import Failure, Success

_settings = load_settings()
_RECOGNITION_BASE_URL = get_recognition_base_url(_settings)


class FaceEnrollInput(BaseModel):
    image_path: str = Field(description="登録に使う画像のローカルパス（attached_images の要素）")


class FaceEnrollValue(BaseModel):
    enrolled: bool = Field(description="登録に成功したか")
    faces_detected: int = Field(description="検出された顔の数")


FaceEnrollResult = Success[FaceEnrollValue] | Failure


def run_face_enroll(input_: FaceEnrollInput) -> FaceEnrollResult:
    """画像を読み込み、FaceRecognition サービスに顔登録リクエストを送る。"""
    try:
        with open(input_.image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")
    except OSError as e:
        return Failure(reason=f"image_read_error: {e}", retryable=False)

    try:
        resp = requests.post(
            f"{_RECOGNITION_BASE_URL}/face/enroll",
            json={"image_b64": image_b64},
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        return Failure(reason="recognition_service_unreachable", retryable=True)
    except Exception as e:
        return Failure(reason=f"http_error: {e!r}", retryable=False)

    if resp.status_code == 400:
        err = resp.json().get("error", "")
        if err == "no_face_detected":
            return Failure(reason="画像に顔が検出されませんでした", retryable=False)
        if err == "multiple_faces":
            count = resp.json().get("count", 0)
            return Failure(
                reason=f"複数の顔が検出されました（{count}人）。1人だけ写った画像で再試行してください",
                retryable=True,
            )
        return Failure(reason=f"enroll_error: {err}", retryable=False)

    if resp.status_code != 200:
        return Failure(reason=f"enroll HTTP {resp.status_code}: {resp.text[:200]}", retryable=True)

    body = resp.json()
    return Success[FaceEnrollValue](
        value=FaceEnrollValue(
            enrolled=body.get("status") == "enrolled",
            faces_detected=body.get("faces_detected", 1),
        ),
        summary="マスターの顔を登録しました。",
    )
