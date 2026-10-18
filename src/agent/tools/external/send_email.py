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

"""メール送信ツール。

SMTP 経由でマスターにメールを送信する。
原則として自由行動モード（mode=2）時のみ、1 ターン 1 回まで使用する。

送信内容はそのままでは記憶に残らないため、
使用後は次の inner_thoughts に件名・本文の要約を必ず記録すること（Orchestrator プロンプトで指示）。

権限レベル: 3 (external)
副作用: 実際にメールを送信する
"""
from __future__ import annotations

import smtplib
from email.mime.text import MIMEText

from pydantic import BaseModel, Field

from agent.config import get_email_settings, load_settings
from agent.schemas import Failure, Success

_settings = load_settings()


class SendEmailInput(BaseModel):
    subject: str = Field(description="メールの件名（50 文字以内推奨）")
    body: str = Field(description="メールの本文（1000 文字以内推奨）")


class SendEmailValue(BaseModel):
    subject: str
    body_preview: str = Field(description="本文の先頭 100 文字")
    recipient: str


SendEmailResult = Success[SendEmailValue] | Failure

_TIMEOUT_SEC = 30


def run_send_email(input_: SendEmailInput) -> SendEmailResult:
    """SMTP 経由でマスターにメールを送信する。"""
    try:
        cfg = get_email_settings(_settings)
    except KeyError as exc:
        return Failure(reason=f"email_settings_missing: {exc}", retryable=False)

    try:
        msg = MIMEText(input_.body, "plain", "utf-8")
        msg["Subject"] = input_.subject
        msg["From"] = cfg["smtp_user"]
        msg["To"] = cfg["recipient"]

        with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"], timeout=_TIMEOUT_SEC) as smtp:
            smtp.starttls()
            smtp.login(cfg["smtp_user"], cfg["smtp_password"])
            smtp.sendmail(cfg["smtp_user"], [cfg["recipient"]], msg.as_string())

        value = SendEmailValue(
            subject=input_.subject,
            body_preview=input_.body[:100],
            recipient=cfg["recipient"],
        )
        side_effect = (
            f"send_email(subject={input_.subject!r}, "
            f"to={cfg['recipient']!r}, "
            f"body_chars={len(input_.body)})"
        )
        return Success[SendEmailValue](
            value=value,
            side_effects=[side_effect],
            summary=f"メール送信完了: 件名={input_.subject!r}",
        )
    except TimeoutError:
        return Failure(reason="smtp_timeout", retryable=True)
    except smtplib.SMTPAuthenticationError:
        return Failure(reason="smtp_auth_error", retryable=False)
    except smtplib.SMTPException as exc:
        return Failure(reason=f"smtp_error: {exc!r}", retryable=True)
    except Exception as exc:
        return Failure(reason=f"unexpected: {exc!r}", retryable=False)
