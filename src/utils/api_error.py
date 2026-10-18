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

"""外部 API のエラーを、利用者に見せられる一文へ翻訳する。

**なぜ要るか**: これまで、会話が作れなかったときも声が作れなかったときも、
**利用者の画面には何も出なかった**（ログにだけ出て、ミクは黙る）。
残高切れ・APIキーの誤り・一時的な上限は**利用者にしか直せない**のに、
何が起きたか伝わらないと直しようがない。

**方針**（2026-09-20 本人判断）:

1. **出どころを先頭に出す** —— `[Google AI Studio]` / `[Fish Audio]` / `[本体]`。
   利用者はまずここで切り分ける。
2. **エラーコードをそのまま見せる** —— README の表と突き合わせられるように。
3. **メッセージ本文で判定しない** —— Google 公式が
   「アプリケーション ロジックの `code` フィールドを使って処理せよ」と書いており、
   **本文の英語は予告なく変わる**。見るのは `code` と HTTP ステータスだけ。
4. ★**知らないコードは「不明」に落とす** —— 分類は親切さを足すだけで、
   **当たらなくても壊れない**。Google がコードを増やしても動き続ける。

出典: https://ai.google.dev/gemini-api/docs/api-errors （2026-09-20 確認）
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional, Tuple

#: 画面の末尾に必ず付ける導線。README の見出し名と一致させること
GUIDE = "詳しくは README の「ミクが返事をしてくれないとき」をご覧ください。"

#: Gemini の `code`（公式の一覧）→ 利用者向けの説明と対処
#: 一覧に無いコードはここに無いだけで、`_BY_HTTP` か「不明」へ落ちる
_GEMINI_BY_CODE: dict[str, str] = {
    "payment_required": "クレジット残高が空のようです。\nAI Studio のお支払いページで補充してください（補充するまで直りません）。",
    "authentication": "APIキーが正しくないようです。\nスタート画面の「設定」タブで入れ直してください。",
    "permission_denied": "APIキーにこの操作の権限がないようです。\nAI Studio でキーの権限を確認してください。",
    "failed_precondition": "課金が有効になっていないようです。\nAI Studio で請求先アカウントを設定してください。",
    "rate_limit_exceeded": "一時的に上限に達しています。少し待つと戻ります。",
    "quota_exceeded": "1日の上限に達しました。明日まで待つか、有料枠をご検討ください。",
    "too_many_requests": "短い時間に呼び出しが多すぎました。少し待つと戻ります。",
    "service_unavailable": "Google 側が混み合っているようです。時間をおいて試してください。",
    "api_error": "Google 側で問題が起きているようです。時間をおいて試してください。",
    "deadline_exceeded": "応答が時間内に返りませんでした。もう一度試してください。",
    "model_not_found": "指定されたモデルが見つかりません。設定タブのモデル名を確認してください。",
}

#: `code` が読めなかったときの受け皿（HTTP ステータスだけで言えること）
_BY_HTTP: dict[int, str] = {
    400: "リクエストが受け付けられませんでした。設定を確認してください。",
    401: "APIキーが正しくないようです。\nスタート画面の「設定」タブで入れ直してください。",
    402: "残高が空のようです。補充してください（補充するまで直りません）。",
    403: "この操作の権限がないようです。APIキーを確認してください。",
    429: "一時的に上限に達しています。少し待つと戻ります。",
    500: "相手側で問題が起きているようです。時間をおいて試してください。",
    503: "相手側が混み合っているようです。時間をおいて試してください。",
    504: "応答が時間内に返りませんでした。もう一度試してください。",
}

#: 再試行しても無駄なもの（残高・キー・権限・設定）。★402 は公式に「再試行しない」と明記
_NO_RETRY_HTTP = (400, 401, 402, 403, 404)


#: Google の正規ステータス（`status` フィールド）→ こちらのコード。
#: ★**実機で分かったこと（2026-09-20・VM 検証）**: SDK 経由だと HTTP の番号ではなく
#: この語彙で降ってくる。**APIキーが無効なときは 401 ではなく `400 INVALID_ARGUMENT`**
#: （`reason: API_KEY_INVALID`）。REST のエラー表だけ見ていたら取り逃がしていた
_STATUS_TO_CODE: dict[str, str] = {
    "UNAUTHENTICATED": "authentication",
    "PERMISSION_DENIED": "permission_denied",
    "RESOURCE_EXHAUSTED": "rate_limit_exceeded",
    "FAILED_PRECONDITION": "failed_precondition",
    "UNAVAILABLE": "service_unavailable",
    "INTERNAL": "api_error",
    "DEADLINE_EXCEEDED": "deadline_exceeded",
    "NOT_FOUND": "model_not_found",
}

#: `details` の中の `reason`（いちばん具体的な手がかり）→ こちらのコード
_REASON_TO_CODE: dict[str, str] = {
    "API_KEY_INVALID": "authentication",
    "API_KEY_SERVICE_BLOCKED": "permission_denied",
    "SERVICE_DISABLED": "failed_precondition",
    "BILLING_DISABLED": "failed_precondition",
    "RATE_LIMIT_EXCEEDED": "rate_limit_exceeded",
}


def _extract(exc: BaseException) -> Tuple[Optional[int], Optional[str]]:
    """例外から (HTTP ステータス, code) を取り出す。取れなければ None。

    拾う形は2通りある:

    - `pydantic_ai.exceptions.ModelHTTPError` … `status_code` と `body`
    - `google.genai.errors.APIError`（`ClientError` / `ServerError`）…
      `code`・`status`・`details`。★**実際に飛んでくるのはこちら**（VM で確認）
    """
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = None

    # google-genai の APIError 系
    if status is None:
        c = getattr(exc, "code", None)
        if isinstance(c, int):
            status = c

    code_from_status: Optional[str] = None
    gs = getattr(exc, "status", None)
    if isinstance(gs, str):
        code_from_status = _STATUS_TO_CODE.get(gs.upper())

    # `details` は {"error": {..., "details":[{"reason": ...}]}} の形
    reason_code: Optional[str] = None
    det = getattr(exc, "details", None)
    if isinstance(det, dict):
        err = det.get("error") if isinstance(det.get("error"), dict) else det
        for d in (err.get("details") or []):
            if isinstance(d, dict) and isinstance(d.get("reason"), str):
                reason_code = _REASON_TO_CODE.get(d["reason"].upper())
                if reason_code:
                    break

    body: Any = getattr(exc, "body", None)
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            body = {"_raw": body}

    code: Optional[str] = None
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            raw = err.get("code") or err.get("status")
            if isinstance(raw, str):
                code = raw.lower()
            if status is None and isinstance(err.get("code"), int):
                status = int(err["code"])

    if status is None:
        # 文字列の中に "status_code: 402" の形で残っていることがある
        m = re.search(r"status_code[:=]\s*(\d{3})", str(exc))
        if m:
            status = int(m.group(1))

    # **具体的なものを優先する**: reason（API_KEY_INVALID 等）> status 語彙 > body の code
    return status, (reason_code or code_from_status or code)


def looks_like_llm_error(exc: BaseException) -> bool:
    """この例外は外部 LLM API 由来か（＝出どころを `[本体]` にしないでよいか）。

    HTTP ステータスだけで判定していたが、**実機では
    `google.genai.errors.ClientError` が飛んでくる**（`status_code` を持たない）。
    ここを取り違えると、キーの誤りが `[本体]` として出てしまう（2026-09-20 に実際に出た）。
    """
    if isinstance(getattr(exc, "status_code", None), int):
        return True
    if isinstance(getattr(exc, "code", None), int) and hasattr(exc, "status"):
        return True
    return type(exc).__module__.startswith("google.genai")


def describe(
    exc: BaseException,
    *,
    source: str = "Google AI Studio",
) -> dict:
    """例外を利用者向けの1通知に翻訳する。

    Args:
        exc: 捕まえた例外
        source: 出どころの表示名（`Google AI Studio` / `Fish Audio` / `本体`）

    Returns:
        {"message": 画面に出す全文, "code": コード or None,
         "http": HTTP ステータス or None, "retryable": 再試行して意味があるか}
    """
    status, code = _extract(exc)

    if code and code in _GEMINI_BY_CODE:
        body = _GEMINI_BY_CODE[code]
    elif status in _BY_HTTP:
        body = _BY_HTTP[status]
    else:
        # ★分からないものは、**言い換えずに原文を見せる**（2026-09-20 本人判断）。
        # これはミクの発言ではなくシステムの通知なので、取り繕う必要がない。
        # 「どこで・どういうエラーが出たか」がそのまま分かるほうが、報告もしやすい
        body = ("想定していないエラーが起きました。\n"
                f"{_raw_summary(exc)}\n"
                "スタート画面の「ログ」タブに詳しい内容が出ています。")

    tag = _format_tag(code, status)
    head = f"[{source}] {body}"
    return {
        "message": f"{head}\n{tag}{GUIDE}" if tag else f"{head}\n{GUIDE}",
        "code": code,
        "http": status,
        "retryable": status not in _NO_RETRY_HTTP if status is not None else True,
    }


def describe_text(detail: str, *, source: str, http: Optional[int] = None) -> dict:
    """例外を持っていない失敗（設定不足など）を、同じ形の通知にする。"""
    body = _BY_HTTP.get(http) if http is not None else None
    if body is None:
        body = detail
    tag = _format_tag(None, http)
    head = f"[{source}] {body}"
    return {
        "message": f"{head}\n{tag}{GUIDE}" if tag else f"{head}\n{GUIDE}",
        "code": None,
        "http": http,
        "retryable": http not in _NO_RETRY_HTTP if http is not None else True,
    }


#: 画面に出す前に伏せる文字列（APIキーらしきもの）。**誤って画面共有される事故を防ぐ**
_SECRET_RE = re.compile(r"(AIza[0-9A-Za-z_\-]{10,}|[0-9a-f]{32,}|sk-[0-9A-Za-z]{10,})")


def _raw_summary(exc: BaseException, limit: int = 300) -> str:
    """例外の原文を1行に畳んで返す。**キーらしき文字列は伏せる**。

    システムの通知なので言い換えず、そのまま見せる方針
    （2026-09-20 本人指示）。ただし画面に出る以上、秘密は落とす。
    """
    text = f"{type(exc).__name__}: {exc}"
    text = _SECRET_RE.sub("＊＊＊", text)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def _format_tag(code: Optional[str], status: Optional[int]) -> str:
    """`（payment_required / 402）` の形。どちらも無ければ空文字。"""
    parts = [p for p in (code, str(status) if status is not None else None) if p]
    return f"（{' / '.join(parts)}）\n" if parts else ""
