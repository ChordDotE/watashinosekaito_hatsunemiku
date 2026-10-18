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

"""ミクの内部メモ帳の編集ツール。

KV の "miku_note" を**部分的に**書き換える。更新後の全文を渡して上書きするのではなく、
変えたい箇所だけを指定する（コーディング AI のファイル編集と同じ考え方）。

**なぜ編集方式なのか**（2026-08-29 に全文上書きから移行）:
以前は LLM が更新後の全文を生成して置き換えていたが、42,000 字を毎回書き写すことになり、
触るつもりのない箇所まで書き換わっていた。実ログ 145 回ぶんを調べたところ、

  - 変わったページの 33.5% は 1〜2 行の言い換えだけ（意図した編集ではない）
  - その中に転記の破損が混ざっていた（「プレッシャー」→「プレッロッププレッシャー」）
  - 完全に消えた行が 1,322 行あり、うち 1,287 行は現在も戻っていない
  - 1 回の更新に 105.8 秒（直近の中央値）かかっていた

編集方式では指定した箇所以外に手が入らないため、破損も欠落も構造的に起きない。

**メモの構造には一切依存しない**。ページ・見出し・箇条書きのどれを使っていても、
使っていなくても動く（ミクが自分で決めた書き方なので、設置先ごとに変わりうる）。

一致しない・複数ある場合は Failure を返す。呼び出し元（`miku_note_compose`）が
その内容を LLM に見せてやり直させるので、**曖昧なまま保存される経路が無い**。

権限レベル: 2 (memory)
副作用: KV サービスの miku_note を更新する
"""
from __future__ import annotations

import requests
from pydantic import BaseModel, Field

from agent import context_cache
from agent.config import get_kv_base_url, load_settings
from agent.schemas import Failure, Success

_settings = load_settings()
_KV_BASE_URL = get_kv_base_url(_settings)

_KV_KEY = "miku_note"

# Failure.reason に出す文字列の長さ上限。長い old をそのまま返しても読みづらいので頭だけ見せる。
_ECHO_CHARS = 60


class MikuNoteReplaceInput(BaseModel):
    old: str = Field(description="メモの中にある、置き換えたい文字列。一字一句そのまま渡す")
    new: str = Field(default="", description="置き換えた後の文字列。空にするとその箇所を削除する")


class MikuNoteAppendInput(BaseModel):
    text: str = Field(description="メモの末尾に足す文字列")


class MikuNoteEditValue(BaseModel):
    chars_before: int = Field(description="編集前のメモの文字数")
    chars_after: int = Field(description="編集後のメモの文字数")
    excerpt: str = Field(default="", description="書き換えた箇所（先頭 200 文字）")


MikuNoteEditResult = Success[MikuNoteEditValue] | Failure


def _kv_get(key: str) -> str:
    r = requests.get(f"{_KV_BASE_URL}/kv", params={"key": key}, timeout=10)
    if r.status_code == 404:
        return ""
    r.raise_for_status()
    return r.text


def _kv_set(key: str, value: str) -> None:
    r = requests.post(f"{_KV_BASE_URL}/kv", json={"key": key, "value": value}, timeout=10)
    r.raise_for_status()


def _head(text: str) -> str:
    """Failure に載せる抜粋。改行は潰して1行にする。"""
    flat = " ".join(text.split())
    return flat[:_ECHO_CHARS] + ("…" if len(flat) > _ECHO_CHARS else "")


def _success(before: str, after: str, excerpt: str, what: str) -> Success[MikuNoteEditValue]:
    # メモは明示キャッシュの固定部に載っている。書き換えたら中身が変わるので捨てる
    # （次のループで作り直される。2026-08-30）
    context_cache.invalidate("miku_note_updated")
    return Success[MikuNoteEditValue](
        value=MikuNoteEditValue(
            chars_before=len(before),
            chars_after=len(after),
            excerpt=excerpt[:200],
        ),
        side_effects=[f"miku_note {what}(chars={len(before)}→{len(after)})"],
        summary=f"メモを{what}しました（{len(before)} → {len(after)} 文字）",
    )


def run_miku_note_replace(input_: MikuNoteReplaceInput) -> MikuNoteEditResult:
    """メモの中の一箇所を書き換える（`new` が空なら削除）。

    `old` がメモに無い、または 2 箇所以上ある場合は Failure を返す。
    どちらも「どう直せばよいか」を reason に書く —— 呼び出し元がそのまま LLM に見せるため。
    """
    if not input_.old:
        return Failure(reason="old が空です。置き換えたい文字列を渡してください", retryable=False)

    try:
        note = _kv_get(_KV_KEY)
    except requests.exceptions.ConnectionError:
        return Failure(reason="kv_service_unreachable", retryable=True)
    except Exception as exc:
        return Failure(reason=f"unexpected: {exc!r}", retryable=False)

    if not note:
        return Failure(
            reason="メモがまだ空です。まず note_append で書き足してください",
            retryable=False,
        )

    hits = note.count(input_.old)
    if hits == 0:
        return Failure(
            reason=(
                f"「{_head(input_.old)}」がメモに見つかりません。"
                "メモを読み直して、実際にある文字列をそのまま渡してください"
                "（省略したり言い換えたりしないこと）"
            ),
            retryable=False,
        )
    if hits > 1:
        return Failure(
            reason=(
                f"「{_head(input_.old)}」がメモの {hits} 箇所にあります。"
                "前後の行も含めて長くし、1 箇所だけに当てはまるようにしてください"
            ),
            retryable=False,
        )

    updated = note.replace(input_.old, input_.new, 1)
    try:
        _kv_set(_KV_KEY, updated)
    except requests.exceptions.ConnectionError:
        return Failure(reason="kv_service_unreachable", retryable=True)
    except Exception as exc:
        return Failure(reason=f"unexpected: {exc!r}", retryable=False)

    what = "削除" if not input_.new else "書き換え"
    return _success(note, updated, input_.new or input_.old, what)


def run_miku_note_append(input_: MikuNoteAppendInput) -> MikuNoteEditResult:
    """メモの末尾に書き足す。

    既存のどこにも属さない新しい話題を書くときに使う。
    メモが空のとき（初回）もここから始まる。
    """
    if not input_.text.strip():
        return Failure(reason="text が空です。書き足す内容を渡してください", retryable=False)

    try:
        note = _kv_get(_KV_KEY)
    except requests.exceptions.ConnectionError:
        return Failure(reason="kv_service_unreachable", retryable=True)
    except Exception as exc:
        return Failure(reason=f"unexpected: {exc!r}", retryable=False)

    # 既存メモとの間に空行を1つ挟む（メモが空なら何も挟まない）
    updated = f"{note.rstrip()}\n\n{input_.text.strip()}" if note.strip() else input_.text.strip()

    try:
        _kv_set(_KV_KEY, updated)
    except requests.exceptions.ConnectionError:
        return Failure(reason="kv_service_unreachable", retryable=True)
    except Exception as exc:
        return Failure(reason=f"unexpected: {exc!r}", retryable=False)

    return _success(note, updated, input_.text, "追記")
