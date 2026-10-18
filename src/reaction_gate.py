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

"""会話中リアクションの頻度制御（Gateway の STT プロキシから使う）。

STT interim ごとに LLM 判定を素直に投げると、コスト・不自然さ（反応の連発）の両面で
問題になるため、「いつ判定してよいか」「いつ反応してよいか」を本クラスで一元管理する。

制御ルール（docs/plan/conversation_reaction_study.md）:
  1. in-flight 1件制限: 判定中に来た interim は判定しない（判定 ~0.5s < interim 間隔 ~1s）
  2. リアクション後クールダウン: 一度反応したら cooldown_secs（settings の
     conversation_reaction.cooldown_seconds、既定5秒）は判定しない。
     長い発話では5秒おきに再反応しうる（連発の不自然さと「ずっと無反応」のバランス）
  3. テキスト成長チェック: 前回判定時からテキストが十分伸びていなければ判定しない
  4. 最小文字数: 断片（「あの」等）への過剰反応を防ぐ

※「意味的に薄い内容に反応しない」判断そのものは LLM 側（filler_compose の
  「確信がなければ none」「迷ったら none」指示）が担う。本クラスはタイミング制御のみ。

LLM・ネットワークに依存しない純粋ロジック（テスト: tests/test_reaction_gate.py）。
"""
from __future__ import annotations

import time
from typing import Callable

MIN_CHARS = 6          # これ未満のテキストは判定しない
MIN_GROWTH = 4         # 前回判定時からの最低増加文字数
DEFAULT_COOLDOWN_SECS = 5.0


class ReactionGate:
    """1つの入力ストリーム（STT セッション / テキスト入力）に対するリアクション判定のゲート。

    Args:
        cooldown_secs: リアクション後、次の判定を再開するまでの秒数
        min_chars: 判定に回す最小文字数（既定 MIN_CHARS。タイピングは大きめを推奨）
        min_growth: 前回判定時からの最低増加文字数（既定 MIN_GROWTH）
        time_fn: 現在時刻取得関数（テスト用に差し替え可能。既定 time.monotonic）
    """

    def __init__(
        self,
        cooldown_secs: float = DEFAULT_COOLDOWN_SECS,
        min_chars: int = MIN_CHARS,
        min_growth: int = MIN_GROWTH,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cooldown_secs = cooldown_secs
        self._min_chars = min_chars
        self._min_growth = min_growth
        self._time_fn = time_fn
        self._in_flight = False
        self._cooldown_until = 0.0
        self._last_judged_text = ""

    def should_judge(self, interim_text: str) -> bool:
        """この interim テキストで LLM 判定を起動してよいか。

        True を返した場合、呼び出し側は判定を開始し、完了時に
        on_judge_done() を必ず呼ぶこと。
        """
        text = (interim_text or "").strip()
        # テキストが前回判定時より縮んだ（削除された）ら、成長基準を現在値にリセットする。
        # 縮み自体では判定しない（消している最中に反応するのは不自然）が、
        # 削除ぶんが「成長の借金」として残らないようにする
        if len(text) < len(self._last_judged_text):
            self._last_judged_text = text
            return False
        if self._in_flight:
            return False
        if self._time_fn() < self._cooldown_until:
            return False
        if len(text) < self._min_chars:
            return False
        if len(text) < len(self._last_judged_text) + self._min_growth:
            return False
        self._in_flight = True
        self._last_judged_text = text
        return True

    def on_judge_done(self, reacted: bool) -> None:
        """判定完了時に呼ぶ。reacted=True ならクールダウンを開始する。"""
        self._in_flight = False
        if reacted:
            self._cooldown_until = self._time_fn() + self._cooldown_secs

    def on_final(self) -> None:
        """発話確定（final）時に呼ぶ。テキスト成長基準をリセットする。

        クールダウンは時間ベースのため final を跨いで継続する
        （発話末尾で反応した直後、次の発話冒頭で即再反応するのを防ぐ）。
        """
        self._last_judged_text = ""
