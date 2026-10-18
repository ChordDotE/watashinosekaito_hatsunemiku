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

"""**形を変えたときに残す工程の出力へ、新しい形の形式の検査を当てる**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §4-2 の3・F11）。

形を変える（`change_form`）と、両方の形にある工程は前の版の出力を残せる。ただし新しい形の決まりに合わない出力
（アカペラにした編成に楽器が残っている・インストにした設計の終わり方が vocal・歌ものに戻した計画が前奏に声を置いている ほか）は
残せないので、その工程も作り直す。どの工程を作り直すかは機械が決める（`song_gate.form_change` の `check`）。

    song_gate.form_change(old, new, check=lambda step: form_findings(song, step, new))

当てるのは**形で決まりが変わる形式の検査だけ**（答えが1つに決まるもの）で、**各工程の Step の検査と同じ部品を呼ぶ**（検査を2本書かない）:

| 工程 | 部品（工程の検査も同じものを呼ぶ） |
|---|---|
| 設計 | `song_design.DesignStep(form).check`（終わり方 vocal は歌ものだけ） |
| 編成 | `instrumentation.InstrumentationStep(form, 設計の区間).form_check`（楽器0本・ドラム none・lead・lead_sections） |
| 伴奏 | `song_arrange.lead_gain_problems`（編成の lead と lead_gain） |
| ハモリ | `harmony.HarmonyPlanStep(設計の区間, form).check`（歌の無い区間に置けるのはアカペラだけ） |
| 声部 | `harmony.voice_findings`（いまのハモリの計画の狙いの区間・番号に合うか） |

ほかの工程（テーマ・コード・グルーヴ・旋律・詞の側・歌い方・歌の崩し・彩り・奏法）には形で変わる形式の決まりが無いので、所見は出さない。
前の工程の出力は、作ったときにその形の検査を通っている（形に依らない検査は当て直さない）。
出力が無い工程（その形で走らない・まだ作っていない）は所見なし。保存から読んだ出力（dict）が型に戻せなければ、
「読めない」を所見にする（確かめられない出力は残さない＝作り直す）。
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def _ns(x: Any) -> Any:
    """型に戻せない保存の dict を、属性で読める形にする（検査の部品は属性で読む）。"""
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_ns(v) for v in x]
    return x


def _as(model_cls: Any, value: Any) -> Any:
    """工程の出力を型に戻す（保存から読んだ曲は dict のまま持っている・`Song.load_record`）。戻せなければ属性で読める形。
    走らなかった工程の空の出力（`skipped`・F1）は出力が無いもの（None）として返す。"""
    from song_model import made                             # noqa: E402
    if not made(value):
        return None
    if isinstance(value, model_cls) or not isinstance(value, dict):
        return value
    try:
        return model_cls.model_validate(value)
    except Exception:                                       # noqa: BLE001（古い記録など。属性で読めるかは下の検査が試す）
        return _ns(value)


def _sections(song: Any) -> list:
    from song_design import Design                          # noqa: E402
    design = _as(Design, getattr(song, "design", None))
    return list(getattr(design, "sections", None) or [])


def _bar_ticks(song: Any) -> int:
    from song_design import Design, bar_ticks              # noqa: E402
    design = _as(Design, getattr(song, "design", None))
    ts = getattr(design, "time_signature", None) or getattr(getattr(song, "score", None), "time_signature", None) or "4/4"
    return bar_ticks(ts)


def _check_step(song: Any, step: str, form: str) -> list[str]:
    if step == "設計":
        from song_design import Design, DesignStep         # noqa: E402
        design = _as(Design, getattr(song, "design", None))
        return [] if design is None else DesignStep(form).check(design)
    if step == "編成":
        from instrumentation import Instrumentation, InstrumentationStep   # noqa: E402
        plan = _as(Instrumentation, getattr(song, "instrumentation", None))
        return [] if plan is None else InstrumentationStep(form, _sections(song)).form_check(plan)
    if step == "伴奏":
        from instrumentation import Instrumentation        # noqa: E402
        from song_arrange import ArrangePlan, lead_gain_problems   # noqa: E402
        plan = _as(ArrangePlan, getattr(song, "arrangement", None))
        if plan is None:
            return []
        iplan = _as(Instrumentation, getattr(song, "instrumentation", None))
        return lead_gain_problems(plan, getattr(iplan, "lead", None))
    if step == "ハモリ":
        from harmony import HarmonyPlan, HarmonyPlanStep   # noqa: E402
        plan = _as(HarmonyPlan, getattr(song, "harmony", None))
        return [] if plan is None else HarmonyPlanStep(_sections(song), form).check(plan)
    if step == "声部":
        from harmony import HarmonyPlan, VoicePart, voice_findings   # noqa: E402
        part = _as(VoicePart, getattr(song, "voices", None))
        plan = _as(HarmonyPlan, getattr(song, "harmony", None))
        if part is None:
            return []
        n_aims = len(getattr(plan, "parts", None) or [])
        return voice_findings(part, _bar_ticks(song), n_aims, plan=plan,
                              bars_of=dict(getattr(song, "bars_of", None) or {}) or None)
    return []                                               # 形で変わる形式の決まりが無い工程


def form_findings(song: Any, step: str, form: str) -> list[str]:
    """**`song` に残っている `step` の出力へ、形 `form` の形式の検査を当てた所見**（空なら通った）。

    `step` は `song_gate.STEP_ORDER` の工程名（とテーマ）。`form` は `song_gate.FORM_STEPS` の形（読めなければ `ValueError`）。
    その形で走らない工程・出力の無い工程は所見なし。出力が読めなければ、そのことを所見にする（作り直す側に倒す）。
    """
    from song_gate import FORM_STEPS                        # noqa: E402
    if form not in FORM_STEPS:
        raise ValueError(f"形『{form}』は読めない（{' / '.join(FORM_STEPS)}）")
    if step not in FORM_STEPS[form]:
        return []
    try:
        return list(_check_step(song, step, form))
    except (AttributeError, TypeError, KeyError, ValueError) as e:
        return [f"前の版の{step}の出力が読めず、{form} の形の決まりに合うか確かめられない（{type(e).__name__}: {e}）——作り直す"]
