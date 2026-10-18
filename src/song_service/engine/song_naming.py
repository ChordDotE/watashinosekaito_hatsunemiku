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

"""**曲名の工程**——詞ができたあとで、最後の曲名を決める（2026-10-03・本人了承）。

## 前提・背景

組F2（2026-10-03）で曲名の持ち主をテーマ1つにした（`Theme.title`・曲の形で置き場を変えない）。ただテーマは詞より前に走るので、
**詞を書いてみて中身が動いても、曲名はテーマの思いつきのまま**だった。プロの作詞は「タイトルを先に決めて書き始める」が、
書き終えてから中身に寄せて付け直すこともある。そこで:

- テーマの曲名は**仮の曲名**（`Theme.working_title`）にする。詞の設計・サビの詞・詞はこれを受けて書く（「タイトルが決まっていない
  状態で書き始めることはない」は仮の曲名に対して保つ）
- 詞のあとに工程「曲名」を置き、**最後の曲名と理由**を決める（仮の曲名をそのまま採ってもよい）。どの形でも走る
  （インストでは詞の系統が走らないので旋律のあと）
- 曲名を読む工程は無い（編成・伴奏・彩りの見出しに出ても音楽の入力ではない・`song_gate.DEPS`）。曲名だけを作り直しても
  ほかは作り直さず、楽譜の曲名は組み立て（`song_place.make_score`）が毎回この工程の値を写す

## このファイル

| 名前 | 中身 |
|---|---|
| `Naming` | 出力（曲名・理由）。曲名は日本語の欄なので前後の空白を全角・半角とも取り除く（仕様書 §0-6） |
| `NamingStep` | 形式の検査（空・改行）と差分の直し（`StepFixer`） |
| `naming_prompt` | 依頼文（頼み・テーマ・設計の要点・詞の設計・サビの詞・詞／インストは旋律の狙い） |
| `run_naming` | 工程（`song_compose._compose_once_body` が詞のあとに呼ぶ） |

触れる軸: 継続性（曲名は曲の一覧・フォルダ・ミクの話題で残り続ける名前）。費用は flash の1呼び出しぶん。
"""
from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_ai import Agent

import song_context as ctx                                 # LLM の呼び出し口（`ctx._call`＝再生が差し替える）・申し送り
from song_fix import StepFixer, guard_patch
from song_flow import _rec
from song_llm import pick_model
from song_model import PatchStatus, StepStatus, made
from song_prompts import prompt


def _one_line_problem(title: str) -> str:
    """曲名の形式の所見（無ければ空）——空でない・改行が無い・かぎかっこで囲まない。"""
    t = (title or "").strip()
    if not t:
        return "曲名（title）が空——曲名を1行で書く"
    if len(t.splitlines()) > 1:
        return f"曲名（title）『{t}』に改行がある——曲名は1行で書く（改行を入れない）"
    # ★囲みは書き方を1通りにするため（2026-10-03）。曲名は見出し・返事で『』に入れて出すので、囲むと『『…』』になる
    #   中に同じかっこがあれば囲みでなく曲名の一部（「好き」と「嫌い」）
    if len(t) >= 2 and (t[0], t[-1]) in (("『", "』"), ("「", "」")) and t[0] not in t[1:-1] and t[-1] not in t[1:-1]:
        return f"曲名（title）{t} をかぎかっこで囲んでいる——曲名だけを書く（囲みを外す）"
    return ""


class Naming(StepStatus):
    """曲名の工程の出力（2026-10-03）。"""
    reason: str = Field(description=(
        "**なぜこの名前にしたか**。仮の曲名をそのまま採ったなら、なぜそのままでよいか（詞の中身と噛み合っているか）を、"
        "変えたなら、詞のどこ（決め台詞・サビの言葉・心の動き）に寄せたかを書く"))
    title: str = Field(description="**曲名**（日本語で・1行。改行を入れない）")

    # ★**日本語の欄なので、前後の空白は全角・半角とも取り除いて読む**（2026-10-03・仕様書 §0-6 の「日本語を書く欄」。
    #   詞・見せる詞と同じ側）。空・改行は型では差し戻さず、形式の検査（`NamingStep`）が所見にして差分で直させる
    @field_validator("title")
    @classmethod
    def _trim(cls, v: str) -> str:
        return (v or "").strip()


class NamingPatch(PatchStatus):
    """曲名の直し（差分）。曲名は欄が1つなので、直すなら曲名を丸ごと書く。"""
    reason: str = Field(description="何をどう直したか（一言）")
    title: str = Field(default="", description="直した曲名（1行・改行を入れない）。直さないなら空のまま")

    @field_validator("title")
    @classmethod
    def _trim(cls, v: str) -> str:
        return (v or "").strip()


_namer = Agent(pick_model("TITLE", "flash"), output_type=Naming, system_prompt=prompt("song_title"))
_naming_fixer = guard_patch(Agent(pick_model("TITLE", "flash"), output_type=NamingPatch,
                                  system_prompt=prompt("song_title_fix")))


class NamingStep(StepFixer):
    """**曲名の形式の検査と差分の直し**（2026-10-03）。値は曲名の工程が書いた `Naming`。

    形式だけ: 曲名が空でない・改行が無い（曲名はフォルダ名・曲の一覧の1行・途中保存のファイル名になる）。
    曲名が詞・テーマと噛み合うか（内容）は仕上げの点検役（`song_finish_gate`）が見る。
    """
    key = "naming"
    label = "曲名"
    agent = _naming_fixer
    return_hint = "直した曲名を `title` に**丸ごと1行で**書いてください（直さないなら空のまま）。"

    def check(self, value) -> list[str]:
        why = _one_line_problem(getattr(value, "title", ""))
        return [why] if why else []

    def show(self, value) -> str:
        return f"【いまの曲名】『{value.title}』\n【理由】{value.reason}"

    def patch_problem(self, value, patch) -> str:
        t = getattr(patch, "title", "") or ""
        if t and (why := _one_line_problem(t)):
            return f"差分の曲名: {why}"
        return ""

    def apply(self, value, patch):
        t = (getattr(patch, "title", "") or "").strip()
        done = []
        if why := self.patch_problem(value, patch):        # 差分の検証（`guard_patch`）が差し戻すので来ない。来たら当てずに記録へ
            done.append(f"当てなかった: {why}")
            t = ""
        new = value.model_copy(update={"title": t}) if t else value.model_copy(deep=True)
        if t:
            done.append(f"曲名を『{t}』に")
        note = "・".join(done) if done else "直しなし"
        if getattr(patch, "trouble", ""):
            note += f"／直さなかった理由: {patch.trouble}"
        return new, note


# ---- 依頼文 ----------------------------------------------------------------------------------

def _display_of(p) -> str:
    """フレーズの見せる詞（無ければ歌う詞）。"""
    return (getattr(p, "display", "") or getattr(p, "text", "") or "").strip()


def _lyrics_text(song) -> str:
    """詞の本文（区間ごと・見せる詞）。旋律のフレーズの順に区間の見出しを付ける。"""
    words, melody = song.lyrics, song.melody
    by = {p.index: p for p in (getattr(words, "phrases", None) or [])}
    out, cur = [], None
    for i, ph in enumerate(getattr(melody, "phrases", None) or [], 1):
        sec = getattr(ph, "section", "")
        if sec != cur:
            cur = sec
            out.append(f"■{sec}")
        if i in by and (d := _display_of(by[i])):
            out.append(f"  {d}")
    return "\n".join(out)


def naming_prompt(song, instruction: str) -> str:
    """**曲名の工程の依頼文**。テーマ・設計・旋律の申し送り（`brief`）に、頼み・仮の曲名・（歌のある形では）詞の設計・サビの詞・詞、
    （インストでは）旋律の狙いを足す。マスターの頼み（宛先『曲名』）と前段の出力の全部は `song_llm.run_sync` が末尾に足す。"""
    theme, design, melody = song.theme, song.design, song.melody
    parts = [ctx.brief(theme, design, None, melody),
             f"【頼み】{instruction or '（無し）'}",
             f"【テーマ】仮の曲名『{theme.working_title}』／何を歌うか: {theme.subject}"
             f"\n  伝えたいこと: {theme.message}\n  心の動き: {theme.arc}",
             f"【設計の要点】{design.concept}"
             + "\n  構成: " + " → ".join(f"{s.kind}(強さ{s.intensity})" for s in design.sections)]
    plan, chorus = song.lyric_plan, song.lyric_chorus
    if made(song.lyrics):
        if made(plan):
            parts.append(f"【詞の設計】決め台詞（hook）「{plan.hook}」（{plan.hook_section} の {plan.hook_line}行目）"
                         f"\n  視点と時制: {plan.viewpoint}")
        if made(chorus) and chorus.phrases:
            parts.append("【サビの詞】\n" + "\n".join(f"  {_display_of(p)}" for p in chorus.phrases if _display_of(p)))
        parts.append("【詞】\n" + _lyrics_text(song))
    else:
        parts.append(f"【旋律の狙い】{melody.reason}"
                     + "\n※この曲は歌詞が無い（詞の工程が走らない形）。曲名は、テーマと旋律の狙いから決める")
    parts.append("この曲の**最後の曲名**を決めてください。仮の曲名をそのまま採ってもかまいません。")
    return "\n\n".join(parts)


def run_naming(st) -> None:
    """**曲名の工程**（詞のあと・インストでは旋律のあと・どの形でも）。

    読む: `song.theme`（仮の曲名）・`song.design`・`song.melody`・`song.lyric_plan`・`song.lyric_chorus`・`song.lyrics`（走った形だけ）。
    書く: `song.naming`。曲名の読み手は `Song.title()`（楽譜 → ここ → 仮の曲名）と、組み立てが楽譜へ写す所（`song_place.make_score`）。
    ★これより前の途中保存は仮の曲名の名前で書かれ、ここで名前が変わると `song_flow._autosave` が新しい名前で書いて古いほうを消す。
    """
    song, verbose = st.song, st.verbose
    out = ctx._call(_namer, naming_prompt(song, st.instruction or song.instruction), "曲名")
    # 形式の検査（空・改行）を差分で直させる。直らなければ曲名の工程で止まる（`StepUnfixed`）。保存から戻した曲名（再生）は検査しない
    out, _ = NamingStep().run(out, song, verbose=verbose)
    song.naming = out
    _rec(song, "naming", out, verbose=verbose)
    if verbose:
        same = "（仮の曲名のまま）" if out.title == song.theme.working_title else f"（仮の曲名『{song.theme.working_title}』から）"
        print(f"  [曲名] 『{out.title}』{same}")
