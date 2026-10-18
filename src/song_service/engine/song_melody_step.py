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

"""**旋律の工程**——依頼して受け取り（`call_melody`）、検査して差分で直させ（`MelodyStep`・`fix_melody`）、点検役に見せる（`run_melody`）。

2026-09-30 に `song_compose` から分けた（型は `song_melody`、依頼文は `song_melody_prompt`）。
工程の流れは `run_melody` の1か所: 依頼文 → 旋律（1案）→ 形式の検査と差分の直し（直らなければ止める＝`StepUnfixed`）
→ 区間名の揃え → 所見（息継ぎ・区間の頭の間）
→ 点検役（毎回）→ accept なら曲に載せる／redo なら旋律だけ作り直す／前の工程が原因なら差し戻し（`RedoFrom`）。
"""
from __future__ import annotations

import math

from pydantic_ai import Agent

import song_checks as ck                                   # 形式の検査の部品
import song_context as ctx                                 # LLM の呼び出し口（`ctx._call`・`ctx.judge`）・申し送り・拍子
import song_finish_facts as ff                             # 跳躍のあとの伸ばしの目安（最終の点検役の表と同じ数え方・2026-10-04）
from harmony import MELODY_LO                              # 音域は harmony が正（主旋律の下端。声部の `SING_LO` とは別・2026-10-05）
from song_design import Q, bar_ticks, key_shift, split_bar_chords, theme_direction_facts
from song_fix import StepFixer, cut_note, guard_patch      # 工程の形・2026-09-28／差分の検証・2026-10-03／黙って切らない・組F2
from song_flow import _maybe_redo_from, _rec, _redo_options
from song_gate import MAX_REDO, Finding
from song_gate_brief import redo_ask                       # 前回の指摘と「指摘への対応」の頼み（2026-10-04・組P2a）
from song_llm import pick_model, solfa_to_midi
from song_melody import (AbcLine, AbcPatch, AbcSketch, BareMelody, _line_slots, _note_name, abc_pitch,
                         abc_sketch_to_melody, grid_per_bar, pitch_with_abc, sketch_record, top_note_of)
from song_melody_prompt import melody_prompt, meter_text
from song_model import StepRecord
from song_prompts import prompt                          # プロンプトの正は prompts/*.md
from song_rhythm import breath_facts, inspect_breath, lead_gaps, melody_numbers


# ★**旋律は1案で作る**（2026-09-30・本人「一旦 A にしようか」）。以前は3案を同時に作って選抜役が選んでいたが、
#   選抜役は旋律の本文（ABC）を読まず数字の札だけで選び、選んだ旋律が点検役に破綻と判定されても旋律の関門には
#   自工程のやり直しが無く進んだ。3案は工程費の約45〜49%。「方針と違う旋律」は内容の判断なので点検役が見る（毎回呼ぶ）。
#   ※3案を同時に走らせていたときは、スレッドごとに Agent を作り直していた（共有すると後始末に失敗した・decisions.md 2026-09-22）。
#     いまは並列で呼ばないので1つを使い回す。並列に戻すなら、そこでスレッドごとに作ること
_melody = Agent(pick_model("MELODY", "flash"), output_type=AbcSketch, system_prompt=prompt("song_melody"))


# ★当てられない差分（範囲外・同じ番号の2つ目・中身の無い直し）は差し戻す（2026-10-03 再レビュー・`MelodyStep.patch_problem`）
_melody_fixer = guard_patch(Agent(pick_model("MELODY", "flash"), output_type=AbcPatch,
                                  system_prompt=prompt("song_melody_fix")))


def call_melody(prompt_text: str, label: str, groove, bpm: float,
                bars: dict[str, int] | None = None) -> "BareMelody":
    """旋律を1回呼んで、**受け取った直後にスケッチを楽譜の形へ整える**（2026-09-09）。

    呼び出しは2箇所（旋律の工程＝点検役のやり直しも同じ所・空のやり直し）あるので、
    **展開をここ1か所に集める**——散らすと、どれか1つで整え忘れる。

    ★**再生・続きから作る・作り直し（`song_replay`）は `BareMelody` を返す**（保存は展開後の `phrases`）。その場合はそのまま通す。
    """
    got = ctx._call(_melody, prompt_text, label)
    if hasattr(got, "phrases"):                            # 保存から戻した旋律（再生・続きから作る・作り直し）
        return got
    if not isinstance(got, AbcSketch):
        raise TypeError(f"旋律が ABC でも楽譜でもない形で返った: {type(got).__name__}")
    # ★**音価は LLM が書いたものを使う**（2026-09-20・測って採用。根拠は `abc_sketch_to_melody` の説明）。
    # ※枠記法（`onsets`）へ戻して小節線の文字数を見る検査（`bar_line_issues`）は 2026-09-30 に外した。ABC から枠へ戻すと
    #   小節線が消えるので、ABC では一度も所見を出していなかった。小節線の位置は、行の長さと一緒に旋律の検査
    #   （`MelodyStep`・`line_shape`・2026-09-30 記-8）が ABC の `|` で見る
    mel = abc_sketch_to_melody(got, groove, bpm=bpm, bars=bars)
    # ★**LLM が書いたものを残す**（2026-09-09）。保存は展開後の `phrases` なので、
    #   本番で壊れたとき**何を書いたのか追えなかった**（実際に追えず、楽譜から逆算した）。
    #   `sketch` に文字列で残しておけば、`.steps.json` から読める。
    mel.sketch = sketch_record(got)
    mel.__dict__["_abc"] = got                             # ★書いた ABC を持っておく（検査して差分で直させる元・`fix_melody`）
    return mel


# ---- 旋律の検査と直し（2026-09-27・`song_fix`）-------------------------------------
#
# ★**機械は旋律を書き換えない**。以前はここで、枠と長さの違う行を伸縮し、最高音を超えた音を1オクターブ下げ、
#   最低音が低いと選抜した案を捨てて作り直し、低い音（シ3）を黙って1オクターブ上げていた（目録の旋律-1/3/5・関門-1）。
#   いまは**所見を旋律の工程へ返し、差分で直させる**。直らなければ旋律の工程で曲づくりを止める（`song_fix.StepUnfixed`・2026-09-30）。
#   どうしても機械が直すもの（行の伸縮・明らかな暴走の音）は `song_fix.record_change` で記録し、後の工程に見せる。

def _sec(name: str, slots: dict[str, list[int]]) -> str:
    """行の区間名を枠の表（設計の区間名で引く）の名前に照らす。**設計との照合と同じ照らし方**（`resolve_section`＝前後の空白だけ取り除く）。
    ★2026-10-03（再レビュー）まではここだけ完全一致で引き、区間名に空白が付くと行の長さ・小節線・行数の検査が全部素通りした
      （設計との照合は通り、`canonize` が綴りをそろえてから組み立てるので、長さの違う行がそのまま鳴った）。照らせなければ書いた名前のまま"""
    return ck.resolve_section(name, slots) or name


def _slots_per_line(got, slots: dict[str, list[int]]) -> list[int | None]:
    """各行の枠（単位数）。区間名が設計に無い行は None（名前の誤りは設計との照合が所見にする）。"""
    seen: dict[str, int] = {}
    out: list[int | None] = []
    for ln in got.lines:
        sec = _sec(ln.section, slots)
        i = seen.get(sec, 0)
        seen[sec] = i + 1
        xs = slots.get(sec) or []
        out.append((xs[i] if i < len(xs) else xs[-1]) if xs else None)
    return out


def line_shape(where: str, abc: str, slot: int, upb: int) -> list[str]:
    """**1行の長さと小節線の位置**（2026-09-30・記-8 本人「A」）。所見は1行に1つまで。

    - 行の長さが枠（`slot`＝グルーヴが決めた小節数×1小節の単位数）と違う → 長さの所見。`|` が2つ以上の小節を区切っていて、
      1小節ちょうどでない小節があれば、どの小節かを同じ所見に添える（行末の小節が短い＝行が早く終わった、もここに入る）
    - 長さは合うのに `|` と `|` のあいだが1小節ちょうどでない → 小節線の所見（`|` を打ち直す）
    - 複数小節の枠を `|` なしで書いた → 小節線の所見（`|` を打つ）

    ★**行はいつも小節の頭から置かれる**（組み立ては前の行の終わりから続けて置き、枠は小節の整数倍）。弱起（食い）は、
      指示書（`song_melody.md`「食い」）のとおり**前の小節の残りを `z` で埋めて書く**（`z14 C2 | …`）。行の頭の小節が短い
      （標準の ABC の弱起 `C2 | …`）と、書いた音は小節の頭で鳴るので、それも小節線の所見にして書き方を添える。
    ★**臨時記号は `|` まで効き**（`notes_of`）、後の工程（詞・伴奏の LLM）も書かれた ABC の `|` を小節線として読むので、
      長さが合っていても `|` の位置は形式として見る。
    ※2026-09-27 に外した小節線の検査（`abc_bar_issues`）の理由は2つ——①本当の事故（3/4 を 4/4 で書く）は行の長さの検査が
      同時に捕まえる（同じ行に所見が2つ）②長さは合うのに小節線だけずれた行で直しを呼んでいた（当時の数え方で約40案中26行・
      「音に影響しない」）。①は長さの所見に添えて1行1所見にした。②は、`|` が臨時記号の範囲と後の工程の読む小節を決めるので
      形式として見る（本人「A」）。2026-09-30 の保存済みの記録では、長さが合って小節線だけずれた行は364行中2行
    """
    from abc_notation import bar_units
    segs = bar_units(abc)
    total = sum(segs)
    off = [(i, u) for i, u in enumerate(segs, 1) if u != upb]
    detail = "・".join(f"{i}小節目が{u}単位" + (f"（{upb}であるべき）" if k == 0 else "")
                       for k, (i, u) in enumerate(off))
    pickup = (f"。行は小節の頭から置かれるので、1小節目も{upb}単位ちょうどに（弱起・食いは前の小節の残りを `z` で埋めて書く: "
              f"`z{upb - 2} C2 | …`）" if len(segs) > 1 and segs[0] < upb else "")
    # ★**許容は 0**（2026-09-29）。行は前の行の終わりから続けて置くので、1単位のずれでも後ろの行が
    #   全部コードと小節線からずれる（以前は±1単位を許し、ずれが区間の中で溜まっていた）
    if total != slot:
        msg = ck.length_is(where, total, slot, "単位", tol=0, then=f"枠は{slot / upb:g}小節。行末の伸ばしと休みも数える")[0]
        return [msg + (f"。小節ごと: {detail}" if len(segs) > 1 and off else "") + pickup]
    if not off:
        return []
    if len(segs) == 1:
        return [f"{where}: 小節線 `|` が無い——{slot // upb}小節ぶんを1つの小節に書いている。"
                f"`|` を{upb}単位ごとに打つ（臨時記号は `|` まで効く）"]
    return [f"{where}: 小節線の位置がずれている——{detail}。`|` を小節ちょうど（{upb}単位ごと）の所へ打ち直す"
            f"（臨時記号は `|` まで効き、後の工程も `|` を小節線として読む）" + pickup]


def rest_line(where: str, abc: str, form: str = "song") -> list[str]:
    """**休みだけの行（音の無い行・`z16`）**の所見（2026-10-02 本人「おすすめで」・仕様書 §1・§7）。

    ★1行＝1フレーズで、詞は1行に1フレーズ、区切りの記号（グルーヴの `breaks`）も1行に1文字当たる。休みだけの行はフレーズにならない
      （`abc_to_phrases`）ので、行と詞・区切りの記号の数が食い違い、休みだけの行の記号の意味も決まらなかった（§17 ず-23 の残り）。
      指示書は以前から「小節を丸ごと空にしない」と教えていたが、検査では止めていなかった。行の数はグルーヴが決めているので、消さずに音を書かせる。
    ※読めない書き方の行は記法の所見（`ABC_LINE`）が先に出るので、読める行だけを見る（同じ行に所見を重ねない）。高さは記法と同じ読み方（`notes_of`）
    """
    from abc_notation import ABC_LINE
    if ABC_LINE.problems(abc) or any(nm for nm, *_ in ABC_LINE.read(abc)):
        return []
    if form == "instrumental":                              # インストは楽器が弾く主旋律（言い方だけ・決まりは同じ・2026-10-03）
        return [f"{where}は休みだけで、音が1つも無い——1行は1フレーズ（グルーヴの区切りの記号も1行に1つ）なので、"
                "休みだけの行は書けない。行は消さずに（区間の行の数はグルーヴのまま）、その行にも主旋律の音を書く（弾かない間は、行の中の `z` で作る）"]
    return [f"{where}は休みだけで、音が1つも無い——1行は1フレーズ（詞が1行に1フレーズ当たり、グルーヴの区切りの記号も1行に1つ）なので、"
            "休みだけの行は書けない。行は消さずに（区間の行の数はグルーヴのまま）、その行にも歌う音を書く（歌わない間は、行の中の `z` で作る）"]


class MelodyStep(StepFixer):
    """**旋律の工程の検査と直し**（2026-09-28・工程が自分の検査を持つ形）。値は旋律の工程が書いた ABC（`AbcSketch`）。

    - 設計との照合（親の決まり）: 区間名が設計と一致するか・区間の並びが設計の順か・**歌のある区間が全部あるか**
    - 記法（`written`＝各行の ABC・`abc_notation.ABC_LINE`）: 読めない書き方・高さの違う音のタイ
    - 工程の検査: 区間ごとの行の数・行の長さが枠と合うか＋`|` が1小節ちょうどの所にあるか（`line_shape`・1行1所見）・**休みだけの行が無いか**（2026-10-02）・最高音と最低音（旋律が書く空間＝C メジャー基準・移調前。所見の高さは国際式の音名に ABC を添える）

    ★**歌のある区間が丸ごと無い旋律も所見にする**（2026-09-30・判-4 本人承認・`need_all_sections`）。以前は行の数の検査が
      書いた区間しか見ず、設計との照合も全区間を求めていなかったので、区間が丸ごと無い旋律が形式を通った
      （保存済みの『明日への足音』v1〜v6 は chorus2 の旋律が無く、その区間は歌が鳴らなかった）。コードの工程と同じ決まり。
    """
    key = "melody"
    label = "旋律"
    agent = _melody_fixer
    need_all_sections = True
    ordered_sections = True
    return_hint = "**直す行だけ**を patches に入れてください（行全体を書き直す。直さない行は書かない）。"

    def __init__(self, sections, slots: dict[str, list[int]], top: int, floor: int, form: str = "song"):
        self.design = list(sections)
        self.slots, self.top, self.floor = slots, top, floor
        #: 曲の形（2026-10-03）。決まりはどの形でも同じで、インストは所見と見せ方の言い方だけが「主旋律」になる
        #: （音域の枠はミクの声のまま・F2。インストの旋律にあとから歌を乗せられるように）
        self.form = form or "song"

    def design_items(self, got):
        return [(f"{no}行目", ln.section, None) for no, ln in enumerate(got.lines, 1)]

    def rename_sections(self, got, mapping):
        return got.model_copy(update={"lines": [ln.model_copy(update={"section": mapping.get(ln.section, ln.section)})
                                                for ln in got.lines]})

    def written(self, got):
        from abc_notation import ABC_LINE
        return ((f"{no}行目", ABC_LINE, ln.abc) for no, ln in enumerate(got.lines, 1))

    def check(self, got):
        from abc_notation import ABC_LINE, abc_to_midi
        upb = grid_per_bar()
        out: list[str] = []
        hi_hits: list[tuple[int, int]] = []
        lo_hits: list[tuple[int, int]] = []
        # ★**行の数も見る**（2026-09-29）。以前は1行ずつの長さしか見ず、行が多いと最後の行と同じ枠で照らされて通り、
        #   区間からはみ出して次の区間の音と重なった（重なった音は鳴らない）。行の数はグルーヴが区間ごとに決めている
        counts: dict[str, int] = {}
        for ln in got.lines:                               # 区間名は設計との照合と同じ照らし方（`_sec`・2026-10-03）
            counts[_sec(ln.section, self.slots)] = counts.get(_sec(ln.section, self.slots), 0) + 1
        for sec, n in counts.items():
            want = len(self.slots.get(sec) or [])
            if want and n != want:
                out.append(f"区間『{sec}』の行が{n}行（グルーヴは{want}行）——"
                           f"{'足りない行を足す' if n < want else '多い行を消すか、ほかの行へまとめる'}（区間の長さは行の枠の合計）")
        for no, (ln, slot) in enumerate(zip(got.lines, _slots_per_line(got, self.slots)), 1):
            if slot:
                out += line_shape(f"{no}行目（{ln.section}）", ln.abc, slot, upb)
            # 読めない書き方・タイの所見は記法（`written`）が出す。高さは記法と同じ読み方（`ABC_LINE.read`＝`notes_of`）で取る
            ks = [abc_to_midi(nm) for nm, _n, _t, _b in ABC_LINE.read(ln.abc) if nm]
            out += rest_line(f"{no}行目（{ln.section}）", ln.abc, self.form)
            hi_hits += [(no, k) for k in ks if k > self.top]
            lo_hits += [(no, k) for k in ks if k < self.floor]
        # ※曲頭の休みは旋律の検査から外した（2026-09-28）。足りなければ組み立てが曲の頭に空の1小節を置く（時間軸の `LEAD_IN`・曲全体を同じだけずらす）
        # ★高さは国際式の音名に ABC の書き方を添える（2026-10-01。以前は階名 `ラ5`。直すのは ABC なので字で照らせるように）
        if hi_hits:
            out.append(f"最高音 {pitch_with_abc(self.top)} を超える音: "
                       + "、".join(f"{no}行目の{pitch_with_abc(k)}" for no, k in hi_hits[:6]) + "。設計の最高音までに収める")
        if lo_hits:
            out.append(f"最低音 {pitch_with_abc(self.floor)} より低い音: "
                       + "、".join(f"{no}行目の{pitch_with_abc(k)}" for no, k in lo_hits[:6])
                       + "。低い所だけ持ち上げる（サビの高さは削らない）")
        return out

    def show(self, got):
        upb = grid_per_bar()
        rows = "\n".join(f"{no:>2}. [{ln.section}] {ln.abc}" + (f"   （枠 {s}単位＝{s / upb:g}小節）" if s else "")
                         for no, (ln, s) in enumerate(zip(got.lines, _slots_per_line(got, self.slots)), 1))
        # ★**区間ごとの行の数と枠も見せる**（2026-09-30・判-4）。丸ごと無い区間は上の一覧に出ないので、足す行の数と長さが分からなかった
        want = "\n".join(f"- {s.kind}: {len(self.slots.get(s.kind) or [])}行（枠 "
                         + "・".join(str(x) for x in (self.slots.get(s.kind) or [])) + "単位）"
                         for s in self.design)
        where = "主旋律のある区間" if self.form == "instrumental" else "歌のある区間"
        return (meter_text(ctx.METER) + f"\n\n【{where}（設計の順）と行の数・各行の枠】\n" + want
                + "\n\n【いまの旋律】（番号・区間・ABC・その行の枠）\n" + rows)

    def patch_problem(self, got, patch) -> str:
        """**差分の番号の所見**（無ければ空・2026-10-03 再レビュー）。直しを頼む間だけ、差分の検証（`song_fix.guard_patch`）が読んで差し戻す。

        - 直す・消す行（`insert` でない項目）: 番号は 1〜いまの行の数・同じ番号は1回だけ・drop も abc も section も無い直しは書けない
        - 足す行（`insert`）: 番号は 0〜いまの行の数（0＝先頭）・section と abc は必須（型の説明のとおり）。同じ番号のあとに何行足してもよい
        ★以前は範囲外の番号・同じ番号の2つ目（後が勝つ）・中身の無い直し・section か abc の無い足す行を `apply` が黙って捨てていた
          （次の検査で同じ所見が出るので音は変わらないが、LLM は直したつもりのまま回を重ねて止まる）。編成・伴奏・声部と同じ形。
        ※`drop` と `insert`・`abc`・`section` を一緒に書いた項目は型（`AbcLinePatch._drop_alone`）が差し戻す（2026-10-03・記B）ので、ここには来ない。
        """
        n = len(got.lines)
        ps = list(getattr(patch, "patches", None) or [])
        edits = [p for p in ps if not getattr(p, "insert", False)]
        nos = [p.no for p in edits]
        out: list[str] = []
        if bad := sorted({x for x in nos if not 1 <= x <= n}):
            out.append(f"行 {'・'.join(map(str, bad))} は【いまの旋律】に無い——行は{n}行なので、直す・消す行の no は 1〜{n}"
                       "（【いまの旋律】の番号をそのまま書く）")
        if dup := sorted({x for x in nos if nos.count(x) > 1}):
            out.append(f"行 {'・'.join(map(str, dup))} を2回以上書いた——1つの行に直しは1つ（abc と section は1つの項目にまとめる）")
        if empty := sorted({p.no for p in edits if 1 <= p.no <= n and not getattr(p, "drop", False)
                            and not ck.strip_ascii_ws(p.abc) and not (p.section or "").strip()}):
            out.append(f"行 {'・'.join(map(str, empty))} は drop も abc も section も無い——直すなら abc か section を書き、"
                       "消すなら drop: true、直さないならその行を書かない")
        for p in ps:
            if not getattr(p, "insert", False):
                continue
            if not 0 <= p.no <= n:
                out.append(f"insert の no={p.no} は【いまの旋律】に無い——足すのは 0（先頭）〜{n}行目のあと")
            if not ck.strip_ascii_ws(p.abc) or not (p.section or "").strip():
                out.append(f"insert には section と abc の両方を書く（no={p.no} のあとに足す行）")
        return ("差分の番号: " + "／".join(dict.fromkeys(out))) if out else ""

    def apply(self, got, patch):
        # 番号はすべて**いまの旋律の番号**で読む（足す・消すで番号がずれないよう、元の並びに印を付けてから組み直す）
        lines = list(got.lines)
        drop: set[int] = set()
        adds: dict[int, list] = {}
        done: list[str] = []
        # ★範囲外・同じ番号の2つ目・中身の無い直し・欄の足りない足す行は、差分の検証（`patch_problem`・`song_fix.guard_patch`）が
        #   差し戻すので来ない（2026-10-03）。型を通らずに来たら当てずに、当てなかったことを直しの記録に書く（以前は黙って捨て、2つ目が勝った）
        if why := self.patch_problem(got, patch):
            done.append(f"当てなかった: {why}")
        seen: set[int] = set()
        for p in (getattr(patch, "patches", None) or []):
            # ABC は半角の記法なので前後は半角の空白とタブだけ取り除く（全角の空白などは残して記法の所見に・§0-6・2026-10-03）
            abc, sec = ck.strip_ascii_ws(p.abc), (p.section or "").strip()
            if getattr(p, "insert", False):
                if 0 <= p.no <= len(lines) and abc and sec:
                    adds.setdefault(p.no, []).append(AbcLine(section=sec, abc=abc))
                    done.append(f"{p.no}行目のあとに足した")
                continue
            if not 1 <= p.no <= len(lines) or p.no in seen:
                continue
            seen.add(p.no)
            if getattr(p, "drop", False):
                drop.add(p.no)
                done.append(f"{p.no}行目を消した")
                continue
            upd = {**({"abc": abc} if abc else {}), **({"section": sec} if sec else {})}
            if upd:
                lines[p.no - 1] = lines[p.no - 1].model_copy(update=upd)
                done.append(f"{p.no}行目")
        out = list(adds.get(0, []))
        for no, ln in enumerate(lines, 1):
            if no not in drop:
                out.append(ln)
            out += adds.get(no, [])
        lines = out
        note = (f"直した（{'・'.join(done)}）" if done else "直しなし")
        if getattr(patch, "trouble", ""):
            note += f"／直さなかった理由: {patch.trouble}"
        return got.model_copy(update={"lines": lines}), note


def fix_melody(melody, song, groove, bpm: float, bars: dict[str, int] | None, sung,
               top: int, floor: int, verbose: bool = False) -> tuple:
    """旋律を `MelodyStep` で検査し、所見があれば差分で直させ、直したら ABC から音符を作り直す。戻り値は (旋律, 残った所見)。

    ★直らなければ `StepUnfixed` で止める（残った所見はいつも空・2026-09-30）。
    ★再生（保存から戻した旋律）は `_abc` を持たないので、何もしない——作り直しで旋律を動かさないため。
    """
    got = melody.__dict__.get("_abc")
    if got is None:
        return melody, []
    gv = {g.kind: g for g in groove.sections} if groove else {}
    step = MelodyStep(sung, _line_slots(gv, bars), top, floor, getattr(song, "form", "song") or "song")   # 形を渡す（2026-10-03）
    fixed, left = step.run(got, song, verbose=verbose)
    if fixed is not got:
        melody = abc_sketch_to_melody(fixed, groove, bpm=bpm, bars=bars)
        melody.sketch = sketch_record(fixed)
        melody.__dict__["_abc"] = fixed
    return melody, left


def _written_lines(melody) -> list[tuple[str, str]] | None:
    """旋律の工程が**書いた行**（区間名, ABC）の並び（息継ぎの所見が区切りの記号を行に当てるため・2026-10-02）。

    いま書いた旋律は `_abc`（直したあとの版）から。保存から戻した旋律（再生・続きから作る）は `_abc` を持たないので、
    同じ行を残した `sketch` の【書かれたABC】（`song_melody.sketch_record` の「区間名: ABC」の行）から読む。
    ABC の記録が無い古い旋律（枠記法の頃）は None（フレーズを行として数える）。
    ※休みだけの行は 2026-10-02 から旋律の検査の所見（`MelodyStep.check`）で書けないので、どの行にも音がある（行とフレーズが1対1）。
    """
    got = melody.__dict__.get("_abc")
    if got is not None:
        return [(ln.section, ln.abc) for ln in got.lines]
    sk = getattr(melody, "sketch", "") or ""
    if ctx._ABC_MARK not in sk:
        return None
    body = sk.split(ctx._ABC_MARK, 1)[1].strip().splitlines()
    return [(sec, abc) for sec, abc in (row.split(": ", 1) for row in body if ": " in row)]


def _beat_len() -> float:
    """1拍の長さ（4分音符＝1）。x/8 の3つ割りの拍子（6/8 など）は付点4分を1拍に数える。"""
    num, den = (int(x) for x in ctx.METER.split("/"))
    return 1.5 if den == 8 and num % 3 == 0 else 1.0


def _line_rows(melody, lines: list[tuple[str, str]] | None) -> list[dict]:
    """旋律を**行ごと**に並べる（点検役の材料の素・2026-10-03 組G1）。位置は区間の頭からの拍（4分＝1）。

    各行: `sec`（区間名）・`no`（区間の何行目か）・`start`（行の頭の位置）・`length`（行の長さ。ABC が無ければ None）・
    `notes`（[(位置, 長さ, 書いた高さ＝C 基準の MIDI)]）・`ties`（音ごとにタイでまとめた続きの音の数）。位置は組み立てと同じ数え方（区間ごとに頭から積む・`rhythm_metrics` と同じ）。
    行の頭は書いた ABC の長さを積んで出す（行は小節の頭から置かれ、長さは枠とちょうど同じ＝形式の検査が保証）。ABC の記録が無い古い旋律は、
    最初の音の小節の頭を行の頭とみなす。※音のある行とフレーズは順に1対1（休みだけの行は形式の検査で書けない）
    """
    from abc_notation import bar_units
    bar = bar_ticks(ctx.METER) / Q
    phrases = list(getattr(melody, "phrases", None) or [])
    heads: list[tuple[float, float]] | None = None
    if lines and len(lines) == len(phrases):
        heads, acc = [], {}
        for sec, abc in lines:
            ln = sum(bar_units(abc)) / 4.0                 # 1単位＝16分
            heads.append((acc.get(sec, 0.0), ln))
            acc[sec] = acc.get(sec, 0.0) + ln
    rows: list[dict] = []
    cursor: dict[str, float] = {}
    nth: dict[str, int] = {}
    for j, p in enumerate(phrases):
        sec = p.section
        at = cursor.get(sec, 0.0) + float(getattr(p, "rest_before", 0) or 0)
        notes = []
        for n in p.notes:
            b = float(n.beats)
            notes.append((at, b, solfa_to_midi(n.solfa, n.octave)))
            at += b + float(getattr(n, "gap_after", 0) or 0)
        cursor[sec] = at
        nth[sec] = nth.get(sec, 0) + 1
        start, length = heads[j] if heads else ((math.floor(notes[0][0] / bar + 1e-9) * bar) if notes else 0.0, None)
        rows.append({"sec": sec, "no": nth[sec], "start": start, "length": length, "notes": notes,
                     "ties": [max(0, int(getattr(n, "ties", 0) or 0)) for n in p.notes]})   # 骨格の種類を数える（2026-10-06）
    return rows


def _chord_at(cells: list[str], t: float, bar: float):
    """区間の頭から `t` 拍の所で鳴るコード（書いたまま＝C 基準）→ (名前, 構成音の集合)。無い・読めなければ None。
    1小節に書いたコードは書いた数で小節を等分する（組み立て `song_place.place_chords` と同じ）。"""
    from song_audio import chord_notes
    i = int(t // bar + 1e-9)
    if not 0 <= i < len(cells):
        return None
    parts = split_bar_chords(cells[i])
    if not parts:
        return None
    k = len(parts)
    name = parts[min(k - 1, int((t - i * bar) / (bar / k) + 1e-9))]
    tones = {x % 12 for x in chord_notes(name)}
    return (name, tones) if tones else None


def _tone_share(cells: list[str], t0: float, d: float, pc: int, bar: float) -> float | None:
    """音が**鳴っている間**のうち、コードの音だった時間の割合（RWC-P の実測と同じ数え方＝`song_pipeline` の「伸ばす音は落ち着く音に置かれているか」の注記。以前の楽譜の検証 `_check_hold_settles` は 2026-10-03 に外した）。"""
    t, end, hit, span = t0, t0 + d, 0.0, 0.0
    while t < end - 1e-9:
        i = int(t // bar + 1e-9)
        k = len(split_bar_chords(cells[i])) if 0 <= i < len(cells) else 1
        seg = bar / max(1, k)
        nxt = min(end, i * bar + (math.floor((t - i * bar) / seg + 1e-9) + 1) * seg)
        if nxt <= t + 1e-9:
            nxt = end
        if c := _chord_at(cells, t, bar):
            span += nxt - t
            hit += (nxt - t) if pc in c[1] else 0.0
        t = nxt
    return (hit / span) if span > 0 else None


def _shape_letters(keys: list[tuple]) -> list[str]:
    """行ごとの形 (全体, 小節ごと, 高さを移しても同じ動きか見る列 or None) → 字の並び。
    同じ形＝同じ字・1小節だけ違う（音の形では、同じ動きを高さだけ移したものも）＝ `'`・ほか＝新しい字。"""
    protos: list[tuple[str, tuple, tuple, tuple | None]] = []
    out: list[str] = []
    for whole, bars, moves in keys:
        if hit := next((L for L, w, _b, _m in protos if w == whole), None):
            out.append(hit)
            continue
        near = next((L for L, _w, b, m in protos
                     if (len(b) == len(bars) > 1 and sum(x != y for x, y in zip(b, bars)) == 1)
                     or (moves is not None and m == moves)), None)
        if near:
            out.append(near + "'")
            continue
        L = chr(ord("A") + len(protos)) if len(protos) < 26 else f"#{len(protos) + 1}"
        protos.append((L, whole, bars, moves))
        out.append(L)
    return out


def _skeleton(r: dict) -> tuple:
    """行の**骨格**＝リズムの形（行の頭からの音の位置・長さ・タイの並び。休みの置き方は位置に出る）。高さは見ない。単位は16分。"""
    ties = r.get("ties") or [0] * len(r["notes"])
    return (round((r["length"] or 0) * 4),
            tuple((round((t - r["start"]) * 4), round(d * 4), k) for (t, d, _p), k in zip(r["notes"], ties)))


def frame_facts(rows: list[dict], moras_of: dict[str, int]) -> tuple[dict[str, str], list[str]]:
    """**言葉を置ける枠とグルーヴのモーラ数・骨格の種類**（旋律の点検役の材料・2026-10-06 本人決定「検査を旋律の点検役へ前倒し」）
    → (区間名 → その区間の1行, 全体の行)。

    『火花になって飛べ』（BPM200）は旋律が全60行を「頭8分休符＋5〜6音目にタイ」の同じ骨格で書き、言葉を置ける枠が約8で固定、
    グルーヴの1行のモーラ数（9〜10）より少なかった。詞は詰めて済ませ（`~` 42か所）、詞の点検役で見つかったときには旋律から
    やり直すと後ろの工程を全部捨てることになった。旋律の点検役の時点で、事実として並べる（良し悪しは断じない）。
    - 枠: 行の歌う音の数（`notes`＝タイでまとめた1音を1つ・休みは音でない）。グルーヴの `moras_per_line` と比べ、少ない行を数える
      ※モーラ数は長音・「っ」「ん」も数えるので、枠が少し少なくても詞が長音や促音・撥音で収めれば詰めずに済むことがある
    - 骨格: `_skeleton` が完全に同じ行を1種類と数える（曲全体と区間ごと）
    """
    sec_line: dict[str, str] = {}
    short_all: list[str] = []
    order = list(dict.fromkeys(r["sec"] for r in rows if r["notes"]))
    for sec in order:
        rs = [r for r in rows if r["sec"] == sec and r["notes"]]
        m = int(moras_of.get(sec, 0) or 0)
        frames = [len(r["notes"]) for r in rs]
        kinds = len({_skeleton(r) for r in rs})
        txt = f"  言葉を置ける枠（行ごとの歌う音の数・タイの続きと休みは数えない）: {' '.join(map(str, frames))}"
        if m:
            short = [(r["no"], m - len(r["notes"])) for r in rs if len(r["notes"]) < m]
            txt += (f"／グルーヴの1行 {m}モーラ——枠がモーラ数より少ない行 {len(short)}/{len(rs)}行"
                    + (f"（{'・'.join(f'{no}行目（{d}足りない）' for no, d in short[:12])}{' ほか' if len(short) > 12 else ''}）"
                       if short else ""))
            short_all += [f"{sec} {no}行目" for no, _d in short]
        txt += f"／骨格 {kinds}種類"
        sec_line[sec] = txt
    out: list[str] = []
    sung_rows = [r for r in rows if r["notes"]]
    if moras_of:
        out.append(f"  枠がグルーヴの1行のモーラ数より少ない行（詰めないと収まらない行）: {len(short_all)}/{len(sung_rows)}行"
                   + (f"（{'・'.join(short_all[:16])}{' ほか' if len(short_all) > 16 else ''}）" if short_all else ""))
    if sung_rows:
        cnt: dict[tuple, int] = {}
        for r in sung_rows:
            k = _skeleton(r)
            cnt[k] = cnt.get(k, 0) + 1
        top = max(cnt.values())
        one = len(cnt) == 1 and len(sung_rows) > 1
        out.append(f"  骨格の種類（行ごとのリズムの形＝音の位置・長さ・タイの並びが完全に同じ行を1種類と数える・曲全体）: "
                   f"{len(cnt)}種類／{len(sung_rows)}行（いちばん多い形が {top}行）"
                   + ("——★全行が同じ1つの骨格" if one else ""))
    return sec_line, out


def melody_facts(melody, sung: list, *, bpm: float = 0.0, chords=None, design=None,
                 lines: list[tuple[str, str]] | None = None, groove=None) -> str:
    """旋律の**区間ごとの事実**（旋律の点検役の材料・2026-09-30・2026-10-03 に組み直した＝組G1）。

    音は渡せないので、**聴けば分かることを数字にする**。設計の方針（区間の強さ・全体方針・歌い方の型）と見比べて
    「方針と違う旋律か」を点検役が判断する材料。**どれも数えた事実で、良し悪しの判定でも目標の範囲でもない**（帯の数字は添えない）。

    区間ごとに並べるもの（`docs/plan/song_viewpoints_review.md` §1-2・`song_viewpoints_by_purpose.md` 末尾の決まったこと）:
    - 高さ: 最高・**重心（長さで重みを付けた平均）**・最低を**鳴る高さ**（区間の調・転調へ移した高さ）で。最初の歌の区間との差も
      ※以前は音の数で割った平均を半音に丸め、C 基準（移調前）で出していた——伸ばした高い音1つと刻んだ低い音の群れが同じ重さになり、
        丸めで差が消え（『茜の空』pre と chorus がどちらも B4）、最後のサビで上げる転調も見えなかった
    - 跳躍: **行の中の隣どうし**だけ。行と行のあいだ（休みをはさむ）は別に出す（以前は区間の音を全部つないで数え、行のまたぎが最大になった）
    - 歌の密度: 音のある小節あたりの音の数と、1秒あたりの音の数（本人「言葉が少ない」「間延び」）
    - 16分: 16分の位置で始まる音のある拍が、2拍以上続く所と、1拍だけの所（本人「基本8分で急に16分が混ざると違和感」）
    - コードの音: 拍の頭の音と伸ばす音（1拍より長い音）がそのときのコードの音か（伸ばす音は鳴っている間のうちコードの音だった時間で数える）
    - 繰り返しと変化: 行ごとのリズムの形・音の形を字で並べる（率でなく並びで見せる。以前の「4音のリズム型の再出現」の率は効かなかった）
    - 全体: 音域・フレーズの本数・1行の音の数・長い音と短い音がコードの音で鳴る時間の割合（RWC-P の実測と同じ数え方。
      以前はその結果が楽譜の検証の記録に出るだけで、直せる旋律の点検役に届いていなかった）
    - 言葉を置ける枠とグルーヴの1行のモーラ数・骨格の種類（`frame_facts`・2026-10-06。`groove` があるときだけモーラ数と比べる）
    ※外した: 1音の長さの平均（拍）——平均は伸ばしを薄め、BPM が無いと秒に直せず、材料に30回出て単独でやり直しの理由になったことが無い
    ※以前は旋律3案の選抜の札（`_melody_card`）の前半だった。1案にしたので点検役へ移した（規約「観点を捨てない」）。
    """
    rows = _line_rows(melody, lines)
    if not any(r["notes"] for r in rows):
        return "【区間ごとの事実】音が無い"
    bar, beat = bar_ticks(ctx.METER) / Q, _beat_len()
    spb = 60.0 / bpm if bpm and bpm > 0 else 0.0
    key_of: dict[str, str] = {}
    shift_of = (lambda _k: 0)
    if design is not None:
        try:
            from song_place import section_keys
            key_of, shift_of = section_keys(design)
        except ValueError:                                  # 読めない調（設計の検査が所見にする）
            pass
    if chords is None:
        ch_map: dict[str, list[str]] = {}
    elif isinstance(chords, dict):
        ch_map = {k: list(v or []) for k, v in chords.items()}
    else:
        ch_map = {c.kind: list(getattr(c, "chords_per_bar", []) or []) for c in getattr(chords, "sections", []) or []}
    meta = {s.kind: s for s in sung}
    order = [s.kind for s in sung] + [r["sec"] for r in rows if r["sec"] not in meta]
    moras_of = {g.kind: int(getattr(g, "moras_per_line", 0) or 0) for g in (getattr(groove, "sections", None) or [])}
    frame_by_sec, frame_all = frame_facts(rows, moras_of)

    def where(r: dict, t: float, *, to: float | None = None) -> str:
        # 位置の「拍」はいつも4分音符で数える（6/8 でも。拍の頭の判定だけ付点4分）
        rel = t - r["start"]
        b_, x = int(rel // bar + 1e-9), round(rel % bar + 1, 2)
        at = f"{r['no']}行目 {b_ + 1}小節目{x:g}拍目"
        if to is not None:
            b2, x2 = int((to - r["start"]) // bar + 1e-9), round((to - r["start"]) % bar + 1, 2)
            at += "〜" + (f"{b2 + 1}小節目" if b2 != b_ else "") + f"{x2:g}拍目"
        return at

    def name(k: int) -> str:
        return _note_name(k)

    out = ["【区間ごとの事実】（機械が数えたもの。良し悪しの判定でも目標の範囲でもない）",
           "  高さは**鳴る高さ**（区間の調へ移した高さ・国際式の音名・`C4`＝中央のド）。書かれた ABC とコードは C 基準"
           "（鳴る高さ＝書いた高さ＋区間の移調量）。位置は「何行目 何小節目（行の頭から）何拍目」で、拍は4分音符を1拍に数える"
           + ("（拍の頭・16分のまとまりは付点4分の拍で見る）" if beat != 1.0 else "") + "。"
           "形の字は、同じ字＝同じ形・`'`＝1小節だけ違う（音の形では、同じ動きを高さだけ移したものも `'`）"]
    first: tuple[str, float, int] | None = None              # 最初の歌の区間（名前・重心・最高）
    allk: list[int] = []
    pairs: list[tuple[float, float]] = []                    # (長さ, コードの音だった時間の割合)
    for sec in dict.fromkeys(order):
        rs = [r for r in rows if r["sec"] == sec and r["notes"]]
        if not rs:
            continue
        sh = shift_of(sec)
        ns = [(r, t, d, k) for r in rs for t, d, k in r["notes"]]
        ks = [k + sh for _r, _t, _d, k in ns]
        allk += ks
        w = sum(d for _r, _t, d, _k in ns) or 1.0
        cen = sum((k + sh) * d for _r, _t, d, k in ns) / w
        s_ = meta.get(sec)
        head = (f"■{sec}（強さ{getattr(s_, 'intensity', '?')}・{getattr(s_, 'bars', '?')}小節・{len(rs)}行"
                + (f"・調 {key_of[sec]}（書いた高さ {sh:+d}半音）" if sec in key_of else "") + "）")
        out.append(head)
        hi = f"最高 {name(max(ks))}・重心 {name(round(cen))}（長さで重みを付けた平均）・最低 {name(min(ks))}"
        if first is None:
            first = (sec, cen, max(ks))
        else:
            hi += f"——{first[0]} との差: 重心 {cen - first[1]:+.1f}半音・最高 {max(ks) - first[2]:+d}半音"
        out.append(f"  高さ: {hi}")
        # 跳躍（行の中の隣どうし）／行と行のあいだ（休みをはさむ）
        inner = [(abs(b[2] - a[2]), r, a, b) for r in rs for a, b in zip(r["notes"], r["notes"][1:])]
        jt = "  跳躍（行の中の隣どうし）: " + (
            f"最大 {(m := max(inner, key=lambda x: x[0]))[0]}半音（{m[1]['no']}行目 {name(m[2][2] + sh)}→{name(m[3][2] + sh)}）"
            if inner else "行の中に音が2つ以上ある行が無い")
        cross = [(abs(b["notes"][0][2] - a["notes"][-1][2]), a, b) for a, b in zip(rs, rs[1:])]
        if cross:
            c = max(cross, key=lambda x: x[0])
            jt += f"／行と行のあいだ（参考・休みをはさむ）: 最大 {c[0]}半音（{c[1]['no']}行目の終わり→{c[2]['no']}行目の頭）"
        out.append(jt)
        # 跳躍のあとの長い伸ばし（2026-10-04・『水たまりのプリズム』22秒の「だ」・9半音跳んで2拍）。数え方は最終の点検役の表と同じ
        #   （`song_finish_facts.leap_holds`＝行の中の隣どうし・前の音とのあいだが1拍以内）。崩しはまだ無いので、旋律だけで分かる部分
        lh = [(r, b, b[2] - a[2]) for r in rs for a, b in zip(r["notes"], r["notes"][1:])
              if abs(b[2] - a[2]) >= ff.LEAP and b[1] >= ff.LEAP_HOLD / ff.Q - 1e-9 and b[0] - (a[0] + a[1]) <= ff.LEAP_GAP / ff.Q + 1e-9]
        if lh:
            out.append(f"  跳躍（{ff.LEAP}半音以上）のあとの長い伸ばし（{ff.LEAP_HOLD / ff.Q:g}拍以上）: {len(lh)}か所（"
                       + "・".join(f"{where(r, b[0])} {name(b[2] - j + sh)}→{name(b[2] + sh)}（{j:+d}半音）{b[1]:g}拍" for r, b, j in lh[:6])
                       + (" ほか" if len(lh) > 6 else "") + "）")
        # 歌の密度
        sb = {i for _r, t, d, _k in ns for i in range(int(t // bar + 1e-9), int((t + d - 1e-6) // bar + 1e-9) + 1)}
        dens = f"  密度: {len(ns)}音・音のある小節 {len(sb)}小節あたり {len(ns) / max(1, len(sb)):.1f}音"
        if spb:
            dens += f"・1秒あたり {len(ns) / (len(sb) * bar * spb):.1f}音（音のある小節の時間で割った）"
        out.append(dens)
        # 16分のつまずき: 16分の位置（8分の裏の裏）で始まる音のある拍を、行ごとに続きでまとめる
        zones, singles = [], []
        for r in rs:
            # 行の頭から数える（行は小節の頭から置かれる。行の長さが枠と違う古い記録でも拍の位置がずれないように）
            s0 = r["start"]
            idx = sorted({int(math.floor((t - s0) / beat + 1e-9)) for t, _d, _k in r["notes"] if round((t - s0) * 4) % 2 == 1})
            run: list[int] = []
            for i in idx + [None]:
                if run and (i is None or i != run[-1] + 1):
                    (zones if len(run) >= 2 else singles).append(
                        where(r, s0 + run[0] * beat, to=s0 + run[-1] * beat if len(run) >= 2 else None))
                    run = []
                if i is not None:
                    run.append(i)
        if zones or singles:
            out.append("  16分の位置で始まる音のある拍: "
                       + (f"2拍以上続く所 {len(zones)}か所（{'・'.join(zones[:6])}{' ほか' if len(zones) > 6 else ''}）" if zones else "2拍以上続く所なし")
                       + "／" + (f"1拍だけ（前後の拍には無い）{len(singles)}か所（{'・'.join(singles[:8])}{' ほか' if len(singles) > 8 else ''}）"
                                if singles else "1拍だけの所なし"))
        else:
            out.append("  16分の位置で始まる音のある拍: なし")
        # コードの音（書いたまま＝C 基準で照らす。旋律もコードも C 基準）
        cells = ch_map.get(sec) or []
        if cells:
            heads_ = [(r, t, k, c) for r, t, _d, k in ns
                      if abs(t / beat - round(t / beat)) < 1e-6 and (c := _chord_at(cells, t, bar))]
            off = [(r, t, k, c) for r, t, k, c in heads_ if k % 12 not in c[1]]
            longs = []
            for r, t, d, k in ns:
                share = _tone_share(cells, t, d, k % 12, bar)
                if share is None:
                    continue
                pairs.append((d, share))
                if d > beat + 1e-9:
                    longs.append((r, t, d, k, share))
            loff = [x for x in longs if x[4] < 0.5]

            def at_(r, t, k, chord) -> str:
                return f"{where(r, t)} `{abc_pitch(k)}` on {chord}"
            out.append(f"  拍の頭の音でコードの音でないもの（高さ・コードは書いたまま）: {len(off)}/{len(heads_)}"
                       + (f"（{'・'.join(at_(r, t, k, c[0]) for r, t, k, c in off[:6])}{' ほか' if len(off) > 6 else ''}）" if off else ""))
            out.append(f"  伸ばす音（{beat:g}拍より長い音）で、鳴っている間の半分以上がコードの音でないもの: {len(loff)}/{len(longs)}"
                       + (f"（{'・'.join(f'{at_(r, t, k, _chord_at(cells, t, bar)[0])}・{d:g}拍' for r, t, d, k, _s in loff[:6])}"
                          f"{' ほか' if len(loff) > 6 else ''}）" if loff else ""))
        # 繰り返しと変化（行ごとの形を字で並べる）
        rk, pk = [], []
        for r in rs:
            nb = int(round((r["length"] or (max(t + d for t, d, _k in r["notes"]) - r["start"])) / bar)) or 1
            by_bar_r: list[list] = [[] for _ in range(max(nb, 1))]
            by_bar_p: list[list] = [[] for _ in range(max(nb, 1))]
            for t, d, k in r["notes"]:
                bi = min(len(by_bar_r) - 1, int((t - r["start"]) // bar + 1e-9))
                by_bar_r[bi].append((round((t - r["start"]) * 4), round(d * 4)))
                by_bar_p[bi].append(k)
            rk.append((tuple(map(tuple, by_bar_r)), tuple(map(tuple, by_bar_r)), None))
            seq = [k for _t, _d, k in r["notes"]]
            pk.append((tuple(map(tuple, by_bar_p)), tuple(map(tuple, by_bar_p)),
                       tuple(b - a for a, b in zip(seq, seq[1:])) if len(seq) > 1 else None))
        out.append(f"  行ごとの形（{len(rs)}行）: リズム {' '.join(_shape_letters(rk))}／音 {' '.join(_shape_letters(pk))}")
        if sec in frame_by_sec:                             # 言葉を置ける枠とモーラ数・骨格の種類（2026-10-06）
            out.append(frame_by_sec[sec])
    lens = [len(r["notes"]) for r in rows]
    out.append(f"■全体: {len(allk)}音・音域 {name(min(allk))}〜{name(max(allk))}（{max(allk) - min(allk)}半音・鳴る高さ）・"
               f"フレーズ {len(lens)}本・1行の音の数 {sorted(set(lens))}")
    out += frame_all
    if len(pairs) >= 16:
        pairs.sort(key=lambda x: x[0])
        q = len(pairs) // 4
        out.append(f"  コードの音で鳴っている時間の割合（音ごとに、鳴っている間で数えた）: 長い音（長さの上位1/4・{pairs[-q][0]:g}拍以上）"
                   f"{sum(s for _d, s in pairs[-q:]) / q * 100:.0f}%・短い音（下位1/4・{pairs[q - 1][0]:g}拍以下）"
                   f"{sum(s for _d, s in pairs[:q]) / q * 100:.0f}%")
    return "\n".join(out)


def melody_context(design, chords, groove) -> str:
    """旋律の点検役の依頼文に足す**判断の文脈**（2026-10-03・組G1）——BPM・拍子・区間の調（転調を含む）・コード進行（区間ごとに1行）・
    グルーヴの区切り。以前は `ctx.brief` だけで、コードは理由の文しか無く、BPM・調・転調・区切りが無かった——点検役が「原因はコード」
    「テンポと合わない」を自分で確かめられなかった（点検役がいちばん効いた仕事は意図と実体の突き合わせで、ジャンル・テンポとの
    食い違いを自分で見つけた例が多い・`docs/plan/song_viewpoints_review.md`）。コード進行は終わり方（サビの手前の緊張・サビと曲の終わりの
    落ち着き）を見る材料にもなる（コードには点検役がいないので、旋律の点検役が見る）。
    """
    if design is None:
        return ""
    from song_design import BREAKS
    bar = bar_ticks(ctx.METER) / Q
    bpm = float(getattr(design, "bpm", 0) or 0)
    key_of: dict[str, str] = {}
    shift_of = (lambda _k: 0)
    try:
        from song_place import section_keys
        key_of, shift_of = section_keys(design)
    except ValueError:
        pass
    ch = ({c.kind: list(getattr(c, "chords_per_bar", []) or []) for c in getattr(chords, "sections", []) or []}
          if chords is not None else {})
    br = {g.kind: "".join(BREAKS.read(g.breaks)) for g in (getattr(groove, "sections", None) or [])}
    sung_word = "主旋律" if ctx.FORM == "instrumental" else "歌"
    out = ["【テンポ・拍子・区間の調・コード進行・区切り（設計・コード・グルーヴの工程が決めたもの）】",
           f"BPM {bpm:g}・拍子 {ctx.METER}" + (f"（1小節 {bar * 60 / bpm:.2f}秒）" if bpm > 0 else ""),
           "コードはコードの工程が書いたまま（C 基準＝旋律の ABC と同じ空間。鳴る調へは区間の移調量だけ移る）。`|` は小節の区切り。"
           "区切りは行ごとに1文字（S＝文が終わる・M＝息を入れる・C＝次の行へそのまま続く）"]
    for s in design.sections:
        k = s.kind
        attrs = [sung_word if s.has_vocal else f"{sung_word}なし",
                 f"強さ{getattr(s, 'intensity', '?')}", f"{getattr(s, 'bars', '?')}小節"]
        if k in key_of:
            attrs.append(f"調 {key_of[k]}（{shift_of(k):+d}半音）")
        if br.get(k):
            attrs.append(f"区切り {br[k]}")
        out.append(f"- {k}（{'・'.join(attrs)}）: " + (" | ".join(ch[k]) if ch.get(k) else "（コードなし）"))
    return "\n".join(out)


def run_melody(st) -> None:
    """**旋律の工程**（2026-09-30 に `song_compose._compose_once_body` から切り出した）。

    読む: `song.theme`・`song.design`・`song.chords`・`song.groove`・`st.ch_map`。
    書く: `song.melody`・`st.sung`（歌のある区間）・`st.top_c`（旋律が書く空間での上限。詞の譜割りが読む）。
    """
    song, verbose = st.song, st.verbose
    theme, design, chords, groove = song.theme, song.design, song.chords, song.groove
    # ① 旋律を先に（歌詞なし）。**これまでの判断を全部渡す**——工程を分けた以上、
    #    前段の意図が届かないと各段が独立に判断してしまう
    #    **セクションは名前でなく `has_vocal` で判定する**（名前を intro/verse/chorus に
    #    限定していたため、Cメロも間奏も作れなかった。2026-08-21 是正）
    #    ※旋律が書く区間名は設計の綴りと完全一致だけ（違えば `MelodyStep` の設計との照合が所見にして直させる・2026-10-02）
    sung = [s for s in design.sections if s.has_vocal]
    _top = top_note_of(theme, design)                  # ★この曲の最高音（設計 or 構想から）
    # ★★**旋律に渡す上限は「移調する前の空間」の数字にする**（2026-09-12）★★
    #
    #   旋律もコードも **C メジャー基準**で書かせ、楽譜を組むときに調のぶんだけ移調する。
    #   なのに上限だけ**移調後の数字（設計の `top_note`）をそのまま渡していた**。
    #
    #   > 実際に起きたこと（『青色へと』2026-09-12）: 構想「声は高め」→ 設計 `top_note=85`
    #   > → 旋律「サビ冒頭から最高音へ跳躍して解放」。旋律は**きっちり 85 まで書いた**。
    #   > ところが調 Bm ぶん **+2** されて 87 になり、下のオクターブ補正が
    #   > **曲全体を1オクターブ下げた**。実体は**最高音 75・最低音 57**
    #   > （ミクの快適域 58 を割り、`score_validate` が所見を出した）。
    #   > **3工程かけて積み上げた「サビで解放する」が、最後の座標決めで 10半音つぶれた。**
    #
    #   ★**意図は届いていた。覆したのはプログラムの座標決めだった**
    #   （型は `docs/dev/song_quality_premises.md` §7）。
    #   ★**転調するぶんも先に見込む**（2026-09-12）。設計は「最後のサビでもう一段上げる」
    #   という判断を `modulations` で書ける。基準の調だけで上限を出すと、
    #   **その転調ぶんがまるごとはみ出す**——実際に A→B（+2）の曲で **key=87**（上限86）が出た。
    #   上限は**いちばん高くなる区間**に合わせる（他の区間はそのぶん低く始まる。
    #   「最後に上げる」と決めたのだから、それが正しい）。
    _shifts = [key_shift(design.key)] + [key_shift(m.key) for m in design.modulations]
    _shift0 = max(_shifts)
    # ★★**上限も下限も、鳴る高さから書く空間へ同じだけ移す**（2026-09-27 監査）★★
    #   以前は上限だけ移し（しかも 70 を割ると黙って 70 へ上げ、低めと決めた曲が最大5半音高くなりえた）、
    #   下限（60）は**移さずに書く空間で比べていた**。A の曲（+9）では「鳴る高さで 69 より下は低すぎる」と
    #   旋律に返していた——下限の決め（ミクの快適域 58 前後）とまるで違う。上限は最も高く鳴る区間、
    #   下限は最も低く鳴る区間に合わせる（書いた音はその区間で最も高く／低く鳴るので）
    _top_c = _top - _shift0                             # 旋律が書く空間での上限
    _bars = {s.kind: int(getattr(s, "bars", 0) or 0) for s in sung}
    # 鳴る高さの下限 → 書く空間。下限は主旋律の下端 `MELODY_LO`（58）に揃えた（2026-09-27 本人判断。以前は 60 で、鳴る 59 を誤報していた）。
    #   2026-10-05 に声部の推奨音域の下端 `SING_LO` を D3（50）へ広げたが、主旋律は歌いやすい帯域を保つので別の値にした
    floor = (62 if theme.genre == "children" else MELODY_LO) - min(_shifts)
    mel_prompt = melody_prompt(theme, design, chords, groove, sung, st.ch_map, _top_c, floor)
    # ★★**旋律は1案で作り、点検役を毎回呼んで、この工程だけのやり直しもできる**（2026-09-30・本人「一旦 A にしようか」）★★
    #   以前は3案を同時に作って選抜役が選んでいた（2026-08-22〜）。記録を調べると、選抜役は旋律の本文（ABC）を読まず
    #   数字の札だけで選び、選んだ旋律が点検役に破綻と判定されても、旋律の関門には自工程のやり直しが無く進んだ。
    #   3案は工程費の約45〜49%。「方針と違う旋律」は内容の判断なので、点検役が本文と区間ごとの事実を読んで決める。
    #   形は設計・詞・伴奏の関門と同じ（上限 `MAX_REDO`）。札の数字（リズム・和音）は曲の記録へ（`melody_numbers`）。
    melody, redo_note = None, ""
    for attempt in range(MAX_REDO + 1):
        _h0 = len(song.history)                            # この回に積んだ記録だけを所見にする（前の回の記録を持ち越さない）
        # ★やり直しでは【前回の指摘】と「`response` に指摘ごとの対応を書く」を足す（2026-10-04・組P2a）。点検役は対応と変わった所を見る
        retry = "" if attempt == 0 else (
            f"\n\n★**前回の旋律は作り直しになりました。**\n{redo_ask(redo_note)}\n"
            f"前回の旋律:\n{(melody.sketch or '')[:6000]}")
        _before = melody                                   # 前の回の稿（点検役に変わった所を並べる・組P2a）
        melody = call_melody(mel_prompt + retry, "旋律", groove, design.bpm, _bars)

        # ★★**検査して、旋律の工程に差分で直させる**（2026-09-27・`song_fix`）★★
        #   行の長さ・小節の長さ・区間名・読めない記法・最高音・最低音を見て、所見があれば旋律へ返す。
        #   以前は機械が直していた（行の伸縮・最高音の1オクターブ下げ・最低音で選抜を捨てて作り直し・区間名の当て推量）。
        #   最低音の下限: 幅で突き返すと LLM は高い方を削る（81.5→76）ので、**下限だけを見る**（2026-09-01 からの判断）
        #   ★直らなければ旋律の工程で止める（`StepUnfixed`・2026-09-30）。以前は残った所見を点検役へ渡して進んでいた
        melody, _ = fix_melody(melody, song, groove, design.bpm, _bars, sung, _top_c, floor, verbose=verbose)

        # ※上限（HARD_TOP）を超えた音を機械が1オクターブ下げる処理は 2026-09-28 に外した（組み立ては一切補正しない）。
        #   旋律の輪郭（跳躍の向き）を変える補正だった。最高音は旋律の工程の検査（`MelodyStep`）が見て直させ、直らなければ旋律の工程で止める
        # **歌が1音も無い**まま先へ進めない。実際に phrases が空のまま vpr まで出来上がり、
        # 自己点検も通ってしまった（『零秒の衝動』0音・2026-08-21）。reason だけ返って
        # 中身が空、という壊れ方をするので、必ずここで止める
        # ※2026-09-30（判-4）から、行の無い旋律は形式の検査（歌のある区間が全部あるか）が差分で直させ、直らなければ止まる。
        #   ここへ来るのは、行はあるが全部が休み（`z`）だけの旋律
        if not any(p.notes for p in melody.phrases):
            if verbose:
                print("      ↻ 旋律が空（フレーズ0）。作り直し")
            # ★**やり直しの指示も記法と一緒に移す**（2026-09-21）。ABC へ移したあとも
            #   ここが旧フィールド名（`onsets`・`pitches`）を言い続けていた。
            #   **いちばん効いてほしい場面**——既に一度失敗している再試行——で、
            #   スキーマに無い名前を指示することになる。
            melody = call_melody(mel_prompt + (
                "\n\n【作り直し】前回は **行が空**でした。理由の文章だけでなく、"
                "**必ず lines に実際の行を入れてください**（各行に section と abc）。"),
                "旋律(空)", groove, design.bpm, _bars)
            # ★作り直した旋律にも同じ検査と差分の直しを掛ける（2026-10-03・再レビュー）。以前はここで作り直した旋律は検査を通らず、
            #   行の長さ・区間名・音域の違う旋律がそのまま点検役と後の工程へ進めた（直らなければ旋律の工程で止まる）
            melody, _ = fix_melody(melody, song, groove, design.bpm, _bars, sung, _top_c, floor, verbose=verbose)
            if not any(p.notes for p in melody.phrases):
                raise RuntimeError("旋律が生成されませんでした（phrases が空）")

        # ★**リズムも測って作り直す**（2026-09-06）。音域と同じ理由——
        #   譜割りの指示を秒基準へ書き換えても **拍頭率 71% → 53%** までしか動かず、
        #   帯（12〜33%）には遠かった。**数値目標を言葉で守らせるより、測って突き返すほうが速い**。
        #   1回だけ。ここは曲の骨なので、直らなければ所見にして先へ送る（関門が判断する）。
        #   ★**採否は件数でなく「帯からの総距離」で決める**（2026-09-06）。件数で比べたら
        #   「拍頭は直したが、ひと息を 17秒にした」という取引を**改善として受け入れた**。
        #   1つ直して1つ壊すと件数は同じでも、距離は正しく悪化する。
        #   **元の案も候補に残す**——作り直しが必ず良くなるとは限らない。
        # ★★**指標でやり直させるのをやめた**（2026-09-09・本人指示）★★
        #
        # ここには「帯からの総距離が縮むまで旋律を作り直す」ループがあった。**それが誤りだった。**
        #
        # | 分かったこと | 実測 |
        # |---|---|
        # | 帯（p25〜p75）は**定義上ヒット曲の半分を外す** | 全帯に収まるヒット曲は **0/909曲**・平均 5.5/10項目が外れる |
        # | 中央値は「正解」でなく**ひとつの型（40%）の位置** | 他の型からは 1.7〜3.0 離れている |
        # | 根拠にした POP909 は**中国のポップス**だった | J-POP は拍頭率 0.19（中国 0.12）・密度 1.71（1.56）・伸ばし 0.80秒（1.05） |
        #
        # つまりこのループは「良くする」のではなく、**全軸を同時に中央へ寄せて
        # 残り60%の作り方を排除する**動作だった。実際、直すたびに別の所が壊れ続けた
        # （休符率を直すとひと息が壊れ、グリッドに乗せると拍頭率が上がり、伸ばしのばらつきが消えた）。
        #
        # 規約どおり **指標は検算（出荷判定）に使い、生成の駆動には使わない**
        # （`.claude/skills/miku-dev/details/coding/generation_design.md`）。
        # 測るのはやめない——`inspect_breath` が所見として出し、**人が読む**。
        # 経緯: `docs/plan/song_expression_map.md`

        # ★**息ができる旋律か**を測って関門へ渡す（2026-09-06 新設）。
        #   旋律には関門が無く、音域の幅しか見ていなかった。本人の指摘
        #   「のびのび歌えてない。詰め込んでる感じ」を追ったら、**休みが 13.6%**（ヒット曲は 31〜44%）で
        #   **6.9秒続けて歌う**箇所があった。1音の長さは普通なので、**間の不足は長さでは見つからない**。
        #   **判定はしない**——切迫感を狙って詰めることはあるので、所見にして LLM に渡す。
        _breath_notes: list[str] = []                      # 記録だけ（点検役には渡さない・2026-09-29）
        # ★区切りの記号は**書いた行**に当てる（休みだけの行はフレーズにならない・2026-10-02）。行は `_written_lines`
        breath = inspect_breath(melody, design.bpm, groove, lines=_written_lines(melody),
                                record_only=_breath_notes)   # ★C の宣言を読んで判定する
        # ★**リズムと和音の数字も記録だけ**（2026-09-30）。以前は旋律3案の選抜の札に載せていた（`melody_numbers`）。
        #   数字は `reason` に、帯の外に出た項目だけ `warnings` に（毎曲「うまくいかなかった工程」に並ばないように）
        _numbers = melody_numbers(melody, design.bpm, chords)
        if _breath_notes or _numbers:
            song.record(StepRecord(step="melody_rhythm",
                                   reason="旋律のリズムの数字（記録だけ・点検役には渡さない）"
                                          + "".join(f"／{x}" for x in _numbers),
                                   warnings=_breath_notes))
        # ★**区間の頭から歌い出しまでが長い**（2小節超）——外した「位置合わせの点検」が持っていた観点のうち、
        #   ほかの検査が見ていなかった1つをここへ移した（2026-09-29・本人「あなたの方で持っている観点は捨てないでね」）。
        #   意図した「間」でありうるので断じず、点検役の材料にする
        breath = list(breath) + [Finding(what=w) for w in lead_gaps(melody)]
        # ※差分で直させても残った形式の所見は、2026-09-30 から点検役へ渡さない（旋律の工程で止まるので、ここへは来ない）
        # ★**機械がした変更も点検役に見せる**（2026-09-27 監査）。行の伸縮・上限超えの下げ・区間名の当て推量を
        #   知らずに判断していた
        breath += [Finding(what=f"（機械がした変更）{r.reason}") for r in song.history[_h0:] if r.step == "melody_machine"]
        if verbose:
            for f in breath:
                print(f"      [旋律の所見] {f.what}")
        # ★**点検役は毎回呼ぶ**（2026-09-30）。以前は所見があるときだけ呼び、方針と違う旋律は誰も見ていなかった。
        #   読ませるのは設計の方針（`brief`＝全体方針・歌い方の型・使う工夫・申し送り）と、旋律の本文（ABC）・区間ごとの事実
        # ★**文脈に BPM・拍子・区間の調・コード進行・区切りを足した**（`melody_context`・2026-10-03 組G1）。材料は鳴る高さ・行ごとの位置で数える
        v = ctx.judge("旋律", melody.reason, getattr(melody, "status", "ok") or "ok", getattr(melody, "trouble", "") or "",
                      breath, attempt=attempt, verbose=verbose,
                      redo_from_options=_redo_options("旋律"), before=_before, after=melody,
                      material=melody_facts(melody, sung, bpm=design.bpm, chords=chords, design=design,
                                            lines=_written_lines(melody), groove=groove)   # 枠とモーラ数・骨格（2026-10-06）
                      + "\n\n" + breath_facts(melody, design.bpm)        # ひと息は毎回の事実＝材料（所見にしない・2026-10-03）
                      + "\n\n【旋律（書かれたとおり）】\n" + cut_note(melody.sketch or "", 6000)    # 切ったら書く（組F2）
                      # ★テーマの方向の比べ方（2026-10-05）。旋律の点検役はテーマも受け持つ。比べた形跡が無いときだけ差し戻す（指示書）
                      + (f"\n\n{_tf}" if (_tf := theme_direction_facts(theme)) else ""),
                      context=f"曲『{theme.working_title}』（仮の曲名）（{ctx._GENRE_NAME[theme.genre]}）\n" + ctx.brief(theme, design, chords)
                      + "\n\n" + melody_context(design, chords, groove))
        song.record(StepRecord(step="melody_gate", reason=v.reason, notes=v.notes or "", guidance=v.guidance or "",
                               status="ok" if v.action == "accept" else "compromised",
                               warnings=[f.what for f in breath], request=v.action))
        # 原因がコード・グルーヴ・設計なら、そこから作り直す。退けた旋律を曲へ載せてから投げる（前回の出力として渡る・2026-10-04 組F1）
        _maybe_redo_from(v, "旋律", song, rejected={"melody": melody})
        if v.action == "accept":
            break
        if v.action == "stop":
            if verbose:
                print("  [関門] ★旋律が直せないので、このまま進めて記録に残す")
            break
        redo_note = v.guidance or "設計の方針（全体方針・歌い方の型・区間の強さ）に沿った旋律に直してください"

    st.log["melody"] = melody.reason
    song.melody = melody
    _rec(song, "melody", melody, verbose=verbose)
    lens = [len(p.notes) for p in melody.phrases]
    if verbose:
        print(f"  [4 旋律] {len(melody.phrases)}フレーズ {sum(lens)}音 音数={lens}\n      理由: {melody.reason}")
    st.sung, st.top_c = sung, _top_c
