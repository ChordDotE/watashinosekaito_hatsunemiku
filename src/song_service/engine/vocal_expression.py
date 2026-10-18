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

"""**歌の表情をフレーズ単位で決める工程**（2026-09-23 新設・案B）。

## なぜ要るか

いままで調声を付けていたのは `song_tuning.py` の**規則だけ**だった:

    1拍以上なら揺らす 0.8 ／ フレーズ頭はしゃくり 0.7 ／ 語尾は抜く ／ 山の区間は音量 0.68

**音の長さと位置しか見ていない。** 歌詞の意味にも旋律の形にも反応しないので、
「**ここは押し殺して**」「**ここは張り上げて**」が指示できなかった。

★**音ごとに LLM へ書かせるのは捨てた**（1曲300音・2026-09-09 に捨てた形へ戻る）。
  **フレーズ単位**にする——1曲 20〜40 フレーズなので現実的で、
  しかも「ここは押し殺して」は**もともとフレーズの話**。本人判断（案B）。

## なぜ伴奏のように工程を切り出さないのか（2026-09-23・本人の問いへの答え）

伴奏の奏法は**専用の工程**（`artic_plan.py`）に切り出したのに、歌は**この工程に混ぜている**。
一貫していないように見えるが、**理由が違う**:

| | 伴奏 | 歌 |
|---|---|---|
| 音の持ち方 | **1小節ぶんのパターン**（`steps`）を繰り返す | **1音ずつ独立**して並んでいる（`vocal_notes`）|
| だから | 「初めの1音だけ」が**構造的に書けない**→ **切り出すしかなかった** | **届かない場所が無い**→ 切り出す必要が構造から出てこない |

さらに**混ぜたほうが良い理由**がある——「このフレーズは**やさしく歌って、終わりを落とす**」は
**1つの解釈から出てくる一続きの判断**。分けると同じフレーズを2回別々に判断することになり、
「張り上げて」なのに「軽く落とす」のような食い違いが起きうる。

★**線引き**: **フレーズ単位で閉じているうちは1工程。音単位の判断が要るようになったら切り出す。**
  「伸ばしの途中でしゃくる」「この1音だけベンド」が要るようになったので、
  2026-09-23 に音程の崩し（旧 `ending`）を `vocal_artic.py`（歌の崩し）へ**切り出した**。
  ★指示書（`song_expression.md`）に「終わり方（ending）」の節が残っていて、**書いても届かない欄を
    決めさせていた**ので 2026-09-27 に消した（目録 表情-4）。ここの出力の型は strength / how / entry / reason だけ
    （強さと歌い回しは 2026-10-06 に1つの `how` から分けた）。

## 効くことを確かめてから作った

`aiExp` は**効かないものが混ざっていた**（ビブラートとフォルマントは V6 で無反応）。
そこで**先に measure した**（`dev_tools/scratch/s5065b362_vpr_aiexp.py`）:

| つまみ | 実測 | |
|---|---|---|
| `amplitude` | 0.0 で -31.3dB / 1.0 で -18.0dB | **効く（13.3dB 差）** |
| `amplitude_attack` | 頭が -8.5 ↔ +6.3 | **効く（14.8dB 差）** |
| `amplitude_release` | 尻が -11.4 ↔ -4.0 | **効く（7.4dB 差）** |
| `pitch_transition_in/out` | 渡りが 100ms → 160ms | **効く** |
| `vibrato_depth` | → `note.vibrato` へ書けば効く（線形）| **効く** |

★**効かないものの上に機能を建てない。** ここに出ていないつまみは使わない。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from song_llm import pick_model                            # noqa: E402
from song_llm import run_sync as _llm_run                  # noqa: E402
from song_model import StepRecord, PatchStatus, StepStatus              # noqa: E402
from song_checks import JsonInt  # noqa: E402（LLM が書く数と真偽の欄の型・2026-10-02 判断5）
from song_prompts import prompt                            # noqa: E402
from song_schema import SongNote, VocalTuning              # noqa: E402
from song_tuning import phrases_by_gap, vibrato_fits       # noqa: E402
from song_usage import record as _rec_usage                # noqa: E402

Q = 480

#: 歌い方の名前（強さ `STRENGTHS`・歌い回し `DELIVERIES`）→ つまみの組み合わせ。**実測で効くと確かめたものだけ**を使う。
#: ★数値は起案値（聴いて詰める前提）。**LLM には名前だけ選ばせる**——
#:   「押し殺して」は程度の話だが、`amplitude=0.32` は人が言う言葉ではない。
#: ★★**2026-09-23 に幅を広げた。** 最初の値では**人が聴き分けられなかった**——
#:   伏せた4本の組分けを外した（崩しのほうは当てた）。測ると:
#:
#:   | | |
#:   |---|---|
#:   | 譜面 | 69/74音が変化（音量 0.45 の一択 → 0.22〜0.85 の7種）|
#:   | 音 | **2秒ごとの差が中央 1.3dB**。3dB を超えるのは 30区間中5つだけ |
#:
#:   ★**1dB の差は、並べて聴いても分からない。** しかも**フレーズ全体が一様に**
#:     大きい／小さいという変化なので、基準が無いと気づけない
#:     ——崩し（しゃくり・落とし）が**局所的な身振り**だから聴き分けられたのと対照的。
#:
#: ★**効きの大きさは測ってある**（`s5065b362_vpr_aiexp.py`）。**そちらへ寄せる**:
#:   `amplitude_attack` **14.8dB**（いちばん大きい）／`amplitude` 13.3dB /
#:   `amplitude_release` 7.4dB ／**ビブラート ±0.52半音**（今日いちばん聴き分けられた）。
#:   最初の版は**音量を主役にしていて、効きの弱いほうに寄せていた**。
#: ★★**2026-10-06 に「強さ」と「歌い回し」の2つの欄に分けた**（本人決定「歌い方と音量は別々に指定すべき」）。
#:   以前は1つの欄（`how`）に8つの名前が並び、`切って`（語尾を短く切る歌い回し）が音量 0.66・頭の出方 0.80、
#:   `伸ばしきる` が音量 0.62 を持っていた。LLM に見せる強さの順にこの2つは入っていなかったので、LLM は音量が上がることを
#:   知らずに選んでいた——『火花になって飛べ』v2 の Aメロで `押し殺して`(0.25) → `切って`(0.66) と隣り合い、本人が
#:   「Aメロ内で音量差がある」と指摘した。いまは**音量（`amplitude`・`amplitude_attack`）は強さだけ**、**語尾の抜け方と揺れ
#:   （`amplitude_release`・`vibrato_depth`）は歌い回しだけ**が持つ。値は以前の表から振り分けただけ（新しく詰めていない）:
#:     ・強さの名前（`ささやく`・`押し殺して`・`しっかり`・`張り上げて`）は音量側だけを残し、持っていた語尾側の値は外した
#:       （語尾は歌い回しに任せる。歌い回しが `ふつう` なら規則の語尾・揺れのまま）
#:     ・歌い回しの名前（`やさしく`・`伸ばしきる`・`切って`）は語尾側だけを残し、持っていた音量側の値は外した
#:       （`やさしく` の音量 0.42 は規則の 0.45 とほぼ同じなので、強さ `ふつう` と組めば以前の音量に近い）
#:   古い記録の写し方は `dev_tools/song_legacy/convert_record.py`（`_OLD_HOW`）。
STRENGTHS: dict[str, dict] = {
    "ささやく":   {"amplitude": 0.12, "amplitude_attack": 0.15},
    "押し殺して": {"amplitude": 0.25, "amplitude_attack": 0.20},
    "ふつう":     {},
    "しっかり":   {"amplitude": 0.70, "amplitude_attack": 0.72},
    "張り上げて": {"amplitude": 0.95, "amplitude_attack": 0.92},
}

#: 歌い回し → 語尾側のつまみ（`amplitude_release`・`vibrato_depth`）。**音量は変えない**（音量は強さだけ・2026-10-06）。
DELIVERIES: dict[str, dict] = {
    "ふつう":     {},
    "やさしく":   {"amplitude_release": 0.88, "vibrato_depth": 0.60},
    "伸ばしきる": {"amplitude_release": 0.95, "vibrato_depth": 0.95},
    "切って":     {"amplitude_release": 0.08, "vibrato_depth": 0.05},
}

#: 入りの表情（フレーズの頭の音だけに掛かる）
#: ★これは **V6 自前の渡り**（`pitch_transition_in`）を伸び縮みさせるもの。
#:   実測で 100ms（V6 の既定 0.5）→ 160ms（0.9）に伸びることを確かめてある（`docs/plan/accomp_articulation.md` 丁）。
#: ★★**`しゃくる` は目盛りの端（1.0）**（2026-09-30・出-12）。以前は 0.75 で、規則（`song_tuning.apply_tuning`）が
#:   **全フレーズの頭に 0.7** を付けているので差が 0.05（実測の傾きで約7ms）しか無く、選んでも効いていなかった
#:   （『孤独のまま』で9フレーズが選び、点検役への応答もこれに頼っていた）。規則の 0.7 は `ふつう` のときに残る値で、
#:   これは変えない（カバーなど歌い方の工程が無い曲も同じ規則で歌う）。`しゃくる` は規則より長く、`まっすぐ入る` は規則より短く——
#:   規則をはさんで両側へ離す。値と規則の関係は指示書（`song_expression.md`）にそのまま書いてある（機械が黙って上書きしない）。
#:   ※実測の傾き（0.4 で 60ms）で見積もると 規則 0.7≈130ms／`しゃくる` 1.0≈175ms／`まっすぐ入る` 0.15≈50ms。聴いて詰める前提
ENTRIES: dict[str, dict] = {
    "ふつう":       {},
    "しゃくる":     {"pitch_transition_in": 1.0},
    "まっすぐ入る": {"pitch_transition_in": 0.15},
}

# ★**歌の奏法（しゃくり・落とし・ベンド）はここでは決めない**（2026-09-23）。
#   一度 `ENDINGS`（フレーズの終わりを落とす）をここに持たせたが、**切り出した**
#   ——本人「**実際の記憶は、伸ばしの途中でしゃくるとか、この1音だけ変化させるとかある。
#   切り出してくれる**」。フレーズ単位では**音の途中**にも**特定の1音**にも届かない。
#   決めるのは `vocal_artic.py`（工程 5.8）。**ここは声の出し方だけ**を持つ。


#: 強さの階段（小さい順）。**隣り合うフレーズで何段まで飛んでよいか**を見るために使う。
#: ★**端どうしを並べると音量差が大きすぎて不自然**——本人の指摘（2026-09-23・
#:   「ささやく」と「張り上げて」を交互に当てた音を聴いて「**音量差が大きすぎる**」）。
#:   指示書に目安として書く（機械は検査しない・2026-09-28）。仕上げの点検役には段差を事実として見せる（`strength_jumps`）。
#: ★2026-10-06 から強さの欄そのものの並び（`ふつう` が真ん中＝規則の音量）。以前は `やさしく` が3段目で、
#:   `切って`・`伸ばしきる` は音量を持つのに階段に入っていなかった。
LOUDNESS = list(STRENGTHS)


def strength_jumps(plan: "ExpressionPlan | None") -> list[tuple[int, int, str, str, int]]:
    """隣り合うフレーズ（番号が続く2つ・どちらも書いたもの）の強さの段差 → (前の番号, 後の番号, 前の強さ, 後の強さ, 段差)。

    段差は `LOUDNESS` の段の数（符号つき・上がれば正）。**事実だけ**——良し悪しは決めない（対比を狙うことがある・2026-09-28）。
    仕上げの点検役の材料に出す（2026-10-06）。"""
    of = {p.no: p.strength for p in (getattr(plan, "phrases", None) or [])}
    return [(no, no + 1, of[no], of[no + 1], LOUDNESS.index(of[no + 1]) - LOUDNESS.index(of[no]))
            for no in sorted(of) if no + 1 in of]


#: 歌い回し `切って` の目安: **最後の音（伸ばし `-` を含めた最後の音節）がこの秒以上**のフレーズには使わないのが標準（2026-10-06・本人決定）。
#: ★発端:『火花になって飛べ』v3 フレーズ4「てのひらこおりつく」の最後の「く」が **2拍（BPM200 で 0.60秒）** で、
#:   `切って`（語尾の抜き 0.08）＋`押し殺して` を選んでいた。短い音なら鋭く切れるが、長い音では伸ばしている途中で音がすぐ落ち、
#:   本人が「語尾が小さい」と指摘した。**秒で持つ**（抜けは時間で効くので、拍だと BPM で目安が動く）。0.5 は実例の 0.60秒より
#:   少し手前に置いた目安で、聴いて測った境目ではない（揺れの条件 `VIBRATO_MIN_SEC` と同じ 0.5秒にそろえて覚えやすく）。
#:   **形式の検査にはしない**（音楽の判断）。指示書に標準として書き、仕上げの点検役には事実として長さを出す（`cut_endings`）。
CUT_LONG_SEC = 0.5


def ending_ticks(seg: list[SongNote]) -> int:
    """フレーズの**最後の音節の長さ**（tick）。最後の音と、その前に続く伸ばし（楽譜の歌詞 `-`）を1つの音節として足す。"""
    if not seg:
        return 0
    j = len(seg) - 1
    while j > 0 and seg[j].lyric == "-":
        j -= 1
    return sum(n.duration_tick for n in seg[j:])


def ending_text(seg: list[SongNote], tempo: float | None) -> tuple[str, float, float | None]:
    """最後の音節 → (歌詞, 拍, 秒)。テンポが分からなければ秒は None。"""
    j = len(seg) - 1
    while j > 0 and seg[j].lyric == "-":
        j -= 1
    beats = ending_ticks(seg) / Q
    return seg[j].lyric, beats, (beats * 60.0 / float(tempo) if tempo and tempo > 0 else None)


def cut_endings(notes: list[SongNote], plan: "ExpressionPlan | None",
                tempo: float | None) -> list[tuple[int, str, float, float | None]]:
    """歌い回し `切って` を選んだフレーズごとの最後の音節 → (番号, 歌詞, 拍, 秒)。**長い順**（目安 `CUT_LONG_SEC` を超えるものが先）。

    **事実だけ**——良し悪しは決めない（短く切る狙いで長い音を選ぶこともありうる）。仕上げの点検役の材料に出す（2026-10-06）。"""
    segs = phrases(notes) if notes else []
    out = []
    for ps in sorted(getattr(plan, "phrases", None) or [], key=lambda p: p.no):
        if ps.how != "切って" or not (1 <= ps.no <= len(segs)):
            continue
        s, e = segs[ps.no - 1]
        out.append((ps.no, *ending_text(notes[s:e], tempo)))
    return sorted(out, key=lambda r: -(r[3] if r[3] is not None else r[2]))


# ※隣り合うフレーズの強さの段差の検査（`MAX_JUMP`）は 2026-09-28 に外した（音楽の判断。指示書には目安として書いてある）

#: LLM が選べる名前（**選択肢の型**）。一覧に無い名前は**型の検証で差し戻る**（目録 表情-2・表情-3）。
#: ★2026-09-27 まで `how` / `entry` はただの文字列で、知らない `entry` は**黙って「ふつう」**になり、
#:   知らない `how` は当てはめで落ちていた（LLM には届かない）。
StrengthName = Literal[tuple(STRENGTHS)]                    # type: ignore[valid-type]
HowName = Literal[tuple(DELIVERIES)]                        # type: ignore[valid-type]
EntryName = Literal[tuple(ENTRIES)]                         # type: ignore[valid-type]


class PhraseStyle(BaseModel):
    """1フレーズの歌い方（強さ・歌い回し・入り方）。**番号は表のものをそのまま書き写す**。"""
    no: JsonInt = Field(ge=1, description="フレーズ番号（表の左端）。**表に無い番号は書かない**")
    # ★2026-10-06: 強さ（音量）と歌い回し（語尾・揺れ）を別の欄に（本人決定）。どちらも既定なし＝必ず書く
    strength: StrengthName = Field(description=f"強さ（音量）: {' / '.join(STRENGTHS)}（**必ず書く**。**この中から1つ**。規則の音量のままなら `ふつう`）")
    how: HowName = Field(description=f"歌い回し（語尾の抜け方・揺れ。音量は変えない）: {' / '.join(DELIVERIES)}（**必ず書く**。**この中から1つ**。変えないなら `ふつう`）")
    # ★2026-10-04（組N2）: 既定「ふつう」をやめて必ず書かせる（省く・「ふつう」と書くが同じ意味だった）
    entry: EntryName = Field(description=f"入り方: {' / '.join(ENTRIES)}（**必ず書く**。**この中から1つ**。特に変えないなら `ふつう`）")
    reason: str = Field(description="なぜそう歌うのか（歌詞の意味・曲の流れから）")

    @field_validator("strength", "how", mode="before")
    @classmethod
    def _strip_how(cls, v: Any) -> Any:
        """前後の空白だけ落とす（名前そのものは変えない）。"""
        return v.strip() if isinstance(v, str) else v

    @field_validator("entry", mode="before")
    @classmethod
    def _strip_entry(cls, v: Any) -> Any:
        """前後の空白だけ落とす（名前そのものは変えない）。省くのも型が差し戻す（2026-10-04・組N2。以前は省くと既定 `ふつう`）。
        ★2026-10-02（洗い出し）: 空 `""`・null は選択肢に無いので型が差し戻す。以前は「書いていない」として黙って `ふつう` にしていた
          （省く・空・null の3通りが同じ意味になる）。"""
        return v.strip() if isinstance(v, str) else v


class ExpressionPlan(StepStatus):
    reason: str = Field(description="この曲をどう歌い分けたか（全体の方針）")
    phrases: list[PhraseStyle] = Field(description="フレーズごとの歌い方（**必ず書く**。変えるフレーズが無ければ空の並び `[]`）")


_expressor: Agent[None, ExpressionPlan] = Agent(
    pick_model("HARMONY", "flash"), output_type=ExpressionPlan,
    system_prompt=prompt("song_expression"),
)


class ExpressionPatch(PatchStatus):
    """歌い方の直し（**差分**）。**直すフレーズだけ**を書く——書かなかったフレーズはいまのまま残る。"""
    # ※`reason` は読み手なし（記録にも依頼文にも出ない）。直す前に考えを書かせる欄として残す（2026-09-30・出-13）
    reason: str = Field(description="何をどう直したか（全体で一言）")
    phrases: list[PhraseStyle] = Field(default_factory=list, description=(
        "直すフレーズだけ（番号・強さ・歌い回し・入り方・理由を**そのフレーズ全体**で書き直す）。直さないフレーズは書かない"))
    drop: list[JsonInt] = Field(default_factory=list, description=(
        "指定ごと取り消すフレーズ番号（表に無い番号を書いてしまったとき等）。取り消したフレーズは規則の表情のまま"))


from song_fix import PatchNumbers, guard_patch           # noqa: E402（差分の検証は工程の形の1つ・2026-10-02）

_expression_fixer: Agent[None, ExpressionPatch] = guard_patch(Agent(
    pick_model("HARMONY", "flash"), output_type=ExpressionPatch,
    system_prompt=prompt("song_expression_fix"),
))


# ---- 検査と直し（2026-09-28・工程が自分の検査を持つ形。`song_fix.StepFixer`）------------------

from song_fix import AssemblyError, StepFixer               # noqa: E402
import song_checks as ck                                    # noqa: E402


def _plan_rows(plan: ExpressionPlan) -> str:
    return "\n".join(f"  {p.no}|{p.strength}|{p.how}|{p.entry}|{p.reason[:60]}"
                     for p in sorted(plan.phrases, key=lambda p: p.no)) or "  （なし）"


class ExpressionStep(PatchNumbers, StepFixer):
    """歌い方: 表の番号か・1フレーズに1つかは形式の検査（部品）。

    - 表に無いフレーズ番号（目録 表情-3。以前は当てはめで黙って落としていた）
    - 同じ番号を2回書いた（以前は後ろのものが黙って勝っていた）
    隣のフレーズの強さの段差（目録 表情-1）は音楽の判断なので 2026-09-28 に外した。
    知らない歌い方・入り方は型の検証で先に差し戻るので、ここには来ない。

    `tbl` は最初の依頼と同じ【フレーズ】の塊（見出しつき）。直させたら `StepRecord(step="expression_fix")`。
    """
    key = "expression"
    label = "歌い方"
    agent = _expression_fixer
    return_hint = "**直すフレーズだけ**を phrases に、取り消す番号を drop に入れてください（直さないフレーズは書かない）。"

    def __init__(self, n_phrases: int, tbl: str):
        self.n_phrases, self.tbl = n_phrases, tbl

    def check(self, plan: ExpressionPlan) -> list[str]:
        nos = [p.no for p in plan.phrases]
        uniq = list(dict.fromkeys(nos))
        out = ck.names_in(((f"フレーズ {no}", str(no)) for no in uniq),
                          [str(i) for i in range(1, self.n_phrases + 1)], "表のフレーズ番号")
        for no in uniq:
            out += ck.count_is(f"フレーズ {no}", nos.count(no), 1, "書いた数", then="1フレーズに1つだけ")
        # ※「隣のフレーズの強さの段差」は 2026-09-28 に外した（本人判断）。強さの急な変化は対比として狙うことがある
        #   ——音楽の判断なので形式としては直させない。LLM が書いた実例も無かった（検証用の道具が作ったもの）
        return out

    def show(self, plan: ExpressionPlan) -> str:
        return f"{self.tbl}\n\n【いまの歌い方】（番号|強さ|歌い回し|入り方|理由）\n{_plan_rows(plan)}"

    def patch_count(self, plan: ExpressionPlan) -> int:
        return self.n_phrases                               # 書き直す番号は表のフレーズ番号（いまの指定の数ではない）

    def patch_items(self, patch: ExpressionPatch) -> list[tuple[int, bool]]:
        return [(p.no, False) for p in patch.phrases or []]

    def patch_problem(self, plan: ExpressionPlan, patch: ExpressionPatch) -> str:
        """書き直す番号は表のフレーズ番号（1〜フレーズの数）、取り消す番号は【いまの歌い方】にある番号（表に無い番号を
        書いてしまった指定を取り消すのが drop の役目なので、表の外でもよい）。同じ番号を書き直しと取り消しの両方に書かない。"""
        out = [w for w in [super().patch_problem(plan, patch)] if w]
        have = {p.no for p in plan.phrases}
        if gone := sorted({no for no in patch.drop or [] if no not in have}):
            out.append(f"差分の番号: 取り消す番号 {'・'.join(map(str, gone))} は【いまの歌い方】に無い（取り消せるのは "
                       f"{' / '.join(map(str, sorted(have))) or 'なし'}）")
        if both := sorted({p.no for p in patch.phrases or []} & set(patch.drop or [])):
            out.append(f"差分の番号: {'・'.join(map(str, both))} を書き直しと取り消しの両方に書いた——どちらか1つにする")
        return "／".join(out)

    def apply(self, plan: ExpressionPlan, patch: ExpressionPatch) -> tuple[ExpressionPlan, str]:
        """差分を**そのまま**当てる（書き直したフレーズは同じ番号の指定を置き換え、drop の番号は取り消す）。
        ★表に無い番号・同じ番号を2回（書き直しと取り消しの両方も）は差分の検証（`song_fix.guard_patch`）が差し戻す（2026-10-02・洗い出し。
          以前は取り消しの表に無い番号を黙って無視し、書き直しの表に無い番号は当ててから次の検査で所見にしていた）。"""
        new = {p.no for p in (patch.phrases or [])}
        drop = set(patch.drop or [])
        kept = [p for p in plan.phrases if p.no not in new and p.no not in drop]
        phrases_ = sorted(kept + list(patch.phrases or []), key=lambda p: p.no)
        parts = []
        if why := self.patch_problem(plan, patch):
            parts.append(f"差分の検証を通らずに来た: {why}")   # 型を通らずに来た差分だけ（当て方は変えない・次の検査が所見にする）
        if new:
            parts.append(f"{len(new)}フレーズを書き直した（{'・'.join(map(str, sorted(new)))}）")
        if drop:
            parts.append(f"{len(drop)}フレーズを取り消した（{'・'.join(map(str, sorted(drop)))}）")
        note = "／".join(parts) or "直しなし"
        if getattr(patch, "trouble", ""):
            note += f"／直さなかった理由: {patch.trouble}"
        return plan.model_copy(update={"phrases": phrases_}), note


def phrases(notes: list[SongNote]) -> list[tuple[int, int]]:
    """フレーズの区切り（音の添字の範囲）。**切り方は `song_tuning` と同じものを使う**
    ——別に書くと、規則で付けた表情とフレーズの境目がずれる。"""
    head, _tail = phrases_by_gap(notes)
    starts = sorted(head) or [0]
    out = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(notes)
        if e > s:
            out.append((s, e))
    return out


def entries_of_heads(notes: list[SongNote], plan: "ExpressionPlan | None") -> dict[int, str]:
    """**フレーズの頭の音 → 歌い方が決めた入り方**（`ENTRIES` の名前）。歌の崩しの表に出す（2026-09-30・出-12）。

    歌い方が書かなかったフレーズ・歌い方が無い曲は `ふつう`（規則の渡りのまま）。フレーズの切り方は `phrases` と同じ。
    ★以前は歌の崩しがこれを知らず、歌い方が `しゃくる` にした頭の音へ `scoop` を重ねていた（『孤独のまま』21・57・81小節）。
    """
    segs = phrases(notes) if notes else []
    out = {s: "ふつう" for s, _e in segs}
    for ps in (getattr(plan, "phrases", None) or []):
        if 1 <= ps.no <= len(segs):
            out[segs[ps.no - 1][0]] = ps.entry
    return out


def table(notes: list[SongNote], bar_ticks: int, *,
          sec_of_note: list[str] | None = None,
          intensity_of: dict[str, int] | None = None,
          tempo: float | None = None,
          keys: list[tuple[int, str]] | None = None) -> str:
    """LLM に見せる表。**歌詞と、どこで歌うか、どれくらいの高さか、その区間の盛り上がり。**

    ★**音域は国際式の音名**（`音域A4-E5`・2026-10-01・判-D）。以前は MIDI の番号（`音域69-76`）で、ほかの表・LLM が書く高さと
      物差しが違った。綴りは曲の表と同じ `song_table.spell_pitch`（`keys`＝`[(開始tick, 調名)]` を渡すと区間の調の ♯／♭。
      高さは実音＝鳴る高さ）。

    ★**位置は `artic_plan.tick_key`、歌詞の伸ばしは `_`（`song_table._hold_shown`）**——ほかの表と同じ書き手（2026-10-01・ず-11）。
      以前は位置をここで別に綴り、伸ばしを楽譜の V6 の歌詞 `-` のまま見せていた。

    ★`tempo` を渡すと**揺れが効く音の数**（音の後半が 0.5秒以上ある音）を末尾に出す（2026-09-27・目録 表情-7）。
      歌い方の揺れはその音にしか付かないので、`伸ばしきる` を選んでも 0 のフレーズでは揺れない。

    ★**最後の音の長さ**（最後の音節＝伸ばしを含む。拍と、`tempo` があれば秒）を末尾に出す（2026-10-06）。`切って` は
      最後の音が短いフレーズに使うのが標準（`CUT_LONG_SEC`）で、LLM がそれを選べるように。

    ★**区間と盛り上がりの列**（2026-09-25）。設計の工程が区間ごとに決めた起伏（1〜5）で、
      **伴奏はこの数字で弾き分けている**。以前はこの表に無く、歌い方だけが歌詞の意味から
      決めていたので、伴奏が静かに弾く A メロを「張り上げる」と決めることがありえた。
      **同じ設計図を見れば、互いを待たなくても大筋は揃う**（どちらが先かに関係なく）。
    """
    from artic_plan import tick_key                          # noqa: E402（位置の書き方は奏法・歌の崩し・曲の表と同じ1つ）
    from song_table import _key_at, _hold_shown, spell_pitch  # noqa: E402（音名の綴り・伸ばしの印は曲の表と同じ1つ）
    lines = []
    for no, (s, e) in enumerate(phrases(notes), 1):
        seg = notes[s:e]
        text = "".join(_hold_shown(n.lyric) for n in seg)
        key = _key_at(keys, seg[0].start_tick)
        lo, hi = (spell_pitch(min(n.key for n in seg), key), spell_pitch(max(n.key for n in seg), key))
        beats = sum(n.duration_tick for n in seg) / Q
        sec = ""
        if sec_of_note and s < len(sec_of_note):
            kind = sec_of_note[s]
            lvl = (intensity_of or {}).get(kind)
            sec = f"{kind}:盛り上がり{lvl}/5|" if lvl is not None else f"{kind}|"
        vib = (f"|揺れ{sum(1 for n in seg if vibrato_fits(n.duration_tick, tempo))}音" if tempo else "")
        _ly, end_b, end_s = ending_text(seg, tempo)
        end = f"|終わりの音{end_b:.1f}拍" + (f"({end_s:.2f}秒)" if end_s is not None else "")
        lines.append(f"  {no}|{tick_key(seg[0].start_tick, bar_ticks)}|{sec}{text}|{len(seg)}音|{beats:.1f}拍|音域{lo}-{hi}{vib}{end}")
    return "\n".join(lines)


def apply_plan(notes: list[SongNote], plan: ExpressionPlan, *,
               tempo: float | None = None) -> tuple[int, list[str]]:
    """決まった歌い方を音へ落とす（**組み立て**）。**その場で書き換え**、(付けた音数, 所見) を返す。

    ★**規則の表情に重ねる。同じつまみは歌い方が上書きする**（2026-09-27 に説明を実体へ合わせた・目録 表情-9）。
      歌い方が持つのは 音量・頭の出方（強さ `STRENGTHS`）・語尾の抜き・揺れ（歌い回し `DELIVERIES`）と、
      頭の音の渡り（`ENTRIES`）。強さと歌い回しは別のつまみを持つので、重ねても互いを上書きしない（2026-10-06）。
      規則（`song_tuning.apply_tuning`）が付けたもののうち、**歌い方が持たないつまみ**（頭の音の渡り＝
      `ふつう` の入りのとき、など）は残る。`ふつう` は何も上書きしない＝規則の表情のまま。
      指示書（`song_expression.md`）に、規則の中身と上書きの関係をそのまま書いてある。
    ★**揺れは、揺れる部分（音の後半）が 0.5秒以上ある音にだけ付ける**（`tempo` を渡したとき。
      `song_tuning.vibrato_fits`・目録 表情-7）。指示書にも書いてある。
    ★表に無い番号・知らない歌い方は、検査（`ExpressionStep.check`）と型で先に返すので、ここへは来ない。
      来たら当てずに所見を返す。呼び側（`plan`）はそれを不具合として `song_fix.AssemblyError` で止める（2026-09-30）。
    """
    segs = phrases(notes)
    warns: list[str] = []
    n_set = 0
    for ps in sorted(plan.phrases, key=lambda p: p.no):
        i = ps.no - 1
        if not (0 <= i < len(segs)):
            warns.append(f"フレーズ {ps.no} は無い（全 {len(segs)} フレーズ）——当てなかった")
            continue
        strength, deliv = STRENGTHS.get(ps.strength), DELIVERIES.get(ps.how)
        if strength is None or deliv is None:
            warns.append(f"フレーズ {ps.no}: 知らない強さ『{ps.strength}』か歌い回し『{ps.how}』——当てなかった")
            continue
        style = {**strength, **deliv}                      # 持つつまみが重ならない（音量側と語尾側）
        entry = ENTRIES.get(ps.entry, {})
        s, e = segs[i]
        for j in range(s, e):
            base = notes[j].tuning.model_dump(exclude_none=True) if notes[j].tuning else {}
            add = dict(style)
            if "vibrato_depth" in add and not vibrato_fits(notes[j].duration_tick, tempo):
                add.pop("vibrato_depth")                   # 揺れが周りきらない短い音（上の注記）
            base.update(add)
            if j == s:                                     # 入りの表情は**頭の音だけ**
                base.update(entry)
            notes[j] = notes[j].model_copy(update={"tuning": VocalTuning(**base)})
            n_set += 1
    return n_set, warns


def plan(notes: list[SongNote], brief_text: str, bar_ticks: int,
         *, verbose: bool = True,
         sec_of_note: list[str] | None = None,
         intensity_of: dict[str, int] | None = None,
         tempo: float | None = None, song=None,
         keys: list[tuple[int, str]] | None = None) -> tuple[int, list[str], ExpressionPlan | None]:
    """**この工程の入口。** 歌詞とフレーズを見せて、歌い方を決めてもらう。

    戻り値の3つ目は**LLM が書いたもの（直させた後）**——呼び側が `Song.expression` へ
    しまうと保存され、再生と「続きから」が使える（2026-09-24）。

    ★**検査して、所見があれば同じ工程に差分で直させる**（2026-09-27・2026-09-28 から `ExpressionStep`）。
      直らなければ歌い方の工程で曲づくりを止める（`song_fix.StepUnfixed`・2026-09-30）。`song` を渡すと
      `StepRecord(step="expression_fix")` を記録する。組み立て（`apply_plan`）が当てられない所を返したら、
      検査と組み立てのずれ＝不具合なので `AssemblyError` で止める（以前は当てずに記録して進んでいた）。
      保存から戻した歌い方（再生・「続きから」）は直させない——直しの呼び出しは保存に無く、再生でも本物の LLM を呼んでしまう。
    `tempo`（BPM）は揺れの条件に使う（渡さないと短い音にも歌い方の揺れが付く）。
    """
    if not notes:
        return 0, [], None
    tbl = table(notes, bar_ticks, sec_of_note=sec_of_note, intensity_of=intensity_of, tempo=tempo, keys=keys)
    head = ("番号|小節:拍|区間:盛り上がり|歌詞|音数|長さ|音域" if sec_of_note
            else "番号|小節:拍|歌詞|音数|長さ|音域") + ("|揺れが効く音" if tempo else "") + "|最後の音の長さ"
    block = f"【フレーズ】（{head}）\n{tbl}"
    ask = f"{brief_text}\n\n{block}\n\nそれぞれ、どう歌いますか。"
    r = _llm_run(_expressor, ask, label="歌い方")
    _rec_usage(_expressor, r, "歌い方")
    got: ExpressionPlan = r.output
    step = ExpressionStep(len(phrases(notes)), block)
    got, _ = step.run(got, song, verbose=verbose)          # 直らなければ止まる（`StepUnfixed`）。保存から戻した歌い方は直させない
    warns: list[str] = [step.result.summary()] if step.result.rounds else []
    n_set, net = apply_plan(notes, got, tempo=tempo)
    if net:
        # ★表の番号・1フレーズ1つは検査が、歌い方の名前は型が保証している（直らなければ上で止まる）ので、ここへは来ない。
        #   来たら検査と組み立ての読み方がずれている不具合（2026-09-30。以前は当てずに記録して進んでいた）
        raise AssemblyError("歌い方の組み立てに、検査を通ったはずの表が来た: " + "／".join(net))
    if verbose:
        print(f"  [歌い方] {len(got.phrases)}フレーズ／{n_set}音に反映"
              f"（{got.reason[:70]}）")
        for line in warns[:5]:
            print(f"      {line}")
    return n_set, warns, got


def _self_report(out, count: str):
    """記録（`song_flow._rec`）へ渡す形: LLM の `reason`・`status`・`trouble` に、組み立ての件数を添える（2026-09-30・出-13）。

    歌い方・歌の崩しで共用。件数は `reason` の末尾に括弧で残す（以前の記録は件数だけだった）。
    """
    from types import SimpleNamespace
    reason = (getattr(out, "reason", "") or "").strip()
    return SimpleNamespace(reason=f"{reason}（{count}）" if reason else count,
                           status=getattr(out, "status", "ok") or "ok",
                           trouble=getattr(out, "trouble", "") or "")


def run_expression(st) -> None:
    """**曲づくりの工程 5.7（歌い方）**——新しい曲の流れ（`song_compose._compose_once_body`）から呼ぶ（2026-09-30 に本体から切り出した）。

    読む: 歌の組み立て（`st.notes`・`st.timeline`・`st.bar`）・`st.brief_text`・`song.design`。書く: `song.expression`（音符の歌い方も）。
    """
    from song_flow import NOT_SWALLOW, _rec                 # noqa: E402（止める・途中で見せる・差し戻しは握らない／記録の形）
    from song_table import key_regions                     # noqa: E402（区間の調＝表の音名の綴り）
    song, verbose, log = st.song, st.verbose, st.log
    design = song.design
    # ---- 5.7 歌い方（フレーズ単位・2026-09-23 新設）----
    #   ★上の規則は**音の長さと位置しか見ていない**ので、歌詞の意味に反応できない。
    #     「ここは押し殺して」「ここは張り上げて」を**フレーズ単位**で決める（本人判断・案B）。
    #     音ごとに書かせる案は捨てた——1曲300音で、2026-09-09 に捨てた形へ戻るため。
    #   ★**規則の上に重ねる**（消さない）。役割が違う——規則は「どの曲でも要ること」、
    #     こちらは「この曲・このフレーズならでは」。
    try:
        # ★**伴奏と同じ設計図（区間ごとの盛り上がり）を渡す**（2026-09-25）。区間は
        #   **この時点の音符から取り直す**——上で作った `sec_of_note` のあとに音符が
        #   足し引きされていると、番号がずれて別の区間の数字を見せてしまう。
        _expr_sec = st.timeline.sections_of(st.notes)
        _expr_int = {s.kind: s.intensity for s in design.sections}
        _n, _ew, song.expression = plan(st.notes, st.brief_text, st.bar, verbose=verbose,
                                        sec_of_note=_expr_sec, intensity_of=_expr_int,
                                        tempo=float(design.bpm), song=song,   # 差分で直させる記録・揺れの条件（段階3）
                                        keys=key_regions(st))   # 音域の音名の綴り（区間の調・2026-10-01・判-D）
        log["expression"] = f"{_n}音に歌い方を反映" + ("／" + "／".join(_ew) if _ew else "")
        # ★**LLM の reason・status・trouble を記録に入れる**（2026-09-30・出-13）。ほかの工程と同じ `_rec` の形。
        #   以前は「278音に歌い方を反映」という件数の文だけが残り、歌い方の方針も「妥協した」という申告も消えていた
        if song.expression is not None:
            _rec(song, "expression", _self_report(song.expression, f"{_n}音に反映"), warnings=_ew, verbose=verbose)
        else:                                               # 歌の音が無い（歌い方の LLM を呼んでいない）
            song.record(StepRecord(step="expression", reason=log["expression"], warnings=_ew))
    except NOT_SWALLOW:                                         # 形式の所見が直らない・組み立ての不具合は止める（握らない・2026-09-30）
        raise
    except Exception as e:                                  # 歌い方で曲全体を落とさない
        import traceback as _tb
        log["expression"] = f"（失敗: {type(e).__name__}: {e}）"
        song.record(StepRecord(step="expression", reason="", status="compromised",
                               warnings=[f"★歌い方を書けなかった: {type(e).__name__}: {e}",
                                         _tb.format_exc()[-600:]]))
        if verbose:
            print(f"  [5.7 歌い方] ★失敗: {type(e).__name__}: {e}")
