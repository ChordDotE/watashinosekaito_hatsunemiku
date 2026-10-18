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

"""**奏法を音ごとに決める工程**（2026-09-23 新設）。

本人の判断:
> 「**数値も毎回書かせた方がいいと思う。パートごとに一律かどうかも違うと思う。
>   なんなら初めの1音だけ深く落とすとか上げるとか、そういったこともあると思う。
>   やっぱり工程を1個切り出したうえで、この音はこの部分のこれだからこれぐらい落とす、
>   そういったものを書かせられないかな。工程は1個増えても構わない**」

## なぜ工程を分けるのか

最初は編曲工程（`song_arrange`）の `SectionPattern` に奏法の欄を足した。
だが **`pattern` は「1小節ぶんのパターン」**で、合う小節すべてに適用される。つまり
**「フレーズの初めの1音だけ深く落とす」が構造的に書けない**。
`articulation_bars`（どの小節に掛けるか）で粗く逃げたが、**音単位には届かなかった**。

そこで**楽譜が組み上がってから**、鳴る音を1つずつ見せて決めさせる。

★**これは声部（`harmony.voices_for`）で当たった形と同じ**——
  **表を見せて、LLM が位置と数値を書き、コードは照合して変換するだけ**。
  2026-09-23 に「こちらが隙間を均等割りしたら 22/38 音が格子から外れた」ので、
  **人間に近い判断を機械が代行しない**。

## 渡すもの

★**他の工程と同じ `brief`（曲の意図・コンセプト・マスターの指示の積み重ね）を渡す**
  （本人指摘「**全体としてどういう曲でどういう意図でなのか、
  そこら辺も加味しないと正しく判断できないんじゃないか**」）。
  `plan_harmony` も `arrange` も同じものを読んでいる。**奏法だけ蚊帳の外にしない。**

## 標準を渡して、外すなら理由を書かせる

本人「**標準でこれぐらい落ちる、と渡したうえで、違う値にするならその理由を考えさせれば
問題ない気がした**」。カタログが持つ標準値を**表に併記**し、書かれた数値で鳴らす。
★2026-10-04（本人「深さを書かないのは良くない。書き忘れたのか標準にしたのかが曖昧になる」）: 深さを持つ奏法は**深さを必ず書く**
（以前は書かなければ標準のまま）。`gliss`・`portamento` は深さを持たない（行き先は次の音）ので書かない。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_ai import Agent

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import articulation as artic                               # noqa: E402
import song_audio                                          # noqa: E402（小節の判定はここだけ）
from song_llm import pick_model                            # noqa: E402
from song_llm import run_sync as _llm_run                  # noqa: E402
from song_checks import JsonBool, JsonFloat, JsonInt  # noqa: E402（LLM が書く数と真偽の欄の型・2026-10-02 判断5）
from song_model import StepRecord, PatchStatus, StepStatus              # noqa: E402
from song_prompts import prompt                            # noqa: E402
from song_checks import POSITION, position_notation        # noqa: E402（位置 `小節:拍` の記法の型）
from song_checks import edge_ws_problem, strip_ascii_ws     # noqa: E402（半角の名前の前後の空白・§0-6・2026-10-03）
from song_table import _key_at, key_regions, spell_pitch   # noqa: E402（音名の綴り・曲の表と同じ1つ・2026-09-30）
from song_schema import NoteArtic, SongScore               # noqa: E402
from song_usage import record as _rec_usage                # noqa: E402
from timbre_map import gm_program                          # noqa: E402

_bar_match = song_audio._bar_match
Q = 480

#: 表に出す音の上限（全パートの合計）。**多すぎると読み飛ばされる**（費用も増える）。
#: ★**パートごとに配る**（2026-09-30・出-11）。以前は上から順に数えて400行で打ち切り、ギターのアルペジオ（576音）が使い切って、
#:   ベース・弦・彩りの対旋律は見出しと「以下省略」だけになった（「奏法は入れない」をギター以外の音を1つも見ずに決めた）。
MAX_ROWS = 400
#: 配った行数に収まらないパートで、続けて見せるひとかたまりの音の数（フレーズの前後が分かるように、飛び飛びにしない）
RUN = 8


def _shares(counts: list[int], budget: int) -> list[int]:
    """**上限の行数をパートに配る**（少ないパートは全部、残りを多いパートで等分＝水を注ぐ配り方）。合計は `budget` 以下。"""
    share = [0] * len(counts)
    left, todo = budget, sorted(range(len(counts)), key=lambda i: counts[i])
    for n, i in enumerate(todo):
        share[i] = min(counts[i], left // (len(todo) - n))
        left -= share[i]
    return share


def _pick(n: int, k: int) -> list[int]:
    """`n` 個の音から `k` 個を選ぶ番号。**曲の頭から終わりまで、`RUN` 個ずつのかたまりを等間隔に**置く（代表を満遍なく）。"""
    if k >= n:
        return list(range(n))
    run = max(1, min(RUN, k))
    m = max(1, k // run)                                   # かたまりの数
    starts = sorted({round(j * (n - run) / max(1, m - 1)) for j in range(m)} if m > 1 else {0})
    got: list[int] = []
    for s in starts:
        got += [i for i in range(s, min(n, s + run)) if not got or i > got[-1]]
    return got[:k]


#: ★**奏法の名前は型で閉じる**（2026-10-02・洗い出し。歌の崩しの `VocalName` と同じ形）。以前はただの文字列で、組み立て
#:   （`articulation.shape`）が大小・前後の空白を無視して読んでいた（`Fall`・`FALL` も通った）。そのパートにできるかは工程の検査が見る
ArticName = Literal[tuple(artic.names())]                    # type: ignore[valid-type]
ArticNameOrKeep = Literal[tuple(["", *artic.names()])]       # type: ignore[valid-type]  # 差分: 空＝変えない


def _strip(v):
    """前後の半角の空白とタブだけ落とす（仕様書 §0-6。名前そのものは変えない）。前後の全角の空白・改行などは型で差し戻す（2026-10-03）。"""
    if isinstance(v, str):
        if why := edge_ws_problem(v):
            raise ValueError(f"奏法の名前{why}")
        return strip_ascii_ws(v)
    return v


class ArticNote(BaseModel):
    """1つの音に掛ける奏法。**位置は表と同じ書き方**で。"""
    part: JsonInt = Field(description="表の左端の「P番号」（そのまま書き写す）")
    at: str = Field(description="鳴り始める位置。**表と同じ書き方**（`12:1.50` ＝12小節目・拍1.50。拍は小数2桁）。"
                                "**拍は 0 から数えます**")
    name: ArticName = Field(description="奏法の名前（表の『できる奏法』に載っているものだけ・小文字で表のとおり）")
    #: ★**0 以下は型で差し戻す**（2026-10-02・洗い出し H1）。以前は 0 を「カタログの基準の形」で鳴らし（楽器の倍率も掛けない
    #:   ＝弦のフォールが表の標準 3.1 ではなく 7 半音）、負の数は大きさとして読んでいた（仕様書 §15 は「大きさ」だけ）
    #: ★**深さを持つ奏法は必ず書く**（2026-10-04・本人「深さを書かないのは良くない。書き忘れたのか標準にしたのかが曖昧になる」）。
    #:   以前は空＝表の標準で、書き忘れと「標準を選んだ」が区別できなかった。`gliss`・`portamento` は深さを持たない（行き先は次の音）ので
    #:   書かない——この2つだけのために欄は空を許し、深さを持つ奏法で空なら型（`_depth_written`）が差し戻す
    depth_st: JsonFloat | None = Field(default=None, gt=0, description=(
        "**何半音動かすか**（0 より大きい数）。**`gliss`・`portamento` 以外は必ず書く**（空・省略は書き直しを頼まれます）。"
        "選ぶ目安は表のそのパートの『できる奏法と標準の深さ』（その楽器のふつう）。曲の性格で浅く・深くし、"
        "標準から外すなら reason に書く。\n"
        "**符号は要りません**（フォールは下、ドゥイットは上、と奏法が決めます。負の数は書けません）。"
        "`gliss`・`portamento` は行き先が次の音で深さを持たないので書きません（書くと所見で返します）"))
    reason: str = Field(description="なぜここでそうするのか。**標準から外すときは必ず理由を書く**")

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, v):
        return _strip(v)

    @model_validator(mode="after")
    def _depth_written(self):
        if self.depth_st is None and artic.has_depth(self.name):
            raise ValueError(f"{self.at} の {self.name} に深さ（depth_st）が無い——{self.name} は深さを必ず書く"
                             "（目安は表のそのパートの『できる奏法と標準の深さ』）")
        return self


class ArticPlan(StepStatus):
    reason: str = Field(description="この曲で奏法をどう使ったか（**使わない判断ならその理由**）")
    # ★必ず書く（2026-10-04・既定値をやめる）。入れないなら空の並び `[]` を書く（書き忘れと「入れない」を分ける）
    notes: list[ArticNote] = Field(description=(
        "掛ける音を並べる。**必ず書く**——入れない曲は空の並び `[]` を書き、理由を reason に。**全部の音に付ける必要はありません**"))


_articulator: Agent[None, ArticPlan] = Agent(
    pick_model("HARMONY", "flash"), output_type=ArticPlan,
    system_prompt=prompt("song_articulation"),
)


class ArticFixItem(BaseModel):
    """奏法の差分1つぶん（2026-09-27・`song_fix`）。**直す奏法だけ**を返させ、機械はそのまま当てる。"""
    no: JsonInt = Field(description="直す奏法の番号（依頼文の「いまの奏法」の番号・1始まり）")
    drop: JsonBool = Field(default=False, description="この奏法をやめるなら true（ほかの直す欄と一緒に書かない）")
    # ★P番号も直せるようにした（2026-09-29）。「P3 の 12:0.00 に音が無い」の所見は P番号の書き違いでも出るのに、
    #   差分に欄が無く、位置しか直せなかった（直しが空振りする）
    part: JsonInt | None = Field(default=None, description="P番号を直すなら新しい P番号（表の左端の『P番号』）。直さないなら空")
    at: str = Field(default="", description="位置を直すなら新しい位置（表と同じ `小節:拍`＝`12:1.50`・拍は 0 からの小数2桁）。直さないなら空")
    name: ArticNameOrKeep = Field(default="", description=(
        "奏法を替えるなら新しい名前（そのパートの『できる奏法』から）。替えないなら空。"
        "**深さを持つ奏法に替えるときは `depth_st` も必ず書く**。`gliss`・`portamento` に替えると深さは無くなる（書かない）。"
        "`gliss`・`portamento` に書いてしまった深さを消すときも、同じ名前をここに書く"))
    depth_st: JsonFloat | None = Field(default=None, gt=0, description=(
        f"深さを直すなら新しい値（0 より大きく **{artic.MAX_DEPTH:g}半音まで**。目安は表の『標準』）。直さないなら空"))
    # ※2026-10-02〜10-04 は深さを空（表の標準）に戻す欄 `depth_standard` があった。深さを必ず書く決まりにしたので外した
    #   （深さを直すときは数を書く。`gliss`・`portamento` の深さを消すのは、その名前を `name` に書く＝深さを持たない奏法にする）
    # ※`why` は読み手なし（記録にも依頼文にも出ない）。直す前に考えを書かせる欄として残す（2026-09-30・出-13）
    why: str = Field(default="", description="どの所見にどう応えたか（一言）")

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, v):
        return _strip(v)

    # ★**当てられない組み合わせは型が差し戻す**（2026-10-03・記B・仕様書 §15）。以前は drop が黙って勝ち（書いた直す欄は読まれない）。
    #   歌い方・伴奏の差分と同じ扱い。2026-10-04 から、深さを持つ奏法へ替えるのに深さが無い・深さを持たない奏法へ替えるのに深さがある、も
    @model_validator(mode="after")
    def _one_way(self):
        if self.drop and (both := [k for k, v in (("part", self.part is not None), ("at", strip_ascii_ws(self.at)),
                                                  ("name", self.name), ("depth_st", self.depth_st is not None)) if v]):
            raise ValueError(f"{self.no}番: drop と {'・'.join(both)} を一緒に書いた——やめるなら drop: true だけ、"
                             f"直すなら {'・'.join(both)} だけ")
        if self.name and artic.has_depth(self.name) and self.depth_st is None:
            raise ValueError(f"{self.no}番: 奏法を {self.name} に替えるのに深さ（depth_st）が無い——深さを持つ奏法に替えるときは"
                             "深さも書く（目安は表の『標準』）")
        if self.name and not artic.has_depth(self.name) and self.depth_st is not None:
            raise ValueError(f"{self.no}番: {self.name} は次の音へ向かう奏法で深さを持たない——depth_st は書かない")
        return self


class ArticFix(PatchStatus):
    """奏法の直し（差分）。**直さない奏法は書かない**——書かなかったものはいまのまま残る。"""
    # ※`reason` は読み手なし（記録にも依頼文にも出ない）。直す前に考えを書かせる欄として残す（2026-09-30・出-13）。
    #   `trouble` は直しの記録に入る（`ArticulationStep.apply`）
    reason: str = Field(description="何をどう直したか（全体で一言）")
    patches: list[ArticFixItem] = Field(default_factory=list, description="直す奏法だけ")


from song_fix import PatchNumbers, guard_patch           # noqa: E402（差分の検証は工程の形の1つ・2026-10-02）

_artic_fixer: Agent[None, ArticFix] = guard_patch(Agent(
    pick_model("HARMONY", "flash"), output_type=ArticFix,
    system_prompt=prompt("song_articulation"),
))


def _artic_fix_empty(p: ArticFixItem) -> bool:
    """drop も直す欄も無い差分（差分の番号の検証が差し戻す）。"""
    return not (p.drop or p.part is not None or strip_ascii_ws(p.at) or p.name or p.depth_st is not None)


def tick_key(tick: int, bar_ticks: int) -> str:
    """tick → **表に出す位置の書き方**（`12:1.50`＝12小節目・拍1.50。拍は 0 始まり・小数2桁）。"""
    bar, rest = divmod(int(tick), bar_ticks)
    return f"{bar + 1}:{rest / Q:.2f}"


def at_key(at: str) -> str | None:
    """書かれた位置 → **表と同じ書き方**（`12:1.50`。前後の半角の空白とタブだけ除く・§0-6）。読めなければ None。
    ★2026-10-01 から小数2桁でない拍（`12:1.5`）、2026-10-02 から途中に空白のある位置（`12 : 1.50`）は読めない（所見は位置の記法の型・§12）。

    ★**位置は表の書き方どうしで照らす**（2026-09-29）。以前は書かれた拍を tick へ直して（`round(拍×480)`）、
      音の tick と完全一致で照らしていた。表は拍を小数2桁で見せるので、3等分した位置（160tick＝`0.33`）は
      158tick になって一致せず、**表に載っている音を書き写しても「音が無い」**になった（歌の `~` で3等分した音・
      1小節に3つのコードで弾き直した音）。表と照合で同じ書き方（`tick_key`）を使えば、表に載った位置は必ず指せる。
    ★読み方は位置の記法の型（`song_checks.POSITION`・2026-09-29）。読めない位置の所見もその型が出す（工程の `written`）。
    """
    got = POSITION.read(at)
    if got is None:
        return None
    bar, beat = got
    return f"{bar}:{beat:.2f}"


def _mono(part) -> bool:
    """単音で鳴るパートか（**和音パートには奏法を付けない**）。"""
    return bool(part.steps) and max(len(s.tones) for s in part.steps) == 1


def near_note_text(at: str, notes: list[tuple[int, int, str]], bar_ticks: int, *, what: str = "音") -> str:
    """**書いた位置に音が無いとき、近い音を示す一文**（2026-10-04 組F3・3‑4）。`notes` は [(開始tick, 終わりtick, 見せ方)]。

    以前は「9:0.10 に歌の音が無い」だけで、前奏に置いたのか・音の途中を指したのか・どの音が近いのかを言わなかった。
    位置は表と同じ書き方（`tick_key`）。読めない位置・音の無い曲は空。"""
    got = POSITION.read(at)
    if got is None or not notes:
        return ""
    t = round((int(got[0]) - 1) * bar_ticks + float(got[1]) * Q)
    ns = sorted(notes)
    prev = next((n for n in reversed(ns) if n[0] <= t), None)
    nxt = next((n for n in ns if n[0] > t), None)
    show = lambda n: f"{tick_key(n[0], bar_ticks)}{n[2]}"     # noqa: E731
    if prev is None:
        return f"——ここは最初の{what}より前（前奏など）。最初の{what}は {show(nxt)}"
    if prev[0] < t < prev[1]:
        return (f"——{show(prev)} の{what}が鳴っている途中（掛けるのは{what}の頭の位置）"
                + (f"。次の{what}は {show(nxt)}" if nxt else ""))
    if nxt is None:
        return f"——ここは最後の{what}より後ろ（後奏など）。最後の{what}は {show(prev)}"
    return f"——近い{what}は 前 {show(prev)}／後 {show(nxt)}（{what}の無い休みの所）"


def _part_list(nums) -> str:
    """パート番号の並び → `P0〜P27`（続いていれば）／`P1・P3・P5`。"""
    xs = sorted(nums)
    if not xs:
        return "無し"
    if xs == list(range(xs[0], xs[-1] + 1)) and len(xs) > 2:
        return f"P{xs[0]}〜P{xs[-1]}"
    return "・".join(f"P{x}" for x in xs)


def playable(score: SongScore) -> list[tuple[int, object, list[tuple[int, int, int]]]]:
    """奏法を掛けられるパートと、**実際に鳴る音**を出す。

    戻り値は `(パート番号, パート, [(tick, 音高, 長さ), …])`。
    **`midi_out` と同じ組み立て規則**を使う——ここだけ別に書くと、
    **表に無い音へ奏法が付く**（あるいは逆）という、いちばん分かりにくいずれ方をする。
    """
    from song_audio import pitched_events                  # noqa: E402（置き方は1つだけ・2026-09-27）
    out = []
    for pi, part in enumerate(score.arrangement.parts):
        if part.drum_steps or part.role == "drums" or not _mono(part):
            continue
        if not artic.family_of(gm_program(part.timbre, part.role))[1]:
            continue                                       # 何もできない楽器（鍵盤など）
        # ★**鳴る音は `song_audio.pitched_events` から取る**（2026-09-27）。ここにも組み立てを書いていて、
        #   書いた高さ（`tone.key`）と1小節2和音の拍位置が `midi_out` と食い違いえた。
        notes = [(start, keys[0], dur) for start, dur, keys, _v in pitched_events(score, part) if len(keys) == 1]
        if notes:
            out.append((pi, part, sorted(notes)))
    # ★**主旋律の楽器も載せる**（2026-10-03・§3「奏法の表に主旋律を載せる」）。P番号は伴奏のパートの次（`lead_index`）。
    #   鳴る音は楽譜の `lead.notes` そのもの（`midi_out._lead_track` が書くのと同じ音）
    lead = score.lead
    if lead.notes:
        from midi_out import LEAD_TRACK                    # noqa: E402（トラック名は書き出しと同じ1つ）
        from song_schema import PartSpec                    # noqa: E402
        stand = PartSpec(role=LEAD_TRACK, bass=False, timbre=lead.timbre, gain=lead.gain)   # 表に出すための見出し（鳴らし方は持たない）
        if artic.family_of(gm_program(lead.timbre, LEAD_TRACK))[1]:
            out.append((lead_index(score), stand,
                        sorted((n.start_tick, n.key, n.duration_tick) for n in lead.notes)))
    return out


def lead_index(score: SongScore) -> int:
    """奏法の表での主旋律の楽器の P番号（伴奏のパートの次）。"""
    return len(score.arrangement.parts)


def apply_artics(score: SongScore, picked: list[NoteArtic]) -> None:
    """決めた奏法を楽譜へ書く。伴奏のパートの分は `arrangement.articulations`、主旋律の楽器の分は `lead.articulations`
    （伴奏のパートを外しても主旋律の奏法がずれないよう、別に持つ・`song_schema.LeadPart`）。"""
    li = lead_index(score)
    has_lead = not score.lead.is_empty()                    # 無い主旋律の楽器（中身が空）には書かない（空のまま・F1）
    score.arrangement.articulations = [a for a in picked if not (has_lead and a.part == li)]
    if has_lead:
        score.lead.articulations = [a for a in picked if a.part == li]


def pitch_name(key: int, tick: int, keys: list[tuple[int, str]] | None) -> str:
    """音高 → 表に出す音名（`A4`・♭ の調の区間なら `Bb4`）。奏法・歌の崩しの表で共用（2026-09-30・記-16 の続き）。

    ★**綴りは曲の表（`song_table.spell_pitch`）と同じ1つ**。以前はここと歌の崩しが ♯ だけの表を別々に持ち、F の曲の B♭ を
      `ラ#` と出していた（F の曲に調の外の音があるように見える）。`keys` は `[(開始tick, 調名)]`（`song_table.key_regions`）で、
      転調した区間はその調で綴る。無ければこれまでどおり ♯。高さは変えない（綴りだけ）。表の音高は LLM が書き返さない欄
      （書き返すのは P番号・位置・名前・深さ）なので、読み手の側は変わらない。
    ★**国際式の音名で出す**（2026-10-01・判-A/B。LLM が書く高さの記法と同じ）。以前は階名（`シ♭4`）
    """
    return spell_pitch(key, _key_at(keys, tick))


def table(score: SongScore, *, keys: list[tuple[int, str]] | None = None) -> str:
    """LLM に見せる表。**奏法を掛けられる音だけ**を並べる。

    ★**その楽器にできる奏法と、標準の深さも併記する**——
      渡さないと外れを書く（実機でベースに `fall` と書かれて弾かれた）。
      本人「標準でこれぐらい落ちる、と渡したうえで、違う値にするなら理由を」。
    ★**上限（`MAX_ROWS`）はパートごとに配る**（2026-09-30・出-11）。全パートの見出しは必ず出し、収まらないパートは
      曲の頭から終わりまで `RUN` 個ずつのかたまりを等間隔に選んで載せ、**省いた所に「ここに◯音あり、省略」と書く**
      （黙って切らない）。以前は上から400行で打ち切り、1パートが使い切っていた。
    ★`keys`（`[(開始tick, 調名)]`・`song_table.key_regions`）を渡すと、音高をその区間の調の綴り（♯／♭）で出す（`pitch_name`）。
    """
    bar_ticks = _bar_ticks(score)
    lines: list[str] = []
    found = playable(score)
    shares = _shares([len(ns) for _pi, _p, ns in found], MAX_ROWS)
    for (pi, part, notes), share in zip(found, shares):
        prog = gm_program(part.timbre, part.role)
        fam, allowed = artic.family_of(prog)
        std = " ".join(f"{a}={artic.standard_depth(a, prog) or '-'}"
                       for a in sorted(allowed))
        keep = _pick(len(notes), share)
        lines.append(f"\n■ P{pi} {part.role}（{fam}・{part.timbre}）／鳴る音 {len(notes)}個"
                     + (f"（**表に載せたのは {len(keep)}個**——多いので、曲の頭から終わりまで {RUN}音ずつ間を空けて抜き出した。"
                        "「省略」の所にも音がある）" if len(keep) < len(notes) else "")
                     + f"\n   できる奏法と標準の深さ（半音）: {std}")
        prev = -1
        for i in keep:
            if i > prev + 1:
                lines.append(f"   …（P{pi} はここに {i - prev - 1}音あり、省略）")
            tick, key, dur = notes[i]
            lines.append(f"   P{pi}|{tick_key(tick, bar_ticks)}|{pitch_name(key, tick, keys)}"
                         f"|{dur / Q:.2f}拍")
            prev = i
        if keep and prev < len(notes) - 1:
            lines.append(f"   …（P{pi} はここから後に {len(notes) - 1 - prev}音あり、省略）")
        elif not keep and notes:                            # パートが多すぎて1行も配れなかった（黙って消さない）
            lines.append(f"   …（P{pi} の {len(notes)}音は、表の上限 {MAX_ROWS}行のため全部省略）")
    return "\n".join(lines) if lines else "（奏法を掛けられるパートがありません）"


def _bar_ticks(score: SongScore) -> int:
    """1小節の tick（拍子の読み手は `song_design.bar_ticks` の1つ。2026-10-01 まではここに同じ計算の写しがあった）。"""
    from song_design import bar_ticks                      # noqa: E402（循環を避けてここで）
    return bar_ticks(score.time_signature or "4/4")


def _position(score: SongScore):
    """この曲の拍子を知っている位置の記法（`song_checks.position_notation`・2026-09-30・記-15）。拍が1小節の長さ以上なら所見。"""
    return position_notation(_bar_ticks(score) / Q)


def _check(out: ArticPlan, score: SongScore) -> list[tuple[int, str, bool]]:
    """奏法の検査。`(番号, 所見, 掛けられないか)`。**所見を返すだけで、直さない**（2026-09-27・目録の 伴奏-10）。

    以前は、楽器にできない奏法・12半音を超える深さ・同時に鳴る音の上の奏法を、MIDI の書き出し（`midi_out`）が
    **黙って落とすか切り詰めて**いた（所見は print だけで記録に残らなかった）。いまは先にここで照合し、奏法の工程へ返す。
    3つ目（掛けられないか）は 2026-09-28 からすべて True（深さの超過も切り詰めずに掛けない）。
    ★**読めない位置はここでは見ない**（2026-09-29）。記法の型（`_position`）が所見を出す（工程の `written`・`_unplayable`）。
      拍が1小節の長さ以上の位置（2026-09-30・記-15）も記法の所見なので、ここで「音が無い」を重ねない。
    """
    bt = _bar_ticks(score)
    pos = _position(score)
    parts = {pi: (p, ns) for pi, p, ns in playable(score)}
    out_: list[tuple[int, str, bool]] = []
    seen: dict[tuple[int, int], int] = {}
    for i, n in enumerate(out.notes, 1):
        key = at_key(n.at)
        if key is None or pos.problems(n.at):              # 読めない・拍が小節をはみ出す位置（所見は記法の型）
            continue
        got = parts.get(n.part)
        if got is None:
            # ★**パートそのものが無い**ことを先に言う（2026-10-04 組F3・3‑4）。以前は「P9 の 5:0.00 に音が無い」と位置の問題に見えた
            out_.append((i, f"P{n.part} の {n.at}: P{n.part} というパートが無い（奏法を掛けられるパートは {_part_list(parts)}）——"
                            "表に載っている P番号と位置をそのまま書き写す", True))
            continue
        hit = [x for x in got[1] if tick_key(x[0], bt) == key]   # 表と同じ書き方で照らす
        if not hit:
            near = near_note_text(n.at, [(x[0], x[0] + x[2], "") for x in got[1]], bt)   # 近い音（2026-10-04 組F3・3‑4）
            out_.append((i, f"P{n.part} の {n.at} に音が無い{near}（表に載っている P番号と位置をそのまま書き写す）", True))
            continue
        part, notes = got
        tick = hit[0][0]                                   # 指した音の tick（以降の照合はこれで）
        prog = gm_program(part.timbre, part.role)
        if artic.shape(n.name, prog) is None:
            out_.append((i, artic.why_not(n.name, prog), True))
            continue
        if (n.part, tick) in seen:
            out_.append((i, f"P{n.part} の {n.at} には {seen[(n.part, tick)]}番でもう奏法を書いた（1つの音に1つ）", True))
            continue
        seen[(n.part, tick)] = i
        # ★深すぎる奏法は**掛けない**（2026-09-28・組み立ては補正しない）。以前は所見にしたうえで 12半音に切り詰めて鳴らした
        if artic.too_deep(n.depth_st):
            out_.append((i, f"深さ {n.depth_st:g}半音は大きすぎる（{artic.MAX_DEPTH:g}半音まで。"
                            f"ベンドの幅の上限で、超えると鳴らせない）", True))
        # ★深さを持たない奏法（gliss・portamento）に書いた深さ・0 以下の深さ（2026-10-02・洗い出し）。以前は黙って無視して鳴らした。
        #   0 以下は型（`gt=0`）が先に返すので、ここへ来るのは差分で名前を gliss に替えた所など
        elif artic.depth_unusable(n.name, n.depth_st):
            out_.append((i, artic.depth_why(n.name, n.depth_st), True))
        _t, _k, dur = hit[0]
        # ★次の音へ滑る奏法（gliss・portamento）は、**次の音までの幅**がベンドの幅に入るか（2026-09-28）。
        #   入らなければ書き出し（`midi_out._bend_msgs`）は掛けない（以前は 12半音先で止まって鳴った）
        pts = artic.shape(n.name, prog)
        if pts and any(p.to_next for p in pts):
            nxt = next((k2 for t, k2, _d in notes if t > tick), None)
            if nxt is not None and abs(nxt - _k) > artic.MAX_DEPTH:
                out_.append((i, f"『{n.name}』は次の音へ滑る奏法だが、次の音まで {abs(nxt - _k)}半音あり、"
                                f"ベンドの幅（{artic.MAX_DEPTH:g}半音）に入らない", True))
        if any(t != tick and t < tick + dur and t + d > tick for t, _k2, d in notes):
            out_.append((i, "同じ時に同じパートの別の音が鳴っている（奏法はベンド＝チャンネル単位なので、"
                            "重なっている音まで一緒に曲がる）", True))
    return out_


from song_fix import AssemblyError, StepFixer               # noqa: E402


def _unplayable(out: ArticPlan, score: SongScore) -> dict[int, str]:
    """組み立ての照合で**掛けられない奏法**（番号→理由）。工程の検査（記法の型 `_position`＋`_check`）と同じ判定。

    ★2026-09-30: 直らなければ奏法の工程で止まる（`StepUnfixed`）ので、組み立て（`plan`）でここが空でないのは不具合。
      以前は「直しても残ったので掛けなかった」と記録して進んでいた（2026-09-28 までは深さだけ 12半音に切り詰めて鳴らしていた）。
    """
    bad: dict[int, str] = {}
    pos = _position(score)
    notation = [(i, why) for i, n in enumerate(out.notes, 1) for why in pos.problems(n.at)]
    # 深さを持つ奏法の空の深さ（2026-10-04。型が必ず書かせるので来ない。以前は空＝楽器の標準で鳴らした）
    no_depth = [(i, f"{n.name} に深さが無い") for i, n in enumerate(out.notes, 1)
                if n.depth_st is None and artic.has_depth(n.name)]
    for i, why in sorted(notation + no_depth + [(i, why) for i, why, _hard in _check(out, score)], key=lambda x: x[0]):
        bad.setdefault(i, why)
    return bad


class ArticulationStep(PatchNumbers, StepFixer):
    """奏法: 位置が読めるか・拍が1小節に収まるか（記法＝`written`・`song_checks.position_notation`）・その位置に音があるか・その楽器にできる奏法か・1音1つか・
    深さの上限・次の音へ滑る幅がベンドの幅に入るか・重なった音の上か（形式の検査。2026-09-27・目録の 伴奏-10）。

    ★**検査は組み立ての照合（`_check`）と同じもの**を使う——`plan` は直したあとも同じ `_check` と同じ記法の型で
      照合するので（`_unplayable`）、2本に分けると片方だけ直る。
    ★直らなければ奏法の工程で曲づくりを止める（`song_fix.StepUnfixed`・2026-09-30）。以前は残ったものを掛けずに
      「直しても残ったので掛けなかった」と記録して進んでいた。
    """
    key = "articulation"
    label = "奏法"
    agent = _artic_fixer
    return_hint = ("**直す奏法だけ**を patches に入れてください（番号で指し、P番号・位置・名前・深さのうち直すものだけ書く。"
                   "やめるなら drop）。直さない奏法は書かないでください。")

    def __init__(self, score: SongScore, tbl: str):
        self.score, self.tbl = score, tbl

    def written(self, out: ArticPlan):
        pos = _position(self.score)                        # 拍子を知っている位置の記法（拍の上限も見る・記-15）
        return ((f"{i}番", pos, n.at) for i, n in enumerate(out.notes, 1))

    def check(self, out: ArticPlan) -> list[str]:
        return [f"{i}番: {why}" for i, why, _hard in _check(out, self.score)]

    def show(self, out: ArticPlan) -> str:
        rows = "\n".join(f"{i}. P{n.part} {n.at} {n.name} 深さ={f'{n.depth_st:g}' if n.depth_st is not None else 'なし（次の音へ）'}"
                         f" — {n.reason}" for i, n in enumerate(out.notes, 1))
        return f"【鳴る音の一覧】（P番号|小節:拍|音高|長さ）\n{self.tbl}\n\n【いまの奏法】\n{rows}"

    def patch_count(self, out: ArticPlan) -> int:
        return len(out.notes)

    def patch_items(self, patch: ArticFix) -> list[tuple[int, bool]]:
        return [(p.no, _artic_fix_empty(p)) for p in patch.patches or []]

    def apply(self, out: ArticPlan, patch: ArticFix) -> tuple[ArticPlan, str]:
        """差分をそのまま当てる。**書かれていない奏法は変えない**（元の案は変えない）。"""
        notes = list(out.notes)
        gone: set[int] = set()
        done: list[str] = []
        items = list(getattr(patch, "patches", None) or [])
        # ★範囲外・重なった番号・中身の無い直しは、差分の検証（`song_fix.guard_patch`・`PatchNumbers`）が LLM へ差し戻すので来ない
        #   （2026-10-02・洗い出し）。型を通らずに来たら当てずに、当てなかったことを直しの記録に書く（以前は黙って捨てた）
        if why := self.patch_problem(out, patch):
            done.append(f"当てなかった: {why}")
        seen: set[int] = set()
        for p in items:
            if not 1 <= p.no <= len(notes) or p.no in seen or _artic_fix_empty(p):
                continue
            seen.add(p.no)
            if p.drop:
                gone.add(p.no)
                done.append(f"{p.no}番をやめた")
                continue
            upd: dict = {}
            if p.part is not None:
                upd["part"] = p.part
            if strip_ascii_ws(p.at):                       # 位置は半角の記法（前後は半角の空白とタブだけ・§0-6）
                upd["at"] = strip_ascii_ws(p.at)
            if p.name:
                upd["name"] = p.name
                if not artic.has_depth(p.name):            # 深さを持たない奏法（gliss・portamento）にした＝深さは無い（説明に書いてある）
                    upd["depth_st"] = None
            if p.depth_st is not None:
                upd["depth_st"] = p.depth_st
            notes[p.no - 1] = notes[p.no - 1].model_copy(update=upd)
            done.append(f"{p.no}番を直した")
        note = "／".join(done) if done else "直しなし"
        if getattr(patch, "trouble", ""):
            note += f"／直さなかった理由: {patch.trouble}"
        return out.model_copy(update={"notes": [n for i, n in enumerate(notes, 1) if i not in gone]}), note


def plan(score: SongScore, brief_text: str,
         *, verbose: bool = True, song=None,
         keys: list[tuple[int, str]] | None = None) -> tuple[list[NoteArtic], list[str], ArticPlan | None]:
    """**この工程の入口。** 楽譜と申し送りを見せて、音ごとの奏法を決めてもらう。

    戻り値は `(掛ける音の一覧, 所見, LLM が書いたものそのもの)`。**位置が表に無いものは
    落として所見に出す**——読めないものを黙って 0 にしない（曲頭に奏法が湧く）。
    3つ目は呼び側が `Song.articulation` へしまうと保存され、再生と「続きから」が
    使える（2026-09-24）。

    ★**検査して所見を返し、差分で直させる**（2026-09-27・`song_fix`・目録の 伴奏-10）。表に無い位置・その楽器に
      できない奏法・12半音を超える深さ・同時に鳴る音の上の奏法を、奏法の工程へ返す。直らなければ奏法の工程で
      曲づくりを止める（`song_fix.StepUnfixed`・2026-09-30。以前は掛けずに所見に出して進んでいた）。保存から戻した奏法は直させない。
      流れは `ArticulationStep`（2026-09-28）。`song` を渡すと `articulation_fix` を記録する。
    `keys`（区間の調・`song_table.key_regions`）は表の音高の綴りに使う。渡さなければ `song` の記録（`key_of`・`timeline`）から引く
    （曲づくりの途中はまだ記録に無いので、呼び側が組み立ての値から渡す・`run_articulation`）。
    """
    if keys is None and song is not None:
        keys = key_regions(song)
    tbl = table(score, keys=keys)
    if tbl.startswith("（"):
        return [], ["奏法を掛けられるパートが無い（単音のパートが無いか、鍵盤だけの編成）"], None
    bar_ticks = _bar_ticks(score)

    ask = (f"曲名『{score.title}』\n\n{brief_text}\n\n"
           f"【鳴る音の一覧】（P番号|小節:拍|音高|長さ）\n{tbl}\n\n"
           f"どの音に、どんな奏法を、どれくらいの深さで掛けますか。")
    r = _llm_run(_articulator, ask, label="奏法")
    _rec_usage(_articulator, r, "奏法")
    step = ArticulationStep(score, tbl)
    out, _ = step.run(r.output, song, verbose=verbose)     # 直らなければ止まる（`StepUnfixed`）。保存から戻した奏法は直させない
    warns: list[str] = [step.result.summary()] if step.result.rounds else []

    # ---- 書かれたものを照合する ----
    # ★検査と同じ照合なので、直らなければ上で止まっていて、ここで掛けられない奏法は無い。来たら不具合として止める（2026-09-30）
    if bad := _unplayable(out, score):
        raise AssemblyError("奏法の組み立てに、検査を通ったはずの表が来た: "
                            + "／".join(f"{i}番: {why}" for i, why in sorted(bad.items())))
    # 書かれた位置 → 指した音の tick（照合と同じく、表の書き方どうしで引く）
    tick_of = {(pi, tick_key(t, bar_ticks)): t for pi, _p, ns in playable(score) for t, _k, _d in ns}
    picked = [NoteArtic(part=n.part, tick=tick_of[(n.part, at_key(n.at))], name=strip_ascii_ws(n.name),
                        depth_st=n.depth_st, reason=n.reason) for n in out.notes]
    if verbose:
        print(f"  [奏法] {len(picked)}か所（{out.reason[:80]}）")
        for line in warns[:5]:
            print(f"      {line}")
    return picked, warns, out


def run_articulation(st) -> None:
    """**曲づくりの工程 8.7（奏法）**——新しい曲の流れ（`song_compose._compose_once_body`）から呼ぶ（2026-09-30 に本体から切り出した）。

    読む: 楽譜（`st.score`・彩りを合わせたあと）・`st.brief_text`。書く: `song.articulation`・楽譜の奏法。
    """
    from song_flow import NOT_SWALLOW, _rec                 # noqa: E402（止める・途中で見せる・差し戻しは握らない）
    song, verbose, log, score = st.song, st.verbose, st.log, st.score
    # ---- 8.7 奏法（音の中で高さが動くもの・2026-09-23 新設）----
    #   ★**楽譜が組み上がってから**でないと呼べない——**鳴る音1つずつ**に掛けるため。
    #     編曲工程のパターンに書く形も試したが、`steps` は1小節ぶんなので
    #     「フレーズの初めの1音だけ深く」が書けなかった（本人指摘）。
    #   ★**他の工程と同じ `brief` を渡す**（本人指摘「全体としてどういう曲でどういう意図か、
    #     そこら辺も加味しないと正しく判断できない」）。
    if score.arrangement.parts or not score.lead.is_empty():
        try:
            # グルーヴの土台（`st.groove_note`）は渡さない（2026-09-30・io レビュー）。ドラムとベースの型は音ごとの弾き方と関係が無く、
            #   グルーヴの出力は依存の木（`DEPS["奏法"]`）で依頼文に丸ごと届いている（二重だった）
            # 表の音高は区間の調で綴る（2026-09-30）。区間の調と時間軸は曲の記録に入る前（`song_compose._finish`）なので、組み立ての値から引く
            _a, _aw, song.articulation = plan(score, st.brief_text, verbose=verbose, song=song, keys=key_regions(st))
            apply_artics(score, _a)                         # 伴奏のパートの分と主旋律の楽器の分を分けて書く（2026-10-03）
            log["articulation"] = "／".join(_aw) if _aw else f"{len(_a)}か所"
            # ★**LLM の reason・status・trouble を記録に入れる**（2026-09-30・出-13）。以前は「0か所」のように件数だけで、
            #   「入れない」判断の理由も自己申告も残らなかった。件数は理由の頭に付ける。掛けられるパートが無くて LLM を
            #   呼ばなかったとき（`song.articulation` が None）は、その所見を理由にする
            if song.articulation is not None:
                _r = _rec(song, "articulation", song.articulation, warnings=_aw, verbose=verbose)
                _r.reason = f"{len(_a)}か所" + (f"——{_r.reason}" if _r.reason else "")
            else:
                song.record(StepRecord(step="articulation", reason=log["articulation"], warnings=_aw))
            # ★書き出し（`midi_out`）で鳴らせなかった所は**記録して進む**（2026-09-30 に残すと決めた）。チャンネルが足りない・
            #   和音パートの奏法・同じチャンネルで音量の違うパートが重なる・音源のキットで読めない など、形式の検査では
            #   保証できない書き出しと音源の限界で、組み立てでしか分からない（`midi_out.assembly_warnings`）
            from midi_out import assembly_warnings           # noqa: E402（ここでだけ使う）
            from song_fix import record_change
            for _w in assembly_warnings(score):
                record_change(song, "articulation", _w, "書き出しの組み立てで鳴らせなかった")
        except NOT_SWALLOW:                                     # 形式の所見が直らない・組み立ての不具合は止める（握らない・2026-09-30）
            raise
        except Exception as e:                              # 奏法で曲全体を落とさない
            import traceback as _tb
            log["articulation"] = f"（失敗: {type(e).__name__}: {e}）"
            song.record(StepRecord(step="articulation", reason="", status="compromised",
                                   warnings=[f"★奏法を書けなかった: {type(e).__name__}: {e}",
                                             _tb.format_exc()[-600:]]))
            if verbose:
                print(f"  [8.7 奏法] ★失敗: {type(e).__name__}: {e}")
