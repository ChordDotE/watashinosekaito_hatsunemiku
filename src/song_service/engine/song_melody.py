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

"""**旋律の型と読み方**——旋律の工程が受け取る形（ABC）と、下流へ渡す形（`BareMelody`）・その変換・1行の枠・音名。

2026-09-30 に `song_compose` から分けた。旋律の工程は3つのファイルに分かれる:

| ファイル | 中身 |
|---|---|
| `song_melody`（ここ） | 型（`BareNote`・`Phrase`・`BareMelody`・`AbcLine`・`AbcSketch`・差分の `AbcPatch`）・ABC からの変換・1行の枠（`_line_slots`）・区間名の揃え・音名 |
| `song_melody_prompt` | 旋律への依頼文（`melody_prompt`・グルーヴ・1行の長さ・ジャンルの実測） |
| `song_melody_step` | 旋律の工程（`run_melody`）・呼び出し（`call_melody`）・検査と直し（`MelodyStep`）・点検役の材料（`melody_facts`） |

依存の向きは ここ → 依頼文 → 工程（逆は無い）。
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from song_checks import JsonBool, JsonInt  # noqa: E402（LLM が書く数と真偽の欄の型・2026-10-02 判断5）
import song_context as ctx                                 # いまの曲の拍子（`ctx.METER`）
from song_design import units_per_bar
from song_model import PatchStatus, StepStatus


# ---- ① 旋律（歌詞なし）--------------------------------------------------------

class BareNote(BaseModel):
    # ※**内部の形**（ABC から機械が起こす・LLM には書かせない・見せない）。LLM に見せる高さは国際式の音名（`_note_name`）か ABC（2026-10-01）。
    #   後の工程へ渡すとき（前段の出力 `song_context.upstream_intents`・前回の出力 `asks_for`）は、この JSON を省いて
    #   【書かれたABC】だけを見せる（`song_context.UPSTREAM_OMIT`・仕様書 §17 ず-7）。ABC の無い古い記録だけ JSON のまま届く
    solfa: str = Field(description="内部の形: 固定ドの階名（ド/レ/ミ/ファ/ソ/ラ/シ・♯可）。C メジャー基準・移調前。機械が ABC から起こす")
    # ★**範囲を持たない**（2026-10-03・再レビュー）。2026-09-27 は 2〜7（その前は 4〜6 で、低い音を読み取り側が黙って丸めていた）。
    #   ABC の記法は `,` `'` をいくつでも書ける（`C,,,16`・`c'''16` は記法として正しい）ので、範囲があると旋律の検査（音域の所見）より先に
    #   受け取り（`call_melody` → `abc_sketch_to_melody`）がこの型の例外で落ちた。書いたとおりに受け、音域は旋律の検査が所見として返し
    #   （`MelodyStep.check` の最高音・最低音）、直らなければ旋律の工程で止まる
    octave: int = Field(default=4, description=(
        "内部の形: 国際式のオクターブの数（4＝中央のドのオクターブ・C4＝MIDI 60）。機械が ABC から起こす"))
    beats: float = Field(gt=0, description="長さ（拍）")
    # ★**上限を持たない**（2026-10-03・再レビュー）。2026-09-27 は 64拍（その前は 2拍で、区間の終わりの休みを前の音へ移すとき超えた分を捨て、
    #   区間が設計より短くなった＝目録の旋律-2）。1行の長さは枠（区間の小節数 ÷ 行数・区間は 32小節まで）で決まり、長い休みは記法として
    #   書ける（`C4 z260 |`）ので、上限があると旋律の検査より先にこの型の例外で落ちた。行の長さは旋律の検査が枠と照らす
    gap_after: float = Field(default=0.0, ge=0, description=(
        "この音のあとに置く**小さな休み**（拍）。**0 でよい場合がほとんど**だが、"
        "**言葉の切れ目・語尾を切る所・跳ねる所**に置くと歌が生きる。"
        "★**大きさは 0.25秒より短く**（拍へ換算すること）。"
        "**0.25秒以上あけたいならそれは息継ぎ**で、フレーズを分けて `rest_before` に置く。"
        "★ヒット曲909曲では、休みの**個数の89%が0.25秒未満のこの小休み**で、"
        "**歌唱時間の11.5%**を占める。これが無いと音符が数珠つなぎになり、"
        "**息継ぎ（フレーズ間）だけでは休符率が21%までしか上がらない**"))
    # ★**タイでまとめた続きの音の数**（2026-10-01・判-H）。旋律は ABC のタイ（`E2- | E2`）を1つの長い音にまとめて持つが、
    #   詞は ABC の歌詞行どおり書いた音ごとに印を書く（続きの音は `_`）。詞の印の数は `lyric_slots.slot_count`（1＋これ）
    ties: int = Field(default=0, ge=0, description="タイでこの音にまとめた続きの音の数（ABC で書いた音の数−1）。機械が書く")


class Phrase(BaseModel):
    section: str = Field(description=(
        "このフレーズが属するセクション。**下に示されたセクション名をそのまま書くこと**"
        "（`verse1` と `verse2` を両方 `verse` と書かない——1番と2番の区別が付かなくなります）"))
    rest_before: float = Field(default=0.0, ge=0, description="このフレーズの前に置く休符（拍）")
    notes: list[BareNote] = Field(description="フレーズの音符（歌詞はまだ付けない）")


class BareMelody(StepStatus):
    """**下流へ渡す旋律の形**（フレーズ・音符・長さ）。旋律の工程は ABC（`AbcSketch`）で受け取り、
    `abc_sketch_to_melody` でこの形へ起こす。保存もこの形で、再生・続きから作る・作り直しはこの形のまま戻る。

    ※LLM に直接この形（音価と隙間を数値）で書かせていたのは 2026-09-09 まで。そのあと枠記法（`onsets`・歌い出しの位置だけ）を
      経て、2026-09-20 に ABC（書いた音価のまま）へ移した。枠記法の型（`SketchLine`・`MelodySketch`）と ABC から枠へ戻す変換
      （`abc_sketch_to_frames`）・枠の小節線の検査（`bar_line_issues`）は 2026-09-30 に外した。
    """
    reason: str = Field(description="旋律の設計（山をどこに置き、どう対比を作ったか）")
    sketch: str = Field(description="旋律の輪郭の一言と、LLM が書いた ABC の記録（`sketch_record`・【書かれたABC】）")
    phrases: list[Phrase] = Field(description="フレーズの並び（曲頭から順に）")


class AbcLine(BaseModel):
    """1行ぶんの旋律を **ABC 記法**で受け取る（2026-09-20 に枠記法から移行）。"""
    section: str = Field(description=(
        "この行が属するセクション。**示されたセクション名をそのまま書くこと**"))
    abc: str = Field(description=(
        "この行の旋律を **ABC 記法**で書く。`L:1/16`（**1単位＝16分音符**）。"
        "**拍子（`M:`）と1小節の単位数は、依頼文の「この曲の拍子」のとおり**（4/4 なら16単位・3/4 なら12単位）。\n"
        "音名は `C`＝中央ハ、`,` で1オクターブ下（`C,`）、小文字で1オクターブ上（`c`）、`'` でさらに上（`c'`）。"
        "**`,` は大文字に・`'` は小文字にだけ**付ける（`c,` ではなく `C`）。"
        "`^`＝♯ `_`＝♭ `=`＝ナチュラル（音名の**前**に**1つだけ**付ける。`^^C` ではなく `D`）。"
        "音の長さは音名のうしろに単位数（`C4`＝1拍・1単位は数字を付けず `C`。`C1` とは書かない）。\n"
        "休みは `z`、小節線は `|`（**`|` と `|` のあいだは1小節の単位数ちょうど**）、"
        "**小節をまたぐ音はタイ `-` で繋ぐ**（`-` は前の音のすぐ後ろにだけ付ける: `E2- | E2`。`| -E2` とは書かない）。\n"
        "例（4/4・8分で8音節）: `C2 D2 E2 D2 C2 D2 E2 F2 |`\n"
        "例（4/4・8分ひとつぶん食って入る）: `z2 C2 D2 E4 D2 C2 D2 |`\n"
        "小節線は `|` 1本だけ（`||`・`|]`・`|:` は書かない）。タイ `-` は同じ高さの次の音へつなぐ（休み `z` には付けない・行の最後に付けない）。\n"
        "★**歌い出しの数（タイでつないだ続きの音は数えない）は、この行のモーラ数と同じにしてください**。"
        # ★2026-10-02（本人「おすすめで」）: 休みだけの行は旋律の工程の検査の所見（`song_melody_step.MelodyStep.check`）
        "**休みだけの行（`z16` のような音の無い行）は書けません**——1行は1フレーズです（検査で所見として返します）"))


class AbcSketch(StepStatus):
    """旋律工程の受け取り型。**中身は `abc_sketch_to_melody` で `BareMelody` へ起こして下流へ渡す**。"""
    reason: str = Field(description="この旋律で何を狙ったか（詞を書く人・編曲する人に向けて）")
    sketch: str = Field(description="全体の輪郭を一言で（**必ず書く**。どこが山か・どう終わるか）")
    lines: list[AbcLine] = Field(description="行の並び（グルーヴが決めた行数と同じだけ）")

    @model_validator(mode="before")
    @classmethod
    def _from_saved(cls, data: Any) -> Any:
        """**保存の形（`BareMelody`）から受け取りの形へ戻す**（2026-10-03・再レビュー）。

        作り直しに巻き込まれた旋律が「前回のまま」を選ぶと、`song_llm.run_sync` が前回の出力（保存の形＝`phrases` と
        `sketch` の【書かれたABC】）をこの型で読み直す。以前は `lines` が無くて読めず、前回のままを捨てて毎回作り直していた
        （LLM 1回の無駄・記録には「前回の出力が今の形に合わない」）。【書かれたABC】は LLM が書いた行そのもの（直しのあとの版・
        `sketch_record`）なので、そこから行を戻す（`sketch_from_saved`）。
        ★読むのは**保存の形だけ**——`phrases` があって `lines` の無い dict で、`sketch` に【書かれたABC】の印があるとき。LLM の出力
          （この型の JSON スキーマに `phrases` は無い）を広く読む道ではない。ABC の記録の無い古い旋律（枠記法の頃）は戻せない（作り直す）。
        """
        if isinstance(data, dict) and "lines" not in data and "phrases" in data:
            got = sketch_from_saved(data)
            if got is not None:
                return got
        return data


def abc_sketch_to_melody(got: "AbcSketch", groove, *, bpm: float,
                         bars: dict[str, int] | None = None) -> "BareMelody":
    """ABC を**書かれた音価のまま**楽譜へ起こす（2026-09-20・測って採用）。

    旧 `expand_sketch`（歌い出しの位置だけ取り、長さは機械が引き算・2026-09-28 に削除）との違いは
    **長さを LLM が書く**こと。ABC は音価を持っているので、
    **「行の途中で音を止めて休む」が初めて書けるようになる**。

    ★**採った根拠**（同じ ABC を2通りに処理して比較・帯は実測の p20〜p80）:

    | | A（機械が引き算）| B（音価そのまま）|
    |---|---|---|
    | 帯に入った指標 | 5/8 | **6/8** |
    | 休符率 | 0.195★ | **0.339** |
    | 伸ばしのばらつき | 0.566 | **0.382** |
    | グリッド率 | 1.000 | **1.000**（崩れない）|

    ★★**採れるようになったのは、プロンプトの教え方を直してから**。
      最初は S/M/C の**3段階の表**を渡しており、出力も3段階に固まって
      **ばらつきが 0.118**（帯 0.27〜0.97）だった。本番のコードには
      **「3段階では原理的に再現できない」**と実測つきで書いてあったのに、
      3段階を教えて「LLM は散らせない」と結論しかけた。
      **法則（行の中身で決まる連続量）を教えたら 0.382 まで戻った。**
      本人の指摘:「既存の曲もランダムに散らせてるわけじゃなくて、
      **意図があってこういうところは伸ばす**みたいなルールがあると思う」。

    ★★**書いた長さのまま置く**（2026-09-28 本人方針「組み立ては一切補正しない」）。以前は枠と長さの違う行を、行末の休みを削ったり
      行全体を比で縮め・伸ばしたりして枠に合わせていた——拍の位置（裏拍・シンコペーション）とコードとの関係が崩れる補正だった。
      行の長さは旋律の工程の検査（`MelodyStep`）が「枠と合うか」を見て差分で直させ、直らなければ点検役へ渡す。
    """
    from abc_notation import abc_to_phrases
    phrases: list[Phrase] = []
    carry = 0.0
    prev_section: str | None = None
    for ln in got.lines:
        # ★**区間をまたいで休みを持ち越さない**（2026-09-20 に実際にずれた）。区間の切れ目の休みは**その区間のもの**
        #   （直前の音の後ろの空きとして持つ＝書いた長さのまま）。
        if prev_section is not None and ln.section != prev_section and carry and phrases:
            last = phrases[-1].notes[-1]
            last.gap_after = last.gap_after + carry
            carry = 0.0
        prev_section = ln.section
        # ★**1行＝1フレーズ**（2026-09-27・目録の時間-6）。行の中の休みは直前の音の `gap_after` として持つ。
        got_p, carry = abc_to_phrases(ln.abc, ln.section, bpm, lead_rest=carry, breath_sec=float("inf"))
        for p in got_p:
            phrases.append(Phrase(
                section=p["section"], rest_before=max(0.0, p["rest_before"]),
                notes=[BareNote(solfa=n["solfa"], octave=n["octave"], beats=n["beats"],
                                gap_after=max(0.0, n["gap_after"]), ties=n.get("ties", 0)) for n in p["notes"]]))
    if carry and phrases:                                  # 最後に残った休みも落とさない
        last = phrases[-1].notes[-1]
        last.gap_after = last.gap_after + carry
    return BareMelody(status=got.status, trouble=getattr(got, "trouble", "") or "",
                      reason=got.reason, sketch=got.sketch, phrases=phrases)


class AbcLinePatch(BaseModel):
    """旋律の差分1行ぶん（2026-09-27・`song_fix`）。**直す行だけ**を返させ、機械はそのまま当てる。

    ★**区間名・行の追加・行の削除も差分で書ける**（2026-09-29）。以前は `abc` の欄しか無く、区間名が設計と違う・
      行の数が足りない／多いという所見が**差分では直らなかった**——区間名が残ると時間軸で知らない区間になり曲づくりが止まった。
    """
    no: JsonInt = Field(description="直す行の番号（依頼文の「いまの旋律」の番号・1始まり）。"
                                "`insert` のときは**この行のあとに**足す（0 なら先頭に）")
    abc: str = Field(default="", description="その行の新しい ABC（**行全体**を書き直す。書き方は旋律の工程と同じ）。"
                                             "区間名だけ直すときは空でよい")
    section: str = Field(default="", description="区間名を直すときだけ書く（設計の区間名をそのまま）。空ならいまのまま")
    insert: JsonBool = Field(default=False, description="true なら**新しい行を足す**（`no` の行のあと・`section` と `abc` は必須）")
    drop: JsonBool = Field(default=False, description="true なら**この行を消す**（`insert`・`abc`・`section` と一緒に書かない）")
    # ※`why` は読み手なし（記録にも依頼文にも出ない）。直す前に考えを書かせる欄として残す（2026-09-30・出-13）
    why: str = Field(default="", description="どの所見にどう応えたか（一言）")

    # ★**消すと足す・直す欄を一緒に書いた項目は型が差し戻す**（2026-10-03・記B・仕様書 §15）。以前は `insert` と `drop` の両方 true で
    #   足すほうになり、`drop` と `abc`・`section` では消すほうが黙って勝った。歌い方・伴奏の差分と同じ扱い
    @model_validator(mode="after")
    def _drop_alone(self):
        if self.drop and (both := [k for k, v in (("insert", self.insert), ("abc", (self.abc or "").strip()),
                                                  ("section", (self.section or "").strip())) if v]):
            raise ValueError(f"no={self.no}: drop と {'・'.join(both)} を一緒に書いた——消すなら drop: true だけ、"
                             f"{'足すなら insert と section と abc' if self.insert else '直すなら abc か section'} だけ")
        return self


class AbcPatch(PatchStatus):
    """旋律の直し（差分）。**直さない行は書かない**——書かなかった行はいまのまま残る。"""
    # ※`reason` は読み手なし（記録にも依頼文にも出ない）。直す前に考えを書かせる欄として残す（2026-09-30・出-13）。`trouble` は直しの記録に入る
    reason: str = Field(description="何をどう直したか（全体で一言）")
    patches: list[AbcLinePatch] = Field(default_factory=list, description="直す行だけ。直さない行は書かない")


# ※`_resolve_sections`（旋律が書いた区間名を設計の綴りへそろえる）は 2026-10-02 に外した（組「洗い出し-型と残り」）。
#   区間名が完全一致だけ（判断6）になり、前後の空白は工程の親 `StepFixer.canonize` が取り除くので、ここで書き換える名前が無くなった。
#   見ていた観点と、いま見ている所: ①名前が設計と合わないと転調・ハモリ・小節数の逆算が黙って外れる → 旋律の工程の設計との照合
#   （`MelodyStep`・`against_design`）が所見にして差分で直させ、直らなければ止まる ②歌のある区間を旋律が書かなかった → 同じ照合の
#   `need_all_sections` ③区間の並び → `ordered_sections`。保存済みの記録の旋律の区間名はどれも設計と完全一致（1119フレーズ・2026-10-02）。


# ---- 1行の枠（区間の小節数 ÷ グルーヴの行数）---------------------------------------

def grid_per_bar() -> int:
    """1小節ぶんの16分音符の数（旋律の枠の単位数）。いまの曲の拍子（`song_context.METER`）から出す。4/4→16 / 3/4→12。

    ★**1行が1小節とは限らない**——設計が 8小節・グルーヴが4行なら **1行 2小節**（4/4 で32単位）。
    以前は「間隔の中央値が8分より広ければ詰める」という固定の基準で補正しており、
    **正しく書かれた2小節の枠を1小節へ潰していた**（2026-09-09『穏やかな空白』・区間が 8小節→5小節）。
    """
    return units_per_bar(ctx.METER)

#: 16分ひとつぶんの拍数（位置も休みもこの倍数に置く）。
GRID_BEAT = 0.25


def _line_slots(gv: dict, bars: dict[str, int] | None) -> dict[str, list[int]]:
    """区間ごとに、**1行が受け持つ枠**（16分いくつぶん）を出す。

    設計が「区間は何小節か」を、グルーヴが「何行か」を決めているので、
    **1行の枠はもう決まっている**——8小節を4行なら 1行 2小節＝32枠。
    LLM が書くのはその枠の**中のリズム**だけ（構造は LLM・座標はプログラム）。
    """
    out: dict[str, list[int]] = {}
    for kind, b in (bars or {}).items():
        ln = int(getattr(gv.get(kind), "lines", 0) or 0)
        if not b or not ln:
            continue
        # ★**1行は「小節の整数倍」でなければならない**（2026-09-10）。
        #   8小節を3行にすると 1行 2.67小節になり、**行の頭が小節の中で毎行ずれる**
        #   ——`--o-o---` の「小節のここで歌い出す」という意味が壊れる。
        #   実測: verse（8小節÷4行＝2小節）は4行とも書いたとおりの位置だったが、
        #   pre（8小節÷3行）は**全行ずれ**、本人に「**20秒あたりの雰囲気が変わるところから
        #   タイミングが合わなくなる**」と言われた（そこが Bメロの入り）。
        #
        base, extra = divmod(max(b, ln), ln)
        out[kind] = [(base + (1 if i < extra else 0)) * grid_per_bar() for i in range(ln)]
    return out


# ※枠記法（`onsets`）の小節線の文字数の検査（`bar_line_issues`・2026-09-11）は 2026-09-30 に枠記法ごと外した。
#   ABC の小節線ごとの単位数の検査（`abc_bar_issues`・2026-09-27）は同じ日に外した（本人判断）が、2026-09-30 に行の長さの所見と
#   1行1つにまとめて戻した（記-8 本人「A」・`song_melody_step.line_shape`）。臨時記号は `|` まで効き、後の工程も `|` を小節線として読むため


def sketch_record(got: "AbcSketch") -> str:
    """LLM が**実際に書いた ABC**を、そのまま読める形で残す。

    保存されるのは展開後の `phrases` なので、本番で旋律が壊れたときに
    **何を書いたのかが追えない**（実際に追えず、楽譜から逆算したことがある・2026-09-09）。

    ★**記法を移したら、記録側も一緒に移す**（2026-09-21）。ABC へ移したあとも
    ここが旧フィールド（`onsets`）を読んでいたため、**オリジナル生成が通しで落ちた**
    （`AttributeError: 'AbcLine' object has no attribute 'onsets'`）。
    工程単体のテストは ABC の変換だけを見ていて、この行を通っていなかった。
    """
    body = "\n".join(f"{ln.section}: {ln.abc}" for ln in (got.lines or []))
    return (got.sketch or "") + _ABC_SEP + body


#: `sketch_record` が輪郭の一言と行の記録のあいだに置く印（`song_context._ABC_MARK` を改行で挟んだもの）
_ABC_SEP = "\n【書かれたABC】\n"


def sketch_from_saved(saved: dict) -> dict | None:
    """**保存の形（`BareMelody` の dict）から受け取りの形（`AbcSketch` の欄）を戻す**——`sketch_record` の逆（2026-10-03・再レビュー）。

    行は `sketch` の【書かれたABC】（「区間名: ABC」の1行ずつ）から、輪郭の一言はその前から、狙い・自己申告はそのまま。
    戻せない（印が無い＝ABC の記録の無い古い旋律・行が「区間名: ABC」の形でない・行が無い）ときは None。
    """
    sk = str(saved.get("sketch") or "")
    if _ABC_SEP not in sk:
        return None
    head, body = sk.split(_ABC_SEP, 1)
    rows = body.split("\n")
    if not rows or not all(": " in r for r in rows):
        return None
    lines = [dict(zip(("section", "abc"), r.split(": ", 1))) for r in rows]
    out = {k: saved[k] for k in ("status", "trouble", "reason") if k in saved}
    return {**out, "sketch": head, "lines": lines}


# ---- 音名・最高音 --------------------------------------------------------------

#: 「おおよその高さ」→ サビの最高音（MIDI・内部の数）。2026-10-04（組N2）までは設計が `top_note` を書かなかったときの既定
#: （76＝E5・81＝A5・85＝C#6）。いまは組み立てが読まず、古い記録の変換（`convert_record._n2_defaults_fill`）が当時の値として使う。
#: 実測でミクが気持ちよく鳴るのは 58〜86（A#3〜D6）。**86 を超えると音量が落ちて痩せる**。
VOICE_TOP = {"低め": 76, "中": 81, "高め": 85}

# ※上限 `HARD_TOP`（90）を超えた音を1オクターブ下げる処理と、設計の最高音を 60〜90 へ丸める処理は 2026-09-28 に外した
#   （組み立ては一切補正しない）。最高音は設計が決め、旋律の検査（`MelodyStep`）が見る。

# ★**LLM に見せる高さは国際式の音名**（2026-10-01・判-A の続き。本人「楽譜（国際式の音名）に寄せる」）。以前はここが階名
#   （`ラ5`・`ド♯4`）で、旋律の依頼文の最高音・最低音・区間の音域、旋律の検査の所見、旋律の点検役の材料が階名だった
#   （旋律は ABC で書くのに、高さの物差しだけ階名と MIDI 番号だった）。綴りは表と同じ `song_table.spell_pitch`
#   （旋律は C 基準・移調前なので調を渡さない＝黒鍵は ♯）。旋律の内部の形（`BareNote.solfa`）は LLM に見せないので階名のまま。
_ABC_LETTERS = ("C", "^C", "D", "^D", "E", "F", "^F", "G", "^G", "A", "^A", "B")


def _note_name(key: int) -> str:
    """MIDI → 国際式の音名（`C4`＝60・`A#5`＝82）。LLM に見せる高さはこれで書く。"""
    from song_table import spell_pitch                      # 表の綴りと同じ1つ
    return spell_pitch(key)


def abc_pitch(key: int) -> str:
    """MIDI → **ABC の音名**（`C`＝60・`c`＝72・`a`＝81・`c'`＝84・`B,`＝59・黒鍵は `^`）。旋律の依頼文と所見で、
    国際式の音名に ABC での書き方を添えるために使う（旋律は ABC で書くので、上限・下限を ABC の字で照らせる）。
    読み手（`abc_notation.abc_to_midi`）で読み戻すと同じ数になる。"""
    acc_letter = _ABC_LETTERS[key % 12]
    octave = key // 12 - 1
    acc, letter = acc_letter[:-1], acc_letter[-1]
    if octave >= 5:
        return acc + letter.lower() + "'" * (octave - 5)
    return acc + letter + "," * (4 - octave)


def pitch_with_abc(key: int) -> str:
    """`A5`（ABC の `a`）——旋律の工程に見せる高さの書き方（国際式の音名に、ABC での書き方を添える）。"""
    return f"{_note_name(key)}（ABC の `{abc_pitch(key)}`）"


def top_note_of(theme, design) -> int:
    """**この曲の最高音**（MIDI）。設計の指定（国際式の音名・`song_design.Design.top_note`・必須）を読む。

    ★設計は 2026-10-01 から最高音を**国際式の音名**で書く（以前は MIDI 番号）。読み手は音名の記法の型（`song_llm.PITCH`）1つ。
      型（`Design._readable_top_note`）が読めない音名を差し戻すので、ここへ読めない値が来たら不具合（黙って既定にしない）。
    """
    # ★**最高音の正は設計の `top_note`**（2026-10-04・組N2）。以前は空なら構想の `voice_height` から `VOICE_TOP` で引いた（二重の既定）。
    #   いまは型が空を差し戻すので、空が来たら組み立ての不具合（保存済みの古い記録は `convert_record` が当時の値を書き込む）
    name = design.top_note
    if not name:
        from song_fix import AssemblyError                 # noqa: E402（循環を避けてここで）
        raise AssemblyError("設計の最高音（top_note）が空——型が差し戻すはず（古い記録なら convert_record を通す）")
    from song_llm import PITCH                              # noqa: E402（音名の読み手は1つ）
    t = PITCH.read(name)
    if t is None:
        raise ValueError(f"設計の最高音『{name}』が音名として読めない（型が差し戻すはず）")
    # ★設計が書いた高さは**そのまま使う**（2026-09-28 に 60〜90 への丸めも外した・組み立ては一切補正しない）。
    #   58〜86 は目安であって壁ではない（2026-09-13・本人「あえて外すこともある」）
    return t
