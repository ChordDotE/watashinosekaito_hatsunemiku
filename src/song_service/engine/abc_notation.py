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

"""ABC 記法と、内部の枠表現（`o`/`-`）の相互変換（2026-09-20）。

**旋律は ABC 記法で書かせる**（`decisions.md` 2026-09-20 ⑦）。独自の枠記法をやめた理由:

・**枠では「食い」が書けなかった**——`onsets` は行の先頭から書くので**枠より前に音を置けず**、
  本番プロンプトは「小節線をまたいで書け」という回避策を教えていた。ABC は弱起が素直に書ける
・**「行の途中で止めて休む」が書けない**（`-` が伸ばしか休みか区別できず、機械が推測している）
・**学習データが桁違いに多い**——ABC は世界中で使われている記法
・却下理由だった拍頭率の差は、**裏拍を促す一文を足したら消えた**
  （2026-09-20 実測: 枠 0.292 / ABC 0.338・帯は 0.09〜0.40。当時の記録は ABC 0.47）

★**内部（`rhythm_metrics` / 検査）は枠表現のまま**。
  変えたのは「LLM に何で書かせるか」の1点だけで、**ABC を受け取った直後にここで枠へ戻す**。
  下流を巻き込まないので、戻すのもこの1か所で済む。

※**枠 → ABC**（`frame_to_abc`・プロンプトの例を訳すため）は本番で使われなくなったので、2026-09-28 に消した。

> ★**（移し終えた・下の `abc_to_phrases` と `song_melody.abc_sketch_to_melody`）** 当時は ABC が**音価をそのまま持っている**のに、
> 枠へ戻すときに捨てて `expand_sketch` が「次の歌い出しまで」で引き算し直していた（`expand_sketch` は 2026-09-28 に消した）。
> そこを使えば「行の途中で止めて休む」が本当に書けるようになる。**別の改修として分けてあった。**
>
> ★**理由は「危ないから」ではない**（2026-09-20 に言い方を訂正）。
> 2026-09-08 の事故は `beats: 0.5` という**自由な小数**を書かせていたために
> **半端な値1つで以降が全部ずれ、68音中50音（74%）がグリッドの外**へ出たもので、
> **ABC の音価は16分の整数個＋「1小節＝16単位ちょうど」の検算が効く**ので別物
> （`song_quality_premises.md` にもそう訂正されている）。
> 分けたのは**影響範囲**のため——当時は**長さの決定を `expand_sketch` が全部持っており**
> （各音の長さ・行末の伸ばし・行のあとの休み）、そこを LLM 側へ移すと
> **伸ばし・息継ぎ・休符率・ひと息の秒がまとめて動く**。独立して測りながらやる。

## 約束

・`L:1/16`（1単位＝16分音符）。拍子は曲の拍子（1小節の単位数は 4/4 16・3/4 12・6/8 12・2/4 8＝`song_design.units_per_bar`）
・音の高さは ABC の標準（`C`＝中央ハ=MIDI 60、`,`で1オクターブ下、`'`で1オクターブ上、
  小文字で1オクターブ上）。`^`＝♯、`_`＝♭
・休符は `z`、小節線は `|`（1本だけ・`||` は書けない）、タイは `-`
・定義（正）は記法の仕様書 `docs/components/song_notation.md` §1
"""
from __future__ import annotations

import re

UNITS_PER_BAR = 16
#: 階名 → ピッチクラス。**旋律の内部の形**（`abc_to_phrases` が `BareNote.solfa` に書く階名）の読み手で、LLM には見せない
#: （2026-10-01。以前は「本番の `pitches` はこの表記」と書いていたが、伴奏・彩りの `pitches` は国際式の音名になった＝`song_llm.PITCH`。
#: LLM に見せる旋律の高さは ABC そのものか、国際式の音名＝`song_melody._note_name`）。下の `solfa_to_midi` を使うのは試験だけ
_SOLFA = {"ド": 0, "レ": 2, "ミ": 4, "ファ": 5, "ソ": 7, "ラ": 9, "シ": 11}
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def solfa_to_midi(s: str) -> int:
    """`"ミ4"` `"♯ファ5"` → MIDI 番号。読めなければ 60。"""
    s = (s or "").strip()
    acc = 0
    while s and s[0] in "♯#♭b":
        acc += 1 if s[0] in "♯#" else -1
        s = s[1:]
    m = re.match(r"^(ド|レ|ミ|ファ|ソ|ラ|シ)(-?\d+)$", s)
    if not m:
        return 60
    return (int(m.group(2)) + 1) * 12 + _SOLFA[m.group(1)] + acc


#: ★**音名の文法**（仕様書 §1・2026-10-02 組「洗い出し-コードとABC」）: 臨時記号は1つまで（`^` `_` `=`）、オクターブの記号は
#:   **大文字に `,`・小文字に `'` だけ**（同じ高さの書き方を1通りに・本人決定 2026-10-02「おすすめで」）。以前は臨時記号を何個でも・混ぜても
#:   足し合わせて読み（`^^^C`＝D#・`^_C`＝C・`=^C`＝C#）、`c,`（＝`C`）・`C,'`（＝`C`）も読んでいた。書けない書き方は `unreadable` の所見
_NOTE_NAME = re.compile(r"(?P<acc>[\^_=]?)(?:(?P<up>[A-G])(?P<down>,*)|(?P<low>[a-g])(?P<upm>'*))")


def abc_to_midi(token: str) -> int:
    """ABC の音名（長さ・タイを除いた `^F,`・`c'` の形）→ MIDI 番号。文法（`_NOTE_NAME`）に合わなければ `ValueError`
    （読み方は `notes_of` が作る音名だけを渡す。検査が先に所見にするので届かない）。"""
    m = _NOTE_NAME.fullmatch(token)
    if not m:
        raise ValueError(f"ABC の音名として読めない: {token!r}")
    acc = {"^": 1, "_": -1}.get(m["acc"], 0)
    if m["up"]:
        return (4 + 1 - len(m["down"])) * 12 + _LETTER_PC[m["up"]] + acc
    return (5 + 1 + len(m["upm"])) * 12 + _LETTER_PC[m["low"].upper()] + acc


#: ★**読み方は文法どおり**（2026-10-02）: 休み `z` は臨時記号・オクターブの記号を付けない（以前は `^z4`・`z,4` も休みとして読んだ）。
#:   長さは半角の数字で頭に 0 を付けない（`digit19 , { digit }`。以前は `\d` が全角の `２` も、`z02` の `02` も読んだ）。
#:   文法の外の字はここで読まずに残り、`unreadable` が所見にする
_TOKEN = re.compile(r"(?:(?P<z>z)|" + _NOTE_NAME.pattern + r")(?P<num>[1-9][0-9]*)?(?P<frac>/[0-9]+)?(?P<tie>-?)")

#: ★**時間を持たない書き方**（2026-10-01・標準とのずれの点検）。ABC の標準では、注釈 `%…`（行の終わりまで）・和音名の引用 `"Am"`・
#:   装飾 `!trill!`・装飾音 `{g}`・行の中のフィールド `[K:G]` は**音の長さを持たない**。以前は中の字を音名として読んでいた
#:   （`"Am"C4` の `A`・`{g}A4` の `g`・`% ab` の `a` `b` が1単位の音になり、行の長さも高さも変わった）。
#:   読み方は標準どおり「時間を持たない」にし、鳴らせない（和音名・装飾・装飾音・フィールド）ものは `unreadable` が所見にする。
_NO_TIME = re.compile(r"%.*$|\"[^\"]*\"?|![^!]*!|\{[^}]*\}?|\[[A-Za-z]:[^\]]*\]?", re.M)


def _strip_no_time(abc: str) -> str:
    return _NO_TIME.sub(" ", abc or "")


def notes_of(abc: str) -> list[tuple[str | None, int, bool, int]]:
    """ABC → [(音名 or None=休み, 単位数, タイ, 小節番号)]。**読み方はここ1か所**（2026-09-27）。

    ★**臨時記号は小節の終わりまで続く**（ABC の決まり）。`^F4 F4` の2つ目もファ♯。
      以前は1音ごとに読んでいて、2つ目がファ（半音違い）になっていた。
    ★**同じ音名なら、ほかのオクターブにも効く**（2026-10-01・判-G 本人「標準に寄せる方式で全体的に」）。ABC 2.1 の既定
      （`%%propagate-accidentals pitch`）どおり、`^F4 f4 F,4` は3つともファ♯。以前は同じオクターブにだけ効かせていた
      （楽譜の流儀）。保存済みの ABC 840行で、読み替えて高さが変わる行は0だった。
    ★時間を持たない書き方（注釈・和音名の引用・装飾・装飾音・行の中のフィールド）は読み飛ばす（`_NO_TIME`）。
    """
    out: list[tuple[str | None, int, bool, int]] = []
    for bi, bar in enumerate(_strip_no_time(abc).split("|")):
        held: dict[str, str] = {}                          # 音名（大文字）→ 臨時記号（オクターブを問わない）
        for m in _TOKEN.finditer(bar):
            num, frac, tie, acc = m["num"], m["frac"], m["tie"], m["acc"]
            n = int(num) if num else 1
            if frac:                                       # `/2` 等は16分より短い＝1単位に丸める（`unreadable` が所見にする）
                n = max(1, n // int(frac[1:]))
            if m["z"]:
                out.append((None, n, False, bi))
                continue
            letter = m["up"] or m["low"]
            k = letter.upper()
            if acc:
                held[k] = acc
            out.append((held.get(k, "") + letter + (m["down"] or m["upm"] or ""), n, bool(tie), bi))
    return out


#: ★**定義より広く読んでいた書き方**（2026-10-02・組「洗い出し-コードとABC」・仕様書 §1 の文法）。以前は所見なしに読んでいた。
#:   どれも音名・休みに付く印なので、`_leftover` は消して詰める
_GLUED_RULES = [
    (re.compile(r"(?<=[A-Ga-gz,'])0+[0-9]+"), "長さの数字の頭に 0 を付けない（`z02` → `z2`）"),
    (re.compile(r"[０-９]+"), "数字は半角で書く（全角の `２` は読めない・`z２` → `z2`）"),
    # ★**長さ 1 を数字で書く**（2026-10-02 本人「Cだけ」・仕様書 §17 ず-34）。以前は `C1` を `C` と同じに読み、同じ長さが2通りだった
    (re.compile(r"(?<=[A-Ga-gz,'])1(?![0-9])"), "1単位は数字を書かない（`C1` → `C`・`z1` → `z`）"),
    (re.compile(r"[\^_=]{2,}(?=[A-Ga-g])"),
     "臨時記号は1つだけ（♯ `^`・♭ `_`・ナチュラル `=`）——ダブル♯・ダブル♭（`^^C`・`__E`）は書けないので異名の簡単な名前で書き"
     "（`^^C` → `D`・`__E` → `D`）、混ぜた `^_C`・`=^C` も書かない"),
    (re.compile(r"[\^_=]+(?=z)|(?<=z)[,']+"), "休み `z` には臨時記号・オクターブの記号（`^z4`・`z,4`）を付けない（休みは `z` と単位数だけ）"),
    (re.compile(r"(?<=[A-G]),*'[,']*|(?<=[a-g])'*,[,']*"),
     "オクターブの記号は大文字に `,`・小文字に `'` だけ（同じ高さの書き方は1通り: `c,` → `C`・`C'` → `c`・`C,'` → `C`・`c,,` → `C,`）"),
]

#: 読めない（または丸めて読む）記法。**黙って落とさず、所見にして旋律の工程へ返す**（2026-09-27）
_UNREADABLE = [
    # ★行の中のフィールド `[K:G]` もヘッダーの文で返す（2026-10-01。以前は「和音は書けない」になっていた）
    (re.compile(r"(^|\s)[A-Za-z]:|\[[A-Za-z]:[^\]]*\]?"), "ヘッダー（`K:`・`M:`・`L:` など・行の中の `[K:G]` も）は書かない"
                                                        "（拍子と単位は依頼文のとおり・調は C で書く）"),
    (re.compile(r'"[^"]*"?'), "和音名などの引用（`\"Am\"`）は書かない（鳴らない。コードはコードの工程が決める）"),
    (re.compile(r"\(\d+(?::\d*)*"), "連符（`(3` など）は読めない。16分の単位で書く"),
    # ★**スラー**（2026-10-01・標準とのずれの点検）。ABC では「つなげて歌う」の指定だが、ここでは鳴らし方に効かず黙って消えていた
    (re.compile(r"\((?!\d)|\)"), "スラー（`(` `)`）は読めない（つなげて歌う指定は鳴らない）。書かない（同じ高さを伸ばすならタイ `-`）"),
    # ★**装飾音**（2026-10-01）。ABC では時間を持たない飾りの音だが、以前は中の音名を1単位の音として読んでいた
    (re.compile(r"\{[^}]*\}?"), "装飾音（`{g}`）は読めない（鳴らない）。歌う音は1単位以上の音として書く"),
    (re.compile(r"/+\d*"), "16分より短い音（`C/2` など）は読めない。1単位＝16分が最小"),
    # ★終止線・始まりの線・番括弧は和音と別の文で返す（2026-09-30・記-11。以前は `|]` が「和音は書けない」になっていた）
    (re.compile(r"\[(?![|\d])|(?<!\|)\]"), "和音（`[CEG]`）は書けない。歌は1音ずつ"),
    (re.compile(r"\|\]|\[\|"), "終止線・始まりの線（`|]`・`[|`）は書かない。小節線は `|` だけ"),
    # ★**二重の縦棒・空の小節**（2026-10-01・仕様書 §17 ず-14）。ABC の `||` は区切りの二重線だが、ここの文法は縦棒1本だけ
    #   （`|` と `|` のあいだは音か休みの入った1小節）。以前は所見にならず、縦棒1本と同じに読んでいた（小節は増えない）。
    #   読み方は変えず、書き方を1通りにするために所見にする
    (re.compile(r"\|[ \t]*\|"), "二重の縦棒 `||`（`|` と `|` のあいだが空）は書かない。小節線は `|` 1本だけで、"
                                "`|` と `|` のあいだには音か休み `z` を書く"),
    # ★**黙って消えていた書き方**（2026-09-30・記-11）。`_TOKEN` が読み飛ばすので、所見なしに時間が縮む・繰り返さずに1回だけ鳴っていた
    #   （行の長さの所見が別の理由で出るだけだった）。読み替えはせず所見にする
    (re.compile(r"[xXZ]\d*"), "`x`・`X`（見えない休み）・`Z`（小節まるごとの休み）は読めない（黙って消えて時間が縮む）。"
                              "休みは `z` に単位数を付けて書く（1小節休むなら、1小節ぶんの単位数の `z`）"),
    (re.compile(r"\|:|:\||::"), "反復記号（`|:` `:|`）は読めない（繰り返されずに1回だけ鳴る）。繰り返すなら、その小節を書き並べる"),
    (re.compile(r"\[\d|\|\d"), "番括弧（`|1` `[2`）は読めない。繰り返すなら、その小節を書き並べる"),
    (re.compile(r"![^!]*!|[~.HLMOPSTuv](?=[A-Ga-g^_=])"), "装飾記号は読めない"),
    # ★**付点のリズム記号**（2026-09-27・監査）。`C2>D2` は ABC では C3 D1（前を付点・後ろを短く）だが、
    #   `_TOKEN` は `>` `<` を読み飛ばすので **C2 D2（均等）として黙って鳴っていた**。読み替えはせず所見にする
    #   （音価を数字で書き直してもらう＝書いた人が長さを決める）。
    (re.compile(r"[<>]"), "付点のリズム記号（`C2>D2`＝C3 D1・`C2<D2`＝C1 D3）は読めない（均等に読まれてしまう）。"
                          "音価を数字で書く（`C3 D`）"),
    # ★**後ろに付けた ♯・♭**（2026-09-27・要否の棚卸しで発見）。`C#2` は `_TOKEN` が `C`（ナチュラル・1単位）と読み、
    #   `#2` を読み飛ばしていた——**高さも長さも黙って変わる**（実出力で3曲・約19行）。ABC の臨時記号は前に付ける
    (re.compile(r"[A-Ga-g][,']*[#♯♭]\d*"), "臨時記号を音名の後ろに付けている（`C#2`）。ABC では前に付ける（♯は `^C2`・♭は `_C2`）。"
                                           "後ろに付けるとナチュラルの1単位として読まれてしまう"),
    # ★**ABC の臨時記号（`^` `_` `=`）を後ろに付けた `A_2`**（2026-09-30・出-14①）。`_TOKEN` は `A`（臨時記号なし・1単位）と読み、
    #   `_2` を読み飛ばしていた（『孤独のまま』の 14・30・42行目。長さの検査で偶然捕まった）。後ろに次の音名が続く `A_B`
    #   （A と B♭）は正しい書き方なので所見にしない
    (re.compile(r"[A-Ga-g][,']*\d*[\^_=]+\d*(?![\^_=]*[A-Ga-g])"),
     "臨時記号を音名の後ろに付けている（`A_2`・`C^2`）。ABC では前に付ける（♭は `_A2`・♯は `^C2`・ナチュラルは `=A2`）。"
     "後ろに付けると、臨時記号の無い音の1単位として読まれてしまう"),
    # ★**長さ 0**（2026-10-01）。`C0` は時間を持たない音として読まれ、行の長さと歌い出しの数が食い違う
    (re.compile(r"(?<=[A-Ga-gz,'])0+(?![0-9])"), "長さ 0 の音（`C0`）は書けない。長さは1単位以上"),
    *_GLUED_RULES,
    (re.compile(r"[^\S \t]+"), "音のあいだの空白は半角の空白かタブだけ（全角の空白・改行などは書けない）"),
]
#: 音名・休みに付いた印の所見（`_leftover` で消して詰める＝後ろの長さを「音名から離れた数字」と数えない）
_GLUED = {p for p, _w in _GLUED_RULES}

_COMMENT = re.compile(r"%.*$", re.M)
_INLINE_FIELD = re.compile(r"\[[A-Za-z]:[^\]]*\]?")


def _leftover(text: str) -> str:
    """**どの読み方にも当たらずに読み飛ばされる文字**（2026-10-01）。所見にした書き方・音・休み・小節線・空白を除いた残り。
    `y`（間隔）と行末の `\\`（行の続き）は ABC でも音にならない印なので残りに数えない。"""
    s = text
    for pat, _why in _UNREADABLE:
        s = pat.sub("" if pat in _GLUED else " ", s)       # 音名・休みに付いた印は消して詰める（`z,4` の `4` を離れた数字にしない）
    s = _TOKEN.sub(" ", s)
    return re.sub(r"[ \t|y\\]", "", s)                    # 区切りの空白は半角の空白とタブだけ（仕様書 §1 の `ws`・2026-10-02）


def unreadable(abc: str) -> list[str]:
    """この行で読めない（または丸めて読む）記法。空なら全部読める。

    ★注釈 `%…` は ABC の標準どおり読み飛ばす（音にならない・所見にしない）。
    ★**どの読み方にも当たらない文字も所見にする**（2026-10-01・標準とのずれの点検）。以前は `_TOKEN` に合わない字を黙って
      読み飛ばしていた——小節線の後ろの `-`（`E2- | -E2`。ABC ではタイは音のすぐ後ろにしか付けない）、音名から離れた数字
      （`C 2` が1単位の C になる）、そのほかの字。
    """
    # 前後は半角の空白とタブだけ取り除く（仕様書 §0-6・2026-10-03 判断4 本人決定）。前後の全角の空白・NBSP・改行などは
    #   `ABC_LINE`（`half_width`）が先に所見にする。途中の全角の空白などは下の「読めない文字」の所見
    t = _COMMENT.sub(" ", strip_ascii_ws(abc))
    out = [why for pat, why in _UNREADABLE[:1] if pat.search(t)]
    t = _INLINE_FIELD.sub(" ", t)                         # 行の中のフィールドはヘッダーの文だけで返す（和音の文を重ねない）
    out += [why for pat, why in _UNREADABLE[1:] if pat.search(t)]
    rest = _leftover(t)
    if "-" in rest:
        out.append("タイ `-` は音のすぐ後ろに付ける（`E2- | E2`）。小節線や空白の後ろの `-`（`| -E2`・`C2 -C2`）は"
                   "ABC では書けない（読み飛ばされる）。前の音から伸ばすなら前の音の後ろにだけ `-` を書く")
    if re.search(r"\d", rest):
        out.append("音名から離れた数字がある（`C 2`）——長さは音名のすぐ後ろに書く（`C2`）。離れた数字は読み飛ばされる")
    other = sorted(set(re.sub(r"[-\d]", "", rest)))
    if other:
        out.append(f"読めない文字（{' '.join(other)}）がある——音名・`z`・長さの数字・`^` `_` `=`・`,` `'`・タイ `-`・"
                   "小節線 `|` だけで書く（ほかの字は読み飛ばされる）")
    return out


def abc_to_frame(abc: str) -> tuple[str, list[str]]:
    """ABC → (枠, 階名の並び)。**本番の指標へ渡すために戻す**（階名は内部の形・LLM には見せない）。

    ・タイ（`-`）で繋がった音は**1つの歌い出し**として数える（枠の `o` は歌い出しの数）
    ・`z` は休み＝`-`
    """
    grid: list[str] = []
    pitches: list[str] = []
    tied = False
    for name, n, tie, _bar in notes_of(abc):
        if name is None:
            grid += ["-"] * n
            tied = False
            continue
        if tied:                                           # 前の音の続き（歌い出しではない）
            grid += ["-"] * n
        else:
            grid += ["o"] + ["-"] * (n - 1)
            pitches.append(_abc_to_solfa(name))
        tied = tie
    return "".join(grid), pitches


def _abc_to_solfa(token: str) -> str:
    key = abc_to_midi(re.sub(r"\d+$", "", token))
    names = ["ド", "♯ド", "レ", "♯レ", "ミ", "ファ", "♯ファ", "ソ", "♯ソ", "ラ", "♯ラ", "シ"]
    return f"{names[key % 12]}{key // 12 - 1}"


def bars_of(abc: str, units_per_bar: int = UNITS_PER_BAR) -> int:
    grid, _ = abc_to_frame(abc)
    return -(-len(grid) // units_per_bar)


def tie_issues(abc: str) -> list[str]:
    """**高さの違う音をタイでつないだ所**（2026-09-29）。タイは前の音の続きとして読むので、後ろの音の高さは鳴らない。

    直さずに所見を返す（組み立ては書かれたとおりに前の音を伸ばす＝`abc_to_phrases`）。
    """
    out: list[str] = []
    prev: str | None = None
    prev_bar = -1
    tied = False
    # ★**つなぐ相手の無いタイ**（2026-10-01・標準とのずれの点検）。ABC のタイは「同じ高さの次の音」へつなぐ印。以前は
    #   休みに付けた `z2-`・後ろが休みの `C4- z4`・行の最後の `C4-` を黙って読み飛ばしていた（伸びずに切れる／次の行の頭の音は
    #   言い直しになる）。1行＝1フレーズなので、行をまたいでつなぐ読み方はしない（伸ばすならその行の中で長さを書く）
    if re.search(r"z\d*(?:/\d*)?-", _strip_no_time(abc)):
        out.append("休み `z` にタイ `-` は付けない（タイは音どうしをつなぐ印。休みは単位数で長さを書く）")
    for name, _n, tie, bar in notes_of(abc):
        if name is None:
            if tied and prev is not None:
                out.append(f"`{prev}-` のあとが休み——タイは同じ高さの次の音へつなぐ印。伸ばすなら休みを外して音の長さを足し、"
                           "切るなら `-` を外す")
            prev, tied = None, False
            continue
        if tied and prev is not None and abc_to_midi(prev) != abc_to_midi(name):
            # 小節線をまたぐタイで、後ろに臨時記号を付け直さなかった（`^C4- | C4`）のは同じ音の続きとして読む
            same_letter = bar != prev_bar and prev.lstrip("^_=") == name.lstrip("^_=") and name == name.lstrip("^_=")
            if not same_letter:
                out.append(f"`{prev}-` のあとの `{name}` は高さが違う——タイは同じ高さの音どうし（後ろの音の高さは鳴らない）。"
                           "高さを変えるなら `-` を外す")
        prev, prev_bar, tied = name, bar, tie
    if tied and prev is not None:
        out.append(f"行の最後の `{prev}-` は次の行へつながらない（1行＝1フレーズで、次の行の頭の音は言い直しになる）。"
                   "伸ばすならこの行の中で長さを書き、`-` を外す")
    return out


# ★**旋律の1行の記法＝型1つ**（2026-09-29・`song_checks.Notation`）。所見は読めない書き方（`unreadable`）と
#   高さの違う音のタイ（`tie_issues`）、読み方は `notes_of`——組み立て（`abc_to_phrases`・`abc_to_frame`）も検査の長さ・音域も
#   `notes_of` で読むので、検査と組み立てで読み方が2本にならない。旋律の工程（`song_melody_step.MelodyStep.written`）が宣言する
from song_checks import Notation, half_width, strip_ascii_ws   # noqa: E402

#   前後の空白は半角の空白とタブだけ取り除き、ほかの空白（全角・NBSP・改行など）は所見（`half_width`・仕様書 §0-6・2026-10-03）
ABC_LINE = Notation("ABC の旋律行", half_width(lambda abc: unreadable(abc) + tie_issues(abc)), notes_of)


def bar_units(abc: str) -> list[int]:
    """**`|` で区切った小節ごとの単位数**（2026-09-30・記-8 本人「A」で戻した）。読み方は `notes_of` と同じ（小節番号も `notes_of` の）。

    音も休みも無い区切り（行頭・行末の `|`、`||` のあいだ）は小節に数えない（`||` は記法の所見になる＝`unreadable`）。
    旋律の工程の検査（`song_melody_step.MelodyStep.check`）が、行の長さと一緒に「`|` と `|` のあいだが1小節ちょうどか」を見る。
    ※2026-09-27 に一度外した（`abc_bar_issues`）。外した理由と、戻した形（行の長さの所見とまとめて1行1つ）は `line_shape` の説明
    """
    per: dict[int, int] = {}
    for _nm, n, _t, b in notes_of(abc):
        per[b] = per.get(b, 0) + n
    return [per[b] for b in sorted(per)]


# ※枠へ戻してから小節線を見る検査（`bar_line_issues`）は、`abc_to_frame` が小節線を捨てるので効いておらず、2026-09-30 に消した。


# ---------------------------------------------------------------------------
# ★**音価をそのまま使う経路**（2026-09-20・測って採用）
#
# `abc_to_frame` は歌い出しの位置だけを取る（長さを引き算し直していた `expand_sketch` は 2026-09-28 に消した）。
# こちらは **ABC が書いた音価をそのまま**使う——これで初めて
# **「行の途中で音を止めて休む」**が書ける（枠記法では `-` が伸ばしか休みか区別できなかった）。
#
# 採否の根拠と、教え方を間違えていた経緯は `song_melody.abc_sketch_to_melody` の説明にある。
def abc_to_phrases(abc: str, section: str, bpm: float,
                   *, breath_sec: float = 0.40,
                   lead_rest: float = 0.0) -> tuple[list[dict], float]:
    """ABC → フレーズの素（`Phrase` に渡せる dict の並び）。**音価をそのまま使う**。

    休みは**長さで分ける**——`breath_sec` 未満は直前の音の `gap_after`（語尾を切る・跳ねる）、
    以上は**息継ぎ**なのでフレーズを切る。境目はプロンプトの表（息継ぎ 0.4〜0.7秒）に合わせた。

    ★**行末の休みは次の行へ持ち越す**（`lead_rest` で受け、残りを返す）。
      落としていたときは**曲全体が短くなり**、密度・ひと息・拍頭率までまとめて狂った
      ——「音価を使うと悪くなる」ように見えたが、**時間を落としていただけ**だった。
    ★**続いた休みは1つにまとめてから判定する**。`z4 z8` を別々に見ると
      どちらも「息継ぎ未満」になり、足し込んだ `gap_after` が上限（2拍）を超える。
    """
    spb = 60.0 / float(bpm)
    units: list[tuple[str | None, int, int]] = []          # (音名 or None, 単位数, タイでまとめた続きの音の数)
    tied = False
    for name, n, tie, _bar in notes_of(abc):
        if name is None:
            units.append((None, n, 0))
            tied = False
            continue
        if tied and units and units[-1][0] is not None:    # タイ＝前の音の続き（詞の印は続きの音にも1つ要る＝`ties`・判-H）
            units[-1] = (units[-1][0], units[-1][1] + n, units[-1][2] + 1)
        else:
            units.append((name, n, 0))
        tied = tie

    merged: list[tuple[str | None, int, int]] = []
    for name, n, t in units:
        if name is None and merged and merged[-1][0] is None:
            merged[-1] = (None, merged[-1][1] + n, 0)
        else:
            merged.append((name, n, t))

    phrases: list[dict] = []
    cur: list[dict] = []
    rest_before = float(lead_rest)
    tail = 0.0
    if merged and merged[-1][0] is None and any(nm for nm, _, _ in merged):
        tail = merged.pop()[1] / 4.0                       # ★行末の休みは**必ず**次の行へ持ち越す（伸縮で削れる場所）
    for name, n, ties in merged:
        beats = n / 4.0                                    # 1単位＝16分＝0.25拍
        if name is None:
            if not cur:                                    # フレーズの前の休み
                rest_before += beats
            elif beats * spb >= breath_sec:                # 息継ぎ＝フレーズを切る
                phrases.append({"section": section, "rest_before": rest_before, "notes": cur})
                cur, rest_before = [], beats
            else:                                          # 行の中の休み＝直前の音に付ける（2拍で捨てない・2026-09-27）
                cur[-1]["gap_after"] = round(cur[-1].get("gap_after", 0.0) + beats, 4)
            continue
        key = abc_to_midi(name)
        # ★**高さは書いたとおり**（2026-09-27）。以前は4〜6へ丸めていて、`B,`（シ3）がシ4になった
        #   ——1オクターブ変わるのに何も出さず、最低音の検査も丸めた後の値を見て効かなかった。
        #   音域の判断は旋律の検査（所見を返す）と設計・点検役に任せる。
        cur.append({"solfa": _abc_to_solfa(name).rstrip("0123456789-"),
                    "octave": key // 12 - 1,
                    "beats": beats, "gap_after": 0.0, "ties": ties})
    if cur:
        phrases.append({"section": section, "rest_before": rest_before, "notes": cur})
        rest_before = 0.0
    return phrases, rest_before + tail                     # 残りは次の行へ持ち越す
