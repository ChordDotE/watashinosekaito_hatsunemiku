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

"""**形式の検査の部品**（2026-09-28・`docs/plan/song_step_objects.md`）。

工程（`song_fix.StepFixer`）は、ここの部品を引数を変えて並べるだけにする。部品は観点ごとに1つで、所見の文の形もここで揃える。

- 形式の検査＝答えが1つに決まるもの（読めるか・名前が選択肢にあるか・個数や長さが枠と合うか・範囲に収まるか）。
  機械が判定し、所見を返して差分で直させる。
- **読めるか（記法）は `Notation`**（記法1つ＝型1つ・2026-09-29）。工程は `written` で宣言し、照合は親の `notation_findings`。
  工程の `check` には記法の判定を書かない（書くと2本になる）。
- **内容の検査（音楽として・言葉としておかしいか）はここに置かない**。機械は数字で断じず、点検役に本文を読ませる
  （規約 `.claude/skills/miku-dev/details/coding/song_service_editing.md`「工程の作り方」5）。

どの部品も (所見の文のリスト) を返す。空なら合格。`where` は「3行目（verse）」「パート2 の verse」のような場所の言い方。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Iterable

from pydantic import Strict

# ---- LLM が書く数と真偽の欄の型（2026-10-02・洗い出しの判断5 本人「おすすめで」）--------------------------------
#
# pydantic の既定は寛容で、整数の欄が `"84"`・`84.0`・`true` を、小数の欄が `"0.85"` を受けて黙って数に読み替えていた
# （洗い出し §1 の型の欄で 82通り中 51通り）。**LLM が書く欄（工程の出力・差分・道具の引数）の数と真偽は、ここの型だけを使う**。
# 定義は記法の仕様書 `docs/components/song_notation.md` §15「数の書き方」:
#   整数の欄＝JSON の整数だけ／小数の欄＝JSON の数（整数でも小数でもよい）だけ／真偽の欄＝`true` `false` だけ。文字列・真偽を数として読まない。
# ★モデル全体に `ConfigDict(strict=True)` を掛けない——Python の dict から検証する経路（pydantic_ai の出力・再生の `model_validate`）で
#   Enum が文字列を、tuple が list を受けなくなる。欄ごとに掛ける。JSON スキーマ（LLM に渡す形）は変わらない。
JsonInt = Annotated[int, Strict()]
JsonFloat = Annotated[float, Strict()]
JsonBool = Annotated[bool, Strict()]


def json_int(v: Any) -> Any:
    """**整数の選択肢（`Literal[8, 16]`）の前に置く検査**。`Literal` には `Strict` を掛けられず、`8.0` を 8 と読むので、
    整数（真偽を除く）でなければ差し戻す（`BeforeValidator(json_int)`）。"""
    if type(v) is not int:
        raise ValueError(f"整数で書く（{v!r} は整数ではない。文字列・小数・真偽は書けない）")
    return v


# ---- 前後の空白（2026-10-03・洗い出しの判断4 本人決定・仕様書 `docs/components/song_notation.md` §0-6）--------------------
#
# **半角だけで書く記法**（旋律の ABC・コード名・調名・音名・打点・位置 `小節:拍`・`at_bars`・区切りの記号 `breaks`・打楽器名・
# ドラムキット名・奏法名・声と歌い方の名前・形 `form` ほか。どれがどちらかは仕様書 §0-6 の表）は、前後で取り除くのを
# **半角の空白とタブだけ**にする。全角の空白・NBSP・細い空白・幅のない空白・改行が前後に付いていたら読み替えずに所見にする
# ——半角だけの記法に全角が入るのは書き間違いの印（本人）。**日本語を書く欄**（詞・見せる詞・声部の言葉・カバーの歌詞のかな）は
# 全角の空白が普通なので、これまでどおり前後の空白を全角・半角とも取り除く（Python の `strip()`）。
# ★検査も組み立ても、半角の記法の前後はここの2つだけで読む（読み方を2本にしない）。以前（2026-10-02 まで）はどの欄も `strip()` で
#   前後の Unicode の空白を全部取り除いて読んでいた（洗い出し §1 で 441通り）。

#: 半角の記法の前後で取り除く字（半角の空白とタブ）
ASCII_WS = " \t"
#: 幅のない空白（`str.isspace()` が空白と見ない字）。前後に付いていたら所見にする
_ZERO_WIDTH = "\u200b\u200c\u200d\u2060\ufeff"
_WS_NAME = {"\u3000": "全角の空白", "\xa0": "NBSP（改行しない空白）", "\u2009": "細い空白", "\u200a": "細い空白",
            "\u202f": "細い空白", "\n": "改行", "\r": "改行", "\u2028": "改行", "\u2029": "改行"}


def strip_ascii_ws(text: Any) -> str:
    """**半角だけで書く記法の前後の空白を取り除く**（半角の空白とタブだけ）。ほかの空白は残す（`edge_ws_problem` が所見にする）。"""
    return str(text or "").strip(ASCII_WS)


def _edge_ws(c: str) -> bool:
    return c.isspace() or c in _ZERO_WIDTH


def trim_edge_ws(text: Any) -> str:
    """前後の空白を種類を問わず取り除いた形——**所見に添える書き直し方と、残りの所見を見るためだけ**（読み替えには使わない。
    1つの文字列を返す検査は `edge_ws_problem` の所見に、この形で見た残りの所見を続ける）。"""
    s = str(text or "")
    i, j = 0, len(s)
    while i < j and _edge_ws(s[i]):
        i += 1
    while j > i and _edge_ws(s[j - 1]):
        j -= 1
    return s[i:j]


def edge_ws_problem(text: Any) -> str:
    """半角だけで書く記法の前後に、半角の空白・タブ以外の空白（全角の空白・NBSP・細い空白・幅のない空白・改行）が付いていれば
    その所見の文（無ければ空）。取り除いた形を添える（読み替えはしない）。"""
    s = strip_ascii_ws(text)
    if not s or not (_edge_ws(s[0]) or _edge_ws(s[-1])):
        return ""
    names = "・".join(dict.fromkeys(_WS_NAME.get(c, "幅のない空白" if c in _ZERO_WIDTH else f"空白の字（U+{ord(c):04X}）")
                                    for c in (s[0], s[-1]) if _edge_ws(c)))
    fixed = trim_edge_ws(s)
    return (f"『{text}』の前後に{names}が付いている——半角の字だけで書く欄なので、前後に余計な文字を付けない"
            + (f"（『{fixed}』と書く）" if fixed else ""))


def half_width(problems: Callable[[str], list[str]]) -> Callable[[str], list[str]]:
    """**半角だけで書く記法の所見**に、前後の余計な空白の所見（`edge_ws_problem`）を先に足す。残りの所見は余計な空白を除いた形で見る
    （同じ字を「読めない文字」として2回言わない）。記法の型（`Notation`）の `problems` を包んで使う。"""
    def run(text: str) -> list[str]:
        why = edge_ws_problem(text)
        if not why:
            return problems(text)
        return [why] + problems(trim_edge_ws(text))
    return run


def names_in(items: Iterable[tuple[str, str]], allowed: Iterable[str], what: str) -> list[str]:
    """**名前が選択肢にあるか**。`items` は (場所, 書いた名前)。`what` は「設計の区間名」「使える奏法」など。"""
    ok = list(dict.fromkeys(allowed))
    return [f"{where}: 『{name}』は{what}に無い（使えるのは {' / '.join(ok)}）"
            for where, name in items if name not in ok]


def count_is(where: str, got: int, want: int, what: str, *, then: str = "") -> list[str]:
    """**個数が合うか**（小節数・行数・音の数・打点の数）。`then` は合わないと組み立てがどうなるか。"""
    if got == want:
        return []
    return [f"{where}: {what}が{got}（{want}であるべき）" + (f"——{then}" if then else "")]


def length_is(where: str, got: float, want: float, unit: str, *, tol: float = 0.0, then: str = "") -> list[str]:
    """**長さが枠と合うか**（1行の単位数・1小節の文字数など）。`tol` は許す幅（同じ観点はどこでも同じ幅を使う）。"""
    if abs(got - want) <= tol:
        return []
    return [f"{where}: 長さが{got:g}{unit}（枠は{want:g}{unit}ちょうど）" + (f"——{then}" if then else "")]


def in_range(where: str, value: float | None, lo: float | None, hi: float | None, what: str,
             *, show=lambda v: f"{v:g}") -> list[str]:
    """**範囲に収まるか**（音域・深さ・音量）。`show` は値の見せ方（音名など）。"""
    if value is None:
        return []
    if lo is not None and value < lo:
        return [f"{where}: {what}が{show(value)}（{show(lo)}より低い）"]
    if hi is not None and value > hi:
        return [f"{where}: {what}が{show(value)}（{show(hi)}より高い）"]
    return []


# ---- 記法（2026-09-29・`docs/plan/song_step_objects.md`「記法1つ＝型1つ」）--------------------------------

@dataclass(frozen=True)
class Notation:
    """**記法1つ＝型1つ**（2026-09-29 本人了承）。`problems(text)` がその記法の所見（空なら読める）、`read(text)` が読み方。

    本人「記法と形式は工程につき1つに集約したほうがよい。見てるのは特定の記法のパターンに当てはまるかどうかでしょ」
    「記法と型を一致させた方がいい」。以前は記法の判定が工程ごとの `check` の中にばらばらにあり、同じ記法（位置 `小節:拍`）を
    工程ごとに別の文・別の読み方で見ていた。

    - 工程は `StepFixer.written` で「出力のどこに・どの記法の文字列があるか」を宣言するだけ。照合は親（`notation_findings`）の1か所
    - **組み立ても同じ型の `read` で読む**——検査と組み立てで読み方が2本になると「検査は通ったのに鳴らない」ができる
    - 型は**読み方の置き場に置く**（ABC は `abc_notation.ABC_LINE`・詞は `song_lyrics.LYRIC_LINE` など）。一覧は計画の文書
    """
    name: str
    problems: Callable[[str], list[str]]
    read: Callable[[str], Any] | None = None


def notation_findings(items: Iterable[tuple[str, Notation, str]]) -> list[str]:
    """**記法の照合**（親が持つ1か所）。`items` は工程が宣言した (場所, 記法, 書いた文字列)。所見は「場所: なぜ」。"""
    return [f"{where}: {why}" for where, nt, text in items for why in nt.problems(text)]


#: ★**位置の文法は仕様書 `docs/components/song_notation.md` §12 のとおり**（2026-10-01・ず-8）: 小節は 1 以上（先頭に 0 を付けない）、
#:   拍は小数2桁（`1.50`・整数部も先頭に 0 を重ねない）、数字は ASCII だけ。以前は `\d`（全角の数字も当たる）と
#:   `[\d.]+` で読み、`12:1.5`・`12:1.333`・`12:.5`・`１２:１.５０` も読めて、同じ位置の書き方が何通りもあった。
#:   表（`artic_plan.tick_key`）はいつも小数2桁で出すので、書き写せばこの形になる。
#: ★**前後の空白だけ取り除き、途中の空白は書けない**（2026-10-02 本人「おすすめで」・§0-6・§17 ず-22）。以前は `:` の前後の空白も
#:   無視して読み（`12 : 1.50`）、書き方が何通りにもなった。空白は記法全体で「区切り」の意味（詞では音の区切り・コードの欄では和音の区切り）で、
#:   区切りの要らない位置では書けない字にする（10/2 に調名で決めたのと同じ方針）。所見に書き直した形を添える（`_position_problems`）
_POSITION_RE = re.compile(r"([1-9][0-9]*):((?:0|[1-9][0-9]*)\.[0-9]{2})", re.ASCII)
#: 小数2桁でないだけの拍（`12:1.5`・`12:1.333`・`12:.5`・`12:2`）。読まないが、所見で正しい書き方を添える
_LOOSE_BEAT_RE = re.compile(r"([1-9][0-9]*):([0-9]*\.?[0-9]*)", re.ASCII)
#: 空白（半角・全角・タブ・改行。Python の `\s` は全角の空白も含む。調名の `song_design._SPACE` と同じ）
_SPACE = re.compile(r"\s")


def _read_position(at: str) -> tuple[int, float] | None:
    """`12:1.50` → (12, 1.5)（小節は1始まり・拍は0始まり・小数2桁）。読めなければ None。前後の半角の空白とタブだけ取り除く
    （途中の空白・前後の全角の空白などは読めない・§0-6）。"""
    m = _POSITION_RE.fullmatch(strip_ascii_ws(at))
    if not m:
        return None
    return int(m.group(1)), float(m.group(2))


def _position_problems(at: str, bar_beats: float | None = None) -> list[str]:
    raw = strip_ascii_ws(at)                               # 前後の全角の空白などは `half_width` が先に所見にする
    # ★**途中の空白は書けない**（2026-10-02・上の説明）。空白を除いた形を添え、その形にもほかの所見があれば続けて書く（所見は1つにまとめる＝
    #   型の検証は最初の1つだけを返すため）。空白を除いた形は書き直し方を示すためだけで、読み替えはしない
    if _SPACE.search(raw):
        compact = "".join(raw.split())
        why = "途中に空白を入れない——空白は記法の「区切り」の印で、位置 `小節:拍` には区切りが要らない"
        if not _LOOSE_BEAT_RE.fullmatch(compact):          # 空白を除いても `小節:拍` の形にならない（`5 1.50`＝`:` の代わりに空白）
            return [f"位置が読めない（at={at!r}。{why}。表と同じ `小節:拍`＝半角の数字で、小節は1から・拍は0からの小数2桁"
                    "（`12:1.50`）で書く）"]
        rest = _position_problems(compact, bar_beats)
        return [f"位置 {at!r} の{why}" + (f"（`{compact}` と書く）" if not rest else f"。空白を除いた `{compact}` も: {rest[0]}")]
    got = _read_position(raw)
    if got is None:
        m = _LOOSE_BEAT_RE.fullmatch(raw)
        if not (m and m.group(2) not in ("", ".")):
            return [f"位置が読めない（at={at!r}。表と同じ `小節:拍`＝半角の数字で、小節は1から・拍は0からの小数2桁"
                    "（`12:1.50`）で書く）"]
        bar, beat = int(m.group(1)), float(m.group(2))
        if bar_beats is None or beat < bar_beats - 1e-9:
            return [f"位置 {raw} の拍は小数2桁で書く（表と同じ `{bar}:{beat:.2f}` の形。3等分の位置は `0.33`・`0.67`、8分の3等分は `0.17`・`0.83`）"]
        got = (bar, beat)                                  # 小数2桁でなく、しかも1小節をはみ出す拍 → 下の所見（`13:0.00` の形を添える）
    if bar_beats is not None and got[1] >= bar_beats - 1e-9:
        last = bar_beats - 0.25
        return [f"位置 {raw} の拍が1小節の長さ（{bar_beats:g}拍）を超えている——拍は 0 から数え、この曲の1小節は "
                f"`0.00`〜`{last:.2f}`。次の小節の頭なら `{got[0] + int(got[1] // bar_beats)}:0.00` のように小節を進めて書く"]
    return []


#: **表の位置 `小節:拍`**（奏法・歌の崩し・声部）。★読み方はここ1か所（2026-09-29）——以前は奏法（`artic_plan.at_key`）と
#: 声部（`harmony.at_to_tick`）が同じ正規表現を別々に持ち、`1.2.3` を片方は「読めない」、片方は例外で落としていた。
#: これは拍子を知らない形（読めるかだけ）。工程の検査は拍子を渡した `position_notation` を使う
POSITION = Notation("表の位置（小節:拍）", half_width(_position_problems), _read_position)


def position_notation(bar_beats: float) -> Notation:
    """**拍子を知っている位置の記法**（2026-09-30・記-15）。`bar_beats` は1小節の4分音符の数（`song_design.beats_per_bar`）。

    ★以前は拍の上限を見ず、4/4 で `12:4.00` が読めて、声部では**13小節の頭**に置かれた（奏法・歌の崩しは表の位置と照らすので
      「その位置に音が無い」の所見になり、理由が分かりにくかった）。拍が1小節の長さ以上なら所見にする。読み方（`read`）は `POSITION` と同じ。
    """
    return Notation(f"表の位置（小節:拍・1小節 {bar_beats:g}拍）",
                    half_width(lambda at: _position_problems(at, bar_beats)), _read_position)


# ---- 工程をまたぐ整合性（親が持つ・2026-09-28）----------------------------------------------------------

def section_key(name: str) -> str:
    """区間名の大文字小文字・前後の空白・全角空白を無視した形——**所見に設計の綴りを添えるためだけ**（`section_hint`）。
    照らし合わせには使わない（2026-10-02 から区間名は設計の綴りと完全一致だけ＝`resolve_section`）。"""
    return str(name or "").strip().replace("　", "").lower()


def resolve_section(name: str, table) -> str | None:
    """区間名を、区間名で引く表（設計の区間名の並び・`bars_of` などの辞書）の名前に照らす。**設計の綴りと完全に一致するときだけ**
    その名前を返し、ほかは None。

    ★**区間名の照らし合わせ方はここ1か所**（2026-09-28）。2026-09-28〜10-01 は大文字小文字・前後の空白・途中の全角空白を無視して
      照らし（2026-09-06『青空の境界線』で設計 `Intro` と編曲 `intro` の食い違いから伴奏が全滅した事故の対策）、`StepFixer.canonize` が
      設計の綴りへ書き換えていた。**2026-10-02 本人「おすすめで」（判断6・別名とゆるい照らし方を外す）から完全一致だけ**——ゆるく照らすと
      書き方が何通りにもなり、照らさない読み手（完全一致で引く辞書）で黙って外れる所が残る。違う綴りは設計との照合（`against_design`）が
      所見にし、設計の綴りを添えて（`section_hint`）その工程に直させる（直らなければその工程で止まる）。
      前方一致（chorus→chorus1）は以前からしない（別の区間でありうる）。
      **前後の空白だけは取り除く**（全角の空白・改行も。区間名は設計で決めた綴りに照らす名前で、半角だけの記法ではない＝仕様書 §0-6 の表）
      ——取り除いた形を `canonize` が設計の綴りとして記録する。
    """
    if name in table:
        return name
    s = section_id(name)
    return s if s and s in table else None


def section_id(name: str) -> str:
    """区間名の**照らす形**（前後の空白を全角も取り除く。`resolve_section` と同じ決まり）。

    表の側（いまのグルーヴ・コードの区間名）にも空白が付きうる所で、**2つの区間名が同じ区間か**を見るときに両方へ当てる
    （`resolve_section` は書いた名前の側だけ取り除くので、表の側に空白が付いていると当たらない）。2026-10-03・再レビュー。
    以前はグルーヴ・コードの直しと旋律の検査が完全一致で引き、空白付きの区間名が別の区間として足された／検査が素通りした。
    """
    return str(name or "").strip()


def section_hint(name: str, table) -> str | None:
    """設計の綴りと大文字小文字・途中の空白だけが違う区間名なら、**設計の綴り**を返す（無ければ None）。所見に添えるためだけ——読み替えない。"""
    want = section_key(name)
    if not want or resolve_section(name, table) is not None:
        return None
    return next((k for k in table if section_key(k) == want), None)


def section_name_problem(name: str, table, *, what: str = "設計の区間名") -> str:
    """区間名が表（設計の区間名）に無いときの所見の文（「『Verse』は設計の区間名に無い——設計の綴りは『verse』…」）。あれば空。"""
    if resolve_section(name, table) is not None:
        return ""
    ok = " / ".join(dict.fromkeys(table))
    hint = section_hint(name, table)
    if hint is not None:
        return (f"『{name}』は{what}に無い——設計の綴りは『{hint}』（区間名は設計と1字も違わずに書く。"
                f"大文字小文字・空白の違いも別の名前として扱う）")
    return f"『{name}』は{what}に無い（使えるのは {ok}）"


def against_design(items: Iterable[tuple[str, str, int | None]], sections, *, need_all: bool = False,
                   what: str = "", ordered: bool = False) -> list[str]:
    """**区間名と小節数が設計と合うか**（工程をまたぐ整合性。決まりはここ1か所・所見は書いた工程へ返る）。

    `items` は工程が書いたもの (場所, 区間名, 小節数 or None)。`sections` は設計の区間（`kind`・`bars`）。
    `need_all` なら、設計の区間のうち工程が1つも書かなかったものも所見にする（`what` はその工程が書くもの）。
    名前は設計の綴りと完全一致だけ（`resolve_section`・2026-10-02 から。以前は大文字小文字・空白の違いを吸収して `StepFixer.canonize` が
    設計の綴りへそろえていた）。違う名前は所見にし、大文字小文字・空白だけの違いなら設計の綴りを添える（`section_name_problem`）。
    """
    names = [s.kind for s in sections]
    bars = {s.kind: int(getattr(s, "bars", 0) or 0) for s in sections}
    items = list(items)
    out = [f"{w}: {why}" for w, n, _b in items if (why := section_name_problem(n, names))]
    items = [(w, resolve_section(n, names) or n, b) for w, n, b in items]   # 前後の空白だけ取り除いた設計の綴り（§0-6）
    for w, n, b in items:
        if n in bars and b is not None and bars[n] and b != bars[n]:
            out.append(f"{w}: {b}小節ぶん（設計は{bars[n]}小節）")
    if need_all:
        seen = {n for _w, n, _b in items}
        out += [f"区間『{k}』の{what}が無い（設計にある区間。書かないとこの区間は鳴らない）" for k in names if k not in seen]
    if ordered:
        # 書いた順に区間が並んでいるか。同じ区間を離れた所で2回書く／設計と違う順で書くと、組み立ては区間ごとに
        # 続けて置くので並びが入れ替わって鳴る（「verse→chorus→verse」が「verse+verse→chorus」に）
        runs: list[str] = []
        for _w, n, _b in items:
            if n in bars and (not runs or runs[-1] != n):
                runs.append(n)
        if len(runs) != len(set(runs)) or runs != [n for n in names if n in runs]:
            out.append(f"区間の並びが設計と違う（書いた順: {'→'.join(runs)}）。設計の順（{'→'.join(names)}）で、"
                       f"同じ区間は続けて書く（組み立ては区間ごとに続けて置くので、並びが入れ替わって鳴る）")
    return out
