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

"""伴奏の置き方（パラメトリック ArrangementSpec を音の並びへ）。

固定パターン名でなく、PartSpec.steps（StepEvent の配列）を「各コード小節に適用」して
任意のリズム/ボイシングを生成する（`pitched_events`）。ドラムは drum_steps（フィルも LLM が `at_bars` で書いたパターン）。
どの小節で鳴るか（`_bar_match`）・曲の小節数（`song_bars`）・和音の読み（`chord_notes`）もここが1つだけ持つ。

★鳴らすのは `midi_out`（MIDI）→ SoundFont（`render_sf`）。ここにあった numpy の合成音（`render_accompaniment`・
  `_render_pitched`・`_render_drums`）は 2026-09-29 に消した——SoundFont が使えないときの落とし先で、
  ピアノを鉄琴の形で・タムをキックで鳴らしていた（代わりのもので鳴らす）。いまは落とさず止める（`song_pipeline._render_accomp`）。
"""
from __future__ import annotations

import re
from typing import NamedTuple

from song_checks import strip_ascii_ws                     # 半角の記法の前後の空白（§0-6・2026-10-03）
from song_schema import PartSpec, SongScore

_ROOTS = {"C": 60, "D": 62, "E": 64, "F": 65, "G": 67, "A": 69, "B": 71}
Q = 480

#: ★**`tones=inversion`（転回形で滑らかにつなぐ）の印**（2026-09-27・目録の 伴奏-1）。`Tone.degree` にこの値を入れる。
#:   以前は `tones=root` のベース（役割名が bass か `register_octave<=-2`）の根音を、**機械が黙って**3度・5度の転回へ
#:   差し替えていた（並行5度を減らすため・2026-09-11）。LLM は「根音」と書いたつもりで、違う音が鳴っていた。
#:   いまは **`root` は根音**、転回でつなぐのは **LLM が `inversion` を選んだ打鍵だけ**（どの転回かは `bass_line` の
#:   声部進行で決まる——その選び方ごと LLM が選んだ、という扱い。指示書に書いてある）。
INVERSION = -1

#: ★**`tones=arp_up` / `arp_down`（分散和音）の印**（2026-09-27）。`Tone.degree` にこの値を入れる。
#:   以前は `_grid_steps` が `degree = 打点の番号 % 3` で書いていたので、**7度・テンションが鳴らず**（Am7 が Am の分散）、
#:   打点の番号を**小節の頭から**数えるため、1小節に和音が2つある小節では**後半の和音が前の和音の続きの順**から始まった。
#:   いまは組み立て（`pitched_events`）が**その和音の構成音（`chord_notes` の全部）を下から順に**たどり、
#:   **小節の頭と和音が替わった所で、その和音の最初の音から**始める。何をするかは伴奏の指示（`tones` の説明）に書いてある。
ARP_UP = -2
ARP_DOWN = -3

#: 転回形の高さを比べる基準のオクターブ（ベースの `register_octave`）。`bass_line` の高さはこのオクターブでの高さで、
#: 鳴らすときは**そのパートのオクターブとの差だけ**ずらす（`register_octave` は度数で書いた音に効く、の規則どおり）。
_INV_REF = -2


def _slash_key(notes: list[int], slash: int, oct_: int) -> int:
    """分数コードの分母の音の高さ（根音に積んだ高さ `notes[0]+oct_` から近いほう）。
    **比べる高さ（`bass_line`）と鳴らす高さ（`pitched_events`）はこの1つで決める**。"""
    base = notes[0] + oct_
    k = base + (slash - base) % 12
    return k - 12 if k - base > 6 else k


def bass_line(chords) -> dict[int, tuple[int, int]]:
    """**`tones=inversion` の打鍵が弾く音**をコードごとに選ぶ。戻り値は `{コードの開始tick: (度数, 高さ)}`。

    高さは **`register_octave=-2`（`_INV_REF`）で鳴らすときの MIDI の高さ**。`pitched_events` はこの高さを
    **そのまま**（パートのオクターブが違えば、その差のオクターブだけずらして）鳴らす。

    ★**なぜ転回形か**（2026-09-11）: ベースが常に根音だと、コード変化の 85% で並行5度が出ていた
      （洋楽ヒット141曲のピアノ×ベースは p95 で 0.071、うちは 0.241〜0.300）。伴奏は度数で音を引くので
      オクターブを置き直しても変わらず、**弾く度数そのもの**を変える必要があった。選び方は ①ベースの動きが小さい
      ②根音ばかりにしない（和声の定石・試算で並行5度 0.875 → 0.125〜0.200）。
      （度数だけを返す薄い包み `bass_degrees` は呼ばれていなかったので 2026-09-28 に消した）

    ★**比べた高さと鳴らす高さを1つにした**（2026-09-27）。以前は選ぶときに低音域へ畳んだ高さ
      （`36 + (音-36)%12`）で「前の低音にいちばん近い音」を比べ、鳴らすときは根音から積んだ高さ（`notes[度数]`）で
      弾いていたので、3度・5度が根音より1オクターブ上に来る和音（Am の C・F の C など）で**選んだ音の1オクターブ上**が鳴り、
      下降ラインが崩れた（『風は海を越えて』で36和音中7か所。A2 G2 F2 E2 D2 C2 のはずが C3 へ跳ねた）。
    """
    out: dict[int, tuple[int, int]] = {}
    prev: int | None = None
    ref = _INV_REF * 12
    for span in sorted(chords, key=lambda c: c.start_tick):
        ns = chord_notes(span.chord)
        slash = chord_bass_pc(span.chord)
        if slash is not None and ns:                       # 分数コードは低音が決まっている（`pitched_events` が弾く）
            prev = _slash_key(ns, slash, ref)
            continue
        # sus の 4度・2度は転回の低音にしない（和音が崩れて聞こえる）
        cands = [(deg, 36 + (ns[deg] - 36) % 12 + o * 12)
                 for deg in range(min(3, len(ns))) for o in (0, 1)
                 if not (deg == 1 and ns[1] - ns[0] not in (3, 4))]
        if not cands:                                      # ★`N.C.` 等。そこは弾かない（落とさない）
            continue
        if prev is None:
            best = min(cands, key=lambda x: abs(x[1] - 43))
        else:                                              # 動きが小さい／根音以外にも少し寄せる
            best = min(cands, key=lambda x: abs(x[1] - prev) + (0.0 if x[0] == 0 else 1.5))
        out[int(span.start_tick)] = best
        prev = best[1]
    return out


def arp_order(notes: list[int], down: bool = False) -> list[int]:
    """分散和音でたどる順＝**その和音の構成音（`chord_notes` の全部＝7度・6度・テンションを含む）を下から**（下行は上から）。
    `tones=chord` の打鍵が同時に鳴らす音と同じ集まり（構成音の決め方は `chord_notes` の1か所）。"""
    xs = sorted(notes)
    return xs[::-1] if down else xs


# ---- コード名の文法（記法の仕様書 §3）——検査（`song_design.chord_problem`・`chord_readable`・`CHORD_BAR`）と
#      組み立て（`chord_notes`・`chord_bass_pc`）が**この1つ**で読む（2026-10-02・組「洗い出し-コードとABC」）----------------
#
# ★**種類は決まった順に1つずつ書く**（本人決定 2026-10-02「字の袋を並びに」）。以前の文法は種類の字の**袋**
#   （`{ 6/9 | maj | m | 7 | # | b | ( | ) | , … }` を何個でも・どの順でも）で、`C7b`・`Cadd`・`C7(9`・`Cmm7`・`C79`・`Am7,` が通り、
#   組み立ては分からない字を黙って無視していた（部分文字列で読む）。全角の数字（`Am７`）も `\d` が通し、組み立ては全角の所を捨てて
#   7度が鳴らず、先頭の 0（`C7(b09)`・`Bbmaj07`）は ♭の付かない9度・短7度で鳴っていた（洗い出しの H2）。
# ★**同じ和音の書き方は1通り**（本人決定 2026-10-02「おすすめで」・判断3）: 和音なしは `N.C.`、長三和音は `C`（`CM`・`Cmaj` は書けない）、
#   短調は `m`（`min` は書けない）、長7は `maj7`（`M7`・`Δ` は書けない）、減は `dim`（`°`）、半減7は `m7b5`（`ø`・`m7-5`）、
#   増三和音は `aug`（`+`）・7度つきの増5度は `7#5`、♭5 は `b5`（`-5`）。
#   7の和音の**数字は、含むナチュラルのテンションのうちいちばん低いもの**（9 があれば `9`・無くて 11 があれば `11`・13 だけなら `13`・
#   無ければ `7`。`C7(9)` は `C9`・`C7(11)` は `C11`・`C7(9,13)` は `C9(13)`）。括弧には**変化したテンションと、数字より上のナチュラル**を
#   低い順に（`C7(b9,#9)`・`C9(#11)`・`C13(b9)`・`Cmaj9(13)`。`-9` `+9` `+11` `-13` と括弧の無い `C7b9` は書けない）。
#   ※組み立ては数字の 11・13 の和音に9度を足さない（`C13`＝根・3・5・♭7・13。省くのは普通の積み方・2026-09-30 から同じ）。
#   書けない書き方は検査が書き直した形を添えて所見にする（`song_design.chord_problem`）。
#
#   種類 = [ m | dim | aug ] [ 6 | 7 | 9 | 11 | 13 | maj7 | maj9 | maj11 | maj13 ] [ b5 | #5 ] [ sus4 | sus2 ] [ add9 | add11 ]
#          [ "(" テンション { "," テンション } ")" ]
#
#   組み合わせの決まりは `_chord_shape`。数字は半角だけ・頭に 0 を付けない（文法が数字を名前で持つので、`７` `07` は読めない）。
_CHORD_RE = re.compile(
    r"^(?P<root>[A-G][#b]?)"
    r"(?P<q>(?P<tri>m|dim|aug)?(?P<ext>maj7|maj9|maj11|maj13|6|7|9|11|13)?(?P<five>[b#]5)?(?P<sus>sus[24])?"
    r"(?P<add>add(?:9|11))?(?:\((?P<paren>[^()]*)\))?)"
    r"(?:/(?P<bass>[A-G][#b]?))?$")
#: 括弧の中に書ける音（低い順＝書く順）と、根音からの半音
TENSION_ORDER = {"b5": 6, "b6": 8, "b9": 13, "9": 14, "#9": 15, "11": 17, "#11": 18, "b13": 20, "13": 21}
_NATURAL = ("9", "11", "13")


class ChordParts(NamedTuple):
    """`parse_chord` の結果。`root` は根音の高さ（C4〜B4 の MIDI 番号・変化記号ぶんずらす）、`bass` は分数の低音の高さのクラス。"""
    root: int
    tri: str                                               # "" | m | dim | aug
    ext: str                                               # "" | 6 | 7 | 9 | 11 | 13 | maj7 | maj9 | maj11 | maj13
    five: str                                              # "" | b5 | #5
    sus: str                                               # "" | sus4 | sus2
    add: str                                               # "" | add9 | add11
    paren: tuple[str, ...]                                 # 括弧の中の音（低い順）
    bass: int | None


def _pc(note: str) -> int:
    return _ROOTS[note[0]] + {"#": 1, "b": -1}.get(note[1:2], 0)


def _chord_shape(tri: str, ext: str, five: str, sus: str, add: str, paren: tuple[str, ...] | None) -> bool:
    """**種類の組み合わせの決まり**（並びの文法 `_CHORD_RE` の上で、同じ和音が2通りに書けないようにする）。"""
    num = ext.removeprefix("maj")                           # 7 / 9 / 11 / 13 / 6 / ""
    has7 = num in ("7", *_NATURAL)
    if tri == "aug" and (ext or five or sus or add or paren is not None):
        return False                                       # 7度つきの増5度は `C7#5`（`Caug7` は書けない）
    if tri == "dim" and (ext not in ("", "7") or five or sus or add):
        return False                                       # 減三和音 `dim`・減7 `dim7` だけ
    if five and not (has7 and (tri == "" or (tri == "m" and five == "b5" and not ext.startswith("maj")))):
        return False                                       # 5度の変化は7の和音の数字の後ろ（短調では半減7 `m7b5`・`m9b5` …だけ）
    if sus and tri:
        return False                                       # sus は3度の代わり（`m` と一緒に書けない）
    if add and has7:
        return False                                       # 7度のある和音は数字と括弧（`C9`・`C7(11)`）。add は7度の無い和音だけ
    if paren is None:
        return True
    if not paren or len(set(paren)) != len(paren) or any(t not in TENSION_ORDER for t in paren):
        return False
    if [TENSION_ORDER[t] for t in paren] != sorted(TENSION_ORDER[t] for t in paren):
        return False                                       # 低い順に書く（`C7(#9,b9)` は書けない）
    if has7:                                               # 7の和音（減7 `dim7` も）
        if any(t in ("b5", "b6") for t in paren):
            return False                                   # 7の和音の ♭5 は括弧の外（`C7b5`）、♭6 は `b13`
        low = int(num) if num in _NATURAL else 99
        if any(t in _NATURAL and int(t) <= low for t in paren):
            return False                                   # 数字より下・同じナチュラルは括弧に書かない（`C7(9)` → `C9`・`C11(9)` → `C9(11)`）
        have = set(paren) | ({num} if num in _NATURAL else set())
        if {"9", "b9"} <= have or {"9", "#9"} <= have or {"11", "#11"} <= have or {"13", "b13"} <= have:
            return False                                   # 同じ度数のナチュラルと変化を重ねない（`C9(b9)`）
        return True
    # 7度の無い和音の括弧は ♭5（長三和音だけ・`C(b5)`）と ♭6（`Cm(b6)`）だけ。テンションは add で書く（`C(9)` → `Cadd9`）
    return not ext and not add and all(t == "b6" or (t == "b5" and tri == "") for t in paren)


def parse_chord(name: str) -> ChordParts | None:
    """**コード名を文法（仕様書 §3）どおりに読む**。読めなければ None（`N.C.` も None——和音が無い）。

    前後は半角の空白とタブだけ取り除く（§0-6・2026-10-03。前後の全角の空白・改行などは読めない＝検査の所見）。検査（`song_design.chord_readable`・`chord_problem`）と組み立て（`chord_notes`・`chord_bass_pc`）が
    これ1つで読む——検査が通したものだけが鳴り、組み立ては書いていない字を読み飛ばさない。
    """
    m = _CHORD_RE.match(strip_ascii_ws(name))
    if not m:
        return None
    paren = tuple(m["paren"].split(",")) if m["paren"] is not None else None
    if not _chord_shape(m["tri"] or "", m["ext"] or "", m["five"] or "", m["sus"] or "", m["add"] or "", paren):
        return None
    return ChordParts(_pc(m["root"]), m["tri"] or "", m["ext"] or "", m["five"] or "", m["sus"] or "", m["add"] or "",
                      paren or (), _pc(m["bass"]) % 12 if m["bass"] else None)


def chord_notes(name: str) -> list[int]:
    """コード名 → 構成音 [root, 3rd, 5th, (7th/6th), テンション…]（degree index で参照する）。

    ★**積み方（オクターブ）は決めない**。ベースが根音以外を弾くかは `bass_line` が決める。

    ★**読めない綴りは空で返す**（2026-09-23）。`N.C.`（ノーコード＝そこは和音を鳴らさない）が
      1つ入っていただけで **`KeyError: 'N'` で曲が丸ごと落ちた**。`N.C.` は音楽的に正しい書き方で、
      コード工程が書いてくる。**読めない名前で落とさず、その区間を鳴らさないだけにする**
      （`_chord_root` は前から None を返す作りで、**こちらだけが落ちていた**）。
      `N.C.` 以外の読めない名前は検査（`song_design.chord_problem`）が所見にし、組み立ての入口（`song_place.place_chords`）が止めるので届かない。
    ★**文法（`parse_chord`）で読む**（2026-10-02）。以前は種類を部分文字列で探し（`"7" in q` など）、書いていない字・並びの崩れた字を
      黙って無視していた（`C7b` は C7・`Cadd` は C・`Am７` は Am・`C7(b09)` は ♭の付かない9度）。
    ★並びは **[根, 3度（sus なら4度/2度）, 5度, 7度/6度, 変化したテンション（♭9 ♯9 ♯11 ♭13）, 13・11・9]** ——度数 0〜2 の意味は変えない
      （sus・dim・aug・♭5・テンションを読むようにした 2026-09-27、種類の中の変化記号を標準どおり読むようにした 2026-10-01・判-K の並びのまま）。
    """
    p = parse_chord(name)
    if p is None:
        return []
    root = p.root
    third = {"m": 3, "dim": 3}.get(p.tri, 4)
    third = {"sus4": 5, "sus2": 2}.get(p.sus, third)
    fifth = {"dim": 6, "aug": 8}.get(p.tri, 7)
    fifth = {"b5": 6, "#5": 8}.get(p.five, fifth)
    if "b5" in p.paren:                                    # 長三和音の ♭5（`C(b5)`）
        fifth = 6
    notes = [root, root + third, root + fifth]
    num = p.ext.removeprefix("maj")
    if p.ext.startswith("maj"):
        notes.append(root + 11)
    elif p.ext == "7" and p.tri == "dim":
        notes.append(root + 9)                             # 減7
    elif num in ("7", *_NATURAL):
        notes.append(root + 10)
    elif "b6" in p.paren:
        notes.append(root + 8)                             # ♭6（`Cm(b6)`・判-K）
    elif p.ext == "6":
        notes.append(root + 9)
    tens = [t for t in p.paren if t not in ("b5", "b6")]
    for mark in ("b9", "#9", "#11", "b13"):
        if mark in tens:
            notes.append(root + TENSION_ORDER[mark])
    naturals = set(tens) | {p.add[3:], num}                # 数字の 9・11・13 はそのナチュラルのテンション（下は足さない）
    for mark in ("13", "11", "9"):
        if mark in naturals:
            notes.append(root + TENSION_ORDER[mark])
    return notes


def chord_bass_pc(name: str) -> int | None:
    """分数コードの低音（`G/F` の F）の高さのクラス（0〜11）。分数でない・読めないなら None（2026-09-27）。
    ★以前は分子しか読まず、**ベースが分数コードの低音を弾いていなかった**（クリシェ Am→Am/G#→Am/G が崩れた）。
    ★文法（`parse_chord`）で読む（2026-10-02）。以前は `/` の後ろの頭の字だけを読み、和音の側が読めない名前でも低音を返していた。"""
    p = parse_chord(name)
    return None if p is None else p.bass


def _bar_match(bar_idx: int, n_bars: int, bar_filter: str) -> bool:
    """その小節でこのパートが鳴るか。**`bar_filter` の解釈はここだけ**（2026-09-01 統合）。

    ## 決めごと（守らないと1小節ずれる）

    ・`bar_filter` の中の小節番号は **1始まり・両端を含む**（`"9-16"` は 9,10,…,16）
    ・引数の `bar_idx` は **0始まり**（曲頭の小節が 0）。**混ざりやすいので注意**
    ・カンマ区切りは**飛び飛び**（`"1-4,17-24"`）。min〜max でまとめてはいけない
      ——間の小節まで鳴ってしまう
    ・**解釈できない値は「鳴らさない」**

    ## なぜ直したか

    以前は `all / latter / first` しか解さず、**それ以外を「全小節で鳴る」**に
    倒していた（`return True`）。編曲が書く小節指定（ため・フィル・フェードの
    `"42-42"` など）を渡すと、**全パートが全小節で鳴りっぱなし**になる。
    書き間違いが「全部の小節で鳴る」という最悪の形で出る作りでもあった。

    同じ判定が**4箇所**にあって食い違っていたのを、ここへ集約している
    （経緯は `test/song_quality_bench/bar_range.py` の冒頭）。
    """
    f = (bar_filter or "all").strip()
    if f in ("", "all"):
        return True
    if f == "latter":
        return bar_idx >= n_bars // 2
    if f == "first":
        return bar_idx < n_bars // 2
    for seg in f.replace("、", ",").split(","):
        seg = seg.strip()
        if not seg:
            continue
        try:
            if "-" in seg:
                a, b = (int(x) for x in seg.split("-", 1))
            else:
                a = b = int(seg)                           # **単独の小節番号も受ける**
        except ValueError:
            continue                                       # 読めない断片は無視（全体は落とさない）
        if a - 1 <= bar_idx <= b - 1:                      # 1始まり → 0始まり
            return True
    return False                                           # 解釈できなければ**鳴らさない**


def bars_of_filter(bar_filter: str, n_bars: int) -> set[int]:
    """その指定が指す小節の集合（**1始まり**）。検査や集計に使う。"""
    return {b for b in range(1, n_bars + 1) if _bar_match(b - 1, n_bars, bar_filter)}


def to_bar_filter(bars: set[int] | list[int]) -> str:
    """小節番号の集合 → `bar_filter` の文字列（**1始まり・両端を含む**）。

    連続するものは範囲にまとめる（`{3,4,5,9}` → `"3-5,9"`）。
    **飛びは繋げない**——繋ぐと鳴らないはずの小節で鳴る。
    """
    xs = sorted(set(bars))
    if not xs:
        return "all"
    out, a, prev = [], xs[0], xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append((a, prev))
        a = prev = x
    out.append((a, prev))
    return ",".join(f"{x}-{y}" if x != y else f"{x}" for x, y in out)


def song_bars(score: SongScore) -> int:
    """曲の小節数。**コードの個数ではなく、いちばん後ろの tick から出す**。

    ★2026-09-06 に新設。それまで各所が `len(score.chords)` を小節数として使っていたが、
    **1小節に2つコードを置ける**（`chords_per_bar` の `"Am|F"`）ので、両者は一致しない。
    実例: コード 42 個／実際は 29 小節（13 小節が2つに割れていた）。
    """
    ends = [c.start_tick + c.duration_tick for c in score.chords]
    ends += [n.start_tick + n.duration_tick for n in score.vocal_notes]
    bt = score_bar_ticks(score)
    return max(1, -(-max(ends, default=bt) // bt))         # 切り上げ


def score_bar_ticks(score: SongScore) -> int:
    """その曲の1小節の tick 数（**拍子から**・2026-09-27）。以前は 4/4 の決め打ち（`BEATS * Q`）で、
    3/4 の曲では小節数も小節の位置も 4/4 で数えていた（ドラムが曲の途中で終わる・`bar_filter` がずれる）。"""
    from song_design import bar_ticks
    return bar_ticks(score.time_signature or "4/4")


def pitched_events(score: SongScore, part: PartSpec) -> list[tuple[int, int, list[int], int]]:
    """音程パートの `steps`（1小節のパターン）を曲に置く。戻り値は **(開始tick, 長さ, 鳴る高さの並び, 強さ)**。

    ★**置き方はここ1つ**（2026-09-27）。numpy（当時の `_render_pitched`・2026-09-29 に削除）と MIDI（`midi_out.score_to_midi`）が
      別々に書いていて食い違っていた——**書いた高さ（`tone.key`）を MIDI 側だけ捨て**、対旋律・リフが
      根音の連打になっていた（例: ミレドラ → ミミミミ）。2本あると片方だけ直る。

    決めごと:
      ・**拍の位置は小節の頭から数える**。1小節に和音が2つある小節では、後半の拍は後半の和音で鳴る
        （以前は和音ごとに頭から数えていたので、**後半に書いた音が消え、前半の形をなぞり直していた**）
      ・前の和音から鳴り続けている音は、**和音が変わったところで新しい和音で弾き直す**（全音符の和音が後半も鳴る）
      ・**高さを書いた音は、書いた高さで鳴る**。`register_octave` は度数で書いた音にだけ効く
        （彩りは「E5」と絶対の高さで書くので、足すと1〜2オクターブ上へ二重にずれた）。和音が変わっても切らない
      ・和音を同時に鳴らす打鍵（度数 0・1・2 がそろう）は、**7度・6度・テンションも鳴らす**（Fmaj7 が F に聞こえない）
      ・**`root` は根音**。転回形（`bass_line`）は **`tones=inversion`（`INVERSION`）の打鍵だけ**（2026-09-27・伴奏-1。
        以前はベースの `root` を機械が黙って転回に差し替えていた）
      ・**分数コード**（`G/F`）では、ベースのパート（`part.bass`）の `root` と、`inversion` の打鍵は
        **分母の音**を弾く（コード記号の意味どおり＝組み立て。指示書に書いてある）。ベースのパートは和音に7度・テンションを積まない
      ・**ベースかどうかは `part.bass` だけを読む**（2026-10-02 本人決定・案b。伴奏の LLM が `PartPlan.bass` に明示する）。
        以前は表示用の `role` の文字（bass で始まる）と `register_octave <= -2` から推し量っていた
      ・**`inversion` は `bass_line` が選んだ高さそのもの**を鳴らす（パートのオクターブがベースと違えば、その差だけずらす）。
        以前は選んだ度数を根音から積み直して鳴らし、選んだ音の1オクターブ上が鳴ることがあった（2026-09-27）
      ・**`arp_up` / `arp_down` は、その和音の構成音（7度・テンションを含む）を下から（上から）順に**たどる。
        **小節の頭と、和音が替わった所で、その和音の最初の音から**始める（2026-09-27。以前は3音だけを小節の頭から数えた）
      ・**鳴らせる高さ（MIDI 0〜127）の外の音は鳴らさない**（2026-09-28・組み立ては補正しない。以前は 0・127 に張り付けた）
    """
    bt = score_bar_ticks(score)
    n_bars = song_bars(score)
    is_bass = part.bass
    walk = any(t.degree == INVERSION for st in part.steps for t in st.tones)
    inv = bass_line(score.chords) if walk else {}
    arp_ats = sorted({float(st.at_beat) for st in part.steps
                      if any(t.degree in (ARP_UP, ARP_DOWN) and getattr(t, "key", None) is None for t in st.tones)})
    out: list[tuple[int, int, list[int], int]] = []
    for span in sorted(score.chords, key=lambda c: c.start_tick):
        s0, s1 = int(span.start_tick), int(span.start_tick + span.duration_tick)
        notes = chord_notes(span.chord)
        slash = chord_bass_pc(span.chord)
        for bar_i in range(s0 // bt, max(s0 // bt + 1, -(-s1 // bt))):   # 2小節以上続く和音は、その小節すべて
            if not _bar_match(bar_i, n_bars, part.bar_filter):
                continue
            bar0 = bar_i * bt
            from_ = max(bar0, s0)                          # 分散和音を数え始める所（小節の頭か、和音が替わった所）
            for st in part.steps:
                on = bar0 + int(st.at_beat * Q)
                end = on + int(st.dur_beats * Q)
                if on >= bar0 + bt:                        # 小節からはみ出したパターンは置かない
                    continue
                # ★**鳴らせる高さ（MIDI 0〜127）の外は鳴らさない**（2026-09-28・組み立ては補正しない）。
                #   以前は 0・127 に張り付けて別の高さで鳴らしていた。書いた高さの外れは伴奏の検査が所見にする——
                #   区間の移調量を足した**鳴る高さ**で（`song_arrange._sounding_issues`・2026-10-03。以前は移調前の高さだけ見ていて、
                #   移すと 127 を超える音がここで黙って鳴らなかった）。楽譜を移調する道具は `song_edits.transpose_problem` で断る
                keyed = [int(t.key) for t in st.tones if getattr(t, "key", None) is not None and 0 <= int(t.key) <= 127]
                if keyed and s0 <= on < s1:
                    out.append((on, max(1, min(end, bar0 + bt) - on), keyed, st.velocity))
                degs = [t for t in st.tones if getattr(t, "key", None) is None]
                if not degs:
                    continue
                if s0 <= on < s1:
                    start = on
                elif on < s0 < end:                        # 前の和音から鳴り続けている → ここで弾き直す
                    start = s0
                else:
                    continue
                dur = min(end, s1) - start
                if dur <= 0:
                    continue
                keys: list[int] = []
                for tone in degs:
                    oct_ = (tone.octave + part.register_octave) * 12
                    low = tone.degree == INVERSION or (is_bass and tone.degree == 0)
                    if low and slash is not None and notes:
                        keys.append(_slash_key(notes, slash, oct_))
                        continue
                    if tone.degree == INVERSION:
                        if s0 in inv:                      # 選んだ高さそのもの（オクターブはベースとの差だけずらす）
                            keys.append(inv[s0][1] + oct_ - _INV_REF * 12)
                        continue
                    if tone.degree in (ARP_UP, ARP_DOWN):
                        if notes:                          # この和音で何番目の打鍵か（小節の頭／和音が替わった所から数える）
                            i = sum(1 for a in arp_ats if from_ <= bar0 + int(a * Q) < start)
                            seq = arp_order(notes, down=tone.degree == ARP_DOWN)
                            keys.append(seq[i % len(seq)] + oct_)
                        continue
                    if 0 <= tone.degree < len(notes):
                        keys.append(notes[tone.degree] + oct_)
                if not is_bass and {0, 1, 2} <= {t.degree for t in degs}:
                    o0 = next(t.octave for t in degs if t.degree == 0)
                    keys += [x + (o0 + part.register_octave) * 12 for x in notes[3:]]
                keys = [k for k in keys if 0 <= k <= 127]          # 鳴らせる高さの外は鳴らさない（張り付けない・2026-09-28）
                if keys:
                    out.append((start, dur, keys, st.velocity))
            # ★**打鍵の無い和音を機械が足すことはしない**（2026-09-27 に一度足して、同日に外した）。
            #   `-` で伸ばした音は上の「弾き直し」で後半の和音にも鳴るので、後半が無音になるのは
            #   **`_` で止めたとき**＝ため・キメとして LLM が書いた無音。足すとそれを上書きする
            #   （プログラムが黙って意図を変える）。規則は伴奏の指示（`_GRID`）で LLM に知らせてある。
    return out
