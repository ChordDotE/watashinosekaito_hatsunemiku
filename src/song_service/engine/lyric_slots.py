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

"""詞の**譜割り**——フレーズの詞を「旋律の1音ごとの印」として読む（2026-09-29）。

★**ABC 記法の歌詞行（`w:`）に寄せた**（本人判断 2026-09-29）。詞は以前「かなの列」で、どの音にどのモーラを
  乗せるかを持っていなかった。組み立てが先頭から1音1モーラで当てるだけなので、「ここは2モーラを1音に詰める」を
  **書く手段が無かった**（『これが証だ』の「さびついたいいわけ」＝8音に9モーラ）。人の作詞（曲先）は、デモを
  口ずさみながら言葉をはめるので、**言葉を選ぶ時点で譜割りも決まっている**——詞の工程に言葉と印を一緒に書かせる。

    さ び つ い た い~い わ け    ← 空白で次の音へ。1つの印＝旋律の1音
    そ ら _ _ に                  ← `_` は前の音節をもう1音伸ばす（メリスマ）
    き み * の な ま え           ← `*` はその音を歌わない

- `~` で別々の音節を1つの音に詰める（`い~い`＝「言い」・`ん~で`）＝**その音を等分して歌う**。
  ★**詰めの標準は促音・撥音を前の音に収めるときだけ**（`け~っ`・`ほ~ん`・2026-10-06 本人決定・`STD_JOIN_MORAS`）。ふつうの2音節の詰めは
  記法としては書けるが標準外で、詞の工程の形式の検査（`song_lyrics.LyricsStep`）が理由（`PhraseLyric.join_reason`）を求める（`plain_joins`）
  ★**V6 は1つの音に2モーラを歌えない**（2026-09-29 実測・`dev_tools/scratch/s5065b362_twomora.py`）——
  「かた」を1音に渡すと libresvip が発音記号を作れず「あ」1音、発音記号を `k a t a` とつないでも「か」1音のまま。
  だから組み立てが音を等分して1音1モーラに割る（書かれた詰めを音にするだけで、書かれたものは変えない）
- `~` の無い1つの区切りは**1つの音節**。長音（`にい`・`びょお`）は前のモーラのまま1音で伸ばす。
  長音でないモーラ（`しん` の「ん」・`かっ` の「っ」）は、`~` と同じく等分して鳴らし分ける（`Slot` の説明）
- **空白の無い詞は、まるごと1つの区切り（1音）として読む**。4モーラ以上なら書式の所見（`problems`）、
  フレーズが2音以上なら印の数の所見（詞の工程の `LyricsStep`）で返る。
  ※2026-09-29〜30 は、空白も印も無い詞を以前の形（1文字＝1モーラ＝1音）として読んでいた。2026-09-30 に外した（古い記録は読まない）
- `|`（小節）は使わない。ABC では「残りの音を飛ばして次の小節へ」の意味を持ち、フレーズと小節の境目が
  一致しない曲で数え違いを誘う。ABC の `-`（語の中の音節の区切り）も使わない（区切りは空白だけ）。どちらも
  `song_lyrics.check_kana` が標準での意味と書き方を添えて所見にする（2026-10-01）
- ★**タイでつないだ音は、書いた音の数だけ印を書く**（2026-10-01・判-H 本人「標準に寄せる方式で全体的に」）。ABC の歌詞行 `w:` は
  タイの2つの音を別々に数える。旋律の組み立て（`abc_to_phrases`）はタイを1音にまとめ、まとめた続きの音の数を `ties` に持つ。
  詞はその音に `1 + ties` 個の印を書き、**続きの音の印は `_`**（前の音節を伸ばす。歌わない音の続きなら `*`）。
  組み立ては続きの `_` を前の音に含める（1つの長い音のまま＝以前と同じ鳴り方）。数え方は `slot_count`・照らし方は `tie_problems`。
  ※2026-09-29〜10-01 はタイの2音を1つの印と数えていた（LilyPond の既定と同じ）
- **区切りは空白だけ**。`・` `|` `/` は読み飛ばさず、かなとして数える（ひらがな以外なので `song_lyrics.check_kana` の所見に出る）。
  ※2026-09-30 まで、以前の形で語の切れ目に書かれがちだった `・` `|` `/` も空白と同じく黙って落としていた——
  指示書「記号は空白と `~ _ *` だけ」と食い違い、譜割りつきの詞でも書き違いが所見にならなかった
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: 1つの音に詰めてよいモーラ数の上限。それ以上は旋律の側で音を足すべき（区切りの空白が抜けたのかもしれない）
SQUEEZE_MAX = 3
HOLD = "_"
SKIP = "*"
JOIN = "~"

_BLANK = " 　\t\n\r"                                     # 空白だけ（`・` `|` `/` は落とさない。上の説明）

#: **カバーの歌詞のかなの行の切れ目**（2026-10-03 本人決定・仕様書 §2「カバーの歌詞のかな」）。改行1つ＝元の歌詞の行の切れ目。
#: 割り当て・強制整列で字を数えるときは、この**定義された区切り**を除いて数える（黙った読み飛ばしではない）。
#: 割り当ての LLM には行ごとに見せる（行の切れ目は材料。機械は行の切れ目で割り当てを強制しない）
KANA_LINE_BREAK = "\n"


def kana_text_lines(text: str) -> list[str]:
    """カバーの歌詞のかな → **元の歌詞の行**（前後の空白は取り除く・§0-6）。空なら []。"""
    t = (text or "").strip()
    return t.split(KANA_LINE_BREAK) if t else []


def kana_text_chars(text: str) -> list[str]:
    """カバーの歌詞のかな → **歌うかなの字の並び**（行の切れ目＝定義された区切りを除く）。"""
    return [c for line in kana_text_lines(text) for c in line]

#: **前の字と繋がって1モーラになる小書きのかな**（拗音＋外来語の小書き母音）。
#: 「っ」は入れない——促音はそれ自体で1モーラを占める（`song_place._sustain_sokuon` が扱う）。
#: ★「ゎ」は入れない（2026-10-01・仕様書 §17 ず-16 本人決定）。「ゎ」「ゔ」は**書けない字**（`song_lyrics.check_kana` の所見・
#:   「わ」「ぶ」で書く）。以前はここだけ「ゎ」を小書きとして前の字につなぎ、歌える字の一覧には無かった（数え方と検査が食い違っていた）
_SMALL_KANA = "ゃゅょぁぃぅぇぉ"


def moras(text: str) -> list[str]:
    """かな列 → **モーラ列**。拗音（きゃ・しゅ 等）は2文字で1モーラなので繋げる。

    文字数で数えると「しんこきゅうお」が7と出るが、歌う単位は
    し・ん・こ・きゅ・う・お の**6モーラ**。1モーラ＝1音符なので、ここを間違えると
    音符への割り当てがずれる（拗音が「き」「ゅ」に分かれて歌えなくなる）。

    ★**空白は落とす**（2026-08-22）。作詞側は読みやすさのために
    「こたえなんか わからなくても」のように**語の切れ目へ空白を入れる**が、
    空白は歌う単位ではない。数に入れると**空白の音符**ができて V6 で歌えなくなる
    （実測: 303音のうち2音が空白のまま書き出された）。
    ※2026-09-29 から詞の空白は「次の音へ」の印になった（`slots`）。ここは印を外したあとのかなを数える。

    ★**小書き母音（ぁぃぅぇぉ）も前と繋げる**（2026-09-15）。外来語の
    「ふぉ」「てぃ」「うぃ」「しぇ」「つぁ」は**2文字で1モーラ**で、拗音とまったく同じ。
    ヤ行（ゃゅょ）だけ繋いでいたので、**「フォーマット」が『ふ』『ぉ』の2音に割れて**
    「ふ・お・お・ま・っ・と」と歌われた（2026-09-15『からっぽだ』の note[266]）。
    ※小書きの「っ」は**別**（それ自体で1モーラ）。
    """
    out: list[str] = []
    for c in text:
        if c in _BLANK or c in (HOLD, SKIP, JOIN):         # 空白・印は歌わない
            continue
        if c in _SMALL_KANA and out:                       # 拗音・外来語の小書き母音は前と繋がる
            out[-1] += c
        else:
            out.append(c)
    return out


_ROWS = {"あ": "あかさたなはまやらわがざだばぱゃぁ", "い": "いきしちにひみりぎじぢびぴぃ",
         "う": "うくすつぬふむゆるぐずづぶぷゅぅ", "え": "えけせてねへめれげぜでべぺぇ",
         "お": "おこそとのほもよろをごぞどぼぽょぉ"}
_VOWEL_OF = {c: v for v, row in _ROWS.items() for c in row}


def _vowel(mora: str) -> str:
    """モーラの母音（拗音は小書きの字で決まる）。「ん」「っ」は空。"""
    return _VOWEL_OF.get(mora[-1], "") if mora else ""


def _is_long(prev: str, mora: str) -> bool:
    """`mora` が `prev` の**長音**（前の母音をそのまま伸ばす）か。「にい」の「い」・「びょお」「きょう」の「お」「う」。"""
    v = _vowel(prev)
    return mora in "あいうえお" and bool(v) and (mora == v or (mora == "う" and v == "お"))


def _syllable_parts(syl: str) -> tuple[str, ...]:
    """1つの音節（`~` で区切らない塊）の中で、**鳴らし分けるモーラ**。長音は前のモーラに含めて1つの音にする。"""
    out: list[str] = []
    for m in moras(syl):
        if out and _is_long(out[-1], m):
            continue                                        # 長音: 前の音のまま伸ばす（V6 は1音の中で母音を保つ）
        out.append(m)
    return tuple(out)


def parts_of(cell: str) -> tuple[str, ...]:
    """**1つの音に書いたかな**（1つの区切り）→ その音の中で**鳴らし分けるもの**。2つ以上なら、その音を等分して歌う。

    `~` で区切った別々の音節はそれぞれ鳴らし分け、`~` の無い塊の中の長音（`にい`・`きょう`）は前のモーラのまま1つにする。
    ★**1つの音にかなを当てる経路は、全部これで読む**（2026-09-29）。新しい曲（`slots`）とカバーの割り当て
      （`align_reference.align_full_lyrics_jp`・`align_cover.align_full_lyrics_audio`）で等分の規則が2本あり、
      カバーは1文字ずつ割っていた——「かあ」の「あ」を言い直し、「きゃ」を「き」「ゃ」の2音に割っていた。
    """
    return tuple(p for syl in (cell or "").split(JOIN) for p in _syllable_parts(syl))


@dataclass(frozen=True)
class Slot:
    """旋律の1音ぶんの詞。`kind` は sing（`parts` を歌う。2つ以上なら等分）／hold（前の音節を伸ばす）／skip（歌わない）。

    `moras` は書かれたかなのモーラ（歌うかな・数の検査に使う）、`parts` は**その音の中で鳴らし分けるもの**。
    ★**長音は音節の一部**（2026-09-29・本人「記法でもそこはそう書くものなの？」）。楽譜の歌詞は音節の単位で付け、
      "day" も「びょう」も1つの音にそのまま書く。ABC・LilyPond の `~` は**別々の音節**を1つの音にまとめる印で、長音には使わない。
      だから `にい`（1つの区切り）は「に」を1音ぶん伸ばし、`い~い`（「言い」の2音節）は等分して「い」「い」と歌い直す。
      最初の実装は両方を等分し、『駆け抜けろスニーカー』の「す に~い か~あ」で後ろの母音が別の音として立ち上がった
      （2026-09-13 の「まあ」の再発音と同じ形）
    """
    kind: str
    moras: tuple[str, ...] = ()
    token: str = ""
    parts: tuple[str, ...] = ()


def slots(text: str) -> list[Slot]:
    """フレーズの詞 → **旋律の1音ごと**の並び。空白で区切った1つが1音（空白の無い詞は全体で1音）。"""
    out: list[Slot] = []
    # ★**区切りは定義の空白だけ**（2026-10-02・組「洗い出し-コードとABC」・仕様書 §2 の `sep`＝半角・全角の空白・タブ・改行）。
    #   以前は Python の `split()` で NBSP・細い空白なども区切りに読んでいた。前後の空白（どれも）は取り除く（詞は日本語の欄・§0-6・判断4＝2026-10-03 本人決定）。
    #   定義の外の空白は印の中の字になり、`song_lyrics.check_kana` が「ひらがな以外」の所見にする
    for tok in (t for t in re.split(r"[ 　\t\n\r]+", (text or "").strip()) if t):
        if tok == HOLD:
            out.append(Slot("hold", (), tok))
        elif tok == SKIP:
            out.append(Slot("skip", (), tok))
        else:
            out.append(Slot("sing", tuple(moras(tok)), tok, parts_of(tok)))
    return out


def ties_of(note) -> int:
    """音にタイでまとめた続きの音の数（`BareNote.ties`・`SongNote.ties`）。"""
    return max(0, int(getattr(note, "ties", 0) or 0))


def slot_count(notes) -> int:
    """**詞の印の数**＝ABC で書いた音の数（タイの続きの音も1つと数える・ABC の歌詞行 `w:` と同じ・2026-10-01 判-H）。"""
    return sum(1 + ties_of(n) for n in notes)


def by_note(ss: list[Slot], notes) -> list[tuple[Slot | None, list[Slot]]]:
    """印を音ごとに分ける: `[(頭の印, タイの続きの音の印…), …]`（音の数だけ。印が足りない音は `(None, [])`）。
    詞の検査（`tie_problems`）と組み立て（`song_place.fit_phrase`）が同じ分け方で読む。"""
    out: list[tuple[Slot | None, list[Slot]]] = []
    pos = 0
    for n in notes:
        need = 1 + ties_of(n)
        got = ss[pos:pos + need]
        pos += need
        out.append((got[0], list(got[1:])) if got else (None, []))
    return out


def tie_ok(head: Slot, cont: Slot) -> bool:
    """タイの続きの音の印として正しいか: 頭が歌う・伸ばす音なら `_`、頭が歌わない `*` なら `*`。"""
    return cont.kind == ("skip" if head.kind == "skip" else "hold")


def tie_problems(ss: list[Slot], notes) -> list[str]:
    """**タイの続きの音の印**（2026-10-01・判-H）。ABC の歌詞行では、タイでつないだ後ろの音には `_` を書く（前の音節を伸ばす）。
    ほかの印を書くと、つながった1つの音の中で言い直すことになり、組み立てにはその置き場が無い。数のずれは呼び手が `slot_count` で見る。"""
    out: list[str] = []
    for j, (head, conts) in enumerate(by_note(ss, notes), 1):
        bad = [c.token for c in conts if head is not None and not tie_ok(head, c)]
        if bad:
            out.append(f"{j}音目はタイで{1 + len(conts)}つの音をつないだ音——続きの音の印は `{HOLD}`（前の音節を伸ばす。"
                       f"歌わない音なら `{SKIP}`）。いまは「{' '.join(bad)}」")
    return out


def sung_kana(text: str) -> str:
    """印を外した**歌うかな**だけ（表示の突き合わせ・hook の照合・`lyrics_full` に使う）。"""
    return "".join(m for s in slots(text) for m in s.moras)


def _head_sokuon(ss: list[Slot]) -> list[str]:
    """**前に歌う音の無い「っ」**（フレーズの頭・`*` の直後）。所見は最初の1つだけ。"""
    sung_before = False
    for s in ss:
        if s.kind == "skip":
            sung_before = False                             # 歌わない音のあとも、保つ相手がいない
            continue
        if s.kind != "sing":                                # `_`（前の音を伸ばす）は歌う音を増やさない
            continue
        for m in (s.parts or s.moras):
            if m == "っ" and not sung_before:
                return [f"「{s.token}」: 前に歌う音の無い『っ』（フレーズの頭など）は、保って切る前の音が無い"
                        "（促音は前の音を保って終わり際を切るもの）。前のフレーズの最後の音に付けるか、言葉を変える"]
            sung_before = sung_before or m != "っ"
    return []


def long_vowel_joins(text: str) -> list[str]:
    """**`~` の後ろが前の音節の長音に見える所**（`に~い`・`か~あ`・`こ~お`・`きょ~う`）。点検役の材料（2026-09-30・記-7）。

    長音は1つの区切りに書く（`にい`＝前の音のまま伸ばす）。`~` でつなぐと別の音節として等分され、後ろの母音が
    立ち上がる（『駆け抜けろスニーカー』の「す に~い か~あ」）。ただし**別の音節が同じ母音で続く語**もある
    （「言い」＝`い~い`・「場合」＝`ば~あ~い`）ので、答えは1つに決まらない＝内容。形式の所見にはせず、
    詞の点検役に事実として渡す（`song_lyrics.run_lyrics`）。
    """
    out: list[str] = []
    for s in slots(text):
        if s.kind != "sing" or JOIN not in s.token:
            continue
        syls = [x for x in s.token.split(JOIN) if x]
        for a, b in zip(syls, syls[1:]):
            ma, mb = moras(a), moras(b)
            if ma and mb and _is_long(ma[-1], mb[0]):
                out.append(s.token)
                break
    return out


#: **詰めの標準で前の音に収めてよいモーラ**（2026-10-06 本人決定）。1つの音に詰めるのは、促音「っ」・撥音「ん」を前の音に収めるときだけが標準。
#: ふつうの2音節（`あ~す`＝明日・1つの区切りの `せい`）を1つの音に詰めるのは標準外——J-POP でも速い曲は詰めずに音を細かく刻む（8分→16分2つ）。
#: 実例: 『火花になって飛べ』（BPM200）で `~` の詰めが42か所（うち1音節0.075秒が16か所）、「何小節かに一回最後の音が詰められている」と聞こえた
STD_JOIN_MORAS = ("っ", "ん")


def squeezed_syllables(s: Slot) -> tuple[str, ...]:
    """歌う印 `s` の中で鳴らし分けるもの（`parts`）のうち、**促音・撥音を除いた音節**。2つ以上なら、ふつうの2音節を1つの音に詰めている。"""
    if s.kind != "sing":
        return ()
    return tuple(p for p in (s.parts or tuple(s.moras)) if p not in STD_JOIN_MORAS)


def plain_joins(text: str) -> list[str]:
    """**ふつうの2音節以上を1つの音に詰めた印**（標準外の詰め・2026-10-06）。`~` でつないだもの（`あ~す`）も、1つの区切りに続けて
    書いたもの（`せい`）も同じ（どちらも組み立てが音を等分する）。促音・撥音を前の音に収めた詰め（`け~っ`・`し~ん`・`しん`）と長音（`にい`）は入らない。"""
    return [s.token for s in slots(text) if len(squeezed_syllables(s)) >= 2]


def join_problem(token: str) -> str:
    """1つの印（1音ぶんのかな）の **`~` の置き方**の所見（空文字なら所見なし）。仕様書 §2 の文法 `syllable , { "~" , syllable }`。
    ★**`~` の片側だけ・2つ続けて**（2026-10-02・組「洗い出し-コードとABC」）。以前は `ん~`・`~い`・`い~~い`・`い ~い`（`~い` の印）の
      空の側を黙って捨てていた（`parts_of`）。詞の印（`problems`）とカバーの割り当ての1音ぶん（`song_lyrics.kana_cell_problems`・
      2026-10-02 組「歌詞のかなと見せる詞」）が同じこの1つで見る"""
    if JOIN in token and not all(token.split(JOIN)):
        return (f"「{token}」: `{JOIN}` は前後の音節をつなぐ印——片側だけ・2つ続けては書かない"
                f"（前後に空白を入れずに `い~い` の形。つなぐ相手が無いなら `{JOIN}` を外す）")
    return ""


def problems(text: str) -> list[str]:
    """譜割りの**書式**の所見（形式の検査。数が音と合うかは呼び手が `len(slots(text))` で見る）。"""
    out: list[str] = []
    ss = slots(text)
    if ss and ss[0].kind == "hold":
        out.append(f"頭の『{HOLD}』は伸ばす相手がいない（フレーズの最初の音は歌う印にする）")
    # `*`（歌わない）のあとの `_` も伸ばす相手がいない（2026-09-29）——歌わない音を伸ばすことになる
    for a, b in zip(ss, ss[1:]):
        if a.kind == "skip" and b.kind == "hold":
            out.append(f"『{SKIP}』のあとの『{HOLD}』は伸ばす相手がいない（歌わない音は伸ばせない）")
            break
    # ★**フレーズ頭の「っ」**（2026-09-30・出-14④）。促音は前の音を保って終わり際を切る（`song_place._sustain_sokuon`）が、
    #   前に歌う音が無いと保つ相手がいない。以前は所見にならず、「っ」のまま V6 へ送られていた
    out += _head_sokuon(ss)
    for s in ss:
        if s.kind != "sing":
            continue
        # ★**前の字から離れた小書きのかな**（2026-09-30・出-14）。「き ょ」と2つの印に分けると、「ょ」が1つの音になる
        #   （『孤独のまま』の1回目。そのときは印の数の検査が偶然捕まえた）。拗音は前の字と同じ印に書く
        if any(syl[:1] in _SMALL_KANA for syl in s.token.split(JOIN) if syl):
            out.append(f"「{s.token}」: 小書きのかな（ゃゅょぁぃぅぇぉ）が前の字から離れている——拗音は前の字と同じ印に書く（「きょ」）")
        if (HOLD in s.token or SKIP in s.token):
            out.append(f"「{s.token}」: 『{HOLD}』『{SKIP}』は1つの印として空白で区切る")
        elif not s.moras and JOIN in s.token:
            # 2026-10-01: ABC の歌詞行と同じく、`~` は空白を入れずに前後の音節をつなぐ（`い ~ い` だと `~` だけの印＝空の音になる）
            out.append(f"「{s.token}」: `{JOIN}` は前後に空白を入れずにつなぐ（`い~い`）")
        elif w := join_problem(s.token):
            out.append(w)
        elif not s.moras:
            out.append(f"「{s.token}」: 歌うかなが無い")
        elif len(s.moras) > SQUEEZE_MAX:
            out.append(f"「{s.token}」は1音に{len(s.moras)}モーラ——1音に詰めるのは{SQUEEZE_MAX}モーラまで"
                       "（区切りの空白が抜けていないか。足りないなら旋律の音を足す）")
    return out
