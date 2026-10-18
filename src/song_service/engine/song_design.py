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

"""5ステップ版の作曲（試作）。現行の2段（plan→melody）と**同じ入口・同じ出力型**で比較できるようにする。

## 現行との違い

| | 現行 | この試作 |
|---|---|---|
| 工程 | plan（テーマ・BPM・コード・歌詞を一度に）→ melody | **テーマ → 設計 → コード → 詞 → 旋律**の5段 |
| 説明 | 残らない（.in/.out のトレースのみ）| **各段が `reason` を返す**＝なぜそうしたかが残る |
| 構成 | 無し（歌詞は1本の文字列）| セクション（種別・小節数・モーラ配列）|
| 休符 | **無し**（音符を隙間なく詰める）| **あり**（`rest_before`）|
| 音価 | 実測で 0.5拍グリッドに貼り付き | 16分（0.25）を含む3段階以上を要求 |
| 音域 | ド4〜ミ5（E5 で頭打ち）| ド4〜ソ5（G5）|
| 拍頭 | 実測 89% がコードトーン | 7割程度に緩め、**経過音・テンションを促す** |

休符が無いことが効いている: 間を作るには音を伸ばすしかなく、それがロングトーン率
0.496（プロの4.7倍）の正体だった。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, field_validator, model_validator
from pydantic_ai import Agent

HERE = Path(__file__).resolve().parent
# ※ テスト環境では各モジュールが `_bootstrap` を自前で読んでいたが、本番は
#   `song_service` の入口（`song_runner` / `_bootstrap`）が先に sys.path を通すので不要。

from song_arrange import arrange, to_spec                              # noqa: E402（bar_filter の拡張もここで効く）
from era_styles import ERA_GUIDE, INTRO_GUIDE                        # noqa: E402（様式はテーマで固定する）
from song_llm import PITCH, pick_model, solfa_to_midi                # noqa: E402（PITCH＝国際式の音名の型・最高音の読み手）
from song_model import PatchStatus, StepStatus                                    # noqa: E402（全工程の自己申告欄）
from song_checks import JsonBool, JsonFloat, JsonInt, json_int  # noqa: E402（LLM が書く数と真偽の欄の型・2026-10-02 判断5）
from song_checks import edge_ws_problem, half_width, strip_ascii_ws, trim_edge_ws   # noqa: E402（半角の記法の前後の空白・§0-6・2026-10-03）
from song_schema import ChordSpan, SongNote, SongScore              # noqa: E402
from song_schema import TICKS_PER_QUARTER as Q                      # noqa: E402
from timbre_map import DRUM_NAMES                                    # noqa: E402（グルーヴのドラムの打楽器名＝伴奏と同じ一覧）
from song_prompts import prompt                          # プロンプトの正は prompts/*.md

BAR = Q * 4                                                # 4/4 のときの1小節（既定）


def bar_ticks(time_signature: str) -> int:
    """拍子 → 1小節の tick 数（4分音符 = Q）。4/4→1920 / 3/4→1440 / 6/8→1440。

    ★**読めない拍子は `ValueError`**（2026-09-30）。以前は黙って 4/4（1920）にしていた。曲づくりの拍子は設計の型
      （`TimeSignature`＝選べる4つ）で閉じたので、ここへ読めない形が来るのは不具合。
    """
    try:
        num, den = (int(x) for x in time_signature.split("/"))
        return max(Q, Q * 4 * num // den)
    except (ValueError, ZeroDivisionError, AttributeError) as e:
        raise ValueError(f"拍子『{time_signature}』が読めない（`4/4` の形）") from e


def grid_slots(time_signature: str) -> int:
    """1小節に入る8分音符の数（伴奏のリズムグリッドの長さ）。4/4→8 / 3/4→6 / 6/8→6。"""
    return max(2, bar_ticks(time_signature) // (Q // 2))


def units_per_bar(time_signature: str) -> int:
    """1小節に入る16分音符の数（旋律の ABC `L:1/16` の単位数）。4/4→16 / 3/4→12 / 6/8→12 / 2/4→8。

    ★**拍子から数を出すのはここと `bar_ticks` / `grid_slots` / `beats_per_bar` だけ**（2026-09-27）。
      旋律の枠が `GRID_PER_BAR = 16`、ABC が `UNITS_PER_BAR = 16`、グルーヴへ渡す秒数が `bars * 4` と
      **4/4 の決め打ち**になっていて、3/4 の曲（『風に呼ばれたら』）で旋律が1行8拍・伴奏が1行6拍になり、
      歌が区間ごとに遅れてサビが後奏に乗った（12小節＝27秒）。
    """
    return max(4, bar_ticks(time_signature) // (Q // 4))


def beats_per_bar(time_signature: str) -> float:
    """1小節の4分音符の数。4/4→4 / 3/4→3 / 6/8→3 / 2/4→2。

    **長さ・BPM・位置 `小節:拍`・`notes_per_beat` の「拍」はこれ（4分音符）で数える**（MIDI・V6 と同じ）。
    感じる拍（拍の頭・拍の数）は `beat_heads`——6/8 は付点4分×2拍なので、ここ（3）とは数が違う。
    """
    return bar_ticks(time_signature) / Q


def _compound(time_signature: str) -> bool:
    """複合拍子か（分母が8で分子が3の倍数・6以上＝6/8・9/8・12/8）。拍は付点4分。"""
    num, den = (int(x) for x in time_signature.split("/"))
    return den == 8 and num % 3 == 0 and num >= 6


def beat_heads(time_signature: str) -> list[int]:
    """1小節の**拍の頭**の位置（16分の単位・**1始まり**）。4/4→[1,5,9,13] / 3/4→[1,5,9] / **6/8→[1,7]** / 2/4→[1,5]。

    ★**6/8 は付点4分×2拍**（複合2拍子・2026-09-30 出-5／記-6）。以前は全工程で「1拍＝4分音符」と数え、6/8 を
      「3拍・拍の頭は 1・5・9 単位目」（3/4 の割り方）と伝えていた。実例『茜色のワルツ』（6/8・BPM72）で、
      旋律は 3/4 の型で書き、グルーヴのスネアは 6/8 どおり7単位目に置き、周回ごとに 3/4 と 6/8 を行き来した。
    """
    bar_ticks(time_signature)                              # 読めない拍子はここで ValueError
    step = 6 if _compound(time_signature) else 16 // int(time_signature.split("/")[1])
    return list(range(1, units_per_bar(time_signature) + 1, max(1, step)))


def meter_text(ts: str, *, positions: bool = False) -> str:
    """**この曲の拍子を、依頼文で伝える1か所**（2026-09-30・出-5）。旋律・グルーヴ・声部の依頼文はここから出す。

    指示書（`song_melody.md`・`song_groove.md`）の数と例は 4/4 のもの。拍子が違う曲の正確な数は**ここで計算して渡す**。
    単位は**1始まり**で数える（記-5）。`positions=True` なら、表の位置 `小節:拍` の数え方（`position_text`）も添える。

    ★2026-09-27 に旋律の依頼文として作り、グルーヴは `song_outline` が別の文（「1小節＝3拍・8分音符6個」）を持っていた。
      6/8 の拍を「付点4分×2」と定めたとき、伝える所が2つだと片方だけ直るので1つにした。
    """
    upb, bpb, g8 = units_per_bar(ts), int(beats_per_bar(ts)), grid_slots(ts)
    heads = beat_heads(ts)
    h16 = "・".join(map(str, heads))
    h8 = "・".join(str((h - 1) // 2 + 1) for h in heads)
    if _compound(ts):
        beat = (f"**1小節＝{upb}単位＝{len(heads)}拍**（**拍は付点4分**＝8分音符3つのまとまり×{len(heads)}・"
                f"8分音符{g8}個。4分音符×{bpb} の {bpb}/4 拍子とは揺れ方が違います）")
    else:
        beat = f"**1小節＝{upb}単位＝{bpb}拍**（4分音符×{bpb}・8分音符{g8}個）"
    out = (f"■**この曲の拍子: {ts}**（ABC は `M:{ts}`・`L:1/16`＝1単位が16分音符）。{beat}。"
           f"拍の頭は {h16} 単位目（8分の格子なら {h8} 番目）、最後の拍は {heads[-1]}〜{upb} 単位目。"
           "単位も格子も1始まりで数えます。")
    if _compound(ts):
        out += (f"\n  **テンポ（BPM）・『1拍に◯個』（`notes_per_beat`）・長さの『拍』・位置の『小節:拍』は4分音符を1拍として数えます**"
                f"（この曲の1小節は4分音符{bpb}つぶん。拍の頭の数え方とは別です）。")
    if ts != "4/4":
        out += (f"\n  **指示の中の 4/4 の数（1小節＝16単位・8分音符8個・拍の頭 1・5・9・13・4拍目など）は、"
                f"この曲では上の数で読み替えてください。**")
    if positions:
        out += "\n  " + position_text(ts)
    return out


def position_text(ts: str) -> str:
    """表の位置 `小節:拍` の数え方（声部・奏法・歌の崩しの表と同じ `artic_plan.tick_key`）。"""
    bpb = beats_per_bar(ts)
    heads = "・".join(f"`:{(h - 1) / 4:.2f}`" for h in beat_heads(ts))
    # ★「小数2桁」を言葉でも書く（2026-10-02・字コードの組）。以前は形 `:0.00` を見せるだけで、読み手（`song_checks._read_position`）が
    #   小数2桁でない拍（`12:1.5`）を所見にすることは書いていなかった（仕様書 §0-4・§12）
    return (f"位置 `小節:拍` の拍は**4分音符＝1.00・0始まり・小数2桁**（小節の頭が `:0.00`。`:1.5` ではなく `:1.50`）。"
            f"この曲の1小節は `:0.00`〜`:{bpb - 0.25:.2f}`、拍の頭は {heads}。")

# ---- キーの移調 ---------------------------------------------------------------
# 旋律もコードも **C メジャー / A マイナーで書かせて、最後に移調する**。
# LLM は C 基準で考えるほうが音を正確に思い出せるため（solfège CoT の前提。当時は固定ドの階名で書かせていた。
# 2026-10-01 から LLM が書く・見る高さは旋律が ABC・ほかは国際式の音名で、どちらも C 基準のまま）。
#: 音名（＋変化記号1つ）→ ピッチクラス。★**全部の綴りを持つ**（2026-09-27・監査）。以前は手書きで
#:   「D♭」「G♭」「C♯」などが抜けており、`transpose_chord` はそのコードを**黙って移調しなかった**。
#: ★**変化記号は ASCII の `#` `b` を1つだけ**（2026-10-02 本人「2通りに書けるものは良くない」「標準に寄せて1つにそろえる」・仕様書 §0-1・§17 ず-1）。
#:   以前は記号の ♯♭（調名は全角の ＃ も）を同じ意味の別綴りとして読んでいた。いまは書けない綴りとして所見にし、`#` `b` で書かせる
#:   （文字で書くコード譜の標準。音名 `Bb4`（`song_llm.PITCH`）と同じ書き方）。`_SYMBOL_ACCIDENTAL` は書き直し方を示すためだけに持つ
_PITCH = {L + a: (pc + d) % 12
          for L, pc in {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}.items()
          for a, d in (("", 0), ("#", 1), ("b", -1))}
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


#: 調名の読み方（2026-09-27・監査）。以前は `_PITCH` の表に載った綴りしか読めず、
#: 「D♭」「G♭」「C♭」「Dmaj」「A minor」などが**黙って C（0）になっていた**。
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACCIDENTAL = {"#": 1, "b": -1}
#: 書けない記号の変化記号 → ASCII の綴り（所見の書き直し方に使う。読み替えはしない）
_SYMBOL_ACCIDENTAL = {"♯": "#", "♭": "b", "＃": "#"}
#: ★**変化記号を2つ以上重ねた音名**（`Ebb`・`C##`・2026-10-02 本人決定）。異名の簡単な名前で書かせる（`Ebb` → `D`）
#:   同じ向きを重ねたものだけ（`Gb#11` は根音 `Gb`＋テンション `#11` と読む）
_DOUBLE_ACCIDENTAL = re.compile(r"(?P<letter>[A-G])(?P<acc>#{2,}|b{2,})")


def _symbol_accidental_problem(what: str, name: str) -> str | None:
    """記号の ♯ ♭（全角の ＃）で書いた変化記号の所見（コード名・調名で共通）。書き直した綴りを添える。"""
    if not any(c in name for c in _SYMBOL_ACCIDENTAL):
        return None
    fixed = "".join(_SYMBOL_ACCIDENTAL.get(c, c) for c in name)
    return (f"{what}『{name}』: 変化記号は記号の ♯ ♭ ではなく ASCII の `#` と `b` で書く"
            f"（『{name}』 → `{fixed}`。文字で書くコード譜の書き方・音名 `Bb4` と同じ）")


def _simple_name(letter: str, acc: str) -> str:
    """音名＋重ねた変化記号 → 異名の簡単な名前（白鍵は幹音・黒鍵は重ねた記号の向き）。`Ebb` → `D`・`C##` → `D`・`Fbb` → `Eb`。"""
    pc = (_LETTER_PC[letter] + sum(_ACCIDENTAL[a] for a in acc)) % 12
    return (_FLAT_NAMES if acc.startswith("b") else _NAMES)[pc]


def _double_accidental_problem(what: str, name: str, at: str = "^") -> str | None:
    """変化記号を2つ以上重ねた音名（`Ebb`・`C##`）の所見。`at` は音名の現れる所（コード名は頭と `/` の後ろ）。"""
    m = re.search(rf"(?:{at}){_DOUBLE_ACCIDENTAL.pattern}", name)
    if not m:
        return None
    simple = _simple_name(m["letter"], m["acc"])
    fixed = name[:m.start("letter")] + simple + name[m.end("acc"):]
    how = f"`{name}` → `{fixed}`" if fixed == simple else f"`{m['letter']}{m['acc']}` → `{simple}`。『{name}』 → `{fixed}`"
    why = f"{what}『{name}』: 変化記号を2つ重ねた音名（`Ebb`・`C##`）は書けない——異名の簡単な名前で書く（{how}）"
    if name[m.end("acc"):m.end("acc") + 1].isdigit():       # `Bbb9` は B♭ の ♭9 のつもりかもしれない
        why += "。種類の ♭9 などのつもりなら7度の後ろに括弧で書く（`Bb7(b9)`）"
    return why
#: ★**長調・短調の書き方は1通り**（2026-10-02 本人「調名の書き方を1通りにする」・仕様書 §4）: 長調は**接尾なし**（`F`・`Bb`）、
#:   短調は **`m`** だけ（`Dm`・`F#m`）。以前は下の綴りも同じ意味の別綴りとして読んでいた（`F major`・`Dmaj`・`A minor`・`ニ短調` の接尾）。
#:   いまは書けない綴りとして所見にし、書き直した形を添える（`F major` → `F`・`D minor` → `Dm`）。読み替えはしない——この表は所見の書き直し方のためだけに持つ
_MAJOR_ALIAS = {"M", "maj", "Maj", "MAJ", "major", "Major", "MAJOR", "ma", "dur", "Dur", "メジャー", "長調"}
_MINOR_ALIAS = {"min", "Min", "MIN", "minor", "Minor", "MINOR", "mi", "moll", "Moll", "マイナー", "短調"}


#: 空白（半角・全角・タブ・改行。Python の `\s` は全角の空白も含む）
_SPACE = re.compile(r"\s")


def parse_key(key: str) -> tuple[int, bool] | None:
    """調名 → (主音のピッチクラス 0〜11, 短調か)。**読めなければ None**（黙って C にしない）。

    読むもの: 音名 A〜G（大文字）＋ 変化記号（ASCII の `#` `b` を**1つまで**・異名同音もそのまま＝Db も C# も 1）
    ＋ 短調なら `m`（長調は何も付けない）。**前後の半角の空白とタブだけ**取り除いてから読む（仕様書 §0-6・§4。
    前後の全角の空白・NBSP・改行などは読まない＝所見は `key_problem`・2026-10-03 判断4 本人決定）。
    例: `Db`→(1, False) / `Gb`→(6, False) / `Cb`→(11, False) / `D`→(2, False) / `F#m`→(6, True) / `Am`→(9, True)
    ★記号の ♯♭・全角の ＃・変化記号2つ（`Ebb`）は読まない（2026-10-02・§0-1。所見は `key_problem`）。
    ★`m` 以外の接尾（`M`・`maj`・`major`・`min`・`minor`・`メジャー`・`短調` …）も読まない（2026-10-02 本人「1通りにする」。所見は `key_problem`）。
    ★**途中の空白も読まない**（`A m`・`F #`・2026-10-02 本人「おすすめで」）。以前はどこに入っていても除いて読んでいた（書き方が何通りにもなる）。
      空白は記法全体で「区切り」の意味（コードの欄では和音の区切り＝`Am F`）なので、区切りの要らない調名では書けない字にする。所見は `key_problem`
    """
    k = strip_ascii_ws(key)                                 # 前後の半角の空白とタブだけ除く
    if _SPACE.search(k):                                    # 途中の空白は読まない（所見は `key_problem` が書き直した形を添える）
        return None
    # ★**音名は大文字だけ**（2026-10-01・標準とのずれの点検）。以前は小文字の `a` を A の長調として読んでいたが、
    #   小文字の音名を短調とする流儀（`a`＝イ短調・music21 など）もあり、どちらにも決まらない。所見にして書き直させる
    if not k or k[0] not in _LETTER_PC:
        return None
    pc = _LETTER_PC[k[0]]
    i = 1
    if i < len(k) and k[i] in _ACCIDENTAL:                  # 変化記号は1つまで（2つ目は接尾として読めず None）
        pc += _ACCIDENTAL[k[i]]
        i += 1
    rest = k[i:]
    if rest == "m":
        return pc % 12, True
    if rest == "":
        return pc % 12, False
    return None


def _key_rewritten(k: str) -> str | None:
    """書けない綴りの調名 → 同じ調の正の綴り（所見に添える書き直し方。読み替えには使わない）。

    接尾（`major`・`minor` …）・記号の ♯♭（全角の ＃）・重ねた変化記号をまとめて直す（`E♭ major` → `Eb`・`D minor` → `Dm`・
    `Ebb minor` → `Dm`）。直し方が1つに決まらない綴り（小文字・ドイツ式・`Am7` など）は None。"""
    m = re.fullmatch(r"([A-G])([#b♯♭＃]*)(.*)", k)
    if not m:
        return None
    letter, acc, rest = m[1], m[2].translate(str.maketrans(_SYMBOL_ACCIDENTAL)), m[3]
    if rest in ("", "m"):
        minor = rest == "m"
    elif rest in _MAJOR_ALIAS or rest in _MINOR_ALIAS:
        minor = rest in _MINOR_ALIAS
    else:
        return None
    if len(acc) > 1:
        if len(set(acc)) != 1:                              # `#b` のような混ぜ書きは直し方が決まらない
            return None
        head = _simple_name(letter, acc)
    else:
        head = letter + acc
    return head + ("m" if minor else "")


def key_problem(key: str) -> str | None:
    """調名の所見（読めれば None）。設計の型の検証と、設計の工程の所見の両方で使う。"""
    if parse_key(key) is not None:
        return None
    # ★**前後に全角の空白・NBSP・改行などを付けない**（2026-10-03 判断4 本人決定・仕様書 §0-6）。取り除いた形にほかの所見があれば続ける
    if why := edge_ws_problem(key):
        rest = key_problem(trim_edge_ws(key))
        return f"調{why}" + (f"。そのうえで: {rest}" if rest else "")
    raw = strip_ascii_ws(key)
    k = "".join(raw.split())                                # 途中の空白を除いた形（書き直し方を探すためだけ・読み替えはしない）
    # ★**接尾は1通り**（2026-10-02 本人決定・仕様書 §4）。長調は接尾なし・短調は `m`。書き直した形を添える（`F major` の空白もこれで直る）
    m = re.fullmatch(r"([A-G])([#b♯♭＃]*)(.*)", k)
    if m and (m[3] in _MAJOR_ALIAS or m[3] in _MINOR_ALIAS):
        fixed = _key_rewritten(k)
        how = f"『{key}』 → `{fixed}`" if fixed else "長調は `F`・短調は `Dm` の形"
        return (f"調『{key}』: 長調・短調の書き方は1通り——長調は何も付けない（`F major` → `F`）、短調は `m` だけ"
                f"（`D minor` → `Dm`）。`{m[3]}` は書けない（{how}）")
    # ★**途中に空白を入れない**（2026-10-02 本人「おすすめで」・仕様書 §0-6・§4）。空白は記法全体で「区切り」（コードの欄では和音の区切り）
    #   なので、区切りの要らない調名では書けない。書き直した形を添える（`A m` → `Am`・`F #` → `F#`・`E ♭` → `Eb`）
    #   書き直し方が1つに決まらない綴り（`a m`・`Fis moll`）は下の「読めない」の所見（書き方の形を示す）
    if _SPACE.search(raw) and (fixed := _key_rewritten(k)):
        return (f"調『{key}』: 調名の途中に空白を入れない——空白は「区切り」の印（コードの欄では和音の区切り）で、"
                f"調名には区切りが要らない（『{key}』 → `{fixed}`）")
    # ★変化記号は ASCII の `#` `b` を1つだけ（2026-10-02 本人決定・仕様書 §0-1・§4・ず-1）。記号の ♯♭ と重ねた変化記号は書き直し方を添える
    if k[:1] in _LETTER_PC and (why := _symbol_accidental_problem("調", k) or _double_accidental_problem("調", k)):
        return why
    return (f"調『{key}』が読めない——音名（大文字の C D E F G A B）＋変化記号（# / b）＋短調なら m、の形で空白を入れずに書く（長調は何も付けない）"
            "（例 C / Eb / F# / Db / Am / C#m / Bbm。小文字の `a`・日本語の『イ短調』・ドイツ式の `Fis` は `Am`・`F#` のように書く）")


def key_shift(key: str) -> int:
    """キー名 → C メジャー基準からの半音数。短調は**相対長調**で見る（Am→0 / Em→7）。

    ★**読めない調は `ValueError`**（2026-09-27・監査）。以前は黙って 0（C）にしていたので、
      「D♭」の曲が C で鳴っても誰も気づけなかった。空・None は「決めていない」＝設計の既定の C（0）。
      呼び側で所見にしたいときは、先に `key_problem` で確かめる。
    """
    if not strip_ascii_ws(key):
        return 0
    got = parse_key(key)
    if got is None:
        raise ValueError(key_problem(key))
    semi, minor = got
    return (semi + 3) % 12 if minor else semi        # Am は C の短3度下＝+3 で相対長調に合わせる


def split_bar_chords(cell: str) -> list[str]:
    """1小節ぶんの表記 → **その小節で鳴るコードの並び**（2026-09-21）。

    ★**空白区切りが標準**。一般のコード譜では**1小節に2つ書けば、それぞれが半分ずつ**受け持ち、
      **`|` は小節の区切り**を意味する。ここでは逆に **`|` を「小節の中の分割」**に使っていた
      ——同じリポジトリの旋律（ABC）・ドラムタブでは `|` が小節線なので、**ここだけ意味が逆**だった。

    ★**区切りは空白だけ**（2026-09-30）。`Am|F` は1つの読めないコード名になり、記法の所見（`CHORD_BAR`＝`chord_problem`）で
      コードの工程に差分で直させる。※2026-09-21〜30 は、`|` で保存された古い曲（50曲中33曲）を再生するために `|` も
      黙って受けていた（リリース前なので古い記録は読まない・本人）。
    ★**区切りは半角の空白だけ**（2026-10-02・仕様書 §3 の文法 `ws = " "`）。以前は Python の `split()` でタブ・全角の空白・NBSP も
      区切りとして読んでいた。いまはそれらを含む塊は1つの読めないコード名になり、`chord_problem` が所見にする。
      前後は半角の空白とタブだけ取り除く（§0-6・2026-10-03 判断4 本人決定。前後の全角の空白・改行などは `CHORD_BAR` の所見）。
    """
    return [x for x in strip_ascii_ws(cell).split(" ") if x]


#: ★**コード名の読み方は組み立てと同じ1つ**（2026-10-02・本人決定「字の袋を並びに」「同じ意味の書き方を1通りに」）。
#:   文法（並び）と組み合わせの決まりは `song_audio.parse_chord`（仕様書 §3）。以前ここにあった `_CHORD_RE` は種類の字の袋
#:   （`{ maj | m | 7 | # | b | ( | ) | , | \d+ … }` を何個でも）で、`C7b`・`Cadd`・`C7(9`・`Cmm7`・`C79`・`Am7,`・全角の数字 `Am７`・
#:   先頭の 0 `C7(b09)` が通り、組み立ては分からない字を黙って無視していた（洗い出しの H2・§2-2 の C 群）
from song_audio import TENSION_ORDER, parse_chord                   # noqa: E402
from song_audio import _CHORD_RE                                    # noqa: E402（移調で根音・種類・低音を分ける）

#: ノーコード（そこは和音を鳴らさない）。読める＝移調しない（`chord_notes` も空で返す）。
#: ★**`N.C.` だけ**（2026-10-02 本人決定・判断3）。以前は `N.C`・`NC` も同じ意味で読んでいた（所見に書き直した形を添える）
NO_CHORD = frozenset({"N.C."})


def chord_readable(name: str) -> bool:
    """コード名を**組み立てが読めるか**（文法 `song_audio.parse_chord` で読める、または `N.C.`）。

    組み立て（`song_place.place_chords`・`song_resize`・カバーの C 基準の表 `song_arrange_cover.c_basis_chords`）が
    「検査を通ったはずのコード名か」を確かめる部品（2026-10-02）。以前は `song_place` が非公開の `_CHORD_RE` を直接読んでいた。
    ★2026-10-02 から検査と同じ文法で読む——読めるコード名はどれも書けるコード名（`chord_problem` が None）。
    """
    n = strip_ascii_ws(name)
    return n in NO_CHORD or parse_chord(n) is not None

#: ★**標準の書き方だが、組み立てが標準と違う音で鳴らしていた種類・書き直し方が1つに決まらない種類**（2026-09-30・記-12 本人「記法側で」）。
#:   2026-10-02 から文法（並び）が読まないので鳴らないが、なぜ書けないか・どう書くかを所見に添える。種類（根音と分数の低音を除いた部分）に当てる。
#:   同じ和音の別の書き方（`CM7`・`C7(9)`・`C+`・`Cadd2`・`C7sus`・`C7+5`・`Cm-6`・`C(9)` …）は `_chord_rewritten` が書き直した形を添える
#:   ※`11`・`13` の和音（`C11`・`Cm11`・`C13`）は組み立てが9度を足さない（省くのは普通の積み方）。9度も鳴らすなら `C9(11)`・`C9(13)`
_CHORD_NG: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^5$|no\d|omit"),
     "パワーコード・3度を抜く書き方（`C5`・`Cno3`・`Comit3`）は書けない（3度も鳴る）——3和音で書く（`C5` → `C`）。"
     "♭5 のつもりなら `C7b5`・`Cm7b5`・`C(b5)`"),
    (re.compile(r"\^"), "`^`（ジャズのメジャーセブン）は書けない——`maj7` と書く（`C^7` → `Cmaj7`）"),
    (re.compile(r"alt"), "`alt`（オルタード）は書けない——テンションを書き並べる（`C7alt` → `C7(b9,#9)`・`C7#5(b9)`）"),
    (re.compile(r"6/?\(?9"), "6/9（`C6/9`・`C69`・`C6(9)`）は書けない——`C6add9` と書く"),
    (re.compile(r"^[24]$"), "`C2`・`C4` は書けない——足した音は `add9`・`add11` と書く（`C2` → `Cadd9`）"),
    (re.compile(r"sus2.*sus4|sus4.*sus2"), "`sus2` と `sus4` は一緒に書けない（片方しか鳴らない）——`Csus4add9` のように書く"),
    (re.compile(r"(?:dim|°)(?:9|11|13)"), "dim に9度などは書けない——`dim7` と書く"),
]

#: 標準の綴りだが文法が読まない種類（`Maj7`・`ma7`・`mi7`・`Min`）。読めない所見に書き直し方を添える（2026-10-01）
_CHORD_SPELLING = re.compile(r"^[A-G][#b]?(?:Maj|MAJ|Ma(?!j)|ma(?!j)|mi(?!n)|Min|MIN|Mi(?!n))")
#: 全角の数字（書けない。所見に半角で書き直した形を添える）
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
#: 書き直した形を探すときだけ使う、種類の字の切り方（読み替えには使わない）
_LOOSE_QUALITY = re.compile(r"maj|min|dim|aug|sus|add|m|M|Δ|°|ø|\+|-|#|b|[0-9]+|[(),]")


def _chord_rewritten(name: str) -> str | None:
    """書けない綴りのコード名 → **同じ和音の書ける綴り**（所見に添える書き直し方。読み替えには使わない）。決まらなければ None。

    同じ意味の書き方を1通りにした（2026-10-02 本人決定・判断3）ときの書き直し方: `CM`・`Cmaj` → `C`、`Cmin` → `Cm`、`CM7`・`CΔ` → `Cmaj7`、
    `CmM7` → `Cmmaj7`、`C°` → `Cdim`、`Cø`・`Cm7-5` → `Cm7b5`、`C+`・`C(#5)` → `Caug`、`Caug7`・`C7+5` → `C7#5`、`Csus` → `Csus4`、
    `C7(9)` → `C9`、`C7(11)` → `C11`、`C7b9`・`C7-9` → `C7(b9)`、`C7(b9,13)` → `C13(b9)`、`C(9)` → `Cadd9`、`Cadd2` → `Cadd9`、
    `Cm-6` → `Cm(b6)`、`C7(b09)`・`Am７` → `C7(b9)`・`Am7`、`C7(13,9)` → `C9(13)`。
    """
    n = (name or "").strip().translate(_FULLWIDTH_DIGITS)
    if re.fullmatch(r"N\.?C\.?", n):
        return "N.C."
    m = re.fullmatch(r"([A-G][#b]?)(.*?)(?:/([A-G][#b]?))?", n)
    if not m:
        return None
    root, q, bass = m[1], m[2], m[3]
    toks: list[str] = []
    pos = 0
    while pos < len(q):
        t = _LOOSE_QUALITY.match(q, pos)
        if not t:
            return None
        toks.append(t.group())
        pos = t.end()
    tri, ext, maj, sus = "", "", False, ""
    fives: set[str] = set()
    adds: set[str] = set()
    tens: set[str] = set()
    paren = False
    i = 0
    while i < len(toks):
        t = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        num = int(nxt) if nxt.isdigit() else None
        if t in ("m", "min") or (t == "-" and i == 0):
            tri = tri or "m"
        elif t in ("dim", "°"):
            tri = "dim"
        elif t == "aug" or (t == "+" and i == 0):
            fives.add("#5")
        elif t == "ø":
            tri, ext = "m", ext or "7"
            fives.add("b5")
            if nxt == "7":
                i += 1
        elif t in ("maj", "M", "Δ"):
            maj = True
            if t == "Δ" and num is None:
                ext = ext or "7"
        elif t == "sus":
            if nxt in ("2", "4"):
                i += 1
            sus = "sus" + (nxt if nxt in ("2", "4") else "4")
        elif t == "add":
            if num not in (2, 4, 9, 11):
                return None
            adds.add({2: "9", 4: "11"}.get(num, str(num)))
            i += 1
        elif t in ("b", "#", "-", "+"):
            if num is None:
                return None
            s = "b" if t in "b-" else "#"
            i += 1
            if num == 5:
                fives.add(s + "5")
            elif num == 6 and s == "b":
                tens.add("b6")
            elif s + str(num) in TENSION_ORDER:
                tens.add(s + str(num))
            else:
                return None
        elif t.isdigit():
            k = int(t)
            if ext or paren:
                if k not in (9, 11, 13):
                    return None
                tens.add(str(k))
            elif k in (6, 7):
                ext = str(k)
            elif k in (9, 11, 13):
                ext = "7"
                tens.add(str(k))
            else:
                return None
        elif t == "(":
            paren = True
        elif t == ")":
            paren = False
        i += 1
    if len(fives) > 1 or len(adds) > 1:
        return None
    has7 = ext == "7"
    if has7:
        # 7の和音の数字は、含むナチュラルのテンションのうちいちばん低いもの（`C7(9)` → `C9`・`C7(11)` → `C11`・`C7(9,13)` → `C9(13)`）
        nat = sorted((t for t in tens if t in ("9", "11", "13")), key=int)
        if nat:
            ext = nat[0]
            tens.discard(nat[0])
    if maj and ext not in ("", "6"):
        ext = "maj" + ext
    five = next(iter(fives), "")
    if "b6" in tens and has7:
        tens.discard("b6")
        tens.add("b13")
    if tri == "m" and five == "b5" and not has7:
        tri, five = "dim", ""                              # `Cm(b5)` → `Cdim`
    if five == "#5" and not has7 and tri == "":
        tri, five = "aug", ""                              # `C+`・`C(#5)` → `Caug`
    if five == "b5" and not has7 and tri == "":
        tens.add("b5")
        five = ""                                          # 長三和音の ♭5 は括弧 `C(b5)`
    if not has7:
        for t in ("9", "11"):
            if t in tens:                                  # `C(9)` → `Cadd9`
                tens.discard(t)
                adds.add(t)
        if len(adds) > 1:
            return None
    out = (root + tri + ext + five + sus + "".join("add" + a for a in adds)
           + (f"({','.join(sorted(tens, key=TENSION_ORDER.get))})" if tens else "") + (f"/{bass}" if bass else ""))
    return out if out != (name or "").strip() and parse_chord(out) is not None else None


#: 書けるコード名の書き方（所見に添える。指示書 `song_chords.md` と同じ並び）
CHORD_HOW = ("根音（C D E F G A B）＋変化記号（# / b）のあとに、種類を決まった順に1つずつ: m / dim / aug → "
             "6 / 7 / 9 / 11 / 13 / maj7 / maj9 / maj11 / maj13（7の和音の数字は、含むナチュラルのテンションのうちいちばん低いもの）→ "
             "b5 / #5 → sus4 / sus2 → add9 / add11 → 括弧のテンション（変化したテンション b9 #9 #11 b13 と、数字より上のナチュラル 11 13 を"
             "低い順に `,` で区切る・7度のある和音だけ）。分数なら `/` のあとに低音"
             "（例 Am7 / F#m7b5 / C7sus4 / G7(b9) / C9(13) / C13(b9) / Cmaj9 / Csus4add9 / G/B / N.C.）")


def chord_problem(name: str) -> str | None:
    """コード名の所見（読めれば None）。**移調・鳴らす前に確かめる入口**（2026-09-27・監査）。

    コードの工程の検査（`song_compose.chords_findings`）で所見にし、差分で直させるのに使う。
    ★**読めるかは組み立てと同じ文法**（`song_audio.parse_chord`・2026-10-02）。読めないものは、なぜ書けないか・どう書くかを添える。
    """
    # ★**前後に全角の空白・NBSP・改行などを付けない**（2026-10-03 判断4 本人決定・仕様書 §0-6）。前後は半角の空白とタブだけ取り除く
    if why := edge_ws_problem(name):
        rest = chord_problem(trim_edge_ws(name))
        return f"コード名{why}" + (f"。そのうえで: {rest}" if rest else "")
    n = strip_ascii_ws(name)
    # ★**記号の ♯ ♭（全角の ＃）は書けない**（2026-10-02 本人「2通りに書けるものは良くない」・仕様書 §0-1・§3・ず-1）。
    #   以前は `#` `b` と同じ意味の別綴りとして読んでいた。書き直した綴りを添えて ASCII で書かせる
    if n[:1] in _LETTER_PC and (why := _symbol_accidental_problem("コード名", n)):
        ascii_ = n.translate(str.maketrans(_SYMBOL_ACCIDENTAL))
        if not chord_readable(ascii_) and (fx := _chord_rewritten(ascii_)):
            why += f"。種類の書き方も1通りに（`{ascii_}` → `{fx}`）"          # `C7♭9` → `C7b9` → `C7(b9)`（2026-10-02・判断3）
        return why
    # ★**変化記号を2つ重ねた音名**（`Ebb`・`C##`・根音と分数の低音）は書けない（2026-10-02 本人決定）。文法は
    #   根音 `Eb`＋種類 `b…` と読みかけるので先に見る。異名の簡単な名前で書かせる
    if why := _double_accidental_problem("コード名", n, at="^|/"):
        return why
    if chord_readable(n):
        return None
    fixed = _chord_rewritten(n)
    how = f"（『{name}』 → `{fixed}`）" if fixed else ""
    # ★**区切りは半角の空白だけ**（2026-10-02・§3 の文法）。タブ・全角の空白は1つのコード名の中の字になる
    if re.search(r"\s", n):
        return (f"コード名『{name}』: 小節の中のコードの区切りは半角の空白だけ（タブ・全角の空白は書けない）——"
                "`Am F` のように半角の空白1つで区切る")
    # ★**和音なしは `N.C.` だけ**（2026-10-02 本人決定・判断3）。以前は `N.C`・`NC` も同じ意味で読んでいた
    if fixed == "N.C.":
        return f"コード名『{name}』: 和音を鳴らさない小節の書き方は `N.C.` だけ（`N.C`・`NC` は書けない）"
    # ★**短調の `-`（ジャズの書き方 `C-`・`C-7`）は読めない扱い**（2026-09-29）。以前の組み立ては `-` を知らず長三和音で鳴らしていた
    if re.match(r"^[A-G][#b]?-", n):
        return f"コード名『{name}』の `-`（短調）は読めない——短調は `m` で書く（C- → Cm・C-7 → Cm7）"
    if n.translate(_FULLWIDTH_DIGITS) != n:
        return f"コード名『{name}』: 数字は半角で書く（全角の数字は読めない）{how}"
    if re.search(r"(?<![0-9])0[0-9]", n):
        return f"コード名『{name}』: 数字の頭に 0 を付けない{how}"
    q = (re.fullmatch(r"[A-G][#b]?(.*?)(?:/[A-G][#b]?)?", n) or [None, ""])[1]
    if why := next((w for pat, w in _CHORD_NG if pat.search(q)), None):
        return f"コード名『{name}』: {why}"
    if _CHORD_SPELLING.match(n):
        return (f"コード名『{name}』: `Maj7`・`ma7`・`mi7`・`Min` の綴りは読めない——長7は `maj7`（`CMaj7` → `Cmaj7`）、"
                "短調は `m`（`Cmi7` → `Cm7`・`CMin` → `Cm`）で書く")
    if fixed:
        # ★**同じ和音の書き方は1通り**（2026-10-02 本人決定・判断3）。以前は `CM7`・`CΔ`・`Cmin`・`C+`・`Cø`・`C7-5`・`C7(9)`・`C7b9`
        #   なども同じ意味で読み、種類の字の並びも問わなかった（`Cmm7`・`C77`）
        return (f"コード名『{name}』は書けない——『{name}』 → `{fixed}` と書く（同じ和音の書き方は1通りで、"
                "種類は決まった順に1つずつ: m / dim / aug → 6 / 7 / 9 / 11 / 13 / maj7 / maj9 … → b5 / #5 → sus4 / sus2 → add9 / add11 → "
                "括弧のテンションを低い順に。7の和音の数字は、含むナチュラルのテンションのうちいちばん低いもの）")
    return f"コード名『{name}』が読めない——{CHORD_HOW}"


# ★**1小節ぶんのコードの記法＝型1つ**（2026-09-29・`song_checks.Notation`）。読み方は `split_bar_chords`（その小節で鳴るコードの並び）、
#   所見はそのコード名のそれぞれ（`chord_problem`）。組み立て（`song_place.place_vocal` のコード表・`build_score`）も同じ `split_bar_chords` で読む。
#   コードの工程（`ChordsStep.written`）が宣言する
from song_checks import Notation, resolve_section, section_hint, section_id    # noqa: E402

#: ★**グルーヴの区切りの記号**（S/M/C を1行1文字・2026-09-30）。旋律・詞・息継ぎの所見が行ごとに読む。数はグルーヴの工程の検査が見る
#:   前後は半角の空白とタブだけ取り除く（前後の全角の空白・改行などは所見・`half_width`・§0-6・2026-10-03）。数を数える所も `strip_ascii_ws` で読む
BREAKS = Notation("区切りの記号",
                  half_width(lambda s: ([f"S/M/C 以外の文字（{' '.join(sorted(set(strip_ascii_ws(s)) - set('SMC')))}）がある——"
                                         "S＝文が終わる・M＝息を入れる・C＝そのまま続く、の3つだけ"]
                                        if set(strip_ascii_ws(s)) - set("SMC") else [])),
                  lambda s: list(strip_ascii_ws(s)))

def _chord_bar_problems(cell) -> list[str]:
    """1小節ぶんのコードの所見。★**空の要素も所見**（2026-10-01・ず-10）。以前は `""` が読めるコードの無い小節として通り、
    その小節は和音もベース・リフも鳴らなかった（`N.C.` と同じになる）。和音を鳴らさない小節は `N.C.` と書く（仕様書 §3）。"""
    names = split_bar_chords(str(cell))
    if not names:
        return ["小節が空——1小節に1つ以上のコードを書く（和音を鳴らさない小節は `N.C.`）"]
    return [w for w in (chord_problem(p) for p in names) if w]


CHORD_BAR = Notation("1小節ぶんのコード", half_width(_chord_bar_problems), split_bar_chords)


def transpose_chord(name: str, shift: int, key: str | None = None) -> str:
    """コード名を移調する（`Am7` → `Bm7`、`C/E` → `D/F#` のように分母も動かす）。

    ★**読めないコード名は `ValueError`**（2026-09-27・監査）。以前は黙ってそのまま返していたので、
      移調した曲の中でそのコードだけ元の調で鳴っていた。先に `chord_problem` で確かめて所見にすること。
      ノーコード（`N.C.`）は読める——移調するものが無いのでそのまま返す。
    ★**移した先の調 `key` を渡せば、根音と分数の低音をその調で綴る**（2026-10-01・判-I 本人「標準に寄せる方式で全体的に」）。
      以前は ♯ だけで綴り、F の曲の B♭ が `A#` と保存されていた（一般のコード譜は調号の向きで綴る）。綴りの部品は表と同じ
      `song_table.spell_chord`（2本目を作らない）。鳴る音は変わらない（読み手はどれも ♭ を読む・`render_base` も 2026-10-01 から）。
      `key` を渡さなければ以前どおり ♯ で綴る（調の分からない呼び側）。
    """
    n = strip_ascii_ws(name)
    if n in NO_CHORD:
        return name
    if shift:
        if parse_chord(n) is None:                         # 読めるかは組み立てと同じ文法（2026-10-02）
            raise ValueError(chord_problem(name))
        m = _CHORD_RE.match(n)

        def _mv(note: str) -> str:
            return _NAMES[(_PITCH[note] + shift) % 12]
        name = _mv(m["root"]) + (m["q"] or "") + (f"/{_mv(m['bass'])}" if m["bass"] else "")
    if key:
        from song_table import spell_chord                 # 綴りの部品は表と同じ1つ（循環 import を避けて呼ぶ時に読む）
        name = spell_chord(name, key)
    return name


# ---- 各ステップの出力型 -------------------------------------------------------

# ★**閉じた選択肢は型（`Literal`＝enum）で縛る**（2026-09-30 本人「選択肢は enum にしよう」・判-1）。
#   以前はただの文字列で、選択肢に無い綴りは検査も無く、読む側の `.get(x, 既定)` が黙って既定値にしていた
#   （genre は「ミドルのポップス」扱い・ending は `fade` 以外の綴りでフェードしない・拍子は小節の長さが黙って 4/4）。
#   外れたら pydantic_ai が LLM へ差し戻し、直らなければ `song_llm.run_sync` がその工程で止める（`StepUnfixed`）。
#   値は**読む側が実際に使う値**（`song_context._GENRE_NAME`・`brief` の表・`song_melody.VOICE_TOP` など）。
#   音源や曲で変わる選択肢（キット・音色・奏法・区間名）と調名は型にしない（形式の検査・調名は型の検証）。
Genre = Literal["pop", "children", "ballad", "anthem"]
VoiceTone = Literal["落ち着いた", "息づかい", "かわいい", "低め", "標準"]
VoiceHeight = Literal["低め", "中", "高め"]
PitchCenter = Literal["low", "mid", "high"]
#: 組み立て（`bar_ticks`・`units_per_bar`・`beats_per_bar`・vpr/MIDI の書き出し）が扱える拍子だけ。
#: 1小節の拍数が整数にならない拍子（5/8・7/8 など）は、旋律の枠の拍の頭（`song_melody_prompt.meter_text`）が数えられない
TimeSignature = Literal["4/4", "3/4", "6/8", "2/4"]
SingingStyle = Literal["続ける", "区切る", "詰める", "保つ"]
Device = Literal["食い", "伸ばし", "跳躍", "同音反復", "転調"]
Ending = Literal["fade", "hit", "vocal", "loop"]
DrumGrid = Annotated[Literal[8, 16], BeforeValidator(json_int)]   # `8.0` を 8 と読まない（整数の選択肢・判断5）

# ★**LLM が書く欄に既定値を持たせない**（2026-10-04・本人「書き忘れたのか標準にしたのかが曖昧になる。全部書いてもらったほうが
#   いい」・組N2）。以前はテーマ・設計・グルーヴの欄の多くが省けて、省くと型の既定（pop・標準・中・2020年代のJ-POP・C・4/4・fade …）に
#   なっていた。点検役も人も、書いた値か書き忘れかを見分けられなかった。いまは全部必須で、書き忘れは型が差し戻し、直らなければ
#   その工程で止まる（`song_llm.run_sync` の `StepUnfixed`）。「無し」は空の並び `[]`・null・0 を選んだことが分かる形で書かせる。
#   保存済みの記録は `dev_tools/song_legacy/convert_record._n2_defaults_fill` が当時の既定を書き込む（音は同じ）。
class ThemeApproach(BaseModel):
    """テーマの方向の案1つ（`Theme.approaches`・2026-10-05）。"""
    direction: str = Field(description=(
        "この案の**方向**を短く（例: 直球／暗示（別の物に託す）／視点をずらす（別の人物・別の時点から語る）／逆説）"))
    gist: str = Field(description=(
        "この案での**曲の芯**を一〜二文で——表で何を歌い、聴き終えたときに何が残るか"))


#: 「方向の案」の最少の数（2026-10-05・本人承認「異なる方向を2〜3案出して1つ選び、理由を書く」）。上限は置かない
#: （上限を書くと上限が既定値になる・`Design.devices` の実例）
APPROACHES_MIN = 2


def approach_problems(approaches, chosen, chosen_reason) -> list[str]:
    """**テーマの方向の比べ方の形式**（2026-10-05）。案が2つ以上・方向と中身が空でない・方向が重ならない・選んだ番号が案の中・
    選んだ理由が空でない。直球か凝った案かの良し悪しは見ない（内容＝旋律の点検役が `theme_direction_facts` を読む）。"""
    out = []
    aps = list(approaches or [])
    if len(aps) < APPROACHES_MIN:
        out.append(f"approaches（方向の案）が{len(aps)}つ——**異なる方向の案を{APPROACHES_MIN}つ以上**書いて比べる"
                   "（直球・暗示・視点をずらす など。直球を選ぶ曲でも、比べたうえで選ぶ）")
    for i, a in enumerate(aps, 1):
        if not (a.direction or "").strip() or not (a.gist or "").strip():
            out.append(f"approaches の{i}番: 方向（direction）と曲の芯（gist）の両方を書く（空は書けない）")
    names = [(a.direction or "").strip() for a in aps if (a.direction or "").strip()]
    if dup := sorted({n for n in names if names.count(n) > 1}):
        out.append(f"approaches の方向『{'』『'.join(dup)}』が重なっている——案ごとに違う方向にする")
    if aps and not 1 <= (chosen or 0) <= len(aps):
        out.append(f"chosen（選んだ案の番号）{chosen} は案の中に無い（1〜{len(aps)}）")
    if not (chosen_reason or "").strip():
        out.append("chosen_reason（選んだ理由）が空——ほかの案と比べて、なぜこの案がこの曲に合うかを書く"
                   "（直球を選んだなら、なぜ直球が合うか）")
    return out


def theme_direction_facts(theme) -> str:
    """**テーマの方向の比べ方の事実**（旋律の点検役の材料・2026-10-05）。案の数と方向・選んだ案・選んだ理由の字数だけ。
    良し悪し（直球か凝った案か）は書かない——比べた形跡が無いか（理由がほかの案に触れない等）は点検役が読んで決める。"""
    aps = list(getattr(theme, "approaches", None) or [])
    if not aps:
        return ""
    k = getattr(theme, "chosen", 0) or 0
    why = (getattr(theme, "chosen_reason", "") or "").strip()
    rows = "／".join(f"{i}. {a.direction}「{a.gist}」" for i, a in enumerate(aps, 1))
    pick = f"{k}（{aps[k - 1].direction}）" if 1 <= k <= len(aps) else f"{k}（案の中に無い）"
    return ("【テーマの方向の比べ方（機械が数えたもの・良し悪しの判定ではない）】\n"
            f"  方向の案: {len(aps)}つ——{rows}\n"
            f"  選んだ案: {pick}\n"
            f"  選んだ理由: {len(why)}字「{why}」")


class Theme(StepStatus):
    reason: str = Field(description="なぜこのテーマにしたか（依頼のどこを汲んだか）")
    # ★**方向を比べてから芯を決める**（2026-10-05・本人「選択肢がいろいろある中で、あえて直球を選んで歌詞にするならよいが、
    #   検討せずになんとなく作るのはクオリティ不足」・『朝光のトランク』の詞が素直すぎた）。以前は message・arc を1本道で決めていて、
    #   暗示・象徴・視点をずらすといった別の方向を考えた形跡が残らなかった。決め台詞の `hook_candidates`→`hook` と同じ「候補→選ぶ」の形。
    #   形式（案が2つ以上・選んだ番号・理由が空でない）は型が差し戻す（`_approaches`）。比べ方の良し悪しは旋律の点検役（`theme_direction_facts`）
    approaches: list[ThemeApproach] = Field(description=(
        "**曲の芯を決める前に、異なる方向の案を2〜3つ**（**必ず書く**）。例: 直球（言いたいことをそのまま言う）／"
        "暗示（言いたいことを別の物・出来事に託し、言葉では言わない）／視点をずらす（別の人物・別の時点から語る）。"
        "案ごとに `direction`（方向）と `gist`（その案での曲の芯）を書く。方向は案ごとに違うものにする"))
    chosen: JsonInt = Field(description="選んだ案の番号（approaches の1始まり）。message・arc・scene などは、この案の方向で書く")
    chosen_reason: str = Field(description=(
        "**なぜその案にしたか**（**必ず書く**）——ほかの案と比べて、この依頼・この曲に何が合うか。"
        "直球を選んでよい。そのときは、なぜ直球がこの曲に合うかを書く"))

    @model_validator(mode="after")
    def _approaches(self):
        """方向の比べ方の形式（`approach_problems`）を型で差し戻す。走らなかった出力（`skipped`）は見ない。"""
        if self.status == "skipped":
            return self
        if bad := approach_problems(self.approaches, self.chosen, self.chosen_reason):
            raise ValueError("／".join(bad))
        return self

    genre: Genre = Field(description=(
        "曲の種類（**必ず書く**）。**依頼の言葉から、次の4つのどれかを選ぶ**——"
        "pop=J-POP・ロック・バンドなど一般的な歌もの（ほかの3つに当たらなければこれ）／"
        "children=童謡・わらべうた・子供向け・手遊び歌／"
        "ballad=バラード・子守唄・静かな曲／"
        "anthem=応援歌・行進曲・合唱。"
        "種類によって作法が正反対になる（例: ポップスは音域を広く使うが、童謡は1オクターブに収める）"))
    # ★**声をここで決める**（2026-09-09 新設）。本人の指摘:
    #   「テンポに対してちょっと声が高かったかも。もっと低いトーンとか、ダークを使うとか、
    #     ソフトにするとか、できなかったのかな? **コンセプトの時点で検討した形跡はあった?**」
    #   → **無かった**。声は終盤の「編成」工程が独断で決めており、
    #     テーマ・設計は声に一切触れていなかった。しかも `Dark` も `Cute` も持っているのに、
    #     選択肢として現れていなかった（実際に選ばれたのは毎回 SOFT × Breath）。
    #
    #   現実の制作は逆で、**歌う人の声に合わせてキーを決めるのが最初のステップ**
    #   （さくなろ道場ほか）。こちらは**歌い手が最初から決まっている**（ミク）のだから、
    #   なおさら最初に決められる。「コンセプトが決まれば BPM も音色も自然と決まる」のと同じ従属関係。
    # ※2026-09-30 に既定を「迷ったら」の値（標準・中）にし、2026-10-04（組N2）に既定をやめて必須にした
    voice_tone: VoiceTone = Field(description=(
        "**この曲をどんな声で歌うか**（**必ず書く**。**次の5つから1つ選ぶ**）。曲想から決まります。\n"
        "・**落ち着いた** … 暗め・大人びた（夜・別れ・回想）\n"
        "・**息づかい** … 息を多く含む・ささやき寄り（静かな曲・内省）\n"
        "・**かわいい** … 明るく高め（元気な曲・子供向け）\n"
        "・**低め** … 少年寄りの落ち着き（クールな曲）\n"
        "・**標準** … 迷ったらこれ\n"
        "**テンポと合わせて考えてください**——ゆったりした曲に高く明るい声を当てると浮きます"))
    # ★**高さの正は設計の `top_note`**（2026-10-04・組N2）。ここは構想の「おおよその高さ」で、設計が `top_note` を選ぶ目安と、
    #   申し送り・楽器の主旋律の音域の目安になる。以前は `top_note` が空なら旋律の工程がここから最高音を引いていた（二重の既定）
    voice_height: VoiceHeight = Field(description=(
        "**おおよその高さ**（**必ず書く**。**次の3つから1つ選ぶ**）: **低め** / **中** / **高め**。"
        "設計がこの高さに合わせてサビの最高音（`top_note`）を音名で決めます。\n"
        # ★高さは国際式の音名で見せる（2026-10-01・仕様書 §0-1「MIDI 番号は LLM に見せない」。以前は 58〜86・76/81/85 の番号）
        "ミクが気持ちよく鳴るのは **A#3〜D6**（それ以上も出るが音量が落ちて痩せる）。\n"
        "低め＝サビの最高音が E5 前後／中＝A5 前後／高め＝C#6 前後 になります。\n"
        "**ゆったりした曲・落ち着いた曲は「低め」〜「中」**。速く元気な曲は「高め」でよい"))
    # ★**仮の曲名**（2026-10-03・本人了承「曲名の工程を足す」）。詞の設計・サビの詞・詞はこれを受けて書く
    #   （「タイトルが決まっていない状態で書き始めることはない」を仮の曲名に対して保つ）。**最後の曲名は詞のあとの工程『曲名』
    #   （`song_naming`・記録の `naming`）が決める**（そのまま採ってもよい）。`Song.title()` は 楽譜 → 曲名の工程 → これ。
    #   それまで（組F2）は欄名 `title` で、ここで決めた名前が最後まで曲名だった。古い記録の `title` は比較用の変換
    #   （`convert_record._naming_step`）がこの欄へ移す。「日本語で」は詞の設計の曲名の欄（判-11〜組F2）から移した
    working_title: str = Field(description=(
        "**仮の曲名**（**日本語で**）。詞の設計と詞はこれを受けて書きます。"
        "最後の曲名は詞のあとの工程『曲名』が決めます（この名前をそのまま採ることもあります）"))
    subject: str = Field(description="何を歌うか（一文）")
    message: str = Field(description=(
        "**この曲で伝えたいこと**を一文で。聴き終えたときに何が残ってほしいか。"
        "『寂しい』『楽しい』のような感情の名前ではなく、**言い切りの言葉**にする"
        "（例:「終わってしまうと分かっていても、その時間は無駄ではなかった」"
        "「誰にも見せない弱さこそが、前に進む力になる」）。"
        "**ここが曲の芯**で、後続の詞はこれを言うために書かれます"))
    arc: str = Field(description=(
        "**曲を通しての心の動き**（どこから始まり、どう変わり、どこへ着地するか）。"
        "例:「①去っていく季節を眺めているだけ → ②失う怖さに気づく → "
        "③それでも確かにあった時間を肯定する」。"
        "**同じことを繰り返さない**ために要る——特にフルコーラスでは、"
        "1番と2番で語り手の立ち位置が変わっていないと、ただの繰り返しになります"))
    scene: str = Field(description="情景・舞台（一文）")
    emotion: str = Field(description="核となる感情（一語〜数語）")
    # ---- 誰が歌っているか（2026-08-22 追加）----
    # ヒット曲の作り手は**人物を先に立ててから詞を書く**。優里は『かくれんぼ』で
    # 架空の男女を頭に置き、キャラクターが確立していたから『ドライフラワー』を
    # 女性目線で書けたと話している。うちのテーマ工程は「焦燥と衝動」のような**気分**しか
    # 出しておらず、誰が何を見て言っているのかが無かった。だから詞が情景の羅列になった。
    singer: str = Field(description=(
        "**誰が歌っているか**（**必ず書く**）。年齢・立場・性格を一文で"
        "（例:「就職して2年目、地元を出たまま帰っていない女性」"
        "「部活を辞めようか迷っている高校生」）。"
        "**『私』ではなく具体的な人物**にする——曖昧なままだと詞が誰の言葉でもなくなります"))
    listener: str = Field(description=(
        "**誰に向かって言っているか**（**必ず書く**。例:「別れた相手」「昔の自分」「まだ会っていない誰か」"
        "「誰にも言わず自分に」）。ここが決まると**言葉づかい**が決まります"))
    moment: str = Field(description=(
        "**どの瞬間の話か**（**必ず書く**）。時間・場所を具体的に"
        "（例:「終電を逃した明け方の駅前」「引っ越しの荷造りを終えた空の部屋」）。"
        "曲全体を通しての心の動き（arc）は、**この一瞬の中で起きる**のが自然です"))
    keywords: list[str] = Field(description="詞に使いたい言葉 3〜6個")
    # ★2026-10-04（組N2）: 以前は既定「2020年代のJ-POP」で、説明は LLM に選ばせるのに省くと J-POP に固定された
    era_style: str = Field(description=(
        "**どの時代・どの様式の音楽として作るか**（**必ず書く**。例:「2020年代のJ-POP」「1980年代シティポップ」"
        "「2010年代前半のボカロ曲」「フィンランドの伝統ポルカ」「1960年代モータウン」）。\n"
        "**ここで決めた様式に、構成・イントロの長さ・編成・リズムのすべてが従います。**\n"
        "工程ごとにばらばらに決めると**チグハグな曲**になるので、ここで大枠を固定する。\n"
        "依頼に時代や様式の指定があればそれに従い、無ければ曲想に合うものを選ぶ"))
    era_reason: str = Field(description=(
        "（**必ず書く**）**その様式を選んだ理由**と、**様式が要求すること**を具体的に。"
        "例:「2020年代のJ-POPなので、イントロは2小節以内で即座に歌に入り、"
        "Bメロは短く、全体を3分以内に収める」"))


class Section(BaseModel):
    """曲の区間。**名前は自由**——下流は名前ではなく `has_vocal` と `intensity` で判断する。

    以前は `intro / verse / pre / chorus / outro` の5種に限定していたため、
    **Cメロも間奏も存在せず、定番から外れた構造が作れなかった**（2026-08-21 に指摘を受けて是正）。
    """
    kind: str = Field(description=(
        "セクションの名前。**自由に付けてよい**。定番は intro / verse(Aメロ) / pre(Bメロ) / "
        "chorus(サビ) / bridge(Cメロ) / interlude(間奏) / outro。**これ以外でもよい**——"
        "曲想に合うなら drop / hook / break / 語り / 大サビ など、独自の呼び方や"
        "定番にない構造を作って構いません。**空は書けない**（前後の空白は取り除いて保存します）"))

    # ★**区間名そのものの文法**（2026-10-03・再レビュー 記G・仕様書 §15-3）。以前は空・前後の空白つきが型を通り、後の工程は
    #   設計の空白つきの綴りをそのまま書かされた（照らす側 `song_checks.section_id` は前後の空白を取り除くので、取り除いて書くと当たらない）。
    #   取り除き方は照らす側と同じ `section_id`（Python の `strip()`＝全角の空白・改行も・§0-6 の「設計で決める名前」）。空は差し戻す
    @field_validator("kind")
    @classmethod
    def _kind(cls, v: str) -> str:
        k = section_id(v)
        if not k:
            raise ValueError("区間名 kind が空——区間には名前を付ける（verse・chorus・Aメロ など。空白だけも空と同じ）")
        return k

    # ★「歌があるか」から「主旋律があるか」へ（2026-10-03・仕様書 §6・§15-4）。インストは楽器が主旋律（旋律の工程の音）を弾く区間。
    #   読み方は形に依らず1つ（歌もの・アカペラでは歌う区間＝以前と同じ）
    has_vocal: JsonBool = Field(description=(
        "この区間に**主旋律があるか**（**区間ごとに必ず書く**。true / false。歌もの・アカペラは歌う区間、インストは楽器が主旋律を弾く区間。"
        "イントロ・間奏・アウトロは false）"))
    intensity: JsonInt = Field(ge=1, le=5, description=(
        "**盛り上がりの度合い**（**区間ごとに必ず書く**。1=最も静か・薄い ／ 5=最も厚い・熱い）。伴奏の厚みの基準になる。"
        "**サビが必ず5とは限らない**——静かに終わるサビ、Cメロが最も熱い曲もある"))
    # ★**上限は置かない**（2026-10-03・`docs/plan/song_viewpoints_review.md` §2 c2）。以前は 1〜32 で、読み手（旋律の枠・コード・伴奏）は
    #   小節数を割り算に使うだけで上限を持たないのに、カバーの原曲に 33 小節以上の区間があると構成推定で止まった。下限 1 は要る（0 小節の区間は枠が無い）
    bars: JsonInt = Field(ge=1, description="小節数（1 以上）")
    # ★`mora_plan`（各フレーズのモーラ数）は 2026-09-30 に外した（出-13）。どの工程のコードも読まず、行数と1行のモーラ数は
    #   グルーヴが決める。それでも設計の出力は丸ごと後ろの全工程へ渡る（`song_context.upstream_intents`）ので、
    #   グルーヴと食い違う数（『茜色のワルツ』で設計のサビ [8,8,5,5,7]＝5行・グルーヴ 8行×6モーラ）が旋律・詞へ流れていた。
    #   古い記録の `mora_plan` は読み飛ばす（pydantic の既定＝知らない欄は無視）
    pitch_center: PitchCenter = Field(description=(
        "音域の狙い（**区間ごとに必ず書く**。**low / mid / high のどれか**。主旋律の無い区間も書く）。"
        "**intensity と一致させる必要はありません**"
        "（静かだが高い、熱いが低い、どちらもあり得る）"))


class KeyChange(BaseModel):
    """**曲の途中でキーを変える**（転調）。J-POP・ボカロでは定番の手法。

    以前は Design.key が曲に1つしかなく、**ラスサビの半音上げすら作れなかった**
    （2026-08-21 のセルフチェックで判明）。楽譜側は絶対音高なので器は元から転調できており、
    足りなかったのは**設計でキーを1つしか持てなかったこと**だけだった。
    """
    at_section: str = Field(description="このセクションの**頭から**キーが変わる（sections の kind と同じ名前で書く）")
    key: str = Field(description=("変わった後のキー（C / Am / Eb など。Design.key と同じ書き方。"
                                  "長調は何も付けない・短調は `m` だけ（`D major`・`Dmaj`・`D minor` とは書かない）。"
                                  "変化記号は `#` と `b` を1つだけ——記号の ♯ ♭ と、2つ重ねた Ebb・C## は書けない。"
                                  "途中に空白を入れない（`A m` ではなく `Am`））"))
    reason: str = Field(description="なぜここで転調するか（例: 最後のサビをもう一段持ち上げるため）")

    @field_validator("key")
    @classmethod
    def _readable_key(cls, v: str) -> str:
        """**読めない調は型で差し戻す**（2026-09-27・監査）。以前は黙って C で鳴っていた。"""
        why = key_problem(v)
        if why:
            raise ValueError(why)
        return v


#: **設計の最高音 `top_note` の下限**（MIDI・`D4`。2026-10-03 再レビュー 記F・仕様書 §6）。旋律の工程は鳴る高さの下限
#: （ミクの快適域の下端 `harmony.SING_LO`＝58、子ども向けは 62＝`song_melody_step` の `floor`）と最高音を C 基準へ移して枠にする。
#: 最高音が下限より低いと旋律が書ける高さが無い。転調の無い曲で、どの曲の種類でも矛盾しない値＝2つの下限の高いほう
#: （値の突き合わせはテスト `test_design_groove_notation.py`）
TOP_NOTE_MIN = 62


class Design(StepStatus):
    reason: str = Field(description="なぜこのテンポ・キー・構成にしたか")
    concept: str = Field(description=(
        "**この曲をどう作るかの全体方針**（後続の全工程がこれに従う）。"
        "何を聴かせたい曲か／どこが山で、そこへどう持っていくか／リズムと言葉の density はどうするか／"
        "どう終わるか。**具体的に**書く（例:「Aメロは低く抑えて言葉を刻み、サビの頭で一気に最高音へ"
        "跳ばして解放する。休符は短く、息継ぎの余裕を与えない切迫感で通す」）"))
    # ★**BPM は4分音符の数**（2026-09-30・出-5）。説明が無く、6/8 の曲（『茜色のワルツ』BPM72）は4分音符＝72＝付点4分＝48 の
    #   とても遅い 6/8 になった（6/8 の BPM はふつう付点4分で数える）
    bpm: JsonInt = Field(ge=60, le=200, description=(
        "テンポ。**どの拍子でも4分音符で数える**（1分間の4分音符の数・MIDI と同じ）。"
        "6/8 で「付点4分＝72」のつもりなら **108**（付点4分は4分音符1.5個ぶん）と書く"))
    key: str = Field(description=(
        "曲のキー（**必ず書く**。C / D / Eb / F / G / A / Bb などの長調、Am / Bm / Dm / Em などの短調）。"
        # ★長調は接尾なし・短調は `m` の1通り（2026-10-02 本人「調名の書き方を1通りにする」）
        "長調は何も付けない・短調は `m` だけ（`F major`・`Fmaj`・`D minor`・`Dmin` とは書かない）。"
        # ★記号の ♯♭ は 2026-10-02 から書けない（本人「2通りに書けるものは良くない」・§0-1）
        "変化記号は `#` と `b` を1つだけ（記号の ♯ ♭ と、2つ重ねた Ebb・C## は書けない。Ebb は D と書く）。"
        # ★途中の空白は 2026-10-02 から書けない（本人「おすすめで」・空白は記法全体で区切りの意味）
        "途中に空白を入れない（`A m`・`F #` ではなく `Am`・`F#`）。"
        "明るい曲は長調、切ない曲は短調。**旋律とコードは C メジャー / A マイナーで考えて書き、"
        "最後にこのキーへ移調する**ので、ここでは雰囲気に合うキーを選ぶだけでよい"))

    @field_validator("key")
    @classmethod
    def _readable_key(cls, v: str) -> str:
        """**読めない調は型で差し戻す**（2026-09-27・監査）。以前は `key_shift` が黙って C（0）にしていた
        （「D♭」「Dmaj」も C で鳴った）。pydantic_ai はこの文を設計の LLM へ返して書き直させる。"""
        why = key_problem(v)
        if why:
            raise ValueError(why)
        return v

    time_signature: TimeSignature = Field(description=(
        "拍子（**必ず書く**。**次の4つのどれか**）。**4/4**（基本・ほかの3つの揺れが曲想でなければこれ）／**3/4**（ワルツ・ゆったり揺れる・4分音符×3拍）／"
        "**6/8**（舟歌・子守唄・バラードのうねり。**付点4分×2拍の2拍子系**＝8分音符3つずつで揺れる）／"
        "**2/4**（ポルカ・マーチ・跳ねる民族舞曲）。"
        "★**「3拍子」「ワルツ」と頼まれたら 3/4**。6/8 は2拍子系なので、6/8・8分の6と頼まれたときと、"
        "その揺れ（大きな2つの拍の中で3つずつ揺れる）が曲想のときだけ選ぶ。"
        "曲想に合うなら 4/4 以外を選んでよい"))
    # ★**歌い方の型と、使う工夫を、ここで宣言する**（2026-09-09 新設）。
    #   本人の直感:「**初めの音楽作成コーディネーターの構想時点で、
    #     工夫や使い方などほとんど決めるべきなんじゃないか**」
    #   調査で裏が取れた——音楽プロデューサーは「コンセプトが決まれば BPM も自然と決まる。
    #   使う音色やジャンル感、**ビートのパターン**などもだいたい決まってくる」と言う。
    #   つまり**リズムの性格は構想の従属変数**で、区間ごとに独立に決めるものではない。
    #
    #   それまでは旋律工程のプロンプトに「常にこうしろ」と書いてあるだけで、
    #   **どの曲でも同じ指示**が出ていた。だから作風を選べず、
    #   「指示を1つ足すと別の何かが壊れる」を繰り返した（2026-09-07〜09 に4回）。
    # ★**サビの最高音を数字で決める**（2026-09-09 新設）。
    #   現実の判断基準そのまま——「キーを決める重要なポイントは、**その曲の一番高い音が
    #   ボーカリストが音楽的に出せるかどうか**。高すぎれば下げ、余裕がありすぎるなら上げる」
    #   （TRIVISION STUDIO）。
    #
    #   それまで音域は `pitch_center` の **low / mid / high という語だけ**で渡していた。
    #   実際に「verse は low」と決めた曲で、**曲全体の中央が D5(74)・最高音 E6(88)**
    #   （上限 86 超え）になり、本人に「テンポに対して声が高かった」と言われた。
    #   **語では効かない。数字で渡す。**
    # ★**国際式の音名で書かせる**（2026-10-01・本人「楽譜（国際式の音名）に寄せる」）。以前は MIDI 番号（`85`）で、旋律の外で
    #   LLM が書く高さ（伴奏・彩り・声部の音名）と物差しが違った。高さの範囲は以前と同じ（0＝自動 → 空・上限 96＝`C7`）。
    #   読み手は音名の記法の型（`song_llm.PITCH`）1つ。保存済みの記録の数は `dev_tools/song_legacy/convert_record.upgrade` が音名へ写す
    # ★**最高音の正はここ**（2026-10-04・組N2）。以前は既定が空で、空なら旋律の工程が構想の `voice_height` から
    #   表（`song_melody.VOICE_TOP`＝低め E5・中 A5・高め C#6）で引いていた（二重の既定）。いまは必ず音名で書き、空は型が差し戻す
    top_note: str = Field(description=(
        "**サビでいちばん高くなる音**（**必ず書く**）。国際式の音名（`C4`＝中央のド・`E5`・`F#5`・`Bb5`。変化記号は `#` と `b`、"
        "オクターブの数字は必ず付ける）で、**実際に鳴る高さ**（この曲のキーで）を書く。"
        "構想の「おおよその高さ」（申し送りの ■歌う高さ）を、下の範囲の中で曲ごとの1音に決める。\n"
        "目安の範囲: **低め C5〜F#5**／**中 F#5〜B5**／**高め B5〜D6**。\n"
        "`A#3`〜`D6` は「鳴ることを確かめた範囲」で、**壁ではありません**。`D6` を超えると"
        "音量が落ちて痩せますが、それを表現として使うなら書けます（上限 `C7`。**`E6` を超えるのは狙うときだけ**）。"
        "**`D6` を超えるなら理由を `reason` に書くこと**——点検役が読み、理由の無い超過は意図の無い高さとして扱います。\n"
        "**構想で決めた「おおよその高さ」に合わせ、範囲の中で曲ごとに決めること。**"
        "下限は `D4`（これより低いと、旋律の最低音の枠より低くなり旋律が書けない）"))

    @field_validator("top_note")
    @classmethod
    def _readable_top_note(cls, v: str) -> str:
        """**読めない音名・上限（`C7`＝96）を超える高さ・下限（`D4`＝62）より低い高さは型で差し戻す**（2026-10-01。上限は以前の `le=96` と同じ。
        下限は 2026-10-03 再レビュー 記F・`TOP_NOTE_MIN`）。記法の所見は伴奏・声部の音名と同じ文（`PITCH.problems`）。"""
        if not strip_ascii_ws(v):                           # 前後は半角の空白とタブだけ取り除く（全角の空白などは PITCH の所見・§0-6）
            # 空は書き忘れ（2026-10-04・組N2。以前は空＝声の高さから自動）
            raise ValueError("top_note が空——サビでいちばん高くなる音を国際式の音名で書く。構想の高さが 低め なら C5〜F#5・"
                             "中 なら F#5〜B5・高め なら B5〜D6 の中から1音（例 `A5`）")
        if bad := PITCH.problems(v):
            raise ValueError("top_note: " + bad[0])
        if PITCH.read(v) > 96:
            raise ValueError(f"top_note: 音名『{v}』は上限（`C7`）より高い")
        if PITCH.read(v) < TOP_NOTE_MIN:
            raise ValueError(f"top_note: 音名『{strip_ascii_ws(v)}』は下限（`D4`）より低い——旋律の最低音の枠（ミクの快適域の下端 `A#3`、"
                             "子ども向けの曲は `D4`）より低いと、旋律が書ける高さが無くなる。サビでいちばん高くなる音を `D4` 以上で書く"
                             "（目安は `C5`〜`D6`）")
        return strip_ascii_ws(v)
    # ※4つのどれも「迷ったら」の値ではない。null は「型を決めない」を選んだ印で、申し送りに載らない
    #   （2026-10-04・組N2 で既定 None をやめ、null も明示で書かせる）
    singing_style: SingingStyle | None = Field(description=(
        "**この曲の歌い方の型**（**必ず書く**。**次の4つから1つ選ぶ**。どれにも当てはめない曲だけ null と書く）。"
        "実測で J-POP はこの4つに分かれる:\n"
        "・**続ける** … 切れ目なくつなぐ（レガート・語り）。休みが少なく、1音が長い\n"
        "・**区切る** … 短く区切って畳み掛ける（スタッカート・掛け合い）。休みが多く、伸ばしが短い\n"
        "・**詰める** … 言葉数を多く入れる。1音が短く、密度が高い\n"
        "・**保つ** … 一音一音をたっぷり伸ばす。密度が低く、伸ばしが長い\n"
        "**題材と心の動きから決める**（静かな回想なら『続ける』、畳み掛ける決意なら『区切る』）。"
        "決めた型は**旋律・グルーヴ・作詞の全部が従う**"))
    # ★`max_length` は付けない——**上限を書くと、上限が既定値になる**（下の説明文の実例: 「0〜3個」と書いたら毎回3つ全部、
    #   「2〜4文字あける」と書いたら全部4文字）。スキーマの上限も LLM に見えるので同じことが起きうる。
    #   絞り込みは指示（下の説明文）と、`reference_text` 側で**先頭2つだけ使う**ことで担保する。
    devices: list[Device] = Field(description=(
        "**この曲で使う工夫**（**必ず書く**。**下の5つから選ぶ**。使わないなら空の並び `[]` と書く＝工夫なしを選んだ）。★**ふつうは1〜2個に絞る**——"
        "たくさん使うと、どれも効かなくなります（「これが効いている曲だ」と分かるのは、**絞ったとき**）。\n"
        "3つ以上は、**あえて工夫を重ねる曲にする意図があるときだけ**。そのときは、なぜ重ねるのかを `concept` に書く"
        "（書いた工夫は全部、旋律の工程へ指示が出ます。機械は数を絞りません）。\n"
        "> ★実際に起きた失敗（2026-09-09）: 「0〜3個」と書いたら**毎回3つ全部**が選ばれた。"
        "同じ日に「音節は2〜4文字あける」と書いたら**全部4文字**、"
        "「拍の頭を避けて」と書いたら**拍頭ばかり**になった——"
        "**上限を書くと、上限が既定値になります。**\n"
        "**選ばなかった工夫は使われません**（旋律工程へ指示が出ません）。\n"
        "・**食い** … 小節の最後の8分から出して次の小節へまたぐ（前のめり・推進力）\n"
        "・**伸ばし** … 区切りの大きいところで長く保つ（余韻・着地）\n"
        "・**跳躍** … 4〜8度の跳び上がりで山を作る（開放感）\n"
        "・**同音反復** … 同じ音を続けて言葉を立てる（語りかけ・切迫）\n"
        "・**転調** … 途中でキーを上げる（後半の高揚）\n"
        "**心の動きのどこで効かせたいか**を `concept` に書くこと"))
    # ★2026-09-29: 以前は「合計小節数は 8〜24」と小節で書いていて、指示書の「尺は秒で決める」と食い違っていた
    #   （小節で考えると速い曲ほど短くなる——BPM200 の 24小節は 29秒）
    sections: list[Section] = Field(description=(
        "曲の並び。**歌のある区間の長さは秒で決める**（指定が無ければ1番ぶん＝歌の区間が合計 45〜75秒。"
        "小節数 ＝ 秒数 × BPM ÷ 60 ÷ 1小節の4分音符の数（4/4＝4・3/4 と 6/8＝3・2/4＝2））。1区間は最低 8小節。"
        "**区間名（kind）は曲の中で1つずつ**（2番は `verse2`、ラスサビは `chorus_last` のように別の名前にする）。"
        "**主旋律のある区間（has_vocal が true）を1つ以上**入れる"))

    @field_validator("sections")
    @classmethod
    def _has_lead(cls, v: list) -> list:
        """**主旋律のある区間（`has_vocal` が true）が1つ以上**（区間が0も同じ）を型で差し戻す（2026-10-03・再レビュー 記D・仕様書 §6）。
        以前は0でも型を通り、旋律の工程で枠が無く落ちた（原因の工程で止める）。主旋律＝歌もの・アカペラは歌う区間、インストは楽器が
        主旋律を弾く区間で、どの形でも1つ以上要る（形を知らなくても決まるので型に置く）。"""
        if not any(s.has_vocal for s in v):
            raise ValueError(f"主旋律のある区間（has_vocal が true）が1つも無い（区間 {len(v)}個）——主旋律の区間"
                             "（歌もの・アカペラは歌う区間、インストは楽器が主旋律を弾く区間）を1つ以上入れる。"
                             "前奏・間奏・後奏だけでは旋律を書く所が無い")
        return v

    @field_validator("sections")
    @classmethod
    def _unique_kinds(cls, v: list) -> list:
        """**同じ区間名が2つ以上あれば型で差し戻す**（2026-09-30 本人承認・調名と同じ形）。

        区間名は「その区間のコード・詞・旋律・伴奏」を結ぶ鍵で、重なると1番と2番の区別が付かず時間軸が組めない。
        答えが1つに決まる（形式）ので、以前の「点検役へ所見として渡す」（`song_compose.inspect_design`）から型へ移した。
        どちらを 1番・どれをラスサビにするかは設計の LLM が名前を付け直して決める（機械は番号を振らない）。
        """
        seen: dict[str, int] = {}
        for s in v:
            k = getattr(s, "kind", "")
            seen[k] = seen.get(k, 0) + 1
        dup = [f"区間名『{k}』が{n}回ある" for k, n in seen.items() if n > 1]
        if dup:
            raise ValueError("／".join(dup) + "——区間名は曲の中で1つずつ（2番は verse2 のように別の名前にする）")
        return v

    @model_validator(mode="after")
    def _modulation_sections(self):
        """**転調の区間名（`at_section`）が設計の区間名にあるか**を型で差し戻す（2026-09-30・判-2 本人承認）。

        組み立て（`song_place`・`song_context` の区間の調）は `resolve_section` で引き、
        合わなければ**どの区間にも当たらず、転調が黙って効かなかった**。区間名の重複（`_unique_kinds`）と同じく、
        答えが1つに決まる形式で、設計の工程の中で見られるので型に置く。照らし方は同じ `song_checks.resolve_section`
        （2026-10-02 から**完全一致だけ**・判断6。以前は大文字小文字・空白の違いを吸収した。違いが大文字小文字・空白だけなら綴りを添える）。
        """
        names = [s.kind for s in self.sections]
        bad = [m.at_section for m in self.modulations if resolve_section(m.at_section, names) is None]
        if bad:
            hints = [f"『{b}』→『{h}』" for b in bad if (h := section_hint(b, names))]
            raise ValueError("転調の at_section " + "・".join(f"『{b}』" for b in bad) + " が区間名に無い——"
                             f"sections の kind と1字も違わない名前で書く（区間名: {' / '.join(names)}。大文字小文字・空白の違いも別の名前）。"
                             + (f"綴りを合わせる: {'・'.join(hints)}。" if hints else "")
                             + "合わない名前だと、転調がどの区間にも効かない")
        # ★**1つの区間に転調は1つだけ**（2026-10-03・再レビュー 記E・本人決定・仕様書 §6）。区間の調は区間名で引く表
        #   （`song_place`・`song_context` の区間の調）にするので、以前は同じ区間へ2つ書くと後のものが黙って勝った。
        #   照らし方は上と同じ（`resolve_section`＝前後の空白だけ取り除く）
        at = [resolve_section(m.at_section, names) for m in self.modulations]
        if dup := [k for k in dict.fromkeys(at) if at.count(k) > 1]:
            raise ValueError("同じ区間への転調を2つ以上書いた: "
                             + "・".join(f"『{k}』に{at.count(k)}つ（"
                                        + " / ".join(m.key for m, a in zip(self.modulations, at) if a == k) + "）" for k in dup)
                             + "——1つの区間に転調は1つだけ（区間の頭から調が変わる）。区間の途中で転調したいなら、"
                               "sections で区間を分けて書き、分けた後ろの区間の頭に転調を書く")
        return self
    # ★**終わり方を設計の時点で決める**（2026-08-23 ユーザー指摘）。
    # 「最後が弱い」と言われた曲は、アウトロが2小節あるだけで**どう終わるかを誰も決めていなかった**。
    # 構成を決めるここで方針を出し、編曲がそれに従う。
    ending: Ending = Field(description=(
        "**どう終わるか**（**必ず書く**。**次の4つのどれか**）。ラスサビの後をどうするかは、ここで決めないと誰も決めません:\n"
        "  fade    … **余韻を残して減らしていく**（4〜8小節かけて薄くする）。バラード・静かな曲\n"
        "  hit     … **最後に全部鳴らして切る**（ジャーン、で終わる）。ロック・応援歌・盛り上げて終わる曲\n"
        "  vocal   … **歌だけ残して消える**（伴奏が先に止み、最後の一言が残る）。余韻の強い終わり方。"
        "**歌もののときだけ**（アカペラ・インストでは選べない——依頼文の【曲の形】）\n"
        "  loop    … 終わった感じを作らず、そのまま切る（ショート動画・TikTok 向け）\n"
        "**アウトロの小節数もこれに合わせて決めてください**——"
        "`fade` なのに2小節しかないと、減らす間もなく終わります（4〜8小節は要る）。"
        "`hit` なら1〜2小節で足ります"))
    ending_reason: str = Field(description=(
        "その終わり方を選んだ理由（**必ず書く**）。**曲の言いたいこと（message）とどう繋がるか**を書く"))
    modulations: list[KeyChange] = Field(description=(
        "**転調**（**必ず書く**。転調しないなら空の並び `[]` と書く＝転調なしを選んだ）。定番は\n"
        "・**ラスサビで半音〜全音上げる**（C → C# / C → D）——最後にもう一段持ち上げる\n"
        "・**Cメロだけ平行調へ**（C → Am / Am → C）——雰囲気を変えて戻ってくる\n"
        "・**同主調へ**（C → Cm / Am → A）——同じ主音で明暗を切り替える\n"
        "**必ず入れるものではありません**。曲が転調を必要としているときだけ使い、"
        "**必要なら2回以上**入れてもよい。転調したあと元に戻す場合も、戻る先を1件として書く。"
        "**1つの区間に転調は1つだけ**（区間の頭から変わる。区間の途中で変えたいなら、sections で区間を分けて書く）"))


#: **コード名の書き方**（型の説明に入れる文・1か所）。コードの工程 `SectionChords.chords_per_bar` と、
#: 尺の変更 `song_resize.NewSection.chords`・その依頼文が同じ文を使う（2026-10-02・組「洗い出し-型と残り」。以前は尺の変更の説明に
#: 書き方が無く、コードの工程と同じ決まりで所見になるのに LLM は知らなかった）。
# ★2026-09-30（記-12）: 書けないコード名を型の説明にも書く（以前は所見の文で初めて知った）
# ★2026-10-02（本人決定「字の袋を並びに」「同じ意味の書き方を1通りに」）: 種類は決まった順に1つずつ・同じ和音の書き方は1通り。
#   文法と組み合わせの決まりは `song_audio.parse_chord`（仕様書 §3）
CHORD_WRITING = (
        "書けるコード名: 根音（C D E F G A B）＋変化記号（# / b）のあとに、種類を**決まった順に1つずつ**: "
        "m / dim / aug → 6 / 7 / 9 / 11 / 13 / maj7 / maj9 / maj11 / maj13 → b5 / #5 → sus4 / sus2 → add9 / add11 → 括弧のテンション。"
        "例 C・Am・G7・Am7・Fmaj7・Cmmaj7・C6・Am6・C9・Am9・Am11・G13・Fmaj9・Cadd9・Amadd9・Csus4・Csus2・G7sus4・Bdim・Bdim7・Bm7b5・"
        "Caug・G7#5・G7b5・C6add9。"
        "**同じ和音の書き方は1通り**: 長三和音は根音だけ（CM・Cmaj は書けない）、短調は m（Cmin は書けない）、長7は maj7（CM7・CΔ は書けない）、"
        "減は dim（C°）、半減7は m7b5（Cø・Cm7-5 は書けない）、増三和音は aug（C+）・7度つきの増5度は 7#5（Caug7 は書けない）。"
        "7の和音の数字は、含むナチュラルのテンション（9 11 13）のうちいちばん低いもの（C7(9) → C9・C7(11) → C11・Cmaj7(9) → Cmaj9）。"
        "括弧には変化したテンション（b9 #9 #11 b13）と、数字より上のナチュラルを低い順に `,` で（G7(b9)・G7(b9,#9)・C9(13)・G13(b9)・"
        "Cmaj9(#11)。G7b9・G7-9・G7+9・C7(9,13) は書けない）。11・13 の和音は9度を足さずに鳴る（9度も鳴らすなら C9(11)・C9(13)）。"
        "7度の無い和音に足す音は add9・add11（C(9) → Cadd9）、♭6 は Cm(b6)、長三和音の ♭5 は C(b5)。"
        "分数は G/B、和音なしは N.C.（N.C・NC は書けない）。"
        # ★変化記号は ASCII の `b` `#` で教える（2026-10-01・ず-1・仕様書 §0-1・§3 の「正の綴り」）
        "変化記号は `#` と `b`（Bbmaj7・F#m7b5）。記号の ♯ ♭ は書けない。根音・低音に変化記号を2つ重ねない（Ebb → D・C## → D）。"
        "数字は半角で、頭に 0 を付けない。"
        "**1要素に1つ以上**書く（空の要素は書き直しを頼む。和音を鳴らさない小節は N.C.）。"
        "**書けない**: パワーコード C5・3度抜き no3/omit3・`^`・alt・6/9（→ C6add9）・sus2 と sus4 を一緒に（→ Csus4add9）・"
        "dim9 など（→ dim7）・短調の `-`（C-7 → Cm7）・Maj7/ma7/mi7/Min（→ maj7・m7・m）・ドイツ式の H（→ B）")


class SectionChords(BaseModel):
    kind: str
    chords_per_bar: list[str] = Field(description=(
        # ★**空白区切りへ**（2026-09-21）。それまで `Am|F` と `|` で区切らせていたが、
        #   **一般のコード譜では `|` は小節の区切り**で、意味が逆だった。
        #   同じリポジトリの旋律（ABC）とドラムタブでは `|` が小節線なので、**ここだけ孤立**していた。
        #   ※`|` を黙って受ける読み方は 2026-09-30 に外した（`split_bar_chords`）。書けば読めないコード名の所見になる。
        "1要素=1小節（C / Am / F / G7 等）。**1小節で2つ以上鳴らすときは `Am F` のように空けて並べる**"
        "（コード譜と同じ書き方。**並べた数で小節を均等に分ける**＝2つなら前半2拍・後半2拍）。"
        "動きを出したい所だけに使い、全部を割らないこと。\n" + CHORD_WRITING))


class Chords(StepStatus):
    reason: str = Field(description="進行の狙い（サビ前の緊張・サビ末の着地など）")
    sections: list[SectionChords]




# ---- エージェント -------------------------------------------------------------

# ロールは**工程ごとに別名**にする。同じ名前を共有すると `MIKU_<ROLE>_MODEL` での上書きが
# 巻き添えになり、「テーマだけ flash に落とす」といった調整ができない（2026-08-20）。
# テーマは flash（2026-08-20 実測）。出力が数行で判断も単純なため、pro との差が出なかった
# （『晩夏の忘れ物』vs『ラムネの瓶に沈む夕陽』——曲種の判定も具体性も同等）。
# 旋律だけは pro を維持する——flash に落とすと最短音符 0.25→0.42・ロングトーン率 0.35→0.50 と
# **譜割りが崩れ**、今回の改修の中心が台無しになることを実測で確認した。
# テーマは **pro に戻した**（2026-08-21）。いったん flash に落としたが、
# 実機の試聴で「曲で伝えたいものが見えてこない・構想が甘い」と指摘された。
# 出力が短いことと判断が軽いことは別で、**ここは曲の芯を決める工程**だった。
# ---- グルーヴ（2026-09-07 新設）------------------------------------------------
# ★**歌より先にノリを作る**。プロは「ドラムパターンとコードだけでも良い」オケの上で
#   鼻歌を歌ってメロディを作る（ドラム→ベース→仮ピアノ→仮メロディ）。
#   こちらは旋律工程が**コード名しか受け取っておらず、鳴っているリズムを知らなかった**——
#   拍頭率 0.55（ヒット曲 0.12〜0.33）＝「朗読のよう」の原因。
#   実験で、グルーヴを渡すと**型の再出現 0.57→0.88・拍頭率 0.55→0.36** と動いた。

class GrooveLine(BaseModel):
    # ★一覧は伴奏のドラムの行と同じ `timbre_map.DRUM_NAMES`（2026-10-02 から GM の打楽器 35〜81 を全部名前で書ける）。
    #   以前は14種を手で書いていて、検査（`GrooveStep.check`）が通す名前と食い違っていた
    # ★2026-10-02: 打楽器名は GM の正式な名前だけ（以前の短い名前 kick・hihat … と別名はやめた・書き方は1通り）
    inst: str = Field(description=(
        "打楽器名（伴奏のドラムの行と同じ名前＝GM の打楽器の正式な名前を小文字・`_` つなぎにしたもの）: " + " / ".join(DRUM_NAMES)
        + "。kick・snare・hihat のような略称は書けない。開いたハイハットは open_hi_hat の行に書く（closed_hi_hat の行に `o` は書けない）。"
        "よく使うのは bass_drum_1 / acoustic_snare / closed_hi_hat / crash_cymbal_1 / ride_cymbal_1 / low_tom ほか"))
    pattern: str = Field(description=(
        # ★**編曲工程と凡例を揃える**（2026-09-21）。それまで**同じ記号が逆の意味**だった——
        #   ここは `-`＝鳴らさない／`_`＝前を伸ばす、編曲は `-`＝前を伸ばす／`_`＝そこで切る。
        #   しかもここで書いた文字列は**そのまま編曲工程のプロンプトへ貼られる**ので、
        #   前段の例を見て後段が書くと**凡例が逆に働く**。ドラムタブに揃えて解消した。
        "**ドラムタブ（drum tab）**で書く。1小節＝**8分音符8個**（4/4。拍子が違う曲は依頼文の数・3/4 と 6/8 なら6個。倍の文字数で16分）。\n"
        "**`o`＝ドラムを叩く**（bass_drum_1 / acoustic_snare / low_tom ほか）／"
        "**`x`＝シンバル類**（closed_hi_hat / crash_cymbal_1 / ride_cymbal_1 ほか）／"
        # 2026-10-02（ず-9・本人「2通りに書けるものは良くない」）: 以前は `_` も休みとして読んでいた。休みは `-` だけ・`_` は所見
        "**大文字 `O` `X`＝アクセント**／**`.`＝ゴーストノート**／**`-`＝休み**（休みは `-` だけ。`_` は書けない＝検査で所見として返します）。"
        # 2026-10-02（打楽器名を GM の正式な名前に一本化）: 判-J（10/1）の「hihat の行の `o` は開いたハイハット」は、
        #   名前が closed_hi_hat／open_hi_hat に分かれたので、開くのは open_hi_hat の行だけにした（closed_hi_hat の行の `o` は所見）
        "開いたハイハットは open_hi_hat の行に書く（closed_hi_hat の行の打点は `X` `x` `.` だけ。`o` は書けない＝検査で所見として返します）。\n"
        "★**書くのは1小節ぶんだけ**（小節ごとに繰り返されます）。**2小節続けて書かないこと**。"
        "**途中に空白を入れない**（`o--- o---` ではなく `o---o---`。書くと検査で所見として返します）。\n"
        # ★2026-09-30（記-2）: snare の例が `---o---o` で、「2・4拍」のつもりが8文字では2拍裏・4拍裏に鳴っていた
        "例（4/4・8文字）: bass_drum_1 `o---o---`（1・3拍）／ acoustic_snare `--o---o-`（2・4拍）／ closed_hi_hat `x-x-x-x-`。"
        "6/8（6文字・拍の頭は1・4文字目）なら bass_drum_1 `o-----` ／ acoustic_snare `---o--`"))


class SectionGroove(BaseModel):
    kind: str = Field(description="区間名（設計で決まったものをそのまま）")
    drums: list[GrooveLine] = Field(description="この区間のドラム。薄い区間は1〜2本でよい")
    # ★2026-10-04（組N2）: 以下の欄は既定をやめて**区間ごとに必ず書く**（以前は bass `X---X---`・lines/moras 0・
    #   notes_per_beat 1.0・tail_rest 0.30・drum_grid 8・breaks 空が省いたときの値）。歌の無い区間は lines・moras_per_line を 0、
    #   breaks を空、notes_per_beat・tail_rest を null と書く（「無し」を選んだ印）
    bass: str = Field(description=(
        # ★**ベースはドラムタブではない**（音程楽器なので音が伸びる）。記号が一部違うので明記する
        #   ——同じプロンプトの中で凡例が2つあるのに、書き分けていなかった（2026-09-21）。
        "ベースのリズム（**区間ごとに必ず書く**。歌の無い区間も、その区間のベースの刻みを書く。同じ8分グリッド）。"
        "**X x . ＝叩く（強さ）／`-`＝前の音を伸ばす／"
        "`_`＝そこで音を切る**。例: `XxXxXxXx`（8分で刻む）／`X-------`（白玉＝1小節伸ばす）／"
        "`X_______`（頭で1発、あとは無音＝サビ前の溜め）"))
    # ★**範囲の上限は型に置かない**（2026-10-03・`song_viewpoints_review.md` §2 c3〜c6・d4）。行数は「小節数まで」（`GrooveStep.check`）だけが
    #   読み手の決まり（`song_melody._line_slots` は任意の行数に割れる）。1行のモーラ数・密度・行末の空きは機械が数として読まず
    #   依頼文に出すだけなので、説明の数は**目安**として残し、型で差し戻すのは読み手が壊れる値だけ:
    #   `notes_per_beat` の 0 以下（読み手が `or 1.0` で黙って 1.0 と読む）と `tail_rest` の 1 以上（行に音節を置く場所が無くなる）
    lines: JsonInt = Field(ge=0, description=(
        "**この区間の歌は何行か**（**区間ごとに必ず書く**。小節数まで）。同じ区間の中では行の長さを揃える"
        "（1行目と2行目が同じ型だから聴き手は乗れる）。"
        "**イントロ・間奏・アウトロなど歌が無い区間は 0**"))
    moras_per_line: JsonInt = Field(ge=0, description=(
        "**1行あたりのモーラ数**（**区間ごとに必ず書く**）。歌のある区間は **6〜12** が目安。"
        "★**`tail_rest` が 0.1 未満（休まない）のときは 16 くらいまで**"
        "（1行2小節を8分で埋めると16モーラになる。12 で止めると休みだらけになり、"
        "宣言と逆のぶつ切りになります）。"
        "**5以下だと単語1つしか入らず「単語を並べただけ」になる**（実際に起きた失敗）。"
        "『モーラ数 × 0.29秒』が1行の時間の目安で、区間の秒数に収まること。"
        "**歌のある区間は 1 以上**（0 は検査で所見として返します）。**歌が無い区間は 0**"))
    notes_per_beat: JsonFloat | None = Field(gt=0, description=(
        "**1拍に音節をいくつ置くか**（**区間ごとに必ず書く**。歌のある区間は数、**歌の無い区間は null**。歌っているあいだの密度。**1拍＝4分音符**で、BPM と同じ数え方——6/8 でも同じ）。"
        "★**旋律と詞の両方が読む**。\n"
        "★**この数字がそのまま密度になります**: **1秒あたりの音数 ＝ これ × BPM ÷ 60**。\n"
        "入るモーラの合計は **拍数 × これ**（`tail_rest` は掛かりません）。\n"
        "**ふつうの曲は 1秒あたり 1.4〜1.8音**（J-POP 73曲の実測・中央 1.78）なので、"
        "**BPM95 なら 1.1／BPM120 なら 0.85／BPM200 なら 0.65** あたりが『ふつう』。\n"
        "**言葉を増やしたいならこの数字だけを上げます**（動かす数字は1つ）。"
        "0 より大きい数（目安は 0.5〜2.0）"))
    tail_rest: JsonFloat | None = Field(ge=0.0, lt=1.0, description=(
        "**行末をどれだけ空けるか**（**区間ごとに必ず書く**。歌のある区間は数、**歌の無い区間は null**。"
        "行の長さに対する割合。0 以上 1 未満・目安は 0〜0.5）。★**旋律と詞の両方が読む**。\n"
        "★**これは「置き方」であって、入る量ではありません**（2026-09-18）——"
        "モーラの合計に**掛かりません**。音を伸ばすか休むかの違いを決めるだけです。\n"
        "**0.30＝ふつう**（音節を行の前半〜中ほどに固め、余りを行末の休みにする。"
        "息を継ぐ場所がはっきりして言葉が立つ）／"
        "**0.0＝休まない**（4つの拍に均して置き、次の行へ流し込む。"
        "★ボカロは息継ぎが要らないので、人が歌えない長さまで続けられる）。\n"
        "★**0.1 を下回ると、関門は「息が続かない」を言いません**（狙ってやったとみなす）"))
    # ★**刻みはドラムの行の文字数だけで決まる**（2026-10-03・再レビュー 記M・仕様書 §15）。この欄は伴奏の依頼文の案内
    #   （`song_band.run_arrangement`「ドラムの刻みは16分」）にだけ使い、行の文字数と食い違えば `GrooveStep.check` の所見
    drum_grid: DrumGrid = Field(description=(
        "**ドラムの刻みの細かさ**（**区間ごとに必ず書く**。**8 か 16 のどちらか**。8＝8分・16＝16分）。編曲工程への案内。\n"
        "★**実際の刻みは `drums` の行の文字数で決まります**——この区間のどれかの行を16分の文字数（4/4 なら16文字）で書いたら 16、"
        "全部8分の文字数（4/4 なら8文字）なら 8 と書く（食い違うと検査で所見として返します）。\n"
        "**BPM150 以上は 16** を選ぶこと——速さは数字ではなく**1秒に何回刻むか**で耳に届きます。"
        "BPM200 で4分刻み（300ms）は **BPM100 の8ビートと同じ密度**で、半分の速さに聞こえます。\n"
        "★**薄い区間でも刻みは落とさない**（薄さはパート数と音域で作る）"))
    breaks: str = Field(description=(
        "**行ごとの区切りの強さ**（**区間ごとに必ず書く**。歌の無い区間は空の文字 `\"\"`）。`lines` と同じ数だけ、1行1文字で並べる。"
        "**S**＝そこで文が終わる（大きく区切る）／**M**＝句点は無いが息を入れる／"
        "**C**＝次の行へそのまま続く。例（4行）: `CMCS`。**歌が無い区間は空**。"
        "★これは**旋律と詞の両方が使う**——旋律は『S の行は長く伸ばして次に長く休む、"
        "C の行は短く切ってすぐ続ける』、詞は『S の行で文を終える』。"
        "**区間の最後の行はふつう S**（そこで区間が終わるため）"))


def groove_number(g: "SectionGroove", name: str) -> float:
    """**歌のある区間の密度・置き方**（`notes_per_beat`・`tail_rest`）を数で読む（2026-10-04・組N2）。null は「歌の無い区間」の印で、
    歌のある区間の null はグルーヴの検査（`GrooveStep.check`）が差し戻すので、ここへ来たら組み立ての不具合（以前は `or 1.0`・`or 0.30`
    のような既定で黙って読んでいた）。"""
    v = getattr(g, name)
    if v is None:
        from song_fix import AssemblyError                 # noqa: E402（循環を避けてここで）
        raise AssemblyError(f"グルーヴの区間『{g.kind}』の {name} が null のまま、歌のある区間として読まれた——グルーヴの検査が返すはず")
    return float(v)


class Groove(StepStatus):
    reason: str = Field(description="このノリをどう作ったか（旋律を書く人に向けて一言）")
    feel: str = Field(description=(
        "ノリの一言。例『頭を食い気味に入り、2・4で踏み込む』『ゆったり跳ねる』"))
    sections: list[SectionGroove]


_theme = Agent(pick_model("THEME", "flash"), output_type=Theme, system_prompt=(
    prompt("song_theme") + ERA_GUIDE
))

_design = Agent(pick_model("DESIGN", "flash"), output_type=Design, system_prompt=(
    prompt("song_design") + INTRO_GUIDE
))

_chords = Agent(pick_model("CHORD", "flash"), output_type=Chords, system_prompt=(
    prompt("song_chords")
))


# ---- コードの工程の検査と直し（2026-09-28・工程が自分の検査を持つ形。`song_fix.StepFixer`）----

class ChordsPatch(PatchStatus):
    """コードの差分。直す区間だけを丸ごと書き直す。"""
    reason: str = Field(description="何をどう直したか（全体で一言）")
    sections: list[SectionChords] = Field(default_factory=list,
                                          description="直す区間だけ（区間名は設計の名前そのまま）。直さない区間は書かない。"
                                                      "抜けている区間はここに書けば足される")
    drop: list[str] = Field(default_factory=list,
                            description="**消す区間の名前**（設計に無い名前の区間など）。名前を直すときは、正しい名前で sections に書き、"
                                        "間違った名前をここに書く")

    # ★同じ区間の2つ目は、以前は前の直しへ黙って上書きされた（後が勝つ）。グルーヴの差分（`GroovePatch._unique_patch_kinds`）と同じ検査
    #   （2026-10-03・再レビュー）。区間名は前後の空白を取り除いた形で照らす（`song_checks.section_id`＝`resolve_section` と同じ決まり）
    @field_validator("sections")
    @classmethod
    def _unique_patch_kinds(cls, v):
        kinds = [section_id(p.kind) for p in v]
        if dup := sorted({k for k in kinds if kinds.count(k) > 1}):
            raise ValueError("sections に同じ区間を2回以上書いた: " + "・".join(f"『{k}』" for k in dup) + "——1つの区間は1つにまとめる")
        return v


_chords_fixer = Agent(pick_model("CHORD", "flash"), output_type=ChordsPatch, system_prompt=prompt("song_chords_fix"))


from song_fix import StepFixer, guard_patch                        # noqa: E402
import song_checks as ck                                           # noqa: E402
from song_checks import section_id                                 # noqa: E402（区間名の照らす形・2026-10-03）



class GroovePatchItem(BaseModel):
    """グルーヴの差分1区間ぶん。"""
    kind: str = Field(description="直す区間名（設計の名前そのまま）")
    # ★行数・モーラ数は「0＝変えない」（2026-10-01・判-E）。ドラムの行だけを直す差分を書けるように（以前は必須で 1 以上）
    lines: JsonInt = Field(default=0, ge=0, description="新しい行数（0 なら変えない・小節数まで）")
    moras_per_line: JsonInt = Field(default=0, ge=0, description="新しい1行のモーラ数（0 なら変えない）")
    breaks: str = Field(default="", description="区切りの記号（S/M/C を行数と同じ文字数。空なら変えない）")
    # ★2026-10-02: 行数 0 の区間の区切りの記号を所見にした（`GrooveStep.check`）ので、消す手段が要る。`breaks` の空は「変えない」なので別の欄に
    clear_breaks: JsonBool = Field(default=False, description=(
        "true なら**この区間の区切りの記号を消す**（歌の無い＝行数 0 の区間に書いてしまったとき）。breaks と一緒には書かない"))
    # ★2026-10-03（再レビュー）: 歌の無い区間に書いた行数を所見にした（`GrooveStep.check`・仕様書 §15「歌の無い区間は 0」）ので、0 に戻す手段が要る。
    #   `lines`・`moras_per_line` の 0 は「変えない」なので別の欄に（`clear_breaks` と同じ形）
    clear_lines: JsonBool = Field(default=False, description=(
        "true なら**この区間の行数とモーラ数を 0 にする**（歌の無い区間に行数を書いてしまったとき）。lines・moras_per_line と一緒には書かない"))
    drums: list[GrooveLine] | None = Field(default=None, description=(
        "この区間の**ドラムの行を全部書き直したもの**（直さない行も含めて全部。書いた並びで置き換わる）。"
        "空（null）なら変えない"))
    # ★**足す区間の密度・置き方・刻み・ベースも書ける**（2026-10-02・洗い出し）。以前は差分にこの欄が無く、抜けた区間を足すと
    #   機械が型の既定（notes_per_beat 1.0・tail_rest 0.30・drum_grid 8・bass X---X---）で黙って埋めていた（LLM は書きようがなかった）。
    #   書かなければ「変えない」。足す区間では必ず書く（2026-10-04・組N2。以前は本体の型の既定で埋めた。`GrooveStep.patch_problem`）
    notes_per_beat: JsonFloat | None = Field(default=None, gt=0, description=(
        "新しい1拍の音節の数（`SectionGroove.notes_per_beat` と同じ意味）。null なら変えない"
        "（**いまのグルーヴに無い区間を足すときは必ず書く**——歌の無い区間なら null のまま）"))
    tail_rest: JsonFloat | None = Field(default=None, ge=0.0, lt=1.0, description=(
        "新しい行末を空ける割合（`SectionGroove.tail_rest` と同じ意味）。null なら変えない（**いまのグルーヴに無い区間を足すときは必ず書く**——歌の無い区間なら null のまま）"))
    drum_grid: DrumGrid | None = Field(default=None, description=(
        "新しいドラムの刻み（8 か 16）。null なら変えない（**いまのグルーヴに無い区間を足すときは必ず書く**）。"
        "ドラムの行の文字数と合わせる（どれかの行が16分の文字数なら 16、全部8分なら 8）"))
    bass: str | None = Field(default=None, description=(
        "新しいベースのリズム（`SectionGroove.bass` と同じ書き方）。null なら変えない（**いまのグルーヴに無い区間を足すときは必ず書く**）"))
    drop: JsonBool = Field(default=False, description="true なら**この区間を消す**（設計に無い名前の区間など）。ほかの欄は書かない")
    why: str = Field(default="", description="どの所見にどう応えたか（一言）")

    # ★**当てられない組み合わせを型で差し戻す**（2026-10-02・洗い出し）。以前は `breaks` と `clear_breaks` の両方を書くと `breaks` が勝ち、
    #   `drop` と一緒に書いたほかの欄は黙って読まなかった（LLM は直したつもりで、所見が残る）
    @model_validator(mode="after")
    def _one_meaning(self):
        if self.breaks and self.clear_breaks:
            raise ValueError(f"区間『{self.kind}』: breaks と clear_breaks を一緒に書いた——書き直すなら breaks だけ、消すなら clear_breaks: true だけ")
        if self.clear_lines and (self.lines or self.moras_per_line):
            raise ValueError(f"区間『{self.kind}』: clear_lines と lines・moras_per_line を一緒に書いた——"
                             "書き直すなら lines・moras_per_line だけ、0 にするなら clear_lines: true だけ")
        if self.drop and (self.lines or self.moras_per_line or self.breaks or self.clear_breaks or self.clear_lines
                          or self.drums is not None
                          or self.notes_per_beat is not None or self.tail_rest is not None or self.drum_grid is not None
                          or self.bass is not None):
            raise ValueError(f"区間『{self.kind}』: drop: true と一緒にほかの欄を書いた——消すなら drop: true だけ（直すなら drop を書かない）")
        return self


class GroovePatch(PatchStatus):
    reason: str = Field(description="何をどう直したか（全体で一言）")
    patches: list[GroovePatchItem] = Field(default_factory=list, description=(
        "直す区間だけ。直さない区間は書かない。**1つの区間は1つだけ**（同じ区間名を2つ書かない）"))

    # ★同じ区間名の2つ目は、以前は前の直しへ黙って上書きされた（欄ごとに後が勝つ）。2026-10-02・洗い出し
    #   区間名は前後の空白を取り除いた形で照らす（`section_id`・2026-10-03 再レビュー。以前は ` verse` と `verse` を別の区間と数えた）
    @field_validator("patches")
    @classmethod
    def _unique_patch_kinds(cls, v):
        kinds = [section_id(p.kind) for p in v]
        if dup := sorted({k for k in kinds if kinds.count(k) > 1}):
            raise ValueError("patches に同じ区間を2回以上書いた: " + "・".join(f"『{k}』" for k in dup) + "——1つの区間は1つにまとめる")
        return v


_groove_fixer = Agent(pick_model("CHORD", "flash"), output_type=GroovePatch, system_prompt=prompt("song_groove_fix"))


class GrooveStep(StepFixer):
    """グルーヴ: 区間名は設計と照合（親の決まり）。工程の検査は歌のある区間の行数（あるか・1以上・小節数以下）と区切りの記号の数、
    ドラムの行の打楽器名と1小節の長さ。記法は区切りの記号（`BREAKS`）とドラムの行の打点（伴奏のドラムの行と同じ `pattern_notation(drum=True)`・2026-10-01 判-E）。
    行数 0（歌の無い区間）なのに区切りの記号が書いてあるのも所見（2026-10-02。消すのは差分の `clear_breaks`）。
    歌のある区間の1行のモーラ数 0 と、`drum_grid` とドラムの行の文字数（刻み）の食い違いも所見（2026-10-03・再レビュー 記J・記M）。
    `time_signature` はドラムの行の1小節の文字数（`grid_slots`）を決める。

    ★「1行が2小節を超えない」の検査と、直らなければ機械が行を倍にする最後の網は 2026-09-28 に外した（音楽の判断のため）。
    """
    key = "groove"
    label = "グルーヴ"
    agent = _groove_fixer
    return_hint = "**直す区間だけ**を patches に入れてください。"

    def __init__(self, sections, time_signature: str = "4/4", form: str = "song"):
        self.design = list(sections)
        self.slots = grid_slots(time_signature or "4/4")    # ドラムの行の1小節の文字数（16分なら倍）
        #: 曲の形（2026-10-03）。グルーヴの決まりはどの形でも同じ（`has_vocal`＝主旋律のある区間・ドラムの行の記法も同じ。
        #: アカペラのドラム・ベースの行は鳴らないがノリの参考として書くので、書いたものは同じ決まりで見る）。インストは所見の言い方だけ「主旋律」
        self.form = form or "song"

    def design_items(self, g):
        return [(f"グルーヴの区間『{x.kind}』", x.kind, None) for x in getattr(g, "sections", [])]

    @staticmethod
    def _drum_where(x, d) -> str:
        return f"区間『{x.kind}』のドラム {d.inst or '（打楽器名なし）'}"

    def rename_sections(self, g, mapping):
        new = g.model_copy(deep=True)
        for x in new.sections:
            x.kind = mapping.get(x.kind, x.kind)
        return new

    def written(self, g):
        # 区切りの記号（S/M/C）は記法（`BREAKS`）が見る。数（行数と同じか）は下の `check`
        yield from ((f"区間『{x.kind}』の区切り『{x.breaks}』", BREAKS, x.breaks)
                    for x in getattr(g, "sections", []) if strip_ascii_ws(x.breaks))
        # ★**ドラムの行は伴奏と同じ打点の記法**（2026-10-01・判-E 本人「A」）。以前は記法の検査が無く（記法7「検査なし」）、
        #   伴奏へ土台として見せるだけだった。途中の試聴（`song_preview`）がこの行を伴奏と同じ読み手で鳴らすので、読める形を保証する
        from song_arrange import pattern_notation           # noqa: E402（循環を避けてここで）
        nt = pattern_notation(self.slots, drum=True)        # ドラムの行の記法（休みは `-` だけ・2026-10-02 ず-9）
        yield from ((self._drum_where(x, d), nt, d.pattern)
                    for x in getattr(g, "sections", []) for d in (getattr(x, "drums", None) or []))

    def check(self, g):
        # ※「1行が2小節を超えない」は 2026-09-28 に外した（本人判断）。ゆっくりのバラードで1行を長く取るのは表現としてありうる
        #   ——音楽の判断なので形式としては直させない（2026-09-11 の事故は長い伸ばしの問題で、旋律の息継ぎの所見が見る）
        # ★**歌のある区間の行数**を見る（2026-09-29）。以前は何も見ず、歌のある区間が抜ける・行数が 0 だと旋律に枠が無く
        #   長さの検査も効かなかった。行数が小節数より多いと、1行1小節でも区間に入りきらず旋律の枠が区間より長くなった
        out: list[str] = []
        # ★**同じ区間を2回書かない**（2026-10-02・洗い出し・コードの H14 と同じ形）。区間ごとのグルーヴは区間名で引く表にするので、
        #   2つあると後のものが勝ち、先のものは黙って消えていた（親の照合 `against_design` は旋律だけ重複を見る）
        # ★区間名は**設計との照合と同じ照らし方**（`resolve_section`＝前後の空白だけ取り除く・2026-10-03 再レビュー）。以前はここだけ
        #   完全一致で引き、` verse` と書くと「歌のある区間がグルーヴに無い」と誤って言い、`verse` と ` verse` を別の区間と数えた
        names = [s.kind for s in self.design]
        key = lambda k: resolve_section(k, names) or section_id(k)          # noqa: E731
        kinds = [key(x.kind) for x in getattr(g, "sections", [])]
        out += [f"区間『{k}』のグルーヴを{kinds.count(k)}回書いている——1つの区間は1回だけ書く（2つあると後のものだけが使われる）。"
                "直しでこの区間を書くと、1つ目を直して2つ目以降を消す（2つ目の値を使うなら、その値を直しに書く）"
                for k in dict.fromkeys(kinds) if kinds.count(k) > 1]
        have = {key(x.kind): x for x in getattr(g, "sections", [])}
        sung = "主旋律のある区間" if self.form == "instrumental" else "歌のある区間"   # 言い方だけ（2026-10-03）
        for s in self.design:
            if not s.has_vocal:
                continue
            x = have.get(s.kind)
            if x is None:
                out.append(f"{sung}『{s.kind}』がグルーヴに無い——旋律の行の枠が決まらない（区間名・行数・モーラ数を書く）")
            elif not int(x.lines or 0):
                out.append(f"{sung}『{s.kind}』の行数が 0——{sung}は1行以上")
            elif int(x.lines) > int(s.bars):
                out.append(f"区間『{s.kind}』は{s.bars}小節なのに{x.lines}行——1行は最低1小節なので、行数は小節数以下にする")
            # ★**歌のある区間の1行のモーラ数は 1 以上**（2026-10-03・再レビュー 記J・仕様書 §15）。以前は 0 が通り、
            #   旋律・詞の依頼文が「0モーラ」になった（直しの差分の 0 は「変えない」なので、1 以上を書けば直る）
            if x is not None and not int(x.moras_per_line or 0):
                out.append(f"{sung}『{s.kind}』の1行のモーラ数が 0——{sung}は 1 以上（目安 6〜12。"
                           "旋律と詞がこの数で1行の言葉の量を決める）")
            # ★**歌のある区間の密度・置き方は数で書く**（2026-10-04・組N2）。null は「歌の無い区間」の印で、以前の既定
            #   （1.0・0.30）で黙って埋めない。直しの差分の null は「変えない」なので、数を書けば直る
            if x is not None:
                miss = [f for f in ("notes_per_beat", "tail_rest") if getattr(x, f) is None]
                if miss:
                    out.append(f"{sung}『{s.kind}』の {'・'.join(miss)} が null——null は歌の無い区間の印。{sung}は数を書く"
                               "（notes_per_beat＝1拍の音節の数・目安 BPM95 なら 1.1／BPM120 なら 0.85。"
                               "tail_rest＝行末を空ける割合・0.30 がふつう、0.0 は休まない）")
            # ★**区切りの記号の数**（2026-09-30・本人了承）。旋律・詞・息継ぎの所見が行ごとに読むのに、誰も見ていなかった
            #   （足りない行は黙って M 扱い・空なら全行 M）
            if x is not None and int(x.lines or 0) and len(strip_ascii_ws(x.breaks)) != int(x.lines):
                out.append(f"区間『{s.kind}』の区切りの記号『{x.breaks}』が{len(strip_ascii_ws(x.breaks))}文字（行数は{x.lines}）——"
                           "S/M/C を1行1文字で、行数と同じ数だけ書く（旋律と詞がこれを行ごとに読む）")
        # ★**行数 0 なのに区切りの記号が書いてある区間**（2026-10-02・本人「2通りに書けるものは良くない」のついで）。
        #   以前は上の数の検査が行数 1 以上の区間しか見ず、歌の無い区間の `breaks` は黙って通っていた（誰も読まない＝書いた意図が届かない）。
        #   歌のある区間で行数 0 のものは上の「行数が 0」が先に言う（行数を書けば、数の検査がこの区切りを見る）
        vocal = {s.kind for s in self.design if s.has_vocal}
        quiet = {s.kind for s in self.design if not s.has_vocal}
        for x in getattr(g, "sections", []):
            # ★**歌の無い区間の行数は 0**（2026-10-03・再レビュー・仕様書 §15 の `lines`「歌の無い区間は 0」・型の説明も同じ）。
            #   以前は所見にならず、書いた行数は誰も読まなかった（書いた意図が届かない）。0 に戻すのは差分の `clear_lines`
            if int(x.lines or 0) and key(x.kind) in quiet:
                what = "主旋律の無い区間" if self.form == "instrumental" else "歌の無い区間"
                out.append(f"区間『{x.kind}』は{what}なのに行数が{x.lines}——{what}の行数は 0"
                           "（直しでは clear_lines: true で行数とモーラ数を 0 に。区切りの記号も書いてあれば clear_breaks: true。"
                           f"{'主旋律' if self.form == 'instrumental' else '歌'}を入れる区間かどうかは設計が決める）")
            if not int(x.lines or 0) and strip_ascii_ws(x.breaks) and key(x.kind) not in vocal:
                if self.form == "instrumental":
                    out.append(f"区間『{x.kind}』は行数 0（主旋律の無い区間）なのに区切りの記号『{x.breaks}』が書いてある——"
                               "主旋律の無い区間の区切りは空にする（直しでは clear_breaks: true。主旋律を入れる区間なら行数を書く）")
                else:
                    out.append(f"区間『{x.kind}』は行数 0（歌の無い区間）なのに区切りの記号『{x.breaks}』が書いてある——"
                               "歌の無い区間の区切りは空にする（直しでは clear_breaks: true。歌を入れる区間なら行数を書く）")
        # ★**ドラムの行の打楽器名と1小節の長さ**（2026-10-01・判-E）。伴奏のドラムの行と同じ検査（`timbre_map.drum_canon`・
        #   `song_arrange._length_issue`）。編成（キット）はまだ決まっていないので、キットにあるかは見ない（伴奏の工程が見る）
        from song_arrange import _clean_grid, _length_issue  # noqa: E402（循環を避けてここで）
        from timbre_map import DRUM_NAMES, drum_name_problem, hat_line_problem   # noqa: E402
        for x in getattr(g, "sections", []):
            for d in getattr(x, "drums", None) or []:
                where = self._drum_where(x, d)
                # ★2026-10-02: 名前は GM の正式な名前だけ（別名は読まない）。近い GM の名前を添える・closed_hi_hat の行の `o` も所見
                if why := drum_name_problem(d.inst or ""):
                    out.append(f"{where}: {why}（書ける名前: {' / '.join(DRUM_NAMES)}）")
                elif hat := hat_line_problem(d.inst or "", d.pattern or ""):
                    out.append(f"{where}: {hat}")
                out += [i.finding() for i in _length_issue(where, d.pattern, self.slots, drum=True)[0]]
            # ★**刻みは行の文字数だけで決まる**（2026-10-03・再レビュー 記M・仕様書 §15）。`drum_grid` は伴奏の依頼文の案内にだけ使うので、
            #   行と食い違えば所見にする（以前は刻みを2か所に書いて照らさず、伴奏へ「16分」と言いながら土台の行は8分のことがあった）。
            #   どれかの行が 2N 文字なら 16・全部 N 文字なら 8（行ごとに混ぜてよい・§8）。長さが N でも 2N でもない行は長さの所見が言うので数えない
            lens = [len(_clean_grid(d.pattern or "")) for d in getattr(x, "drums", None) or []]
            lens = [n for n in lens if n in (self.slots, self.slots * 2)]
            if lens:
                grid = 16 if self.slots * 2 in lens else 8
                if int(x.drum_grid) != grid:
                    out.append(f"区間『{x.kind}』の drum_grid が {x.drum_grid} なのに、ドラムの行は"
                               + (f"{self.slots * 2}文字（16分）の行がある" if grid == 16 else f"全部{self.slots}文字（8分）")
                               + f"——刻みは行の文字数で決まる。drum_grid を {grid} に直す（"
                               + (f"8分で刻むなら行を{self.slots}文字に書き直す）" if grid == 16
                                  else f"16分で刻むなら行を{self.slots * 2}文字に書き直す）"))
        return out

    def show(self, g):
        return (f"【ドラムの行の1小節】{self.slots}文字（16分なら{self.slots * 2}文字）\n\n"
                "【いまのグルーヴ】\n" + "\n".join(
                    f"- {x.kind}: {x.lines}行 × {x.moras_per_line}モーラ 区切り『{x.breaks}』"
                    + "".join(f"\n    ドラム {d.inst}: {d.pattern}" for d in (getattr(x, "drums", None) or []))
                    for x in g.sections))

    def patch_problem(self, g, patch) -> str:
        """**いまのグルーヴに無い区間を足す直しは、区間の欄を全部書く**（2026-10-04・組N2）。以前は書かなかった欄を本体の型の既定
        （bass `X---X---`・drum_grid 8・notes_per_beat 1.0・tail_rest 0.30・行数 0）で黙って埋めていた。差分の空は「いまのまま」だが、
        足す区間には「いまの値」が無い。歌のある区間（設計の `has_vocal`）は密度・置き方・行数・モーラ数・区切りも要る。"""
        have = {section_id(x.kind) for x in getattr(g, "sections", [])}
        sung = {section_id(s.kind) for s in self.design if s.has_vocal}
        out = []
        for p in getattr(patch, "patches", None) or []:
            k = section_id(p.kind)
            if getattr(p, "drop", False) or k in have:
                continue
            miss = [f for f, v in (("bass", p.bass), ("drum_grid", p.drum_grid)) if v is None]
            if k in sung:
                miss += [f for f, v in (("lines", p.lines or None), ("moras_per_line", p.moras_per_line or None),
                                        ("breaks", p.breaks or None), ("notes_per_beat", p.notes_per_beat),
                                        ("tail_rest", p.tail_rest)) if v is None]
            if miss:
                out.append(f"いまのグルーヴに無い区間『{p.kind}』を足すなら {'・'.join(miss)} も書く——足す区間には「いまの値」が無いので、"
                           "機械は既定で埋めない（drums も書くか、書かずに空のドラムで足すかを決める）")
        return "／".join(out)

    def apply(self, g, patch):
        new = g.model_copy(deep=True)
        done, idle = [], []
        # ★区間名は前後の空白を取り除いた形どうしで照らす（`section_id`＝`resolve_section` と同じ決まり・2026-10-03 再レビュー）。
        #   以前は完全一致で引き、差分の ` verse` が既定値の新しい区間として足され、`canonize` のあと同じ区間が2つになって
        #   後のもの（既定値）が勝ち、密度とドラムが変わった
        same = lambda x, p: section_id(x.kind) == section_id(p.kind)          # noqa: E731
        for p in getattr(patch, "patches", None) or []:
            if getattr(p, "drop", False):
                if any(same(x, p) for x in new.sections):
                    new.sections = [x for x in new.sections if not same(x, p)]
                    done.append(f"{p.kind}を消した")
                else:                                       # 無い区間の drop（2026-10-02 から記録に書く。以前は黙って何もしなかった）
                    idle.append(f"{p.kind}（drop したがグルーヴに無い区間）")
                continue
            hit = next((x for x in new.sections if same(x, p)), None)
            if hit is None:
                # ★**抜けた区間は足す**（2026-09-29）。以前は既にある区間しか直せず、抜けた区間の所見が直らなかった。
                #   足した区間のドラムは空（伴奏の工程はグルーヴのドラムを土台として読むだけ）。
                #   密度・置き方・刻み・ベースは差分に書いたもの（2026-10-02 から書ける）。2026-10-04（組N2）から既定で埋めない——
                #   書いていない欄は `patch_problem` が直しの LLM へ差し戻す。ここで型が通らなければ組み立ての不具合（検証を通らずに来た）
                hit = SectionGroove(kind=p.kind, drums=[], bass=p.bass, drum_grid=p.drum_grid, lines=p.lines,
                                    moras_per_line=p.moras_per_line, breaks=p.breaks,
                                    notes_per_beat=p.notes_per_beat, tail_rest=p.tail_rest)
                new.sections.append(hit)
            elif not (p.lines or p.moras_per_line or p.breaks or p.clear_breaks or p.clear_lines or p.drums is not None
                      or p.notes_per_beat is not None or p.tail_rest is not None or p.drum_grid is not None or p.bass is not None):
                idle.append(f"{p.kind}（直す欄が無い）")             # 何も書いていない直し（2026-10-02 から「直した」と数えない）
                continue
            if any(same(x, p) and x is not hit for x in new.sections):
                # 同じ区間が2つ以上（検査の所見）: 直しでその区間を書いたら1つ目を直し、2つ目以降を消す（所見の文で知らせている・2026-10-02）
                new.sections = [x for x in new.sections if not same(x, p) or x is hit]
                done.append(f"{p.kind}の2つ目以降を消した")
            if getattr(p, "clear_lines", False):            # 歌の無い区間の行数を 0 に（2026-10-03・書いた差分を当てるだけ）
                hit.lines, hit.moras_per_line = 0, 0
            if p.notes_per_beat is not None:
                hit.notes_per_beat = p.notes_per_beat
            if p.tail_rest is not None:
                hit.tail_rest = p.tail_rest
            if p.drum_grid is not None:
                hit.drum_grid = p.drum_grid
            if p.bass is not None:
                hit.bass = p.bass
            if p.lines:
                hit.lines = p.lines
            if p.moras_per_line:
                hit.moras_per_line = p.moras_per_line
            if p.breaks:
                hit.breaks = p.breaks
            elif getattr(p, "clear_breaks", False):         # 書いた差分を当てるだけ（2026-10-02）
                hit.breaks = ""
            if p.drums is not None:                         # ドラムの行は区間ごとに丸ごと差し替える（判-E）
                hit.drums = [d.model_copy() for d in p.drums]
            done.append(p.kind)
        note = f"{len(done)}区間を直した（{'・'.join(done)}）" if done else "直しなし"
        note += f"／当てる所が無かった: {'・'.join(idle)}" if idle else ""
        if getattr(patch, "trouble", ""):                  # 直さなかった理由を直しの記録と止めたときの文へ（旋律・設計と同じ・2026-10-03 再レビュー）
            note += f"／直さなかった理由: {patch.trouble}"
        return new, note


class ChordsStep(StepFixer):
    """コード: 区間名と小節数は設計と照合（親の決まり）、読めないコード名は記法（`written`＝各小節・`CHORD_BAR`）。
    工程固有の検査は無い（「1小節に3つ以上」は 2026-09-28 に外した）。

    以前は区間名が合わないと区間ごと和音が鳴らず、小節数が違うと機械が繰り返し／捨て、読めないコード名は
    C 以外の調で作曲を落とした——どれも黙っていた（2026-09-27 監査）。
    """
    key = "chords"
    label = "コード"
    agent = _chords_fixer
    need_all_sections = True
    return_hint = "**直す区間だけ**を sections に入れてください。"

    def __init__(self, sections):
        self.design = list(sections)

    def design_items(self, c):
        return [(f"コードの区間『{x.kind}』", x.kind, len(x.chords_per_bar)) for x in c.sections]

    def rename_sections(self, c, mapping):
        new = c.model_copy(deep=True)
        for x in new.sections:
            x.kind = mapping.get(x.kind, x.kind)
        return new

    def written(self, c):
        return ((f"{x.kind} の{i}小節目『{cell}』", CHORD_BAR, cell)
                for x in c.sections for i, cell in enumerate(x.chords_per_bar, 1))

    def check(self, c):
        # 読めないコード名は記法（`written`）が見る。区間名と小節数は親の決まり（設計との照合）
        # ※「1小節に3つ以上」の所見は 2026-09-28 に外した: 組み立てが書いた数で小節を均等に分けるようになった
        # ★**同じ区間を2回書かない**（2026-10-02・洗い出しの H14）。区間ごとのコードは区間名で引く表（`song_outline.run_chords` の
        #   `ch_map`・`Song.chords_of`）にするので、2つあると後のものが勝ち、先のものは黙って消えていた。区間名の照らし方は
        #   設計との照合と同じ（`resolve_section`）。親の照合（`against_design`）は旋律だけ並びと重複を見る
        names = [s.kind for s in self.design]
        seen: dict[str, int] = {}
        for x in c.sections:
            k = resolve_section(x.kind, names) or x.kind
            seen[k] = seen.get(k, 0) + 1
        # 差分の `drop` は同じ名前の区間をすべて消し、`sections` は同じ名前の最初の1つだけを置き換える（`apply`）。
        #   1回の差分で直せるように「drop で消して、残すコードを sections に1つ書く」と返す
        return [f"区間『{k}』のコードを{n}回書いている——1つの区間は1回だけ書く（2つあると後のものだけが使われる）。"
                f"drop に『{k}』を書く（同じ名前の区間がすべて消える）と同時に、残すコードを sections に『{k}』として1つだけ書く"
                for k, n in seen.items() if n > 1]

    def show(self, c):
        want = "\n".join(f"- {s.kind}: {s.bars}小節" for s in self.design)
        rows = "\n".join(f"- {x.kind}: {' / '.join(x.chords_per_bar)}" for x in c.sections)
        return "【設計の区間と小節数】\n" + want + "\n\n【いまのコード】\n" + rows

    def apply(self, c, patch):
        new = c.model_copy(deep=True)
        done, idle = [], []
        # ★区間名は前後の空白を取り除いた形どうしで照らす（`section_id`＝`resolve_section` と同じ決まり・2026-10-03 再レビュー）。
        #   2026-10-02（H14）に書いた名前の側だけ `resolve_section` で照らしたが、いまのコードの側に空白が付いていると当たらず、
        #   同じ区間を新しく足していた（後のものが勝つ）
        # ★**区間を消せる**（2026-09-29）。以前は足すか上書きだけで、設計に無い名前の区間の所見が直らず残り続けた
        for k in getattr(patch, "drop", None) or []:
            if any(section_id(x.kind) == section_id(k) for x in new.sections):
                new.sections = [x for x in new.sections if section_id(x.kind) != section_id(k)]
                done.append(f"{k}を消した")
            else:                                           # 無い区間の drop は記録に書く（2026-10-03・グルーヴと同じ。以前は黙って何もしなかった）
                idle.append(f"{k}（drop したがコードに無い区間）")
        for p in getattr(patch, "sections", None) or []:
            hit = next((x for x in new.sections if section_id(x.kind) == section_id(p.kind)), None)
            if hit is not None:
                hit.chords_per_bar = list(p.chords_per_bar)
            else:
                new.sections.append(SectionChords(kind=p.kind, chords_per_bar=list(p.chords_per_bar)))
            done.append(p.kind)
        note = f"{len(done)}区間を直した（{'・'.join(done)}）" if done else "直しなし"
        note += f"／当てる所が無かった: {'・'.join(idle)}" if idle else ""
        if getattr(patch, "trouble", ""):                  # 直さなかった理由を直しの記録と止めたときの文へ（2026-10-03 再レビュー）
            note += f"／直さなかった理由: {patch.trouble}"
        return new, note


_groove = Agent(pick_model("CHORD", "flash"), output_type=Groove, system_prompt=(
    prompt("song_groove")
))


# ---- 詞先（旧）の工程は 2026-09-02 に削除 ------------------------------------
# `Lyrics` / `Melody` / `SectionLyrics` / `SectionMelody` の型、`_lyrics` / `_melody` の
# エージェント、通しの `compose_v2()`。
# **詞を先に書いて各モーラに1音ずつ当てる**やり方で、実測すると
#   ・全フレーズが「8分を並べて最後に伸ばす」の**一択**
#   ・フレーズ開始が4拍ごとに固定、休符が全部 0.5拍
#   ・別のフレーズ3つが**音高まで完全に同一**
# となった（『Morning Step!』）。**旋律が詞の形に完全に従属**するためで、
# 配分の数値目標は満たしていた——**数値を満たしても音楽にならない**という実例。
# いまは曲先（`song_compose`）が、旋律を先に書いてから枠に合う詞を当てる。
# ★**テーマ・設計・コード（`_theme` / `_design` / `_chords`）は曲先でも使う**ので残す。


# ---- 設計の工程の検査と直し（2026-10-03・曲の形・`docs/plan/song_acappella_instrumental.md` §3・仕様書 §15-4）----------------
#
# ★設計の形式は型（区間名の重複・転調の区間名・調名）が見るが、**曲の形で決まる決まり**は型に書けない（型は形を知らない）。
#   形は工程の引数で渡す（グルーヴの拍子 `GrooveStep(sections, time_signature)` と同じ渡し方）。
#   終わり方 `vocal`（伴奏が先に止み歌だけ残る）は、アカペラ（伴奏が無い）・インスト（歌が無い）では意味が無い＝答えが1つに決まる形式。
#   機械は終わり方を選び直さず、所見で設計の LLM に差分で直させる（直らなければ設計の工程で止まる）。
#   形を変えたときに残す設計へ当てる検査（`song_form_checks.form_findings`）も同じ `check` を呼ぶ。

class DesignPatch(PatchStatus):
    """設計の直し（差分）。終わり方だけ（形の所見で直すのはここだけ）。"""
    reason: str = Field(description="何をどう直したか（一言）")
    ending: Ending | None = Field(default=None, description="新しい終わり方（fade / hit / vocal / loop のどれか）。直さないなら null")
    ending_reason: str = Field(default="", description="新しい終わり方を選んだ理由（曲の言いたいこととどう繋がるか）。直さないなら空")


# ★当てられない差分（理由の無い終わり方）は差し戻す（2026-10-03・再レビュー・`DesignStep.patch_problem`）
_design_fixer = guard_patch(Agent(pick_model("DESIGN", "flash"), output_type=DesignPatch, system_prompt=prompt("song_design_fix")))


def ending_problems(design, form: str = "song") -> list[str]:
    """**終わり方が曲の形と合うか**の所見（無ければ空）。`vocal` は歌ものだけ。"""
    if form == "song" or design.ending != "vocal":
        return []
    what = {"acappella": "アカペラは伴奏が無い（「伴奏が先に止み、歌だけ残る」が起きない）",
            "instrumental": "インストは歌が無い（「歌だけ残して消える」が起きない）"}.get(form, f"形『{form}』では使えない")
    return [f"終わり方が vocal——{what}。fade / hit / loop のどれかに直す（アウトロの小節数もその終わり方に合っているか見る）"]


class DesignStep(StepFixer):
    """**設計の、曲の形で決まる決まり**の検査と差分の直し（2026-10-03）。いまは終わり方（`ending_problems`）だけ。

    呼ぶ場所は設計の直後（`song_outline.run_design`）で `DesignStep(song.form).run(design, song)` → (設計, 残った所見＝いつも空)。
    歌もの（`form="song"`）では所見が出ない（以前と同じ）。保存から戻した設計は直させない（親の `run`）。
    """
    key = "design"
    label = "設計"
    agent = _design_fixer
    return_hint = "**直すのは所見の出た欄（終わり方 ending とその理由）だけ**です。テンポ・キー・構成・方針は変えません。"

    def __init__(self, form: str = "song"):
        self.form = form or "song"

    def check(self, design) -> list[str]:
        return ending_problems(design, self.form)

    def show(self, design) -> str:
        rows = "\n".join(f"  - {s.kind}: {s.bars}小節" + ("" if s.has_vocal else "（主旋律なし）")
                         for s in design.sections)
        return (f"【いまの設計の終わり方】ending: {design.ending}（理由: {design.ending_reason or 'なし'}）\n"
                f"【構成】\n{rows}\n【全体方針】{design.concept}")

    def patch_problem(self, design, patch: DesignPatch) -> str:
        """**終わり方と理由は一緒に直す**（2026-10-03・再レビュー）。以前は終わり方だけ変えて理由を空にすると、前の終わり方の理由が
        そのまま残った（後の工程と点検役は、新しい終わり方に古い理由が付いたまま読む）。型の説明どおり、終わり方を直すなら理由を書き、
        直さないなら理由も空（差分の検証 `song_fix.guard_patch` が頼む間だけ読む）。"""
        ending, why = getattr(patch, "ending", None), (getattr(patch, "ending_reason", "") or "").strip()
        if ending is not None and not why:
            return ("ending を直すなら ending_reason（新しい終わり方を選んだ理由）も書く——書かないと前の終わり方の理由"
                    f"（『{(design.ending_reason or 'なし')[:60]}』）が残る")
        if ending is None and why:
            return "ending_reason だけ書いた——終わり方を直さないなら ending_reason も空（直すなら ending も書く）"
        return ""

    def apply(self, design, patch: DesignPatch) -> tuple:
        trouble = f"／直さなかった理由: {patch.trouble}" if getattr(patch, "trouble", "") else ""
        # ★当てられない書き方（理由の無い終わり方・終わり方の無い理由）は差分の検証が差し戻すので来ない。型を通らずに来たら
        #   当てずに、当てなかったことを記録に書く（2026-10-03・ほかの工程の差分と同じ）
        if why := self.patch_problem(design, patch):
            return design, f"当てなかった: {why}" + trouble
        if getattr(patch, "ending", None) is None:
            return design, "直しなし" + trouble
        upd = {"ending": patch.ending, "ending_reason": patch.ending_reason}
        return design.model_copy(update=upd), f"終わり方を {design.ending} → {patch.ending}" + trouble
