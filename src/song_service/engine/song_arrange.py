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

"""伴奏を **LLM に書かせる**（現行はテンプレート3種の固定パターン）。

## 現行の何が問題か

`_style_arrangement(style)` が band / pop / music_box の3つから `ArrangementSpec` を返すだけで、

- 曲ごとに編曲が変わらない（同じスタイルなら伴奏は同一）
- **1小節ぶんのパターンを全小節に繰り返す**（`PartSpec.steps` は「1小節パターン」）。
  `bar_filter` は all / latter / first の3値しかないので、**Aメロとサビで伴奏が変わらない**
- ドラムは8ビート固定＋N小節ごとに同じフィル

## この試作

LLM に **1小節を8分音符8個のグリッド**（`"x-x-x-x-"`）で書かせ、コードで `StepEvent` /
`DrumHit` へ展開する。生の `StepEvent` を書かせないのは、ネストが深くて LLM が壊しやすいのと、
**音楽的な言葉（どこで鳴らすか）に集中させたい**ため。

セクション対応には小節範囲（`"9-16"`）を解する判定が要る。
これは 2026-09-01 に `song_audio._bar_match` へ統合済み（**判定はそこ1つだけ**）。
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, NamedTuple

from pydantic import BaseModel, Field

from pydantic_ai import Agent

import song_audio                                          # **どの小節で鳴るかの判定はここだけ**
from era_styles import DYNAMICS_GUIDE                       # メリハリの目安（強さ→音の差）
from song_audio import ARP_DOWN, ARP_UP, INVERSION         # `tones=inversion`・分散和音の印（伴奏-1）
from song_llm import PITCH, pick_model
from song_llm import run_sync as _llm_run
from song_checks import JsonBool, JsonFloat, JsonInt  # noqa: E402（LLM が書く数と真偽の欄の型・2026-10-02 判断5）
from song_checks import edge_ws_problem, strip_ascii_ws, trim_edge_ws  # noqa: E402（半角の記法の前後の空白・§0-6・2026-10-03）
from song_usage import record as _rec_usage         # 費用の集計
from song_model import StepRecord, PatchStatus, StepStatus
from song_schema import ArrangementSpec, DrumHit, PartSpec, StepEvent, Tone
from timbre_map import DRUM_NAMES, drum_canon, drum_name_problem, gm_name, hat_line_problem, kit_drum_menu, kit_drums
from song_prompts import prompt                          # プロンプトの正は prompts/*.md

# ---- どの小節で鳴るかの判定 ---------------------------------------------------
# ★**判定は `song_audio._bar_match` が1つだけ持つ**（2026-09-01 に本番へ統合）。
# かつては編曲・MIDI・本番の3か所に別実装があって**食い違っていた**——
# 例えば単独の小節番号 `"7"` を、編曲側は範囲と見なさず本番へ流し、本番は未知の値を
# 「全小節で鳴る」と解釈していた。つまり `"7"` と書くと**全小節で鳴った**。
# テスト環境ではここで本番の関数を差し替えて回避していたが、**本体を直したので不要**。
_bar_match_v2 = song_audio._bar_match


# ---- LLM に書かせる中間表現 ---------------------------------------------------

#: ★**ドラムは「ドラムタブ」で書かせる**（2026-09-21）。
#: それまで独自の凡例（`X`/`x`/`.` の強弱だけ）で、**`o` を一度も使っていなかった**。
#: 世界標準のドラムタブは **`x`＝シンバル・ハイハット／`o`＝ドラムの打点**（キック・スネア・タム）、
#: **大文字がアクセント**、**`-`＝休み**、**`|`＝小節線**。
#: 名乗らずに記号を一から定義していたので、**LLM が持っているドラムタブの知識が働かなかった**
#: （旋律を ABC へ移したのと同じ理由）。
#: ※ドラムタブは「書き手ごとに記号が違うので**必ず凡例を付ける**」のが作法なので、凡例は残す。
#
# 楽器ごとのベロシティ基準差（キックは前に出し、金物は奥へ引く）。書かれていない楽器は 0（書かれたとおりに鳴らす）。
# ★**凡例に書く**（2026-09-27・目録の 打-4）。以前は LLM が知らず、`X` のハイハットがキックの `o` より弱く鳴ることを
#   前提にできなかった。ミキサーの既定値として残し、値は下の `_DRUM_LEVELS` でここから凡例へ出す（書き写さない）。
# ★2026-10-02: 鍵は GM の正式な名前（`timbre_map.GM_DRUM`。以前の短い名前 kick・snare・hihat … から写した・値は同じ）
_DRUM_BASE = {"bass_drum_1": 8, "acoustic_snare": 4, "hand_clap": 2, "side_stick": -6,
              "closed_hi_hat": -14, "open_hi_hat": -10, "pedal_hi_hat": -16,
              "crash_cymbal_1": -4, "ride_cymbal_1": -10, "ride_bell": -8,
              "maracas": -18, "tambourine": -12, "open_triangle": -14, "hi_wood_block": -8,
              # 2026-10-02: GM の 35〜81 を全部名前にしたとき、上にある楽器の2つ目・同じ類（シンバル・トライアングル・ウッドブロック）に同じ差
              "acoustic_bass_drum": 8, "electric_snare": 4, "crash_cymbal_2": -4, "chinese_cymbal": -4,
              "splash_cymbal": -4, "ride_cymbal_2": -10, "mute_triangle": -14, "low_wood_block": -8}
_DRUM_LEVELS = (
    "★**打楽器ごとに、書いた強さへ機械が一定の差を足します**（ミキサーの既定値。キックは前・金物は奥）: "
    + " / ".join(f"{k} {v:+d}" for k, v in _DRUM_BASE.items())
    + "（強さの目安 `X`/`O`=110・`x`/`o`=88・`.`=62 に足す。ここに無い楽器は 0）。"
    "ハイハットを前に出したいときは `X` で書く")

_DRUM_TAB = (
    "**ドラムタブ（drum tab）**で書く。1小節を**8分音符8個**で表す"
    "（**倍の16文字を書けば16分**。行ごとに変えてよい——ハイハットだけ16分、という形は普通）。\n"
    "記号（ドラムタブの標準）: "
    "**`o`＝ドラムを叩く**（bass_drum_1 / acoustic_snare / low_tom ほか）／"
    "**`x`＝シンバル類を叩く**（closed_hi_hat / crash_cymbal_1 / ride_cymbal_1 ほか）／"
    "**大文字 `O` `X`＝アクセント（強く）**／**`.`＝ゴーストノート（とても弱く）**／"
    "**`-`＝休み（鳴らさない）**。"
    # ★2026-10-02（打楽器名を GM の正式な名前に一本化）: 判-J（2026-10-01）では hihat の行の `o` を開いたハイハットで鳴らしていた。
    #   名前が closed_hi_hat／open_hi_hat に分かれたので、開くのは open_hi_hat の行だけ（書き方は1通り）。closed_hi_hat の行の `o` は所見
    "**開いたハイハットは `open_hi_hat` の行に書く**（`closed_hi_hat` の行の打点は `X` `x` `.` だけ。`o` を書くと検査で所見として返します）。"
    "4拍裏で開くなら closed_hi_hat `x-x-x-x-` ＋ open_hi_hat `-------x` のように行を分ける。\n"
    "★**書くのは1小節ぶんだけ**（そのパターンが小節ごとに繰り返されます）。"
    # ★2026-10-02（本人「おすすめで」・空白は記法全体で「区切り」の意味）: 以前は途中の空白を無視して読んでいた。いまは所見
    "**途中に空白を入れない**（`o--- o---` ではなく `o---o---`。書くと検査で所見として返します）。"
    # ★2026-10-01（記法の仕様書 §8 と突き合わせ）: 「直さずに残ると最初の `|` までの1小節だけが鳴る」と書いていたが、
    #   2026-09-30 から直らなければ工程で止まる（`StepUnfixed`）。読み方（最初の `|` までの1小節だけ）と止まることを書く
    "**`|`（小節線）で区切って2小節以上を書かないでください**——読むのは**最初の `|` までの1小節だけ**なので、"
    "検査で所見として返します（直らずに残ると曲づくりはこの工程で止まります）。"
    "`|` を書かずに8文字の小節を2つ続けると、16文字＝16分の1小節として読むので**倍の速さで**鳴ります。"
    "小節ごとに変えたいときは、パターンを分けて `at_bars` で小節を指定します。\n"
    # ★スネアの例を直した（2026-09-30・記-2）。以前は `---o---o` を「2・4拍」と説明していたが、8文字では 4・8 番目＝
    #   2拍裏・4拍裏（組み立ては 1.5拍・3.5拍に置く）。2・4拍は 8文字の 3・7 番目、16文字の 5・13 番目（グルーヴ側と同じ数え方）
    "例（4/4 の8ビート）: bass_drum_1 `o---o---` ／ acoustic_snare `--o---o-`（8文字の 3・7 番目＝2・4拍＝バックビート）／ "
    "closed_hi_hat `x-x-x-x-` ／ 16分で刻むなら closed_hi_hat `xxxxxxxxxxxxxxxx`・acoustic_snare `----o-------o---`"
    "（16文字の 5・13 番目＝2・4拍）。"
    "3拍子・6/8 の拍の頭は依頼文の拍子の説明で数える。\n"
    # ★2026-10-02（ず-9・本人「2通りに書けるものは良くない」「標準に寄せて1つにそろえる」）: 以前は `_` も休みとして読んでいた
    #   （記録の伴奏のドラムに242行ある）。ドラムタブの標準どおり休みは `-` だけにし、`_` は記法の所見にした
    "★**打楽器は伸びない**ので、伸ばす記号はここには無い。**鳴らさない所は必ず `-`** で書く"
    "（`_` は音程パートの「そこで切る」の字で、ドラムの行では書けない＝検査で所見として返します。`--o-__o_` ではなく `--o---o-`）。\n"
    + _DRUM_LEVELS)

_GRID = ("1小節を8分音符8個で表す。**倍の16文字を書けば16分グリッド**になる"
         "（速い曲の刻みはこちら。行ごとに変えてよい）。**強弱を文字で書き分ける**: "
         "X=強く（アクセント） x=普通 .=弱く **-=前の音を伸ばす** **_=そこで音を切る（無音）**"
         # ★2026-10-02（ず-9・本人「2通りに書けるものは良くない」）: 以前は `O` `o` も `X` `x` と同じ強さで読んでいた。いまは所見
         "（打点は `X` `x` `.` だけ。`O` `o` はドラムの行だけの字で、ここでは書けない——検査で所見として返します）。\n"
         "例 'X-x-X-x-'（1・3拍にアクセント）／'Xx.xXx.x'（8分連打で強弱をつける）／"
         "'x-------'（**1小節伸ばす＝白玉**）／**'x_______'（頭で1発だけ鳴らして、あとは無音）**。\n"
         "★**`-` と `_` の違いが「ため」を作ります。** `-` は音が鳴り続け、`_` は止まります。\n"
         "  サビ前で溜めたいなら **'x_______'**——`'x-------'` だと1小節鳴りっぱなしで止まりません。\n"
         "  'X-x-____'（2拍鳴らして、後半は無音）のように**途中から切る**こともできます。\n"
         "  **'________'（打点なし）はその小節を鳴らさない**という書き方です（`at_bars` で指した小節だけ止める、など）。\n"
         "★**1小節にコードが2つ以上ある小節**（コード表の `Am F`）では、小節を**書いた数で均等に分けた区切り**ごとに"
         "和音が替わります。パターンは**小節の頭から**数えて置き、打点はその時点の和音で鳴り、`-` で伸ばした音は和音が"
         "変わる所で新しい和音に弾き直します。**`_` で止めると、その先の和音は鳴りません**（ためならそれでよい）。\n"
         "★**コードが置かれていない小節**（コード表に書かれていない小節）では、音程パートは鳴りません"
         "（`pitches`・リフも含む。ドラムは鳴ります）。`N.C.`（ノーコード）の所では、コードから選ぶ音（`tones`）は鳴らず、"
         "`pitches`・リフの音だけが鳴ります。\n"
         "**全部 x にしない**——プロの伴奏はベロシティの幅が105（ほぼ全域）で動いていて、"
         "強弱の平坦さが打ち込みっぽさの主因になります\n"
         # ★目録の 伴奏-8（2026-09-27）。以前は1小節を超えた分を黙って捨てていた
         "★**書くのは1小節ぶんだけ**（8文字か16文字ちょうど。3拍子なら6文字か12文字。`|`（小節線）は書かない）。"
         # ★2026-10-02（本人「おすすめで」・空白は記法全体で「区切り」の意味）: 以前は途中の空白を無視して読んでいた。いまは所見
         "**途中に空白を入れない**（`x-x- x-x-` ではなく `x-x-x-x-`）。"
         # ★2026-10-01: 「足りない所は無音・ほかの文字も無音」と組み立ての読み方を書いていたが、2026-09-30 から直らなければ工程で止まる
         "1小節を超えて書いた分・足りない分・`|` で区切った2小節目以降・`X x . - _` 以外の文字（ドラムの `o` `O` も・途中の空白も）は検査で所見として返します"
         "（直らずに残ると曲づくりはこの工程で止まります）。小節ごとに変えたいならパターンを分けて `at_bars` で指す")


_AT_BARS = (
    "**区間の中のどの小節に当てるか**（**必ず書く**。区間の全小節なら `all` と書く）。\n"
    "  all    … 区間の全小節（普通はこれ）\n"
    "  last   … **区間の最後の1小節だけ**（サビ前の「ため」・フィル・キメ）\n"
    "  first  … 区間の最初の1小節だけ（頭で1発入れる）\n"
    "  `4,8`  … 区間の**4小節目と8小節目**だけ（フレーズ終わりのフィル）\n"
    "  `1-2`  … 区間の1〜2小節目だけ（ビルドアップの前半）\n"
    "★**同じ楽器・同じ区間で `all` と併用してよい**——特定の小節を指したほうが優先され、\n"
    "  `all` は**残りの小節**を受け持ちます（二重に鳴りません）。\n"
    "  例: ベースに `all`＝'x---x---' と `last`＝**'x_______'** を書けば、\n"
    "  **区間の最後の1小節だけ、頭で1発鳴らして止まる**（＝ため）。\n"
    "  ※ここで 'x-------' と書くと**1小節鳴りっぱなし**になり、ためになりません（`_` で切ること）\n"
    # ★目録の 伴奏-6（2026-09-27）。以前は読めない・範囲外の指定を黙って「区間の全小節」にし、`all` の小節を奪っていた
    "★**小節の番号はその区間の中で数え、区間の小節数を超えない**こと。番号は半角の数字（`４` のような全角は読めない）で"
    "頭に 0 を付けない（`04` ではなく `4`）、"
    "範囲は小さいほうから（`3-1` は書けない）。**途中に空白を入れない**（`1, 3-4`・`1 - 2` ではなく `1,3-4`・`1-2`）。"
    "読めない指定・範囲外の番号は検査で所見として返します"
    # ★2026-10-01: 「直さずに残ると読めた小節だけで鳴る」と書いていたが、2026-09-30 から直らなければ工程で止まる
    "（直らずに残ると曲づくりはこの工程で止まります）"
)


def _voice_note(k: int) -> str:
    """MIDI の高さ → 国際式の音名（`pitches` と同じ書き方・同じ物差し。60＝`C4`）。綴りは表と同じ `song_table.spell_pitch`
    （C 基準＝調号の無い調なので黒鍵は ♯ の `C#4`）。★2026-10-01 まで階名（`ド4`・`ド♯4`）だった（判-A）。"""
    from song_table import spell_pitch                      # noqa: E402（循環を避けて遅延）
    return spell_pitch(k)


#: 根音を置く高さ（`register_octave` 0 のとき）。**組み立て（`song_audio.chord_notes`）の値から出す**（書き写さない）
_ROOT_LO, _ROOT_HI = min(song_audio._ROOTS.values()), max(song_audio._ROOTS.values())


def root_range_c(shift: int = 0) -> tuple[int, int]:
    """`register_octave` 0 の根音の範囲を、**C 基準（移調前）の物差し**で（2026-10-01）。

    組み立て（`song_audio.chord_notes`）は区間の調へ移したコード名の根音を、**鳴る高さで** `_ROOT_LO`〜`_ROOT_HI`（C4〜B4）に置く。
    C 基準で書いたコード・`pitches`・歌の音域と見比べるには、区間の移調量 `shift` だけ戻す（F の曲＝+5 なら G3〜F#4）。"""
    return _ROOT_LO - shift, _ROOT_HI - shift


def _voicing_text() -> str:
    """**`tones` で弾く音をどの高さに積むか**（組み立て `song_audio.pitched_events` の積み方）を LLM に見せる文（2026-09-30・出-9）。

    ★以前は指示書に無く、`register_octave` の例も「パッドは0」だったので、そのまま書くと和音がミクの歌う高さに重なった
      （『孤独のまま』のサビで、ピアノの和音の139/204音・ギターの140/192音が歌の音域の中）。機械は積み方を変えない——
      積み方と高さを見せて、LLM が `register_octave` で選ぶ。例の音は `chord_notes` から作る（組み立てと食い違わない）。
    ★**高さは C 基準（移調前）の物差しで見せる**（2026-10-01）。以前は根音の範囲を鳴る高さ（C4〜B4）で書き、歌の音域・`pitches` は
      C 基準で見せていたので、C 以外の調の曲では移調量の分だけ物差しが食い違っていた。この文は型の説明・指示書にも入る（曲ごとに
      変えられない）ので、範囲そのものは依頼文の【歌の音域】の区間ごとの行（`vocal_range_text`・`root_range_c`）に出し、ここは
      積み方（根音からの距離）と、移調の無い区間の例だけを書く。
    """
    ex = " ／ ".join(f"{c} → {'・'.join(_voice_note(k) for k in song_audio.chord_notes(c))}" for c in ("C", "Am", "G7"))
    return ("**`tones` で弾く音の高さ（機械の積み方）**: `register_octave` 0 のとき、"
            "**根音を依頼文の【歌の音域】の区間ごとの「`tones` の根音」の範囲（1オクターブ）に置き、その上へ基本形で積みます**"
            "（3度・5度、7度やテンションがあればさらに上。転回やオープンな並べ方にはしません）。"
            "範囲は `pitches`・歌の音域と同じ C 基準（移調前）の物差しで出しています"
            f"（区間の調へ移した鳴る高さでは、どの調でも根音が {_voice_note(_ROOT_LO)}〜{_voice_note(_ROOT_HI)} に入るので、"
            f"C 基準で見ると区間の移調量の分だけずれます）。移調の無い区間の例: {ex}。"
            "三和音なら根音の範囲の上端から5度上まで、7度まで積むと短7度上まで届きます"
            "——**歌の中心と同じ高さになりやすい**ので、依頼文の歌の音域と見比べてください。"
            "`arp_up`/`arp_down` は同じ音を1つずつ、`root` は根音だけ、`fifth` は根音と5度。"
            "**`register_octave` は1つで1オクターブ**動かします（-1 なら根音の範囲が1オクターブ下、+1 なら1オクターブ上）。"
            "ベースのパート（`bass: true`）は7度・テンションを積まず、分数コードでは root を分母の音で弾きます。"
            "**`pitches`・リフは書いた高さのまま**鳴ります（`register_octave` は効かない）")


def _register_text() -> str:
    """`register_octave` の説明（高さは `_voicing_text` と同じ所から出す・C 基準の範囲は依頼文の区間ごとの行・2026-10-01）。"""
    rows = "／".join(f"{o:+d}＝その{abs(o)}オクターブ{'下' if o < 0 else '上'}" if o else
                    "0＝依頼文の【歌の音域】の区間ごとの「`tones` の根音」の範囲（**歌の中心と重なりやすい**）"
                    for o in range(-2, 3))
    return ("オクターブ。**`tones`（コードの構成音）で弾く音にだけ効く**——`pitches`・リフで高さを書いた音は書いた高さで鳴る。"
            f"1つで1オクターブ: {rows}（移調の無い区間なら 0＝根音 {_voice_note(_ROOT_LO)}〜{_voice_note(_ROOT_HI)}・"
            f"-2＝{_voice_note(_ROOT_LO - 24)}〜{_voice_note(_ROOT_HI - 24)}）。和音はそこから上へ積む。"
            "**依頼文の歌の音域と見比べて決める**（歌と同じ高さで和音を厚く鳴らすと歌が埋もれる。ベースは -2 がふつう）")


class SectionPattern(BaseModel):
    """**同じ楽器の、セクションごとに違う弾き方**（Aメロは白玉／サビは8分刻み、など）。"""
    # ★説明を実体に合わせた（2026-09-29）。以前は「verse / pre / chorus / intro / outro」と決まった名前のように書いていたが、
    #   区間名は設計が自由に付ける（`verse1`・`chorus_last` など）。照らし方は `song_checks.resolve_section`
    sections: list[str] = Field(description=(
        "このパターンを使う区間。**依頼文の構成に並んだ区間名を1字も違わずに**書く（1つのパターンを複数の区間に使ってよい）。"
        "構成に無い名前（大文字小文字・空白だけ違うものも）は検査で所見として返します"))
    pattern: str = Field(description=_GRID)
    # ★`inversion` を足した（2026-09-27・目録の 伴奏-1）。以前はベースの `root` を機械が黙って転回形へ差し替えていた
    tones: str = Field(description=(
        "root=**根音**のみ / chord=和音 / arp_up=分散上行 / arp_down=分散下行 / fifth=根音と5度 / "
        "**inversion=転回形で滑らかにつなぐ**（コードごとに根音・3度・5度から、前のコードの低音にいちばん近い高さの音を選び、"
        "**選んだ高さのまま**鳴らす（高さは `register_octave` -2 のベースの音域〈鳴る高さで C2〜B3〉で選び、"
        "`register_octave` が -2 でないパートはその差のオクターブだけずらして同じ線をなぞる）。"
        "ベースを根音で跳ばさず、並行5度を避けたいとき）。\n"
        "※**chord** はコードの構成音を全部同時に鳴らす（`Am7` なら A・C・E・G。7度・6度・テンションも鳴る）。\n"
        "※**arp_up / arp_down** は、**chord と同じ構成音**（7度・テンションを含む。`Am7` なら4音）を、打点ごとに"
        "**下から順に（arp_down は上から順に）1音ずつ**たどり、最後まで行ったら最初の音へ戻る。"
        "**小節の頭と、コードが替わった所（1小節に複数あるコードの2つ目以降）では、そのコードの最初の音から**始まる。\n"
        "※**分数コード**（`G/F`）では、ベースのパート（`bass: true`）の root と、どのパートでも inversion は"
        "**分母の音（F）**を弾きます。\n"
        "※ここに無い値は検査で所見として返します（直らずに残ると曲づくりはこの工程で止まります。根音などで代わりに鳴らすことはしません）\n"
        # ★積み方と高さを見せる（2026-09-30・出-9）
        "※" + _voicing_text()))
    # ★2026-10-04（組N2）: pitches・use_riff・at_bars も既定をやめて必ず書かせる（以前は 空・false・all）
    pitches: list[str] = Field(description=(
        "（**必ず書く**。コードの構成音から選ぶ伴奏・刻みなら空の並び `[]`）"
        "★**自分の旋律を書くときだけ**音名を並べる（対旋律・リフ・ソロ）。2026-09-17 追加。\n"
        "**国際式の音名**で、`pattern` の音の数だけ並べる（例 `[\"E5\",\"G5\",\"A5\",\"G5\"]`）。"
        "`C4` が中央のド、変化記号は `#` と `b`（`F#4`・`Bb3`）、**オクターブの数字は必ず付ける**。"
        "C メジャー基準（移調前）で書く（依頼文の歌の音域もこの書き方・この物差しで出ています）。移調は機械がやります"
        "（転調する曲では、区間ごとにその区間の調へ移します。1つのパターンが転調の前後の区間をまたいでも、それぞれの調で鳴ります）。\n"
        "**入れると `tones` は無視され、コードに関係なく、書いた高さでこの音が鳴ります**"
        "（`register_octave` は効きません——歌の上を行くなら、歌の最高音より上の音をここに書く）。\n"
        "空の並び `[]` なら `tones` でコードの構成音から選びます（伴奏・刻みはこちら）。\n"
        # ★目録の 伴奏-9（2026-09-27）。以前は余りを黙って捨て、足りない打点と読めない階名を黙って無音にしていた
        "※**`pattern` の打点（`X` `x` `.`）の数とちょうど同じ数**にしてください。"
        "数が合わない・読めない音名（階名の `ミ5`・記号の ♯♭・オクターブ無しの `E` など）は、検査で所見として返します"))
    use_riff: JsonBool = Field(description=(
        "★**ここでリフを鳴らすか**（**必ず書く**。true / false。2026-09-17 追加）。`true` にすると、"
        "`ArrangePlan.riff` に書いた音形が**そのまま**鳴ります（`pitches` も `tones` も無視）。\n"
        "**リフは曲に1つ**で、**イントロ・間奏・最後の3か所で同じものを鳴らす**のが定石です。"
        "★**音形を書き写さないでください**——1か所（`riff`）に定義して、"
        "使う所から**指す**。書き写すと、どこかで変わってリフでなくなります。\n"
        "※リフ（riff）が空のまま `true` にすると検査で所見として返します（`pitches`・`tones` で代わりに鳴らすことはしません。"
        "直らずに残ると曲づくりはこの工程で止まります）"))
    # ★**奏法はここに書かない**（2026-09-23 に専用の工程へ切り出した）。
    #   一度ここに `articulations` を置いたが、**`pattern` は1小節ぶん**なので
    #   「フレーズの初めの1音だけ深く落とす」が書けなかった（本人指摘）。
    #   奏法は**楽譜が組み上がってから、音1つずつに**掛ける（`artic_plan.py`）。
    at_bars: str = Field(description=_AT_BARS)


class PartPlan(BaseModel):
    # ★**既定値を置かない**（2026-09-13）。`default=1` だったため、LLM が書き忘れると**全パートが
    #   1番の楽器（ベース）で鳴った**——『この衝動』で歪みギター・クリーンギター・ピアノの役割が
    #   全部 GM35 のベース音色になり、関門は見抜いたのに直す経路が無かった。必須にして書かせる。
    # ★上限を 8 → **10** に（2026-09-27）。編成は 2〜10本を選べる（`Instrumentation.instruments`）のに、ここが 8 までで
    #   9・10番目の楽器を指せなかった
    # ★上限を 10 → **15** に（2026-10-03・§2 c7）。編成の本数の上限（`instrumentation.MAX_INSTRUMENTS`＝旋律のチャンネルの数）と同じ
    instrument: JsonInt = Field(ge=1, le=15, description=(
        "**編成で決まった楽器の番号**（プロンプトの【この曲の編成】に 1. 2. …と並んでいる番号）。"
        "**パートごとに違う番号になるのが普通**（ベースは 1、ギターは 2、のように）。"
        "**そこに無い楽器は使えません**——編成は前の工程で決まっています"))
    # ★`role` は表示用（2026-10-02 本人決定・案b）。以前は組み立て（`song_audio.pitched_events`）が `role` の文字（bass で始まるか）と
    #   `register_octave <= -2` からベースかを推し量り、分数コードの低音と和音の積み方を変えていた。ベースかどうかは `bass` に明示させる。
    #   `drums` だけは書けない（書き出しが打楽器として扱う・`_issues` の所見）
    role: str = Field(description=("このパートの役割を表す短い名前（bass / chord / arpeggio / lead 等・表示用。"
                                   "鳴り方は変わらない——ベースかどうかは `bass` に書く）。"
                                   "`drums` は書けない（打楽器のパートの印。ドラムは drums に書く）"))
    bass: JsonBool = Field(description=(
        "このパートが**ベース**なら `true`、それ以外は `false`（必ず書く）。`true` のパートは、`tones` の root を"
        "分数コード（`G/F`）では**分母の音（F）**で弾き、和音（chord）に7度・6度・テンションを積まない（根音・3度・5度だけ）。"
        "`role` の名前や `register_octave` からは決めない（書いた値だけで決まる）"))
    patterns: list[SectionPattern] = Field(description=(
        "**セクションごとの弾き分け**。同じ楽器でも場所で変える（例: ベースは verse で 'x---x---'、"
        "chorus で 'xxxxxxxx'）。**sections が重ならないように**分けること——2つのパターンが同じ区間の同じ小節を受け持つのは書けない"
        "（at_bars で小節を指したパターンと、同じ区間の all のパターンは重ならない＝all は残りの小節。検査で所見として返します）"))
    # ★説明を実体に合わせた（2026-09-27）。以前は「ボーカル1.0基準」と書いていたが、歌との釣り合いは書き出しで
    #   伴奏全体を一律に合わせる（`score_to_vpr._accomp_gain_db`・`v6_export` の `vocal_lead_db`／`accomp_trim_db`）ので、
    #   gain が効くのは**パートどうしの比**だけ
    gain: JsonFloat = Field(ge=0.1, le=1.2, description=(
        "このパートの音量。**伴奏のパートどうしの比**です（0.7 がふつうの大きさ。目安 0.3〜0.8）。"
        "**歌との釣り合いは書き出しで伴奏全体をまとめて合わせる**（歌が伴奏より一定だけ前に出るよう、伴奏全体の音量を"
        "機械が決める）ので、全パートを同じだけ上げ下げしても歌との釣り合いは変わりません。"
        "歌を前に出したい・引っ込めたいときに gain で調整しないでください。"
        "区間の強さによる音量差（指示書の表）は、この値に掛かります"))
    # ★上限を +1 → **+2** にした（2026-09-11）。「対旋律を歌の上へ置いて」と指示しても
    #   **+1 では歌（サビで 74〜84）の上へ出られず**、指示に従いようがなかった。
    #   実測でぶつかり 29.1%。**スキーマが守れない指示を禁じていた**という型の欠落。
    # ★例「パッドは0」をやめた（2026-09-30・出-9）。0 は根音が C4〜B4 で、そのまま書くと和音が歌の高さに重なった。
    #   高さの対応を見せて、歌の音域と見比べて選んでもらう
    register_octave: JsonInt = Field(ge=-2, le=2, description=_register_text())


class DrumLine(BaseModel):
    """打楽器1つ分の1小節パターン。**楽器を自由に選べる**（従来はキック・スネア・ハイハットの3つ固定だった）。"""
    # ★一覧は `timbre_map.GM_DRUM` から出す（2026-09-27・目録の 打-3）。以前は知らない名前をシェイカーで鳴らしていた
    # ★**選べるのは編成のキットにある打楽器だけ**（2026-09-29）。一覧は【この曲の編成】のドラムの行に、音源から読んで出す
    #   （`timbre_map.kit_drum_menu`）。以前はどのキットでも同じ一覧を見せ、キットに無い打楽器を機械が別の番号へ運んでいた
    inst: str = Field(description=(
        "打楽器名。**【この曲の編成】のドラムの行に並んだ「このキットで鳴らせる打楽器」の名前だけ**"
        "（キットによって入っている打楽器が違います。かっこ内はその音の中身）。"
        "名前の全体は " + " / ".join(DRUM_NAMES) + " ですが、キットに無いものは鳴りません。\n"
        "**名前は GM（General MIDI）の打楽器の正式な名前を小文字・`_` つなぎにしたものだけ**（`Bass Drum 1` → `bass_drum_1`・"
        "`Closed Hi Hat` → `closed_hi_hat`・`Pedal Hi-Hat` → `pedal_hi_hat`）。略称（kick・snare・hihat・bd・hh など）は書けません。"
        "開いたハイハットは open_hi_hat の行に書く（closed_hi_hat の行に `o` は書けない）。"
        "オーケストラ用キットではタム（low_tom ほか）の番号がティンパニです（一覧のかっこ内で分かります）。"
        "**一覧に無い名前・キットに無い打楽器は検査で所見として返します**"
        "（近い打楽器やキットの別の音で代わりに鳴らすことはしません。直らずに残ると曲づくりはこの工程で止まります）"))
    pattern: str = Field(description=_DRUM_TAB)


class DrumSectionPattern(BaseModel):
    sections: list[str] = Field(description=(
        "このパターンを使う区間。**依頼文の構成に並んだ区間名を1字も違わずに**書く"
        "（構成に無い名前は、大文字小文字・空白だけ違うものも検査で所見として返します）"))
    lines: list[DrumLine] = Field(description=(
        "このセクションで鳴らす打楽器を並べる。**必要な数だけ**——"
        "ハイハットだけの薄い場所もあれば、クラップやタムを足して盛り上げる場所もある"))
    at_bars: str = Field(description=_AT_BARS)


class DrumPlan(BaseModel):
    patterns: list[DrumSectionPattern] = Field(description=(
        "**セクションごとの叩き分け**。Aメロはハイハットだけ、サビは8ビート全開、のように変える。"
        "鳴らさないセクションは単に書かない。2つのパターンが同じ区間の同じ小節を受け持つのは書けない"
        "（at_bars で指したパターンと同じ区間の all は重ならない＝all は残りの小節）"))
    # ★2026-10-04（組N2）: 既定 0.55 をやめて必ず書かせる（音程パートの gain は以前から必須）
    gain: JsonFloat = Field(ge=0.1, le=1.2, description=(
        "ドラム全体の音量（**必ず書く**。目安 0.4〜0.7）。音程パートの gain と同じ物差しの、**パートどうしの比**（歌との釣り合いは書き出しで伴奏全体を合わせる）"))
    # ★**`fill_every_bars`（N小節ごとに決まった形のフィル）は外した**（2026-09-27・目録の 打-2）。
    #   書いた小節を機械がタム16分＋クラッシュで丸ごと差し替えていた（曲頭から数える・4拍前提で 3/4 ではみ出す）。
    #   フィルは `at_bars`（`last` や `4,8`）で LLM が自分で書く。前の版の記録に残っていても読める（知らない鍵は無視）。


class ArrangePlan(StepStatus):
    reason: str = Field(description="編曲の狙い（どの楽器をどこで足し引きして起伏を作ったか）")
    riff: list[str] = Field(description=(
        "（**必ず書く**。使わない曲は空の並び `[]`）★**この曲のリフ**（曲を通して繰り返す短い句）。音名を **3〜6音**並べる"
        "（例 `[\"G5\",\"E5\",\"C5\",\"D5\"]`）。`pitches` と同じ書き方（国際式の音名・C メジャー基準）。\n"
        "**ここに1つだけ定義して**、鳴らしたいパターンで `use_riff: true` と指してください。\n"
        "★**イントロで出し、間奏で戻し、最後にもう一度**——"
        "**同じ音形が戻ってくること**がリフの本体です。聴き手が『この曲の音』として掴みます。\n"
        "使わない曲では空の並び `[]` と書きます（リフなしを選んだ印）"))
    # ★「2〜4本」をやめた（2026-09-29）。指示書の【編成の目安】は曲の種類で2〜6本、編成は2〜10本を選べ、指示書は
    #   「選ばれた楽器は原則すべて使う」と頼んでいる——ここだけ上限が低く、食い違っていた（本数を機械が縛る所は無い）
    parts: list[PartPlan] = Field(description=(
        "音程のあるパート。本数は指示書の【編成の目安】が目安（曲の種類で2〜6本）。"
        "**編成で選ばれた楽器は原則すべて使う**"))
    # ★**ドラムの持ち主は伴奏**（2026-09-30・出-8）。以前は「静かな曲では無しでよい」と書いていて、編成が standard と決めた
    #   『孤独のまま』で伴奏が理由も書かずに null を返し、ドラムが1音も鳴らなかった（検査も点検役も気づかなかった）
    # ★2026-10-04（組N2）: drums・lead_gain も既定 null をやめ、null も明示で書かせる
    drums: DrumPlan | None = Field(description=(
        "ドラム（**必ず書く**。ドラム無しは null）。**鳴るのはここに書いたものだけ**です（グルーヴのドラムは土台の指定で、それ自体は鳴りません——写して鳴らす）。"
        "**編成がドラムのキットを決めたら必ず書く**（静かにしたいなら、鳴らす区間を絞る・金物やリムだけにする。"
        "書かないと検査で所見として返します。ドラム無しにするのは編成の判断＝「ドラム: none」）。"
        "**編成が「ドラム: none」なら null と書く**——中身を書いたら検査で所見として返します"))
    # ★**`accomp_gain`（伴奏全体の音量）は外した**（2026-09-27・目録の 音量-4）。どこにも効いていなかった。
    #   歌と伴奏の釣り合いは vpr の伴奏トラックの初期音量が決める。前の版の記録に残っていても読める。
    # ★**主旋律の楽器の音量**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §3・定義は仕様書 §15）。編成が `lead` を
    #   書いたときだけ書く・必ず書く（無い・余計に書いたは伴奏の検査が見る）。組み立ては既定値で埋めない
    lead_gain: JsonFloat | None = Field(ge=0.1, le=1.2, description=(
        "**主旋律を弾く楽器（編成の `lead`）の音量**（**必ず書く**。編成に主旋律の楽器が無ければ null）。音程パートの gain と同じ物差しの、パートどうしの比。"
        "編成が主旋律の楽器を決めたときだけ書く（決めていなければ null）"))


#: 区間の**強さ（intensity 1〜5）→ 音量差（dB）**。いちばん強い区間を 0 とした相対値（由来は下の `intensity_db` の上）。
#: ★指示書へは**ここから**書き出す（`<<INTENSITY_DB>>`・2026-09-27・目録の 伴奏-4）。以前の指示書は「上の dB は
#:   機械的に付けます」と、目安の表（強さ1で −8〜−12dB）を機械の値のように見せていた。実際に付くのは強さ1でも −4dB。
_INTENSITY_DB = {1: -4.0, 2: -1.5, 3: -0.8, 4: -0.3, 5: 0.0}


def _intensity_text() -> str:
    return " / ".join(f"強さ{k}: {v:+.1f}dB" if v else f"強さ{k}: 0dB" for k, v in sorted(_INTENSITY_DB.items()))


_arranger: Agent[None, ArrangePlan] = Agent(
    pick_model("ARRANGE", "flash"), output_type=ArrangePlan,
    system_prompt=(
        prompt("song_arrange").replace("<<DYNAMICS_GUIDE>>", DYNAMICS_GUIDE)
        .replace("<<INTENSITY_DB>>", _intensity_text())
        .replace("<<VOICING>>", _voicing_text())               # 和音の積み方と高さ（2026-09-30・出-9）
    ),
)


# ---- ArrangementSpec への展開 -------------------------------------------------

# 強弱の文字 → ベロシティ。プロの実測は中央値88・標準偏差16.6・幅105 で、
# 従来の「表拍90／裏拍72」の2値では平坦すぎた（打ち込みっぽさの主因）。
#: ★**`o`／`O` も打点として受ける**（2026-09-21・ドラムタブに合わせた）。
#: ドラムタブの世界標準では **`x`＝シンバル・ハイハット／`o`＝ドラムの打点**（キック・スネア・タム）で、
#: **大文字がアクセント**。それまで `o` を一度も使っておらず、**標準の書き方が通らなかった**。
#: 強さの対応は `x`＝`o`（ふつう）、`X`＝`O`（強く）、`.`＝ゴーストノート。
#: ★**`o` `O` はドラムの行だけ**（2026-10-02・ず-9・本人「2通りに書けるものは良くない」「標準に寄せて1つにそろえる」）。
#:   以前は音程パートの `o` も `x` と同じ強さで読み、ドラムの行の `_` も `-` と同じ休みで読んでいた（書き方が2通り）。
#:   いまは音程パートの打点は `X x .`（`_PITCHED_VEL`）・休みは `-`（伸ばす）と `_`（切る）、ドラムの行の打点は `X x O o .`（`_VEL`）・
#:   休みは `-` だけ（ドラムタブの標準）。それ以外は記法の所見（`_pattern_format`）。
_VEL = {"X": 110, "O": 110, "x": 88, "o": 88, ".": 62}
_PITCHED_VEL = {c: _VEL[c] for c in "Xx."}
#: 打点以外に書ける字（音程パート: `-`＝前の音を伸ばす・`_`＝そこで切る／ドラム: `-`＝休み）
_PITCHED_REST, _DRUM_REST = "-_", "-"


def _bar_segments(pattern: str) -> list[str]:
    """グリッド文字列を**小節線 `|` で区切った小節ごと**の並びにする（前後の半角の空白とタブだけ落とす・両端の `|` は小節線として読むだけ）。

    `"|x-x-x-x-|"` → `["x-x-x-x-"]`、`"x-x-x-x-|o-o-o-o-"` → `["x-x-x-x-", "o-o-o-o-"]`。
    ★**途中の空白は落とさない**（2026-10-02 本人「おすすめで」・仕様書 §0-6・§8・§17 ず-22）。以前は行の中の空白（半角・全角・タブ）を
      どこでも落として読み、`x-x- x-x-` も `x - x - …` も同じ1小節になった（書き方が何通りにもなる）。空白は記法全体で「区切り」の意味で、
      区切りの要らない打点のパターンでは書けない字にした（10/2 に調名で決めたのと同じ方針）。途中の空白は記法の所見（`_pattern_format`）が
      書き直した形を添えて返し、組み立てへは届かない（ここで読めば空白は読めない文字＝その所は無音）
    ★**両端の縦棒は1本ずつだけ取る**（2026-10-02・洗い出し）。以前は `strip("|")` で何本でも取り、`||x-x-x-x-||` も1小節として読んだ
      （仕様書 §8 の文法は `[ "|" ] , bar , [ "|" ]`）。2本目からは空の小節として残り、記法の所見（`_pattern_format`）が「縦棒が続いている」と返す
    ★**前後の全角の空白・改行などは落とさない**（2026-10-03・判断4 本人決定・§0-6）。読めない文字として残り、記法の所見（`_pattern_format`）が返す
    """
    body = strip_ascii_ws(pattern)
    body = body[1:] if body.startswith("|") else body
    body = body[:-1] if body.endswith("|") else body
    return body.split("|") if body else []


def _bars_written(pattern: str) -> list[str]:
    """`_bar_segments` から空の小節（縦棒が続いた所）を除いたもの——**検査が長さ・打点を数えるためだけ**（縦棒の所見は `_pattern_format` が別に出す。
    空の小節を1小節目と数えて「0文字」「空」まで重ねて言わないため）。組み立ての読み方（`_clean_grid`）は変えない。"""
    return [s for s in _bar_segments(pattern) if s]


def _compact(pattern: str) -> str:
    """空白をすべて除いた形——**所見に添える書き直し方を作るためだけ**（読み替えには使わない・`song_design.key_problem` と同じ扱い）。"""
    return "".join((pattern or "").split())


def _clean_grid(pattern: str) -> str:
    """グリッド文字列の**1小節目**（小節線と前後の空白を落としたもの・2026-09-21。途中の空白は 2026-10-02 から落とさない＝`_bar_segments`）。

    ★**ドラムタブは小節線 `|` を書くのが作法**なので、`HH|x-x-x-x-|` のように来る。
      落とさずに数えると**そこから後ろが1コマずれる**——旋律の枠で実際に起きた事故と同じ型
      （LLM が小節の間に空白を入れ、それを1文字と数えて**以降が16分ひとつぶんずれた**。
      グリッド率は 1.000 のままなので**数字では捕まらない**・2026-09-10）。
    ★**`|` で区切った2小節目以降は読まない**（2026-09-29）。以前は `|` を落として続けて読んでいたので、
      8文字の小節を2つ書くと16文字＝16分の1小節として読み、**所見なしに倍の密度で鳴った**。
      パターンは1小節ぶんの作りなので、書かれた1小節目だけを訳し、2小節目以降は記法の検査（`_pattern_format`・型は `pattern_notation`）が所見にする。
    """
    segs = _bar_segments(pattern)
    return segs[0] if segs else ""


def _grid_unit(pattern: str, slots: int) -> tuple[int, float]:
    """パターンの**文字数から**「1小節を何コマに割ったか」と1コマの拍数を出す（2026-09-15）。

    ★**それまで1コマ＝8分に決め打ちしていた**（`at_beat = i * 0.5`）。つまり
    **ドラムに16分を書く手段が無かった**——旋律は16文字グリッドなのに、リズムだけ8文字。

    実害: BPM200 の曲でハイハットが `'x-x-x-x-'`（4分＝**300ms ごと**）になり、
    **BPM100 の8ビートと同じ密度**になった。本人に「**200なのかもしれないけど
    半分の100ぐらいに聞こえちゃった**」と言われた（2026-09-15『初期化させて』）。

    内部表現（`at_beat` は小数）は最初から16分を置けるので、**足かせはここだけ**だった。
    **8文字はこれまでどおり8分**、**倍の文字数を書いたら16分**として読む。
    """
    n = len(_clean_grid(pattern))
    beats = slots * 0.5                                    # 1小節の拍数（4/4→4.0・3/4→3.0）
    if n >= slots * 2:
        return slots * 2, beats / (slots * 2)
    return slots, beats / slots


#: 国際式の音名（`E5`）→ MIDI。読めなければ None（その音は置かない）。
def _pitch_of(text: str, shift: int = 0) -> int | None:
    """`"E5"` のような国際式の音名（C メジャー基準）を MIDI へ。

    ★`shift` は**移調ぶん**。`degree` と違い絶対音高は後から追従しないので、ここで足す。
    ★読み方は音名の記法の型（`song_llm.PITCH`＝国際式・2026-10-01・判-A）。検査（伴奏・彩りの `written`）と同じ1つ。
      以前は固定ドの階名＋オクターブ（`ミ5`）で、2026-09-30 まではここで文字列を切り分けてオクターブの無い `ミ` を黙って `ミ4` と読んでいた。
      読めない音名は検査が記法の所見にし、伴奏・彩りの工程へ返す（直らなければその工程で止まる）。
    """
    k = PITCH.read(text)
    return None if k is None else k + shift


def _grid_steps(pattern: str, tones: str, slots: int = 8,
                pitches: list[str] | None = None, shift: int = 0) -> list[StepEvent]:
    """強弱つきグリッド文字列 → StepEvent 列。`slots` は1小節の8分の数（4/4=8・3/4=6・6/8=6）。

    **文字数が倍なら16分グリッド**として読む（`_grid_unit`）。

    ★**書かれたとおりに訳すだけで、補正しない**（2026-09-28 本人方針「組み立ては一切補正しない」）:
      ・音が伸びるのは**後ろに `-` が続く所だけ**。`_`・読めない文字・**パターンの終わり**で止まる
        （以前は1小節に満たないパターンの最後の音を**小節の終わりまで伸ばし**、読めない文字も `-` と同じに伸ばしていた）
      ・`tones` が知らない値なら**その打点は鳴らさない**（以前は根音で鳴らした）。`pitches` があれば `tones` は見ない
      ・どちらも伴奏の形式の検査（読めない文字は記法の `_pattern_format`・長さは `_length_issue`・`tones` は `_pattern_issues`）が先に所見にして伴奏の工程へ返している
      ・曲づくりでは、直らなければ伴奏の工程で止まり（`StepUnfixed`）、`to_spec` も入口で検査と同じ判定をして止める（2026-09-30）
        ので、上の「鳴らさない」は曲づくりからは届かない（この関数を単独で呼んだときの読み方として残してある）
    """
    pattern = _clean_grid(pattern)                         # 小節線 `|` と前後の空白を落としてから数える
    slots, unit = _grid_unit(pattern, slots)
    # 打点は `X x .` だけ（2026-10-02・ず-9。`o` `O` はドラムの行の字で、ここでは読めない文字＝記法の所見）
    hits = [i for i, c in enumerate(pattern[:slots]) if c in _PITCHED_VEL]
    if not hits:
        return []
    tones = strip_ascii_ws(tones)                          # 検査（`_pattern_issues`）と同じ読み方（前後は半角の空白とタブだけ・§0-6）
    if not pitches and tones not in TONES:                 # 知らない `tones` は鳴らさない（検査が所見にしている）
        return []
    end = min(slots, len(pattern))                         # 書かれた長さ（1小節に満たなければ、その先は無音）
    out = []
    for n, i in enumerate(hits):
        # ★**`-` が続く所だけ伸ばす**（2026-08-23 に `_` で切れるようにし、2026-09-28 に「`-` 以外では止まる」にした）。
        #   `x-------` は1小節伸ばす（白玉）、`x_______` は1発で止まる（ため）。
        # ★**パターンの長さで打ち切る**（2026-09-23 修正）。`"x-x-"`（4文字）のような短いパターンで
        #   `IndexError` になり、編曲が丸ごと落ちていた。2026-09-28 からは**最後の音も書いた長さまで**
        #   （小節の終わりまで伸ばさない）。
        stop = i + 1
        while stop < end and pattern[stop] == "-":
            stop += 1
        dur = (stop - i) * unit
        vel = _PITCHED_VEL[pattern[i]]
        if tones == "chord":
            t = [Tone(degree=0, octave=0), Tone(degree=1, octave=0), Tone(degree=2, octave=0)]
        elif tones == "fifth":
            t = [Tone(degree=0, octave=0), Tone(degree=2, octave=0)]
        # ★分散和音は**印だけ**を置き、どの音かは組み立て（`song_audio.pitched_events`）が和音ごとに決める（2026-09-27）。
        #   以前は `degree = n % 3`（小節の頭からの打点の番号）で、7度・テンションが鳴らず、1小節2和音の後半が前の和音の続きの順になった
        elif tones == "arp_up":
            t = [Tone(degree=ARP_UP, octave=0)]
        elif tones == "arp_down":
            t = [Tone(degree=ARP_DOWN, octave=0)]
        elif tones == "inversion":                         # 転回形でつなぐ（どの構成音かは `song_audio.bass_line`）
            t = [Tone(degree=INVERSION, octave=0)]
        else:                                              # root（知らない値は上で鳴らさないことにした・伴奏-7）
            t = [Tone(degree=0, octave=0)]
        # ★**自分の旋律があれば、コード構成音より優先する**（2026-09-17）。
        #   対旋律・リフ・ソロは「いま鳴っているコードのどの音か」では書けない。
        if pitches:
            _k = _pitch_of(pitches[n], shift) if n < len(pitches) else None
            t = [Tone(degree=0, octave=0, key=_k)] if _k is not None else []
        out.append(StepEvent(at_beat=i * unit, dur_beats=dur, tones=t, velocity=vel))
    return out


# 楽器ごとのベロシティ基準差 `_DRUM_BASE` は上（凡例 `_DRUM_TAB` の手前）に置いた（2026-09-27・凡例へ出すため）。


def _grid_drums(plan: DrumSectionPattern, slots: int = 8,
                have: Mapping[str, str] | None = None) -> list[DrumHit]:
    """ドラムも強弱を文字で受ける。表裏の差はここで作らず、書かれたとおりに鳴らす。

    ★**行ごとに文字数を見る**（2026-09-15）——ハイハットだけ16文字（16分）で刻み、
    キックとスネアは8文字（8分）、という**混在を許す**。速い曲はこの形になる。

    `have` は編成のキットで鳴らせる打楽器（`timbre_map.kit_drums`）。渡すと、**キットに無い打楽器の行は置かない**
    （2026-09-29。検査が先に所見で返している。None＝キットで絞れない）。
    ※曲づくりでは、知らない打楽器名・キットに無い打楽器は伴奏の工程で止まり、`to_spec` も入口で止める（2026-09-30）ので、
      下の「置かない」は届かない（この関数を単独で呼んだときの読み方として残してある）。
    """
    out = []
    for line in plan.lines:
        # ★**打楽器名は GM の正式な名前だけ**（2026-10-02。以前は別名・大文字・空白入りを正準名にそろえて読んでいた）。
        #   知らない名前は検査（`arrange_findings`）が先に所見にして返す。★残ったら**その打楽器は鳴らさない**（2026-09-28）。
        inst = drum_canon(line.inst or "")
        if inst is None or (have is not None and inst not in have):
            continue
        base = _DRUM_BASE.get(inst, 0)
        # ★closed_hi_hat の行の `o`／`O` は検査の所見（`timbre_map.hat_line_problem`・開くのは open_hi_hat の行・2026-10-02）。
        #   ここへ届いたら（単独で呼んだとき）その打点は鳴らさない（閉じた・開いたのどちらかを当て推量で選ばない）
        no_o = inst == "closed_hi_hat"
        pat = _clean_grid(line.pattern)                    # 1小節目（小節線 `|` と前後の空白を落としてから数える）
        n_slots, unit = _grid_unit(pat, slots)
        # ★**1小節ぶんしか読めないことを、黙って通さない**（2026-09-21）。
        #   ドラムタブに移して `|`（小節線）を教えたら、LLM が**2小節ぶん**書くようになった
        #   （`o--o--o-o--o--o-|o---------------` のように**後半が明確に違う**例が実際に出た）。
        #   2026-09-29 から、`|` で区切った2小節目以降（記法の `_pattern_format`）・1小節を超えた文字（`_length_issue`）は検査が所見にして伴奏の工程へ返し、
        #   直らなければ伴奏の工程で止まる（2026-09-30。以前は残ったときの動き＝読んだのは1小節目だけ、を warns に書いて進んだ）
        for i, c in enumerate(pat[:n_slots]):
            if c in _VEL:
                if no_o and c in "oO":
                    continue
                out.append(DrumHit(inst=inst, at_beat=i * unit,
                                   velocity=max(20, min(127, _VEL[c] + base))))
    return out


# 音色表は timbre_map が正（編曲・MIDI で同じ表を見る）。
# かつては音色を7種に絞り、`piano → bell`（鉄琴）のように**寄せて**いた。numpy 音源に
# 定義が無い名前を渡すと KeyError で伴奏が落ちたためだが、その結果
# **LLM がピアノのつもりで書いた曲が鉄琴で鳴っていた**（2026-08-21 発覚）。
# ★2026-09-29: numpy 音源への近似の登録（`register_numpy_timbres`）と、楽器の一覧が無いとき役割名から音色を推す
#   `safe_timbre` は消した。楽器は編成の工程が決める（編成が作れなければ曲づくりを止める・`to_spec`）。


# 区間の**強さ（intensity 1〜5）→ 音量差（dB）**。いちばん強い区間を 0 とした相対値。
#
# なぜ機械で当てるか（2026-08-22）: 設計は強さ1〜5と幅を宣言していたのに、
# 実際に鳴った音は**全区間で 2.8dB しか差が無かった**（サビとAメロの差 0.3dB）。
# `PartPlan.gain` は**曲全体で1つ**しか無く、区間ごとの音量という概念が存在しなかったため。
# 業界の目安では 1〜2dB=わずか / 3〜6dB=はっきり / 7dB以上=劇的 なので、0.3dB は無いのと同じ。
#
# ★**値は控えめにする**。当初 -10〜0 にしたら **21.8dB** も開いて落ちサビがほぼ無音になった。
# 実測すると**編曲の楽器の出し入れだけで 5.0dB** 出ている——起伏の本体は編曲であって、
# ここは**足りない分を少し足すだけの下駄**。深追いすると編曲の意図を壊す。
# この値で 5.0dB → 8dB 程度（目安の「はっきり分かる」帯）に収まる。
# 値の表 `_INTENSITY_DB` は、指示書へ書き出すために上（`_arranger` の手前）へ移した（2026-09-27）。


def intensity_db(level: int | None) -> float:
    return _INTENSITY_DB.get(int(level) if level is not None else 3, -4.0)


_to_filter = song_audio.to_bar_filter                       # 小節集合 → 文字列（実体は song_audio）


# ★区間名の照らし合わせ方は `song_checks.resolve_section` の1か所（2026-09-28）。伴奏はここから使う。
#   2026-09-06『青空の境界線』: 設計 `Intro` と編曲 `intro` の食い違いで、完全一致の照合が全パートを黙って落とし、
#   伴奏が 2秒の無音になった。2026-09-28〜10-01 は大文字小文字・空白の違いを吸収して読んでいたが、2026-10-02（判断6・本人「おすすめで」）から
#   **設計の綴りと完全一致だけ**。違う綴りは伴奏の検査（`_section_issue`）が設計の綴りを添えた所見にして直させ、直らなければ伴奏の工程で止まる
#   （黙って落ちることはもう無い——2026-09-30 から組み立ての入口 `to_spec` も同じ検査で止める）
from song_checks import Notation, resolve_section, section_hint   # noqa: E402
from song_timeline import LEAD_IN                                # noqa: E402


def _song_sections(bars_of: Mapping[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """時間軸の区間から、**曲頭の空き小節（`LEAD_IN`）を除いたもの**（2026-09-29）。

    `LEAD_IN` は組み立てが曲の頭に置く空の小節で、設計の区間ではなく、歌も伴奏も鳴らない（`song_timeline`）。
    以前は `timeline.bars_of()` をそのまま使っていたので、伴奏の直しの区間表に混ざり、組み立ての見回りが毎回
    「区間『lead_in』に伴奏が1つも無い」と出していた。伴奏は設計の区間だけを見る（`lead_in` と書いた区間名も所見になる）。
    """
    return {k: v for k, v in (bars_of or {}).items() if k != LEAD_IN}


def fade_span(bars_of: dict[str, tuple[int, int]], min_bars: int = 2) -> tuple[int, int] | None:
    """フェードアウトを掛ける小節（1始まり・両端を含む）＝**曲の最後の区間**（2026-09-27）。

    最後の区間が `min_bars` に満たなければ、手前の区間をつないで `min_bars` 以上にする
    （1小節では下り坂にならない）。区間が無ければ None。
    """
    spans = sorted(bars_of.values(), key=lambda ab: (ab[1], ab[0]))
    if not spans:
        return None
    b0, b1 = spans[-1]
    for a, _b in reversed(spans[:-1]):
        if b1 - b0 + 1 >= min_bars:
            break
        b0 = min(b0, a)
    return (b0, b1) if b1 - b0 + 1 >= min_bars else None


def apply_fade_out(spec: ArrangementSpec, bars_of: dict[str, tuple[int, int]],
                   outro: str | None = None, n_bars: int = 0, depth_db: float = -14.0,
                   *, song: Any = None) -> int:
    """`ending='fade'` のとき、**曲の最後の区間に音量の下り坂**を作る（2026-08-23・対象を直した 2026-09-27）。

    ## なぜ機械がやるのか

    フェードアウトは編曲ではなく**ミックスの操作**（フェーダーを下げる）。
    設計が `fade` と決めたのだから、実行は機械でよい（＝設計の決定を音にする**組み立て**）。
    何をするかは伴奏の指示書（`song_arrange.md`）に書いてあり、やったことは `song` の記録に残す
    （`song_fix.record_change`・後の工程と作り直しの LLM に見える）。

    そして**編曲側には表現手段が無い**——`gain` は区間に1つしか持てず、
    `at_bars` で変えられるのはパターン（何を鳴らすか）だけ。
    実際、編曲は「4小節かけて静かにフェードアウトさせたい」と書きながら、
    **4小節とも同じパターン**しか出力できなかった（2026-08-23 実測）。

    区間の小節ごとにパートを分け、**1小節ずつゲインを下げる**。
    最後の小節で `depth_db` まで落ちる。

    ★**対象は曲の最後の区間**（`fade_span`・2026-09-27 目録の 音量-1）。以前は呼び手が
      「**最後の歌なし区間**」を選んで渡しており、**歌で終わる曲では途中の間奏がフェードした**。
      `outro` は旧い呼び出しの受け口で、**対象の選択には使わない**（渡されても曲の最後を下げる）。
    """
    fs = fade_span(bars_of)
    if fs is None or n_bars <= 1:
        return 0
    b0, b1 = fs
    span = list(range(b0, b1 + 1))
    keep, added = [], 0
    for p in spec.parts:
        inside = [b for b in span if _bar_match_v2(b - 1, n_bars, p.bar_filter)]
        if not inside:
            keep.append(p)
            continue
        outside = [b for b in range(1, n_bars + 1)
                   if b not in span and _bar_match_v2(b - 1, n_bars, p.bar_filter)]
        if outside:                                        # 区間の外はそのまま残す
            keep.append(p.model_copy(update={"bar_filter": _to_filter(set(outside))}))
        for b in inside:                                   # 区間の中は1小節ずつ下げる
            k = span.index(b) / max(1, len(span) - 1)      # 0.0（頭）〜1.0（最後）
            # ★下げた値のまま（2026-10-03・記L）。以前は 0.1 で止め、小さいパートは最後の小節まで下がりきらなかった
            g = p.gain * (10 ** (depth_db * k / 20))
            keep.append(p.model_copy(update={"bar_filter": f"{b}-{b}", "gain": g}))
            added += 1
    spec.parts = keep
    if added and song is not None:
        from song_fix import record_change                 # noqa: E402（循環を避けてここで）
        names = [k for k, (a, b) in sorted(bars_of.items(), key=lambda kv: kv[1]) if a <= b1 and b >= b0]
        record_change(song, "arrangement",
                      f"終わり方 fade: 最後の区間（{'・'.join(names)}＝{b0}〜{b1}小節）の音量を"
                      f"1小節ずつ下げ、最後の小節で {depth_db:.0f}dB にした（{added}パートを小節ごとに分けた）",
                      "設計が終わり方を fade と決めた。伴奏のパートは区間に音量を1つしか持てないので、下り坂は組み立てで作る")
    return added


# ---- 伴奏の検査（2026-09-27・`song_fix`）---------------------------------------------
#
# ★**機械は伴奏を書き換えない**。以前は `to_spec` が組み立てながら黙って直していた（目録の 伴奏-2/3/6/7/8/9・打-1/3）:
#   全パートが同じ楽器なら役割順に割り当て直す／編成に無い番号は役割名から音色を当てる／読めない `at_bars` は区間の全小節／
#   知らない `tones` は根音／1小節を超えた分・余った `pitches` は捨てる／知らない打楽器はシェイカー／「ドラムなし」なら落とす。
#   いまは**組み立ての前に検査して所見を返し、伴奏の工程に差分（パート単位）で直させる**（`fix_arrangement`）。
#   直しても残ったら、組み立ては下の `fallback` のとおりに動き、**それを所見と記録に残す**（`to_spec` の warns・`record_change`）。
#   ★2026-09-28 から `fallback` はどれも「鳴らさない／読めた所だけ鳴らす」。組み立ては補正しない（代わりの音・延長・当て推量をしない）。
#   ★2026-09-30 から、直しても残ったらその工程で曲づくりを止める（`song_fix.StepUnfixed`）。`to_spec` も入口で同じ検査を当て、
#     所見があれば不具合として止める（`AssemblyError`）ので、`fallback` のとおりに組み立てが動くことはもう無い。
#     `fallback` の文は、組み立てがどう読むか（読めた1小節目だけ・無音として読む…）の説明として残してある。

#: `tones` に書ける値（`SectionPattern.tones` の説明と同じ）
TONES = ("root", "chord", "fifth", "arp_up", "arp_down", "inversion")


class Issue(NamedTuple):
    """検査で見つけた1件。`fallback` は**組み立てがそこをどう読むか**（2026-09-30 から、残ったら工程で止まるので実際には使わない。説明として残す）。"""
    where: str
    what: str
    fallback: str = ""

    def finding(self) -> str:
        return f"{self.where}: {self.what}"

    def warn(self) -> str:
        return self.finding() + (f"——**組み立てでは{self.fallback}**" if self.fallback else "")


#: `at_bars` の項目1つ（番号か範囲）。★**数字は半角だけ**（2026-10-01・仕様書 §11・§17 ず-15）。以前は `int()` で読んでいたので、
#:   全角の数字（`４`）・`+3`・`1_0` も読めた（Python の `int` は全角数字と `_` 区切りを受ける）。
#: ★**途中の空白は書けない**（2026-10-02 本人「おすすめで」・仕様書 §0-6・§11・§17 ず-22）。以前は項目の前後と `-` の前後の空白を無視して読み
#:   （`1, 3-4`・`1 - 2`）、書き方が何通りにもなった。空白は記法全体で「区切り」の意味で、`at_bars` の区切りは `,` なので空白は要らない。
#:   前後の空白だけ取り除き、途中に空白のある項目は「読めない指定」にして、所見（`_at_issue`）に空白を除いた形を添える
#: ★**番号の頭に 0 を付けない**（2026-10-02・洗い出し・仕様書 §11 `number = digit19 , { digit }`）。以前は `[0-9]+` で `04`・`01-02` も
#:   `4`・`1-2` と同じに読んでいた（書き方が2通り）。`0` だけの番号も読めない指定（以前は「範囲外」）
_AT_ITEM = re.compile(r"([1-9][0-9]*)(?:-([1-9][0-9]*))?")
#: 頭に 0 を付けた番号（所見に書き直した形を添えるためだけ。読み替えない）
_AT_ZERO = re.compile(r"([0-9]+)(?:-([0-9]+))?")


def _read_at_bars(at: str, n: int) -> tuple[set[int] | None, list[str], list[str]]:
    """`at_bars` → 区間の中の位置（1始まり）の集合。`all` は None。読めない断片・範囲外の断片も返す（直さない）。
    文法は仕様書 `docs/components/song_notation.md` §11（`all`／`first`／`last`／番号・範囲を `,` か `、` で並べる。大文字小文字は問わない。
    前後の半角の空白とタブだけ取り除く——途中に空白のある項目・前後の全角の空白などは読めない断片）。"""
    spec = strip_ascii_ws(at or "all").lower()              # 前後は半角の空白とタブだけ（全角の空白などは読めない断片・§0-6・2026-10-03）
    if spec in ("", "all"):
        return None, [], []
    if spec == "first":
        return {1}, [], []
    if spec == "last":
        return {n}, [], []
    got: set[int] = set()
    bad: list[str] = []
    out: list[str] = []
    for tok in spec.replace("、", ",").split(","):
        m = _AT_ITEM.fullmatch(tok)                        # 項目の前後の空白も読まない（`1, 3` の ` 3` は読めない断片）
        if m is None:                                      # 全角の数字・頭に 0（`04`）・`odd`・`1〜2`・空の項目（`1,,3`）は読めない指定
            bad.append(tok or "（空の項目）")
            continue
        x = int(m[1])
        y = int(m[2]) if m[2] is not None else x
        inside = {o for o in range(x, y + 1) if 1 <= o <= n}
        if x > y or len(inside) < y - x + 1:
            out.append(tok)
        got |= inside
    return got, bad, out


def _at_issue(where: str, at: str, sections: list[str], bars_of: Mapping[str, tuple[int, int]]) -> list[Issue]:
    out: list[Issue] = []
    for s in sections or []:
        k = resolve_section(s, bars_of)
        if k is None:
            continue
        a, b = bars_of[k]
        n = b - a + 1
        got, bad, rng = _read_at_bars(at, n)
        if not (bad or rng):
            continue
        # ★途中の空白は書けない（2026-10-02）。空白を除いた形を添える（書き直し方を示すためだけ・読み替えはしない）
        # ★前後に全角の空白・改行などを付けない（2026-10-03・判断4 本人決定・§0-6）。取り除いた形を添える
        edge = edge_ws_problem(at)
        spaced = any(c.isspace() for c in trim_edge_ws(at))
        # ★頭に 0 を付けた番号（`04`・2026-10-02）。書き直した形を添える（読み替えはしない）
        zero = [(x, "-".join(str(int(g)) for g in m.groups() if g is not None)) for x in bad
                if (m := _AT_ZERO.fullmatch(x)) and all(g is None or int(g) > 0 for g in m.groups())]
        msg = (f"at_bars『{at}』"
               + (f" ／{edge}" if edge else "")
               + (f" の読めない指定: {'・'.join(repr(x) if any(c.isspace() for c in x) else x for x in bad)}" if bad else "")
               + (f" ／区間『{k}』は実際には{n}小節なので範囲外: {'・'.join(rng)}" if rng else "")
               + (f" ／途中に空白を入れない——空白は記法の「区切り」の印で、at_bars の区切りは `,` だけ（『{_compact(at)}』のように詰めて書く）"
                  if spaced else "")
               + (" ／番号の頭に 0 を付けない（" + "・".join(f"『{x}』→『{y}』" for x, y in zero) + "）" if zero else "")
               + f"（書けるのは all / first / last か、半角の数字の番号 1〜{n} と範囲 `1-2`（小さいほうから）を `,` で並べたもの・"
                 "番号の頭に 0 を付けない・空白を入れない）")
        fb = ("このパターンはその区間で鳴らさなかった（同じ区間の `all` のパターンが全小節を受け持つ）" if not got
              else "読めた小節だけで鳴らした")
        out.append(Issue(where, msg, fb))
    return out


def _overlap_issues(head: str, patterns: list, bars_of: Mapping[str, tuple[int, int]]) -> list[Issue]:
    """**同じパートの2つのパターンが、同じ区間の同じ小節を受け持つ**所見（2026-10-03・再レビュー 記I・仕様書 §11）。

    受け持つ小節は組み立て（`to_spec`）と同じ読み方: `at_bars` で指したパターンはその小節、`all`（と空）のパターンは同じ区間で
    指されていない残りの小節。だから「指したもの」と「`all`」は重ならず、重なるのは**指したものどうしが同じ小節**か、
    **`all` どうし**（残りの小節を両方が受け持つ）。ドラム（`drums.patterns`）も1つのパートとして同じに見る。
    設計に無い区間名・読めない `at_bars` は別の所見（`_section_issue`・`_at_issue`）なので、ここでは読めた小節だけで見る。
    ★以前は定義が無く（型の説明は「sections が重ならないように」と頼むだけ）、重なった小節では両方が鳴った。
    """
    explicit: dict[str, dict[int, list[int]]] = {}         # 区間 → 区間の中の小節（1始まり）→ そこを指したパターンの番号
    alls: dict[str, list[int]] = {}                        # 区間 → all のパターンの番号
    for k, sp in enumerate(patterns or [], 1):
        at = strip_ascii_ws(sp.at_bars).lower()
        for s in dict.fromkeys(getattr(sp, "sections", None) or []):
            key = resolve_section(s, bars_of)
            if key is None:
                continue
            a, b = bars_of[key]
            if at in ("", "all"):
                if k not in alls.setdefault(key, []):
                    alls[key].append(k)
                continue
            got, _bad, _rng = _read_at_bars(at, b - a + 1)
            for off in sorted(got or ()):
                if k not in (lst := explicit.setdefault(key, {}).setdefault(off, [])):
                    lst.append(k)
    out: list[Issue] = []
    for key, (a, b) in sorted(bars_of.items(), key=lambda kv: kv[1]):
        ex = explicit.get(key, {})
        clash: dict[tuple[int, ...], list[int]] = {}       # 重なったパターンの組 → 小節
        for off in range(1, b - a + 2):
            ks = ex.get(off) or (alls.get(key) or [])
            if len(ks) > 1:
                clash.setdefault(tuple(ks), []).append(off)
        for ks, offs in clash.items():
            both = "・".join(f"パターン{k}" for k in ks)
            out.append(Issue(head, f"{both}が区間『{key}』の{'・'.join(map(str, offs))}小節目を両方受け持つ——1つの小節を受け持つ"
                                   "パターンは1つだけ（1つのパターンにまとめるか、at_bars で小節を分ける。all のパターンは、"
                                   "ほかのパターンが at_bars で指した小節を譲る）", "重なった小節では両方が鳴る"))
    return out


def _section_issue(where: str, sections: list[str], bars_of: Mapping[str, tuple[int, int]]) -> list[Issue]:
    if not bars_of:
        return []
    if not sections:
        return [Issue(where, "sections が空（どの区間で鳴らすかが無い）", "このパターンは鳴らさなかった")]
    # ★区間名は設計の綴りと完全一致だけ（2026-10-02・判断6 本人「おすすめで」）。以前は大文字小文字・空白の違いを吸収して読んでいた。
    #   大文字小文字・空白だけの違いなら、設計の綴りを添える（読み替えない）
    return [Issue(where, f"区間『{s}』はこの曲に無い（区間: {'・'.join(bars_of)}）"
                  + (f"——設計の綴りは『{h}』（区間名は設計と1字も違わずに書く。大文字小文字・空白の違いも別の名前）"
                     if (h := section_hint(s, bars_of)) else ""), "その区間では鳴らさなかった")
            for s in sections if resolve_section(s, bars_of) is None]


def _pattern_format(where: str, pattern: str, slots: int, *, drum: bool) -> list[Issue]:
    """**打点のパターンの記法**（小節線 `|` と、音程パートは `X x . - _`・ドラムの行は `X x O o . -`）の所見: 2小節以上書いた・
    読めない文字・空（2026-09-29 に長さの検査から分けた）。

    `slots` は1小節の文字数（読めない文字は、組み立てが読む1小節ぶんの中だけ見る）。`drum` は**どちらの記法か**
    （2026-10-02・ず-9 から書ける字が違う。以前は所見の言い方だけに効いていた）。
    工程の所見は記法の型（`pattern_notation`）を通して出る。組み立ての入口の照合（`to_spec`）も `_issues` が同じここを呼ぶ。
    """
    vel, rest = (_VEL, _DRUM_REST) if drum else (_PITCHED_VEL, _PITCHED_REST)
    out: list[Issue] = []
    # ★**途中の空白は書けない**（2026-10-02 本人「おすすめで」・仕様書 §0-6・§8・§17 ず-22）。以前は行の中の空白をどこでも落として読んでいた。
    #   空白を除いた形を添え、下の所見（小節線・読めない文字）もその形で見る——空白を直したあとにまた別の所見で差し戻さないため。
    #   空白を除いた形は書き直し方を示すためだけで、組み立て（`_clean_grid`）は空白を落とさない（読み替えない）
    src = pattern
    # ★**前後に全角の空白・改行などを付けない**（2026-10-03・判断4 本人決定・仕様書 §0-6）。前後は半角の空白とタブだけ取り除く。
    #   取り除いた形を添え、下の所見はその形で見る（読み替えない——組み立ては読めない文字として読むので1字ずれる）
    if why := edge_ws_problem(pattern):
        src = trim_edge_ws(pattern)
        out.append(Issue(where, f"pattern{why}", "前後の空白の字を読めない文字として読んだ（打点がずれる）"))
    if any(c.isspace() for c in strip_ascii_ws(src)):
        src = _compact(src)
        out.append(Issue(where, f"pattern『{pattern}』の途中に空白がある——空白は記法の「区切り」の印で、打点のパターンには入れない"
                                f"（1小節ぶんを詰めて『{src}』のように書く）",
                         "空白の所は読めない文字として読んだ（打点も伸ばしも無い）"))
    # ★**縦棒が続いている**（`||x-x-x-x-|`・`x-x-x-x-||`・2026-10-02・洗い出し）。以前は両端の縦棒を何本でも取って1小節として読んでいた。
    #   縦棒は1小節の両端に1本ずつだけ（仕様書 §8）。下の所見（小節ぶん・読めない文字）は空の小節を除いた形で見る
    if len(_bar_segments(src)) != len(_bars_written(src)) or strip_ascii_ws(src) == "||":
        bars = _bars_written(src)
        out.append(Issue(where, f"pattern『{pattern}』で縦棒（`|`）が続いている——縦棒は1小節の両端に1本ずつだけ書ける"
                                + (f"（『{bars[0]}』のように縦棒なしで書く）" if len(bars) == 1 else ""),
                         "縦棒のあいだの空の小節を1小節目として読んだ（打点なし）"))
    segs = _bars_written(src)
    pat = segs[0] if segs else ""
    n_slots, _u = _grid_unit(pat, slots)
    hits = sum(1 for c in pat[:n_slots] if c in vel)
    # ★**`|`（小節線）で区切って2小節以上書いたもの**（2026-09-29）。以前は `|` を落として続けて読み、8文字の小節を
    #   2つ書くと16分の1小節として倍の密度で鳴った（16文字なので長さの検査も通り、所見が出なかった）
    if len(segs) > 1:
        later = sum(1 for s in segs[1:] for c in s if c in vel)
        out.append(Issue(where, f"pattern『{pattern}』は `|`（小節線）で区切って{len(segs)}小節ぶん書いてある。"
                                f"**1小節ぶんだけ**、`|` を書かずに（小節ごとに変えるならパターンを分けて at_bars で指す）",
                         "最初の `|` までの1小節だけ読んだ" + (f"（2小節目以降の打点{later}個は鳴らない）" if later else "")))
    # ★**休みだけのパターン（`________`・`--------`）は所見にしない**（2026-09-27）。「その小節は鳴らさない」という正当な
    #   書き方（キメ・ため・`at_bars` で特定の小節だけ止める）で、以前は所見にして直しの LLM 呼び出しを1回増やしていた。
    #   所見にするのは、**読めない文字**があるとき・何も書いていないときだけ（本当の書き誤り）
    # ★**打点があっても、読めない文字は所見にする**（2026-09-28）。組み立ては読めない文字を無音として読み、
    #   前の音もそこで止める（以前は `-` と同じに前の音を伸ばしていた＝当て推量）
    odd = sorted({c for c in pat[:n_slots] if c not in vel and c not in rest})
    if odd:
        fb = ("このパターンは鳴らさなかった" if not hits
              else "読めない文字の所は無音にした（前の音もそこで止まる）" if not drum
              else "読めない文字の所は鳴らさなかった")
        # ★**もう一方の記法の字は、書き直し方を添える**（2026-10-02・ず-9）。音程パートの `o` `O` と、ドラムの行の `_` は
        #   以前は黙って `x` `X`・`-` と同じに読んでいた。記録には伴奏のドラムの行の `_` が242行あった（音程パートの `o` は0行）
        if drum:
            legend = "ドラムの行で使える文字は 打点 X x O o .・休み - だけ"
            fix = pat.replace("_", "-") if "_" in odd else ""
            hint = f"。ドラムの休みは `-` だけ（`_` は書けない）——『{fix}』のように書く" if fix else ""
        else:
            legend = "音程パートで使える文字は 打点 X x .・前の音を伸ばす -・そこで切る _ だけ"
            fix = pat.replace("o", "x").replace("O", "X") if set(odd) & {"o", "O"} else ""
            hint = (f"。音程パートの打点は `X` `x` `.`（`o` `O` はドラムの行の字で、ここでは書けない）——"
                    f"`o` は `x`、`O` は `X` に書き直す（『{fix}』）") if fix else ""
        out.append(Issue(where, f"pattern『{pattern}』に読めない文字がある: {'・'.join(odd)}（{legend}）{hint}", fb))
    elif not pat:
        out.append(Issue(where, f"pattern『{pattern}』が空", "このパターンは鳴らさなかった"))
    return out


def pattern_notation(slots: int, *, drum: bool = False) -> Notation:
    """**打点のパターンの記法＝型1つ**（2026-09-29・`song_checks.Notation`）。1小節の文字数（`slots`）は曲の拍子で決まるので、
    曲ごとに作る。読み方は `_clean_grid`（1小節目）——組み立て（`_grid_steps`・`_grid_drums`）も同じもので読む。
    `drum` で音程パート（打点 `X x .`・`-` 伸ばす・`_` 切る）とドラムの行（打点 `X x O o .`・`-` 休み）を分ける（2026-10-02・ず-9。
    以前はどちらも同じ型で、`o` と `_` が両方で通った）。"""
    kind = "ドラムの行" if drum else "音程パート"
    return Notation(f"打点のパターン・{kind}（1小節{slots}文字・16分なら{slots * 2}文字）",
                    lambda p: [i.what for i in _pattern_format("", p, slots, drum=drum)], _clean_grid)


def _written(plan: Any, *, allow_drums: bool = True) -> list[tuple[str, str, bool]]:
    """伴奏・彩りが**打点のパターンを書いた所** (場所, パターン, ドラムか)。工程の `written` と組み立ての記録（`_issues`）で同じ並び。
    「ドラム: none」の曲のドラムは読まない（ドラムを書いたこと自体が所見になる）。"""
    out: list[tuple[str, str, bool]] = []
    for pi, p in enumerate(getattr(plan, "parts", None) or [], 1):
        for k, sp in enumerate(p.patterns or [], 1):
            out.append((f"パート{pi}（{p.role}）のパターン{k}（区間 {'/'.join(sp.sections or [])}）", sp.pattern, False))
    drums = getattr(plan, "drums", None)
    if allow_drums and drums is not None and drums.patterns:
        for k, dp in enumerate(drums.patterns, 1):
            for ln in dp.lines or []:
                out.append((f"ドラムのパターン{k}（区間 {'/'.join(dp.sections or [])}）の {ln.inst}", ln.pattern, True))
    return out


def _written_pitches(plan: Any) -> list[tuple[str, str]]:
    """伴奏・彩りが**音名を書いた所** (場所, 音名)——`pitches` の1音ずつと `riff` の1音ずつ（2026-10-01・判-A）。
    記法は国際式の音名（`song_llm.PITCH`）。工程の `written` と組み立ての入口の照合（`_issues`）で同じ並び。
    `use_riff` のパターンの `pitches` は鳴らない（リフが鳴る）ので読まない（以前の検査と同じ）。彩りが骨組みのリフを指すときの
    骨組みのリフは、骨組みの工程で読んである。"""
    out: list[tuple[str, str]] = []
    for pi, p in enumerate(getattr(plan, "parts", None) or [], 1):
        for k, sp in enumerate(p.patterns or [], 1):
            if sp.use_riff:
                continue
            head = f"パート{pi}（{p.role}）のパターン{k}（区間 {'/'.join(sp.sections or [])}）の pitches"
            out += [(f"{head} {j}音目", x) for j, x in enumerate(sp.pitches or [], 1)]
    out += [(f"リフ（riff）の{j}音目", x) for j, x in enumerate(getattr(plan, "riff", None) or [], 1)]
    return out


def _length_issue(where: str, pattern: str, slots: int, *, drum: bool) -> tuple[list[Issue], int]:
    """1小節の長さの検査。戻り値は (所見, 1小節の中の打点の数)。記法（`|`・読めない文字・空）は `_pattern_format`。"""
    vel = _VEL if drum else _PITCHED_VEL                   # 打点の字は記法ごと（2026-10-02・ず-9）
    # 途中の空白は記法の所見（`_pattern_format`）が書き直した形を添えて返すので、長さはその形（空白を除いた形）で数える
    #   （2026-10-02。空白を1文字と数えて「9文字」と言うと、空白を除いた書き直しと食い違う）。読み替えはしない
    segs = _bars_written(_compact(pattern))                # 縦棒が続いた所の空の小節は数えない（所見は `_pattern_format`・2026-10-02）
    pat = segs[0] if segs else ""
    n_slots, _u = _grid_unit(pat, slots)
    hits = sum(1 for c in pat[:n_slots] if c in vel)
    out: list[Issue] = []
    if len(pat) not in (slots, slots * 2):
        extra = sum(1 for c in pat[n_slots:] if c in vel)
        if len(pat) > n_slots:
            fb = (f"先頭の{n_slots}文字（1小節）だけ読んだ"
                  + (f"（後ろの打点{extra}個は鳴らない）" if extra else ""))
        else:
            # ★2026-09-28: 以前は音程パートの最後の音を小節の終わりまで伸ばしていた（組み立ての補正）。いまは書いた長さのまま
            fb = ("足りない所は鳴らさなかった" if drum
                  else "足りない所は無音として読んだ（最後の音も書いた長さのまま。小節の終わりまで伸ばさない）")
        out.append(Issue(where, f"pattern『{pattern}』が{len(pat)}文字。**1小節ぶんだけ**、{slots}文字"
                                f"（16分なら{slots * 2}文字）ちょうどに（小節ごとに変えるならパターンを分けて at_bars で指す）", fb))
    return out, hits


def _section_shift(section: str, bars_of: Mapping[str, tuple[int, int]], shift_of: Any | None) -> int:
    """その区間の移調量（半音）。区間名は時間軸の綴りへ照らしてから引く。引けない区間は 0（移さない）。
    組み立て（`to_spec`）と伴奏の検査（`_sounding_issues`）が同じここで引く——2本あると片方だけ直る。"""
    if shift_of is None:
        return 0
    try:
        return int(shift_of(resolve_section(section, bars_of) or section))
    except Exception:                                      # noqa: BLE001（移調量が引けない区間は移さない）
        return 0


def _sounding_issues(where: str, name: str, seq: list[str], sections: list[str],
                     bars_of: Mapping[str, tuple[int, int]], shift_of: Any | None) -> list[Issue]:
    """**書いた高さ（`pitches`・リフ）を区間の調へ移した鳴る高さが MIDI 0〜127 に収まるか**（2026-10-03・再レビュー）。

    書いた高さ（C 基準）が 0〜127 かは記法（`song_llm.PITCH`）が見るが、鳴るのは区間の移調量を足した高さで、
    組み立て（`song_audio.pitched_events`）は外れた音を鳴らさない（張り付けない）。以前は移調前の高さしか見ておらず、
    移すと 127 を超える音が所見なしに黙って鳴らなかった。所見にして伴奏・彩りの工程に直させる。
    読めない音名・無い区間名はほかの所見（記法・`_section_issue`）なので、ここでは見ない。
    """
    if shift_of is None or not seq:
        return []
    out: list[Issue] = []
    for s in sections or []:
        if resolve_section(s, bars_of) is None:
            continue
        sh = _section_shift(s, bars_of, shift_of)
        bad = [(j, x, k + sh) for j, x in enumerate(seq, 1)
               if (k := PITCH.read(x)) is not None and not 0 <= k + sh <= 127]
        if bad:
            out.append(Issue(where, f"区間『{s}』では移調量 {sh:+d} 半音を足して鳴るので、{name} の "
                                    + "・".join(f"{j}音目 {x}（鳴る高さ {v}）" for j, x, v in bad)
                                    + " が MIDI で鳴らせる高さ（0〜127＝C-1〜G9）を超える。書く高さ（C 基準）を下げる"
                                      "（オクターブを下げる など）",
                             "その音は鳴らさなかった"))
    return out


def _pattern_issues(where: str, sp: SectionPattern, slots: int, bars_of: Mapping[str, tuple[int, int]],
                    riff: list[str], shift_of: Any | None = None) -> list[Issue]:
    out = _section_issue(where, sp.sections, bars_of)
    li, hits = _length_issue(where, sp.pattern, slots, drum=False)
    out += li
    riff_on = bool(sp.use_riff)
    if riff_on and not riff:
        # ★2026-09-28: 以前はリフの代わりに pitches（無ければ tones）で鳴らしていた（組み立ての代用）。いまは鳴らさない
        out.append(Issue(where, "use_riff: true なのに、指す先のリフ（riff）が空",
                         "このパターンは鳴らさなかった（pitches・tones で代わりに鳴らすことはしない）"))
        out += _at_issue(where, sp.at_bars, sp.sections, bars_of)
        return out
    seq = list(riff) if riff_on else list(sp.pitches or [])
    name = "リフ（riff）" if riff_on else "pitches"
    if seq:
        # ※音名が読めるか・鳴らせる高さ（MIDI 0〜127）かは記法（`song_llm.PITCH`・`_written_pitches`）が見る（2026-10-01）。
        #   以前はここで「読めない階名」「MIDI 0〜127 の外」を見ていた（2026-09-28 までは組み立てが 0・127 に張り付けて鳴らしていた）
        if hits and len(seq) != hits:
            fb = (f"余りの{len(seq) - hits}音は鳴らさなかった" if len(seq) > hits
                  else f"足りない{hits - len(seq)}個の打点は鳴らさなかった")
            out.append(Issue(where, f"{name} が{len(seq)}音なのに、pattern の1小節ぶんの打点が{hits}個。数をそろえる", fb))
        out += _sounding_issues(where, name, seq, sp.sections, bars_of, shift_of)   # 移したあとの鳴る高さ（2026-10-03）
    elif strip_ascii_ws(sp.tones) not in TONES:
        # ★2026-09-28: 以前は根音（root）として鳴らしていた（組み立ての当て推量）。いまは鳴らさない
        out.append(Issue(where, f"tones『{sp.tones}』は使えない（{' / '.join(TONES)}）", "このパターンは鳴らさなかった"))
    out += _at_issue(where, sp.at_bars, sp.sections, bars_of)
    return out


def _issues(plan: Any, *, bars_of: Mapping[str, tuple[int, int]], instruments: list | None = None,
            allow_drums: bool = True, slots: int = 8, base_riff: list[str] | None = None,
            drum_kit: str = "", notation: bool = True, need_drums: bool = False,
            shift_of: Any | None = None) -> list[Issue]:
    """伴奏（`ArrangePlan`）・彩り（`ColourPlan`）の検査。**所見を返すだけで、直さない**。

    `shift_of`（区間 → 移調量・組み立て `to_spec` に渡すのと同じもの）を渡すと、書いた高さ（`pitches`・リフ）を区間の調へ移した
    **鳴る高さ**が 0〜127 に収まるかも見る（`_sounding_issues`・2026-10-03）。渡さなければ見ない（移調量が分からない呼び方）。

    `base_riff` は彩りの検査のとき骨組みのリフ（彩りが `riff` を空にして `use_riff` を書くと、骨組みのリフを指す）。
    `need_drums` は**編成がドラムのキットを決めたか**（2026-09-30・出-8）。真なら、伴奏が `drums` を書かなかったことを所見にする
    （ドラムの持ち主は伴奏——グルーヴのドラムは土台の指定で鳴らない）。組み立て（`to_spec`）の入口では見ない
    （書かれていないドラムは組み立てが何かを決める所ではなく、鳴らないだけ）。
    `drum_kit` は編成のドラムの種類。渡すと、打楽器が**そのキットにあるか**も見る（2026-09-29・`timbre_map.kit_drums`。
    キットの中身が読めないとき・キット名が読めないときは見ない——読めないキット名は編成の工程の所見）。
    `notation` が偽なら打点のパターンの記法（`_pattern_format`）を除く——工程（`ArrangeStep`）は記法を `written` で宣言し、
    親の照合（`song_checks.notation_findings`）が出すので、`check` はそれ以外だけ（2026-09-29）。組み立ての記録は全部を使う。
    """
    out: list[Issue] = []
    if notation:                                           # 記法が先（工程の所見の並び＝親の `findings` と同じ）
        out += [iss for w, pat, drum in _written(plan, allow_drums=allow_drums)
                for iss in _pattern_format(w, pat, slots, drum=drum)]
        out += [Issue(w, why, "その音は鳴らさなかった")      # 音名（`pitches`・`riff`）の記法（2026-10-01）
                for w, x in _written_pitches(plan) for why in PITCH.problems(x)]
    bars_of = _song_sections(bars_of)                      # 曲頭の空き小節（lead_in）は伴奏の区間ではない
    parts = list(getattr(plan, "parts", None) or [])
    riff = list(getattr(plan, "riff", None) or []) or list(base_riff or [])
    n_inst = len(instruments or [])
    # ※「全パートが同じ楽器番号」は 2026-09-28 に形式の検査から外した（音楽の判断——ピアノだけの編曲なら正しい）。
    #   点検役の材料として `song_band.inspect_arrange` が渡す
    for pi, p in enumerate(parts, 1):
        head = f"パート{pi}（{p.role}）"
        if n_inst and not 1 <= p.instrument <= n_inst:
            out.append(Issue(head, f"楽器番号 {p.instrument} は編成に無い（1〜{n_inst}）", "このパートは鳴らさなかった"))
        # ★**音程パートの role に `drums` は書けない**（2026-10-02・洗い出し）。書き出し（`midi_out`）は role が `drums` のパートを
        #   打楽器として扱うので、音程パートの音が1音も鳴らなかった（所見なし）。ドラムは `drums` に書く
        if (p.role or "").strip() == "drums":
            out.append(Issue(head, "role『drums』は打楽器のパートの印なので、音程パートには書けない——"
                                   "役割の名前を変える（ドラムは drums に書く）", "このパートは1音も鳴らない（打楽器として扱われる）"))
        for k, sp in enumerate(p.patterns or [], 1):
            out += _pattern_issues(f"{head}のパターン{k}（区間 {'/'.join(sp.sections or [])}）", sp, slots, bars_of, riff,
                                   shift_of)
        out += _overlap_issues(head, p.patterns, bars_of)   # 同じ区間の同じ小節を2つのパターンが受け持たない（記I・2026-10-03）
    drums = getattr(plan, "drums", None)
    if need_drums and allow_drums and not (drums is not None and any(dp.lines for dp in (drums.patterns or []))):
        # ★2026-09-30（出-8）: 以前は書かなくても所見にならず、編成が standard と決めた曲でドラムが1音も鳴らなかった
        out.append(Issue("ドラム", f"編成がドラムを使う（キット『{drum_kit or 'standard'}』）と決めたのに drums が無い。"
                                  "グルーヴのドラムは土台の指定で、それ自体は鳴らない——**鳴るのは伴奏の drums に書いたものだけ**。"
                                  "drums を書く（グルーヴの刻みを写す。静かにしたいなら鳴らす区間を絞る・金物やリムだけにする）",
                         "ドラムは1音も鳴らない"))
    if drums is not None and drums.patterns:
        if not allow_drums:
            out.append(Issue("ドラム", "編成が「ドラム: none」（ドラムを使わない）と決めたのに drums を書いた。"
                                      "drums を消し（drop_drums）、ノリはベースや和音の刻みで出す", "ドラムを鳴らさなかった"))
        else:
            have = kit_drums(drum_kit) if drum_kit else None
            menu = " / ".join(have) if have is not None else " / ".join(DRUM_NAMES)
            for k, dp in enumerate(drums.patterns, 1):
                where = f"ドラムのパターン{k}（区間 {'/'.join(dp.sections or [])}）"
                out += _section_issue(where, dp.sections, bars_of)
                for ln in dp.lines or []:
                    w = f"{where}の {ln.inst}" if strip_ascii_ws(ln.inst or "") else f"{where}の打楽器名が空の行"
                    # ★**空の名前（半角の空白だけも）も名前の検査に通す**（2026-10-03・再レビュー）。以前は空なら検査を飛ばし、
                    #   組み立て（`_grid_drums`）はその行を所見なしに黙って鳴らさなかった。仕様書 §15-2「GM の正式な名前だけ」
                    canon = drum_canon(ln.inst or "")
                    if canon is None:
                        # ★2026-09-28: 以前は部分一致かシェイカーの「近い名前」で鳴らしていた。いまは鳴らさない
                        # ★2026-10-02: 名前は GM の正式な名前だけ（別名は読まない）。近い GM の名前を添えて書き直させる
                        out.append(Issue(w, f"{drum_name_problem(ln.inst)}（書ける名前: {menu}）",
                                         "この打楽器は鳴らさなかった（近い名前の打楽器で代わりに鳴らすことはしない）"))
                    elif canon and have is not None and canon not in have:
                        # ★2026-09-29: 以前はキットの別の番号へ運ぶか、番号のまま（オーケストラ用キットではティンパニ）鳴らしていた
                        out.append(Issue(w, f"打楽器『{ln.inst}』は編成のキット『{drum_kit}』に無い"
                                            f"（このキットで鳴らせる打楽器: {menu or 'なし'}）",
                                         "この打楽器は鳴らさなかった（キットの別の音や近い打楽器で代わりに鳴らすことはしない）"))
                    elif hat := hat_line_problem(canon or "", ln.pattern):
                        # ★2026-10-02: 開いたハイハットは open_hi_hat の行に書く（以前の判-J は hihat の行の `o` を開いて鳴らした）
                        out.append(Issue(w, hat, "`o` の打点は鳴らさなかった"))
                    out += _length_issue(w, ln.pattern, slots, drum=True)[0]
                out += _at_issue(where, dp.at_bars, dp.sections, bars_of)
            out += _overlap_issues("ドラム", drums.patterns, bars_of)   # ドラムも1つのパートとして（記I・2026-10-03）
    return out


def arrange_findings(plan: Any, *, bars_of: Mapping[str, tuple[int, int]], instruments: list | None = None,
                     allow_drums: bool = True, slots: int = 8, base_riff: list[str] | None = None,
                     drum_kit: str = "", need_drums: bool = False, shift_of: Any | None = None) -> list[str]:
    """伴奏・彩りの検査（所見の文だけ）。`bars_of` は**時間軸の実際の小節**（設計の小節数と違うことがある・時間-1）。
    記法（打点のパターン）が先で、`ArrangeStep.findings`（記法＝`written`＋`check`）と同じ並び。
    `shift_of`（区間 → 移調量）を渡すと、移したあとの鳴る高さも見る（`_sounding_issues`・2026-10-03）。"""
    return [i.finding() for i in _issues(plan, bars_of=bars_of, instruments=instruments,
                                         allow_drums=allow_drums, slots=slots, base_riff=base_riff,
                                         drum_kit=drum_kit, need_drums=need_drums, shift_of=shift_of)]


def _inst_label(no: int, instruments: list | None) -> str:
    """楽器番号 → `楽器1（Grand Piano）`（編成の一覧が無ければ番号だけ）。"""
    if instruments and 1 <= no <= len(instruments):
        return f"楽器{no}（{gm_name(instruments[no - 1][1])}）"
    return f"楽器{no}"


def _lead_row(plan: Any, ins: Any, instruments: list | None) -> list[str]:
    """**主旋律の楽器の行**（編成の `lead`・`lead_sections` と伴奏の `lead_gain`・2026-10-03・判-25）。
    編成が主旋律の楽器を決めたか、伴奏が `lead_gain` を書いたときだけ（歌もので重ねない曲の表は以前のまま）。"""
    lead = getattr(ins, "lead", None)
    gain = getattr(plan, "lead_gain", None)
    if lead is None and gain is None:
        return []
    secs = list(getattr(ins, "lead_sections", None) or [])
    who = _inst_label(lead, instruments) if lead is not None else "なし（編成は主旋律の楽器を決めていない）"
    where = "・".join(secs) if secs else "主旋律のある区間の全部"
    vol = f"lead_gain={gain:g}" if gain is not None else "lead_gain なし（書いていない）"
    return [f"■主旋律の楽器（編成の lead）: {who}／弾く区間（lead_sections）: {where}／音量 {vol}"
            "（音は旋律の工程の音を機械がそのまま置く＝パートの行には出ない。音量はパートの gain と同じ物差し）"]


def pattern_table(plan: Any, instruments: list | None = None, *, base_riff: list[str] | None = None,
                  ins: Any = None) -> str:
    """伴奏・彩りの**パターンの表**（伴奏の点検役の材料・彩りに見せる骨組み）。**行を削らず、字数でも切らない**（2026-09-30・出-6）。

    ★以前の表には**リフ（`riff`・`use_riff`）・音量（`gain`）・音域（`register_octave`）の欄が無く**、点検役は
      「共通リフに旋律の指定が無く、ただの和音弾き」と読み違え、その所見が彩りへ渡り、彩りが同じピアノにもう1本リフを足した
      （イントロ・間奏・アウトロで別のリズムのリフが2本鳴った）。彩りに見せる要約は同じ欄を落としたうえ1200字で切れ、
      パッドと弦の行が丸ごと見えず、弦も二重になった。**同じ楽器にリフ・旋律が重なるかは、この表の楽器名と印で分かる**
      （判定は点検役・彩りの LLM）。
    ★`ins`（編成）を渡すと**主旋律の楽器・弾く区間・音量**（`lead`・`lead_sections`・`lead_gain`）の行も出す
      （2026-10-03・判-25）。インストと主旋律を楽器で重ねる歌もので「主旋律が伴奏の上で聞こえるか」を見られるのは伴奏の点検役だけで、
      以前はその材料が表に無かった。彩りに見せる骨組みには渡さない（彩りは `lead_gain` を持たない）。
    """
    riff = list(getattr(plan, "riff", None) or [])
    lines = [("リフ（riff）: " + "・".join(riff) + "（`リフ` の行で、この音形がそのまま鳴る）") if riff else
             ("リフ（riff）: なし" + (f"（`リフ` の行は骨組みのリフ {'・'.join(base_riff)} を指す）" if base_riff else ""))]
    if ins is not None:
        lines += _lead_row(plan, ins, instruments)
    melodic: list[str] = []
    for pi, p in enumerate(getattr(plan, "parts", None) or [], 1):
        inst = _inst_label(p.instrument, instruments)
        lines.append(f"■パート{pi} {p.role}／{inst}／音量 gain={p.gain:g}"
                     f"／音域 register_octave={p.register_octave:+d}（tones の音にだけ効く）"
                     + ("／ベース（bass: true）" if p.bass else ""))
        for sp in p.patterns or []:
            at, secs = (sp.at_bars or "all"), "/".join(sp.sections or [])
            if sp.use_riff:
                what = "**リフ**（riff の音形をそのまま）"
            elif sp.pitches:
                what = f"**旋律**={'・'.join(sp.pitches)}"
            else:
                what = f"tones={sp.tones}"
            if sp.use_riff or sp.pitches:
                melodic.append(f"  {inst} {'リフ' if sp.use_riff else '旋律'}: 区間={secs} 小節={at}")
            lines.append(f"   {sp.pattern:<16} {what}  小節={at}  区間={secs}")
    if hasattr(plan, "drums"):
        drums = plan.drums
        if drums is not None and drums.patterns:
            lines.append(f"■ドラム／音量 gain={drums.gain:g}")
            for dp in drums.patterns:
                for ln in dp.lines or []:
                    lines.append(f"   drum:{(ln.inst or ''):<12} {ln.pattern:<16} 小節={dp.at_bars or 'all'}  "
                                 f"区間={'/'.join(dp.sections or [])}")
        else:
            lines.append("■ドラム: なし（drums を書いていない）")
    if melodic:
        lines += ["【自分の旋律（リフ・pitches）を鳴らしている所】"] + melodic
    return "\n".join(lines)


def merge_colour(plan: ArrangePlan, colour: "ColourPlan") -> ArrangePlan:
    """骨組み（伴奏）と彩りを**1つの編曲案に合わせた写し**を返す（元の2つは変えない・2026-09-27）。

    ★**リフはそれぞれのパターンの持ち物にする**（目録の 伴奏-5）。以前は合わせるときに
      `merged.riff = colour.riff or merged.riff` としていたので、**彩りのリフが伴奏のリフを上書き**し、
      伴奏が `use_riff` で指した所まで彩りの音形で鳴った（伴奏の書いたリフは消えていた）。
      ここでは**彩りのパートの `use_riff` を彩りのリフで先に解いて `pitches` にする**
      （`to_spec` が `use_riff` を解くのと同じ訳し方＝組み立て）。伴奏のパートは伴奏の `riff` を指したまま。
      彩りのリフが空で `use_riff` を書いたパートは、これまでどおり伴奏のリフを指す。
    """
    merged = plan.model_copy(deep=True)
    for cp in colour.parts:
        cp = cp.model_copy(deep=True)
        if colour.riff:
            cp.patterns = [sp.model_copy(update={"pitches": list(colour.riff), "use_riff": False})
                           if sp.use_riff else sp for sp in cp.patterns]
        merged.parts.append(cp)
    # `merged.riff` は伴奏のリフのまま（伴奏がリフを書かずに `use_riff` を指した所へ、彩りの音形を流し込まない）
    return merged


def to_spec(plan: ArrangePlan, bars_of: dict[str, tuple[int, int]], slots: int = 8,
            instruments: list[tuple[int, int]] | None = None, allow_drums: bool = True,
            warns: list[str] | None = None,
            intensity_of: dict[str, int] | None = None,
            shift_of: "Any | None" = None, drum_kit: str = "", lead: int | None = None) -> ArrangementSpec:
    """LLM の編曲案 → ArrangementSpec。セクション名を小節範囲（bar_filter）へ変換する。

    `lead` は編成の主旋律の楽器の番号（`Instrumentation.lead`・1始まり）。使われなかった楽器の所見で「使った」に数える
    （主旋律の楽器は伴奏のパートに書かないので、渡さないとインストで誤報する・2026-10-03 組G3）。

    `instruments` は**編成工程で決まった GM 番号の並び**。PartPlan.instrument はその何番目かを指す。
    ★**無ければ組み立てない**（2026-09-29・`ValueError`）。以前は楽器の一覧が無いと役割名から音色を推して鳴らしていた
      （`safe_timbre`・最後は pluck）。編成が作れなければ曲づくりを止める（`song_band.run_instrumentation` は編成の失敗を上げる）。

    `drum_kit` は編成のドラムの種類。そのキットに無い打楽器は検査の所見（2026-09-29）。

    ★**入口で伴奏の形式の検査（`_issues`）と同じ判定をし、所見があれば `song_fix.AssemblyError` で止める**（2026-09-30）。
      曲づくりでは直らなければ伴奏・彩りの工程で止まる（`StepUnfixed`）ので、ここへは検査を通った形しか来ない。
      「ドラムなし」の曲のドラム（`allow_drums=False`）も同じ——以前はここで機械的に落としていた
      （2026-08-22、神楽の曲に洋楽のドラムが入ったため。プロンプトで頼むだけでは守られなかった）。

    `intensity_of` は**区間 → 強さ(1〜5)**。渡すと**区間ごとに音量差を付ける**。
    1つの PartPlan は既にセクションごとの弾き分けを持つので、
    **同じ強さのセクションだけをまとめて**別々の PartSpec にすれば区間別の音量になる。
    """
    def db_of(kind: str) -> float:
        if not intensity_of:
            return 0.0
        key = resolve_section(kind, intensity_of)          # 設計の綴りと完全一致だけ（2026-10-02。検査が保証している）
        return intensity_db(intensity_of.get(key)) if key else 0.0

    def by_level(sections: list[str]) -> dict[float, list[str]]:
        """セクションを**強さ（dB）ごとに束ねる**（同じ音量のものだけ小節範囲を繋ぐ）。"""
        out: dict[float, list[str]] = {}
        for s in sections:
            out.setdefault(db_of(s), []).append(s)
        return out

    def shift_in(section: str) -> int:
        """その区間の移調量（半音）。伴奏の検査（`_sounding_issues`）と同じ `_section_shift` で引く。"""
        return _section_shift(section, bars_of, shift_of)

    def by_level_key(sections: list[str], keyed: bool) -> dict[tuple[float, int], list[str]]:
        """セクションを**強さ（dB）と移調量ごとに束ねる**（2026-09-29）。

        ★`pitches`・リフ（書いた高さ）は絶対音高なので、`degree` と違って**あとからコードに追従しない**——
          区間の調へここで移す（2026-09-17）。以前は**パターンの最初の区間の移調量**を全区間に掛けていたので、
          転調をまたぐパターンは転調のあとも元の調で鳴った。書いたものを**区間ごとにその区間の調へ訳す**（組み立て）。
          高さを書いていないパターン（`tones`）はコードの側が区間ごとに移るので、移調量で分けない。
        """
        out: dict[tuple[float, int], list[str]] = {}
        for s in sections:
            out.setdefault((db_of(s), shift_in(s) if keyed else 0), []).append(s)
        return out

    def scaled(base: float, db: float) -> float:
        # ★**下げた値のまま**（2026-10-03・再レビュー 記L・仕様書 §15）。以前は 0.1〜1.2 に丸め、gain 0.1 のパートは強さ1の区間でも
        #   下がらなかった（目録に無い黙った丸め）。`db` は 0 以下（`_INTENSITY_DB`）なので、書いた gain（0.1〜1.2）を超えない
        return base * (10 ** (db / 20))

    def rng(sections: list[str]) -> str:
        spans = sorted(bars_of[k] for k in
                       (resolve_section(s, bars_of) for s in sections) if k)
        if not spans:
            return "all"
        merged = [list(spans[0])]                          # 隣接する範囲だけ繋ぐ（飛びは別範囲のまま残す）
        for a, b in spans[1:]:
            if a <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return ",".join(f"{a}-{b}" for a, b in merged)

    def bars_in(section: str, at_bars: str) -> set[int]:
        """区間の中の**何小節目か**を、曲全体の小節番号へ直す（1始まり・両端を含む）。"""
        key = resolve_section(section, bars_of)            # 設計の綴りと完全一致だけ（2026-10-02。検査が保証している）
        if key is None:
            return set()
        a, b = bars_of[key]
        got, _bad, _rng = _read_at_bars(at_bars, b - a + 1)
        if got is None:
            return set(range(a, b + 1))
        # ★**読めた小節だけ**（2026-09-28・組み立ては補正しない）。1つも読めなければ**どの小節でも鳴らさない**。
        #   以前は解釈できなければ区間の全小節で鳴らし（当て推量）、同じ区間の `all` のパターンから小節を奪っていた。
        #   読めない・範囲外は検査（`_issues`）が所見にして編曲へ返し、残ったときはこの動きを warns に書く（伴奏-6）
        return {a + off - 1 for off in got}

    def warn(msg: str) -> None:
        # 同じ所見を重ねない（※2026-09-30 から、彩りを合わせた組み直しは別のリストに集める＝`song_band.run_colour`・判-17②）
        if warns is not None and msg not in warns:
            warns.append(msg)

    # ★**楽器が要るのは音程のパートがあるときだけ**（2026-10-01・判-E）。ドラムだけの案（途中の試聴でグルーヴのドラムを鳴らす
    #   `song_preview.groove_drums`）は編成の前に組むので、楽器の一覧が無い
    if plan.parts and not instruments:
        raise ValueError("編成（どの楽器で演奏するか）が無いので、伴奏を組み立てられない"
                         "（楽器を役割名から推して鳴らすことはしない）")
    bars_of = _song_sections(bars_of)                      # 曲頭の空き小節（lead_in）は伴奏の区間ではない（見回りにも出さない）
    # ★**検査が保証している形しか来ない**（2026-09-30）。伴奏・彩りの形式の所見は工程（`ArrangeStep`）が差分で直させ、
    #   直らなければその工程で曲づくりを止める（`song_fix.StepUnfixed`）。だから組み立ての「鳴らさない」分岐
    #   （編成に無い楽器番号・空のリフ・読めない at_bars・知らない打楽器…＝`Issue.fallback`）へは届かない。
    #   来たら検査と組み立てのずれ＝不具合なので止める。以前（2026-09-27〜）は所見と「組み立てでは鳴らさなかった」を
    #   warns に書いて進み、その前は黙って直していた（目録の 伴奏-2/3/6/7/8/9・打-1/3）
    if left := _issues(plan, bars_of=bars_of, instruments=instruments, allow_drums=allow_drums, slots=slots,
                       drum_kit=drum_kit, shift_of=shift_of):
        from song_fix import AssemblyError                 # noqa: E402（循環を避けてここで）
        raise AssemblyError("伴奏の組み立てに、検査を通ったはずの形でないものが来た: "
                            + "／".join(i.finding() for i in left[:8]) + (f" ほか{len(left) - 8}件" if len(left) > 8 else ""))

    # 1つの PartPlan（楽器1本）を、**セクションごとに別々の PartSpec** へ展開する。
    # 本番の PartSpec は「1小節パターン × 小節範囲」しか持てないため、弾き分けはこの形でしか表せない
    # （ヒット曲の実測では 75小節に 19種類のリズム型。1楽器1パターンでは届かない）。
    parts = []
    used: set[int] = set()                                 # 編成のどれが実際に使われたか
    # ★★**全パートが同じ楽器番号でも、割り当て直さない**（2026-09-27・目録の 伴奏-2）★★
    #   2026-09-13 に『この衝動』で全パートが 1番（ベース）を指し、曲全体がベース音色になったので、
    #   ここで役割の順に割り当て直していた（プランそのものを書き換えていた）。いまは書いたとおりに鳴らし、
    #   点検役の材料（`song_band.inspect_arrange`・2026-09-28）にする（ソロ楽器の編曲なら正しいので、形式としては直させない）。
    for p in plan.parts:
        # **楽器は編成工程で決まっている**。番号で引いて GM 音色名に直す
        # （音色名を自由記述させると、鳴らせない名前や曲に合わない音が混ざる）。
        # ※編成に無い番号は上の検査が止める（2026-09-30。以前はそのパートを鳴らさずに進み、2026-09-27 より前は役割名から音色を推した）
        sel = instruments[p.instrument - 1]
        # 音色は **gm<bank>-<番号>** で持ち回る（音源固有の追加音色も指せるように）
        timbre = f"gm{sel[0]}-{sel[1]}"
        used.add(p.instrument)
        # **特定の小節を指したパターンが優先**（`all` は残りの小節を受け持つ）。
        # こうしないと「区間全体に x---x--- ／ 最後の小節だけ x-------」を書いたとき
        # 最後の小節で両方が鳴って濁る。上書きの意味で書けるようにする。
        # ★**譲る小節は、解決した区間名で持つ**（2026-09-28）。書いたままの名前で持つと、`Intro` の指名と
        #   `intro` の `all` が別の区間として扱われ、指名した小節で両方が鳴る（区間名の表記ゆれ・2026-09-06 と同じ型）
        claimed: dict[str, set[int]] = {}
        for sp in p.patterns:
            if strip_ascii_ws(sp.at_bars).lower() in ("", "all"):
                continue
            for s in sp.sections:
                claimed.setdefault(resolve_section(s, bars_of) or s, set()).update(bars_in(s, sp.at_bars))

        for sp in p.patterns:
            # ★**リフは1か所（`plan.riff`）に定義して、ここから指す**（2026-09-17）。
            #   書き写す形にすると、どこかで音形が変わって**リフでなくなる**——
            #   実際、区間をまたぐ同一性を持てる欄が無かったので、
            #   「イントロ・間奏・最後で同じ句」と頼んでも**間奏の1か所しか入らなかった**。
            # ※`use_riff` なのにリフが空、は上の検査が止める（2026-09-30。2026-09-28 からは鳴らさずに進み、その前は pitches・tones で代わりに鳴らした）
            if sp.use_riff:
                _pit = plan.riff
            else:
                _pit = sp.pitches
            at = strip_ascii_ws(sp.at_bars).lower()
            is_all = at in ("", "all")
            for (db, _sh), secs in by_level_key(sp.sections, bool(_pit)).items():   # 強さ・調が違えば別パートに分ける
                steps = _grid_steps(sp.pattern, sp.tones, slots, _pit, _sh)
                if not steps:
                    continue
                bars: set[int] = set()
                for s in secs:
                    got = bars_in(s, at)
                    if is_all:
                        got -= claimed.get(resolve_section(s, bars_of) or s, set())   # 指名された小節は譲る
                    bars |= got
                if not bars:
                    continue
                parts.append(PartSpec(role=p.role, bass=p.bass, timbre=timbre, gain=scaled(p.gain, db),
                                      register_octave=p.register_octave,
                                      bar_filter=_to_filter(bars), steps=steps))
    n_pitched = len(parts)                                 # 音程パートの数（ドラムを足す前に数える）
    # ※編成が「ドラムなし」なのに書かれたドラムは上の検査が止める（2026-09-30・打-1。以前は鳴らさずに記録して進んだ）
    if plan.drums:
        have = kit_drums(drum_kit) if drum_kit else None    # キットで鳴らせる打楽器（None＝絞れない）
        d_claimed: dict[str, set[int]] = {}
        for dp in plan.drums.patterns:
            if strip_ascii_ws(dp.at_bars).lower() in ("", "all"):
                continue
            for s in dp.sections:
                d_claimed.setdefault(resolve_section(s, bars_of) or s, set()).update(bars_in(s, dp.at_bars))
        for dp in plan.drums.patterns:
            hits = _grid_drums(dp, slots, have)
            if not hits:
                continue
            at = strip_ascii_ws(dp.at_bars).lower()
            is_all = at in ("", "all")
            for db, secs in by_level(dp.sections).items():   # ドラムも区間ごとに強弱を付ける
                bars: set[int] = set()
                for s in secs:
                    got = bars_in(s, at)
                    if is_all:
                        got -= d_claimed.get(resolve_section(s, bars_of) or s, set())
                    bars |= got
                if not bars:
                    continue
                # ★決まった形のフィルで差し替えることはしない（2026-09-27・打-2）。書いたとおりに鳴らす
                parts.append(PartSpec(role="drums", bass=False, gain=scaled(plan.drums.gain, db),
                                      bar_filter=_to_filter(bars), drum_steps=hits))
    if warns is not None:
        # ★**編成の主旋律の楽器（`lead`）も「使った」に数える**（2026-10-03・組G3）。主旋律の楽器は伴奏のパートに書かず
        #   （音は旋律の工程の音を楽譜の `lead` に置く）、以前はここで数えず、インストで毎回「使われなかった」と誤報していた
        if lead:
            used.add(int(lead))
        if unused := [i for i in range(1, len(instruments) + 1) if i not in used]:
            warn("編成で選んだのに使われなかった楽器: "
                 + ", ".join(gm_name(instruments[i - 1][1]) for i in unused))

    # ★**区間に伴奏のパートが1つも無い**ことだけを事実として書く（2026-10-03・組G3）。区間の**全部の小節**で見る。
    #   以前は区間の1小節目と最後の小節だけで見ていて、さらに「1小節10音以下で低音なし」を**薄すぎる**と出し
    #   「曲が始まった感じにならない」と断じていた（閾値と断じる文）。薄い・厚いは点検役が【区間ごとに鳴る楽器】の表
    #   （`song_band.arrange_facts`・実際に鳴る小節で数える）を読んで判断する
    if warns is not None and bars_of:
        _total = max(b for _a, b in bars_of.values())
        for kind, (a, b) in bars_of.items():
            if not any(_bar_match_v2(x - 1, _total, p.bar_filter) for p in parts for x in range(a, b + 1)):
                warn(f"区間『{kind}』に**伴奏が1つも無い**")

    # ★**伴奏が丸ごと消えたら、ここで止める**（2026-09-06）。
    #   編曲がパートを書いたのに1つも残らないのは、**芸術的な選択ではなく故障**
    #   ——区間名が解決できない類のバグでしか起きない。これまでは「区間に伴奏が無い」という
    #   所見にして**関門の LLM の判断へ回していた**ため、通ってしまい
    #   **2秒の無音を伴奏として同梱した曲が出荷された**（『青空の境界線』2026-09-06）。
    #   薄い・無音の区間は表現としてありうるが、**全区間ゼロ**はありえない。
    #   ★**ドラムの有無で判定しない**——ドラムだけ生き残ると `parts` は空でなくなるが、
    #   曲としては「打楽器しか鳴らない」状態で、やはり故障。**音程パートの数**で見る。
    if plan.parts and not n_pitched:
        known = ", ".join(sorted(bars_of)) or "(空)"
        used = sorted({s for p in plan.parts for pat in p.patterns for s in (pat.sections or [])})
        raise ValueError(
            f"編曲は {len(plan.parts)} パート書いたのに、楽譜に1つも残らなかった。"
            f"区間名が解決できていない・楽器番号が編成に無い可能性が高い／編曲が指す区間: {', '.join(used) or '(無し)'}"
            f"／設計の区間: {known}")
    return ArrangementSpec(parts=parts, drum_kit=drum_kit or "")


def vocal_range_text(vocal_range: dict[str, tuple[int, int]] | None, shift_of=None) -> str:
    """歌の音域（区間ごと＋曲全体の最低〜最高）と、区間ごとの `tones` の根音の範囲（`register_octave` 0）。
    伴奏・彩りの依頼文と伴奏の点検役の材料で同じ書き方（2026-09-30・出-9）。

    ★**どちらも C 基準（移調前）・国際式の音名**（2026-10-01）。歌の音域は旋律の書いた高さ（移調前）、根音の範囲は
      `root_range_c`（鳴る高さの C4〜B4 から区間の移調量を戻す）。`shift_of`（区間 → 移調量・`song_place.section_keys`＝旋律・コード・
      `pitches` を移すのと同じもの）が無ければ移調量 0 として出す（カバーも 2026-10-01 から推定した調の `shift_of` を渡す・判-C）。
      以前は根音の範囲だけ鳴る高さ（C4〜B4 決め打ち）で、C 以外の調の曲では歌の音域と物差しが移調量の分だけ食い違っていた。
    """
    if not vocal_range:
        return ""
    lo = min(a for a, _b in vocal_range.values())
    hi = max(b for _a, b in vocal_range.values())

    def roots(kind: str) -> str:
        r0, r1 = root_range_c(int(shift_of(kind)) if shift_of is not None else 0)
        return f"{_voice_note(r0)}〜{_voice_note(r1)}"
    return (f"・**曲全体**: 歌は **{_voice_note(lo)}〜{_voice_note(hi)}**（この曲の旋律の最低〜最高）\n"
            + "\n".join(f"・**{k}**: 歌は **{_voice_note(a)}〜{_voice_note(b)}** ／ `tones` の根音（`register_octave` 0）は {roots(k)}"
                        for k, (a, b) in vocal_range.items())
            + "\n（どれも `pitches` と同じ C 基準・移調前の物差し。鳴るときは機械が区間の調へ移す）")


def _counterpoint_text(vocal_range: dict[str, tuple[int, int]] | None, shift_of=None) -> str:
    """**歌の音域と、そこを避ける指示**（2026-09-11 新設）。

    編成プロンプトには「歌と同時に同じ高さで鳴らすと喧嘩する」と書いてあったのに、
    **編曲工程は歌の音域を受け取っていなかった**——守りようがなかった。
    実測で対旋律の **144音中 86音**が歌と同時・4半音以内で鳴っていた。
    ★曲全体の最低〜最高と、`tones` の和音をどの高さに積むかも並べる（2026-09-30・出-9）。
    ★根音の範囲も C 基準で区間ごとに（`vocal_range_text`・2026-10-01）。
    """
    if not vocal_range:
        return ""
    rows = vocal_range_text(vocal_range, shift_of)
    return ("\n\n★★**歌とぶつからない高さに置いてください**★★\n" + rows
            + "\n※`tones` の和音・アルペジオは `register_octave` 0 だと、上の区間ごとの「`tones` の根音」から"
              "上へ積みます（指示書・`tones` の説明）。上の歌の高さと重なるかを見て `register_octave` を決めてください"
            + "\n**対旋律（歌に絡むパート）を、歌と同じ高さで同時に鳴らすと、どちらも輪郭を失います**"
              "（対位法の基本）。次のどれかにしてください:\n"
              "・**歌の上を行く**——`pitches` で書くなら**歌の最高音より上の音を直接書く**／"
              "コードの構成音（`tones`）で弾くなら `register_octave` を **+2** にする\n"
              "・**歌の下を行く**——`pitches` なら歌の最低音より下の音を書く／`tones` なら `register_octave` を **-1 以下**\n"
              "※**`pitches` の音は書いた高さでそのまま鳴り、`register_octave` は効きません**"
              "（上の歌の高さは `pitches` と同じ書き方・同じ物差し＝国際式の音名・C メジャー基準）\n"
              "・**歌の合間だけ鳴らす**——`at_bars` で歌の休んでいる小節を選ぶ\n"
              "コードを刻むパートも、**歌の中心と同じ高さに厚みを集めない**でください。\n\n"
              "★★**対旋律を消さないでください**★★\n"
              "**減らすのではなく、置く場所を変える**のが対位法です。\n"
              "> ★実際に起きた失敗（2026-09-11）: この指示を出したら、3回中2回で**対旋律を丸ごと省いて**きた。ぶつかりは 0 になるが、それは回避であって対位法ではない。\n"
              "**伴奏が痩せると「どの曲も同じに聞こえる」**という別の問題になります。")


def chord_row(cells: list[str]) -> str:
    """区間のコードを、伴奏・彩りに見せる形 `1:Am | 2:F G | …` に（小節は区間の中で1始まり＝`at_bars` と同じ数え方。
    1小節に複数のコードは §3 の書き方のまま空白で区切る）。伴奏（`arrange`）と彩り（`add_colour`）で同じ形（2026-10-01）。

    ★以前の伴奏の依頼文は小節を `-` でつないでいた（`Am-F-C-G`）。`-` はコード名の中で ♭（`C7-9`）にも使うので
      `C7-9` の小節が `C7`・`9` の2小節に見え、1小節に2つのコード（`Am F`）の区切りとも見分けにくかった。"""
    return " | ".join(f"{i}:{c}" for i, c in enumerate(cells, 1))


_CHORD_ROW_NOTE = ("区間ごとに `小節:コード` を ` | ` で区切る。小節はその区間の中で数える（`at_bars` と同じ）。"
                   "1小節に `Am F` のように並んでいれば、並べた数で小節を均等に分けて替わる。"
                   "コード名は `pitches` と同じ物差し＝C メジャー基準・移調前")


def arrange(title: str, bpm: int, sections: list, chords_of: dict[str, list[str]],
            genre: str = "ミドルのポップス", brief: str = "", time_signature: str = "4/4",
            vocal_range: dict[str, tuple[int, int]] | None = None, shift_of=None) -> ArrangePlan:
    """sections は compose_v2.Section のリスト。

    `brief` は**それまでの工程の判断**（テーマ・全体方針・コード・旋律の意図）。
    これが無いと伴奏だけが独立に組まれ、旋律の狙いと噛み合うかが運任せになる。
    `shift_of`（区間 → 移調量）は、`tones` の根音の範囲を歌の音域と同じ C 基準で見せるため（`vocal_range_text`・2026-10-01）。
    """
    desc = "\n".join(
        f"- {s.kind}: {s.bars}小節 **強さ{getattr(s, 'intensity', 3)}/5** "
        f"{'歌あり' if getattr(s, 'has_vocal', True) else '**歌なし**'} "
        f"コード {chord_row(chords_of.get(s.kind, [])) or '（なし）'}" for s in sections)
    # 拍子から8分音符の数を出す。かつては 4/4=8 / 3/4・6/8=6 / **それ以外も8** と決め打ちで、
    # **2/4（ポルカ・マーチ）で1小節の倍のパターンを書かせていた**（2026-08-22 修正）。
    from song_design import grid_slots                      # compose_v2 ⇄ arrange_v2 の循環を避けて遅延
    slots = grid_slots(time_signature)
    _r = _llm_run(_arranger, 
        (brief + "\n\n" if brief else "") +
        f"曲名『{title}』BPM{bpm} **拍子={time_signature}** **曲の種類={genre}**\n"
        + (f"※この曲は {time_signature} なので、**グリッドは8分音符 {slots} 個**で書いてください"
           f"（4/4 の8個ではありません）。\n" if slots != 8 else "")
        + f"構成（コードは{_CHORD_ROW_NOTE}）:\n{desc}\n\n"
        f"**全体方針と旋律の意図に合う伴奏**を組んでください"
        f"（切迫感を狙った旋律なら隙間を作らず刻む、伸びやかな旋律なら余白を残す）。\n"
        f"セクションごとに厚みを変えて起伏を作ること。"
        f"編成は目安表の「{genre}」の列を軸に、曲想で調整してください。"
        + _counterpoint_text(vocal_range, shift_of), label="伴奏")   # ラベルで頼まれごと・前回の出力・「前回のまま」が付く
    _rec_usage(_arranger, _r, "伴奏")                      # 費用集計から漏れていた（2026-08-23）
    return _mark_replayed(_r)


def _mark_replayed(r: Any) -> Any:
    """LLM の結果から出力を取り出す。**保存から戻した出力（再生・続きから）には印を付ける**——
    `fix_arrangement` がそれを直させない（作り直しで伴奏を動かさない・旋律の `fix_melody` と同じ扱い）。"""
    out = r.output
    if type(r).__name__ == "_Replayed":
        try:
            out.__dict__["_replayed"] = True
        except Exception:                                   # noqa: BLE001（印を付けられない型は直すほうへ倒す）
            pass
    return out


# ---- 彩り（対旋律・リフ）を別工程にする（2026-09-17）-----------------------------
# ★**なぜ分けたか**: 伴奏プロンプトが 6,913文字まで膨らみ、**後から足した節ほど効かなく
#   なっていた**。同じ Gemini・同じ依頼・同じ素材で、プロンプトを展開の節だけ（3,143文字）に
#   縮めたら **対旋律 0箇所 → 4箇所 / 繰り返し「同じ」→「変えた」/ at_bars 10 → 24箇所**。
#   ＝**書き足すほど、書き足した当のものが効かなくなる**。
#   （同じ結論を旋律側でも一度測っていた——拍頭率 0.520 が短いプロンプトで 0.194）
#
#   だから**プロンプトを足すのではなく、工程を分けて短いプロンプトを構造で作る**。
#   詞を4段にしたのと同じ手。骨組み（コード・ベース・ドラム）は従来どおりで、
#   ここは**彩りだけ**を見る。

class ColourPlan(StepStatus):
    """**彩り**——対旋律とリフだけ。骨組みには触らない。"""
    reason: str = Field(description=(
        "何をどこに置いたか（置かない判断をしたなら、その理由）。"
        "**歌のどこが空いていたから、そこへ入れた**という書き方にする"))
    riff: list[str] = Field(description=(
        "**この曲のリフ**（**必ず書く**。使わないなら空の並び `[]`）。国際式の音名を 3〜6音（例 `[\"G5\",\"E5\",\"C5\"]`・`pitches` と同じ書き方）。"
        "**ここに1つだけ**定義し、鳴らす所で `use_riff: true` と指す"))
    parts: list[PartPlan] = Field(description=(
        "**足すパートだけ**（**必ず書く**。骨組みのパートは書かない）。ふつう 1〜2本。"
        "彩りが邪魔になる曲では空の並び `[]`（置かないを選んだ印）——そのときは `reason` に理由を書く"))


_colorist: Agent[None, ColourPlan] = Agent(
    pick_model("ARRANGE", "flash"), output_type=ColourPlan,
    # ★区間の強さ → 音量差も渡す（2026-09-27）。彩りのパートも骨組みと合わせて `to_spec` を通るので同じ差が掛かるのに、
    #   以前は伴奏の指示書にしか書いておらず、彩りは知らずに gain を決めていた
    system_prompt=prompt("song_arrange_color").replace("<<INTENSITY_DB>>", _intensity_text()),
)


def add_colour(title: str, bpm: int, sections: list, inst_desc: str,
               vocal_frames: dict[str, str], skeleton: str = "",
               request: str = "", voices: str = "",
               vocal_range: dict[str, tuple[int, int]] | None = None,
               chords_of: dict[str, list[str]] | None = None, time_signature: str = "",
               shift_of=None) -> ColourPlan:
    """**彩りを足す**。`vocal_frames` は区間ごとの歌の枠（旋律の ABC を区間の全行 ` / ` でつないだもの。
    2026-09-30 まで説明は旧い枠記法の「`o` が歌い出し・`-` が休み」のままで、中身は ABC の1行目だけだった）。

    ★`chords_of`・`time_signature`（2026-09-30・出-7）: **コード表と拍子**。以前は渡しておらず、指示書は
      「コード表に書かれていない小節では鳴りません」と渡していない表を指し、対旋律の「ド-シ-ラ」を G/B・G・Em7・G7 の
      どの小節にも同じ形で置いた（G/B ではベースと短9度）。拍子も無かった。コード名は `pitches` と同じ物差し
      （C メジャー基準・移調前。転調は機械が両方に同じだけ掛ける）。
    ★`skeleton` は `pattern_table`（行を削らず・字数で切らない・リフ／音量／音域の欄つき・出-6）。

    ★**歌の枠を渡すのが要**。「歌のすき間に入れる」と言葉で書いても、
    **どこが空いているかを知らなければ置きようがない**。実際、手で枠を読んで
    「各行の2小節目の4拍目が空いている」と伝えたときだけ、対旋律が3区間とも入った。

    ★`voices`（2026-09-25）: **重ねた声（合いの手・追っかけ）が入っている所**。
      彩りも声部も「歌のすき間」を使うのに、以前は**互いを見ていなかった**——
      彩りが先に走り、声部の表には伴奏が載っていない。合いの手とギターの対旋律を
      両方頼むと、**同じすき間で重なる**ことがありえた。**声を先に決め、彩りがそこを避ける**
      （ポップスでは歌が主役で、楽器の飾りは歌の邪魔をしない所へ入れる）。
    """
    desc = "\n".join(f"- {s.kind}: {s.bars}小節 強さ{getattr(s, 'intensity', 3)}/5 "
                     f"{'歌あり' if getattr(s, 'has_vocal', True) else '**歌なし**'}"
                     for s in sections)
    frames = "\n".join(f"  {k:<10} {v}" for k, v in vocal_frames.items() if v)
    from song_design import grid_slots                      # noqa: E402（循環を避けて遅延・`arrange` と同じ）
    slots = grid_slots(time_signature) if time_signature else 8
    chord_rows = "\n".join(f"  {s.kind:<10} " + chord_row(chords_of.get(s.kind, []))
                           for s in sections if chords_of.get(s.kind)) if chords_of else ""
    _r = _llm_run(_colorist, (
        f"曲名『{title}』BPM{bpm}"
        + (f" **拍子={time_signature}**（`pattern` は1小節＝8分音符 {slots} 個＝{slots}文字、16分なら {slots * 2}文字）"
           if time_signature else "")
        + f"\n構成:\n{desc}\n\n"
        # describe() は見出し【この曲の編成】から始まるので、二重に付けない
        + ((inst_desc if inst_desc.startswith("【この曲の編成】") else f"【この曲の編成】\n{inst_desc}") + "\n\n"
           if inst_desc else "")
        + (f"【コード表】（{_CHORD_ROW_NOTE}。**対旋律・リフの音は、その小節のコードに合わせて選ぶ**。"
           f"コードが書かれていない小節・区間では、音程のあるパートは鳴りません）\n{chord_rows}\n\n" if chord_rows else "")
        + (f"【骨組み（もう出来ている伴奏）】\n{skeleton}\n\n" if skeleton else "")
        + (f"【歌の枠】区間ごとの歌の旋律（ABC・C メジャー基準・移調前。`|` は小節線・` / ` は行の区切り）。"
           f"**`z`（休み）が続く所と、長く伸ばす音の所が置き場**\n{frames}\n\n"
           if frames else "")
        + (f"【重ねた声（合いの手・追っかけ・ハモリ）が入っている所】**ここには対旋律・リフを置かない**"
           f"——声と楽器の飾りが同じすき間を取り合うと、どちらも聞こえなくなる\n{voices}\n\n"
           if voices else "")
        + (_counterpoint_text(vocal_range, shift_of).lstrip("\n") + "\n\n" if vocal_range else "")
        # ★**マスターの指示とは書かない**（2026-09-27）。ここに来るのは差し戻しの理由など（`song_context.CARRY`）で、
        #   マスターの頼みは台帳から `song_llm.run_sync` が足す。以前は【マスターの指示】と見出しを付けていて、
        #   選抜の独り言（他の案の惜しい点・選抜は 2026-09-30 に外した）がマスターの指示として届いていた
        + (f"【前の工程からの申し送り（差し戻しの理由など。マスターの指示ではない）】"
           f"\n{request}\n\n" if request else "")
        + "**対旋律とリフを足してください。**"), label="彩り")
    _rec_usage(_colorist, _r, "彩り")
    return _mark_replayed(_r)


# ---- 伴奏・彩りの直し（差分）（2026-09-27・`song_fix`）----------------------------------
#
# 検査（`_issues`）で所見があれば、**同じ工程に所見を返し、直すパートだけ**を書き直させる（差分）。
# 差分の単位は**パート1本（番号で指す・丸ごと差し替え／消す）**と、ドラム（丸ごと差し替え／消す）とリフ。
# 書かなかったパートは仕組みとして変わらない。流れは `song_fix.fix_loop`（検査→差分→再検査・上限2回）。

class PartFix(BaseModel):
    """差し替えるパート1本。"""
    no: JsonInt = Field(description="直すパートの番号（依頼文の「いまの伴奏」の『パートN』の N・1始まり）")
    part: PartPlan | None = Field(default=None, description=(
        "そのパートを**丸ごと書き直したもの**（直さない所も含めて全部書く）。`remove` が true なら空でよい"))
    remove: JsonBool = Field(default=False, description="このパートを消す（鳴らさない）なら true")
    # ※`why` は読み手なし（記録にも依頼文にも出ない）。直す前に考えを書かせる欄として残す（2026-09-30・出-13）
    why: str = Field(default="", description="どの所見にどう応えたか（一言）")


class ArrangePatch(PatchStatus):
    """伴奏・彩りの直し（差分）。**直さないパートは書かない**——書かなかったパートはいまのまま残る。"""
    # ※`reason` は読み手なし（記録にも依頼文にも出ない）。直す前に考えを書かせる欄として残す（2026-09-30・出-13）。
    #   `trouble` は直しの記録に入る（`_apply_arrange_patch`）
    reason: str = Field(description="何をどう直したか（全体で一言）")
    parts: list[PartFix] = Field(default_factory=list, description="直すパートだけ。直さないパートは書かない")
    drums: DrumPlan | None = Field(default=None, description=(
        "ドラムを直すなら、ドラム全体を書き直したもの（直さないなら空のまま。彩りでは使わない）"))
    drop_drums: JsonBool = Field(default=False, description="ドラムを消すなら true（編成が「ドラム: none」のとき）")
    # ★**空は「直さない」・消すのは `drop_riff`**（2026-10-03・再レビュー 記H・仕様書 §15）。ほかの差分の欄（`lead_sections` と
    #   `clear_lead_sections`・`drums` と `drop_drums`）と同じ意味にした。以前は `[]` を書くとリフを空にし、彩りでは自分のリフが消えて
    #   `use_riff` が骨組みのリフを指し、黙って骨組みの音形で鳴った
    riff: list[str] | None = Field(default=None, description=(
        "リフを直すなら新しいリフ（書き直した全体）。直さないなら書かない（空 `[]` も直さない）。消すなら drop_riff"))
    drop_riff: JsonBool = Field(default=False, description=(
        "リフを消すなら true（riff と一緒に書かない。彩りのリフを消すと、彩りの `use_riff` は骨組みのリフを指す）"))
    # ★主旋律の楽器の音量（2026-10-03・仕様書 §15）。伴奏だけ（彩りは書かない）
    lead_gain: JsonFloat | None = Field(default=None, ge=0.1, le=1.2, description=(
        "主旋律の楽器の音量（lead_gain）を直すなら新しい値（0.1〜1.2）。直さないなら空のまま。彩りの直しでは書かない"))
    drop_lead_gain: JsonBool = Field(default=False, description=(
        "lead_gain を消すなら true（編成が主旋律の楽器を決めていないとき）。彩りの直しでは書かない"))


_arrange_fixer: Agent[None, ArrangePatch] = Agent(
    pick_model("ARRANGE", "flash"), output_type=ArrangePatch,
    system_prompt=prompt("song_arrange_fix"),
)


def patch_problems(patch: Any, n_parts: int, *, colour: bool = False) -> list[str]:
    """**伴奏・彩りの差分で、当てられない書き方**の所見（2026-10-02・洗い出し）。差分の検証（`ArrangeStep.patch_problem`）と `apply` で同じ文を使う。

    ★以前は `_apply_arrange_patch` が、範囲外のパート番号・同じ番号の2つ目（後が勝つ）・`part` も `remove` も無い直し・
      `remove` と `part` の両方（消すほうが勝つ）・彩りの差分の `drums`／`drop_drums`（彩りはドラムを持たない）・`drums` と `drop_drums` の両方
      （消すほうが勝つ）を**黙って捨てていた**——LLM は直したつもりで、所見は残ったまま次の回へ進んだ（編成の `InstrumentFix.no` を
      2026-10-02 に直したのと同じ形）。パートの数は曲ごとに違うので型には書けない——頼んでいる伴奏・彩りのパートの数で見る。
    """
    label = "彩り" if colour else "伴奏"
    fixes = list(getattr(patch, "parts", None) or [])
    nos = [fx.no for fx in fixes]
    out: list[str] = []
    if bad := [n for n in nos if not 1 <= n <= n_parts]:
        out.append(f"parts のパートの番号 {'・'.join(map(str, bad))} は【いまの{label}】に無い——パートは{n_parts}本なので、"
                   f"番号は 1〜{n_parts}（【いまの{label}】の『パートN』の N をそのまま書く。パートを足す差分は無い）")
    if dup := sorted({n for n in nos if nos.count(n) > 1}):
        out.append(f"parts にパート{'・'.join(map(str, dup))}を2回以上書いた——1本につき1つだけ（書き直すなら part に丸ごと1つ）")
    for fx in fixes:
        if fx.remove and fx.part is not None:
            out.append(f"パート{fx.no}: remove と part を一緒に書いた——消すなら remove: true だけ、書き直すなら part だけ")
        elif not fx.remove and fx.part is None:
            out.append(f"パート{fx.no}: part も remove も無い——書き直すなら part に丸ごと、消すなら remove: true（直さないパートは書かない）")
    if colour and (getattr(patch, "drums", None) is not None or getattr(patch, "drop_drums", False)):
        out.append("彩りはドラムを持たない——drums・drop_drums は書かない")
    elif getattr(patch, "drop_drums", False) and getattr(patch, "drums", None) is not None:
        out.append("drop_drums と drums を一緒に書いた——消すなら drop_drums: true だけ、書き直すなら drums だけ")
    if colour and (getattr(patch, "lead_gain", None) is not None or getattr(patch, "drop_lead_gain", False)):
        out.append("彩りは主旋律の楽器の音量を持たない——lead_gain・drop_lead_gain は書かない")
    elif getattr(patch, "drop_lead_gain", False) and getattr(patch, "lead_gain", None) is not None:
        out.append("drop_lead_gain と lead_gain を一緒に書いた——消すなら drop_lead_gain: true だけ、直すなら lead_gain だけ")
    if getattr(patch, "drop_riff", False) and getattr(patch, "riff", None):   # 空の riff は「直さない」（記H・2026-10-03）
        out.append("drop_riff と riff を一緒に書いた——消すなら drop_riff: true だけ、書き直すなら riff だけ")
    return out


def lead_gain_problems(plan: Any, lead: int | None) -> list[str]:
    """**主旋律の楽器の音量（`lead_gain`）が、編成の主旋律の楽器（`lead`）と合うか**の所見（2026-10-03・仕様書 §15・§15-4）。

    編成が `lead` を書いたら伴奏は `lead_gain` を必ず書き、書いていなければ書かない。組み立ては既定値で埋めない（書いていなければ
    この所見で伴奏の工程に直させ、直らなければ伴奏の工程で止まる）。伴奏の検査（`ArrangeStep.check`）と、形を変えたときに残す
    伴奏へ当てる検査（`song_form_checks.form_findings`）が同じこれを呼ぶ。
    """
    gain = getattr(plan, "lead_gain", None)
    if lead is not None and gain is None:
        return [f"編成が主旋律の楽器（楽器{lead}）を決めたのに lead_gain が無い——主旋律の楽器の音量を lead_gain に書く"
                "（0.1〜1.2・音程パートの gain と同じ物差しのパートどうしの比）"]
    if lead is None and gain is not None:
        return [f"編成が主旋律の楽器を決めていない（lead が無い）のに lead_gain（{gain:g}）を書いた——lead_gain は書かない"
                "（差分の drop_lead_gain）"]
    return []


def _dump(m: Any) -> str:
    return json.dumps(m.model_dump(), ensure_ascii=False)


from song_fix import StepFixer, guard_patch                 # noqa: E402（工程の形・2026-09-28）

# ★**当てられない差分は差し戻す**（2026-10-02・`patch_problems`）。直しを頼む間だけ工程の `patch_problem` を置き、
#   差分の検証（`song_fix.guard_patch`）がこの文を伴奏・彩りの LLM へ返して書き直させる（直らなければその工程で止まる）。
#   同じ日に作った伴奏だけの検証 `_patch_in_range`・`_PATCH_CTX` は、工程の形の1つへまとめた（組「洗い出し-型と残り」）
guard_patch(_arrange_fixer)


class ArrangeStep(StepFixer):
    """**伴奏・彩りの工程の検査と直し**（2026-09-28・工程が自分の検査を持つ形）。伴奏（`ArrangePlan`）と彩り（`ColourPlan`）は
    同じ検査（`_issues`）と同じ差分（`ArrangePatch`＝パート1本を番号で丸ごと差し替え／消す・ドラム・リフ）で、`key`・`label` だけが違う。
    打点のパターン（伴奏の pattern・ドラムの行）の記法は `written`（`pattern_notation`）で宣言し、`check` はそれ以外（2026-09-29）。
    工程の所見（`findings`）は `arrange_findings` と同じもの・同じ並び。

    直しても残ったら、その工程（伴奏／彩り）で曲づくりを止める（`song_fix.StepUnfixed`・2026-09-30）。以前は `settle` で
    「組み立てでは鳴らさなかった」を記録して進んでいた。
    `bars_of` は**時間軸の実際の小節**（伴奏には設計の小節数を見せているので、食い違いはここで所見になる）。
    """
    agent = _arrange_fixer
    return_hint = ("**直すパートだけ**を parts に入れてください（番号で指し、そのパートを丸ごと書き直す）。"
                   "直さないパートは書かないでください。")

    def __init__(self, *, colour: bool = False, bars_of: Mapping[str, tuple[int, int]],
                 instruments: list | None = None, allow_drums: bool = True, slots: int = 8,
                 inst_desc: str = "", base_riff: list[str] | None = None, drum_kit: str = "",
                 need_drums: bool = False, lead: int | None = None, shift_of: Any | None = None):
        self.key = "colour" if colour else "arrangement"
        self.label = "彩り" if colour else "伴奏"
        # 区間表と検査は設計の区間だけ（曲頭の空き小節 lead_in を小節表に混ぜない・2026-09-29）
        # `need_drums`（編成がキットを決めたのに drums が無いのを所見にする・出-8）は伴奏だけ。彩りはドラムを持たない
        # `shift_of`（区間 → 移調量）は書いた高さを移したあとの鳴る高さを見るため（2026-10-03・組み立て `to_spec` と同じもの）
        self.kw = dict(bars_of=_song_sections(bars_of), instruments=instruments, allow_drums=allow_drums, slots=slots,
                       base_riff=base_riff, drum_kit=drum_kit or "", need_drums=bool(need_drums) and not colour,
                       shift_of=shift_of)
        self.inst_desc = inst_desc
        #: 編成の主旋律の楽器（`Instrumentation.lead`・2026-10-03）。伴奏の `lead_gain` の決まりを見る（彩りは見ない）
        self.lead = None if colour else lead

    def written(self, plan):
        # 音程パートとドラムの行は別の記法（2026-10-02・ず-9: 打点の `o` はドラムだけ・`_` は音程パートだけ）
        nts = {d: pattern_notation(self.kw["slots"], drum=d) for d in (False, True)}
        yield from ((w, nts[drum], pat) for w, pat, drum in _written(plan, allow_drums=self.kw["allow_drums"]))
        yield from ((w, PITCH, x) for w, x in _written_pitches(plan))   # 音名（国際式・2026-10-01）

    def check(self, plan):
        # 打点のパターンと音名の記法は `written`（親の照合）。ここはそれ以外（区間・長さ・音の数・tones・at_bars・楽器番号・打楽器名…）
        out = [i.finding() for i in _issues(plan, notation=False, **self.kw)]
        # 主旋律の楽器の音量（2026-10-03）。伴奏だけ（彩りの型に lead_gain は無い）
        return out + ([] if self.key == "colour" else lead_gain_problems(plan, self.lead))

    def show(self, plan):
        bars_of, instruments = self.kw["bars_of"], self.kw["instruments"]
        base_riff, slots = self.kw["base_riff"], self.kw["slots"]
        secs = "\n".join(f"- {k}: **実際には{b - a + 1}小節**（曲の {a}〜{b} 小節目）"
                         for k, (a, b) in sorted(bars_of.items(), key=lambda kv: kv[1]))
        insts = self.inst_desc or "\n".join(f"  {i}. {gm_name(g)}（{bk}-{g}）"
                                            for i, (bk, g) in enumerate(instruments or [], 1))
        rows = [f"■パート{i}: {_dump(p)}" for i, p in enumerate(getattr(plan, "parts", None) or [], 1)]
        if hasattr(plan, "drums"):
            rows.append("■ドラム: " + (_dump(plan.drums) if plan.drums else "なし"))
        rows.append("■リフ（riff）: " + json.dumps(list(getattr(plan, "riff", None) or []), ensure_ascii=False)
                    + (f"（空なら骨組みのリフ {json.dumps(list(base_riff), ensure_ascii=False)} を指す）" if base_riff else ""))
        # 主旋律の楽器の音量（2026-10-03）は、編成が主旋律の楽器を決めたか、伴奏が書いたときだけ見せる（以前の見せ方を変えない）
        if self.key != "colour" and (self.lead is not None or getattr(plan, "lead_gain", None) is not None):
            rows.append(f"■主旋律の楽器の音量（lead_gain）: {getattr(plan, 'lead_gain', None)}"
                        + (f"（主旋律の楽器＝編成の楽器{self.lead}）" if self.lead is not None
                           else "（編成は主旋律の楽器を決めていない）"))
        return (f"【この曲の区間】（**at_bars はこの実際の小節数で数える**）\n{secs}\n\n"
                + (f"【この曲の編成】（instrument はこの番号）\n{insts}\n\n" if insts else "")
                + (f"【ドラム】{'使う' if self.kw['allow_drums'] else '**使わない**（編成が none と決めた）'}"
                   + ("（**鳴るのは drums に書いたものだけ**。グルーヴのドラムは土台の指定で、それ自体は鳴らない）"
                      if self.kw["need_drums"] else "")
                   + (f"。キット『{self.kw['drum_kit']}』で鳴らせる打楽器（inst はこの名前だけ）: "
                      f"{kit_drum_menu(self.kw['drum_kit'])}" if self.kw["allow_drums"] and self.kw["drum_kit"] else "")
                   + "\n\n" if hasattr(plan, "drums") else "")
                + f"【1小節の文字数】{slots}文字（16分なら{slots * 2}文字）\n\n"
                + f"【いまの{self.label}】\n" + "\n".join(rows))

    def patch_problem(self, plan, patch) -> str:
        """当てられない差分の所見（`patch_problems`。差分の検証 `song_fix.guard_patch` が頼む間だけ読む・2026-10-02）。"""
        return "／".join(patch_problems(patch, len(getattr(plan, "parts", None) or []), colour=self.key == "colour"))

    def apply(self, plan, patch):
        return _apply_arrange_patch(plan, patch)


def _apply_arrange_patch(plan: Any, patch: Any) -> tuple[Any, str]:
    """差分をそのまま当てる（＝組み立て）。**書かれていないパートは変えない**。元の案は変えない（写しを返す）。

    ★当てられない書き方（範囲外の番号・同じ番号の2つ目・part も remove も無い ほか）は差分の検証（`patch_problem`・`song_fix.guard_patch`）が差し戻すので来ない
      （2026-10-02。以前はここで黙って捨てた）。型を通らずに来たら（保存から戻した差分など）、当てなかったことを直しの記録に書く。"""
    new = plan.model_copy(deep=True)
    parts: list[Any] = list(new.parts)
    done: list[str] = []
    if why := patch_problems(patch, len(parts), colour=not hasattr(new, "drums")):
        done.append("差分の書き方の所見（範囲外の番号・一緒に書けない直しは当てなかった）: " + "／".join(why))
    gone: set[int] = set()
    for fx in (getattr(patch, "parts", None) or []):
        if not 1 <= fx.no <= len(parts):
            continue
        if fx.remove:
            gone.add(fx.no)
            done.append(f"パート{fx.no}を消した")
        elif fx.part is not None:
            parts[fx.no - 1] = fx.part.model_copy(deep=True)
            done.append(f"パート{fx.no}を書き直した")
    new.parts = [p for i, p in enumerate(parts, 1) if i not in gone]
    if hasattr(new, "drums"):
        if getattr(patch, "drop_drums", False):
            if new.drums is not None:
                done.append("ドラムを消した")
            new.drums = None
        elif getattr(patch, "drums", None) is not None:
            new.drums = patch.drums.model_copy(deep=True)
            done.append("ドラムを書き直した")
    # ★空の riff は「直さない」・消すのは drop_riff（2026-10-03・記H）。両方書いた差分は差分の検証が差し戻すので来ない
    #   （型を通らずに来たら、上の「差分の書き方の所見」に書いたうえで消すほうを当てない＝どちらも当てない）
    both = bool(getattr(patch, "drop_riff", False) and getattr(patch, "riff", None))
    if getattr(patch, "drop_riff", False) and not both:
        if new.riff:
            done.append("リフを消した")
        new.riff = []
    elif getattr(patch, "riff", None) and not both:
        new.riff = list(patch.riff)
        done.append("リフを書き直した")
    # 主旋律の楽器の音量（2026-10-03）。伴奏だけ（彩りの差分の lead_gain は差分の検証が差し戻すので来ない）
    if hasattr(new, "lead_gain"):
        if getattr(patch, "drop_lead_gain", False):
            if new.lead_gain is not None:
                done.append(f"lead_gain を消した（{new.lead_gain:g} → なし）")
            new.lead_gain = None
        elif getattr(patch, "lead_gain", None) is not None:
            done.append(f"lead_gain を {new.lead_gain} → {patch.lead_gain:g}")
            new.lead_gain = patch.lead_gain
    note = "／".join(done) if done else "直しなし"
    if getattr(patch, "trouble", ""):
        note += f"／直さなかった理由: {patch.trouble}"
    return new, note


def _kit_of(song: Any) -> str:
    """曲の編成が決めたドラムの種類（編成が無ければ空）。伴奏の検査は前の工程（編成）を曲から丸ごと読む。"""
    return (getattr(getattr(song, "instrumentation", None), "drum_kit", "") or "") if song is not None else ""


def _lead_of(song: Any) -> int | None:
    """曲の編成が決めた主旋律の楽器の番号（`Instrumentation.lead`・2026-10-03。編成が無い・書いていなければ None）。"""
    return getattr(getattr(song, "instrumentation", None), "lead", None) if song is not None else None


def fix_arrangement(plan: ArrangePlan, song: Any, *, bars_of: Mapping[str, tuple[int, int]],
                    instruments: list | None = None, allow_drums: bool = True, slots: int = 8,
                    inst_desc: str = "", verbose: bool = False,
                    drum_kit: str | None = None, need_drums: bool | None = None,
                    shift_of: Any | None = None) -> tuple[ArrangePlan, list[str]]:
    """伴奏を検査し、所見があれば**伴奏の工程に差分で直させる**（`ArrangeStep`）。戻り値は (伴奏, 残った所見＝いつも空)。

    呼ぶ場所は `arrange()` の直後・`to_spec` の前（`song_band.run_arrangement`／やり直しは `run_arrange_gate`）。直したら `StepRecord("arrangement_fix")`。
    直しても残ったら伴奏の工程で曲づくりを止める（`song_fix.StepUnfixed`・2026-09-30）。
    `bars_of` は**時間軸の実際の小節**（伴奏には設計の小節数を見せているので、食い違いはここで所見になる）。
    `drum_kit` を渡さなければ、曲の編成（`song.instrumentation.drum_kit`）のキットで打楽器を見る（2026-09-29）。
    `need_drums` を渡さなければ、**曲に編成があり、ドラムを使う（`allow_drums`）なら**、`drums` が無いことを所見にする
    （2026-09-30・出-8。ドラムの持ち主は伴奏）。編成の無い呼び方（検査だけの試験など）では見ない。
    主旋律の楽器の音量（`lead_gain`）は曲の編成の `lead` と照らす（2026-10-03・`lead_gain_problems`）。
    `shift_of`（区間 → 移調量・`to_spec` に渡すのと同じもの）を渡すと、`pitches`・リフを移したあとの鳴る高さが 0〜127 に
    収まるかも見る（2026-10-03・再レビュー）。**`to_spec` に `shift_of` を渡す呼び手は、ここにも同じものを渡す**
    （渡さないと、ここで所見にならずに `to_spec` の入口の照合で `AssemblyError` になる）。
    """
    if need_drums is None:
        need_drums = allow_drums and getattr(song, "instrumentation", None) is not None
    return ArrangeStep(bars_of=bars_of, instruments=instruments, allow_drums=allow_drums, slots=slots,
                       inst_desc=inst_desc, drum_kit=_kit_of(song) if drum_kit is None else drum_kit,
                       need_drums=need_drums, lead=_lead_of(song), shift_of=shift_of).run(plan, song, verbose=verbose)


def fix_colour(colour: ColourPlan, song: Any, *, bars_of: Mapping[str, tuple[int, int]],
               instruments: list | None = None, slots: int = 8, inst_desc: str = "",
               base_riff: list[str] | None = None, verbose: bool = False,
               shift_of: Any | None = None) -> tuple[ColourPlan, list[str]]:
    """彩りも同じ形で検査し、差分で直させる。`base_riff` は骨組み（伴奏）のリフ。記録は `colour_fix`・`colour_machine`。
    （彩りはドラムを持たないので、キットは見ない）。`shift_of` は `fix_arrangement` と同じ（移したあとの鳴る高さ）。"""
    return ArrangeStep(colour=True, bars_of=bars_of, instruments=instruments, slots=slots, inst_desc=inst_desc,
                       base_riff=base_riff, shift_of=shift_of).run(colour, song, verbose=verbose)
