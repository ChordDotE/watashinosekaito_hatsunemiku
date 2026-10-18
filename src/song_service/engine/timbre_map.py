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

"""**音色の正準リスト**。編曲(song_arrange)・MIDI 出力(midi_out)が同じ表を見る。

音色名を3か所で別々に持つと必ずずれる。実際、プロンプトが `piano` を選択肢に挙げていたのに
numpy 側に定義が無く、**伴奏の書き出しが KeyError で落ちた**（2026-08-21）。
その後 `piano → bell`（鉄琴）へ寄せる回避を入れたため、今度は
**LLM がピアノのつもりで書いた曲が鉄琴で鳴っていた**。表を1つにしてこの類を止める。

    LLM が選ぶ名前 ──→ GM_PROGRAM  … SoundFont / HALion（実音）

★numpy 合成音の近似（`NUMPY_APPROX` 等・SoundFont が使えないときの落とし先）は 2026-09-29 に消した。
  伴奏は SoundFont で鳴らし、鳴らせなければ止めて原因を返す（`song_pipeline._render_accomp`）。
"""
from __future__ import annotations

from song_checks import edge_ws_problem, strip_ascii_ws, trim_edge_ws   # 半角の名前の前後の空白（仕様書 §0-6・2026-10-03）

import functools
import re

# ---- GM 音色（program number）------------------------------------------------
GM_PROGRAM = {
    # 従来からある名前（後方互換）
    "pad": 89,          # Warm Pad
    "bass": 33,         # Finger Bass
    "pluck": 45,        # Pizzicato
    "lead": 81,         # Saw Lead
    "bell": 11,         # Vibraphone
    "strings": 48,      # String Ensemble
    "guitar": 27,       # Clean Electric Guitar
    # ここから実音（numpy 版には無かったもの）
    "piano": 0,         # Acoustic Grand
    "e_piano": 4,
    "organ": 18,        # Rock Organ
    "ac_guitar": 25,    # Steel Guitar
    "dist_guitar": 30,
    "pick_bass": 34,
    "synth_bass": 38,
    "brass": 61,        # Brass Section
    "sax": 65,
    "flute": 73,
    "choir": 52,        # Choir Aahs
    "harp": 46,
    "marimba": 12,
    "glocken": 9,
    "accordion": 21,
    "square_lead": 80,
    "saw_pad": 91,
}

TIMBRE_ALIAS = {"keys": "piano", "synth": "pad", "drums": "pad", "arpeggio": "pluck",
                "epiano": "e_piano", "electric_piano": "e_piano", "acoustic_guitar": "ac_guitar",
                "eguitar": "guitar", "distortion_guitar": "dist_guitar", "vibraphone": "bell",
                "vibes": "bell", "voice": "choir", "vocal": "choir", "horn": "brass",
                "trumpet": "brass", "violin": "strings", "cello": "strings", "synth_lead": "lead"}


def parse_gm(name: str) -> int | None:
    """`gm61` / `gm1-61`（bank付き）から **音色番号**（1始まり）を取り出す。

    編成工程が音源の全音色から選ぶようになったので、音色は `gm<bank>-<番号>` の形で持ち回る
    （24種の名前付き音色では表せない・2026-08-22）。
    """
    b, n = parse_gm_bank(name)
    return n


def parse_gm_bank(name: str) -> tuple[int, int | None]:
    """`gm61` → (0, 61) ／ `gm1-61` → (1, 61) ／ それ以外 → (0, None)。"""
    k = (name or "").strip().lower()
    if not k.startswith("gm"):
        return 0, None
    body = k[2:]
    bank = 0
    if "-" in body:
        head, _, body = body.partition("-")
        if not head.isdigit():
            return 0, None
        bank = int(head)
    if body.isdigit() and 1 <= int(body) <= 128:
        return bank, int(body)
    return 0, None


def _lookup(name: str) -> str | None:
    """名前 → 正準名。**見つからなければ None**（既定へ落とす判断は呼び出し側に任せる）。"""
    k = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not k:
        return None
    if k in GM_PROGRAM:
        return k
    if k in TIMBRE_ALIAS:
        return TIMBRE_ALIAS[k]
    for n in GM_PROGRAM:                                    # 部分一致（"warm pad" → pad）
        if n in k:
            return n
    return None


def canon_timbre(name: str, role: str = "") -> str:
    """LLM が書いた音色名を**正準名**へ。未知なら役割から推測し、最後は pluck に落とす。"""
    return _lookup(name) or _lookup(role) or "pluck"


def gm_program(timbre: str, role: str = "") -> int:
    """音色名 → GM プログラム。**timbre が正**（読めなければ役割、最後は pluck）。

    ※numpy 音源の頃に汎用音色へ寄せられたデータ（timbre='bell' / role='piano' など）を役割の側で読み直す例外
      （`_LEGACY_FALLBACK`・`_ROLE_WORDS`）は 2026-09-30 に外した（古い記録は読まない）。
    """
    if (n := parse_gm(timbre)) is not None:                # 編成工程が選んだ GM 番号
        return n - 1                                       # GM は1始まり・プログラム番号は0始まり
    t, r = _lookup(timbre), _lookup(role)
    return GM_PROGRAM[t or r or "pluck"]


# ---- GM ドラム（note number・ch10）-------------------------------------------
#: ★**打楽器の名前は GM（General MIDI）の正式な名前だけ**（2026-10-02・本人決定「おすすめのとおりで良い」。書き方は1通り）。
#:   General MIDI Level 1 の Percussion Key Map（35〜81）の名前を、**小文字にし、空白と `-` を `_` に**したもの
#:   （`Closed Hi Hat` → `closed_hi_hat`・`Pedal Hi-Hat` → `pedal_hi_hat`・`Low-Mid Tom` → `low_mid_tom`。
#:   `Hi` と `High` は GM の名前のまま書き分ける＝`hi_bongo`・`high_timbale`）。綴りの決まりは記法の仕様書 §15-2 の1か所。
#:   以前は短い独自名（`kick`・`snare`・`hihat` …）が正式な名前で、GM の正式名と昔からの略称（`bd`・`hh`・`floor_tom` …）を
#:   別名（`DRUM_ALIAS`）で読んでいた＝同じ打楽器に名前が何通りもあった。`floor_tom` は GM のフロアタム（41・43）ではなく 45（Low Tom）を指していた。
#:   **別名は受け付けない**（知らない名前は所見にし、近い GM の名前を示して書き直させる＝`drum_name_problem`）。
GM_DRUM = {
    "acoustic_bass_drum": 35, "bass_drum_1": 36, "side_stick": 37, "acoustic_snare": 38, "hand_clap": 39,
    "electric_snare": 40, "low_floor_tom": 41, "closed_hi_hat": 42, "high_floor_tom": 43, "pedal_hi_hat": 44,
    "low_tom": 45, "open_hi_hat": 46, "low_mid_tom": 47, "hi_mid_tom": 48, "crash_cymbal_1": 49,
    "high_tom": 50, "ride_cymbal_1": 51, "chinese_cymbal": 52, "ride_bell": 53, "tambourine": 54,
    "splash_cymbal": 55, "cowbell": 56, "crash_cymbal_2": 57, "vibraslap": 58, "ride_cymbal_2": 59,
    "hi_bongo": 60, "low_bongo": 61, "mute_hi_conga": 62, "open_hi_conga": 63, "low_conga": 64,
    "high_timbale": 65, "low_timbale": 66, "high_agogo": 67, "low_agogo": 68, "cabasa": 69,
    "maracas": 70, "short_whistle": 71, "long_whistle": 72, "short_guiro": 73, "long_guiro": 74,
    "claves": 75, "hi_wood_block": 76, "low_wood_block": 77, "mute_cuica": 78, "open_cuica": 79,
    "mute_triangle": 80, "open_triangle": 81,
}
assert list(GM_DRUM.values()) == list(range(35, 82)), "GM の打楽器 35〜81 に1つずつ名前が無い"
assert all(re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", k) for k in GM_DRUM), "打楽器名は小文字と数字を `_` でつないだ形"
#: 番号 → 名前（道具の返事 `accomp_edit.drum_shown` が見せる名前。名前と番号は1対1）
GM_DRUM_NAME = {n: k for k, n in GM_DRUM.items()}

#: ★**所見の手がかりだけ**（読まない）。知らない名前のとき「近い GM の名前」を添えるための表で、**この名前で書いても鳴らない**。
#:   以前の正式な名前と別名（2026-10-02 まで `DRUM_ALIAS` で読んでいたもの）→ GM の名前。
_DRUM_HINT = {
    "kick": "bass_drum_1", "kick2": "acoustic_bass_drum", "bd": "bass_drum_1", "bassdrum": "bass_drum_1",
    "bass_drum": "bass_drum_1", "kick_drum": "bass_drum_1", "concert_bass_drum": "bass_drum_1",
    "snare": "acoustic_snare", "snare2": "electric_snare", "sd": "acoustic_snare", "snare_drum": "acoustic_snare",
    "rim": "side_stick", "cross_stick": "side_stick", "rimshot": "side_stick",
    "clap": "hand_clap", "handclap": "hand_clap", "claps": "hand_clap",
    "hihat": "closed_hi_hat", "closedhat": "closed_hi_hat", "hat": "closed_hi_hat", "hh": "closed_hi_hat",
    "hi_hat": "closed_hi_hat", "closed_hat": "closed_hi_hat", "hihat_closed": "closed_hi_hat", "closed_hihat": "closed_hi_hat",
    "openhat": "open_hi_hat", "oh": "open_hi_hat", "open_hihat": "open_hi_hat", "open_hat": "open_hi_hat", "hihat_open": "open_hi_hat",
    "pedalhat": "pedal_hi_hat", "pedal_hat": "pedal_hi_hat", "hihat_pedal": "pedal_hi_hat", "foot_hat": "pedal_hi_hat",
    "crash": "crash_cymbal_1", "crash2": "crash_cymbal_2", "crash_cymbal": "crash_cymbal_1", "cym": "crash_cymbal_1",
    "cymbal": "crash_cymbal_1", "ride": "ride_cymbal_1", "ride2": "ride_cymbal_2", "ride_cymbal": "ride_cymbal_1",
    "china": "chinese_cymbal", "splash": "splash_cymbal",
    "tom": "low_mid_tom", "tom_low": "low_tom", "tom_mid": "low_mid_tom", "tom_mid_high": "hi_mid_tom", "tom_high": "high_tom",
    "mid_tom": "low_mid_tom", "hi_tom": "high_tom", "floor_tom": "low_floor_tom",
    "tom_floor_low": "low_floor_tom", "tom_floor_high": "high_floor_tom",
    "timpani": "low_tom", "timpano": "low_tom",
    "shaker": "maracas", "perc": "maracas", "tamb": "tambourine", "tambourin": "tambourine",
    "conga": "open_hi_conga", "conga_hi": "open_hi_conga", "conga_lo": "low_conga", "conga_mute": "mute_hi_conga",
    "bongo": "hi_bongo", "bongo_lo": "low_bongo", "timbale": "high_timbale", "timbales": "high_timbale",
    "timbale_lo": "low_timbale", "agogo_hi": "high_agogo", "agogo_lo": "low_agogo",
    "whistle_short": "short_whistle", "whistle_long": "long_whistle", "guiro_short": "short_guiro", "guiro_long": "long_guiro",
    "wood": "hi_wood_block", "woodblock": "hi_wood_block", "wood_block": "hi_wood_block", "wood_lo": "low_wood_block",
    "cuica_mute": "mute_cuica", "cuica_open": "open_cuica", "triangle": "open_triangle", "triangle_mute": "mute_triangle",
}
assert all(v in GM_DRUM for v in _DRUM_HINT.values()) and not set(_DRUM_HINT) & set(GM_DRUM)


def drum_canon(inst: str) -> str | None:
    """打楽器名 → GM の正式な名前（`GM_DRUM` の鍵）。**書いた名前がそのまま GM の名前のときだけ**（前後の半角の空白とタブだけ無視。
    前後の全角の空白・改行などは読まない＝所見は `drum_name_problem`・§0-6・2026-10-03）。
    別名・大文字・空白や `-` の入った書き方は読まない（知らなければ None。所見の文は `drum_name_problem`）。

    編曲の検査（`song_arrange._issues`）・グルーヴの検査・伴奏を手で直す道具（`tools/accomp.py`）が同じ1つで読む。
    """
    k = strip_ascii_ws(inst)
    return k if k in GM_DRUM else None


def drum_hint(inst: str) -> str:
    """知らない打楽器名に**近い GM の名前**（無ければ空）。所見の文に添えるだけで、読み替えはしない。"""
    k = (inst or "").strip().lower().replace("-", "_").replace(" ", "_")
    if k in GM_DRUM:                                       # 大文字・空白・`-` で書いた GM の名前（`Bass Drum 1`）
        return k
    if k in _DRUM_HINT:
        return _DRUM_HINT[k]
    import difflib
    got = difflib.get_close_matches(k, [*GM_DRUM, *_DRUM_HINT], n=1, cutoff=0.6)   # 綴りの誤り（`snaer`・`bass_drum1`）
    return (_DRUM_HINT.get(got[0], got[0])) if got else ""


def drum_name_problem(inst: str) -> str:
    """打楽器名が GM の正式な名前でなければ所見の文（読めれば空）。近い GM の名前があれば添える。"""
    if drum_canon(inst) is not None:
        return ""
    if why := edge_ws_problem(inst):                       # 前後に全角の空白・改行などを付けない（2026-10-03・§0-6）
        rest = drum_name_problem(trim_edge_ws(inst))
        return f"打楽器名{why}" + (f"。そのうえで: {rest}" if rest else "")
    near = drum_hint(inst)
    return (f"打楽器名『{inst}』は使えない（打楽器は GM の正式な名前を小文字・`_` つなぎで書く"
            + (f"——『{near}』のように" if near else "") + "）")


def hat_line_problem(inst: str, pattern: str) -> str:
    """`closed_hi_hat` の行に `o` `O` を書いていれば所見の文（2026-10-02）。開いたハイハットは `open_hi_hat` の行に書く（書き方は1通り）。

    以前は hihat の行の `o` を開いたハイハットで鳴らしていた（2026-10-01・判-J・一般のドラムタブどおり）。名前が閉じた・開いたに
    分かれたので、閉じたハイハットの行で開くと書けると、同じ音に書き方が2通りになる。黙って閉じたまま鳴らすと、ドラムタブの
    読み方（`o`＝開く）で書いた意図と違う音になる。だから所見にして、開く打点を `open_hi_hat` の行へ移させる。
    """
    if drum_canon(inst) != "closed_hi_hat" or not any(c in "oO" for c in (pattern or "")):
        return ""
    keep = "".join("-" if c in "oO" else c for c in pattern)
    opened = "".join(c if c in "oO|" or c.isspace() else "-" for c in pattern)
    return (f"closed_hi_hat の行に `o` は書けない（閉じたハイハットの打点は `X` `x` `.`）。開いたハイハットは open_hi_hat の行に書く"
            f"——closed_hi_hat『{keep}』と open_hi_hat『{opened}』のように分ける")


#: 編曲に見せる打楽器名（`GM_DRUM` の鍵＝GM の番号順）。**知らない名前は所見にして返す**（2026-09-27）
DRUM_NAMES = tuple(GM_DRUM)


def drum_note(inst: str) -> int | None:
    """打楽器名 → GM の番号。**知らない名前は None**（鳴らさない）。

    ★**知らない名前は、編曲の検査が先に所見にして編曲へ返す**（2026-09-27・打-3）。直しても残ったら**鳴らさない**
      （2026-09-28・組み立ては補正しない）。以前はここが最後の網で、部分一致かシェイカーの近い音へ寄せていた。
    """
    c = drum_canon(inst)
    return GM_DRUM[c] if c is not None else None


# ★numpy 合成音の近似（`NUMPY_APPROX`・`_GM_RANGE_APPROX`・`register_numpy_timbres`・`_NUMPY_DRUM`）は
#   2026-09-29 に消した。SoundFont が使えないときに伴奏を numpy の合成音へ黙って落とし、ピアノを鉄琴の形で・
#   タムをキックで・シンバルをハイハットで鳴らしていた（代わりのもので鳴らす）。いまは落とさず止める（`song_pipeline._render_accomp`）。


# ---- HALion Sonic のスロット割り当て -------------------------------------------
# HALion は GM のプログラムチェンジで音色を選べない（16スロットに読み込む方式で、
# 読み込みは MediaBay=GUI の担当。CLI からは切り替えられない・2026-08-21 実測）。
# そこで **スロット番号＝MIDI チャンネル**を規約で固定し、人が一度だけ仕込む。
# ch10 をドラムにしてあるのは MIDI の慣習に合わせるため（人が迷わない）。
HALION_SLOT = {
    "piano": 1, "e_piano": 2, "organ": 3, "ac_guitar": 4, "guitar": 5, "dist_guitar": 6,
    "bass": 7, "pick_bass": 7, "synth_bass": 8, "strings": 9,
    # 10 はドラムキット（音程パートには割り当てない）
    "brass": 11, "sax": 11, "choir": 12, "flute": 12,
    "pad": 13, "saw_pad": 13, "lead": 14, "square_lead": 14, "pluck": 14,
    "bell": 15, "glocken": 15, "marimba": 15, "accordion": 15,
    "harp": 16,
}


def halion_channel(timbre: str, role: str = "") -> int:
    """音色 → HALion のスロット番号（=MIDI チャンネル・1始まり）。未知は pad の枠へ。"""
    return HALION_SLOT.get(canon_timbre(timbre, role), 13)


# ---- GM 128音色の全リスト --------------------------------------------------------
# 音色を24種に絞っていたのは「編曲プロンプトに一覧を書く」都合だった。**編成を決める工程**を
# 分けたので、そこでは128種すべてを対象にできる（2026-08-22）。
# 番号は GM の 1始まり（プログラム番号は -1 したもの）。
GM_NAMES = [
    "アコースティックピアノ", "ブライトピアノ", "エレクトリックグランド", "ホンキートンク",
    "エレピ1(ローズ)", "エレピ2(FM)", "ハープシコード", "クラビネット",
    "チェレスタ", "グロッケン", "オルゴール", "ビブラフォン",
    "マリンバ", "シロフォン", "チューブラーベル", "ダルシマー",
    "ドローバーオルガン", "パーカッシブオルガン", "ロックオルガン", "教会オルガン",
    "リードオルガン", "アコーディオン", "ハーモニカ", "タンゴアコーディオン",
    "ナイロンギター", "スチールギター", "ジャズギター", "クリーンギター",
    "ミュートギター", "オーバードライブギター", "ディストーションギター", "ギターハーモニクス",
    "アコースティックベース", "指弾きベース", "ピックベース", "フレットレスベース",
    "スラップベース1", "スラップベース2", "シンセベース1", "シンセベース2",
    "バイオリン", "ビオラ", "チェロ", "コントラバス",
    "トレモロストリングス", "ピチカート", "ハープ", "ティンパニ",
    "ストリングス合奏1", "ストリングス合奏2", "シンセストリングス1", "シンセストリングス2",
    "合唱(アー)", "声(ウー)", "シンセボイス", "オーケストラヒット",
    "トランペット", "トロンボーン", "チューバ", "ミュートトランペット",
    "フレンチホルン", "ブラスセクション", "シンセブラス1", "シンセブラス2",
    "ソプラノサックス", "アルトサックス", "テナーサックス", "バリトンサックス",
    "オーボエ", "イングリッシュホルン", "ファゴット", "クラリネット",
    "ピッコロ", "フルート", "リコーダー", "パンフルート",
    "ブロウボトル", "尺八", "ホイッスル", "オカリナ",
    "リード(スクエア)", "リード(ノコギリ)", "リード(カリオペ)", "リード(チフ)",
    "リード(チャラン)", "リード(ボイス)", "リード(5度)", "リード(ベース+リード)",
    "パッド(ニューエイジ)", "パッド(ウォーム)", "パッド(ポリシンセ)", "パッド(クワイア)",
    "パッド(ボウ)", "パッド(メタリック)", "パッド(ハロー)", "パッド(スイープ)",
    "FX(雨)", "FX(サウンドトラック)", "FX(クリスタル)", "FX(アトモスフィア)",
    "FX(ブライトネス)", "FX(ゴブリン)", "FX(エコー)", "FX(SF)",
    "シタール", "バンジョー", "三味線", "琴",
    "カリンバ", "バグパイプ", "フィドル", "シャナイ",
    "ティンクルベル", "アゴゴ", "スチールドラム", "ウッドブロック",
    "太鼓", "メロディックタム", "シンセドラム", "リバースシンバル",
    "ギターフレットノイズ", "ブレスノイズ", "海岸", "鳥のさえずり",
    "電話", "ヘリコプター", "拍手", "銃声",
]
assert len(GM_NAMES) == 128, f"GM 音色は128個でなければならない（現在 {len(GM_NAMES)}個）"

# LLM に見せるときのカテゴリ（GM の並びは8音色ごとに系統が変わる）
GM_CATEGORIES = [
    (1, 8, "ピアノ"), (9, 16, "鍵盤打楽器・ベル"), (17, 24, "オルガン・アコーディオン"),
    (25, 32, "ギター"), (33, 40, "ベース"), (41, 48, "弦楽器(ソロ)・ハープ・ティンパニ"),
    (49, 56, "アンサンブル・合唱"), (57, 64, "金管"), (65, 72, "サックス・木管"),
    (73, 80, "笛・尺八"), (81, 88, "シンセリード"), (89, 96, "シンセパッド"),
    (97, 104, "シンセ効果音"), (105, 112, "民族楽器(シタール・琴・三味線)"),
    (113, 120, "打楽器・太鼓"), (121, 128, "効果音"),
]


def gm_menu() -> str:
    """128音色をカテゴリごとに並べた一覧（編成を決める工程のプロンプトに埋める）。"""
    out = []
    for lo, hi, cat in GM_CATEGORIES:
        names = " / ".join(f"{i}:{GM_NAMES[i - 1]}" for i in range(lo, hi + 1))
        out.append(f"  【{cat}】{names}")
    return "\n".join(out)


def gm_name(num: int) -> str:
    return GM_NAMES[num - 1] if 1 <= num <= len(GM_NAMES) else f"GM{num}"


# GM のドラムキット（SoundFont 側のプログラム番号。GM 準拠の音源は概ねこの並び）
DRUM_KITS = {"standard": 0, "room": 8, "power": 16, "electronic": 24,
             "tr808": 25, "jazz": 32, "brush": 40, "orchestra": 48}


#: ★**キットによって鍵盤の中身が違う**（2026-09-27・GeneralUser-GS の bank128 を読んで確かめた）。
#:   オーケストラ用キット（48）は **41〜53 番がティンパニ**で、ハイハット（42/44/46）・クラッシュ（49）・
#:   ライド（51/53）の番号に置くと**全部ティンパニで鳴る**（8分のハイハットがティンパニの連打になる）。
#:   ブラシのキットはクラップ（39）の番号がブラシのスネア、効果音のキットはどの番号も効果音。
#:   **中身は音源から読む**（`sf2_presets.drum_layout` → `kit_drums`）——手で表を書くと、音源を差し替えたときにずれる。
#:
#: ★2026-09-29: 以前はキットに無い楽器を**選んだあとで黙って別の番号へ運び**（`kit_remap`）、
#:   運び先も無ければ**番号のまま**（＝ティンパニ）鳴らしていた。いまは**キットの中身から「この楽器は何番」を読み**（`kit_notes`）、
#:   見つかったものだけを伴奏の選択肢に見せ（`kit_drum_menu`）、検査で見て（`song_arrange._issues`）、その番号で鳴らす（`midi_out`）。
#:   キットに無い打楽器は鳴らさない。
#:
#: GM の番号 → その番号に入っているはずの楽器の手がかり（サンプル名に含まれる語）。**GM の番号の意味**で、音源には依らない。
#: タムに「timpani」を入れているのは、オーケストラ用キットでタムの番号がティンパニなのは**意図どおり**だから
#: （編成が orchestra を選んだ曲で、伴奏は low_tom をティンパニのつもりで書く。選択肢にもサンプル名つきで見せる）。
#: 「sine」は 808 のキットのタム・コンガ（サイン波で作る音）、「click」は 808 のリムショット。
DRUM_EXPECT: dict[int, tuple[str, ...]] = {
    35: ("kick", "bass drum", "taiko"), 36: ("kick", "bass drum", "taiko"),
    37: ("rim", "stick", "click"), 38: ("snare",), 40: ("snare",), 39: ("clap",),
    42: ("hat",), 44: ("hat",), 46: ("hat",),
    49: ("crash", "cymbal"), 57: ("crash", "cymbal"), 51: ("ride", "cymbal"), 59: ("ride", "cymbal"),
    53: ("ride bell", "ride", "bell"),
    41: ("tom", "timpani", "sine"), 43: ("tom", "timpani", "sine"), 45: ("tom", "timpani", "sine"),
    47: ("tom", "timpani", "sine"), 48: ("tom", "timpani", "sine"), 50: ("tom", "timpani", "sine"),
    54: ("tambourine",), 56: ("cowbell",), 60: ("bongo",), 63: ("conga", "sine"), 64: ("conga", "tumba", "sine"),
    65: ("timbale",), 70: ("maracas", "shaker", "cabasa"), 76: ("wood",), 81: ("triangle",),
    # 2026-10-02（GM の 35〜81 を全部名前で書けるようにしたとき）。GeneralUser-GS の全キットのサンプル名を読んで確かめた
    # （`dev_tools/scratch/s5065b362_conform_打点ドラム.py kit`）。電子キットの 52 は「Reverse Cymbal」で、chinese_cymbal ではないので当たらない
    52: ("chinese", "china"), 55: ("splash",), 58: ("vibra",), 61: ("bongo",), 62: ("conga", "sine"),
    66: ("timbale",), 67: ("agogo",), 68: ("agogo",), 69: ("cabasa",), 71: ("whistle",), 72: ("whistle",),
    73: ("guiro",), 74: ("guiro",), 75: ("clave",), 77: ("wood",), 78: ("cuica",), 79: ("cuica",), 80: ("triangle",),
}
assert all(n in DRUM_EXPECT for n in GM_DRUM.values()), "GM_DRUM の番号に DRUM_EXPECT の手がかりが無い"


def _sf_kits() -> dict[int, str]:
    """音源のドラムキット一覧 {プログラム番号: 名前}。読めなければ空（そのときは GM の並び `DRUM_KITS` で読む）。"""
    from sf2_presets import drum_kits                      # noqa: E402（循環を避けてここで）
    from sf2_render import soundfont
    return drum_kits(soundfont() or "")


@functools.lru_cache(maxsize=16)
def kit_layout(kit_program: int) -> dict[int, str]:
    """キット（bank 128・`kit_program`）の {鍵盤: サンプル名}。音源から読めなければ空。"""
    from sf2_presets import drum_layout                    # noqa: E402
    from sf2_render import soundfont
    return drum_layout(soundfont() or "", kit_program)


#: 打楽器名 → そのキットで**その楽器が入っている鍵盤を探す手がかり**（サンプル名がすべての語を含む。どれか1組）。
#: ★2026-09-29 本人「サウンドフォントで取るべきものが取れてないってこと？」——そのとおりで、音源はキットごとに
#:   「どの鍵盤に何が入っているか」を全部持っているのに、GM の番号（標準キットの並び）でしか引いていなかった。
#:   GS のオーケストラ用キットは 41〜53 番をティンパニに使うので、ハイハットは 27〜29 番・ライドは 30 番・クラッシュは 57 番に
#:   置いてある（キットが不足しているのでも間違っているのでもない）。GM の番号に入っていなければ、ここで同じ楽器の鍵盤を探す。
#:   **選択肢に出すのはこうして見つかったものだけ**で、伴奏の LLM はそれを見て選ぶ（黙って別の楽器へ運ぶ `kit_remap` とは違う）。
#: タム・コンガ・ボンゴ・ティンバレスは低い／高いを名前で見分けられないので探さない（GM の番号の中身どおり）。
#: 探すのは**名前で取り違えようがない**楽器だけ。1語で探すと効果音のキットで「Car-Crash」（車の衝突音）がクラッシュに、
#: 「Sine-Triangle」（波形の名前）がトライアングルに当たった（2026-09-29 に全キットを読んで確かめた）。
DRUM_FIND: dict[str, tuple[tuple[str, ...], ...]] = {
    "bass_drum_1": (("kick",), ("bass drum",)), "acoustic_snare": (("snare",),), "hand_clap": (("clap",),),
    "closed_hi_hat": (("hat", "closed"),), "pedal_hi_hat": (("hat", "pedal"),), "open_hi_hat": (("hat", "open"),),
    "crash_cymbal_1": (("crash", "cymbal"),), "ride_cymbal_1": (("ride", "cymbal"),), "ride_bell": (("ride", "bell"),),
}
assert set(DRUM_FIND) <= set(GM_DRUM)


def kit_notes(kit: str) -> dict[str, tuple[int, str]] | None:
    """編成の `drum_kit` で**鳴らせる打楽器** {正準名: (そのキットでの鍵盤, サンプル名)}（2026-09-29）。**中身は音源から読む**。

    1. 打楽器名の GM の番号に、その楽器（`DRUM_EXPECT` の手がかり）が入っていればその番号（標準的なキットはすべてこれ）
    2. 入っていなければ、キットの中から**同じ楽器のサンプル名**の鍵盤を探す（`DRUM_FIND`。GM の番号に近いもの）
    3. どちらも無ければ、そのキットには無い（選択肢に出さず、鳴らさない）
    キットの指定が無い（空・none）・読めない・音源から中身を読めないときは None（呼び手は「キットで絞れない」として扱う）。
    """
    prog = drum_program(kit)
    if prog is None:
        return None
    lay = kit_layout(prog)
    if not lay:
        return None
    out: dict[str, tuple[int, str]] = {}
    for name, n in GM_DRUM.items():
        s = lay.get(n) or ""
        if s and any(w in s.lower() for w in DRUM_EXPECT[n]):
            out[name] = (n, s)
            continue
        for words in DRUM_FIND.get(name, ()):
            hit = min((k for k, smp in lay.items() if all(w in (smp or "").lower() for w in words)),
                      key=lambda k: (abs(k - n), k), default=None)
            if hit is not None:
                out[name] = (hit, lay[hit])
                break
    return out


def kit_drums(kit: str) -> dict[str, str] | None:
    """`kit_notes` の {正準名: サンプル名}（選択肢と検査が使う）。"""
    notes = kit_notes(kit)
    return None if notes is None else {k: s for k, (_n, s) in notes.items()}


def kit_drum_menu(kit: str) -> str:
    """伴奏の工程に見せる**打楽器の選択肢**（そのキットで鳴らせるものだけ・サンプル名つき）。

    キットの中身が読めないとき（音源が .sf3 等）は GM の一覧（`DRUM_NAMES`）をそのまま見せる。
    """
    k = strip_ascii_ws(kit)
    if k == "none":                                        # 書けるのは小文字の none だけ（2026-10-02・`drum_program`）
        return "（ドラムを使わない）"
    if kit_finding(k):
        return f"（ドラムの種類『{k}』が読めないので、ドラムは鳴りません）"
    have = kit_drums(k)
    if have is None:
        return " / ".join(DRUM_NAMES)
    return " / ".join(f"{n}（{s}）" for n, s in have.items()) or "（このキットで鳴らせる打楽器が無い）"


def kit_names() -> dict[str, int]:
    """**書けるドラムキットの名前 → プログラム番号（0始まり）**。1つのキットに名前は1つ（2026-10-02・判断6 本人「おすすめで」）。

    GM の並び（`DRUM_KITS`）にあるキットは**その名前**（`standard`・`orchestra` …・音源の番号が GM の並びと同じとき）、ほかの音源のキットは
    **`番号:音源のキット名`**（1始まりの番号・`2:Standard 2`・`27:Dance`）。音源の一覧が読めなければ `DRUM_KITS` の名前だけ。
    編成の依頼文の【ドラムキット】（`kit_menu`）・型の説明・検査（`kit_finding`）・書き出し（`drum_program`）が同じこの表を使う。
    """
    kits = _sf_kits()
    if not kits:
        return dict(DRUM_KITS)
    gm = {p: k for k, p in DRUM_KITS.items()}
    return {gm.get(p) or f"{p + 1}:{n}": p for p, n in sorted(kits.items())}


def kit_menu() -> str:
    """編成に見せる【ドラムキット】の一覧（`kit_names` の名前・GM の名前のキットはかっこに音源のキット名）。`none` は呼び手が添える。"""
    kits = _sf_kits()
    return " / ".join(f"{name}（{kits[p]}）" if kits and ":" not in name else name for name, p in kit_names().items())


def drum_program(kit: str) -> int | None:
    """編成の `drum_kit` → ch10 に出すプログラム番号（0始まり）。読めない・none・空なら None（2026-09-27）。

    ★**書けるのは `kit_names` の名前を1字も違わずに**（前後の半角の空白とタブだけ取り除く・2026-10-02・判断6 本人「おすすめで」。
      前後の全角の空白・改行などは 2026-10-03 から読まない＝`kit_finding` の所見・§0-6）。以前は大文字小文字を
      無視し、`49:何か` は番号だけ・全角の数字も読み、音源のキット名（`Orchestral`）・**先頭5字の前方一致**（`e`→electronic・`s`→standard・
      `electronic・tr808`→electronic）でも拾っていた（同じキットに書き方が何通りもあった）。違う書き方は編成の検査（`kit_finding`）が所見にする。
    ★**音源に無い番号は読めないとする**（2026-09-29）。以前は 0〜127 に丸めて渡し、音源に無い番号は音源の既定のキットで鳴っていた。
    """
    k = strip_ascii_ws(kit)
    if not k or k == "none":
        return None
    return kit_names().get(k)


def _kit_hint(kit: str) -> str | None:
    """読めないキット名に**添える一覧の名前**（大文字小文字だけ違う・音源のキット名だけ・番号が同じ）。所見に添えるためだけ——読み替えない。"""
    kits, low = _sf_kits(), (kit or "").strip().lower()
    head, tail = low.split(":", 1)[0].strip(), low.split(":", 1)[-1].strip()
    if low == "none":
        return "none"
    for name, p in kit_names().items():
        if name.lower() == low or (kits.get(p) or "").lower() == tail or (head.isascii() and head.isdigit() and int(head) - 1 == p):
            return name
    return None


def kit_finding(kit: str) -> str:
    """編成の `drum_kit` が読めなければ所見の文（読めれば空・`none` と空も空）。★読めなければドラムは鳴らさない（目録の 打-6・2026-09-29）。
    空は楽譜の `ArrangementSpec.drum_kit` の「音源の既定」なので、ここでは所見にしない（編成の工程の検査が空を所見にする）。"""
    k = strip_ascii_ws(kit)
    if not k or k == "none" or drum_program(k) is not None:
        return ""
    if why := edge_ws_problem(kit):                        # 前後に全角の空白・改行などを付けない（2026-10-03・§0-6）
        rest = kit_finding(trim_edge_ws(kit))
        return f"ドラムの種類{why}" + (f"。そのうえで: {rest}" if rest else "")
    names = kit_names()
    near = _kit_hint(k)
    return (f"ドラムの種類『{k}』は音源のキットに無い（読めない）。このままだとドラムは鳴らない"
            f"（代わりのキットで鳴らすことはしない）。"
            f"【ドラムキット】の一覧の名前（{' / '.join(names)} / none）を1字も違わずに1つだけ書く"
            + (f"——『{near}』のように" if near else ""))


# LLM に見せる音色一覧（プロンプトへ埋める）
TIMBRE_MENU = (
    "  鍵盤   piano / e_piano / organ / accordion\n"
    "  ギター ac_guitar（アコギ）/ guitar（クリーン）/ dist_guitar（歪み）\n"
    "  ベース bass（指弾き）/ pick_bass / synth_bass\n"
    "  弦・管 strings / brass / sax / flute / choir / harp\n"
    "  鍵盤打 bell / glocken / marimba\n"
    "  シンセ pad / saw_pad / lead / square_lead / pluck"
)
