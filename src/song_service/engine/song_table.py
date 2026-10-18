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

"""**曲を1枚で見る表**を組む（2026-09-23 に `tools/diagnosis.py` から移した）。

`歌詞|音高|小節:拍|長さ|コード|和音での役割|ドラム|<追加声部>` の並びと、
小節ごとの編成表。**良し悪しは何も判断しない**——materials を並べるだけ。

★**なぜ engine へ移したか**: もともと `inspect_song`（ミクが「違和感がある」と
  言われたときに呼ぶ道具）の中にあった。だが**表そのものは道具ではない**——
  楽譜から組み立てるだけで、LLM も呼ばず、判断も入らない。
  声部を書く工程（`harmony.write_voices`）が同じ表を欲しくなったとき、
  `engine` から `tools` を呼ぶことになり、**層の向き（道具 → 本体）が壊れる**。

  | 層 | 中身 | 向き |
  |---|---|---|
  | `tools/` | ミクの音楽担当が呼ぶ道具 | → `engine` を使う |
  | `engine/` | 曲を作る本体 | **`tools` を使わない** |

  写しを作って逃げる手もあったが、**表が2つになれば片方は必ず古くなる**
  （同じ形の事故を 2026-09-22 に3回踏んでいる）。**置き場を正した。**

`inspect_song` はこれを呼ぶ薄い包みになっている。
"""
from __future__ import annotations

import re
from pathlib import Path

from song_checks import strip_ascii_ws                     # 半角の記法の前後の空白（§0-6・2026-10-03）
from song_schema import SongScore

# ★`accomp_edit` / `song_pipeline` は**関数の中で読む**（移す前と同じ形のまま）。
#   `song_pipeline` はこのモジュールを使う側にもなりうるので、**上で読むと循環する**。

_PITCH = "C C# D D# E F F# G G# A A# B".split()
#: 和音の中での役割。**「外」＝コードに無い音**（悪いとは限らない——経過音・テンションのこともある）
_DEGREE = {0: "root", 2: "9th", 3: "m3", 4: "3rd", 5: "11th", 7: "5th", 9: "13th", 10: "7th", 11: "M7"}


# ============================ 音名・コード名の綴り（2026-09-30・記-16）============================
#
# ★**黒鍵（ピッチクラス 1 3 6 8 10）を、曲・区間の調に合わせて ♯ か ♭ で綴る**。以前は常に ♯ だけで、
#   F の曲の B♭ が `A#`・`A#5` と出ていた（表を読む声部・コーディネーターの LLM には、F の曲に調の外の音があるように見える）。
#   高さは変えない（綴りだけ）。読み手（音名の `song_llm.pitch_name_to_midi`・コード名の `song_design._CHORD_RE`・
#   `song_audio.chord_notes`）はどれも `b` を読める。
#   決め方: ①その調の音階の音（短調は相対長調の7音＋上げた6度・7度）は音階どおり ②残りの黒鍵は、調号が ♭ の調なら ♭・
#   ♯ の調なら ♯ ③C／Am（調号なし）と調が分からないときは**これまでどおり**（音名は ♯・コード名は書かれたまま）。
#   E♯・B♯・C♭・F♭ のような白鍵の綴りは使わない（白鍵は常に幹音の名前）。
# ★**綴りは国際式の音名**（2026-10-01・判-A/B）: 変化記号は ASCII の `#` と `b`（`Bb4`・`A#4`・コード名 `Bb`）——LLM が書く音名
#   （`song_llm.PITCH`）・コード記号と同じ書き方。以前は ♭ を記号で `B♭4` と出し、奏法・歌の崩しの表は階名（`シ♭4`）だった。
_LETTERS = "CDEFGAB"
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_BLACK = (1, 3, 6, 8, 10)


def black_key_names(key_name: str | None) -> dict[int, str]:
    """調名 → 黒鍵の綴り `{ピッチクラス: 'Bb'}`。C／Am・空・読めない調は `{}`（＝これまでどおり ♯）。"""
    from song_design import parse_key                      # 調名の読み手は1つ（設計の型の検証・移調と同じ）
    got = parse_key(key_name or "")
    if got is None:
        return {}
    tonic, minor = got
    li = _LETTERS.index("".join((key_name or "").split())[0].upper())
    mli, mpc = ((li + 2) % 7, (tonic + 3) % 12) if minor else (li, tonic)   # 相対長調の主音（文字とピッチクラス）

    def _name(letter: str, pc: int) -> str | None:
        d = (pc - _LETTER_PC[letter]) % 12
        return letter if d == 0 else letter + "#" if d == 1 else letter + "b" if d == 11 else None

    scale: dict[int, str] = {}
    for i, step in enumerate((0, 2, 4, 5, 7, 9, 11)):
        pc = (mpc + step) % 12
        if (n := _name(_LETTERS[(mli + i) % 7], pc)):
            scale[pc] = n
    flat = any(n.endswith("b") for n in scale.values())
    sharp = any("#" in n for n in scale.values())
    if minor:                                              # 短調の上げた6度・7度（導音）は音階の文字のまま上げる
        for deg, step in ((5, 9), (6, 11)):
            pc = (tonic + step) % 12
            if pc not in scale and (n := _name(_LETTERS[(li + deg) % 7], pc)):
                scale[pc] = n
    if not flat and not sharp:                             # C／Am: 調号が無く、決める根拠が無い
        return {}
    # 音階の外の黒鍵は調号の向きで（黒鍵の半音上はいつも白鍵なので、♭ はその幹音に付ける）
    return {pc: scale.get(pc) or (_PITCH[(pc + 1) % 12] + "b" if flat else _PITCH[pc]) for pc in _BLACK}


def spell_pitch(k: int, key_name: str | None = None) -> str:
    """MIDI 番号 → 国際式の音名（`C4`＝60・F の曲の B♭ は `Bb4`）。調が無ければ ♯ で綴る（`A#4`）。

    ★**LLM に高さを見せる所はどこもこれで綴る**（2026-10-01・判-A/B/D）: 曲の表（`song_table`）・声部の推奨音域・奏法と歌の崩しの表
      （`artic_plan.pitch_name`）・歌い方の表の音域（`vocal_expression.table`）・伴奏と彩りの依頼文の歌の音域と和音の積み方
      （`song_arrange._voice_note`）・ハモリ追加の主旋律（`song_llm.generate_added_voice`）。読み手は `song_llm.pitch_name_to_midi` 1つ。
    """
    # 調が無ければ調名の読み手（`song_design`）を読まない——型の説明を組む `song_arrange` の読み込み中にも呼ばれ、読むと循環する
    name = (black_key_names(key_name) if key_name else {}).get(k % 12) or _PITCH[k % 12]
    return name + str(k // 12 - 1)


def spell_chord(chord: str, key_name: str | None = None) -> str:
    """コード名の根音と分数の低音を、調に合わせて綴り直す（`A#maj7/D#` → F の曲なら `Bbmaj7/Eb`）。**種類の字は触らない**。

    調が無い・C／Am・読めない綴り（`N.C.` など）は書かれたまま返す。
    ★変化記号は ASCII の `#` `b` だけ読む（2026-10-02・仕様書 §0-1。記号の ♯♭ の名前は読めない綴りとして書かれたまま返す）。
    """
    names = black_key_names(key_name)
    if not names:
        return chord

    def _one(note: str) -> str:
        pc = (_LETTER_PC[note[0]] + (1 if note[1:] == "#" else -1 if note[1:] == "b" else 0)) % 12
        return names.get(pc, note)

    m = re.fullmatch(r"([A-G][#b]?)(.*?)(?:/([A-G][#b]?))?", strip_ascii_ws(chord))
    if not m:
        return chord
    root, q, bass = m.groups()
    return _one(root) + q + (f"/{_one(bass)}" if bass else "")


def key_regions(song) -> list[tuple[int, str]]:
    """曲の記録（`Song`）→ `[(開始tick, 調名), …]`（曲頭から順）。区間の調（`key_of`）か時間軸が無ければ空。

    区間の調は組み立て（`song_place`）が決めて曲の記録に残したもの（`Song.key_of`＝区間→調）をそのまま使い、
    区間の位置は時間軸（`Song.timeline`）から引く——転調の決め方をここで写し直さない。
    曲頭の空き小節（`LEAD_IN`）のように調の無い区間は、前の区間の調のまま（曲頭なら最初の調）。
    ★**時間軸を持たない曲（カバー）は区間の小節（`Song.bars_of`）から引く**（2026-10-01・判-C）。カバーも推定した調で C 基準から
      移すので、声部の表の頭に調の名前を出し、奏法の表をその調で綴る（以前は時間軸が無いので空＝♯ で綴っていた）。
    """
    key_of, tl = getattr(song, "key_of", None) or {}, getattr(song, "timeline", None)
    if key_of and tl is None and getattr(song, "bars_of", None) and getattr(song, "score", None) is not None:
        from song_design import bar_ticks
        bt = bar_ticks(song.score.time_signature or "4/4")
        out = []
        for kind, (b0, _b1) in sorted(song.bars_of.items(), key=lambda kv: kv[1][0]):
            k = key_of.get(kind)
            if k and not (out and out[-1][1] == k):
                out.append((0 if not out else (int(b0) - 1) * bt, k))
        return out
    if not key_of or tl is None:
        return []
    out: list[tuple[int, str]] = []
    for kind, t0, _t1 in tl.key_ranges():
        k = key_of.get(kind)
        if not k or (out and out[-1][1] == k):
            continue
        out.append((0 if not out else t0, k))
    return out


def shift_regions(shift_at, bar_ticks: int, n_bars: int) -> list[tuple[int, int]]:
    """tick → 移調量（`harmony.section_shift_at`）→ `[(開始小節, 移調量), …]`（移調量が変わる小節だけ・曲頭から順）。
    表の頭（C 基準の断り書き）と声部の推奨音域の行が使う。小節の頭の移調量で数える。"""
    out: list[tuple[int, int]] = []
    for b in range(max(1, n_bars)):
        sh = int(shift_at(b * bar_ticks))
        if not out or out[-1][1] != sh:
            out.append((b + 1, sh))
    return out


def c_basis_header(shift_at, bar_ticks: int, n_bars: int, keys: list[tuple[int, str]] | None = None) -> str:
    """C 基準（移調前）で見せる表・依頼文の頭の断り書き（区間ごとの移調量と、分かれば調の名前）。

    ★**声部の表（`song_table`）とハモリ追加の依頼文（`song_llm.generate_added_voice`）が同じ1つを使う**（2026-10-02・ず-12①）。
    移調量は `shift_at`（tick → 移調量・`harmony.section_shift_at`）から小節の頭で引く（`shift_regions`）。
    """
    return ("音高・コード名は **C メジャー／A マイナー基準（移調前）** で出す（旋律・コードを書いたときと同じ物差し）。"
            "鳴るときは機械が区間の調へ移す: " + " → ".join(
                f"{b}小節〜 {sh:+d}半音" + (f"（{k}）" if (k := _key_at(keys, (b - 1) * bar_ticks)) else "")
                for b, sh in shift_regions(shift_at, bar_ticks, n_bars)))


def c_basis_pitch(k: int, tick: int, shift_at) -> str:
    """鳴る高さ（MIDI）→ C 基準の音名（その位置の区間の移調量だけ戻す・♯ で綴る）。声部の表・ハモリ追加の主旋律が使う。"""
    return spell_pitch(int(k) - int(shift_at(tick)))


def c_basis_chord(chord: str, tick: int, shift_at) -> str:
    """鳴るコード名 → C 基準のコード名（その位置の区間の移調量だけ戻す）。和音なし（`N.C.`）と読めないコード名は書かれたまま（表示だけ）。"""
    from song_design import transpose_chord
    try:
        return transpose_chord(chord, -int(shift_at(tick)))
    except ValueError:
        return chord


def _hold_shown(lyric: str) -> str:
    """表の歌詞の欄: 楽譜の伸ばす歌詞（V6 の `-`）を、LLM が書く伸ばす印 `_`（詞・声部と同じ）で見せる（2026-09-30・記-4 の残り）。

    ★**LLM に歌詞を見せる表はどれもこれを通す**（2026-10-01・ず-11）: 曲の表・歌の崩しの表（`vocal_artic.table`）・
      歌い方の表（`vocal_expression.table`）・ハモリ追加の主旋律（`song_llm.generate_added_voice`）。以前は歌の崩しと歌い方の表が
      楽譜の `-` のまま見せていた。
    """
    return "_" if lyric == "-" else lyric


def _key_at(keys: list[tuple[int, str]] | None, tick: int) -> str | None:
    if not keys:
        return None
    cur = keys[0][1]
    for t0, k in keys:
        if t0 > tick:
            break
        cur = k
    return cur


# ============================ 伴奏の打点の図の記号（2026-09-30・記-14）============================
#
# ★**伴奏の書き方（`song_arrange._GRID`・`_DRUM_TAB`）と同じ記号にする**: `X`＝強く・`x`＝ふつう・`.`＝弱く、打点の無い枠は `-`。
#   以前は `o`＝弱く・`.`＝鳴っていない で、伴奏の記法（`o`＝ふつう・`.`＝弱い打点）と逆だった。しかも弱いの境目が 60 で、
#   書いた `.`（62）が `x`（ふつう）と出ていた。境目は書いた強さ（`_VEL`: X=110・x=88・.=62）の中間。
#   ドラムは組み立てが打楽器ごとに一定の差（`_DRUM_BASE`: キック +8・ハイハット −14 …）を足しているので、それを戻してから比べる
#   ——戻さないと `x` で書いたハイハット（74）と `.` で書いたキック（70）が同じ強さに見え、書いた記号と区別できない。
_NO_HIT = "-"
_RANK = {_NO_HIT: 0, ".": 1, "x": 2, "X": 3}


def _hit_mark(vel: int) -> str:
    """書いた強さ（ドラムは打楽器ごとの差を戻したもの）→ 伴奏の記法の記号。"""
    from song_arrange import _VEL                          # 記号 → 強さの表は伴奏の組み立てのもの1つ
    strong, normal, weak = _VEL["X"], _VEL["x"], _VEL["."]
    if vel >= (strong + normal) / 2:
        return "X"
    return "x" if vel >= (normal + weak) / 2 else "."


def _drum_base_by_note() -> dict[int, int]:
    """GM の打楽器の番号 → 組み立てが足した強さの差（`song_arrange._DRUM_BASE`）。"""
    from song_arrange import _DRUM_BASE
    from timbre_map import GM_DRUM
    return {GM_DRUM[k]: v for k, v in _DRUM_BASE.items() if k in GM_DRUM}


def _chord_root(chord: str) -> int | None:
    """コード名 → ルートのピッチクラス。読めない綴りは None（判定から外す）。

    ★**変化記号は ASCII の `#` `b` だけ読む**（2026-10-02・仕様書 §0-1・§3。和音の読み手 `song_audio.chord_notes`・
      `chord_bass_pc`・綴りの部品 `spell_chord` と同じ）。以前はここだけ記号の ♯♭ も同じ意味で読んでいた。"""
    c = (chord or "").split("/")[0].strip()
    if not c or c[0].upper() not in "CDEFGAB":
        return None
    root = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[c[0].upper()]
    for ch in c[1:]:
        if ch == "#":
            root += 1
        elif ch == "b":
            root -= 1
        else:
            break
    return root % 12


def song_table(score: SongScore, song_dir: Path, *, start_bar: int = 1, end_bar: int = 0,
               keys: list[tuple[int, str]] | None = None,
               shift_at=None, lead_track: str = "") -> tuple[str, int, int, int]:
    """表を組む。戻り値は (本文, 開始小節, 終了小節, 出した音数)。

    ★**インスト（`form=instrumental`・歌0音）も組む**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §7-2）。
      歌の行の代わりに**主旋律の行**（伴奏 MIDI の `lead_track` という名前のトラックの音・歌の行と同じ列から歌詞を除いたもの）と、
      いつもの小節ごとの伴奏の表を出す。`lead_track` が分からない・見つからないときは、そう書いて小節の表だけ出す（推し量らない）。
      歌もの・アカペラはいままでどおり（歌0音なら「未生成」）。

    `song_dir` には `accompaniment.mid` / `mix.json` があれば読む（無くても組める）。
    `keys`（`[(開始tick, 調名)]`・`key_regions(song)`）を渡すと、音名・コード名をその調の綴り（♯／♭）で出す（2026-09-30・記-16）。
    楽譜（`SongScore`）は調を持たないので、渡されなければこれまでどおり ♯ で綴る。
    `shift_at`（tick → その区間の移調量・`harmony.section_shift_at`）を渡すと、音高・コード名を**移調前（C メジャー／A マイナー基準）**
    で出す（2026-09-30・記-3。声部に見せる表＝声部が書く高さと同じ物差し）。C 基準の表は調号の無い調なので ♯ で綴る
    （`keys` は調の名前を頭の断り書きに出すためだけに使い、綴りには使わない）。和音での役割・声部と主旋律の差は移しても変わらない。
    """
    from song_pipeline import _chord_tones                 # ★循環を避けて関数の中で読む
    from artic_plan import tick_key                        # 位置の書き方（奏法・歌の崩しの表と同じ1つ）
    from song_design import bar_ticks, position_text
    from accomp_edit import NO_CHORD_MARK                  # 和音なしの書き方（推定が和音の取れない小節に書くものと同じ）

    s = score
    inst = s is not None and getattr(s, "form", "song") == "instrumental"
    if s is None or (not s.vocal_notes and not inst):
        return "曲が未生成。先に compose か load_existing_song を。", 0, 0, 0
    Q = 480
    # ★**1小節の長さは曲の拍子から**（2026-09-30・記-1）。以前は `Q * 4`（4/4）の決め打ちで `小節:拍` を出し、声部は曲の拍子の
    #   小節長で読む（`harmony.at_to_tick`）ので、3/4・6/8 の曲では表の `2:0.00` の音と同じ位置に書いた声部が1拍前に置かれ、
    #   後ろの小節ほどずれた。位置の書き方も奏法・歌の崩しの表と同じ `tick_key` にした
    ts = s.time_signature or "4/4"
    BAR = bar_ticks(ts)
    BPB = BAR / Q                                          # 1小節の4分音符の数
    notes = sorted(s.vocal_notes, key=lambda n: n.start_tick)
    last_bar = (int(max(n.start_tick + n.duration_tick for n in notes) // BAR) + 1) if notes else 1   # インストは伴奏の長さから下で決まる

    # コードは**楽譜のものが正**。無ければ（カバー等）伴奏MIDIから推定する
    spans = sorted(s.chords, key=lambda c: c.start_tick)
    est: list[str] = []
    tracks = []
    mid = song_dir / "accompaniment.mid"
    if mid.exists():
        try:
            from accomp_edit import estimate_bar_chords, load_accomp
            tracks = load_accomp(mid)
            # ★**伴奏は歌より後ろまで続くことがある**（後奏）。歌の最後で切ると、
            #   伴奏だけの小節が表から落ちる。曲の長さは**両方の長いほう**で取る。
            _spb = 60.0 / float(s.tempo)
            _acc_end = max((n.start_sec + n.dur_sec) / _spb for t in tracks for n in t.notes)
            last_bar = max(last_bar, int(-(-round(_acc_end * 4) / 4 // BPB)))
            if not spans:
                # 小節は曲の拍子で切る（2026-10-01。以前は拍子を渡さず 4/4 で切り、3/4 のカバーで推定した和音が表の小節とずれた）
                est = estimate_bar_chords(tracks, float(s.tempo), 0, last_bar, beats_per_bar=BPB)
        except Exception:
            tracks = []

    # ★**範囲は必ず曲の中に収める**（2026-09-20 に落ちた）。`end_bar` は LLM が決めるので、
    #   前の曲の小節数（98）をそのまま渡してきた。45小節の曲で `IndexError` になり、
    #   **診断の途中で止まった**。呼ぶ側の数字を信じない。
    b0 = max(1, min(start_bar, last_bar))
    b1 = min(max(b0, end_bar or last_bar), last_bar)

    def chord_at(tick: int) -> str:
        for c in spans:
            if c.start_tick <= tick < c.start_tick + c.duration_tick:
                return c.chord
        # ★**コードの無い位置（楽譜にも推定にも無い）は `N.C.`**（2026-10-02）。以前は記法に無い `-` を出していた
        #   （推定が和音の取れない小節に書く印は同じ日に `-` から `N.C.` にした・`accomp_edit.NO_CHORD_MARK`）。表は見せるだけで、
        #   和音での役割は根音が読めないので `?`
        bar = tick // BAR
        return est[bar] if bar < len(est) else NO_CHORD_MARK

    spb = 60.0 / float(s.tempo)
    hits = sorted(n.start_sec / spb for t in tracks if t.is_drums for n in t.notes)

    out = [f"『{s.title}』 BPM{s.tempo:.0f} {s.time_signature} / "
           + ("インスト（歌は無い・主旋律は楽器が弾く）" if inst else f"歌 {len(notes)}音") + " / "
           f"{last_bar}小節" + (f"（{b0}〜{b1}小節を表示）" if (b0, b1) != (1, last_bar) else ""),
           position_text(ts)]
    if shift_at is not None:
        # ★**声部に見せる表は C 基準（移調前）**（2026-09-30・記-3）。旋律・コード・伴奏の音名と同じ物差しで見せ、声部も同じ物差しで
        #   書かせて、組み立てが区間ごとに移す（`harmony.voice_to_track`）。以前は表も声部も実音で、同じ `ミ5` が工程で違う高さを指した
        #   断り書きと C 基準の綴りはハモリ追加の依頼文と同じ部品（`c_basis_header`・`c_basis_pitch`・`c_basis_chord`・2026-10-02）
        out.append(c_basis_header(shift_at, BAR, last_bar, keys))

        def pname(k: int, tick: int) -> str:
            return c_basis_pitch(k, tick, shift_at)

        def cname(tick: int) -> str:
            return c_basis_chord(chord_at(tick), tick, shift_at)
    else:
        if keys:
            out.append("調: " + " → ".join(f"{t0 // BAR + 1}小節〜 {k}" for t0, k in keys)
                       + "（音名・コード名はこの調の綴りで出す。♭ の調なら `Bb4`・`Eb` のように `b` で書く）")

        def pname(k: int, tick: int) -> str:
            return spell_pitch(k, _key_at(keys, tick))

        def cname(tick: int) -> str:
            return spell_chord(chord_at(tick), _key_at(keys, tick))
    if s.extra_voices:
        # ★音量まで出す（2026-09-20）。同じ3度でも **gain が違えば「浮く」かどうかが変わる**。
        out.append("追加声部: " + " / ".join(
            f"『{v.title}』{len(v.notes)}音 音量{v.gain_db:+.1f}dB" for v in s.extra_voices))
    # ★**調声（歌い方）もまとめて出す**（2026-09-20 の棚卸しで抜けが判明）。
    #   1音ずつ出すと 323行ぶん増えるので、**値の種類**だけ示す。全部同じなら「一律」。
    tun: dict[str, set] = {}
    for n in notes:
        for f, val in (n.tuning.model_dump() if n.tuning else {}).items():
            if val is not None:
                tun.setdefault(f, set()).add(round(float(val), 2))
    if tun:
        out.append("調声（歌い方）: " + " / ".join(
            f"{f}={next(iter(vs)):.2f}" if len(vs) == 1 else
            f"{f}={min(vs):.2f}〜{max(vs):.2f}（{len(vs)}種）" for f, vs in sorted(tun.items())))
    mj = song_dir / "mix.json"
    if mj.exists():
        try:
            import json as _json
            out.append("音量の設定(mix.json): " + _json.dumps(
                _json.loads(mj.read_text(encoding="utf-8")), ensure_ascii=False))
        except Exception:
            pass
    # ★**歌詞をひと続きの文としても出す**。音符の列だけだと語として読みにくく、
    #   「いかあならし」のような**日本語として壊れた箇所**が目に入らない（実際に見落とされていた）。
    #   ★★**`lyrics_full` ではなく音符から組む**（2026-09-20）。
    #     23曲を突き合わせたら**1曲で食い違っていた**（`lyrics_full` 166字 / 音符 161字）。
    #     区間を書き直しても `lyrics_full` が更新されないため。
    #     **実際に歌うのは音符のほう**なので、そちらから組めばずれようがない。
    #   ★伸ばしは同じ表の行と同じ `_` で見せる（`_hold_shown`・2026-10-03・再レビュー）。以前は通し文だけ楽譜の `-` のままで、
    #     同じ表に伸ばしの書き方が2通り出ていた。`lyrics_full` との突き合わせは楽譜の形（`-`）どうしで比べる
    _raw = "".join(n.lyric for n in notes)
    _line = "".join(_hold_shown(n.lyric) for n in notes)
    if _line:
        out.append(f"歌詞（音符から組んだ通し文・『_』は伸ばし）: {_line}")
        if s.lyrics_full and s.lyrics_full.replace("\n", "").replace(" ", "") != _raw:
            out.append("※`lyrics_full`（作ったときの詞）とは食い違っている"
                       "——**編集で音符側だけが変わった**。歌われるのは上の通し文。"
                       "**`lyrics_display`（字幕・投稿に使う見せる歌詞）も同じだけ古い**"
                       "——人に見せる前に書き直すこと。")
    # ★**見せる歌詞を出す**（2026-09-21）。`lyrics_full` は歌うためのかな表記（を→お・長音は母音）で
    #   読み物にならないので、字幕・投稿には使えない。無い曲もある（カバーなど、見せる歌詞を書かない曲）。
    if getattr(s, "lyrics_display", ""):
        # ★行番号を付ける（2026-10-12）。`replace_region` の `display` は行番号で指す
        out.append("見せる歌詞（字幕・投稿用。1行＝1フレーズ。`replace_region` の `display` で行番号を指して直す）:\n"
                   + "\n".join(f"  {i}: {t}" for i, t in enumerate(s.lyrics_display.split("\n"), 1)))
    elif _line:
        out.append("※**見せる歌詞（漢字まじり）が無い曲**（カバーなど、見せる歌詞を書かない曲）。"
                   "字幕や投稿に使うなら書き起こしが要る。")
    # ★**追加声部（ハモリ）の音符も渡す**（2026-09-20 本人の指摘:
    #   「ハモリはどこでどう渡すかみたいな情報ってちゃんと入ってるんだっけ」）。
    #   それまで出していたのは `追加声部: 『third_down』114音` という**1行だけ**で、
    #   114音の中身は一度も渡っていなかった。
    #   ★**実害が出た**——ハモリの違和感を訴えた試験で、ミクは「主旋律 C6 に対しハモリ A5」と
    #   **具体的な音名**を答えたが、それは見たのではなく**「third_down という名前だから3度下だろう」
    #   という推測**だった（数字は結果的に合っていた）。正しい振る舞いは
    #   「材料に出ていないので確かめられない」と言うこと。**材料が無いこと自体に気づけない**のが
    #   一番危ないので、**主旋律の横に並べて**渡す（突き合わせを機械側でやる・この表の原則）。
    hv = [{n.start_tick: n for n in v.notes} for v in s.extra_voices]
    main_ticks = {n.start_tick for n in notes}
    #: 主旋律と同じ位置に無いハモリの音（輪唱など）。これも落とさずに**独立した行**で出す
    orphans = sorted(((n.start_tick, vi, n) for vi, v in enumerate(s.extra_voices)
                      for n in v.notes if n.start_tick not in main_ticks),
                     key=lambda x: x[0])
    out.append("")
    if not inst:
        out.append("歌詞|音高|小節:拍|長さ|コード|和音での役割|ドラム"
                   + "".join(f"|{v.title}" for v in s.extra_voices))
    if s.extra_voices:
        out.append(f"※追加声部の欄は `音高(主旋律との差・半音)`。**−が下・＋が上**。"
                   f"その位置に音が無ければ `-`"
                   + (f"。主旋律とずれた位置の音は `[{s.extra_voices[0].title}のみ]` の行で出す"
                      if orphans else ""))

    def _silence(from_tick: int, to_tick: int, tail: str = "") -> str | None:
        """歌が無い区間を**小節つきで**印す。★これが無いと前奏・間奏・後奏が表から消える
        （2026-09-20 に本人が指摘: 「歌詞が無いところが LLM に渡らなくない？」）。
        小節を書くのは、**下の編成の表と突き合わせられるようにする**ため——
        拍数だけ出しても「何小節目の話か」が分からず、伴奏が何をしているか見に行けない。"""
        gap = (to_tick - from_tick) / Q
        if gap < 0.5:
            return None
        bf, bt = from_tick // BAR + 1, max(from_tick, to_tick - 1) // BAR + 1
        where = f"{bf}小節" if bf == bt else f"{bf}〜{bt}小節"
        return f"--- 歌なし {gap:.2f}拍（{where}）{tail}---"

    def _row(lyric: str, key: int, tick: int, dur: int, extra: str) -> str:
        ch = chord_at(tick)
        root = _chord_root(ch)
        deg = "?" if root is None else _DEGREE.get((key - root) % 12, "外")
        if root is not None and deg == "外" and (key % 12) in _chord_tones(ch):
            deg = "構成音"                                   # 7th/9th 以外の綴り（sus/dim 等）の救済
        bt = tick / Q
        return (f"{lyric}|{pname(key, tick)}|{tick_key(tick, BAR)}|"
                f"{dur / Q:.2f}|{cname(tick)}|{deg}|"
                + ("d" if any(abs(x - bt) < 0.06 for x in hits) else "") + extra)

    # 主旋律と、主旋律からずれた位置のハモリを**1本の並び**にしてから出す
    events = [(n.start_tick, 0, n) for n in notes] + [(t, 1, (vi, n)) for t, vi, n in orphans]
    prev_end = None
    shown = 0
    for tick, kind, payload in sorted(events, key=lambda e: (e[0], e[1])):
        bar = tick // BAR
        end = tick + (payload.duration_tick if kind == 0 else payload[1].duration_tick)
        if not (b0 - 1 <= bar < b1):
            prev_end = end if prev_end is None else max(prev_end, end)
            continue
        if prev_end is None:                               # 前奏（表示範囲の頭から最初の音まで）
            if (line := _silence((b0 - 1) * BAR, tick, "＝前奏 ")):
                out.append(line)
        elif (line := _silence(prev_end, tick)):
            out.append(line)
        if kind == 0:
            n = payload
            cols = ""
            for vi, hn in ((i, d.get(tick)) for i, d in enumerate(hv)):
                cols += (f"|{pname(hn.key, tick)}({hn.key - n.key:+d})" if hn else "|-")
            out.append(_row(_hold_shown(n.lyric), n.key, tick, n.duration_tick, cols))
        else:                                              # 主旋律と位置が違うハモリの音
            vi, hn = payload
            cols = "".join(f"|{pname(hn.key, tick)}" if i == vi else "|-" for i in range(len(hv)))
            out.append(_row(f"[{s.extra_voices[vi].title}のみ]{_hold_shown(hn.lyric)}",
                            hn.key, tick, hn.duration_tick, cols))
        prev_end = end if prev_end is None else max(prev_end, end)
        shown += 1

    if prev_end is not None and (line := _silence(prev_end, b1 * BAR, "＝後奏 ")):
        out.append(line)                                   # 後奏（最後の音から曲の終わりまで）

    if inst:
        # ★**インストの主旋律の行**（2026-10-03）。主旋律は伴奏の楽譜の1パートとして伴奏 MIDI に入る（§5-2）ので、
        #   その名前のトラックの音を歌の行と同じ列（歌詞を除く）で出す。名前が分からなければ推し量らずにそう書く
        #   音は楽譜の `lead`（主旋律の楽器のパート・`SongScore.lead`）が正。無い古い楽譜だけ伴奏 MIDI のトラックから読む
        lp = s.lead                                        # 無ければ中身が空（F1）
        if lp.notes:
            lead = sorted((n2.start_tick, n2.key, n2.duration_tick) for n2 in lp.notes)
            src = f"楽譜の主旋律のパート・楽器 {lp.timbre}"
        else:
            spb_ = 60.0 / float(s.tempo)
            lead = sorted((round(n2.start_sec / spb_ * Q), n2.key, round(n2.dur_sec / spb_ * Q))
                          for t in tracks if lead_track and t.name == lead_track and not t.is_drums for n2 in t.notes)
            src = f"伴奏 MIDI のトラック『{lead_track}』"
        if not lead:
            out.append("※主旋律は伴奏の楽器の1つが弾いている（下の小節ごとの表のレーン）。"
                       + (f"伴奏 MIDI に主旋律のトラック『{lead_track}』が見つからず、" if lead_track else
                          "主旋律のトラックの名前が分からず、")
                       + "主旋律だけの行は出していない。")
        else:
            out.append(f"主旋律（{src}）: 音高|小節:拍|長さ|コード|和音での役割|ドラム")
            prev_end = None
            for tick, key, dur in lead:
                if not (b0 - 1 <= tick // BAR < b1):
                    prev_end = tick + dur if prev_end is None else max(prev_end, tick + dur)
                    continue
                if (line := _silence((b0 - 1) * BAR if prev_end is None else prev_end, tick,
                                     "＝前奏 " if prev_end is None else "")):
                    out.append(line.replace("歌なし", "主旋律なし"))
                out.append(_row("", key, tick, dur, "")[1:])
                prev_end = tick + dur if prev_end is None else max(prev_end, tick + dur)
                shown += 1
            if prev_end is not None and (line := _silence(prev_end, b1 * BAR, "＝後奏 ")):
                out.append(line.replace("歌なし", "主旋律なし"))

    if tracks:
        # ★**伴奏は「小節あたり何音」では足りない**（2026-09-20 本人の指摘:
        #   「小節内で歌声がない部分も送ってる？そこに伴奏の違和感がある可能性もある」）。
        #   個数だけだと**拍のどこで鳴っているか**が分からず、歌が休んでいる拍で伴奏が
        #   変な所を叩いていても見えない。16分の枠に打点を落として**形として**出す。
        #   ★**畳まない**（同・本人指示）。「前の小節と同じ」で省くと、前の行を覚えて
        #   突き合わせる仕事が LLM に残る。**行をまたぐ参照は取り違えの元**。
        SLOTS = int(round(BPB * 4))                        # 1小節の16分の数（拍子から・4/4=16 / 3/4・6/8=12）
        lanes: dict[str, list[list[str]]] = {}
        counts: dict[str, list[int]] = {}
        progs: dict[str, int] = {}                         # レーン→GM音色（宣言したトラックのもの）
        lows: list[int | None] = [None] * last_bar
        drum_base = _drum_base_by_note()
        for t in tracks:
            ln = "drums" if t.is_drums else t.name
            grid = lanes.setdefault(ln, [[_NO_HIT] * SLOTS for _ in range(last_bar)])
            cnt = counts.setdefault(ln, [0] * last_bar)
            # ★**音色は「宣言したトラック」から取る**（2026-09-20）。
            #   伴奏MIDIは区間ごとにトラックが分かれており、`program_change` を出しているのは
            #   **各チャンネルの最初のトラックだけ**。残りは `load_accomp` が prog0（ピアノ）として
            #   読む。音数の多いほうを代表に選んだら**全部ピアノ**と表示して嘘をついた。
            #   MIDI は正しく、壊れていたのは表示のほう。
            if not t.is_drums and t.program and ln not in progs:
                progs[ln] = t.program
            for n2 in t.notes:
                # ★**先に絶対拍で16分へ量子化してから**小節とスロットを出す。
                #   tick→秒→拍の往復で小節頭が 3.9958拍 のように前へずれており、
                #   小節→スロットの順に丸めると**小節頭の打点を落とす**（実際に落としていた）。
                q = round(n2.start_sec / spb * 4) / 4
                bar, slot = int(q // BPB), int(round((q % BPB) * 4))
                if not (0 <= bar < last_bar and 0 <= slot < SLOTS):
                    continue
                # ★**強弱も文字で持たせる**（2026-09-20 の棚卸しで抜けが判明）。
                #   打点の有無だけだと「のっぺりしている／抑揚が無い」が見えない。
                #   1文字に載せるので**トークンは増えない**。
                mark = _hit_mark(n2.vel - (drum_base.get(n2.key, 0) if t.is_drums else 0))
                if _RANK[mark] > _RANK[grid[bar][slot]]:   # 同じ枠に重なったら強いほうを残す
                    grid[bar][slot] = mark
                cnt[bar] += 1
                if not t.is_drums and (lows[bar] is None or n2.key < lows[bar]):
                    lows[bar] = n2.key
        order = [k for k in ("drums", "bass", "chord", "lead") if k in lanes]
        order += [k for k in lanes if k not in order]
        # ★**音色を出す**（同・棚卸し）。`lead` が歪みギターなのかフルートなのかで、
        #   同じ打点でも聞こえ方がまるで違う。レーン名だけでは分からなかった。
        from accomp_edit import _GM_NAMES
        inst = []
        for k in order:
            if k == "drums":
                inst.append("drums=Drums")
                continue
            p = progs.get(k, 0)                            # MIDI の値（0始まり）。見せる `prog` は1始まり（2026-10-02・案①）
            inst.append(f"{k}={_GM_NAMES.get(p, f'GM{p + 1}')}(prog{p + 1})")
        out.append("")
        out.append("伴奏の音色: " + " / ".join(inst))
        out.append("小節|コード|最低音|" + "/".join(f"{k}(打点)" for k in order)
                   + f"  ※1小節を16分の枠{SLOTS}個で・打点の記号は伴奏の書き方と同じ `X`=強く `x`=ふつう `.`=弱く"
                   "（`O` `o` も `X` `x` と同じ強さなので、ここでは `X` `x` で出す）・`-`=打点なし（前の音が伸びているか、休み）。"
                   "ドラムは打楽器ごとに機械が足す強さの差（キック +8・ハイハット −14 など）を戻して、書いた記号で出す")
        for bar in range(b0 - 1, b1):
            body = "/".join(f"{k[:2]}:{''.join(lanes[k][bar])}({counts[k][bar]})" for k in order)
            out.append(f"{bar + 1}|{cname(bar * BAR)}|"
                       f"{pname(lows[bar], bar * BAR) if lows[bar] is not None else '-'}|{body}")
    else:
        out.append("\n※伴奏MIDIが無いので、コードは楽譜のもの・編成の推移は出せない。")

    out.append(f"\n**良し悪しは何も判断していない**（{shown}音を出した）。"
               "おかしいと思った場所を**小節:拍で名指して**から、直す道具を選ぶこと。"
               "★どの行を見ても説明が付かないなら、**犯人をでっち上げず「材料では説明できない」と言う**。")
    return "\n".join(out), b0, b1, shown
