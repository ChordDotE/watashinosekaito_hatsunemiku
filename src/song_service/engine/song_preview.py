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

"""**途中の段階を、人が聴ける音にする**（2026-09-23 新設）。

## なぜ要るか

本人:「**全部作り切らずに、人間にあえて途中で確認してもらうっていうのもパターンとしてありたい。
ここまで作ってみたんだけど、こんな感じで合ってるかな、っていう提示の仕方もありだと思う**」

ところが**いまは書き出しが `render_final`（`compose`）の一括だけ**で、途中で音になる所が無い。
仮歌（サイン波）は作られるが、**用途は音量見積もり・口の動き・タイミング評価の3つだけ**
——`song_pipeline` のコメントにあるとおり「**どれも人の耳には入らない**」。
**聴ける中間成果物が1つも無い**ので、止めても人が判断できない。

## どこまで出来ているかで、鳴らせるものが変わる

`SongScore` が組み上がるのは**かなり後ろ**（整列・調声・編曲のあと）。それより前の段階には
楽譜がまだ無いので、**そのとき手元にあるものから鳴らす**:

| 段階 | 鳴らせるもの | 材料 |
|---|---|---|
| コード・グルーヴまで | **コードの伴奏**（和音を置くだけ・サイン波）＋**グルーヴのドラム**（SoundFont）。**曲頭の空き小節なし** | 設計・コード・グルーヴ → **本番の組み立て `song_place.place_chords`**・ドラムは **`song_arrange.to_spec` → `_render_accomp`** |
| 旋律まで | 旋律＋コードの伴奏（サイン波）＋**グルーヴのドラム** | 設計・コード・`BareMelody`（歌詞は無くてよい）・グルーヴ → **本番の組み立て `song_place.place_vocal`** |
| **編成・伴奏まで** | **本番と同じ音**（SoundFont の実楽器）＋仮歌 | `SongScore` → **`song_pipeline._render_accomp`** |

★**曲の形で変わる所**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §7-2）: **アカペラ**は伴奏が無いので、旋律の段は旋律だけ
（和音・ドラムを重ねない）、楽譜ができたら歌（主旋律と声部の仮歌）だけ（`preview_voices`）。コード・グルーヴの段はドラムを鳴らさない。
**インスト**は旋律の段を詞なしで置き（`song_place.place_lead`）、楽譜ができたら主旋律の楽器を含む伴奏（本番の `_render_accomp`）で鳴る。
★旋律の段も**置き方は本番の組み立てを通す**（2026-09-30）。前奏・間奏・曲頭の空き小節・拍子の小節の長さ・区間ごとの移調は
組み立てが決めるもので、ここで写し直すと「聴いた位置と高さ」と「出来上がる位置と高さ」が食い違う
（以前は旋律を曲頭から休みなく C 基準で並べ、コードは1音も鳴っていなかった）。コードは旋律より前の工程なので、旋律の段には必ずある。
★**旋律より前の段（コード・グルーヴ）も音にする**（2026-10-01・判-E 本人「a」）。以前は「曲頭の空き小節が旋律で決まる」ので
音にしなかった。いまは空き小節なしの時間軸に、本番と同じ組み立て（区間の調・拍子・1小節に複数のコードの分け方＝`place_chords`）で
置き、**本番とは頭の空きが違うことを返事に添える**（`lead_in_note`）。旋律の費用をかける前に、進行とノリの直しを言える。
★**グルーヴのドラムを鳴らす**（2026-10-01・判-E 本人「a」「B」・コード・グルーヴの段と旋律の段）。グルーヴのドラムは伴奏と同じ
ドラムタブで、グルーヴの工程が同じ記法・打楽器名で検査する（`song_design.GrooveStep`）ので、伴奏の組み立て（`to_spec`）でそのまま読める。
本番でグルーヴのドラム自体は鳴らない（鳴るのは伴奏が写した drums）——試聴では「旋律が乗るノリ」を聴くために鳴らす。
**グルーヴのベースは鳴らさない**（ベースの高さは伴奏の工程が決める）。FluidSynth・SoundFont が無い環境では和音だけにして理由を返す。

★**サイン波でよいのは、楽器がまだ決まっていない段だけ。**
本人:「**伴奏の完成時点で見せてとかだったら、きちんと伴奏の楽器とかで鳴らさないとおかしいよね**」
——楽器を選んだ後までサイン波で聴かせたら判断の材料にならない
（ピアノを指定したのに鉄琴のような音で鳴っていた、という失敗が過去にある）。
だから編成・伴奏の段は**本番の `_render_accomp` をそのまま呼ぶ**。
**別の鳴らし方をここに書かない**——書くと「聴いて OK を出した音」と「出来上がる音」が食い違う。

★**旋律の段は凝らない。** 聴きたいのは「方向が合っているか」だけなので、
伴奏は和音を置くだけでよい（ドラムはグルーヴで決めたものを重ねる・2026-10-01 判-E。それ以外の楽器は足さない）。

## 音量

仮歌の振幅（`PLACEHOLDER_AMP`）は**実歌声に合わせて小さく**してあり、単体で聴くには小さい。
ここは**人が聴くためのもの**なので、書き出す直前に**ピークで正規化**する。
"""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

SR = 44100
Q = 480                                    # 4分音符あたりの tick（song_schema と同じ）


#: 旋律の段の仮の詞（1音に1つ）。組み立て（`song_place.place_vocal`）は詞の印を1音に1つ要るので、まだ詞の無い段でも
#: 同じ組み立てを通せるように置く。サイン波で鳴らすだけなので、何の字でも音は変わらない（歌詞は人に見せない）
_HUM = "ら"


def _as(value, model):
    """工程の出力を型で受け取る（曲づくりの途中の曲は型のまま・記録から読んだ曲は dict）。無ければ None。"""
    if value is None or isinstance(value, model):
        return value
    return model.model_validate(value)


def placed(song) -> tuple[list, list, float]:
    """**旋律とコードを、本番の組み立てで時間軸の上に置く**（2026-09-30）。戻り値は (歌の音符 `SongNote`, コード `ChordSpan`, BPM)。

    ★**組み立ては本番の `song_place.place_vocal` をそのまま通す**——時間軸（前奏・間奏・曲頭の空き小節）・拍子の小節の長さ・
      区間ごとの移調（転調を含む）・1小節に書いた複数のコードの分け方を、ここで写し直さない（2本あると片方だけ直る）。
      以前は旋律を曲頭から休みなく並べ（前奏・間奏が無い・C 基準のまま）、コードはコードの工程の出力（`Chords`）を
      小節の並びとして読めずに**1音も鳴らしていなかった**（1小節の長さも 4/4 の決め打ち）。
    ★詞はまだ無い段なので、1音に1つの仮の詞（`_HUM`）で通す。組み立てが曲の記録に書くこと（曲頭の空き小節など）は
      **本物の曲に書かない**よう、設計と旋律だけを持つ使い捨ての曲へ書かせる。
    検査を通った設計・コード・旋律しか来ない前提（組み立ての決まり）。来なければ組み立ての例外（`AssemblyError`・`ValueError`）がそのまま出る。
    """
    st, design = _placed_state(song)
    return _line(st), st.spans, float(design.bpm)


def _form(song) -> str:
    """曲の形（記録から読んだ dict・欄の無い古い曲は song）。"""
    import song_files
    f = song.get("form") if isinstance(song, dict) else getattr(song, "form", None)
    return song_files.as_form(f)


def _line(st) -> list:
    """旋律の段で鳴らす主旋律の音（歌の音符か、インストなら詞なしで置いた主旋律の楽器の音）。どちらも start_tick・duration_tick・key を持つ。"""
    return st.notes or [ln for _sec, ln in st.lead_notes]


def _placed_state(song):
    """`placed` の組み立てをして、組み立ての状態（時間軸・区間の小節・拍子も）と設計を返す（ドラムを同じ時間軸に置くため）。
    ★組み立ては本番と同じ順（`build_timeline` → `place_vocal`、インストは `place_lead`＝詞なし・2026-10-03）。"""
    from song_design import Chords, Design
    from song_flow import ComposeState
    from song_melody import BareMelody
    from song_model import Song
    from song_place import build_timeline, place_lead, place_vocal   # 組み立ては1か所（2本目を作らない）

    design, chords, melody = (_as(getattr(song, "design", None), Design), _as(getattr(song, "chords", None), Chords),
                              _as(getattr(song, "melody", None), BareMelody))
    if design is None or chords is None or melody is None:
        raise ValueError("設計・コード・旋律のどれかがまだ無い")
    scratch = Song(instruction="")                          # 組み立てが書く記録の受け皿（本物の曲には書かない）
    scratch.design, scratch.melody = design, melody
    st = ComposeState(song=scratch, verbose=False,
                      ch_map={c.kind: c.chords_per_bar for c in chords.sections},   # コードの工程と同じ形（`song_outline.run_chords`）
                      # タイでまとめた音は続きの音の印 `_` も書く（詞と同じ数え方・2026-10-01 判-H）
                      by_index={i: " ".join(_HUM + " _" * int(getattr(n, "ties", 0) or 0) for n in p.notes)
                                for i, p in enumerate(melody.phrases, 1)})
    build_timeline(st)
    if _form(song) == "instrumental":
        place_lead(st)                                      # 詞なしで置く（仮の詞も要らない）
    else:
        place_vocal(st)
    return st, design


def placed_chords(song) -> tuple[list, object, object]:
    """**旋律より前の段（コード・グルーヴ）**: コードを本番の組み立てで時間軸に置く（2026-10-01・判-E 本人「a」）。
    戻り値は (コード `ChordSpan`, 時間軸 `SectionTimeline`, 設計)。

    ★**置き方は本番と同じ**——区間の調・移調量（`song_place.section_keys`）、時間軸（`SectionTimeline.build`・設計の小節数）、
      1小節に書いた複数のコードの分け方（`song_place.place_chords`＝`place_vocal` が使うのと同じ関数）。試聴用の組み立てを別に作らない。
    ★**曲頭の空き小節は置かない**（`lead_bars=0`）。空き小節は旋律の歌い出しで決まる（`place_vocal`）ので、旋律がまだ無い段では
      決められない。本番では頭に1小節空くことがある——呼び手（`show_progress`）がそう添える（`lead_in_note`）。
    """
    from song_design import Chords, Design, bar_ticks
    from song_place import place_chords, section_keys
    from song_timeline import SectionTimeline

    design, chords = _as(getattr(song, "design", None), Design), _as(getattr(song, "chords", None), Chords)
    if design is None or chords is None:
        raise ValueError("設計かコードがまだ無い")
    key_of, shift_of = section_keys(design)
    bar = bar_ticks(design.time_signature)
    timeline = SectionTimeline.build(design.sections, {}, bar, lead_bars=0)
    spans = place_chords(timeline, {c.kind: c.chords_per_bar for c in chords.sections}, shift_of, key_of, bar)
    return spans, timeline, design


def lead_in_note(design) -> str:
    """旋律より前の段の試聴が本番と違う所（曲頭の空き小節）の断り書き。最初の区間が歌なしなら本番も空かないので空。"""
    secs = list(getattr(design, "sections", None) or [])
    if secs and secs[0].has_vocal:
        return ("旋律がまだ無いので、曲頭の空き小節は入れていない"
                "（本番では、歌い出しが曲頭に近いと頭に空の1小節が入り、そのぶん全体が後ろへずれる）")
    return ""


def _groove(song):
    """グルーヴの工程の出力（型のまま・記録から読んだ dict なら型へ）。区間を持たないもの・無ければ None。"""
    from song_design import Groove
    g = getattr(song, "groove", None)
    if isinstance(g, dict):
        return Groove.model_validate(g)
    return g if getattr(g, "sections", None) else None


def groove_spec(song, design, bars_of: dict):
    """**グルーヴのドラムを、伴奏と同じ組み立て（`song_arrange.to_spec`）で楽譜のパートにする**（2026-10-01・判-E）。無ければ None。

    グルーヴのドラムの行は伴奏のドラムと同じドラムタブで、2026-10-01 からグルーヴの工程が同じ記法・打楽器名で検査する
    （`song_design.GrooveStep`）ので、伴奏の読み手（`_grid_drums`・打楽器名は GM の正式な名前・開いたハイハットは open_hi_hat の行）でそのまま読める。
    区間の強さの音量差も伴奏と同じ（`intensity_of`）。キットはまだ決まっていない（編成の前）ので音源の既定。
    **グルーヴのベースは鳴らさない**（ベースの音は伴奏の工程が決める・ここで高さを決めると試聴だけの組み立てになる）。
    """
    from song_arrange import ArrangePlan, DrumLine, DrumPlan, DrumSectionPattern, to_spec
    from song_design import grid_slots
    g = _groove(song)
    if g is None:
        return None
    # 機械が組む試聴の伴奏なので、伴奏の型の必須の欄は値を明示する（2026-10-04 組N2。以前の型の既定＝全小節・音量 0.55・リフ無し）
    pats = [DrumSectionPattern(sections=[x.kind], lines=[DrumLine(inst=d.inst, pattern=d.pattern) for d in x.drums], at_bars="all")
            for x in g.sections if x.drums]
    if not pats:
        return None
    plan = ArrangePlan(reason="グルーヴの土台（途中の試聴）", parts=[], riff=[], lead_gain=None,
                       drums=DrumPlan(patterns=pats, gain=0.55))
    return to_spec(plan, bars_of, grid_slots(design.time_signature), [], True,
                   intensity_of={s.kind: s.intensity for s in design.sections})


def render_drums(song, design, spans, bars_of: dict, path: Path) -> tuple[list[float], str]:
    """グルーヴのドラムを**本番の書き出し（`song_pipeline._render_accomp`＝SoundFont）**で鳴らす。戻り値は (波形, 鳴らさなかった理由)。

    グルーヴが無い・ドラムの行が無いなら ([], "")。組み立てられない（検査の前の古い記録で読めない行がある）・FluidSynth や
    SoundFont が無いときは ([], 理由)——和音だけで鳴らし、理由を返事に添える（代わりの音で鳴らさない）。
    """
    from song_fix import AssemblyError
    from song_pipeline import _render_accomp
    from song_schema import SongScore
    try:
        spec = groove_spec(song, design, bars_of)
    except (AssemblyError, ValueError) as e:
        return [], f"グルーヴのドラムを組み立てられなかった（{e}）"
    if spec is None:
        return [], ""
    score = SongScore(title="preview", tempo=float(design.bpm), time_signature=design.time_signature,
                      vocal_notes=[], chords=list(spans), arrangement=spec)
    acc = path.with_name(path.stem + "_drums.wav")
    try:
        _render_accomp(score, acc)
        x, rate = _read_wav(acc)
    except Exception as e:                                  # noqa: BLE001（音源が無い等。和音だけで鳴らして理由を返す）
        return [], f"グルーヴのドラムを鳴らせなかった（{type(e).__name__}: {e}）"
    finally:
        acc.unlink(missing_ok=True)
        acc.with_suffix(".mid").unlink(missing_ok=True)
    if rate != SR:
        return [], f"グルーヴのドラムの音の形式が違う（{rate}Hz）"
    return x, ""


def _read_wav(p: Path) -> tuple[list[float], int]:
    """wav → (1ch の波形 -1〜1, サンプリング周波数)。2ch はモノラルへ畳む（聴ければよい）。16bit 以外は空。"""
    with wave.open(str(p), "rb") as w:
        n, ch, sw = w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
        fr = w.getframerate()
    if sw != 2:
        return [], fr
    vals = struct.unpack("<%dh" % (len(raw) // 2), raw)
    if ch > 1:
        vals = [sum(vals[i:i + ch]) / ch for i in range(0, len(vals) - ch + 1, ch)]
    return [v / 32768.0 for v in vals], fr


def melody_notes(notes, bpm: float) -> list[tuple[float, float, int]]:
    """組み立てた歌の音符（`SongNote`）→ (開始秒, 長さ秒, midi)。位置と高さは組み立てのまま（秒へ直すだけ）。"""
    spb = 60.0 / max(bpm, 1.0) / Q                          # 1 tick の秒数
    return [(n.start_tick * spb, n.duration_tick * spb, int(n.key)) for n in notes if n.duration_tick > 0]


def chord_notes(spans, bpm: float) -> list[tuple[float, float, int]]:
    """組み立てたコード（`ChordSpan`）を「和音を置くだけ」の伴奏にする。**方向が合っているかを聴くため**の最小限。

    和音の中身は本番と同じ解釈（`song_audio.chord_notes`）で、1オクターブ下に敷く。位置と長さはコードの置き場のまま
    （拍子の小節の長さ・1小節に書いた複数のコードの分け方も組み立てが決めたもの）。
    """
    from song_audio import chord_notes as tones     # 本番と同じ和音の解釈を使う（ずれを作らない）

    spb = 60.0 / max(bpm, 1.0) / Q
    return [(c.start_tick * spb, c.duration_tick * spb, int(k) - 12)
            for c in spans for k in tones(c.chord)]


def write_wav(parts: list[tuple[list[tuple[float, float, int]], float]], path: Path,
              *, tail: float = 0.4, samples: list[tuple[list[float], float]] | None = None) -> Path:
    """(音の列, 相対音量) をいくつか重ねて wav にする。**最後にピークで正規化**する。

    `samples`（(波形, 相対音量)・`SR` の1ch）は、本番の書き出しで鳴らした音（グルーヴのドラム・`render_drums`）を重ねるため。
    """
    end = max((s + d for notes, _ in parts for s, d, _ in notes), default=0.0) + tail
    buf = [0.0] * max(int(end * SR) + 1, max((len(x) for x, _g in samples or []), default=0))
    for x, g in samples or []:
        for i, v in enumerate(x):
            buf[i] += v * g
    for notes, gain in parts:
        for s, d, midi in notes:
            freq = 440.0 * 2 ** ((midi - 69) / 12)
            i0, ln = int(s * SR), int(d * SR)
            for i in range(ln):
                j = i0 + i
                if j >= len(buf):
                    break
                t = i / SR
                env = min(1.0, t / 0.02) * min(1.0, (ln - i) / SR / 0.05)
                buf[j] += gain * env * math.sin(2 * math.pi * freq * t)
    peak = max((abs(v) for v in buf), default=0.0)
    scale = (0.89 / peak) if peak > 0 else 1.0              # 人が聴くので、聴こえる大きさへ
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(struct.pack("<h", int(max(-1.0, min(1.0, v * scale)) * 32767))
                               for v in buf))
    return path


def preview_arrangement(score, path: Path, *, with_vocal: bool = True) -> dict:
    """**編成・伴奏の段では、本番と同じ音で鳴らす。**

    ★本人の指摘（2026-09-23）:「**伴奏の完成時点で見せてとかだったら、
    きちんと伴奏の楽器とかで鳴らさないとおかしいよね**」——そのとおり。
    旋律の段でサイン波なのは**まだ楽器が決まっていないから**であって、
    **楽器が決まった後までサイン波で聴かせたら、判断の材料にならない**
    （ピアノを選んだのに鉄琴のような音で鳴っていた、という失敗が過去にある）。

    だから**本番の `_render_accomp` をそのまま呼ぶ**（SoundFont の実音。鳴らせなければ理由を返す——
    2026-09-29 から合成音へは落とさない）。
    ここで別の鳴らし方を書くと、**聴いて OK を出した音と、出来上がる音が違う**ことになる。

    `with_vocal` のときは仮歌（サイン波）を薄く重ねる——**歌の乗り方**を確かめるため。
    実歌声はまだ無いので、ここはサイン波でよい。
    """
    import wave as _wave

    from song_pipeline import _render_accomp, render_placeholder_vocal

    path.parent.mkdir(parents=True, exist_ok=True)
    acc = path.with_name(path.stem + "_acc.wav")
    try:
        _render_accomp(score, acc)                          # ★本番と同じ経路（音を変えない）
    except Exception as e:
        return {"ok": False, "reason": f"伴奏を鳴らせなかった（{type(e).__name__}: {e}）"}

    tracks: list[tuple[list[float], float]] = []
    rate = SR
    _read = _read_wav                                       # 読み方は1つ（グルーヴのドラムと同じ）

    a, rate = _read(acc)
    if a:
        tracks.append((a, 1.0))
    if with_vocal and getattr(score, "vocal_notes", None):
        voc = path.with_name(path.stem + "_voc.wav")
        try:
            render_placeholder_vocal(score, voc)
            v, vr = _read(voc)
            if v and vr == rate:
                tracks.append((v, 1.0))
            voc.unlink(missing_ok=True)
        except Exception:
            pass
    if not tracks:
        return {"ok": False, "reason": "鳴らせる音が無かった"}

    ln = max(len(t) for t, _ in tracks)
    buf = [0.0] * ln
    for t, g in tracks:
        for i, v in enumerate(t):
            buf[i] += v * g
    peak = max((abs(v) for v in buf), default=0.0)
    scale = (0.89 / peak) if peak > 0 else 1.0
    with _wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(max(-1.0, min(1.0, v * scale)) * 32767))
                               for v in buf))
    acc.unlink(missing_ok=True)
    return {"ok": True, "path": str(path), "seconds": round(ln / rate, 1),
            "with_vocal": len(tracks) > 1}


def preview_voices(score, path: Path) -> dict:
    """**アカペラの楽譜を聴ける音にする**（2026-10-03・§7-2）。伴奏が無いので、仮歌（主旋律＋声部のサイン波・
    `song_pipeline.render_placeholder_vocal`＝本番の仮歌と同じ1か所）だけを、人が聴ける大きさへ正規化して書く。"""
    from song_pipeline import render_placeholder_vocal
    if not getattr(score, "vocal_notes", None):
        return {"ok": False, "reason": "歌の音符が無い"}
    path.parent.mkdir(parents=True, exist_ok=True)
    voc = path.with_name(path.stem + "_voc.wav")
    try:
        render_placeholder_vocal(score, voc)
        v, rate = _read_wav(voc)
    except Exception as e:                                  # noqa: BLE001（鳴らせなければ理由を返す）
        return {"ok": False, "reason": f"歌を鳴らせなかった（{type(e).__name__}: {e}）"}
    finally:
        voc.unlink(missing_ok=True)
    if not v:
        return {"ok": False, "reason": "鳴らせる音が無かった"}
    peak = max((abs(x) for x in v), default=0.0)
    scale = (0.89 / peak) if peak > 0 else 1.0
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(max(-1.0, min(1.0, x * scale)) * 32767)) for x in v))
    return {"ok": True, "path": str(path), "seconds": round(len(v) / rate, 1), "with_vocal": True,
            "voices": len(getattr(score, "extra_voices", None) or []),
            "note": "アカペラなので歌（主旋律と声部・サイン波）だけを鳴らした"}


def preview_melody(song, path: Path) -> dict:
    """**旋律とコードを聴ける音にする**（楽器が決まる前の段）。戻り値は何を鳴らしたかの要約。

    ★楽譜（`SongScore`）はまだ無い段階で呼ばれる。設計・コード・旋律から、本番の組み立て（`placed`）で置いてから鳴らす
      （2026-09-30。以前は `BareMelody` とコードの工程の出力を直に読み、コードは1音も鳴っていなかった）。
    組み立てられなければ（設計・コード・旋律が揃っていない・読めない階名など）、黙って音を抜かずに理由を返す。
    ★**グルーヴのドラムも重ねる**（2026-10-01・判-E 本人「B」）。旋律はグルーヴの上で書かれているので、ノリと一緒に聴く。
      ドラムは本番の組み立てと書き出しで鳴らし（`render_drums`）、鳴らせなければ和音と旋律だけにして理由を返す（`drums_reason`）。
    """
    from song_fix import AssemblyError
    try:
        st, design = _placed_state(song)
    except (AssemblyError, ValueError) as e:                # 読めない階名・揃っていない工程（組み立てでも止まる）
        return {"ok": False, "reason": f"旋律とコードを組み立てられなかった（{e}）"}
    notes, spans, bpm = _line(st), st.spans, float(design.bpm)
    mel = melody_notes(notes, bpm)
    if not mel:
        return {"ok": False, "reason": "旋律から鳴らせる音が取れなかった"}
    path.parent.mkdir(parents=True, exist_ok=True)
    if _form(song) == "acappella":
        # ★**アカペラは歌だけ**（2026-10-03・§7-2）。伴奏が無い曲なので、和音もグルーヴのドラムも重ねない（重ねると、出来上がる音と違う）
        write_wav([(mel, 1.0)], path)
        return {"ok": True, "path": str(path), "notes": len(mel), "chords": 0,
                "seconds": round(max(s + d for s, d, _ in mel), 1), "with_chords": False, "with_drums": False,
                "drums_reason": "", "note": "アカペラなので旋律だけを鳴らした（和音・ドラムは重ねない）"}
    acc = chord_notes(spans, bpm)
    drums, why = render_drums(song, design, spans, st.bars_of, path)
    write_wav([(mel, 1.0)] + ([(acc, 0.22)] if acc else []), path, samples=[(drums, 1.0)] if drums else None)
    return {"ok": True, "path": str(path), "notes": len(mel), "chords": len(spans),
            "seconds": round(max(max(s + d for s, d, _ in mel + acc), len(drums) / SR), 1),
            "with_chords": bool(acc), "with_drums": bool(drums), "drums_reason": why}


def preview_chords(song, path: Path) -> dict:
    """**旋律より前の段（コード・グルーヴ）を聴ける音にする**（2026-10-01・判-E 本人「a」）。戻り値は何を鳴らしたかの要約。

    和音はサイン波（旋律の段と同じ・楽器はまだ決まっていない）、グルーヴがあればそのドラムを本番の書き出しで重ねる。
    置き方は本番と同じ組み立て（`placed_chords`）。**曲頭の空き小節は入れない**（旋律で決まるため）——`note` に断り書きを入れる。
    """
    from song_fix import AssemblyError
    try:
        spans, timeline, design = placed_chords(song)
    except (AssemblyError, ValueError) as e:
        return {"ok": False, "reason": f"コードを組み立てられなかった（{e}）"}
    bpm = float(design.bpm)
    acc = chord_notes(spans, bpm)
    if not acc:
        return {"ok": False, "reason": "コードから鳴らせる音が取れなかった"}
    path.parent.mkdir(parents=True, exist_ok=True)
    if _form(song) == "acappella":                          # アカペラはドラムを鳴らさない（グルーヴはノリの参考・§3）
        drums, why = [], "アカペラなのでグルーヴのドラムは鳴らさない（ノリの参考）"
    else:
        drums, why = render_drums(song, design, spans, timeline.bars_of(), path)
    write_wav([(acc, 0.5)], path, samples=[(drums, 1.0)] if drums else None)
    note = "／".join(x for x in (lead_in_note(design), why) if x)
    return {"ok": True, "path": str(path), "chords": len(spans),
            "seconds": round(max(max(s + d for s, d, _ in acc), len(drums) / SR), 1),
            "with_chords": True, "with_drums": bool(drums), "drums_reason": why, "note": note}


# ---------------------------------------------------------------------------
# **ここまで作った、を人へ出す**（2026-09-23）
#
# 本人:「**途中で止まる場合には、どこまでやったかをメインに返す必要がある。
#   まずコンセプトだったらそこのコンセプトのテキスト。それ以外、音楽作り始めたら wav。
#   完成までいったら vpr 書き出してそこから wav。
#   「ここまでこういう状態でやったので、このファイルを見てください」みたいなのを
#   メインに教える必要がありそう**」
#
# ★**段階によって「見せるもの」が変わる**。テキストで足りる所で音を作っても待たせるだけだし、
#   音でしか分からない所をテキストで出しても判断できない。
# ---------------------------------------------------------------------------

def reached(song) -> str:
    """どこまで作ったか（いちばん後ろまで進んだ工程のキー）。何も無ければ空。"""
    order = ["theme", "design", "chords", "groove", "melody",
             "lyric_plan", "lyric_chorus", "lyrics", "naming", "instrumentation", "arrangement", "harmony"]
    from song_model import made                           # 走らなかった工程（空の出力・F1）は「作った」に数えない
    last = ""
    for key in order:
        if made(getattr(song, key, None)):
            last = key
    return last


def show_progress(song, song_dir: Path) -> dict:
    """**ここまでの状態を、人が受け取れる形にする。**

    戻り値: `stage`（どこまで）／`kind`（"text" か "audio"）／`text` か `path`／`say`（ミクへの一言）／
    `note`（音の断り書き——本番と違う所・鳴らせなかったもの。無ければ空）。
    ★**判断はここでしない**——何をどう言うかはミクが決める。ここは材料を作るだけ。
    ★コード・グルーヴの段も音にする（2026-10-01・判-E 本人「a」。以前は旋律の段からで、ここはテキストだった）。
    """
    stage = reached(song)
    if not stage:
        return {"stage": "", "kind": "none", "say": "まだ何も作れていない"}

    label = {"theme": "テーマ", "design": "設計", "chords": "コード", "groove": "グルーヴ",
             "melody": "旋律", "lyric_plan": "詞の設計", "lyric_chorus": "サビの詞",
             "lyrics": "詞", "naming": "曲名", "instrumentation": "編成", "arrangement": "伴奏",
             "harmony": "ハモリ"}.get(stage, stage)

    # ① コンセプトの段（テーマ・設計）は**テキストで足りる**。音はまだ無いし、作っても待たせるだけ
    if stage in ("theme", "design"):
        parts = []
        for key in ("theme", "design"):
            v = getattr(song, key, None)
            if v is not None:
                r = (getattr(v, "reason", "") or "").strip()
                parts.append(f"【{label if key == stage else ('テーマ' if key == 'theme' else '設計')}】"
                             + (r or str(v))[:600])
        return {"stage": label, "kind": "text", "text": "\n".join(parts),
                "say": f"{label}まで考えたので、方向が合っているか見てほしい"}

    # ② 音楽が始まったら**音で出す**（譜面を見せても人は判断できない）
    out = song_dir / "preview.wav"
    score = getattr(song, "score", None)
    if score is not None and getattr(score, "form", "song") == "acappella":
        res = preview_voices(score, out)                   # アカペラは伴奏が無い——歌（主旋律と声部）だけ（2026-10-03・§7-2）
    elif score is not None and not score.arrangement.is_empty():   # 伴奏が無ければパートが0本（F1）
        res = preview_arrangement(score, out)              # 楽器が決まっていれば本番と同じ音で
    elif stage in ("chords", "groove"):
        res = preview_chords(song, out)                    # 和音（サイン波）＋グルーヴのドラム・曲頭の空き小節なし
    else:
        res = preview_melody(song, out)                    # 旋律とコードを本番の組み立てで置いて（サイン波で）＋グルーヴのドラム
    if not res.get("ok"):
        return {"stage": label, "kind": "text",
                "text": f"{label}まで作ったが、音にできなかった（{res.get('reason')}）",
                "say": f"{label}まで作ったけれど、聴かせる音にできなかった"}
    note = res.get("note") or res.get("drums_reason") or ""
    return {"stage": label, "kind": "audio", "path": res["path"],
            "seconds": res.get("seconds"), "note": note,
            "say": f"{label}まで作ったので、一度聴いてほしい" + (f"（{note}）" if note else "")}
