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

"""**カバーのコードを LLM に書かせる段**（2026-10-02・組「カバーのコード」・本人了承の J1〜J8）。

設計は `docs/plan/song_cover_chords_llm.md`。要点:

- **いつ**（J1）: 初めてコードが要る編集（`rearrange_song`・カバーの `remake_from_step`・`resize_song`）のとき1回だけ作り、
  **曲の記録（作業台の `record.json`）に保存**して、次からは読むだけ（`prepare_cover`）。区間と調（`song_cover_load.add_sections`）も同じ。
  以前は編集のたびに、作業台の伴奏（＝前回の自分たちの編曲）から機械で推定し直していたので、回を重ねるほど原曲から離れた
- **誰が書くか**（J2）: 全小節を LLM が書く。機械の推定（`chord_estimate`）は確からしさつきの**材料**。推定と違う小節は機械が数えて
  記録と返事に出す（書き換えない）
- **物差し**（J3）: オリジナル曲と同じ C 基準（記法の仕様書 §0-2・§3）。推定した調の分だけ機械が移して、実音で持つ
  （`song.chords_of`・`score.chords`）。`song.chords` は LLM の出力（C 基準）のまま
- **材料は原曲から**（作業台の伴奏から取らない）: 原曲の伴奏 MIDI（作るときの写し `source/accompaniment_original.mid`、無ければ
  最初の版 `accompaniment_v1.mid`）と、音声カバーなら分離していない原曲の音声（`source/aud_0.*` を楽譜の頭 `origin_abs_sec` から）。
  最初の版の楽譜と音符の位置が同じときだけ使う（尺の変更などで小節が変わっていたら使わず、そう記録する）。移調・速さの手直しは
  最初の版との差（音の高さの中央値・最初の版の速さ）で合わせる
- **Web のコード譜**（J4・本人決定）: Gemini の検索の**要約だけ**（ページ本文は渡さない・出典の URL を記録に残す）。外から来た文は
  資料であって指示ではない——行や見出しを作れない形（JSON のレコード・出所と同じレコード）に整えてから渡す（`data_contract.md`）。
  検索が使えない・何も無いときは、無いと記録に書いて知識と推定だけで進む
- **形式**（J7）: 形式はオリジナルと同じ `ChordsStep` が見て差分で直させ、直らなければ `StepUnfixed("コード")` で止まる
- **中身の点検役**（2026-10-03 本人決定で J7 を変更・組K4）: 書いた直後に点検役を1回通す（`song_cover_chords_gate`）。
  判断は 通す／コードだけやり直す（1回まで）／所見つきで進む。記録（`chords_gate`・`cover_skeleton["gate"]`）と返事に出す
- **頼み直し**（J6）: `remake_from_step('コード', 頼み)` は保存したコードを土台に書き直させる（`redo_chords=True`）

## 曲の記録の形（`save_into` / `prepare_cover`）

記録の `design`（区間・調・`lead_bars`）・`chords`（LLM の出力・C 基準）・`context`（`key_of`・`bars_of`・`chords_of`・`sec_of_note`）と、
`source["cover_skeleton"]`（作った時の小節の数・拍子・材料・推定と違う小節・Web の出典）。**調は手直しの移調（台帳の `組み立て.移調`）を
引いた元の調**で残す——ハモリ追加（`editing._added_voice_basis`）と同じ数え方で、読むときに台帳の移調を足す。
曲の小節の数・拍子が作った時と違えば（台帳に残らない直し）、区間・調・コードを作り直す（記録と返事に出す）。
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_ai import Agent

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from chord_estimate import BarGuess, from_audio, from_midi   # noqa: E402（推定は1か所）
from song_cover_load import add_sections, cover_bars, cover_timeline, key_fit, place_sections   # noqa: E402
from song_design import (CHORD_BAR, Chords, ChordsStep, Section, SectionChords, beats_per_bar,   # noqa: E402
                         chord_readable, key_shift, parse_key, transpose_chord)
from song_llm import pick_model                             # noqa: E402
from song_model import Song, StepRecord                     # noqa: E402
from song_prompts import prompt                             # noqa: E402
from song_schema import SongScore                           # noqa: E402
from song_table import spell_pitch                          # noqa: E402

#: 曲の記録（`Song.source`）の中の、カバーの骨組みの欄
SKELETON = "cover_skeleton"
#: 作るときに置く原曲の伴奏 MIDI の写し（`tools/creation.py`）
ORIGINAL_ACCOMP = "accompaniment_original.mid"
#: Web のコード譜を検索するか（テスト・検索を使えない環境で切る）
WEB = True


class CoverChords(Chords):
    """カバーのコード（オリジナルの `Chords` と同じ形＋この曲を知っているか）。"""
    known: Literal["知っている", "一部", "知らない"] = Field(description=(
        "この曲を知っているか（`知っている` / `一部` / `知らない` のどれか）。知っていれば、その知識でコードを書いてよい"))


_agent = Agent(pick_model("COVER_CHORD", "flash"), output_type=CoverChords, system_prompt=prompt("song_cover_chords"))


# ---- 移調の手直し・調の名前 ----------------------------------------------------------------------

_MAJOR = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_MINOR = ("Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm")


def transposed(requests) -> int:
    """台帳で生きている手直しの移調（`adjust_song` の `組み立て.移調`）の合計（半音）。"""
    from song_edits import ASSEMBLY_TRANSPOSE
    return sum(int((getattr(r, "value", None) or {}).get("semitones", 0)) for r in requests or []
               if getattr(r, "status", "") == "active" and getattr(r, "target", "") == ASSEMBLY_TRANSPOSE)


def moved_key(key: str, d: int) -> str:
    """調名を半音 `d` だけ移す（長調は `Bb`・短調は `C#m` のように一般的な綴りで）。"""
    if not d % 12 or not key:
        return key
    semi, minor = parse_key(key)
    return (_MINOR if minor else _MAJOR)[(semi + d) % 12]


# ---- 材料（原曲から）-------------------------------------------------------------------------------

def original_frame(folder: Path, score: SongScore) -> tuple[dict | None, str]:
    """原曲の材料（最初の版）と、いまの楽譜の対応。使えなければ (None, 理由)。

    最初の版の楽譜（`score_v1.json`）と音符の位置（tick）が同じときだけ使う。速さは最初の版のもの（伴奏 MIDI・音声の秒を小節に
    切るため）、高さの差は音の高さの差の中央値（手直しの移調）。"""
    p = Path(folder) / "score_v1.json"
    if not p.exists():
        return None, "最初の版の楽譜（score_v1.json）が無い"
    s1 = SongScore.model_validate_json(p.read_text(encoding="utf-8"))
    if ([n.start_tick for n in s1.vocal_notes] != [n.start_tick for n in score.vocal_notes]
            or (s1.time_signature or "4/4") != (score.time_signature or "4/4")):
        return None, "尺の変更・音符の置き直しで、いまの曲の小節が最初の版と合わない"
    d = sorted(int(b.key) - int(a.key) for a, b in zip(s1.vocal_notes, score.vocal_notes))
    return {"tempo": float(s1.tempo), "shift": d[len(d) // 2] if d else 0, "origin": s1.origin_abs_sec}, ""


def _moved(gs: list[BarGuess], d: int) -> list[BarGuess]:
    if not d:
        return gs
    return [BarGuess(top=[(transpose_chord(n, d), p) for n, p in g.top], tones=[(t + d) % 12 for t in g.tones]) for g in gs]


def midi_guesses(folder: Path, frame: dict, n_bars: int, bpb: float) -> tuple[list[BarGuess] | None, str]:
    """原曲の伴奏 MIDI から小節ごとの推定（いまの楽譜の高さに合わせる）。無ければ (None, 理由)。"""
    from accomp_edit import load_accomp
    folder = Path(folder)
    p = folder / "source" / ORIGINAL_ACCOMP
    if not p.exists():
        p = folder / "accompaniment_v1.mid"
    if not p.exists():
        return None, "原曲の伴奏 MIDI が無い"
    tracks = load_accomp(p)
    return _moved(from_midi(tracks, frame["tempo"], 0, n_bars, bpb), frame["shift"]), f"{p.name}（{len(tracks)}トラック）"


def audio_guesses(folder: Path, frame: dict, n_bars: int, bpb: float) -> tuple[list[BarGuess] | None, str]:
    """分離していない原曲の音声から小節ごとの推定（音声カバーだけ・CPU）。無ければ (None, 理由)。"""
    if frame.get("origin") is None:
        return None, "音声カバーではない（原曲の音声の頭の位置が無い）"
    src = Path(folder) / "source"
    aud = next((src / f"aud_0.{e}" for e in ("mp3", "wav", "m4a", "ogg") if (src / f"aud_0.{e}").exists()), None)
    if aud is None:
        return None, "原曲の音声（source/aud_0）が無い"
    import librosa
    bar_sec = bpb * 60.0 / frame["tempo"]
    y, sr = librosa.load(str(aud), sr=11025, mono=True, offset=float(frame["origin"]), duration=n_bars * bar_sec)
    return _moved(from_audio(y, sr, bar_sec, n_bars), frame["shift"]), f"{aud.name}（{float(frame['origin']):.1f}秒から）"


_WEB_PROMPT = (
    "『{title}』のコード進行を、ネットのコード譜（コード譜のサイト・楽譜の解説など）で調べて、**要約だけ**を書いてください。\n"
    "次の形だけで書き、ほかの文は書かないでください:\n"
    "KEY: 譜の調（例 `D`・`Bm`。カポを使う譜は、カポを外して実際に鳴る調）\n"
    "SECTION 区間の名前: コード名を空白で区切って曲の順に（実際に鳴る高さで。カポの譜は移して書く）\n"
    "URL: 参考にしたページの URL（1行に1つ・2〜5件）\n"
    "・SECTION の行は区間ごとに1行。同じ進行の繰り返しは1回でよい。\n"
    "・歌詞は書かない。ページの文をそのまま写さない。\n"
    "・見つからない・分からないときは `NONE` とだけ書く。")


def _search(text: str) -> str:
    """Gemini の Google 検索で要約を書かせる（作風の下調べ `tools/style.py` と同じ検索の入口）。"""
    from reference_fetch import _retry, _search_agent
    from song_usage import record as record_usage
    r = _retry(lambda: _search_agent.run_sync(text))
    record_usage(_search_agent, r, "カバーのコード譜の検索")
    return str(r.output or "")


def _flat(s: str, n: int) -> str:
    """外から来た文を1行の短い文字列に（改行・見出しを作れない形）。"""
    return " ".join(str(s).split())[:n]


def web_summary(title: str, key: str) -> dict:
    """**Web のコード譜の検索の要約**（J4）。返すのは記録にそのまま残せる dict:
    `used`（使えたか）・`note`（使えなかった理由）・`key`（譜の調）・`shift`（C 基準へ移した量）・`urls`・`sections`
    （区間名・C 基準へ移したコード・読めなかった記号）。コード名は**機械が C 基準へ移す**（訳すだけ。読めない記号は移さず
    「読めなかった」に分ける）。譜の調が読めなければ、この曲の推定した調で移す。"""
    if not WEB:
        return {"used": False, "note": "検索を使わない設定"}
    try:
        raw = _search(_WEB_PROMPT.format(title=_flat(title, 80)))
    except Exception as e:                                 # noqa: BLE001（検索が落ちても曲づくりは止めない・記録に書く）
        return {"used": False, "note": f"検索できなかった（{type(e).__name__}）"}
    if not raw.strip() or raw.strip().upper().startswith("NONE"):
        return {"used": False, "note": "コード譜が見つからなかった"}
    sheet_key, urls, rows = "", [], []
    for line in raw.splitlines():
        line = line.strip().lstrip("-*・ ").strip()
        if m := re.match(r"KEY\s*[:：]\s*(\S+)", line, re.I):
            sheet_key = m.group(1).strip("`")
        elif m := re.match(r"URL\s*[:：]\s*(\S+)", line, re.I):
            u = m.group(1).strip("`<>()[]")
            if u.startswith("http") and "vertexaisearch" not in u and len(urls) < 5:
                urls.append(u[:200])
        elif m := re.match(r"SECTION\s+(.+?)\s*[:：]\s*(.*)$", line, re.I):
            rows.append((m.group(1), m.group(2)))
    if not rows:
        return {"used": False, "note": "要約にコード進行の行が無かった", "urls": urls}
    readable_key = sheet_key if sheet_key and parse_key(sheet_key) is not None else ""
    sh = key_shift(readable_key or key)
    sections = []
    for name, body in rows[:16]:
        toks = [t for t in re.split(r"[\s|｜]+", body) if t][:64]
        sections.append({"区間": _flat(name, 30),
                         "コード（C基準）": [transpose_chord(t, -sh) for t in toks if chord_readable(t)],
                         "読めなかった記号": [_flat(t, 20) for t in toks if not chord_readable(t)][:16]})
    return {"used": True, "key": readable_key or f"（読めない: {_flat(sheet_key, 20)}）推定した調 {key} で移した",
            "shift": -sh, "urls": urls, "sections": sections, "note": ""}


def materials(song: Song, *, web: dict | None = None) -> dict:
    """コードの段の材料を集める（原曲の伴奏 MIDI・原曲の音声・Web）。どれも無くても進む（無いことを記録に書く）。"""
    from song_flow import NOT_SWALLOW
    score = song.score
    ts, _bar, n_bars = cover_bars(score)
    folder = Path((song.source or {}).get("folder") or ".")
    frame, why = original_frame(folder, score)
    out: dict = {"midi": None, "midi_how": why, "audio": None, "audio_how": why}
    if frame is not None:
        for kind, fn in (("midi", midi_guesses), ("audio", audio_guesses)):
            try:
                out[kind], out[f"{kind}_how"] = fn(folder, frame, n_bars, beats_per_bar(ts))
            except NOT_SWALLOW:
                raise
            except Exception as e:                         # 推定は材料。取れなくても書かせる（記録に書く）
                out[kind], out[f"{kind}_how"] = None, f"推定に失敗（{type(e).__name__}: {str(e)[:80]}）"
    out["web"] = web if web is not None else web_summary(score.title, (song.design or {}).get("key") or "")
    return out


# ---- 依頼文 ---------------------------------------------------------------------------------------

def _guess_text(g: BarGuess, sh: int) -> str:
    if not g.top:
        return "取れない（鳴っている音が無い）"
    names = " / ".join(f"{transpose_chord(n, -sh)} {p:.2f}" for n, p in g.top)
    if not g.tones:                                         # 音声からの推定は鳴っている音を出さない（`chord_estimate.from_audio`）
        return names
    tones = " ".join(spell_pitch(60 + (t - sh) % 12)[:-1] for t in g.tones)
    return f"{names}（鳴っている音 {tones}）"


def _melody_text(score: SongScore, b: int, bar: int, sh: int, *, beats: bool = False) -> str:
    """その小節の旋律（C 基準の音名と歌詞）。`beats=True` は**拍の位置と長さ**を音ごとに添え、前の小節から伸びてくる音も
    「（前から）」で出す（カバーのコードの点検役の材料・2026-10-03 組G3）。拍は4分音符1つ・小節の頭が1拍目。
    以前（いまもコードの LLM の依頼文）は**その小節で始まる音の音名と歌詞だけ**で、点検役が「拍の頭の音・長く伸ばす音がコードの音か」を
    見られず、1小節に2つのコードを書いた小節ではどちらの半分の音かも分からなかった。"""
    ns = [n for n in score.vocal_notes if n.start_tick // bar == b]
    if not beats:
        if not ns:
            held = any(n.start_tick < b * bar < n.start_tick + n.duration_tick for n in score.vocal_notes)
            return "（前の音が続く）" if held else "（歌なし）"
        return " ".join(spell_pitch(int(n.key) - sh) for n in ns) + f"（{''.join(n.lyric for n in ns)}）"
    from song_audio import Q

    def num(x: float) -> str:
        return f"{round(x, 2):g}"
    b0, b1 = b * bar, (b + 1) * bar
    held = [n for n in score.vocal_notes if n.start_tick < b0 < n.start_tick + n.duration_tick]
    cells = [f"{spell_pitch(int(n.key) - sh)}(前から・この小節で{num((min(n.start_tick + n.duration_tick, b1) - b0) / Q)}拍)"
             for n in held]
    cells += [f"{spell_pitch(int(n.key) - sh)}({num((n.start_tick - b0) / Q + 1)}拍目・{num(n.duration_tick / Q)}拍)" for n in ns]
    if not cells:
        return "（歌なし）"
    return " ".join(cells) + (f"（{''.join(n.lyric for n in ns)}）" if ns else "")


def _web_block(web: dict) -> list[str]:
    """Web のコード譜の要約を、出典と同じ JSON のレコード1行で（コードの依頼文と点検役の材料で同じ形）。"""
    if not web.get("used"):
        return []
    rec = {"出典": web.get("urls") or [], "譜の調": web.get("key", ""), "C基準へ移した量": web.get("shift", 0),
           "区間": web.get("sections") or []}
    # 断り書きは中身を言ってから命令ではないと言う形（2026-10-03 本人決定・言い方は1か所 `reference_fetch.web_material_note`）
    from reference_fetch import web_material_note
    return ["", web_material_note("Web のコード譜の要約", "Web のコード譜を検索した要約を、機械が C 基準へ移したものと、その出典")
            + "（小節の位置は合っていない前提・区間の名前は譜のもの）",
            "【Web のコード譜の要約】", json.dumps(rec, ensure_ascii=False)]


def build_prompt(song: Song, mats: dict, *, request: str = "", prev: CoverChords | None = None, requests=None,
                 review: str = "") -> str:
    """コードの LLM への依頼文（材料は小節ごと・C 基準）。`review` は点検役の差し戻し（`gate_chords` の redo）。"""
    score, d = song.score, song.design
    ts, bar, n_bars = cover_bars(score)
    key, lead = d["key"], int(d.get("lead_bars") or 0)
    sh = key_shift(key)
    fit = ((song.source or {}).get("key") or {}).get("note", "")
    lines = [f"曲名『{score.title}』BPM{score.tempo:.0f} 拍子{ts} 全{n_bars}小節"
             + (f"（歌い出しの前の {lead}小節はどの区間にも入れていない＝書かない）" if lead else ""),
             f"■調: **{key}**（原曲の旋律から推定）。{fit}",
             f"■高さの物差し: **C 基準**で書く。この曲の調 {key} から C/Am へは **{-sh:+d}半音**——あなたのコードは機械が "
             f"{sh:+d}半音移して鳴らす。下の旋律・推定・Web の要約は C 基準へ移してある",
             f"■材料: 原曲の伴奏 MIDI からの推定＝{mats['midi_how'] if mats.get('midi') else 'なし（' + mats['midi_how'] + '）'}"
             f"／原曲の音声からの推定＝{mats['audio_how'] if mats.get('audio') else 'なし（' + mats['audio_how'] + '）'}"
             f"／Web のコード譜＝{'あり' if mats['web'].get('used') else 'なし（' + mats['web'].get('note', '') + '）'}",
             "", "【区間と小節数】（chords_per_bar の数＝小節数）"]
    for s in d["sections"]:
        b0, b1 = song.bars_of[s["kind"]]
        lines.append(f"- {s['kind']}: {s['bars']}小節（曲の {b0}〜{b1}小節目）")
    lines += ["", "【旋律と機械の推定】（小節ごと・C 基準。推定は 上位3つ 確からしさ）"]
    for s in d["sections"]:
        b0, b1 = song.bars_of[s["kind"]]
        lines.append(f"■{s['kind']}")
        for i, b in enumerate(range(b0 - 1, b1), 1):
            row = f"  {i}小節目（曲の{b + 1}小節目）: 旋律 {_melody_text(score, b, bar, sh)}"
            for kind, label in (("midi", "伴奏MIDI"), ("audio", "音声")):
                if mats.get(kind):
                    row += f" ／ {label}: {_guess_text(mats[kind][b], sh)}"
            lines.append(row)
    lines += _web_block(mats["web"])
    if prev is not None:
        lines += ["", "【前回のコード】（これを土台にする。頼まれた所だけ変え、ほかは前回のまま残す。変えた所と理由は reason に）",
                  json.dumps({s.kind: s.chords_per_bar for s in prev.sections}, ensure_ascii=False)]
    if request:
        lines += ["", f"【マスターの頼み】{request}"]
    if review:
        lines += ["", review]
    from song_requests import deliver
    asks = deliver(list(requests or []), "コード")
    if asks:
        lines += ["", asks]
    lines += ["", "全区間・全小節のコードを書いてください。"]
    return "\n".join(lines)


# ---- 書かせて、曲に載せる ----------------------------------------------------------------------------

def write_chords(song: Song, mats: dict, *, request: str = "", prev: CoverChords | None = None,
                 requests=None, review: str = "", verbose: bool = True) -> CoverChords:
    """コードの LLM を呼び（ラベル `コード`＝費用・依頼文の記録・頼み・型の差し戻しが直らなければ止める）、
    形式を `ChordsStep` で検査して差分で直させる（直らなければ `StepUnfixed("コード")`）。"""
    import song_context as sc
    import song_llm
    from song_usage import record as record_usage
    sections = [Section.model_validate(s) for s in song.design["sections"]]
    text = build_prompt(song, mats, request=request, prev=prev, requests=requests, review=review)
    keep, sc.CURRENT_SONG = sc.CURRENT_SONG, song          # 型の差し戻しが直らなかったときの記録の置き場
    try:
        r = song_llm.run_sync(_agent, text, label="コード")
        record_usage(_agent, r, "コード")
        out, _ = ChordsStep(sections).run(r.output, song, verbose=verbose)
    finally:
        sc.CURRENT_SONG = keep
    return out


def place_chords_on(song: Song, chords: CoverChords) -> None:
    """C 基準のコードを区間の調へ移して曲に載せる（組み立て・新しい曲と同じ `song_place.place_chords`）。
    `song.chords`＝LLM の出力（C 基準）・`song.chords_of`＝区間ごとの実音・`score.chords`＝時間軸に置いた実音。"""
    from song_place import keys_shift, place_chords
    ts, bar, _n = cover_bars(song.score)
    key = song.design["key"]
    shift_of = keys_shift(song.key_of, key)
    ch_map = {s.kind: list(s.chords_per_bar) for s in chords.sections}
    spans = place_chords(cover_timeline(song), ch_map, shift_of, song.key_of, bar)
    song.chords = chords
    song.chords_of = {k: [" ".join(transpose_chord(c, shift_of(k), song.key_of.get(k)) for c in CHORD_BAR.read(cell))
                          for cell in cells] for k, cells in ch_map.items()}
    song.score = song.score.model_copy(update={"chords": spans})


def _triad_id(name: str):
    from song_audio import parse_chord
    p = parse_chord(name)
    return None if p is None else (p.root % 12, p.tri in ("m", "dim"))


def _differ_bars(song: Song, chords: CoverChords, guesses: list[BarGuess] | None):
    """推定と違う小節 → ((曲の小節〈0始まり〉, 区間, 区間の中の小節〈1始まり〉, LLM のコード, 推定〈C 基準〉) の並び, 比べた小節, 取れなかった小節)。"""
    if not guesses:
        return [], 0, 0
    sh = key_shift(song.design["key"])
    out, n, miss = [], 0, 0
    for s in chords.sections:
        b0, _b1 = song.bars_of[s.kind]
        for i, cell in enumerate(s.chords_per_bar):
            g = guesses[b0 - 1 + i]
            if not g.top:
                miss += 1
                continue
            n += 1
            est = transpose_chord(g.name, -sh)
            if _triad_id(est) not in {_triad_id(c) for c in CHORD_BAR.read(cell)}:
                out.append((b0 - 1 + i, s.kind, i + 1, cell, est))
    return out, n, miss


def differing(song: Song, chords: CoverChords, guesses: list[BarGuess] | None) -> tuple[list[str], int, int]:
    """LLM のコードと機械の推定（いちばん確からしいもの）が違う小節 → (「N小節目: 推定 X → Y」の並び, 比べた小節, 取れなかった小節)。
    比べるのは根音と長短（推定は3和音しか答えないので、`Am7` と `Am` は同じと数える）。機械は数えるだけで、書き換えない。"""
    rows, n, miss = _differ_bars(song, chords, guesses)
    return [f"{b + 1}小節目: 推定 {est} → {cell}" for b, _k, _i, cell, est in rows], n, miss


#: 中身の点検役の判断 → 返事の言葉（点検役そのものは `song_cover_chords_gate`・2026-10-03 組K4）
GATE_WORD = {"accept": "通した", "redo": "やり直しを求めた", "stop": "所見つきで進んだ"}


def _gate_note(gate: dict) -> str:
    if not gate:
        return ""
    head = "1回やり直させてから" if gate.get("redone") else ""
    return (f"。点検役: {head}{GATE_WORD.get(gate.get('action', ''), gate.get('action', ''))}"
            f"（{_flat(gate.get('reason', ''), 160)}）"
            + (f"。点検役が気になった所: {_flat(gate['notes'], 200)}" if gate.get("notes") else ""))


def _note_of(meta: dict) -> str:
    """返事に出す一文（記録の `cover_skeleton` から）。"""
    if not meta.get("has_chords"):
        return ""
    web = meta.get("web") or {}
    head = ("コードは今回 LLM が書いた" if meta.get("fresh") else
            f"コードは曲の記録のもの（{meta.get('made', '')} に LLM が書いた）を使った——変えたいなら remake_from_step('コード', 頼み)")
    return (f"{head}（材料: 旋律・曲の知識〔{meta.get('known', '?')}〕・{meta.get('basis', '機械の推定なし')}）。"
            f"機械の推定と違う小節: {meta.get('n_differ', 0)}/{meta.get('n_compared', 0)}"
            + (f"（{' ／ '.join(meta.get('differ', [])[:4])}{' ほか' if meta.get('n_differ', 0) > 4 else ''}）"
               if meta.get("differ") else "")
            + (f"。推定が取れなかった小節: {meta['n_missing']}" if meta.get("n_missing") else "")
            + ("。Web のコード譜: 使った（" + " ".join(web.get("urls") or ["出典の URL なし"]) + "）" if web.get("used")
               else f"。Web のコード譜: 使わなかった（{web.get('note', '')}）")
            + _gate_note(meta.get("gate") or {}))


# ---- 入口 ---------------------------------------------------------------------------------------

def prepare_cover(song: Song, record: Song | None = None, *, chords: bool = True, redo_chords: bool = False,
                  request: str = "", verbose: bool = True) -> list[StepRecord]:
    """**カバーの区間・調・コードを用意する**（J1）。曲の記録にあれば読み、無ければ作る。戻り値は今回作った工程の記録。

    `chords=False` は区間と調だけ（書き出しの音量の点検 `check_mix_quality`）。`redo_chords=True` は保存したコードを土台に
    書き直させる（`remake_from_step('コード', 頼み)`）。作ったら `dirty(song)` が真になる——呼び手（道具）が `save_into` で記録へ残す。
    """
    src = song.source = dict(song.source or {})
    offset = transposed(getattr(record, "requests", None))
    meta = dict(((getattr(record, "source", None) or {}).get(SKELETON)) or {})
    ts, _bar, n_bars = cover_bars(song.score)
    made: list[StepRecord] = []
    stale = ""
    d = getattr(record, "design", None) if meta else None
    if isinstance(d, dict) and meta.get("n_bars") == n_bars and meta.get("time_signature") == ts:
        place_sections(song, moved_key(d["key"], offset), [Section.model_validate(s) for s in d["sections"]],
                       int(d.get("lead_bars") or 0))
        if set(record.key_of or {}) == set(song.key_of):   # 区間ごとの調（尺の変更で足した区間の調）も記録のまま
            song.key_of = {k: moved_key(v, offset) for k, v in record.key_of.items()}
        src["key"] = key_fit(song.score.vocal_notes, [], song.design["key"])
    else:
        if meta:
            stale = (f"曲の小節（{n_bars}小節・{ts}）が記録の区間とコードを作った時（{meta.get('n_bars')}小節・"
                     f"{meta.get('time_signature')}）と違うので、区間・調・コードを作り直した")
        made.append(add_sections(song, verbose=verbose))
        src = song.source                                   # `add_sections` が `source` を作り直す（推定した調の合い具合）
        meta = {"n_bars": n_bars, "time_signature": ts, "has_chords": False, "made": f"{datetime.now():%Y-%m-%d %H:%M}"}
        src["skeleton_dirty"] = True
    if not chords:
        src[SKELETON] = meta
        return made
    prev = CoverChords.model_validate(record.chords) if meta.get("has_chords") and getattr(record, "chords", None) else None
    if prev is not None and not redo_chords:
        place_chords_on(song, prev)
        src[SKELETON] = {**meta, "fresh": False}
        src["chords_note"] = _note_of(src[SKELETON])
        return made

    mats = materials(song, web=meta.get("web"))
    guesses = mats.get("midi") or mats.get("audio")
    est = [g.name for g in guesses or [] if g.top]
    src["key"] = key_fit(song.score.vocal_notes, est, song.design["key"])   # 調の合い具合は原曲の伴奏の推定も数える（依頼文にも出す）
    asks = getattr(record, "requests", None)
    out = write_chords(song, mats, request=request, prev=prev if redo_chords else None, requests=asks, verbose=verbose)
    place_chords_on(song, out)
    # 中身の点検役（2026-10-03・組K4）。やり直しは1回まで。点検役は別ファイル（import の向きは点検役 → ここ）なので関数の中で読む
    from song_cover_chords_gate import review_chords
    out, v, gate = review_chords(song, out, mats, request=request, requests=asks, verbose=verbose)
    differ, n_cmp, n_miss = differing(song, out, guesses)
    basis = "・".join(f"{lbl}の推定（{mats[f'{k}_how']}）" for k, lbl in (("midi", "原曲の伴奏 MIDI"), ("audio", "原曲の音声"))
                      if mats.get(k)) or "機械の推定なし"
    meta = {**meta, "has_chords": True, "fresh": True, "made": f"{datetime.now():%Y-%m-%d %H:%M}", "known": out.known,
            "basis": basis, "differ": differ, "n_differ": len(differ), "n_compared": n_cmp, "n_missing": n_miss,
            "web": mats["web"], "gate": gate,
            "no_material": [f"{lbl}: {mats[f'{k}_how']}" for k, lbl in
                            (("midi", "原曲の伴奏 MIDI"), ("audio", "原曲の音声")) if not mats.get(k)]}
    src[SKELETON] = meta
    src["skeleton_dirty"] = True
    src["chords_note"] = _note_of(meta)
    warns = ([stale] if stale else []) + ([f"機械の推定と違う小節 {len(differ)}/{n_cmp}: " + " ／ ".join(differ[:8])] if differ else [])
    warns += ([] if guesses else ["機械の推定が無い（" + " ／ ".join(meta["no_material"]) + "）——旋律と曲の知識だけで書いた"])
    warns += [src["key"]["warning"]] if src["key"].get("warning") else []
    warns += [f"点検役: 所見つきで進んだ——{v.reason}"] if v.action == "stop" else []
    rec = StepRecord(step="chords", reason=f"{out.reason}（{src['chords_note']}）。{src['key']['note']}",
                     status=out.status, trouble=out.trouble, warnings=warns, request=request)
    song.record(rec)
    made.append(rec)
    return made


def dirty(song: Song) -> bool:
    """今回、区間・調・コードを作った（曲の記録へ残す必要がある）か。"""
    return bool((song.source or {}).get("skeleton_dirty"))


def save_into(record: Song, song: Song) -> None:
    """カバーの区間・調・コードを曲の記録（`record`）へ写す。調・コードの実音は**手直しの移調を引いた元の高さ**で残す
    （読むときに台帳の移調を足す・ハモリ追加 `editing._added_voice_basis` と同じ数え方）。"""
    off = transposed(getattr(record, "requests", None))
    d = dict(song.design or {})
    d["key"] = moved_key(d.get("key") or "", -off)
    record.design = d
    record.key_of = {k: moved_key(v, -off) for k, v in (song.key_of or {}).items()}
    record.bars_of = dict(song.bars_of or {})
    record.sec_of_note = list(song.sec_of_note or [])
    record.chords = song.chords
    record.chords_of = {k: [" ".join(transpose_chord(c, -off, record.key_of.get(k)) for c in CHORD_BAR.read(cell))
                            for cell in cells] for k, cells in (song.chords_of or {}).items()}
    meta = {k: v for k, v in ((song.source or {}).get(SKELETON) or {}).items() if k != "fresh"}
    record.source = {**(record.source or {}), SKELETON: meta}


def resized(song: Song) -> None:
    """**尺の変更のあと**: 並べ替えた区間と区間ごとのコード（`chords_of`＝実音）から、C 基準のコード（`song.chords`）と
    骨組みの記録を作り直す（訳すだけ。コピーした区間はコードの LLM が書いたもの、足した区間は尺の変更の LLM が書いたもの）。"""
    from song_fix import record_change
    from song_place import keys_shift
    d = dict(song.design or {})
    d["lead_bars"] = 0                                     # 尺の変更は区間を曲頭から並べる（前奏は新しい構成に入らない）
    song.design = d
    shift_of = keys_shift(song.key_of, d.get("key") or "")
    prev = song.chords
    prev_d = prev.model_dump() if hasattr(prev, "model_dump") else (prev or {})
    song.chords = CoverChords(
        reason=(prev_d.get("reason") or "") + "（尺の変更で区間を並べ替え、区間ごとのコードを写した）",
        known=prev_d.get("known") or "知らない",
        sections=[SectionChords(kind=s["kind"], chords_per_bar=[" ".join(transpose_chord(c, -shift_of(s["kind"]))
                                                                        for c in CHORD_BAR.read(cell))
                                                               for cell in song.chords_of.get(s["kind"], [])])
                  for s in d.get("sections") or []])
    record_change(song, "chords", "尺の変更で区間を並べ替え、コードを区間ごとに写した（新しい区間のコードは尺の変更の LLM が書いたもの）",
                  "コードの段の記録（C 基準）を新しい構成に合わせるため")
    ts, _bar, n_bars = cover_bars(song.score)
    src = song.source = dict(song.source or {})
    src[SKELETON] = {**(src.get(SKELETON) or {}), "n_bars": n_bars, "time_signature": ts, "has_chords": True,
                     "fresh": False, "resized": f"{datetime.now():%Y-%m-%d %H:%M}"}
    src["skeleton_dirty"] = True
