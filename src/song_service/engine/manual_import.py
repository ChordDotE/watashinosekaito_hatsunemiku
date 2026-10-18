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

"""**人が手で直した vpr・MIDI を楽譜に取り込む**（2026-09-26・`docs/plan/manual_edit_import.md` §5）。

楽譜が正のまま、人の直しを楽譜へ移す。**楽譜の言葉で書けるものは楽譜の欄へ**（音の高さ・位置・長さ・歌詞・
音ごとの表情・声の種類・テンポ・楽器を消す）、**書けないものは区間つきで楽譜の「手の直し」の欄へ**
（手で描いたカーブ・発音記号・歌トラックの音量 → `SongScore.hand_edits`／特定の小節の伴奏 → `ArrangementSpec.hand_bars`）。
取り込んだ楽譜から書き出すと、人が直した vpr・MIDI と同じものができる（往復で確かめる）。

差分から動かすので、**触っていない音符は、持っていた情報（歌の崩しの印など）をそのまま保つ**。
"""
from __future__ import annotations

from pathlib import Path

from manual_diff import read_midi, read_vpr
from song_schema import PartSpec, SongNote, SongScore, VocalTuning, VoiceTrack

Q = 480


def _aiexp_to_field() -> dict[str, str]:
    from score_to_vpr import _MAP
    out = {}
    for f, keys in _MAP.items():
        for k in keys:
            out.setdefault(k, f)
    return out


def _tuning_with(t: VocalTuning | None, ai: dict, changed: dict | None = None) -> VocalTuning | None:
    """aiExp の値を `VocalTuning` に写す。`changed` があればその項目だけ、無ければ既定値でない項目を全部。"""
    from score_to_vpr import AIEXP_DEFAULT
    rev = _aiexp_to_field()
    upd = {}
    for k, v in ai.items():
        if k not in rev:
            continue
        if changed is not None and k not in changed:
            continue
        if changed is None and v == AIEXP_DEFAULT.get(k):
            continue
        upd[rev[k]] = float(v)
    if not upd:
        return t
    return (t or VocalTuning()).model_copy(update=upd)


def bars_label(a: int, b: int) -> str:
    """小節の範囲 [a, b)（0始まり）を人が読む形に。"""
    return f"{a + 1}小節" if b <= a + 1 else f"{a + 1}〜{b}小節"


def _regions(ticks: list[int], gap: int) -> list[tuple[int, int]]:
    """変わった位置を、`gap` 以上離れたら別のかたまりにまとめる。"""
    out: list[list[int]] = []
    for t in sorted(ticks):
        if out and t - out[-1][1] <= gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [(a, b) for a, b in out]


def _merge_hand(old: list[dict], new: list[dict]) -> list[dict]:
    """同じトラック・同じ種類・同じカーブで範囲が重なる古い手の直しは、新しいもので置き換える。"""
    def key(e):
        return (e.get("track"), e.get("kind"), e.get("name"))

    def overlap(a, b):
        if a["kind"] == "volume":
            return True
        s0, e0 = a.get("start", 0), a.get("end", a.get("start", 0))
        s1, e1 = b.get("start", 0), b.get("end", b.get("start", 0))
        return s0 <= e1 and s1 <= e0
    kept = [o for o in old if not any(key(o) == key(n) and overlap(o, n) for n in new)]
    return kept + new


def _phoneme_edits(score: SongScore, act: dict) -> list[dict]:
    import tempfile
    from score_to_vpr import score_to_vpr
    tmp = Path(tempfile.mkdtemp()) / "vocal.vpr"
    back = read_vpr(score_to_vpr(score.model_copy(update={"hand_edits": []}), tmp))
    out = []
    for name, a in act.items():
        bm = {n["start"]: n for n in (back.get(name) or {}).get("notes", [])}
        for n in a["notes"]:
            y = bm.get(n["start"])
            if y is not None and n["phoneme"] and y["phoneme"] != n["phoneme"]:
                out.append({"kind": "phoneme", "track": name, "start": n["start"], "phoneme": n["phoneme"]})
    return out


def take_in_vpr(score: SongScore, expected: Path, actual: Path, bar_ticks: int
                ) -> tuple[SongScore, list[str], dict]:
    """vpr の手の直しを楽譜へ。戻り値は (新しい楽譜, 取り込んだことの説明, 台帳へ残す材料)。

    台帳へ残す材料: `{"regions": [(開始, 終わり, 歌の音符の並び)], "voices": [VoiceTrack], "hand": 件数}`
    """
    exp, act = read_vpr(expected), read_vpr(actual)
    lead = score.title
    voices = {v.title: v for v in score.extra_voices}
    said: list[str] = []
    hand: list[dict] = []
    regions: list = []
    new_voices: list[VoiceTrack] = []
    removed: list[str] = []
    choice = dict(score.voice_choice or {})
    vocal_notes = list(score.vocal_notes)
    extra = list(score.extra_voices)

    for name in sorted(set(exp) | set(act)):
        if name not in act:
            if name in voices:
                extra = [v for v in extra if v.title != name]
                removed.append(name)
                said.append(f"声部『{name}』を外した")
            continue
        a = act[name]
        if name not in exp:                                 # 人が足した歌のトラック
            vt = VoiceTrack(title=name, notes=[SongNote(lyric=n["lyric"] or "あ", key=n["key"], start_tick=n["start"],
                                                        duration_tick=max(1, n["dur"]),
                                                        tuning=_tuning_with(None, n["aiExp"]))
                                               for n in a["notes"]])
            extra.append(vt)
            new_voices.append(vt)
            said.append(f"声部『{name}』を足した（{len(vt.notes)}音）")
            continue
        e = exp[name]
        base_list = vocal_notes if name == lead else (voices[name].notes if name in voices else None)
        if base_list is None:
            continue
        by_start = {n.start_tick: n for n in base_list}
        em, am = {n["start"]: n for n in e["notes"]}, {n["start"]: n for n in a["notes"]}
        out, moved = [], []
        for t, n in sorted(am.items()):
            x = em.get(t)
            base = by_start.get(t)
            if x is not None and base is not None:
                ch_ai = {k for k, v in n["aiExp"].items() if x["aiExp"].get(k) != v}
                upd = {}
                if n["key"] != x["key"]:
                    upd["key"] = n["key"]
                if n["lyric"] != x["lyric"]:
                    upd["lyric"] = n["lyric"]
                if n["dur"] != x["dur"]:
                    upd["duration_tick"] = max(1, n["dur"])
                if ch_ai:
                    upd["tuning"] = _tuning_with(base.tuning, n["aiExp"], ch_ai)
                if upd:
                    moved.append(t)
                    base = base.model_copy(update=upd)
                out.append(base)
            else:                                          # 人が足した音
                out.append(SongNote(lyric=n["lyric"] or "あ", key=n["key"], start_tick=t,
                                    duration_tick=max(1, n["dur"]), tuning=_tuning_with(None, n["aiExp"])))
                moved.append(t)
        moved += [t for t in em if t not in am]            # 人が消した音
        if name == lead:
            vocal_notes = out
            for s, t in _regions(moved, 2 * bar_ticks):
                e_end = t + max((n.duration_tick for n in out if n.start_tick == t), default=Q)
                regions.append((s, e_end, [n for n in out if s <= n.start_tick < e_end]))
        elif moved:
            extra = [v.model_copy(update={"notes": out}) if v.title == name else v for v in extra]
            new_voices.append(next(v for v in extra if v.title == name))
        if moved:
            said.append(f"『{name}』の音 {len(set(moved))} か所を楽譜に取り込んだ")
        for cname, evs in a["curves"].items():
            base_ev = e["curves"].get(cname) or []
            diff = ({(x["pos"], x["value"]) for x in evs} ^ {(x["pos"], x["value"]) for x in base_ev})
            if not diff:
                continue
            s, t = min(p for p, _ in diff), max(p for p, _ in diff)
            hand.append({"kind": "curve", "track": name, "name": cname, "start": s, "end": t,
                         "events": [x for x in evs if s <= x["pos"] <= t]})
            said.append(f"『{name}』の手で描いたカーブ {cname} を手の直しの欄に持った")
        if a["volume"] != e["volume"]:
            hand.append({"kind": "volume", "track": name, "events": a["volume"]})
            said.append(f"『{name}』のトラックの音量を手の直しの欄に持った")
        if a["style"] != e["style"] and a["style"]:
            choice["style" if name == lead else "harmony_style"] = a["style"]
            said.append(f"『{name}』の声の種類を {a['style']} にした")

    new = score.model_copy(update={"vocal_notes": sorted(vocal_notes, key=lambda n: n.start_tick),
                                   "extra_voices": extra, "voice_choice": choice})
    # 発音記号: 取り込んだ楽譜から一度書いてみて、人のファイルと違う音だけを手の直しとして持つ
    # （歌詞を変えると V6 が発音記号を付け直すので、元の vpr と比べると「歌詞を変えた音」が全部違って見える）
    hand += _phoneme_edits(new, act)
    n_ph = sum(1 for e in hand if e["kind"] == "phoneme")
    if n_ph:
        said.append(f"発音記号の手直し {n_ph} か所を手の直しの欄に持った")
    new = new.model_copy(update={"hand_edits": _merge_hand(list(score.hand_edits or []), hand)})
    return new, said, {"regions": regions, "voices": new_voices, "removed": removed, "hand": len(hand)}


def _track_info(path: Path) -> dict[str, tuple[int, int]]:
    """トラック名ごとの (チャンネル, 音色番号)。人が足したトラックの音色を引き継ぐため。"""
    import mido
    out: dict[str, tuple[int, int]] = {}
    for i, tr in enumerate(mido.MidiFile(str(path)).tracks):
        name, ch, prog = f"track{i}", None, 0
        for m in tr:
            if m.type == "track_name":
                name = m.name or name
            elif m.type == "program_change":
                prog = m.program
            elif m.type == "note_on" and ch is None:
                ch = m.channel
        if ch is not None:
            out.setdefault(name, (ch, prog))
    return out


def take_in_midi(score: SongScore, expected: Path, actual: Path, bar_ticks: int
                 ) -> tuple[SongScore, list[str], dict]:
    """MIDI の手の直しを楽譜へ。戻り値は (新しい楽譜, 取り込んだことの説明, 台帳へ残す材料)。"""
    from manual_diff import diff_midi
    d = diff_midi(expected, actual, bar_ticks)
    act, _ = read_midi(actual)
    info, info_exp = _track_info(actual), _track_info(expected)
    arr = score.arrangement
    if arr.is_empty():                                     # 伴奏が無い＝パートが0本（F1。以前は None）
        return score, ["伴奏の楽譜が無い曲なので、MIDI の直しは取り込めなかった"], {}
    parts, hand_bars = list(arr.parts), list(arr.hand_bars or [])
    said, bars_note, tempo_new = [], [], None
    from midi_out import _ascii
    role_of = {_ascii(p.role): p.role for p in parts}      # MIDI のトラック名は英数字だけに削ってある
    for mname, x in d.items():
        if mname == "tempo":
            tempo_new = x[1]
            said.append(f"テンポを {x[0]} → {x[1]} にした")
            continue
        name = role_of.get(mname, mname)
        if x.get("track") == "消えた":
            parts = [p for p in parts if p.role != name]
            hand_bars = [h for h in hand_bars if h.get("role") != name]
            said.append(f"楽器『{name}』を外した")
            continue
        notes = act.get(mname, [])
        if x.get("track") == "増えた":
            ch, prog = info.get(mname, (0, 0))
            if ch == 9 and name != "drums":                 # 打楽器は名前でドラム扱いを決めている（midi_out）
                said.append(f"楽器『{name}』は打楽器のチャンネルなので取り込まなかった"
                            "（トラック名を drums にすると取り込める）")
                continue
            # `bass` はパターン（`steps`）の弾き方だけに効く。手で足した楽器はパターンを持たず、手で書いた音をそのまま鳴らすので false
            parts.append(PartSpec(role=name, bass=False, timbre="" if ch == 9 else f"gm{prog + 1}", steps=[]))
            bs = sorted({n[0] // bar_ticks for n in notes})
            ranges = [(bs[0], bs[-1] + 1)] if bs else []
            said.append(f"楽器『{name}』を足した（{len(notes)}音・手で書いた小節として持った）")
        else:
            ranges = [(a, b + 1) for a, b in _regions(list(x["bars"]), 1)]
            said.append(f"楽器『{name}』の " + "・".join(bars_label(a, b) for a, b in ranges)
                        + " を手で書いた小節として持った")
        for a, b in ranges:
            hand_bars = [h for h in hand_bars
                         if not (h.get("role") == name and h["start_bar"] < b and a < h["end_bar"])]
            hand_bars.append({"role": name, "start_bar": a, "end_bar": b,
                              "notes": [list(n) for n in notes if a * bar_ticks <= n[0] < b * bar_ticks]})
            bars_note.append((name, a, b))
    for mname, (ch, prog) in info.items():                # 音色だけ変えた楽器（音符の差には出ない）
        name = role_of.get(mname, mname)
        if ch == 9 or mname not in info_exp or info_exp[mname][1] == prog:
            continue
        parts = [p.model_copy(update={"timbre": f"gm{prog + 1}"}) if p.role == name else p for p in parts]
        said.append(f"楽器『{name}』の音色を GM {prog + 1} 番にした")
    new_arr = arr.model_copy(update={"parts": parts, "hand_bars": hand_bars})
    upd = {"arrangement": new_arr}
    if tempo_new:
        upd["tempo"] = float(tempo_new)
    return score.model_copy(update=upd), said, {"bars": bars_note, "tempo": tempo_new}


def _vocal_hand_desc(edits: list[dict]) -> str:
    """歌の手の直しの中身（種類ごとの数と、拍の範囲）。"""
    kinds: dict[str, int] = {}
    ticks: list[int] = []
    for e in edits:
        k = {"phoneme": "発音記号", "curve": f"カーブ（{e.get('name') or '?'}）", "volume": "音量"}.get(
            e.get("kind", ""), e.get("kind", "?"))
        kinds[k] = kinds.get(k, 0) + 1
        for key in ("start", "end"):
            if isinstance(e.get(key), (int, float)):
                ticks.append(int(e[key]))
    where = (f"・{min(ticks) / Q:g}〜{max(ticks) / Q:g}拍" if ticks else "")
    return "、".join(f"{k}{n}件" for k, n in kinds.items()) + where


def _bars_desc(hb: list[dict]) -> str:
    return "、".join(f"{h.get('role', '?')} の {bars_label(int(h['start_bar']), int(h['end_bar']))}"
                     for h in hb if "start_bar" in h and "end_bar" in h)


def hand_edit_scope(prev: SongScore | None, step: str) -> str:
    """**作り直す工程へ添える「この範囲は人が直している」**（2026-09-27・目録の道具-8）。

    作り直したあと、下が変わっていなければ人の手直しは機械が上に重ねる（`carry_hand_edits`）。
    それを工程が知らないと、作り直した歌い方や伴奏の上に、知らない手直しが黙って乗る。
    **引き継ぐものだけ**を書く（引き継がないものは作り直しの結果が正になる）。無ければ空。
    """
    if prev is None:
        return ""
    from song_edits import notes_redone
    from song_gate import affected
    parts = []
    if prev.hand_edits and not notes_redone(step):
        parts.append(f"歌: {_vocal_hand_desc(prev.hand_edits)}（V6 で人が直した。作り直したあとも上に重ねる）")
    hb = prev.arrangement.hand_bars
    if hb and not (affected(step) & {"設計", "コード", "グルーヴ", "編成", "伴奏"}):
        parts.append(f"伴奏: {_bars_desc(hb)}（MIDI で人が手で書いた小節。パターンより優先して鳴る）")
    if not parts:
        return ""
    return ("【この範囲は人が直している】作り直したあと、次の手直しを機械がそのまま上に重ねる。"
            "ここは人の手が正なので、**合わせて作る**（ぶつかる書き方をしない）:\n"
            + "\n".join(f"  ・{p}" for p in parts))


def carry_hand_edits(prev: SongScore | None, new: SongScore, step: str,
                     carried: list[str] | None = None) -> tuple[SongScore, list[str]]:
    """作り直したあと、前の版の楽譜から手の直しを引き継ぐ（下が変わっていなければ）。

    - 歌の手の直し（カーブ・発音記号・音量）… 旋律・詞を作り直していなければ引き継ぐ
    - 伴奏の手で書いた小節 … 設計・コード・グルーヴ・編成・伴奏を作り直していなければ引き継ぐ
    引き継げなかったものは説明を返す（**黙って消さない**。前の版には残っている）。
    ★**引き継いだものも `carried` に書く**（2026-09-27・目録の道具-8）。以前は黙って重ねていたので、
      作り直した工程もコーディネーターも、人の手直しが上に乗ったことを知らなかった。
      作り直す前に `hand_edit_scope` を工程の依頼文へ添え、あとでこの一覧を返事に載せる。
    """
    if prev is None:
        return new, []
    from song_edits import notes_redone
    from song_gate import affected
    said, upd = [], {}
    if prev.hand_edits:
        if notes_redone(step):
            said.append(f"歌の手の直し {len(prev.hand_edits)} 件（手で描いたカーブ・発音記号など）は、旋律か詞を作り直したので"
                        "引き継いでいない（前の版に残っている）")
        else:
            upd["hand_edits"] = list(prev.hand_edits)
            if carried is not None:
                carried.append(f"歌の手の直し（{_vocal_hand_desc(prev.hand_edits)}）を作り直した歌の上に重ねた")
    hb = prev.arrangement.hand_bars
    if hb and not new.arrangement.is_empty():               # 新しい版に伴奏が無ければ（パートが0本）置き場が無い
        if affected(step) & {"設計", "コード", "グルーヴ", "編成", "伴奏"}:
            said.append(f"伴奏の手で書いた小節 {len(hb)} か所は、伴奏の土台を作り直したので引き継いでいない（前の版に残っている）")
        else:
            upd["arrangement"] = new.arrangement.model_copy(update={"hand_bars": list(hb)})
            if carried is not None:
                carried.append(f"伴奏の手で書いた小節（{_bars_desc(hb)}）を引き継いだ（パターンより優先して鳴る）")
    return (new.model_copy(update=upd) if upd else new), said
