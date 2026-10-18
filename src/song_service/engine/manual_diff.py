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

"""**人が手で直した vpr・MIDI の、どこがどう変わったか**（2026-09-26・`docs/plan/manual_edit_import.md`）。

楽譜が正で、vpr と MIDI は楽譜から毎回作る出力。人が VOCALOID6 や MIDI の道具で直すと、次の書き出しで
**黙って上書きされる**。ここは「何が変わったか」を出すだけの部品（取り込みは `song_edits` / 道具の側）。

見分け方の要点:
- **版を確定するときにファイルの指紋を控える**（`write_manifest`）。どのファイルが変わったかはこれで分かる
- **どこがどう変わったか**は、その版の楽譜から**システムが書いたはずのもの**を作り直して突き合わせる。
  作り方が決まっているので、**同じ名前で上書きされて元が無くても比べられる**
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import song_files                                         # 曲フォルダのファイルの名前の決まり（2026-10-03）

Q = 480
#: 人が直しうるもの（2026-09-26 本人確認: vpr と MIDI）
EDITABLE = (".vpr", ".mid", ".midi")


# ---- 指紋 ------------------------------------------------------------------------------

#: 作業中だけ存在する固定名（`song_files.WORKBENCH`＝`session._VERSIONED`）。セッション中に見ると「増えた」に見えるので数えない
_WORKBENCH = set(song_files.WORKBENCH)


def _counted(name: str) -> bool:
    """指紋を控えるファイルか。**途中の物だけ**（`song_files.kind_of` が `work`）を数える——システムが書き足していくログ（`log_`）・
    機械の印（`_`）と、**完成品（`master_`）は数えない**（2026-10-03・F12〜F15。人が直すのは vpr と MIDI で、完成品の付け替えや
    書き出しを「手の直し」と誤報しない）。作業台の固定名も数えない。"""
    return song_files.kind_of(name) == "work" and name not in _WORKBENCH


def fingerprint(song_dir: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(song_dir.iterdir()):
        if not p.is_file() or not _counted(p.name):
            continue
        out[p.name] = {"sha1": hashlib.sha1(p.read_bytes()).hexdigest(), "size": p.stat().st_size,
                       "mtime": int(p.stat().st_mtime)}
    return out


def write_manifest(song_dir: Path, version: int) -> Path:
    """版を確定したときの指紋を控える（数ミリ秒〜・中身は読むだけ）。"""
    p = song_dir / song_files.manifest(version)
    p.write_text(json.dumps({"version": version, "files": fingerprint(song_dir)}, ensure_ascii=False, indent=1),
                 encoding="utf-8")
    return p


def latest_manifest(song_dir: Path) -> dict | None:
    _base = song_files.split_version(song_files.manifest(1))[0]   # 版を外した名前（`_manifest.json`）
    ms = sorted((p for p in song_dir.iterdir()
                 if p.is_file() and (got := song_files.split_version(p.name)) and got[0] == _base),
                key=lambda p: song_files.version_of(p.name))
    if not ms:
        return None
    try:
        return json.loads(ms[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def changed_files(song_dir: Path) -> tuple[list[dict], bool]:
    """控えと比べて、人が直しうるファイルのうち変わった・増えた・消えたもの。

    戻り値は (一覧, 控えがあったか)。控えが無い（最初の版を確定する前の曲）ときは、人が直しうるファイルを全部候補にする。
    """
    now = fingerprint(song_dir)
    man = latest_manifest(song_dir)
    if man is None:
        return [{"name": n, "how": "候補（控えが無い）"} for n in now if n.lower().endswith(EDITABLE)], False
    old = man.get("files") or {}
    out = []
    for n, f in now.items():
        if not n.lower().endswith(EDITABLE):
            continue
        if n not in old:
            out.append({"name": n, "how": "増えた"})
        elif f["sha1"] != old[n]["sha1"]:
            out.append({"name": n, "how": "変わった"})
    for n in old:
        if n.lower().endswith(EDITABLE) and n not in now:
            out.append({"name": n, "how": "消えた"})
    return out, True


def version_of(name: str, default: int) -> tuple[int, bool]:
    """名前に入っている版番号（`vocal_v6_手直し.vpr` → 6）。無ければ (default, False)。

    システムが付けた名前の版は `song_files.version_of` の1つで読む（F16①）。人が別名で保存したもの（`vocal_v6_手直し.vpr`）は
    `song_files` の版の形ではないので、名前の中の `_v{N}` を拾う（人が付けた名前を読む所はここだけ）。"""
    if (n := song_files.version_of(name)) is not None:
        return n, True
    m = re.search(r"_v(\d+)", name)
    return (int(m.group(1)), True) if m else (default, False)


# ---- 歌（vpr）---------------------------------------------------------------------------

def read_vpr(path: Path) -> dict[str, dict]:
    """歌のトラックごとに、音符（曲頭からの位置）・カーブ・音量・声の種類を読む。"""
    seq = json.loads(zipfile.ZipFile(path).read("Project/sequence.json"))
    out: dict[str, dict] = {}
    for tr in seq.get("tracks", []):
        if tr.get("type") != 2:                            # 2＝歌のトラック（1＝伴奏の音声）
            continue
        notes, curves, style = [], {}, ""
        for pt in tr.get("parts") or []:
            base = int(pt.get("pos", 0))
            style = pt.get("styleName") or style
            for n in pt.get("notes") or []:
                notes.append({"start": base + int(n["pos"]), "dur": int(n["duration"]), "key": int(n["number"]),
                              "lyric": n.get("lyric", ""), "phoneme": n.get("phoneme", ""),
                              "protected": bool(n.get("isProtected")), "aiExp": n.get("aiExp") or {}})
            for c in pt.get("controllers") or []:
                curves.setdefault(c.get("name", ""), []).extend(
                    {"pos": base + int(e["pos"]), "value": e["value"]} for e in c.get("events") or [])
        out[tr.get("name") or f"track{len(out)}"] = {
            "notes": sorted(notes, key=lambda x: x["start"]), "curves": curves,
            "volume": (tr.get("volume") or {}).get("events"), "style": style}
    return out


def diff_vpr(expected: Path, actual: Path) -> dict[str, dict]:
    """歌のトラックごとの差。`expected` はその版の楽譜から作り直した vpr。"""
    exp, act = read_vpr(expected), read_vpr(actual)
    res: dict[str, dict] = {}
    for name in sorted(set(exp) | set(act)):
        if name not in act:
            res[name] = {"track": "消えた"}
            continue
        if name not in exp:
            res[name] = {"track": "増えた", "notes": len(act[name]["notes"])}
            continue
        e, a = exp[name], act[name]
        em, am = {n["start"]: n for n in e["notes"]}, {n["start"]: n for n in a["notes"]}
        d = {"added": [am[t] for t in sorted(set(am) - set(em))],
             "removed": [em[t] for t in sorted(set(em) - set(am))], "changed": []}
        for t in sorted(set(em) & set(am)):
            x, y = em[t], am[t]
            what = {k: (x[k], y[k]) for k in ("key", "lyric", "dur") if x[k] != y[k]}
            if x["phoneme"] != y["phoneme"]:
                what["phoneme"] = (x["phoneme"], y["phoneme"])
            ex = {k: (x["aiExp"].get(k), v) for k, v in y["aiExp"].items() if x["aiExp"].get(k) != v}
            if ex:
                what["aiExp"] = ex
            if what:
                d["changed"].append({"start": t, "lyric": y["lyric"], "what": what})
        cur = {k: v for k, v in a["curves"].items() if v != e["curves"].get(k)}
        if cur:
            d["curves"] = {k: _span(v) for k, v in cur.items()}
        if a["volume"] != e["volume"]:
            d["volume"] = (e["volume"], a["volume"])
        if a["style"] != e["style"]:
            d["style"] = (e["style"], a["style"])
        if any(d[k] for k in ("added", "removed", "changed")) or len(d) > 3:
            res[name] = d
    return res


def _span(events: list[dict]) -> tuple[int, int]:
    ps = [e["pos"] for e in events] or [0]
    return (min(ps), max(ps))


# ---- 伴奏（MIDI）------------------------------------------------------------------------

def read_midi(path: Path) -> tuple[dict[str, list[tuple[int, int, int, int]]], float | None]:
    """トラック名ごとの音 `(開始, 高さ, 長さ, 強さ)`（同じ名前のトラックはまとめる）とテンポ。"""
    import mido
    mid = mido.MidiFile(str(path))
    out: dict[str, list] = {}
    tempo = None
    scale = Q / (mid.ticks_per_beat or Q)
    for i, tr in enumerate(mid.tracks):
        name, t, on = f"track{i}", 0, {}
        notes = []
        for msg in tr:
            t += msg.time
            if msg.type == "track_name":
                name = msg.name or name
            elif msg.type == "set_tempo" and tempo is None:
                tempo = round(mido.tempo2bpm(msg.tempo), 3)
            elif msg.type == "note_on" and msg.velocity > 0:
                on[(msg.channel, msg.note)] = (t, msg.velocity)
            elif msg.type in ("note_off", "note_on"):
                st = on.pop((msg.channel, msg.note), None)
                if st:
                    notes.append((int(st[0] * scale), msg.note, int((t - st[0]) * scale), st[1]))
        if notes:
            out.setdefault(name, []).extend(notes)
    return {k: sorted(v) for k, v in out.items()}, tempo


def _lead_track() -> str:
    """主旋律の楽器のトラック名（書き出す側 `midi_out.LEAD_TRACK` の1つ・2026-10-03）。"""
    from midi_out import LEAD_TRACK, _ascii
    return _ascii(LEAD_TRACK)


def lead_changed(expected: Path, actual: Path) -> bool:
    """主旋律の楽器のトラック（`midi_out.LEAD_TRACK`）の音が、楽譜から作るものと違うか（足した・消した・変えた）。"""
    from collections import Counter
    name = _lead_track()
    return Counter(read_midi(expected)[0].get(name, [])) != Counter(read_midi(actual)[0].get(name, []))


def without_lead(path: Path) -> Path:
    """伴奏 MIDI から主旋律の楽器のトラックを除いた写し（一時フォルダ）。

    ★主旋律の楽器は旋律の工程の音そのもの（楽譜の `lead`）で、伴奏のパートではない。手で直した伴奏 MIDI を取り込むとき、
      このトラックを伴奏の「新しい楽器」や「手で書いた小節」として取り込まない（2026-10-03・担当C の指摘）。
      主旋律を直すなら旋律の工程から作り直す。"""
    import tempfile

    import mido
    name = _lead_track()
    mid = mido.MidiFile(str(path))
    keep = [tr for tr in mid.tracks
            if not any(m.type == "track_name" and m.name == name for m in tr)]
    if len(keep) == len(mid.tracks):
        return path
    out = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
    out.tracks.extend(keep)
    p = Path(tempfile.mkdtemp()) / path.name
    out.save(str(p))
    return p


def diff_midi(expected: Path, actual: Path, bar_ticks: int) -> dict:
    """トラック（楽器）ごと・小節ごとの差。`expected` はその版の楽譜から作り直した MIDI。"""
    (exp, t0), (act, t1) = read_midi(expected), read_midi(actual)
    res: dict = {}
    if t0 != t1:
        res["tempo"] = (t0, t1)
    for name in sorted(set(exp) | set(act)):
        if name not in act:
            res[name] = {"track": "消えた"}
            continue
        if name not in exp:
            res[name] = {"track": "増えた", "notes": act[name]}
            continue
        # ★**数も比べる**（集合にしない）。同じ名前のトラック（区間ごとに分かれた同じ楽器）には
        #   同じ位置の同じ音が重なっていることがあり、集合だと1本から消しても差に出なかった
        from collections import Counter
        e, a = Counter(exp[name]), Counter(act[name])
        if e == a:
            continue
        bars: dict[int, dict] = {}
        for n in (e - a).elements():
            bars.setdefault(n[0] // bar_ticks, {"removed": [], "added": []})["removed"].append(n)
        for n in (a - e).elements():
            bars.setdefault(n[0] // bar_ticks, {"removed": [], "added": []})["added"].append(n)
        res[name] = {"bars": dict(sorted(bars.items()))}
    return res


# ---- 人へ見せる形 ------------------------------------------------------------------------

def where(tick: int, bar_ticks: int) -> str:
    return f"{tick // bar_ticks + 1}小節{(tick % bar_ticks) / Q + 1:g}拍"


def describe_vpr(d: dict[str, dict], bar_ticks: int, limit: int = 12) -> list[str]:
    lines = []
    for name, x in d.items():
        if "track" in x:
            lines.append(f"・歌『{name}』: トラックが{x['track']}")
            continue
        parts = []
        for n in x["changed"][:limit]:
            w = "・".join(f"{k} {v[0]}→{v[1]}" if k != "aiExp" else "表情" for k, v in n["what"].items())
            parts.append(f"{where(n['start'], bar_ticks)}「{n['lyric']}」{w}")
        if x["added"]:
            parts.append(f"足した音 {len(x['added'])}（最初は {where(x['added'][0]['start'], bar_ticks)}）")
        if x["removed"]:
            parts.append(f"消した音 {len(x['removed'])}（最初は {where(x['removed'][0]['start'], bar_ticks)}）")
        for k, (s, e) in (x.get("curves") or {}).items():
            parts.append(f"手で描いたカーブ {k}（{where(s, bar_ticks)}〜{where(e, bar_ticks)}）")
        if x.get("volume"):
            parts.append("トラックの音量")
        if x.get("style"):
            parts.append(f"声の種類 {x['style'][0]}→{x['style'][1]}")
        more = len(x["changed"]) - limit
        lines.append(f"・歌『{name}』: " + " ／ ".join(parts) + (f" ／ ほか {more} か所" if more > 0 else ""))
    return lines


def describe_midi(d: dict, bar_ticks: int, limit: int = 12) -> list[str]:
    lines = []
    if d.get("tempo"):
        lines.append(f"・テンポ {d['tempo'][0]}→{d['tempo'][1]}")
    for name, x in d.items():
        if name == "tempo":
            continue
        if "track" in x:
            lines.append(f"・伴奏『{name}』: トラックが{x['track']}")
            continue
        bs = list(x["bars"].items())
        head = "、".join(f"{b + 1}小節（足した{len(v['added'])}・消した{len(v['removed'])}）" for b, v in bs[:limit])
        lines.append(f"・伴奏『{name}』: {head}" + (f" ほか {len(bs) - limit} 小節" if len(bs) > limit else ""))
    return lines
