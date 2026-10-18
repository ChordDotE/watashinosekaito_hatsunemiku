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

"""人が手で直した vpr・MIDI を確かめる道具（2026-09-26・`docs/plan/manual_edit_import.md`）。"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from pydantic_ai import RunContext

from session import (SONGS, MusicSession, _append_log, _current_version, _trace, _traced,
                     _working_version, next_step, with_scope)
from coordinator import coordinator


def _expected_vpr(song_dir: Path, v: int, score) -> Path:
    """その版の楽譜から、システムが書いたはずの vpr を作り直す（一時フォルダ）。"""
    from score_to_vpr import score_to_vpr
    tmp = Path(tempfile.mkdtemp())
    for src, dst in ((f"mix_v{v}.json", "mix.json"), (f"accompaniment_v{v}.wav", "accompaniment.wav"),
                     (f"placeholder_v{v}.wav", "placeholder.wav"), (f"record_v{v}.json", f"record_v{v}.json")):
        if (song_dir / src).exists():
            shutil.copy2(song_dir / src, tmp / dst)
    return score_to_vpr(score, tmp / "vocal.vpr")


def _expected_mid(score) -> Path:
    from midi_out import score_to_midi
    return score_to_midi(score, Path(tempfile.mkdtemp()) / "accompaniment.mid", include_vocal=False)


def _form_skip(score, name: str) -> str:
    """その形の曲が持たない種類のファイルなら、突き合わせない理由（持つなら空・2026-10-03・§7-2）。
    アカペラは伴奏 MIDI を、インストは vpr を持たない（`song_files.FORM_FILES`）——前の形のファイルが残っていても読まない
    （以前の形のまま取り込むと、アカペラに無音の伴奏を作り、インストに歌の無い vpr を書いてしまう）。"""
    form = getattr(score, "form", "song") or "song"
    low = name.lower()
    if form == "acappella" and low.endswith((".mid", ".midi")):
        return "アカペラなので伴奏 MIDI は持たない——突き合わせない（前の形のファイル）"
    if form == "instrumental" and low.endswith(".vpr"):
        return "インストなので vpr（歌）は持たない——突き合わせない（前の形のファイル）"
    return ""


#: 主旋律の楽器のトラックを手で直していたときの一言（取り込まない理由と行き先）
_LEAD_NOT_TAKEN = ("・主旋律の楽器のトラック（main_melody）も手で直されている——これは楽譜へは取り込めない"
                   "（主旋律は旋律の工程の音そのもの。伴奏の楽器としては扱わない）。旋律を直すなら remake_from_step('旋律', 頼み)")


def _lead_vpr_skip(score) -> str:
    """主旋律を楽器でも重ねている曲（楽譜の `lead`）で、歌（vpr）の手直しを取り込まない理由。重ねていなければ空。
    歌の音符だけ取り込むと、楽器の主旋律が古い旋律のまま残る（黙って古いまま残さない・2026-10-03）。"""
    if score.lead.is_empty():                              # 主旋律の楽器が無い＝中身が空（F1。以前は None）
        return ""
    return ("主旋律を楽器でも重ねている曲なので、歌の音符の手直しを楽譜へ取り込むと楽器の主旋律が古いまま残る——取り込まない。"
            "旋律を直すなら remake_from_step('旋律', 頼み)")


def _bar_ticks(score) -> int:
    """楽譜の1小節の tick。拍子の読み方は `song_design.bar_ticks` の1つ（2026-10-01。以前はここで割り算を写していた）。"""
    from song_edits import score_bar_ticks
    return score_bar_ticks(score)


@coordinator.tool
@_traced
@with_scope
def review_manual_edits(ctx: RunContext[MusicSession], note: str = "") -> str:
    """**マスターが VOCALOID6 や MIDI の道具で手で直したものを確かめる**（読むだけ・曲は変えない）。

    **呼ぶのは、マスターが「V6 で直した」「MIDI をいじった」「このファイル見て」と言ったときだけ**。
    自分から見回らない。先に `load_existing_song` で曲を開いておく。
    `note` … マスターの言葉（なぜ直したか・どうしたかったか）。**意図として台帳に残る**。無ければ空でよい。

    返すもの: 変わったファイルと、**何小節目の何がどう変わったか**（歌は音の高さ・歌詞・長さ・表情・手で描いたカーブ、
    伴奏は楽器ごと・小節ごとの足した音と消した音）。版は名前から見分ける（`vocal_v6_手直し.vpr` → v6）。
    名前から分からないときは、推定した版を添えて返すので、**マスターに確かめる**。
    """
    sid = ctx.deps.song_id
    song_dir = SONGS / sid if sid else None
    if not song_dir or not song_dir.exists():
        return f"曲が未ロード。先に load_existing_song を。（song_id={sid!r}）"
    if ctx.deps.score is None:                             # 台帳を書くので、曲を開いて作業台を用意してから
        return "楽譜が読み込まれていない。先に load_existing_song を。"
    from manual_diff import (changed_files, describe_midi, describe_vpr, diff_midi, diff_vpr, lead_changed,
                             version_of, without_lead)
    from song_schema import SongScore
    cur = _current_version(song_dir) or _working_version(song_dir)
    files, had_manifest = changed_files(song_dir)
    if not files:
        return ("手で直した形跡は見つからなかった（前回この曲を仕上げたときから、vpr も MIDI も変わっていない）。"
                "マスターに、どのファイルを直したか・別の場所に保存していないかを聞く。")
    lines, guessed, counts = [], [], {"vpr": 0, "midi": 0}
    for f in files:
        name, how = f["name"], f["how"]
        if how == "消えた":
            lines.append(f"■ {name}: 消えている（前回はあった）")
            continue
        if why := _form_skip(ctx.deps.score, name):
            lines.append(f"■ {name}（{how}）: {why}")
            continue
        v, named = version_of(name, cur)
        if not named:
            guessed.append(f"{name} → v{v}（名前に版番号が無いので、いまの版とみなした）")
        sp = song_dir / f"score_v{v}.json"
        if not sp.exists():
            lines.append(f"■ {name}（{how}）: 元にした版 v{v} の楽譜が無いので、突き合わせられない")
            continue
        score = SongScore.model_validate_json(sp.read_text(encoding="utf-8"))
        bar = _bar_ticks(score)
        try:
            if name.lower().endswith(".vpr"):
                d = diff_vpr(_expected_vpr(song_dir, v, score), song_dir / name)
                got = describe_vpr(d, bar)
                counts["vpr"] += sum(len(x.get("changed", [])) + len(x.get("added", [])) + len(x.get("removed", []))
                                     for x in d.values())
            else:
                if ctx.deps.faithful:
                    lines.append(f"■ {name}（{how}）: カバーの伴奏 MIDI は原曲の採譜そのものなので、ここでは突き合わせない")
                    continue
                # 主旋律の楽器のトラック（`midi_out.LEAD_TRACK`）は伴奏ではない——伴奏の差から外し、別に知らせる
                exp_mid = _expected_mid(score)
                d = diff_midi(without_lead(exp_mid), without_lead(song_dir / name), bar)
                got = describe_midi(d, bar) + ([_LEAD_NOT_TAKEN] if lead_changed(exp_mid, song_dir / name) else [])
                counts["midi"] += sum(len(b["added"]) + len(b["removed"]) for x in d.values()
                                      if isinstance(x, dict) for b in (x.get("bars") or {}).values())
        except Exception as e:                             # 読めない形式のファイル
            lines.append(f"■ {name}（{how}）: 読めなかった（{type(e).__name__}: {e}）")
            continue
        lines.append(f"■ {name}（{how}・v{v} の楽譜と比べた）" + ("\n" + "\n".join(got) if got else ": 楽譜から作るものと同じ（中身は変わっていない）"))

    body = "\n".join(lines)
    _append_log(sid, "=== 手の直しの確認 ===\n" + (f"マスターの言葉: {note}\n" if note else "") + body)
    _trace(sid, "review_manual_edits.out", files=len(files), vpr=counts["vpr"], midi=counts["midi"],
           manifest=had_manifest)
    from tools.editing import _ledger_note, _note_request
    _rid = _note_request(ctx, (note.strip() or "（理由は聞いていない）") + " ／ 手で直したファイル: "
                         + "・".join(f["name"] for f in files), "手の直し", op="set",
                         value={"files": [f["name"] for f in files], "vpr_changes": counts["vpr"],
                                "midi_changes": counts["midi"]})
    return ("手で直した所（楽譜から作るものと比べた差）:\n" + body + _ledger_note(_rid)
            + ("\n★版の見分けを推定した: " + " ／ ".join(guessed) + "——違っていたらマスターに確かめる" if guessed else "")
            + ("" if had_manifest else "\n※この曲は前回の控えが無いので、vpr と MIDI を全部突き合わせた")
            + "\n台帳に「マスターが手で直したこと」として残した。**まだ楽譜には取り込んでいない**"
              "（このまま作り直したり書き出したりすると、手の直しは上書きされる）。"
              "マスターが「その直しで行く」なら `take_in_manual_edits` で楽譜に取り込む。")


@coordinator.tool
@_traced
def take_in_manual_edits(ctx: RunContext[MusicSession], files: str = "", note: str = "") -> str:
    """**マスターが手で直した vpr・MIDI を楽譜に取り込む**（以後の書き出し・作り直しでも消えない）。

    先に `review_manual_edits` で何が変わったかを見て、**マスターが「その直しで行く」と言ったとき**に呼ぶ。
    `files` … 取り込むファイル名（カンマ区切り）。空なら変わったファイル全部。
    **同じ版の vpr が2つ以上変わっているとき（`vocal_v6.vpr` と `vocal_v6_手直し.vpr` など）は、どれかをマスターに聞いて指す。**
    `note` … マスターの言葉（なぜ直したか）。台帳に意図として残る。

    取り込めるのは**いま開いている版**の直しだけ（前の版の直しは、その版を `load_existing_song` で開いてから）。
    ・歌: 音の高さ・位置・長さ・歌詞・音ごとの表情・声の種類は楽譜の欄へ、手で描いたカーブ・発音記号・トラックの音量は
      楽譜の「手の直し」の欄へ。足した・消した声部も取り込む
    ・伴奏: 手で直した小節は「手で書いた小節」として楽器ごとに持つ（パターンより優先）。消した楽器・足した楽器・音色・テンポも取り込む
    取り込んだあと書き出し直し、**人が直したファイルと同じになるかを確かめて**返す。
    """
    sid = ctx.deps.song_id
    song_dir = SONGS / sid if sid else None
    if not song_dir or not song_dir.exists():
        return f"曲が未ロード。先に load_existing_song を。（song_id={sid!r}）"
    if ctx.deps.score is None:
        return "楽譜が読み込まれていない。先に load_existing_song を。"
    from manual_diff import changed_files, diff_midi, diff_vpr, lead_changed, version_of, without_lead
    from manual_import import bars_label, take_in_midi, take_in_vpr
    from song_edits import ASSEMBLY_TEMPO, ASSEMBLY_VOICE, NOTES, notes_value, span_text
    cur = _current_version(song_dir) or _working_version(song_dir)
    want = [f.strip() for f in files.split(",") if f.strip()]
    got = [f for f in changed_files(song_dir)[0] if f["how"] != "消えた" and (not want or f["name"] in want)]
    if want and len(got) < len(want):
        missing = set(want) - {f["name"] for f in got}
        return f"{' / '.join(sorted(missing))} は、前回から変わったファイルに無い（`review_manual_edits` で一覧を見る）。"
    if not got:
        return "取り込む手の直しが無い（前回この曲を仕上げたときから、vpr も MIDI も変わっていない）。"
    def _skip(name: str) -> str:
        return _form_skip(ctx.deps.score, name) or (_lead_vpr_skip(ctx.deps.score) if name.lower().endswith(".vpr") else "")
    skipped = [f"{f['name']}（{why}）" for f in got if (why := _skip(f["name"]))]
    got = [f for f in got if not _skip(f["name"])]
    if not got:
        return "取り込まなかった: " + " / ".join(skipped) + "。"
    vprs = [f["name"] for f in got if f["name"].lower().endswith(".vpr")]
    mids = [f["name"] for f in got if f["name"].lower().endswith((".mid", ".midi"))]
    if len(vprs) > 1 or len(mids) > 1:
        return ("同じ種類のファイルが2つ以上変わっている（" + " / ".join(vprs + mids) + "）。"
                "どれを取り込むかマスターに聞いて、`files` に指す。")
    other = [n for n in vprs + mids if version_of(n, cur)[0] != cur]
    if other:
        return (f"{' / '.join(other)} は v{version_of(other[0], cur)[0]} の直しで、いま開いているのは v{cur}。"
                "取り込むなら、その版を `load_existing_song` で開いてから。")

    s = ctx.deps.score
    bar = _bar_ticks(s)
    said: list[str] = []
    rows: list[tuple] = []                                 # 台帳へ書く (言葉, 対象, op, scope, value)
    if vprs:
        exp = _expected_vpr(song_dir, cur, s)
        s, got_said, meta = take_in_vpr(s, exp, song_dir / vprs[0], bar)
        said += got_said
        for a, b, notes in meta["regions"]:
            rows.append((f"V6 で手で直した音符（{bars_label(a // bar, (b - 1) // bar + 1)}）", NOTES, "set",
                         span_text(a, b, bar), notes_value(a, b, notes)))   # 範囲は `小節:拍`（表と同じ）
        for vt in meta["voices"]:
            rows.append((f"V6 で手で直した声部『{vt.title}』", ASSEMBLY_VOICE, "set", vt.title,
                         {"track": vt.model_dump(), "replaces": ""}))
        for t in meta["removed"]:
            rows.append((f"V6 で声部『{t}』を消した", ASSEMBLY_VOICE, "remove", t, {"title": t}))
    if mids and not ctx.deps.faithful:
        # 主旋律の楽器のトラックは伴奏の楽器として取り込まない（両方から外して突き合わせる・2026-10-03）
        _exp = _expected_mid(ctx.deps.score)
        s, got_said, meta = take_in_midi(s, without_lead(_exp), without_lead(song_dir / mids[0]), bar)
        said += got_said
        if lead_changed(_exp, song_dir / mids[0]):
            said.append(_LEAD_NOT_TAKEN.lstrip("・"))
        if meta.get("tempo"):
            rows.append((f"MIDI で速さを {meta['tempo']:g} に", ASSEMBLY_TEMPO, "set", "", {"tempo": float(meta["tempo"])}))
    elif mids:
        said.append("カバーの伴奏 MIDI は原曲の採譜そのものなので取り込まなかった")
    if not said:
        return "楽譜から作るものと同じだった（取り込むものが無い）。"

    # 書き出し直す（伴奏 → vpr の順。vpr は伴奏の wav を同梱する）
    from build_lipsync import write_lipsync
    from score_to_vpr import score_to_vpr
    from song_pipeline import _render_accomp, render_placeholder_vocal
    ctx.deps.score = s
    if mids and not ctx.deps.faithful:
        _render_accomp(s, song_dir / "accompaniment.wav")
    if getattr(s, "form", "song") == "instrumental":       # インストは vpr・仮歌・口の動きを持たない（§4-4）
        (song_dir / "score.json").write_text(s.model_dump_json(indent=2), encoding="utf-8")
    else:
        score_to_vpr(s, song_dir / "vocal.vpr")
        (song_dir / "score.json").write_text(s.model_dump_json(indent=2), encoding="utf-8")
        render_placeholder_vocal(s, song_dir / "placeholder.wav")
        write_lipsync(s, song_dir)

    # 往復の確認: 取り込んだ楽譜から書いたものと、人が直したファイルが同じになるか
    rest = []
    try:
        if vprs:
            d = diff_vpr(song_dir / "vocal.vpr", song_dir / vprs[0])
            if d:
                rest.append(f"{vprs[0]}: " + "・".join(d))
        if mids and not ctx.deps.faithful:
            mid = song_dir / "accompaniment.mid"
            d = diff_midi(without_lead(mid if mid.exists() else _expected_mid(s)), without_lead(song_dir / mids[0]), bar)
            if d:
                rest.append(f"{mids[0]}: " + "・".join(d))
    except Exception as e:
        rest.append(f"確かめられなかった（{type(e).__name__}: {e}）")

    from tools.editing import _ledger_note, _note_request
    words = note.strip() or "（理由は聞いていない）"
    _ids = [_note_request(ctx, f"{w} ／ マスターの言葉: {words}", target, op=op, scope=scope, value=value)
            for w, target, op, scope, value in rows]
    _ids.append(_note_request(ctx, f"{words} ／ 手で直したファイルを楽譜に取り込んだ: " + "・".join(vprs + mids), "手の直し",
                              value={"files": vprs + mids, "taken_in": True, "hand_edits": len(s.hand_edits),
                                     "hand_bars": len(s.arrangement.hand_bars)}))
    body = "\n".join(f"・{x}" for x in said)
    _append_log(sid, "=== 手の直しの取り込み ===\n" + (f"マスターの言葉: {note}\n" if note else "") + body
                + ("\n往復の残り: " + " / ".join(rest) if rest else "\n往復: 人が直したファイルと同じになった"))
    _trace(sid, "take_in_manual_edits.out", files=len(vprs + mids), said=len(said), rest=len(rest))
    return ("手の直しを楽譜に取り込んだ:\n" + body
            + ("\n取り込まなかったファイル: " + " / ".join(skipped) if skipped else "")
            + ("\n★書き出し直したものと人が直したファイルに、まだ差がある（取り込めなかった所）: " + " / ".join(rest)
               + "——マスターに伝える" if rest else "\n書き出し直したものは、人が直したファイルと同じになった。")
            + "\n台帳に残したので、あとで工程から作り直しても、下の旋律・伴奏が変わらなければ引き継ぐ。"
            + f"\n{next_step(getattr(s, 'form', 'song'))}"   # 次にやることは形で分ける（インストは render_final・2026-10-03・再レビュー）
            + _ledger_note(*_ids))
