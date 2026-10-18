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

"""**作り直したあとに、手で直したことを当て直す**（2026-09-25・段階3・`docs/plan/song_revision_design.md` §6.5・§10-2）。

## なぜ要るか

作り直しは工程の記録から楽譜を組み立て直すので、**楽譜だけに当てた手直しは消える**。
キーを +2 してから詞を直すと、キーが元に戻る。足した声部も、置き直した音符も消える。
段階2で頼まれたことは台帳に残るようにしたが、**速さ・移調・声部・音符は LLM の工程ではない**ので、
台帳に書いても誰も当てていなかった。

## 考え方

- **曲全体にかかるもの**（速さ・移調）→ そのまま当てる
- **声部の手直し**（消す・移調・音量）→ 同じ名前の声部があれば当てる。消す頼みは、無ければ何もしない
- **旋律に乗っているもの**（足した声部・置き直した音符）→ 旋律と詞が変わっていなければ当てる。
  **変わっていたら当てずに「当て直し待ち」にする**——新しい旋律のどこに当てるかは
  **コーディネーターが新しい楽譜を見て決める**（本人判断 §10-2。機械が同じ拍へ置くと、
  別の言葉・別の音の上に前の音符を貼ることになる）
- 当てる順は**頼まれた順**。移調のあとで置き直した音符は、移調後の高さで記録されている
"""
from __future__ import annotations

from song_design import bar_ticks, transpose_chord
from song_schema import SongNote, SongScore

Qd = 480
#: 台帳の宛先のうち、この段で機械が当てるもの
ASSEMBLY_TEMPO = "組み立て.速さ"
ASSEMBLY_TRANSPOSE = "組み立て.移調"
ASSEMBLY_VOICE = "組み立て.声部"
NOTES = "音符"


# ---- 道具の位置 `小節:拍`（2026-10-01・記法の仕様書 §15-2・ず-18）-------------------------------
#
# ★**コーディネーターが道具に書く位置・道具が見せる位置は、曲の表と同じ `小節:拍`**（小節は楽譜の頭から1始まり・
#   拍は4分音符＝1.00の0始まり・小数2桁）。以前は直す道具（`replace_region`・`inspect_region`・伴奏の道具ほか）だけが
#   楽譜の頭からの拍（`start_beat=21.5`）で、`inspect_song` の表で `6:1.50` と名指した所を、LLM が拍へ換算して書いていた。
#   書き手は表と同じ `artic_plan.tick_key`、読み手は声部と同じ `harmony.at_to_tick`＋記法の型 `song_checks.position_notation`。

def score_bar_ticks(score) -> int:
    """楽譜の1小節の tick（曲の拍子から・`song_design.bar_ticks`）。"""
    return bar_ticks(getattr(score, "time_signature", None) or "4/4")


def at_text(tick: float, bar: int) -> str:
    """tick → 道具が見せる位置 `小節:拍`（表と同じ書き方）。曲頭より前は `曲頭の◯拍前`（書けない位置なのでそう言う）。"""
    from artic_plan import tick_key
    t = int(round(tick))
    return tick_key(t, bar) if t >= 0 else f"曲頭の{-t / Qd:.2f}拍前"


def at_tick(at: str, bar: int) -> int:
    """道具に書かれた位置 `小節:拍` → tick。読めない・拍が1小節の長さ以上なら `ValueError`（所見の文）。"""
    from harmony import at_to_tick
    from song_checks import position_notation
    if bad := position_notation(bar / Qd).problems(at):
        raise ValueError(bad[0])
    return at_to_tick(at, bar)


def span_ticks(start: str, end: str, bar: int) -> tuple[int, int]:
    """道具の区間 `[start, end)`（どちらも `小節:拍`）→ (開始 tick, 終わり tick)。読めない・逆向きなら `ValueError`。"""
    a, b = at_tick(start, bar), at_tick(end, bar)
    if b <= a:
        raise ValueError(f"区間の終わり {end} が始まり {start} より前か同じ（終わりは含まない。"
                         f"小節の終わりまでなら次の小節の頭 `◯:0.00` と書く）")
    return a, b


def span_text(a: float, b: float, bar: int) -> str:
    """tick の区間 → `小節:拍〜小節:拍`（終わりは含まない）。台帳の範囲・道具の返事に使う。"""
    return f"{at_text(a, bar)}〜{at_text(b, bar)}"


# ---- 見せる歌詞（lyrics_display）の行の手直し（2026-10-12・本人了承）----------------------------
# 手直しの道具が音符（歌う詞）だけを書き換え、見せる歌詞（字幕・投稿用の漢字まじりの詞）が古いまま残っていた
# （『てのひらのシグナル』v16 は v7 の詞のまま）。道具が**歌う詞と見せる歌詞を同じ場面で書き換える**ために、
# 見せる歌詞の行（1行＝詞の1フレーズ）に番号を振り、手直しの区間にかかる行を見せる。

#: 見せる歌詞の1行に残してよい長さ（字）。長すぎる行は取り違えの疑い
DISPLAY_LINE_MAX = 60
_KANA_FOLD = str.maketrans({"を": "お", "は": "わ", "へ": "え"})


def _fold(kana: str) -> str:
    """歌うかなと見せる歌詞のかなを照らすための正規化（助詞の発音・カタカナ・伸ばす印）。"""
    out = []
    for ch in kana:
        if "ァ" <= ch <= "ヶ":
            ch = chr(ord(ch) - 0x60)                       # カタカナ → ひらがな
        out.append(ch.translate(_KANA_FOLD))
    return "".join(out)


def sung_kana_of(notes) -> str:
    """音符の歌うかな（伸ばし `-` と `_`・記号は除く）。"""
    return "".join(n.lyric for n in notes if getattr(n, "lyric", "") and n.lyric not in ("-", "_"))


def display_line_spans(song, score) -> list[tuple[int, int] | None]:
    """見せる歌詞の各行（1始まりの順）が、いまの楽譜のどの tick の範囲か（分からなければ None）。

    詞の工程の記録（`song.lyrics.phrases`＝行ごとの歌うかな）を、いまの音符のかなと突き合わせて探す（difflib）。
    手直しで音符のかなが変わっていても、残っている部分で位置を取れる。記録が無い・行の数が見せる歌詞と合わない曲は空の対応。"""
    import difflib

    from lyric_slots import sung_kana
    disp = [ln for ln in (getattr(score, "lyrics_display", "") or "").split("\n")]
    ph = sorted(list(getattr(getattr(song, "lyrics", None), "phrases", None) or []), key=lambda p: getattr(p, "index", 0))
    if not disp or len(ph) != len(disp):
        return [None] * len(disp)
    notes = sorted(getattr(score, "vocal_notes", None) or [], key=lambda n: n.start_tick)
    b_chars, b_pos = [], []                                # 音符のかな（1文字ずつ）と、その音の (開始, 終わり)
    for n in notes:
        if n.lyric and n.lyric not in ("-", "_"):
            for ch in _fold(n.lyric):
                b_chars.append(ch)
                b_pos.append((n.start_tick, n.start_tick + n.duration_tick))
    a_chars, a_line = [], []
    for i, p in enumerate(ph):
        for ch in _fold(sung_kana(getattr(p, "text", "") or "")):
            a_chars.append(ch)
            a_line.append(i)
    spans: list[list[int] | None] = [None] * len(disp)
    sm = difflib.SequenceMatcher(None, a_chars, b_chars, autojunk=False)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            line, (t0, t1) = a_line[blk.a + k], b_pos[blk.b + k]
            cur = spans[line]
            spans[line] = [t0, t1] if cur is None else [min(cur[0], t0), max(cur[1], t1)]
    return [tuple(x) if x else None for x in spans]


def numbered_display(song, score, bar: int, *, near: tuple[int, int] | None = None) -> str:
    """見せる歌詞に行番号と位置（`小節:拍`）を付けて見せる。`near` の区間にかかる行には ◀ を付ける。"""
    disp = (getattr(score, "lyrics_display", "") or "").split("\n")
    spans = display_line_spans(song, score)
    rows = []
    for i, text in enumerate(disp, 1):
        sp = spans[i - 1] if i - 1 < len(spans) else None
        where = f"（{at_text(sp[0], bar)}〜{at_text(sp[1], bar)}）" if sp else "（位置は特定できない）"
        hit = " ◀この区間にかかる" if near and sp and sp[0] < near[1] and near[0] < sp[1] else ""
        rows.append(f"  {i}: {text} {where}{hit}")
    return "\n".join(rows)


def display_problems(edits: dict[int, str], score, new_notes) -> list[str]:
    """渡された見せる歌詞の行の所見（空なら通す）。行の番号の範囲・空・長さ・改行。かなの読みは、同じ区間の歌うかなと照らして警告だけ。"""
    n_lines = len((getattr(score, "lyrics_display", "") or "").split("\n"))
    out = []
    for line, text in edits.items():
        t = (text or "").strip()
        if not 1 <= line <= n_lines:
            out.append(f"行 {line} は見せる歌詞に無い（1〜{n_lines} 行）")
        elif not t:
            out.append(f"行 {line} が空（見せる詞は書いたとおり。歌わない行でも空にしない）")
        elif "\n" in t or len(t) > DISPLAY_LINE_MAX:
            out.append(f"行 {line}: 1行は改行なし・{DISPLAY_LINE_MAX}字以内で書く")
    return out


def stale_display_lines(song, score) -> list[int]:
    """**見せる歌詞が、実際に歌う詞と食い違っていそうな行**（1始まり・2026-10-14）。

    詞の工程の記録の行ごとの歌うかなと、いまの楽譜のその行の位置の歌うかなを比べ、違う行（位置が取れない行も）を返す。
    あとから見せる歌詞を書き直した行（台帳の値 `display` にある行）は除く。記録が無い・行の数が合わない曲は空。
    ★『てのひらのシグナル』の Aメロ4行は v18 で「字幕を揃えて」と頼まれたのに v27 まで古いままだった。"""
    from lyric_slots import sung_kana
    disp = (getattr(score, "lyrics_display", "") or "").split("\n")
    ph = sorted(list(getattr(getattr(song, "lyrics", None), "phrases", None) or []), key=lambda p: getattr(p, "index", 0))
    if not disp or len(ph) != len(disp):
        return []
    edited = {int(k) for r in (getattr(song, "requests", None) or []) for k in ((getattr(r, "value", None) or {}).get("display") or {})}
    notes = sorted(getattr(score, "vocal_notes", None) or [], key=lambda n: n.start_tick)
    out = []
    for i, sp in enumerate(display_line_spans(song, score), 1):
        if i in edited:
            continue
        want = _fold(sung_kana(getattr(ph[i - 1], "text", "") or ""))
        now = _fold(sung_kana_of([n for n in notes if sp and sp[0] <= n.start_tick < sp[1]]))
        if sp is None or want.replace("っ", "") != now.replace("っ", ""):
            out.append(i)
    return out


def apply_display_edits(score, edits: dict[int, str]):
    """見せる歌詞の指定の行を差し替えた `lyrics_display` を返す（行の数は変えない）。"""
    lines = (getattr(score, "lyrics_display", "") or "").split("\n")
    for line, text in edits.items():
        if 1 <= line <= len(lines):
            lines[line - 1] = " ".join((text or "").split())
    return "\n".join(lines)


# ---- 楽譜の操作（編集の道具と作り直しの両方が使う）--------------------------------------

def transpose_score(s: SongScore, semitones: int) -> SongScore:
    """歌・追加声部・**コード**をまとめて移調する。

    ★コードを一緒に動かさないと、歌だけが調を変えて伴奏から離れる（2026-09-07 の事故。
      編曲パートはコードからの度数で書かれているので、コードを動かせば伴奏は追随する）。
    ★**音の高さを直に持つ伴奏の音（`Tone.key`）も動かす**（2026-09-25）。彩りの対旋律・リフは
      コードの度数ではなく絶対音高で書かれている（2026-09-17〜）ので、コードを動かしても追随しない。
      移調のたびに、彩りだけ元のキーで鳴っていた。
    ★**読めないコード名があれば `ValueError`**（2026-09-27・監査）。そのコードだけ元の調で鳴るのを黙って通さない。
      呼ぶ前に `untransposable` で確かめて断る・所見にすること。
    ★**移すと MIDI の 0〜127 からはみ出す音があれば `ValueError`**（2026-10-03・再レビュー）。以前はカバーの伴奏 MIDI
      （`shift_midi_problem`）しか確かめておらず、歌・声部ははみ出したまま保存され、次の読み込みで型の検証に落ちて曲が開けなくなった
      （伴奏の `Tone.key`・主旋律の楽器ははみ出した音だけ黙って鳴らなかった）。呼ぶ前に `transpose_problem` で確かめて断ること。
    """
    k = int(semitones)
    if not k:
        return s
    bad = untransposable(s)
    if bad:
        raise ValueError("移調できない: " + " / ".join(bad))
    if why := transpose_problem(s, k):
        raise ValueError(why)
    upd = {
        "vocal_notes": [n.model_copy(update={"key": n.key + k}) for n in s.vocal_notes],
        "extra_voices": [v.model_copy(update={"notes": [n.model_copy(update={"key": n.key + k})
                                                        for n in v.notes]}) for v in s.extra_voices],
        "chords": [c.model_copy(update={"chord": transpose_chord(c.chord, k)}) for c in s.chords],
    }
    if not s.arrangement.is_empty():
        upd["arrangement"] = s.arrangement.model_copy(update={"parts": [
            p.model_copy(update={"steps": [
                e.model_copy(update={"tones": [t.model_copy(update={"key": t.key + k})
                                               if t.key is not None else t for t in e.tones]})
                for e in p.steps]})
            for p in s.arrangement.parts]})
    # ★**主旋律の楽器も動かす**（2026-10-03・`SongScore.lead`）。高さを直に持つので、コードを動かしても追随しない
    if not s.lead.is_empty():                               # 無い主旋律の楽器は中身が空のまま（F1）
        upd["lead"] = s.lead.model_copy(update={"notes": [n.model_copy(update={"key": n.key + k}) for n in s.lead.notes]})
    return s.model_copy(update=upd)


#: MIDI で鳴らせる高さ（ノート番号）。楽譜の歌・声部・主旋律の楽器は型（`SongNote.key`・`LeadNote.key`）もこの範囲
MIDI_LO, MIDI_HI = 0, 127


def transpose_problem(s: SongScore, semitones: int, *, voice: str | None = None) -> str | None:
    """**移調すると MIDI の 0〜127 からはみ出す音があるか**（2026-10-03・再レビュー）。はみ出すなら理由の文、無ければ None。

    見るのは高さを直に持つ全トラック: 歌（`vocal_notes`）・追加声部（`extra_voices`）・伴奏の高さを書いた音（`Tone.key`＝
    彩りの対旋律・リフ）・主旋律の楽器（`lead`）。度数で書いた伴奏の音はコードに付いて動くので見ない。
    カバーの伴奏 MIDI は楽譜の外なので `shift_midi_problem` が見る。
    `voice` を渡すと、その名前の声部だけを見る（`adjust_voice` の1声部の移調）。

    呼び手（`adjust_song`・`adjust_voice`・作り直しの当て直し `apply_edits`）は、はみ出すなら**何も変えずに**理由を返す
    （端に張り付けない・はみ出した音だけ外さない＝黙って変えない）。
    """
    k = int(semitones or 0)
    if not k:
        return None
    tracks: list[tuple[str, list[int]]] = []
    if voice is None:
        tracks.append(("歌", [n.key for n in s.vocal_notes]))
    tracks += [(f"声部『{v.title}』", [n.key for n in v.notes]) for v in s.extra_voices
               if voice is None or v.title == voice]
    if voice is None:
        tracks += [(f"伴奏のパート{i}（{p.role}）の高さを書いた音",
                    [int(t.key) for e in p.steps for t in e.tones if t.key is not None])
                   for i, p in enumerate(s.arrangement.parts, 1)]
    if voice is None and not s.lead.is_empty():
        tracks.append(("主旋律の楽器", [n.key for n in s.lead.notes]))
    bad = []
    for name, keys in tracks:
        if not keys:
            continue
        lo, hi = min(keys) + k, max(keys) + k
        if lo < MIDI_LO or hi > MIDI_HI:
            bad.append(f"{name}（いちばん低い音が {lo}・高い音が {hi} になる）")
    if not bad:
        return None
    return (f"{k:+d} 半音動かすと MIDI で鳴らせる高さ（{MIDI_LO}〜{MIDI_HI}）からはみ出す音がある: " + "／".join(bad))


def shift_midi(src, dst, *, semitones: int = 0, tempo_from: float = 0.0, tempo_to: float = 0.0) -> int:
    """**伴奏 MIDI を移調・テンポ変更する**（2026-09-27・目録の道具-4）。書き換えた音の数を返す。

    カバーの伴奏は楽譜（`arrangement`）から作られていない（原曲の MIDI・YourMT3 の採譜）。
    `adjust_song` はそれを楽譜から鳴らし直していたので、**移調・テンポ変更で伴奏が空になっていた**。
    楽譜の代わりに伴奏 MIDI そのものを動かす: 打楽器（ch10）以外の音を `semitones` 動かし、
    テンポの印を `tempo_from → tempo_to` の比で置き換える（拍の位置は変えない）。
    ★**MIDI の 0〜127 からはみ出す音があれば `ValueError`**（2026-10-02・洗い出し 2-3）。以前は端に張り付けて、
      その音だけ別の高さで鳴らしていた。呼ぶ前に `shift_midi_problem` で確かめて断ること。
    """
    import mido
    if why := shift_midi_problem(src, semitones):
        raise ValueError(why)
    mf = mido.MidiFile(str(src))
    k = int(semitones or 0)
    ratio = (float(tempo_from) / float(tempo_to)) if tempo_from and tempo_to else 1.0
    n = 0
    for tr in mf.tracks:
        for i, m in enumerate(tr):
            if m.type in ("note_on", "note_off") and k and m.channel != 9:
                tr[i] = m.copy(note=m.note + k)
                n += 1
            elif m.type == "set_tempo" and ratio != 1.0:
                tr[i] = m.copy(tempo=max(1, int(round(m.tempo * ratio))))
    if ratio != 1.0 and mf.tracks and not any(m.type == "set_tempo" for tr in mf.tracks for m in tr):
        mf.tracks[0].insert(0, mido.MetaMessage("set_tempo", tempo=int(round(500000 * ratio)), time=0))  # 印が無い＝120
    mf.save(str(dst))
    return n


def shift_midi_problem(src, semitones: int) -> str:
    """伴奏 MIDI を `semitones` 移調すると MIDI の 0〜127 からはみ出す音があるか（打楽器は動かさないので見ない）。無ければ空。"""
    import mido
    k = int(semitones or 0)
    if not k:
        return ""
    keys = [m.note for tr in mido.MidiFile(str(src)).tracks for m in tr
            if m.type == "note_on" and m.velocity > 0 and m.channel != 9]
    if not keys:
        return ""
    lo, hi = min(keys) + k, max(keys) + k
    if lo >= 0 and hi <= 127:
        return ""
    return (f"伴奏 MIDI を {k:+d} 半音動かすと MIDI で鳴らせる高さ（0〜127）からはみ出す音がある"
            f"（いちばん低い音が {lo}・高い音が {hi} になる）")


def patch_region(notes: list[SongNote], start_tick: int, end_tick: int,
                 new: list[SongNote], trimmed: list[str] | None = None,
                 bar: int = 4 * Qd) -> tuple[list[SongNote], int]:
    """[start_tick, end_tick) の音符を `new` で置き換える。戻り値は (並べ直した音符, 消した数)。

    ★重なりは次の音の開始までに切る（歌は単旋律）。**区間の外の音を切ったときは `trimmed` に書く**
      （2026-09-27・目録の道具-5）——頼まれた範囲の外を黙って変えない。呼び側が返事に添える。
      位置は `小節:拍` で書く（`bar`＝曲の1小節の tick。`trimmed` を渡すときは曲の拍子から渡す）。
    """
    kept = [n for n in notes if not (start_tick <= n.start_tick < end_tick)]
    new_ids = {id(n) for n in new}
    merged = sorted(kept + list(new), key=lambda n: n.start_tick)
    for i in range(len(merged) - 1):                       # 重なり防止: 各音を次音開始までにクリップ
        ov = merged[i].start_tick + merged[i].duration_tick - merged[i + 1].start_tick
        if ov > 0:
            if trimmed is not None and id(merged[i]) not in new_ids:
                trimmed.append(f"範囲の外の「{merged[i].lyric}」（{at_text(merged[i].start_tick, bar)}）を"
                               f"{ov / Qd:g}拍短くした（次の音と重なるため）")
            merged[i] = merged[i].model_copy(
                update={"duration_tick": max(Qd // 8, merged[i + 1].start_tick - merged[i].start_tick)})
    return merged, len(notes) - len(kept)


def region_notes(old: list[SongNote], start_tick: int, end_tick: int,
                 written: list, bar: int) -> tuple[list[SongNote], list[str]]:
    """`replace_region` に書かれた音（`lyric` / `key` / `at`＝`小節:拍` / `beats`）→ 楽譜の音符。`bar` は曲の1小節の tick。
    戻り値は (音符, 歌い方・崩しが付いていない音の一覧（返事に書く文）)。
    位置が読めない・拍が1小節の長さ以上なら `ValueError`（呼び側が書き換えずに返す）。
    言葉は声部と同じ1モーラ（伸ばす印 `_` は楽譜の伸ばす歌詞 `-` に訳す・`harmony._V6_WORD`）。

    ★**元の音と同じ開始位置の音には、元の音の歌い方（`tuning`）・崩し（`artic` / `artic_depth`）を引き継ぐ**
      （2026-09-27・監査）。以前は歌詞・高さ・位置・長さだけで作り直し、歌い方の工程と歌の崩しの工程が
      その位置に決めたことを**黙って捨てていた**。位置で決まる決定を保つのは組み立て（書き換えではない）。
    ★**開始位置が一致しない音には付けない**（どの決定を当てるかは機械には決められない）。
      曲のどこかに歌い方・崩しがあるときだけ、その音を一覧に出す（呼び側が返事に書く＝コーディネーターが知る）。
    ★**書いた音の形が区間の置き換えにならないものは `ValueError`（書き換えずに返す・2026-10-02・洗い出し H11・H12）**:
      ①区間 `[start, end)` の外の `at`（以前はそのまま置き、外の元の音は消えないので同じ位置に2音になった）
      ②書いた音どうしの重なり・同じ位置の2音（歌は単旋律。以前は前の音を次の音の頭まで黙って切った）
      ③書いた音が区間の外の次の元の音に重なる（以前は黙って切った）。長さは書いたとおり（1/8拍未満を伸ばすのはやめた・
      短すぎて tick にすると 0 の長さは型の検証 `harmony.beats_problem` が差し戻す）。並びは開始位置の順にする（中身は変えない）。
    """
    from harmony import _V6_WORD, beats_to_ticks          # 言葉の訳し方・長さの読み方は声部と同じ1つ
    by_tick = {n.start_tick: n for n in old if start_tick <= n.start_tick < end_tick}
    has_expr = any(n.tuning is not None or n.artic for n in old)
    out: list[SongNote] = []
    bare: list[str] = []
    bad: list[str] = []
    for rn in written:
        t = at_tick(rn.at, bar)
        if not (start_tick <= t < end_tick):
            bad.append(f"『{rn.lyric}』の位置 {rn.at} が区間 {span_text(start_tick, end_tick, bar)}（終わりは含まない）の外")
            continue
        src = by_tick.get(t)
        carry = ({"tuning": src.tuning.model_copy() if src.tuning is not None else None,
                  "artic": src.artic, "artic_depth": src.artic_depth} if src is not None else {})
        # ★高さは型の検証（`song_llm.PITCH`）が鳴らせる範囲を保証している——ここで別の高さへ寄せない（以前は範囲外を 67 にしていた）
        out.append(SongNote(lyric=_V6_WORD.get(rn.lyric.strip(), rn.lyric.strip()), key=rn.key,
                            start_tick=t, duration_tick=beats_to_ticks(rn.beats),
                            **carry))
        if src is None and has_expr:
            bare.append(f"{at_text(t, bar)}の『{rn.lyric}』")
    out.sort(key=lambda n: n.start_tick)
    bad += _written_overlaps(out, old, end_tick, bar)
    if bad:
        raise ValueError("／".join(bad) + "。区間の外の音は消えずに残り、歌は1度に1音なので、書いた音は区間の中に重ならずに並べる"
                         "（区間の外まで直すなら、区間を広げてその音も書く）")
    return out, bare


def _written_overlaps(new: list[SongNote], old: list[SongNote], end_tick: int, bar: int) -> list[str]:
    """`replace_region` に書いた音（開始位置の順）の重なりの所見（2026-10-02・洗い出し H12）。
    書いた音どうしと、区間の外の次の元の音（区間の終わり以降で最初の音）との重なりを見る。直さない。"""
    from song_table import _hold_shown                     # 伸ばす歌詞は書いたときの `_` で見せる（表と同じ）
    out: list[str] = []
    shown = lambda n: f"『{_hold_shown(n.lyric)}』（{at_text(n.start_tick, bar)}）"   # noqa: E731
    for a, b in zip(new, new[1:]):
        if a.start_tick == b.start_tick:
            out.append(f"{shown(a)}と{shown(b)}が同じ位置")
        elif a.start_tick + a.duration_tick > b.start_tick:
            out.append(f"{shown(a)}の長さ {a.duration_tick / Qd:g} 拍が次の{shown(b)}に"
                       f" {(a.start_tick + a.duration_tick - b.start_tick) / Qd:g} 拍重なる")
    nxt = min((n for n in old if n.start_tick >= end_tick), key=lambda n: n.start_tick, default=None)
    if new and nxt is not None and new[-1].start_tick + new[-1].duration_tick > nxt.start_tick:
        a = new[-1]
        out.append(f"{shown(a)}の長さ {a.duration_tick / Qd:g} 拍が区間の外の次の音{shown(nxt)}に"
                   f" {(a.start_tick + a.duration_tick - nxt.start_tick) / Qd:g} 拍重なる")
    return out


def notes_value(start_tick: int, end_tick: int, new: list[SongNote], display: dict[int, str] | None = None) -> dict:
    """台帳に残す「置き直した音符」（仕上げたあとの形＝そのまま当て直せる）。`display` は一緒に書き直した見せる歌詞の行（行番号 → 詞）。"""
    out = {"start_tick": int(start_tick), "end_tick": int(end_tick),
           "notes": [n.model_dump() for n in new]}
    if display:
        out["display"] = {str(k): v for k, v in display.items()}
    return out


# ---- 旋律と詞が変わるか ----------------------------------------------------------------

#: 音符（歌う音と言葉）を決める工程。ここを作り直すと、音符に乗った手直しは当てはまらない
NOTE_STEPS = ("旋律", "サビの詞", "詞")


def notes_redone(step: str) -> bool:
    """`step` から作り直すと、旋律か詞も作り直すか（依存の木で決める）。

    ★設計では「旋律と詞の指紋」を比べる案だったが、記録から戻した工程と保存の形が
      細部で揃わない（後から足した欄の既定値など）と、**変わっていないのに変わった**と出る。
      作り直す工程は木で決まっているので、そちらで判定する。
    """
    from song_gate import affected
    return bool(set(NOTE_STEPS) & affected(step))


# ---- 当て直し --------------------------------------------------------------------------

def design_redone(step: str) -> bool:
    """`step` から作り直すと、**設計**（速さ・調を決める工程）も作り直すか。"""
    from song_gate import affected
    return "設計" in affected(step)


def expression_redone(step: str) -> tuple[bool, bool]:
    """`step` から作り直すと、(**歌い方**（tuning を決める工程）, **歌の崩し**（artic を決める工程）) を作り直すか。"""
    from song_gate import affected
    got = affected(step)
    return "歌い方" in got, "歌の崩し" in got


def untransposable(s: SongScore) -> list[str]:
    """楽譜のコードのうち、**読めなくて移調できないもの**の所見（2026-09-27・監査）。空なら全部移調できる。"""
    from song_design import chord_problem
    out: list[str] = []
    for c in s.chords:
        why = chord_problem(c.chord)
        if why and why not in out:
            out.append(why)
    return out


def cover_reapply(requests: list) -> list:
    """**カバーを作り直したあとに当て直す台帳の行**（2026-10-01・別件①）——作り直しで消えるもの（声部）だけ。

    カバーの作り直し（`rearrange_song`・`remake_from_step` の編成／伴奏／ハモリ）は**いまの曲**（作業台・`song_cover_load.find_parts`）
    から組み直すので、移調・速さ・置き直した音符はもう楽譜（と伴奏 MIDI）に入っている。当て直すと二重にかかる
    （移調は足し算なので、+2 の曲が +4 になった）。作り直しで丸ごと差し替わるのは声部（`extra_voices`＝ハモリを付け直す）だけなので、
    足した声部・声部の直しだけを当て直す。オリジナル曲は工程の記録（手直しの前）から組み直すので、全部当て直す（`apply_edits`）。
    """
    return [r for r in requests if getattr(r, "target", "") == ASSEMBLY_VOICE]


def apply_edits(score: SongScore, requests: list, *, notes_changed: bool,
                design_changed: bool = False, tuning_changed: bool = False,
                artic_changed: bool = False) -> tuple[SongScore, list[str], list]:
    """台帳の生きている手直しを、作り直した楽譜へ**頼まれた順に**当てる。

    ★**設計を作り直したときは、速さ・移調を当てない**（2026-09-27・目録の道具-7）。設計の LLM が新しい BPM と調を
      決めているので、前の速さを当てると**設計の決定を黙って戻し**、移調は**新しい調の上に足される**
      （前の調から数えた差なので、意図と違う高さになる）。当てずに「当て直し待ち」にして、
      コーディネーターが新しい設計を見て当て直すか決める（`reapply_note` に理由と新しい値を出す）。
      設計を作り直さない作り直し（詞・伴奏…）では、速さも調も前の版と同じなので、そのまま当てる。

    ★**置き直した音符の歌い方・崩しは工程の決定**（2026-09-27・監査）。台帳に残すのは人・コーディネーターが
      決めたもの（高さ・位置・長さ・詞）で、歌い方（`tuning`）・崩し（`artic`/`artic_depth`）は位置で引き継いだ
      工程の決定にすぎない。だから**歌い方の工程を作り直したとき（`tuning_changed`）は、作り直した楽譜の同じ位置の音の
      `tuning` を使う**（崩しは `artic_changed` で同じ）。以前は台帳の前の値ごと当て直していたので、歌い方を作り直しても
      書き直した区間だけ前の歌い方のままだった。使った値は台帳にも写す（次の作り直しで前の値へ戻さないため）。
      同じ位置に音が無い音は台帳の値のまま。どちらを使ったかは報告の一行（`done`）に書く。
      作り直していなければ台帳の値を使う。

    ★**読めないコード名があって移調できないときは当てない**（「当て直し待ち」・`reapply_note` が理由を出す）。
      移すと 0〜127 をはみ出すとき（曲全体の移調・声部の移調）も同じ（`transpose_problem`・2026-10-03）。

    ★**主旋律を楽器でも重ねている楽譜（`lead`）には、置き直した音符を当てない**（2026-10-03・再レビュー）。台帳の音符は歌だけなので、
      楽器の主旋律とずれる。当て直し待ちにして、コーディネーターが旋律の工程から直すか決める（`replace_region` の断り方と同じ）。

    Returns:
        (当てたあとの楽譜, 当てたことの一覧（報告用）, 当て直し待ちにした行)
    """
    done: list[str] = []
    wait: list = []
    s = score
    for r in requests:
        if r.status != "active" or not r.value:
            continue
        v = r.value
        if r.target in (ASSEMBLY_TEMPO, ASSEMBLY_TRANSPOSE) and design_changed \
                and (v.get("tempo") or v.get("semitones")):
            r.status = "reapply"                            # 設計が新しい速さ・調を決めた＝黙って上書きしない
            wait.append(r)
            continue
        if r.target == ASSEMBLY_TEMPO and v.get("tempo"):
            s = s.model_copy(update={"tempo": float(v["tempo"])})
            done.append(f"速さ {float(v['tempo']):g}")
        elif r.target == ASSEMBLY_TRANSPOSE and v.get("semitones"):
            # 読めないコードを残して移調すると、そのコードだけ元の調で鳴る。作り直した楽譜では高さが変わっているので、
            # 移すと 0〜127 をはみ出すこともある（2026-10-03・はみ出したまま保存すると次に開けない）——どちらも当てずに待たせる
            if untransposable(s) or transpose_problem(s, int(v["semitones"])):
                r.status = "reapply"
                wait.append(r)
                continue
            s = transpose_score(s, int(v["semitones"]))
            done.append(f"移調 {int(v['semitones']):+d}")
        elif r.target == ASSEMBLY_VOICE:
            s, msg, ok = _apply_voice(s, r, notes_changed=notes_changed)
            if ok:
                if msg:
                    done.append(msg)
            else:
                r.status = "reapply"
                wait.append(r)
        elif r.target == NOTES and "notes" in v:
            # 旋律か詞が変わった＝同じ拍に貼ると別の言葉の上に乗る。
            # ★**主旋律を楽器でも重ねている楽譜（`lead`）にも当てない**（2026-10-03・再レビュー）。台帳の音符は歌の音符だけなので、
            #   当てると楽器の主旋律が古い旋律のまま残って歌とずれる（移調は `lead` も動かすのに、ここだけ歌だけ動いていた）。
            #   道具の `replace_region` が楽器の主旋律のある曲を断る（`tools/editing.lead_refusal`）のと同じ考え
            if notes_changed or not s.lead.is_empty():
                r.status = "reapply"
                wait.append(r)
                continue
            new = [SongNote.model_validate(x) for x in v["notes"]]
            how = "歌い方・崩しは台帳の値"
            if tuning_changed or artic_changed:
                new, how = _fresh_expression(s.vocal_notes, int(v["start_tick"]), int(v["end_tick"]), new,
                                             tuning=tuning_changed, artic=artic_changed, bar=score_bar_ticks(s))
                v["notes"] = [n.model_dump() for n in new]   # 台帳も工程の最新の決定に揃える
            merged, _ = patch_region(s.vocal_notes, int(v["start_tick"]), int(v["end_tick"]), new)
            # 詞の全文も音符から組み直す（`replace_region` と同じ・2026-09-27・監査）
            upd = {"vocal_notes": merged, "lyrics_full": "".join(n.lyric for n in merged)}
            if v.get("display") and getattr(s, "lyrics_display", ""):
                # 一緒に書き直した見せる歌詞の行も当て直す（行の数が変わっていなければ。旋律と詞が変わらないときだけここへ来る）
                upd["lyrics_display"] = apply_display_edits(s, {int(k): t for k, t in v["display"].items()})
            s = s.model_copy(update=upd)
            done.append(f"音符 {r.scope or ''}（{how}）".replace(" （", "（"))
    return s, done, wait


def _fresh_expression(notes: list[SongNote], start_tick: int, end_tick: int, new: list[SongNote], *,
                      tuning: bool, artic: bool, bar: int) -> tuple[list[SongNote], str]:
    """置き直した音符に、**作り直した楽譜の同じ位置の音**の歌い方・崩しを写す。戻り値は (音符, 報告の文)。
    `bar` は曲の1小節の tick（報告の位置を `小節:拍` で書く）。"""
    by_tick = {n.start_tick: n for n in notes if start_tick <= n.start_tick < end_tick}
    what = "・".join(w for w, on in (("歌い方", tuning), ("崩し", artic)) if on)
    out: list[SongNote] = []
    fresh, kept = 0, []
    for n in new:
        src = by_tick.get(n.start_tick)
        if src is None:
            out.append(n)
            kept.append(f"{at_text(n.start_tick, bar)}の『{n.lyric}』")
            continue
        upd: dict = {}
        if tuning:
            upd["tuning"] = src.tuning.model_copy() if src.tuning is not None else None
        if artic:
            upd.update(artic=src.artic, artic_depth=src.artic_depth)
        out.append(n.model_copy(update=upd))
        fresh += 1
    msg = f"{what}は作り直した工程の新しい値（{fresh}音）"
    if kept:
        msg += (f"・新しい楽譜の同じ位置に音が無い {len(kept)}音（{'／'.join(kept[:6])}"
                + (" ほか" if len(kept) > 6 else "") + "）は台帳の前の値のまま")
    return out, msg


def _apply_voice(s: SongScore, r, *, notes_changed: bool) -> tuple[SongScore, str, bool]:
    """声部の手直し1件。戻り値は (楽譜, 報告, 当てられたか)。"""
    v = r.value
    voices = list(s.extra_voices)
    if r.op == "add" or v.get("track"):
        if notes_changed:                                  # 足した声部は前の旋律から作ったもの
            return s, "", False
        from song_schema import VoiceTrack
        vt = VoiceTrack.model_validate(v["track"])
        gone = {vt.title, v.get("replaces") or ""}         # 差し替えなら前の声部も外す
        voices = [x for x in voices if x.title not in gone] + [vt]
        return s.model_copy(update={"extra_voices": voices}), f"声部を足す『{vt.title}』", True
    name = v.get("title") or r.scope
    hit = [i for i, x in enumerate(voices) if name and x.title == name]
    if r.op == "remove":
        if not hit:
            return s, "", True                             # 消す相手がもう無い＝当てることが無い
        voices = [x for i, x in enumerate(voices) if i not in hit]
        return s.model_copy(update={"extra_voices": voices}), f"声部を消す『{name}』", True
    if not hit:                                            # 直す相手が見つからない＝どれに当てるかは人が決める
        return s, "", False
    if v.get("transpose") and any(transpose_problem(s, int(v["transpose"]), voice=voices[i].title) for i in hit):
        return s, "", False                                # 作り直した声部を移すとはみ出す＝どうするかは人が決める（2026-10-03）
    for i in hit:
        x = voices[i]
        upd = {}
        if v.get("transpose"):
            upd["notes"] = [n.model_copy(update={"key": n.key + int(v["transpose"])}) for n in x.notes]
        if v.get("gain_db") is not None:
            upd["gain_db"] = float(v["gain_db"])
        voices[i] = x.model_copy(update=upd)
    return s.model_copy(update={"extra_voices": voices}), f"声部を直す『{name}』", True


def reapply_note(wait: list, score: SongScore | None = None) -> str:
    """当て直し待ちの一覧（コーディネーターへ・作り直しの戻り値に載せる）。

    `score` は作り直したあとの楽譜（速さ・移調の当て直し待ちに、設計が新しく決めた速さを添えるため）。
    """
    if not wait:
        return ""
    lines = ["【当て直し待ち】前に手で直したことが、作り直しで**そのままでは当てはまらない**ので、機械は当てていない。"
             "新しい楽譜を見て、同じ意図で当て直す（`replace_region` / `adjust_song` などで。"
             "呼ぶときは `supersedes` にこの id を指す）。**意図が読めない・置き場が見つからないときは当てず、"
             "最終報告でマスターに伝える。**"]
    for r in wait:
        why = ""
        if r.target == ASSEMBLY_TEMPO:
            why = ("設計を作り直したので、設計が新しい速さを決めている"
                   + (f"（いま {float(score.tempo):g}）" if score is not None else "")
                   + "。前の速さを当てると設計の決定を戻す——マスターの頼みが今も要るかで決める")
        elif r.target == ASSEMBLY_TRANSPOSE and score is not None and untransposable(score):
            why = ("読めないコード名があり、移調するとそのコードだけ元の調で鳴る: "
                   + " / ".join(untransposable(score)[:3]) + "——コードを直してから移調する")
        elif (r.target == ASSEMBLY_TRANSPOSE and score is not None and (r.value or {}).get("semitones")
              and (out := transpose_problem(score, int(r.value["semitones"])))):
            why = f"作り直した楽譜を{out}——いまの高さを見て、移す量を決め直す"
        elif r.target == ASSEMBLY_TRANSPOSE:
            why = ("設計を作り直したので、設計が新しい調を決めている。前の移調は前の調から数えた差なので、"
                   "そのまま足すと意図と違う高さになる——いまの高さを見て、要るならあらためて移調する")
        elif r.target == NOTES and score is not None and not score.lead.is_empty():
            why = ("この曲は主旋律を楽器でも重ねて弾いている。台帳の音符は歌の音符だけなので、当てると楽器の主旋律が"
                   "古い旋律のまま残って歌とずれる——旋律を直すなら remake_from_step('旋律', 頼み)（楽器の主旋律も一緒に作り直る）")
        elif (r.target == ASSEMBLY_VOICE and score is not None and (r.value or {}).get("transpose")
              and (out := transpose_problem(score, int(r.value["transpose"]),
                                            voice=(r.value or {}).get("title") or r.scope))):
            why = f"作り直した声部を{out}——いまの高さを見て、移す量を決め直す"
        elif r.target in (NOTES, ASSEMBLY_VOICE):
            why = "旋律か詞が変わったので、前の拍に貼ると別の言葉・別の音に乗る"
        lines.append(f"  [{r.id}] {r.target}" + (f"（前の範囲: {r.scope}）" if r.scope else "")
                     + f": 「{r.words[:200]}」" + (f"\n      なぜ当てていないか: {why}" if why else ""))
    return "\n".join(lines)
