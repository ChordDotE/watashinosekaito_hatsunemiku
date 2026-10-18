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

"""**演奏の工程の呼び出し**——編成・伴奏（点検役とやり直し）・ハモリの計画・声部・彩り。

2026-09-30 に `song_compose._compose_once_body` から切り出した。LLM の型・指示書・工程の検査は各工程のファイル
（`instrumentation`・`song_arrange`・`harmony`）にあり、ここは**前の工程の決定を渡して呼び、伴奏の組み立て（`to_spec`）を
組み直す**呼び出し役。編成・伴奏・ハモリは再生（`song_replay`）が差し替える呼び出し口（`ctx.plan_instruments`・`ctx.arrange`・
`ctx.plan_harmony`）を通す。伴奏の組み立て（`st.spec`）を伴奏・彩りが共有するので、この5つは1つのファイルに置く。
奏法（`artic_plan.run_articulation`）・歌い方（`vocal_expression.run_expression`）・歌の崩し（`vocal_artic.run_vocal_artic`）は各工程のファイルにある。
"""
from __future__ import annotations

import song_context as ctx                                 # 呼び出し口（編成・伴奏・ハモリ・点検役）・申し送り
from instrumentation import describe, instrument_name
from song_arrange import (add_colour, apply_fade_out, fade_span, fix_arrangement, fix_colour, merge_colour,
                          pattern_table, to_spec)
from song_checks import strip_ascii_ws                     # 半角の名前の前後の空白（§0-6・2026-10-03）
from song_design import Q, grid_slots
from song_flow import NOT_SWALLOW, _maybe_redo_from, _rec, _redo_options
from song_fix import cut_note                               # 切ったら書く（2026-10-04）
from song_gate import MAX_REDO, Finding
from song_gate_brief import redo_ask                       # 前回の指摘と「指摘への対応」の頼み（2026-10-04・組P2a）
from song_llm import solfa_to_midi
from song_model import StepRecord

#: 伴奏の自工程のやり直しで、指摘のあとに置く1行（前回の伴奏そのものは `song_context.redo_base` → `asks_for` が末尾に足す・組F4）。
#: 新しい曲（`run_arrange_gate`）とカバー（`song_arrange_cover._arrange_gate`）で同じ文
REDO_BASE_NOTE = ("前回の伴奏は、下の【この工程の前回の出力（作り直す前の版）】にあります。"
                  "**これを土台に、指摘の所だけ直してください**（指摘に関係ない区間・パートは前回のまま残す）。")


def _kit_kind(kit: str) -> str:
    """編成の `drum_kit` → 種類の名前（`DRUM_KITS` の鍵）。読み方は `timbre_map.drum_program` の1か所（2026-09-29）。

    以前は `orchestra` と完全一致でしか比べず、編成に勧めている番号つきの名前（`49:Orchestral`）は標準キットの扱いになり、
    伴奏へ「グルーヴのドラム（ハイハット・クラッシュ）を土台に」と渡っていた。
    ★2026-10-02（判断6）から書けるのは一覧の名前だけ（`timbre_map.kit_names`・1つのキットに名前1つ）なので、GM の並びのキットは
      名前そのもの。番号つきのキット（`27:Dance`）は `DRUM_KITS` に無いので書いたまま返す（大文字小文字も変えない）。
    """
    from timbre_map import DRUM_KITS, drum_program
    prog = drum_program(kit)
    return next((k for k, p in DRUM_KITS.items() if p == prog), strip_ascii_ws(kit))


def _drum_kit(song) -> str:
    """編成が決めたドラムの種類（楽譜の `ArrangementSpec.drum_kit` へ運ぶ・2026-09-27）。"""
    return getattr(getattr(song, "instrumentation", None), "drum_kit", "") or ""


def _voices_desc(tracks, timeline, bar_ticks: int, *, limit: int = 24) -> str:
    """**重ねた声（合いの手・追っかけ・ハモリ）が入っている所**を、彩りへ渡す形に書く。

    1拍より長く空いたら別のかたまりとして数え、`区間 小節:拍〜小節:拍（声部名）` で並べる。
    彩りは**ここを避けて**対旋律・リフを置く（2026-09-25・声部を先に決めるようにした）。
    """
    out: list[str] = []
    for tr in tracks or []:
        notes = sorted(getattr(tr, "notes", None) or [], key=lambda n: n.start_tick)
        run: list = []
        for n in notes + [None]:
            if run and (n is None or n.start_tick - (run[-1].start_tick + run[-1].duration_tick) > Q):
                a, z = run[0].start_tick, run[-1].start_tick + run[-1].duration_tick
                out.append(f"  {timeline.at(a):<8} {a // bar_ticks + 1}:{(a % bar_ticks) / Q:.2f}"
                           f"〜{z // bar_ticks + 1}:{(z % bar_ticks) / Q:.2f}（{getattr(tr, 'title', '声部')}）")
                run = []
            if n is not None:
                run.append(n)
    if len(out) > limit:
        out = out[:limit] + [f"  ほか {len(out) - limit} か所"]
    return "\n".join(out)


def inspect_arrange(spec, bars_of: dict, warns: list[str],
                    sections: list | None = None) -> list[Finding]:
    """伴奏を機械で見る（**事実だけ**。良し悪しは点検役）。区間ごとの楽器の顔ぶれ・歌の音域との重なり・ドラムの細かさは
    所見でなく材料の表（`arrange_facts`・毎回出す）。`bars_of`・`sections` は呼び手の形を変えないために残している（いまは読まない）。"""
    findings: list[Finding] = []
    if warns:
        findings.append(Finding(what="伴奏の組み立てが書いた所見", evidence=cut_note(" ; ".join(warns), 1500)))
    # ★抑揚（音量差）は**ここでは測らない**。まだ鳴らしていないため。
    # 鳴らす前の見積もりは当てにならないと実証済み（見積 7.9dB / 実測 2.9dB）。
    # 仕上げの後に、コーディネーターが `check_mix_quality` で実測して判断する。
    # ★「いちばん静かな区間と厚い区間で楽器の種類数が同じ」「静かな区間に4種以上」（`inspect_skeleton`）は 2026-10-03（組G3）に外した。
    #   区間の1小節目と最後の小節だけで数え（『茜の空』chorus の 2,4,6,8,10 小節のイングリッシュホルンを数えなかった）、
    #   比べる文・数の帯・「音量だけ変えても盛り上がりには聞こえない」と断じる添え書きがあった。いまは `arrange_facts` の表
    # ★**全パートが同じ音色**は、形式の検査から外して点検役の材料にした（2026-09-28 本人判断）。ピアノだけの編曲なら
    #   ベースも和音もピアノで正しい——正解が曲で変わるので、機械は断じず、点検役が編成の意図と照らして判断する。
    #   （2026-09-13『この衝動』で既定値のせいで全パートがベースの音色になった事故は、楽器番号を必須にして断った）
    _tim = [getattr(p, "timbre", "") for p in (getattr(spec, "parts", None) or [])]
    if len(_tim) > 1 and len(set(_tim)) == 1:
        findings.append(Finding(what=f"伴奏の全パート（{len(_tim)}本）が同じ音色 {_tim[0]} で鳴る",
                                evidence="ソロ楽器の編曲なら正しい。編成の意図（楽器ごとの役）と食い違っていないかを見る"))
    return findings


# ★`check_mix`（書き出した音を測って関門へ通す）は**どこからも呼ばれていなかった**ので外した（2026-09-27・目録の関門-5）。
#   書き出した音の抑揚は、コーディネーターが `check_mix_quality`（tools/diagnosis.py）で測って判断している。


def _num(x: float) -> str:
    return f"{round(x, 2):g}"


def _same_pitch_block(mel: list[tuple[int, int, int]], acc: list[tuple[int, int, int, str]], secs: list,
                      bt: int, who: str, sung: list) -> str:
    """【{who}と同じ高さで同時に鳴る伴奏の音】の節（`arrange_facts` の4つ目）。区間ごとに `小節:拍「歌詞」高さ パート` で並べる。
    数えるのは `song_finish_facts.same_pitch`（最終の点検役の区間ごとの表と同じ数え方）。"""
    import song_finish_facts as ff
    from artic_plan import tick_key
    from song_table import spell_pitch
    sec_at = lambda t: next((k for k, a, b in secs if (a - 1) * bt <= t < b * bt), "")   # noqa: E731
    lyric = {int(n.start_tick): (n.lyric or "") for n in sung}
    rows = ff.same_pitch_rows(ff.same_pitch(mel, acc), sec_at, lambda t: tick_key(t, bt), spell_pitch,
                              [k for k, _a, _b in secs], lyric_at=lyric.get)
    head = (f"【{who}と同じ高さで同時に鳴る伴奏の音】（{who}の音ごとに、同じ音名・同じオクターブの伴奏の音が同時に鳴っているパート。"
            f"`小節:拍「歌詞」高さ パート`・小節:拍は拍を0から数える）")
    body = [f"■{k}: {rows[k]}" for k, _a, _b in secs if k in rows]
    return "\n".join([head] + (body or ["（無い）"]))


def arrange_facts(score, bars_of: dict, *, intensity_of: dict | None = None) -> str:
    """伴奏の点検役の材料の表（**機械が数えた事実だけ**・毎回出す・2026-10-03 組G3）。`score` は組み立てた伴奏（`arrangement`）・
    コード（`chords`）・歌（`vocal_notes`）・主旋律の楽器（`lead`）を持つ楽譜。`bars_of` は区間 → (最初の小節, 最後の小節)（1始まり）。

    1. **区間ごとに鳴る楽器**: 伴奏を曲の小節に置き（`song_audio.pitched_events`＝仕上げの点検役・書き出しと同じ置き方）、
       実際に音が鳴る小節を楽器ごとに数える。ドラムは別の行（`bar_filter` の小節）。比べる文・閾値は書かない
    2. **歌の音域と伴奏の音の重なり**: 歌のある区間で、楽器ごとに「その区間の歌の最低〜最高に入る音／その楽器の音」と、
       入る音が歌の鳴っている間と重なる長さ（拍）。鳴る高さで比べる（理論は編曲の音域の住み分け）。インストは主旋律の楽器の音を歌の代わりに使う
    3. **歌と同じ高さで同時に鳴る伴奏の音**（2026-10-04）: 歌の音ごとに、同じ音名・同じオクターブの伴奏の音が同時に鳴るパートと場所
       （`_same_pitch_block`。歌声が二重に聞こえた『水たまりのプリズム』8〜9秒のギター。寄り添う意図なら正しい・読み方は点検役）
    4. **ドラムの打点の細かさ**: 区間ごと・打楽器ごとに、鳴る小節での1拍（4分音符）あたりの打点の数（BPM200 が半分に聞こえた指摘）
    """
    from collections import Counter

    from song_audio import Q as _Q
    from song_audio import bars_of_filter, pitched_events, score_bar_ticks, song_bars
    from song_finish_gate import _part_name                  # 仕上げの点検役と同じ呼び名（使うだけ・書き換えない）
    from song_table import spell_pitch
    from song_timeline import LEAD_IN
    from timbre_map import gm_name, parse_gm_bank

    secs = sorted(((k, a, b) for k, (a, b) in (bars_of or {}).items() if k != LEAD_IN), key=lambda x: x[1])
    if not secs:
        return ""
    spec = getattr(score, "arrangement", None)
    bt = score_bar_ticks(score)
    n_bars = max(song_bars(score), secs[-1][2])
    sec_of_bar = {x: k for k, a, b in secs for x in range(a, b + 1)}

    def bars_hit(on: int, end: int) -> range:
        return range(on // bt + 1, (max(on + 1, end) - 1) // bt + 2)

    order: list[str] = []
    plays: dict[str, dict[str, set[int]]] = {}              # 区間 → 楽器 → 鳴る小節
    notes_in: dict[str, list[tuple[int, int, int, str]]] = {}   # 区間 → (開始, 終わり, 高さ, 楽器)
    drum_bars: dict[str, set[int]] = {}
    drum_hits: dict[str, dict[str, list]] = {}             # 区間 → 打楽器 → [打点の数, 鳴る小節]

    def put(name: str, on: int, end: int, keys: list[int]) -> None:
        if name not in order:
            order.append(name)
        for x in bars_hit(on, end):
            if (k := sec_of_bar.get(x)) is not None:
                plays.setdefault(k, {}).setdefault(name, set()).add(x)
        if (k := sec_of_bar.get(on // bt + 1)) is not None:
            notes_in.setdefault(k, []).extend((on, end, key, name) for key in keys)
        if name != lead_name:
            acc_all.extend((on, end, key, name) for key in keys)

    acc_all: list[tuple[int, int, int, str]] = []           # 伴奏の音（主旋律の楽器を除く）。歌と同じ高さで同時に鳴るかを見る

    lead = getattr(score, "lead", None)
    lead_name = ""
    if lead is not None and getattr(lead, "notes", None):
        _b, prog = parse_gm_bank(lead.timbre or "")
        lead_name = f"主旋律の楽器（{gm_name(prog) if prog is not None else '?'}）"
        for n in lead.notes:
            put(lead_name, n.start_tick, n.start_tick + n.duration_tick, [int(n.key)])
    for p in (getattr(spec, "parts", None) or []):
        if p.drum_steps:
            per_bar = Counter(h.inst for h in p.drum_steps)
            for x in bars_of_filter(p.bar_filter, n_bars):
                if (k := sec_of_bar.get(x)) is None:
                    continue
                drum_bars.setdefault(k, set()).add(x)
                for inst, c in per_bar.items():
                    d = drum_hits.setdefault(k, {}).setdefault(inst, [0, set()])
                    d[0] += c
                    d[1].add(x)
            continue
        name = _part_name(p)
        for on, dur, keys, _vel in pitched_events(score, p):
            put(name, on, on + dur, list(keys))

    def head(k: str, a: int, b: int) -> str:
        inten = (intensity_of or {}).get(k)
        return f"■{k}（{a}〜{b}小節目・{b - a + 1}小節" + (f"・強さ{inten}" if inten is not None else "") + "）"

    out = ["【区間ごとに鳴る楽器】（組み立てた伴奏を曲の小節に置き、実際に音が鳴る小節を数えた。"
           "`楽器 鳴る小節の数/区間の小節数`。ドラムは別の行）"]
    for k, a, b in secs:
        got = plays.get(k, {})
        row = " ・ ".join(f"{nm} {len(got[nm])}/{b - a + 1}" for nm in order if nm in got)
        out.append(f"{head(k, a, b)}: {row or '鳴る楽器なし'}")
        db = drum_bars.get(k)
        out.append(f"   ドラム: {len(db)}/{b - a + 1}小節" if db else "   ドラム: 鳴らない")

    # ---- 歌の音域と伴奏の音の重なり ----
    mel = [(int(n.start_tick), int(n.start_tick + n.duration_tick), int(n.key)) for n in (getattr(score, "vocal_notes", None) or [])]
    who = "歌"
    if not mel and lead_name:
        mel = [(int(n.start_tick), int(n.start_tick + n.duration_tick), int(n.key)) for n in lead.notes]
        who = "主旋律（楽器）"
    out.append("")
    out.append(f"【{who}の音域と伴奏の音の重なり】（{who}のある区間ごと・鳴る高さで比べた。"
               f"`楽器 {who}の音域〈その区間の最低〜最高〉に入る音/その楽器の音`〈和音は1音ずつ数える〉"
               f"・入る音が{who}の鳴っている間と重なる長さ（拍））")
    rows = 0
    for k, a, b in secs:
        t0, t1 = (a - 1) * bt, b * bt
        m = sorted((s, e, key) for s, e, key in mel if t0 <= s < t1)
        if not m:
            continue
        lo, hi = min(x[2] for x in m), max(x[2] for x in m)
        spans: list[list[int]] = []                         # 歌の鳴っている時間（重なりはまとめる）
        for s, e, _k in m:
            if spans and s <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], e)
            else:
                spans.append([s, e])
        cells = []
        for nm in order:
            if lead_name and nm == lead_name:
                continue                                   # 主旋律の楽器は旋律そのもの（重なって当然）なので数えない
            ns = [x for x in notes_in.get(k, []) if x[3] == nm]
            if not ns:
                continue
            inside = [x for x in ns if lo <= x[2] <= hi]
            ov = sum(max(0, min(e, se) - max(s, ss)) for s, e, _key, _n in inside for ss, se in spans)
            cells.append(f"{nm} {len(inside)}/{len(ns)}音・{_num(ov / _Q)}拍")
        out.append(f"■{k}（{who} {spell_pitch(lo)}〜{spell_pitch(hi)}）: " + (" ・ ".join(cells) or "伴奏の音なし"))
        rows += 1
    if not rows:
        out.append(f"（{who}のある区間が無い）")

    # ---- 歌と同じ高さで同時に鳴る伴奏の音（2026-10-04・『水たまりのプリズム』8〜9秒）----
    # 上の「音域に入る音の数」は同じ高さかを見ていなかった（ギターのアルペジオが歌と同じ F#4・A4 を同時に鳴らし、歌声が二重に聞こえた）
    out.append("")
    out.append(_same_pitch_block(mel, acc_all, secs, bt, who, getattr(score, "vocal_notes", None) or []))

    # ---- ドラムの打点の細かさ ----
    out.append("")
    out.append("【ドラムの打点の細かさ】（区間ごと・打楽器ごとに、鳴る小節での1拍〈4分音符〉あたりの打点の数）")
    if not drum_hits:
        out.append("（ドラムなし）")
    else:
        beats = bt / _Q
        for k, a, b in secs:
            hits = drum_hits.get(k)
            if not hits:
                out.append(f"■{k}: 鳴らない")
                continue
            cells = sorted(((inst, c / (len(xs) * beats), len(xs)) for inst, (c, xs) in hits.items()),
                           key=lambda x: -x[1])
            out.append(f"■{k}: " + " ・ ".join(f"{inst} {_num(d)}" + (f"（{n}/{b - a + 1}小節）" if n != b - a + 1 else "")
                                              for inst, d, n in cells))
    return "\n".join(out)


def _gate_score(st, spec):
    """伴奏の点検役が数えるための楽譜（楽譜を組む `song_place.make_score` の前なので、同じ材料で仮に組む）。
    歌（`st.notes`）・コードの配置（`st.spans`）・拍子・主旋律の楽器（`lead_part`＝楽譜に入るのと同じもの）。"""
    from song_place import lead_part
    from song_schema import SongScore
    return SongScore(title=st.title or "", tempo=float(st.song.design.bpm), time_signature=st.ts,
                     vocal_notes=list(st.notes or []), chords=list(st.spans or []), arrangement=spec, lead=lead_part(st))


def run_instrumentation(st) -> None:
    """**編成**（どの楽器で演奏するか・GM 128種から）。

    読む: `song.theme`・`song.design`・`st.brief_text`・曲名（`song.step_title()`＝工程『曲名』→ 仮の曲名）。
    書く: `song.instrumentation`・`st.title`・`st.genre_name`・`st.inst_gms`・`st.inst_desc`・`st.allow_drums`。
    """
    song, verbose = st.song, st.verbose
    theme, design = song.theme, song.design
    # ---- ⑥ 編成（どの楽器で演奏するか・GM 128種から）----
    # 音色を編曲プロンプトの一覧（24種）から選ばせていたが、絞っていたのは
    # **一覧を毎回書く都合**であって音源の制約ではなかった。工程を分けて128種を対象にする
    # （2026-08-22・本人の提案）。編曲側へは**選ばれた数本だけ**を番号で渡す。
    # ★**曲名は工程『曲名』の値**（2026-10-03・どの形でも詞〔インストは旋律〕のあとに走る。無ければテーマの仮の曲名・`Song.step_title`）。
    #   見出しに出すだけで音楽の入力ではない（`song_gate.DEPS` に数えない）。組F2 ではテーマの曲名、判-11 では詞の設計の曲名
    #   （インストだけテーマ）、それより前は詞の曲名を読んだ
    title = song.step_title()
    genre_name = ctx._GENRE_NAME[theme.genre]
    # ★**編成が作れなければ曲づくりを止める**（2026-09-29 本人方針「組み立ては補正しない」）。以前は例外を握って
    #   「従来の音色選択に戻す」と進み、伴奏の組み立て（`to_spec`）が役割名から音色を推して鳴らしていた。
    #   例外はそのまま上げる（旋律・伴奏の工程の失敗と同じ扱い。呼び手の `draft_melody` が失敗を記録して返す）。
    #   以前の `except Exception` は、途中で見せる印（`ShowProgress`）と差し戻し（`RedoFrom`）まで握っていた。
    # 頼まれごと・前回の出力はラベル「編成」で song_llm.run_sync が足す（ここで足すと二重になる）
    iplan = ctx.plan_instruments(title, genre_name, design.bpm, st.brief_text)
    from instrumentation import InstrumentationStep          # noqa: E402（ここでだけ使う）
    # 曲の形（2026-10-03）を渡す: アカペラは楽器0本・ドラム none、インストは主旋律の楽器 `lead` が要る（形ごとの決まりは編成の検査）。
    #   設計の区間を渡すと、主旋律を弾く区間 `lead_sections` を設計と照らす
    iplan, _ = InstrumentationStep(getattr(song, "form", "song") or "song",
                                   design.sections).run(iplan, song, verbose=verbose)
    inst_gms: list[tuple[int, int]] = [(i.bank, i.gm) for i in iplan.instruments]   # (bank, 番号) の並び
    inst_desc = describe(iplan)
    # 編成の検査（`InstrumentationStep`）が空と一覧に無い名前を所見にするので、ここへは一覧の名前か `none` だけが来る（2026-10-02 から小文字の none だけ）
    allow_drums = strip_ascii_ws(iplan.drum_kit) != "none"
    st.log["instrumentation"] = iplan.reason
    song.instrumentation = iplan
    _rec(song, "instrumentation", iplan, verbose=verbose)
    if verbose:
        print("  [6 編成] " + " / ".join(
            f"{instrument_name(i)}({i.part})" for i in iplan.instruments)
            + f"  ドラム={iplan.drum_kit}")
    st.title, st.genre_name = title, genre_name
    st.inst_gms, st.inst_desc, st.allow_drums = inst_gms, inst_desc, allow_drums


def run_arrangement(st) -> None:
    """**伴奏**（骨組み）。点検役とやり直しは `run_arrange_gate`（ハモリの計画のあと）。

    読む: `song.groove`・`song.instrumentation`・`song.melody`（歌の音域）・`song.design`・`st.ch_map`・`st.bars_of`・編成の値。
    書く: `song.arrangement`・`st.groove_note`・`st.vocal_range`・`st.arrange_rec`。
    """
    song, verbose, log = st.song, st.verbose, st.log
    design, melody = song.design, song.melody
    title, genre_name, ts = st.title, st.genre_name, st.ts
    brief_text, inst_desc, inst_gms, allow_drums = st.brief_text, st.inst_desc, st.inst_gms, st.allow_drums
    # 伴奏も**これまでの判断**を受け取る（旋律の意図に噛み合う編成にするため）。
    # なお arrange 側は _call を通らないので、費用集計には含まれない
    #
    # ★**グルーヴで決めたドラムを必ず渡す**（2026-09-07）。旋律は**そのノリの上で**書かれている。
    #   ここで編曲が別のドラムを作ると、**歌が乗ったノリと実際に鳴る音が食い違う**——
    #   グルーヴ工程を入れた意味が消える。
    groove_note = ""
    # ★**編成が「ドラムセットを使わない」と決めたら、グルーヴの刻みはノリの参考にする**（2026-09-26）。
    #   グルーヴは編成より前に決まるので、キック・スネア・ハイハットで書かれる。そこへ「このドラムを土台に」とだけ
    #   渡していたので、編成が「ドラムセットは使わず、ティンパニ等で」と決めた『風の軌跡』（ジブリ風のワルツ）に
    #   クラッシュやキックが入った。**後から決めた音の選択（編成）を、ノリ（グルーヴ）より優先する**。
    #   ノリ＝拍の重さ・刻みの細かさ・跳ね方は保つ（旋律はその上で書かれている）。
    # ★★**ドラムとベースの持ち主は伴奏**（2026-09-30・出-8）★★ グルーヴのドラム・ベースは**土台の指定で、それ自体は鳴らない**。
    #   鳴るのは伴奏が `drums` とベースのパートに書いたものだけ。以前は「このドラムとベースを土台に」とだけ渡していて、
    #   『孤独のまま』で伴奏が `drums: null` を返し（編成は standard）、ドラムが1音も鳴らなかった。
    #   編成がキットを決めたのに drums が無いのは、伴奏の工程の形式の所見（`song_arrange._issues` の `need_drums`）
    _sounds = ("※**グルーヴのドラムとベースは土台の指定で、それ自体は鳴りません。**鳴るのは、あなたが `drums` と"
               "ベースのパートに書いたものだけです。")
    # 書けるのは一覧の名前か小文字の none だけ（2026-10-02・判断6。編成の検査が空と読めない名前を所見にする）
    _kit = strip_ascii_ws(getattr(song.instrumentation, "drum_kit", "")) \
        if getattr(song, "instrumentation", None) is not None else "standard"
    if _kit == "none":
        _drum_use = ("**ドラムは鳴らしません**（編成が「ドラムを使わない」と決めた）。下のドラムの刻みは、"
                     "旋律が乗っている**ノリの参考**です——拍の重さと刻みの細かさを、ベースや和音の刻み方で出してください。"
                     "ベースはこの土台を写して、ベースのパートに書いてください。" + _sounds)
    elif _kit_kind(_kit) in ("orchestra", "brush", "jazz"):
        _kit = _kit_kind(_kit)
        _drum_use = (f"**ドラムキットは {_kit}**（編成が決めた）。下の刻みは**ノリの参考**です——"
                     "拍の重さ・刻みの細かさ・跳ね方は保ったまま、**その打楽器らしい叩き方に置き換えて `drums` に書いてください**"
                     "（orchestra ならティンパニ・大太鼓・シンバルを要所だけ。ハイハットやクラッシュを刻み続けない）。"
                     "ベースはこの土台を写して、ベースのパートに書いてください。" + _sounds
                     + "編成がキットを決めたので、`drums` を書かないと検査で所見として返します。")
    else:
        _drum_use = ("**このドラムとベースを写して、`drums` とベースのパートに書いてください。**"
                     "作り直すと、歌が乗ったノリと食い違います（区間ごとに厚みを変える・静かな区間で抜くのはよい）。" + _sounds
                     + "編成がキットを決めたので、`drums` を書かないと検査で所見として返します。")
    if getattr(song, "groove", None):
        # ★**行ごとの読み方を添える**（2026-10-01・記法の仕様書 §8・§9）。ドラムの行はドラムタブ（`-`＝休み）、bass の行は
        #   音程パートの書き方（`-`＝前の音を伸ばす）で、同じ `-` が逆の意味になる。以前は凡例なしで並べていた
        g_lines = [f"\n\n【グルーヴ（歌より先に決めた土台。**旋律はこの上で書かれています**）】",
                   f"ノリ: {song.groove.feel}",
                   _drum_use,
                   "（読み方: 打楽器の行は `drums` の `pattern` と同じドラムタブ＝`-` は休み。"
                   "`bass` の行は音程パートの `pattern` と同じ書き方＝`-` は前の音を伸ばす・`_` はそこで切る。どの行も1小節ぶん）"]
        for g in song.groove.sections:
            # ★**刻みの細かさをグルーヴから受け取る**（2026-09-16・`drum_grid`）。
            #   BPM200 でハイハットを4分（300ms）にすると **BPM100 の8ビートと同じ密度**で、
            #   曲全体が半分の速さに聞こえる（本人に実際にそう言われた）。
            _dg = g.drum_grid                               # 8 か 16（型 `DrumGrid` で閉じている・2026-09-30）
            g_lines.append(f"■{g.kind}"
                           + (f"　★**ドラムの刻みは {_dg}分**"
                              + (f"（1小節 {grid_slots(ts) * 2}文字のグリッドで書く。**薄い区間でも刻みは落とさない**"
                                 "——薄さはパート数と音域で作る）" if _dg >= 16 else "")))
            # ※文字数は拍子から出す（2026-09-30・io レビュー。以前は「16文字」の決め打ちで、6/8 でも16と伝えていた＝正しくは12）
            for d in g.drums:                               # 打楽器名は GM の正式な名前（2026-10-02・`closed_hi_hat` が入る幅）
                g_lines.append(f"   {d.inst:<14} {d.pattern}")
            g_lines.append(f"   {'bass':<14} {g.bass}")
        groove_note = "\n".join(g_lines)
    # ★★**歌の音域を編曲へ渡す**（2026-09-11 新設）★★
    #
    #   編成プロンプトには「歌と同時に同じ高さで鳴らすと喧嘩する」と書いてあるのに、
    #   **編曲工程は歌の音域を一度も受け取っていなかった**——守りようがなかった。
    #   実測: 対旋律の 144音中 86音が**歌と同時・4半音以内**で鳴っていた（『火花を散らして』）。
    #   対位法の基本（声部を離す）は、**情報が届いて初めて当てられる**。
    # ★**`pitches` と同じ物差し（国際式の音名・C メジャー基準・移調前。2026-10-01 まで固定ドの階名）で渡す**（2026-09-27）。以前は移調後の高さを見せていたので、
    #   固定ドで `pitches` を書く伴奏・彩りから見ると、C 以外の調の曲で歌の高さが最大6半音ずれて見えていた。
    #   彩りにも渡す（以前は歌の枠＝いつ歌うかだけで、高さを知らずにサビの最高音の半音上へ重ねていた）。
    _vr: dict[str, tuple[int, int]] = {}
    for _p in melody.phrases:
        _ks = [solfa_to_midi(n.solfa, n.octave) for n in _p.notes]
        if not _ks:
            continue
        _lo, _hi = _vr.get(_p.section, (999, 0))
        _vr[_p.section] = (min(_lo, min(_ks)), max(_hi, max(_ks)))
    plan = ctx.arrange(title, design.bpm, design.sections, st.ch_map, genre_name,
                       brief_text + groove_note + ("\n\n" + inst_desc if inst_desc else ""), ts,
                       vocal_range=_vr, shift_of=getattr(st, "shift_of", None))   # 頼まれごと・前段の意図・前回の出力はラベル「伴奏」で足る
    #   ↑`shift_of` は `tones` の根音の範囲を歌の音域と同じ C 基準で見せるため（2026-10-01）
    plan, _ = fix_arrangement(plan, song, bars_of=st.bars_of, instruments=inst_gms, allow_drums=allow_drums,
                              slots=grid_slots(ts), inst_desc=inst_desc or "", verbose=verbose,
                              shift_of=getattr(st, "shift_of", None))   # 移したあとの鳴る高さも検査（`to_spec` と同じ・2026-10-03）
    log["arrange"] = plan.reason
    song.arrangement = plan
    arrange_rec = _rec(song, "arrange", plan, verbose=verbose)
    if verbose:
        types = sum(len(p.patterns) for p in plan.parts) + (len(plan.drums.patterns) if plan.drums else 0)
        print("  [7 伴奏] " + " / ".join(
            f"{p.role}[{'|'.join(sp.pattern for sp in p.patterns)}]" for p in plan.parts)
            + (f" / drums[{len(plan.drums.patterns)}型]" if plan.drums else "")
            + f"  → リズム型 計{types}種")
    st.groove_note, st.vocal_range, st.arrange_rec = groove_note, _vr, arrange_rec


def run_harmony_plan(st) -> None:
    """**ハモリの計画**（どこに重ねるかだけ LLM・実音は `run_voices`）と、声の種類の受け渡し。

    読む: `song.instrumentation`（ハモリの方針・声の種類）・`song.design`・`st.brief_text`。書く: `song.harmony`・`voice_style` の選択。
    """
    song, verbose, log = st.song, st.verbose, st.log
    design, iplan, title, brief_text = song.design, song.instrumentation, st.title, st.brief_text
    # ---- ⑧ ハモリ（どこに重ねるかだけ LLM・実音は機械）----
    # ★**編成が「使う」と決めたときだけ呼ぶ**（2026-08-23 ユーザー指摘）。
    # 以前は毎回呼んでいたので、どの曲にも必ずハモリが付いた。一人きりの独白の
    # バラードにも 60音が乗り「過剰。この曲は無い方が良かった」と指摘された。
    # **重ねるのが当たり前ではない**——独白は1人で歌うから独白に聞こえる。
    # **歌い方を決めて後段（vpr の書き出し）へ渡す**（2026-08-23）。
    # 仕組みは前からあったのに繋いでおらず、全曲が `No Effect`（加工なし）で出ていた。
    # 選んだ声は楽譜（`score.voice_choice`・`song_place.make_score` が写す）が持ち、vpr へ焼くのは
    # `score_to_vpr._apply_voice_style`（`voice_choice_for`・声部ごとは `part_styles_for`）。
    import voice_style as _vs
    # ★声・歌い方・ハモリの方針は編成の型が必ず書かせる（2026-10-04・既定値をやめた）ので、ここで既定を補わない
    _vs.set_choice(voice=iplan.voice, style=iplan.vocal_style, harmony_style=iplan.harmony_style)
    log["voice_style"] = (f"{iplan.voice} / {iplan.vocal_style}"
                          + (f" / ハモリ={iplan.harmony_style}" if strip_ascii_ws(iplan.harmony) != "none" else ""))
    if verbose:
        print(f"  [歌い方] {log['voice_style']}"
              + (f" — {iplan.voice_reason[:80]}" if iplan.voice_reason else ""))

    # ★ハモリの方針は小文字の none / accent / full だけ（2026-10-02・洗い出し。以前は大文字小文字・空・no・off も none と読んでいた）。
    #   編成の検査（`InstrumentationStep`）が一覧に無い値を所見にするので、ここへは3つのどれかだけが来る
    want_harmony = strip_ascii_ws(iplan.harmony)
    if want_harmony == "none":
        log["harmony"] = ("使わない"
                          + (f"（{iplan.harmony_reason}）" if iplan.harmony_reason else ""))
        song.record(StepRecord(step="harmony", reason=log["harmony"]))
        if verbose:
            print(f"  [8 ハモリ] **使わない** — {iplan.harmony_reason or '編成の判断'}")
    else:
        try:
            # ★区間名は `plan_harmony` の中で設計と照らし、差分で直させる（2026-09-30・判-14①・`harmony.HarmonyPlanStep`）。
            #   曲を渡すと直しの記録（`harmony_fix`）が残る。直らなければハモリの工程で止まる（`StepUnfixed`＝下の NOT_SWALLOW）
            hplan = ctx.plan_harmony(title, design.sections, brief_text   # 頼まれごと等はラベル「ハモリ」で足る
                                     + f"\n\n**ハモリの方針は『{want_harmony}』**"
                                     + ("（**ここぞという所だけ**——ラスサビだけ等に絞る）"
                                        if want_harmony == "accent" else "（サビでしっかり重ねる）"),
                                     song=song, verbose=verbose)
            # ★**2段目は楽譜を組んでから呼ぶ**（2026-09-23）。声部を書くには
            #   `song_table` の表（小節:拍・コード・既にある声部）が要り、それには楽譜が要る。
            #   ここでは**計画だけ**を決めて置き、実際の音は下の「声部」で書く。
            log["harmony"] = hplan.reason
            song.harmony = hplan
            _rec(song, "harmony", hplan, verbose=verbose)
            if verbose:
                print("  [8 ハモリの計画] " + (" / ".join(
                    f"{'／'.join(p.sections)}: {p.aim}（{p.density}）" for p in hplan.parts)
                    or "なし（単声で通す）"))
        except NOT_SWALLOW:                                # 止める・途中で見せる（`_rec` の `ShowProgress`）・差し戻しは握らない（2026-09-30）
            raise
        except Exception as e:                             # ハモリで曲全体を落とさない
            # ★**失敗を曲の記録に残す**（2026-09-30・判-14②）。以前は画面の表示（`log`）にだけ書き、`song.harmony` が空のまま
            #   なので声部の工程も黙って飛んだ（計画が無いと声部は書かない）。声部の工程が 09-23 に直した形（`run_voices`）にそろえる
            import traceback as _tb
            log["harmony"] = f"（生成に失敗: {type(e).__name__}）"
            song.record(StepRecord(step="harmony", reason="", status="compromised",
                                   warnings=[f"★ハモリの計画を作れなかった: {type(e).__name__}: {e}"
                                             "——計画が無いので声部（ハモリ・合いの手）も作らない",
                                             _tb.format_exc()[-600:]]))
            if verbose:
                print(f"  [8 ハモリ] ★失敗: {type(e).__name__}: {e}")


def _arr_material(plan, instruments=None, vocal_range=None, drum_kit: str = "", shift_of=None, *, ins=None,
                  request: str = "", facts: str = "") -> str:
    """伴奏の点検役に読ませる**パターンの表**（`run_arrange_gate`・やり直しの輪でも同じ）。

    ★**`pitches`（自分の旋律）も出す**（2026-09-17）。出し忘れていたため、関門が「リフの指定が抜けています」と**誤って指摘した**。
    ★★**リフ（`riff`・`use_riff`）・音量（`gain`）・音域（`register_octave`）・楽器名も出す**（2026-09-30・出-6）★★
      表にリフが無く、点検役が「共通リフに旋律の指定が無く、ただの和音弾き」と読み違え、その所見が彩りへ渡って
      同じピアノにリフが2本鳴った。**読ませる材料が欠けていると、点検役は正しく読んだうえで間違う**。表は `song_arrange.pattern_table`
      （彩りに見せる骨組みと同じ）。歌の音域と和音の積み方も並べる（出-9。歌に重なるかは点検役が見る）。
    ★歌の音域と `tones` の根音の範囲は、伴奏の依頼文と同じ C 基準（移調前）・国際式の音名（`vocal_range_text`・2026-10-01）。
    ★`ins`（編成）を渡すと表に**主旋律の楽器・弾く区間・音量**の行が出る（2026-10-03・判-25・`pattern_table`）。
    ★`request` は**マスターの頼み**（カバーの編曲「和楽器風に」など・2026-10-03・判-27）。頼みと合っているかを点検役が見る。
    ★`facts` は**機械が数えた表**（`arrange_facts`＝区間ごとに鳴る楽器・歌の音域との重なり・ドラムの細かさ・2026-10-03 組G3）。末尾に置く。
    """
    from song_arrange import _voicing_text, vocal_range_text
    return ((f"【マスターの頼み】{request}（この伴奏は頼みに応えて組み直したもの。頼みと合っているかも見る）\n" if request else "")
            + "【伴奏のパターン】（`小節=all` は区間ぜんぶ・同じ区間に小節を指した行があればそちらが優先）\n"
            + pattern_table(plan, instruments, ins=ins)
            + (f"\n【編成のドラム】{drum_kit}" if drum_kit else "")
            + (f"\n【歌の音域】（`pitches` と同じ書き方・同じ物差し＝国際式の音名・C メジャー基準）\n"
               f"{vocal_range_text(vocal_range, shift_of)}" if vocal_range else "")
            + f"\n【和音の積み方】{_voicing_text()}"
            + (f"\n\n{facts}" if facts else ""))


def run_arrange_gate(st) -> None:
    """**伴奏の組み立て（`to_spec`）と点検役・自工程のやり直し**。ハモリの計画のあとに呼ぶ（記録の並びを変えないため）。

    読む: `song.arrangement`・`song.design`・`st.bars_of`・`st.shift_of`・編成の値・`st.groove_note`・`st.vocal_range`・
    （点検役の表のため）`st.notes`・`st.spans`・`st.lead_notes`（`_gate_score`）。
    書く: `st.spec`（伴奏の組み立て）・`st.awarn`・`st.intensity_of`・（やり直したら）`song.arrangement`・`st.arrange_rec`。
    """
    song, verbose, log = st.song, st.verbose, st.log
    design, plan, arrange_rec = song.design, song.arrangement, st.arrange_rec
    title, genre_name, ts, bars_of, timeline = st.title, st.genre_name, st.ts, st.bars_of, st.timeline
    inst_gms, allow_drums, inst_desc, shift_of = st.inst_gms, st.allow_drums, st.inst_desc, st.shift_of
    # 編成の判断が伴奏へ届いたかを検査する。**上流の決定を下流が無視する**事故が起きたため
    # （神楽の曲で「ドラムなし」と決めたのに洋楽のドラムセットが入った・2026-08-22）
    awarn: list[str] = []
    # 区間の強さ（intensity）を**実際の音量差**にする。宣言だけで音が変わらなかったため
    intensity_of = {s.kind: s.intensity for s in design.sections}
    _lead_no = getattr(getattr(song, "instrumentation", None), "lead", None)   # 主旋律の楽器も「使った」に数える（組G3）
    spec = to_spec(plan, bars_of, grid_slots(ts), inst_gms, allow_drums, awarn,
                   intensity_of=intensity_of, shift_of=shift_of, drum_kit=_drum_kit(song), lead=_lead_no)

    # **終わり方を音にする**（2026-08-23）。`fade` は音量の下り坂で、編曲側には表現手段が無い
    # （gain は区間に1つ・at_bars で変えられるのはパターンだけ）。ミックスの操作なので機械が実行する。
    # ★対象は**曲の最後の区間**（2026-09-27・目録の音量-1）。以前は「最後の歌の無い区間」を選んでいて、歌で終わる曲では
    #   途中の間奏がフェードしていた。記録は `song_fix.record_change` で残る（最初の1回だけ・作り直しでは記録しない）
    if design.ending == "fade":
        n = apply_fade_out(spec, bars_of, n_bars=timeline.total_bars, song=song)
        if n and verbose:
            print(f"  [終わり方] fade → {fade_span(bars_of)} で音量を下げる（{n}パートを小節ごとに分けた）")
    if awarn:
        log["arrange"] += "".join(f"\n★{w}" for w in awarn)
        arrange_rec.warnings.extend(awarn)
        if verbose:
            for w in awarn:
                print(f"      ★{w}")

    # 伴奏も関門へ通す。**やり直しはしない**——「薄い」は落ちサビ・無音の表現でありうるし、
    # ここでやり直すと編成から組み直しになる。判断だけ記録して、後段の判断材料にする。
    # ★**関門にパターンの表を読ませる**（2026-09-16）。それまで所見しか渡しておらず、
    #   **所見がゼロなら関門を呼びもしなかった**。だから「16小節ぜんぶ `at_bars=all` の
    #   1パターン」「リードが `tones=root`＝根音刻み」を**表を見れば一目**なのに素通りし、
    #   本人に「**間奏が単調**」と言われた。**所見が無くても必ず通す**。
    # ★**機械が数えた表**（区間ごとに鳴る楽器・歌の音域との重なり・ドラムの細かさ）を毎回添える（2026-10-03 組G3・`arrange_facts`）
    _material = _arr_material(plan, inst_gms, st.vocal_range, _drum_kit(song), shift_of,
                              ins=getattr(song, "instrumentation", None),   # 主旋律の楽器の行（判-25）
                              facts=arrange_facts(_gate_score(st, spec), bars_of, intensity_of=intensity_of))
    af = inspect_arrange(spec, bars_of, awarn, design.sections)
    # ★**所見がゼロでも必ず通す**（読ませるものは `_material` で渡している）
    # ★やり直しの輪でも同じ構成（区間の強さ）を見せる（2026-09-30・判-13。「楽器を減らせたか」を強さと見比べるため＝出-10）
    # ★**テンポと拍子も見せる**（2026-10-03 組G3）。ジャンル・テンポとドラムの刻み（【ドラムの打点の細かさ】）の食い違いを点検役が見るため
    _ctx_line = (f"曲『{title}』（{genre_name}）／BPM {design.bpm}・拍子 {ts}／構成: "
                 + " → ".join(f"{s.kind}(強さ{getattr(s, 'intensity', 3)})" for s in design.sections))
    v = ctx.judge("伴奏", log.get("arrange", ""), arrange_rec.status, arrange_rec.trouble,
                  af, verbose=verbose, material=_material, redo_from_options=_redo_options("伴奏"),
                  context=_ctx_line, after=plan)
    # ★所見は**写して**持たせる（2026-09-30・判-17②）。以前は `awarn` そのもの（`st.awarn`）を持たせていて、彩りを合わせた
    #   組み立ての所見がこのリストに後から足され、伴奏の点検役の記録に混ざっていた
    song.record(StepRecord(step="arrange_gate", reason=v.reason, notes=v.notes or "", guidance=v.guidance or "",
                           status="ok" if v.action == "accept" else "compromised",
                           warnings=list(awarn), request=v.action))
    log["arrange_gate"] = f"{v.action}: {v.reason}"
    # ★以前は「やり直すと編成から組み直しになる」という理由で**諦めていた**。
    #   戻れるようになったので、原因が編成にあるなら編成から作り直す（2026-09-01）
    _maybe_redo_from(v, "伴奏", song)
    # ★★**伴奏の自工程やり直し**（2026-09-13 追加）★★
    #   関門が「redo」と言っても、`_maybe_redo_from` は**前の工程**を指すときしか動かず、
    #   伴奏そのものを作り直す経路が無かった（詞・設計・旋律にはある）。
    #   『この衝動』で関門が「歪みギター・クリーンギター・ピアノが全く使われていない＝破綻」と
    #   見抜いたのに、そのまま先へ進んで**全パートがベース音色**で書き出された。
    _att = 0
    while v.action == "redo" and _att < MAX_REDO:
        _att += 1
        _guide = v.guidance or v.reason
        if verbose:
            print(f"  [関門] 伴奏を作り直す（{_att}回目）: {_guide[:100]}")
        # ★**捨てた案の直しの記録を、いまの伴奏のものとして後へ渡さない**（2026-09-27 監査）。消さずに名前を変えて
        #   残す（`*_machine`・`*_fix` でなくなるので、彩り・奏法の依頼文とコーディネーターの要約から外れる）
        for _r in song.history:
            if _r.step in ("arrangement_fix", "arrangement_machine") and not _r.reason.startswith("終わり方"):
                _r.step = "arrangement_discarded"
        _before = plan                                     # 前の回の稿（点検役に変わった所を並べる・組P2a）
        # ★【前回の指摘】と「`response` に指摘ごとの対応を書く」を足す（2026-10-04・組P2a）
        # ★**前回の伴奏を土台として渡す**（2026-10-04 組F4・1‑2）。以前は指摘だけで前の稿が無く、伴奏を一から書き直していた
        #   （直した所以外も変わり、点検役は新しく崩れた所を毎回追った）。その間だけ前回の出力に置く＝差し戻しの周と同じ口
        #   （`asks_for` の【この工程の前回の出力】「これを土台に…変える必要がある所だけ」）
        with ctx.redo_base("伴奏", _before):
            plan = ctx.arrange(title, design.bpm, design.sections, st.ch_map, genre_name,
                               st.brief_text + st.groove_note + ("\n\n" + inst_desc if inst_desc else "")
                               + f"\n\n★**前回の伴奏は作り直しになりました。**\n{redo_ask(_guide)}\n{REDO_BASE_NOTE}", ts,
                               vocal_range=st.vocal_range, shift_of=shift_of)
        plan, _ = fix_arrangement(plan, song, bars_of=bars_of, instruments=inst_gms, allow_drums=allow_drums,
                                  slots=grid_slots(ts), inst_desc=inst_desc or "", verbose=verbose,
                                  shift_of=shift_of)               # 移したあとの鳴る高さも検査（2026-10-03）
        log["arrange"] = plan.reason
        song.arrangement = plan
        arrange_rec = _rec(song, "arrange", plan, request=_guide, verbose=verbose)
        awarn = []
        spec = to_spec(plan, bars_of, grid_slots(ts), inst_gms, allow_drums, awarn,
                       intensity_of=intensity_of, shift_of=shift_of, drum_kit=_drum_kit(song), lead=_lead_no)
        if design.ending == "fade":
            apply_fade_out(spec, bars_of, n_bars=timeline.total_bars)
        if awarn:
            log["arrange"] += "".join(f"\n★{w}" for w in awarn)
            arrange_rec.warnings.extend(awarn)
        af = inspect_arrange(spec, bars_of, awarn, design.sections)
        # ★**やり直した後も、所見がゼロでも必ず表を読ませる**（2026-09-27・目録の関門-4）。以前は所見ゼロなら関門を飛ばし、
        #   表も渡していなかった（2026-09-16 の「所見ゼロでも必ず通す」が、やり直しの輪の中にだけ届いていなかった）
        v = ctx.judge("伴奏", log.get("arrange", ""), arrange_rec.status, arrange_rec.trouble,
                      af, attempt=_att, verbose=verbose,
                      material=_arr_material(plan, inst_gms, st.vocal_range, _drum_kit(song), shift_of,
                                             ins=getattr(song, "instrumentation", None),
                                             facts=arrange_facts(_gate_score(st, spec), bars_of, intensity_of=intensity_of)),
                      # 輪の中では前の工程へは戻らない。★ただし**最後の稿**（やり直しを使い切った・`MAX_REDO`）では、通す／弱い所を書いて進む
                      #   のほかに前の工程への差し戻しも選べる（2026-10-04 組F1・本人決定の案A。回数が残っていれば）
                      redo_from_options=_redo_options("伴奏") if _att >= MAX_REDO else [],
                      context=_ctx_line, before=_before, after=plan)
        song.record(StepRecord(step="arrange_gate", reason=v.reason, notes=v.notes or "", guidance=v.guidance or "",
                               status="ok" if v.action == "accept" else "compromised",
                               warnings=list(awarn), request=v.action))
        log["arrange_gate"] = f"{v.action}: {v.reason}"
        if _att >= MAX_REDO:
            _maybe_redo_from(v, "伴奏", song)
    st.spec, st.awarn, st.intensity_of, st.arrange_rec = spec, awarn, intensity_of, arrange_rec


def run_voices(st) -> None:
    """**声部**（ハモリ・追っかけ・合いの手を実際の音にする）。楽譜ができてから呼ぶ（`song_table` の表が要る）。

    読む: `song.harmony`（計画）・`st.score`。書く: `song.voices`・楽譜の `extra_voices`。
    """
    song, verbose, log, score = st.song, st.verbose, st.log, st.score
    # ★★**声部（ハモリ・追っかけ・合いの手）を書く**（2026-09-23）★★
    #   8番の工程で決めた計画（どの区間に・何のために・どれくらい）を、実際に鳴る音にする。
    #   **ここまで下げたのは、`song_table` の表を渡すため**——表には
    #   `小節:拍`・コード・和音での役割・ドラムの打点・**既にある声部**が並んでいて、
    #   それを見て「どこが空いているか」「何と重なるか」を決められる。
    #   ★**位置も長さも LLM が書く**。こちらは拍を tick に直すだけ
    #   （機械が均等割りしていたときは**16分の格子から22/38音が外れ**、
    #    本人に「入れ方が不自然」と言われた。書かせたら格子外ゼロ）。
    if getattr(song, "harmony", None) and song.harmony.parts:
        try:
            from harmony import voices_for                  # noqa: E402（ここでだけ使う）
            # ※機械が捨てた音・狙いを決めた音の記録（2026-09-27〜）は 2026-09-30 に外した。形式の所見が直らなければ
            #   声部の工程で止まり（`StepUnfixed`）、組み立てで捨てる音は無い（来たら `AssemblyError`）
            # 時間軸の区間（`st.bars_of`）を渡すと、狙いの区間の外に置いた音を声部の検査が見る（2026-09-30・判-16②）
            # 区間 → 移調量（`st.shift_of`＝旋律・コードを移したのと同じもの）を渡すと、声部も C 基準で書かせて区間の調へ移す（2026-09-30・記-3）
            _v, _vw, song.voices = voices_for(song.harmony, score, verbose=verbose, song=song, bars_of=st.bars_of,
                                              shift_of=st.shift_of)
            if _v:
                score.extra_voices = _v
            log["voices"] = "／".join(_vw) if _vw else "（所見なし）"
            # ★**声部は声部の記録に**（2026-09-30・出-13）。以前はハモリの計画の理由で `harmony` の記録をもう1つ作り、声部の LLM の
            #   reason・status・trouble が入らなかった（所見の欄に「声部:理由」が入り、毎曲「うまくいかなかった工程」に並んだ・判-16①）。
            #   理由は記録の reason に入る。所見に「声部:理由」の行を足すのは `harmony.write_voices` の側でやめた（同日。カバーの編曲にも効く）
            _rec(song, "voices", song.voices, warnings=_vw, verbose=verbose)
        except NOT_SWALLOW:                                     # 形式の所見が直らない・組み立ての不具合は止める（握らない・2026-09-30）
            raise
        except Exception as e:                              # 声部で曲全体を落とさない
            # ★**失敗を記録に残す**（2026-09-23）。最初は `log` に入れて `verbose` のときだけ
            #   印字していたため、**実機で声部が0本になったのに、どこにも理由が残らなかった**。
            #   `verbose=False` で走ると**黙って消える**——「計画したハモリが黙って消える」
            #   （2026-08-21）と同じ形を、新しい工程で作っていた。
            import traceback as _tb
            log["voices"] = f"（失敗: {type(e).__name__}: {e}）"
            song.record(StepRecord(step="voices", reason=log.get("harmony", ""),
                                   status="compromised",
                                   warnings=[f"★声部を書けなかった: {type(e).__name__}: {e}",
                                             _tb.format_exc()[-600:]]))
            if verbose:
                print(f"  [8.5 声部] ★失敗: {type(e).__name__}: {e}")


def run_colour(st) -> None:
    """**彩り**（対旋律・リフ）。声部のあとに呼ぶ（声が先・彩りがそこを避ける）。

    読む: `song.melody`（書いた ABC＝歌の枠）・`song.arrangement`（骨組み）・`st.score`（声部）・編成の値・`st.vocal_range`。
    書く: `song.colour`・楽譜の伴奏（骨組み＋彩りを合わせた写し。`song.arrangement` には足さない）。
    """
    song, verbose, log, score = st.song, st.verbose, st.log, st.score
    design, melody, plan = song.design, song.melody, song.arrangement
    title, ts, bars_of, timeline, bar = st.title, st.ts, st.bars_of, st.timeline, st.bar
    inst_gms, allow_drums, inst_desc = st.inst_gms, st.allow_drums, st.inst_desc
    # ★**彩りの組み立ての所見は彩りの記録に**（2026-09-30・判-17②）。以前は伴奏の段階の所見のリスト（`st.awarn`）に足していて、
    #   そのリストを伴奏の点検役の記録（`arrange_gate`）がそのまま持っていたので、彩りの所見が伴奏の記録に後から混ざった。
    #   いまは別のリストに集め、伴奏の段階に無かった所見だけを彩りの記録に入れる（骨組みから続く所見は伴奏の記録にある）
    cwarn: list[str] = []
    # ---- ⑦-2 彩り（対旋律・リフ）——**別工程にする**（2026-09-17）----
    #   ★伴奏プロンプトが 6,913文字まで膨らみ、**後から足した節ほど効かなくなっていた**。
    #     同じ Gemini・同じ依頼・同じ素材で、展開の節だけ（3,143文字）に縮めたら
    #     **対旋律 0箇所 → 4箇所 / 繰り返し「同じ」→「変えた」/ at_bars 10 → 24箇所**。
    #     だから**プロンプトを足すのではなく、工程を分けて短いプロンプトを構造で作る**。
    #   ★**歌の枠を渡すのが要**——「歌のすき間に入れる」と言葉で書いても、
    #     どこが空いているかを知らなければ置きようがない。
    #   ★★**声部のあとへ移した**（2026-09-25・本人判断）★★ 彩りも声部も「歌のすき間」を使うのに、
    #     以前は彩りが先に走り、**互いを見ていなかった**（声部の表には伴奏が載っていない）。
    #     合いの手とギターの対旋律を両方頼むと、**同じすき間で重なる**ことがありえた。
    #     **演奏系（伴奏・彩り・奏法）と声系（ハモリ・声部）が交互に並んでいた**のを系統ごとにまとめ、
    #     **声を先に決めて、彩りがそこを避ける**（ポップスでは歌が主役・楽器の飾りは歌の邪魔をしない）。
    #     奏法は**彩りのあと**のまま（彩りが足したパートにも弾き方を付けるため）。
    try:
        _frames: dict[str, str] = {}
        for _ln in (getattr(melody, "sketch", "") or "").split("\n"):
            _k, _, _v = _ln.strip().partition(": ")
            # ★区間の**全部の行**を渡す（2026-09-30・io レビューの彩りの所見3）。以前は1行目だけで、2行目以降の
            #   すき間が見えなかった。行は ` / ` でつなぐ（ABC の `|` は小節線なので使わない）
            if _k and _v:
                _frames[_k] = f"{_frames[_k]} / {_v}" if _k in _frames else _v
        # ★骨組みは伴奏の点検役と同じ表（`pattern_table`）で、**行を削らず字数で切らない**（2026-09-30・出-6）。以前は
        #   pattern と区間だけを1200字で切り、リフ・音量・音域・楽器が無く、パッドと弦の行が丸ごと見えなかった
        #   （同じピアノにリフが2本・弦が二重になった）
        _skel = pattern_table(plan, inst_gms)
        colour = add_colour(title, design.bpm, design.sections, inst_desc or "",
                            _frames, _skel, request="\n".join(ctx.carry_plain()), vocal_range=st.vocal_range,
                            voices=_voices_desc(score.extra_voices, timeline, bar),
                            chords_of=st.ch_map, time_signature=ts,     # コード表と拍子（出-7）
                            shift_of=getattr(st, "shift_of", None))                       # 根音の範囲を歌の音域と同じ C 基準で（2026-10-01）
        colour, _ = fix_colour(colour, song, bars_of=bars_of, instruments=inst_gms, slots=grid_slots(ts),
                               inst_desc=inst_desc or "", base_riff=plan.riff, verbose=verbose,
                               shift_of=getattr(st, "shift_of", None))   # 移したあとの鳴る高さも検査（2026-10-03）
        song.colour = colour
        # ★★**伴奏の持ち物に足し込まない**（2026-09-25）★★ `song.arrangement` は**骨組みのまま**保存し、
        #   彩りのパートは**合わせた写し**にだけ足して楽譜にする（工程は自分の持ち物だけを持つ）。
        #   以前は `plan` に直接足して `song.arrangement = plan` としていたので、保存される伴奏に
        #   彩りが混ざり、**彩りを作り直しても古い彩りが残り、新しい彩りが鳴らなかった**
        #   （二重を防ぐ判定 `replayed("伴奏")` が作り直しにも効いたため・2026-09-24 に入れたもの）。
        #   分けて持てば、戻しても作り直しても**合わせるのは毎回この一回だけ**になる。
        if colour.parts:
            # ★**リフはそれぞれの持ち物**（2026-09-27・目録の伴奏-5）。以前は彩りのリフが伴奏のリフを上書きしていた
            merged = merge_colour(plan, colour)
            spec = to_spec(merged, bars_of, grid_slots(ts), inst_gms, allow_drums, cwarn,
                           intensity_of=st.intensity_of, shift_of=st.shift_of, drum_kit=_drum_kit(song),
                           lead=getattr(getattr(song, "instrumentation", None), "lead", None))   # 主旋律の楽器も使った（組G3）
            # ★**彩りを合わせたあとにもフェードを当てる**（同日・音量-1）。以前は当て直しておらず、新規の曲ではほぼ効いていなかった
            if design.ending == "fade":
                apply_fade_out(spec, bars_of, n_bars=timeline.total_bars)
            score.arrangement = spec                        # 楽譜はもう組んであるので、差し替える
            st.spec = spec
        _rec(song, "colour", colour, warnings=[w for w in cwarn if w not in (st.awarn or [])], verbose=verbose)
        log["colour"] = colour.reason
        if verbose:
            _n = sum(1 for p in colour.parts for sp in p.patterns
                     if sp.pitches or sp.use_riff)
            print(f"  [7.5 彩り] {'対旋律/リフ %d箇所' % _n if _n else '置かない'}"
                  f"／リフ {'・'.join(colour.riff) or 'なし'} — {colour.reason[:90]}")
    except NOT_SWALLOW:                                         # 形式の所見が直らない・組み立ての不具合は止める（握らない・2026-09-30）
        raise
    except Exception as e:                                  # 彩りで曲全体を落とさない
        # ★**失敗を曲の記録に残す**（2026-09-30・判-17①）。以前は画面の表示（`log`）にだけ書き、記録には何も残らなかった
        #   （彩りが無いまま骨組みだけで鳴る）。声部の工程（`run_voices`）と同じ形
        import traceback as _tb
        log["colour"] = f"（失敗: {type(e).__name__}）"
        song.record(StepRecord(step="colour", reason="", status="compromised",
                               warnings=[f"★彩りを書けなかった: {type(e).__name__}: {e}——骨組みの伴奏だけで鳴る",
                                         _tb.format_exc()[-600:]]))
        if verbose:
            print(f"  [7.5 彩り] ★失敗: {type(e).__name__}: {e}")
