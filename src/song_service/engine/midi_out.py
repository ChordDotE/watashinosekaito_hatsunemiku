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

"""**伴奏を MIDI で書き出す**（音源を差し替えられるようにするための土台）。

これまで伴奏は numpy で倍音を足して直接 wav にしていた。音色が 7 種しかなく、
実音（ピアノ・ブラス・オルガン）が無いため「後ろのオルガンがほぼ毎回同じに聞こえる」と
指摘された（2026-08-21）。原因は音の作り方ではなく、**音源を選べない構造**にあった。

そこで伴奏を2段に分ける:

    編曲データ → **MIDI**（このファイル） → 音源 → wav
                                          ├ SoundFont  … render_sf.py・手作業ゼロ
                                          └ HALion     … 最高音質だが初回だけ GUI が要る
    （numpy の合成音は 2026-09-29 に消した。SoundFont で鳴らせなければ止める・`song_pipeline._render_accomp`）

MIDI を挟むと副産物として **V6 / Cubase で開いて好きな音源を当てられる**（ミクさんが
気に入らない音色を後から差し替えられる）。

ノートの置き方は `song_audio.pitched_events` を呼ぶ（置き方は1つの関数・2026-09-27）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import mido

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import song_design as compose_v2                                          # noqa: E402,F401（bootstrap で sys.path を通す）
import articulation as artic                               # 奏法のカタログ（2026-09-23）
import song_audio                                          # どの小節で鳴るかの判定はここだけ
from song_schema import SongScore                           # noqa: E402
from timbre_map import (drum_canon, drum_note, drum_program, gm_program, halion_channel,   # noqa: E402
                        kit_finding, kit_notes, parse_gm_bank)

Q = 480


# ★**判定は `song_audio._bar_match` が1つだけ持つ**（2026-09-01 に本番へ統合）。
# ここにも独自実装があり、`arrange_v2` とも本番とも微妙に食い違っていた。
# しかも**ドラム分岐だけこの判定を呼び忘れて**おり、ドラムが全小節で鳴って
# 曲の抑揚が 0.3dB になっていた。判定を1つにすれば、少なくとも食い違いは起きない。
_bar_match = song_audio._bar_match
song_bars = song_audio.song_bars                            # 小節数もここが1つだけ持つ


# ドラム(9)を除いた 15 本。パートがこれを超えたら**使い回す**（落とさない）
_CHANNELS = [c for c in range(16) if c != 9]


# ============================ 奏法（2026-09-23）============================
#
# ★**ベンドはチャンネル単位**（実測で確認）。同じチャンネルで鳴っている音は**全部**曲がる。
#   実例『季節を追い越せ』は 46パート／音色4種で、**`lead` と `chord` が同じ `gm0-31`**
#   ＝同じチャンネルだった。このままリードにフォールを掛けると**後ろの和音まで落ちる**。
#
#   そこで2段構えにする:
#     ① **和音を鳴らすパートには奏法を付けない**（そもそも奏法にならない——構成音が
#        そろって平行移動するだけ。実際の演奏でもフォールやスクープは単旋律に付く）
#     ② 奏法を持つパートは、**音色が同じでも別チャンネルへ**分ける
#
#   予算: 編成は最大8本なので 8（音色）＋ 奏法つき2〜4 ＝ 10〜12 で 15 に収まる。
#   **溢れたら奏法を落として所見に出す**（曲が壊れるより浅く鳴るほうがよい）。


def _is_chordal(part) -> bool:
    """このパートは和音を鳴らすか（**1つでも同時発音があれば和音パート**）。"""
    return any(len(st.tones) > 1 for st in part.steps)


def _artic_of(score) -> dict[int, dict[int, object]]:
    """`ArrangementSpec.articulations` を **{パート番号: {tick: NoteArtic}}** に組み直す。

    ★**奏法は音ごと**（2026-09-23）。`steps` の中ではなく、鳴る音の位置に重なっている
      ——同じパターンから鳴る音でも、**フレーズの初めの1音だけ深く**が書ける。
    """
    out: dict[int, dict[int, object]] = {}
    for a in score.arrangement.articulations:
        out.setdefault(int(a.part), {})[int(a.tick)] = a
    return out


def _depth_written(a, where: str) -> None:
    """**深さを持つ奏法の深さが空なら止める**（2026-10-04）。奏法の工程の型（`artic_plan.ArticNote`）が深さを必ず書かせるので、
    空は不具合（`AssemblyError`）。以前は空を楽器の標準（`articulation.shape` の空の読み方）で鳴らしていた。
    古い記録・楽譜の空は変換（`dev_tools/song_legacy/convert_record.py`）がその楽器の標準の深さを書き込む。"""
    if a.depth_st is None and artic.has_depth(a.name):
        from song_fix import AssemblyError                  # noqa: E402
        raise AssemblyError(f"{where}: {a.tick}tick の奏法『{a.name}』に深さが無い——奏法の工程の型が必ず書かせているはずの形でない")


def _ascii(s: str) -> str:
    """MIDI のメタ文字列は latin-1 しか通らない。日本語のトラック名で落ちるので削る。"""
    return "".join(c for c in (s or "") if ord(c) < 128).strip()[:60] or "track"


def assembly_warnings(score: SongScore) -> list[str]:
    """**書き出しの組み立てが、書かれたものを鳴らせなかった所**を返す（ファイルは書かない・2026-09-27）。

    奏法を掛けなかった（和音のパート・チャンネル不足・同時に鳴る音・楽器にできない・深さや行き先がベンドの幅を超える）、
    知らない打楽器名・キットに無い打楽器を鳴らさなかった、読めないキット名のドラムを鳴らさなかった、同じチャンネルで音量の違うパートが重なる、など。以前は print だけで**記録に残らなかった**
    （目録の 伴奏-10・打-3・打-6）。呼び手（`artic_plan.run_articulation`）が `song_fix.record_change` で曲の記録に残す。
    """
    w: list[str] = []
    score_to_midi(score, None, warns=w)
    return w


def score_to_midi(score: SongScore, path: Path | None, include_vocal: bool = False,
                  for_halion: bool = False, warns: list[str] | None = None) -> Path | None:
    """SongScore → MIDI。伴奏は1パート1トラック、ドラムは ch10。`path=None` は書き出さずに所見だけ集める。

    include_vocal=True で歌とハモリも入れる（V6 で開いて確認する用。
    **伴奏 wav を作るときは False**——歌は V6 が歌うので二重に鳴ってはいけない）。

    for_halion=True では **チャンネル＝HALion のスロット番号**にし、プログラムチェンジを出さない。
    HALion は GM のプログラムチェンジで音色を選べず、人が16スロットに仕込む方式のため。
    """
    mid = mido.MidiFile(ticks_per_beat=Q)
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(score.tempo), time=0))
    num, den = (int(x) for x in (score.time_signature or "4/4").split("/"))
    meta.append(mido.MetaMessage("time_signature", numerator=num, denominator=den, time=0))
    meta.append(mido.MetaMessage("track_name", name=_ascii(score.title) or "song", time=0))

    # **チャンネルは「楽器」ごとに割り当てる**（パートごとではない・2026-08-22）。
    #
    # 1つの楽器は**区間ごとに別々の PartSpec** に分かれている（弾き分け・音量差のため）。
    # パート単位で割り当てると、同じピアノが6本あるだけでチャンネルを6本食い、
    # 15本を超えたところで**使い回し**が始まる。使い回されたチャンネルには
    # 両方のトラックが program_change を出すので、**後勝ちで音色が入れ替わる**
    # （実測: 区間ごとの音量差を入れてパートが 23→36 に増えた途端、
    #   音量差が付かないどころか曲全体が 2.3dB 下がった）。
    # 同じ音色は同じチャンネルにまとめれば、パートがいくら分かれても増えない。
    #
    # ★**奏法を持つパートは、音色が同じでも別チャンネルへ**（2026-09-23）。
    #   ベンドはチャンネル単位なので、同居していると和音まで曲がる。
    #
    # ★**役（`role`）が違えば、音色が同じでも別チャンネルへ**（2026-09-27）。パートの音量を CC11 で出すように
    #   したので（下の `_expression_events`）、同じ音色の `lead` と `chord` が1チャンネルに同居すると、
    #   **片方の音量でもう片方の鳴っている音まで動く**。同じ役の区間ごとの分割（弾き分け・強さ・フェード）は
    #   時間が重ならないので同じチャンネルのままでよい。役ごとに分けると 15 本を超える曲だけ、従来どおり音色でまとめる。
    w = warns if warns is not None else []
    artic_at = _artic_of(score)                            # {パート番号: {tick: NoteArtic}}
    _pitched = [p for p in score.arrangement.parts
                if not (p.role == "drums" or p.drum_steps)]
    # ★**主旋律の楽器は専用のチャンネル**（2026-10-03・§5-2）。歌と同じ旋律を単音で弾くので奏法（ベンド）も音量（CC11）も
    #   チャンネル単位で効く——伴奏のパートと同居させない。伴奏はその残りの 14 本で割り当てる（主旋律の無い曲は今までどおり 15 本）
    lead = score.lead if score.lead.notes else None       # 主旋律の楽器が無い曲は中身が空（`LeadPart.is_empty`・F1）
    free = _CHANNELS[1:] if lead is not None and not for_halion else _CHANNELS
    by_role = len({(p.timbre or p.role, p.role) for p in _pitched}) <= len(free) - 2   # 奏法の別チャンネルに余白を残す
    def _base_key(part) -> str:
        key = part.timbre or part.role
        return f"{key}\x00{part.role}" if by_role else key

    # ★**奏法の別チャンネルが、後ろの音色のチャンネルを奪わない**（2026-10-03・§2 c7）。以前の網（`len(ch_of) >= len(free) - 1`）は
    #   その時点の数しか見ないので、前のパートの奏法が別チャンネルを取ったあとに新しい音色が続くと 15 本を超え、
    #   チャンネルを使い回して音色が入れ替わった（楽器 10 本でも、奏法つきの単音のパートが前に多いと起きえた）。
    #   1回目は以前と同じ割り当て（使い回しが起きない曲は何も変わらない）。使い回しが起きる曲だけ、全部の音色のチャンネルを
    #   先に取っておき、残りがあるときだけ奏法の別チャンネルを足す（足りなければ奏法を落とす＝以前の網と同じ扱い）
    def _assign(reserve: set[str]) -> tuple[dict[str, int], dict[int, bool], dict[int, str], list[str]]:
        ch_of: dict[str, int] = {}
        art_ok: dict[int, bool] = {}                       # パート番号 → 奏法を鳴らすか
        ch_key_of: dict[int, str] = {}                     # パート番号 → チャンネルの引き当て名
        aw: list[str] = []
        for pi, part in enumerate(score.arrangement.parts):
            if part.role == "drums" or part.drum_steps:
                continue
            key = _base_key(part)
            use_art = False
            if artic_at.get(pi):
                need = len(ch_of) + len(reserve - set(ch_of)) + (0 if f"{key}\x00bend" in ch_of else 1)
                if _is_chordal(part):
                    # ★**和音パートには付けない**。付けると構成音がそろって平行移動するだけで
                    #   グリッサンドにもフォールにもならない（実際の演奏でも単旋律に付くもの）。
                    aw.append(f"{part.role}: 和音を鳴らすパートなので奏法を付けなかった"
                              f"（ベンドはチャンネル単位なので、和音だと全部の音が一緒に動く）")
                elif len(ch_of) >= len(free) - 1 or (reserve and need > len(free)):
                    # ★最後の網。**溢れたら奏法を落とす**（曲が壊れるより浅く鳴るほうがよい）
                    aw.append(f"{part.role}: チャンネルが足りないので奏法を付けなかった"
                              f"（音色 {len(ch_of)} 種で {len(free)} 本を使い切った）")
                else:
                    key = f"{key}\x00bend"                 # 同じ音色でも別チャンネルにする
                    use_art = True
            art_ok[pi] = use_art
            if key not in ch_of:
                ch_of[key] = free[len(ch_of) % len(free)]
            ch_key_of[pi] = key
        return ch_of, art_ok, ch_key_of, aw

    ch_of, art_ok, ch_key_of, _aw = _assign(set())
    if len(ch_of) > len(free):
        ch_of, art_ok, ch_key_of, _aw = _assign({_base_key(p) for p in _pitched})
    w.extend(_aw)
    done_pc: set[int] = set()                              # program_change を出し終えたチャンネル
    _kit = score.arrangement.drum_kit
    kit_prog = drum_program(_kit)
    kit_have: dict[str, tuple[int, str]] | None = None     # キットで鳴らせる打楽器と、そのキットでの番号（None＝キットで絞れない→GM の番号のまま）
    drop_drums = False                                     # ドラムを鳴らさない（キット名が読めない）
    if for_halion:
        kit_prog = None                                    # HALion は人が仕込む（キットで絞らない）
    elif _kf := kit_finding(_kit):
        # ★**読めないキット名ならドラムは鳴らさない**（2026-09-29・組み立ては補正しない）。以前は音源の既定のキット
        #   （standard）で代わりに鳴らしていた（目録の 打-6）。編成の検査が先に所見で返し、残ったら編成の記録に残っている
        drop_drums = True
        if any(p.drum_steps for p in score.arrangement.parts):
            w.append(_kf.split("。")[0] + "。ドラムは鳴らさなかった（代わりのキットで鳴らさない）")
    else:
        kit_have = kit_notes(_kit)
    _odd_drums: set[str] = set()                           # 知らない打楽器名・キットに無い打楽器（鳴らさなかったもの）
    done_rpn: set[int] = set()                             # ベンド幅を設定し終えたチャンネル
    done_vol: set[int] = set()                             # チャンネル音量（CC7）を出し終えたチャンネル
    pending: list[tuple] = []                              # (トラック, 音, チャンネル, 音量, ベンド, 手書きの音, 役)

    first_of_role: dict[str, int] = {}                     # 楽器ごとの最初のパート（手で書いた小節の置き場）
    for pi, part in enumerate(score.arrangement.parts):
        first_of_role.setdefault(part.role, pi)
    for pi, part in enumerate(score.arrangement.parts):
        is_drum = part.role == "drums" or bool(part.drum_steps)
        events: list[tuple[int, int, int, int]] = []       # (start, key, dur, vel)
        bends: list[tuple[int, int, list, int]] = []       # (start, dur, 折れ点, その音の高さ)
        # ★**小節数はコードの個数ではなく実際の小節数**（2026-09-06 修正）。
        #   `chords_per_bar` は1小節に2つ置ける（`"Am|F"`）ので、両者は一致しない
        #   ——実例でコード42個／実際は29小節。ここは **V6 へ渡す MIDI** を作る側なので、
        #   ずれると「伴奏が設計より早く終わる／ドラムだけ曲より長く鳴る」がそのまま歌に乗る。
        n_bars = song_bars(score)
        bar_ticks = num * Q * 4 // den

        if is_drum and drop_drums:
            continue
        if is_drum:
            for bar_idx in range(n_bars):
                # ★**ドラムにも bar_filter を効かせる**（2026-08-22 修正）。
                # ここに判定が無かったため、ドラムだけ**全小節で鳴っていた**。
                # 「Aメロはドラムを抜く」という編曲の指示が丸ごと無視され、
                # 区間ごとに叩き分けを分割すると**全パターンが同時に鳴る**状態になっていた。
                # 症状: 楽譜上は 24dB の起伏があるのに、鳴らすと 3dB しか差が出ない。
                # 音程パートは判定していたので、**ドラムだけが例外的に漏れていた**。
                if not _bar_match(bar_idx, n_bars, part.bar_filter):
                    continue
                t0 = bar_idx * bar_ticks
                # ★決まった形のフィルで小節を差し替えることはしない（2026-09-27・打-2）
                for h in part.drum_steps:
                    note = drum_note(h.inst)
                    if note is None:
                        # ★**知らない打楽器名は鳴らさない**（2026-09-28・組み立ては補正しない）。以前は部分一致かシェイカーの
                        #   近い音で鳴らしていた。伴奏の検査が先に所見で返し、`song_arrange._grid_drums` も落とすので、
                        #   ここへ来るのは手で作った楽譜だけ（打-3）
                        if h.inst not in _odd_drums:
                            _odd_drums.add(h.inst)
                            w.append(f"知らない打楽器名『{h.inst}』は鳴らさなかった")
                        continue
                    if kit_have is not None and drum_canon(h.inst) not in kit_have:
                        # ★**キットに無い打楽器は鳴らさない**（2026-09-29）。以前はキットの別の番号へ運び（`kit_remap`・
                        #   ハイハット→27 番）、運び先が無ければ番号のまま（オーケストラ用キットではティンパニ）鳴らしていた。
                        #   伴奏の工程にはキットにある打楽器だけを選択肢に見せ、検査が先に所見で返している
                        if h.inst not in _odd_drums:
                            _odd_drums.add(h.inst)
                            w.append(f"打楽器『{h.inst}』はキット『{_kit}』に無いので鳴らさなかった（キットの別の音で代わりに鳴らさない）")
                        continue
                    if kit_have is not None:
                        # ★**そのキットでその楽器が入っている番号で鳴らす**（2026-09-29・`kit_notes`）。オーケストラ用キットの
                        #   ハイハットは 27 番。伴奏の工程はキットの中身（サンプル名つき）を見て選んでいる
                        note = kit_have[drum_canon(h.inst)][0]
                    events.append((t0 + int(h.at_beat * Q), note, Q // 8, h.velocity))
        else:
            # ★**置き方は `song_audio.pitched_events` が1つだけ持つ**（2026-09-27）。ここにも同じ組み立てがあり、
            #   **書いた高さ（`tone.key`）をここだけ捨てて**対旋律・リフが根音の連打になっていた。
            #   転回形・分数コードの低音・1小節2和音の拍位置も、あちらの規則そのまま。
            # ★奏法は**音の位置（tick）で引く**（2026-09-23）。パターンの中ではないので、
            #   同じ `StepEvent` から鳴る音でも**1つずつ別に扱える**。
            mine = artic_at.get(pi, {}) if art_ok.get(pi) else {}
            prog = gm_program(part.timbre, part.role)
            for start, dur, placed, vel in song_audio.pitched_events(score, part):
                events.extend((start, key, dur, vel) for key in placed)
                # ---- 奏法（この音の中でどう動くか）----
                a = mine.get(start)
                if a is not None and len(placed) == 1:
                    _depth_written(a, part.role)
                    pts = artic.shape(a.name, prog, a.depth_st)
                    if pts is None:                        # 楽器にできない・深さがベンドの幅を超える（頭打ちにしない・2026-09-28）
                        why = artic.why_not(a.name, prog, a.depth_st)
                        if why not in w:                   # 同じ理由は1回だけ
                            w.append(f"{part.role}: {why}")
                    else:
                        bends.append((start, dur, pts, placed[0]))
                elif a is not None:
                    w.append(f"{part.role}: {start}tick は同時に{len(placed)}音鳴るので"
                             f"『{a.name}』を掛けなかった")
        # ★**人が手で書いた小節はパターンより優先**（2026-09-26・`ArrangementSpec.hand_bars`）。その小節の
        #   パターンの音を消し、その楽器の最初のトラックに手で書いた音を置く（同じ楽器が区間ごとに分かれているため）
        hand: list = []
        hb = [h for h in score.arrangement.hand_bars if h.get("role") == part.role]
        if hb:
            spans_ = [(int(h["start_bar"]) * bar_ticks, int(h["end_bar"]) * bar_ticks) for h in hb]
            inside = lambda tk: any(s0 <= tk < s1 for s0, s1 in spans_)   # noqa: E731
            events = [ev for ev in events if not inside(ev[0])]
            bends = [bd for bd in bends if not inside(bd[0])]
            if first_of_role.get(part.role) == pi:
                hand = [tuple(n) for h in hb for n in h.get("notes") or []]
        if not events and not hand:
            continue

        tr = mido.MidiTrack()
        mid.tracks.append(tr)
        tr.append(mido.MetaMessage("track_name", name=_ascii(part.role) or "part", time=0))
        if is_drum:
            use_ch = 9                                     # ch10（0始まりで9）がドラム
            # ★**ドラムの種類を音色に出す**（2026-09-27）。編成が `orchestra` と決めても ch10 に音色の
            #   切り替えを出しておらず、ティンパニのつもりの低いタム（当時の名前 tom_low・いまの GM の名前 low_tom）が普通のタムで鳴っていた。
            #   音源（GS 準拠）のオーケストラ用キットは 41〜53 番がティンパニ。HALion は人が仕込むので出さない。
            #   キットごとに鍵盤の中身が違うので、キットに無い打楽器は上のドラムの組み立てで鳴らさずに落としてある（`kit_drums`）。
            if kit_prog is not None and 9 not in done_pc:
                tr.append(mido.Message("program_change", channel=9, program=kit_prog, time=0))
                done_pc.add(9)
        elif for_halion:
            use_ch = halion_channel(part.timbre, part.role) - 1        # スロット番号がそのままチャンネル
        else:
            use_ch = ch_of.get(ch_key_of.get(pi, part.timbre or part.role), _CHANNELS[0])
            if use_ch not in done_pc:                       # 音色指定は**チャンネルにつき1回だけ**
                bank, _ = parse_gm_bank(part.timbre)
                if bank:                                   # 音源固有の追加音色（bank 0 以外）
                    tr.append(mido.Message("control_change", channel=use_ch,
                                           control=0, value=bank, time=0))
                tr.append(mido.Message("program_change", channel=use_ch,
                                       program=gm_program(part.timbre, part.role), time=0))
                done_pc.add(use_ch)
            if bends and use_ch not in done_rpn:
                # ★**ベンド幅を広げる**（RPN 0,0）。既定の ±2半音では
                #   フォール（-7）もグリッサンドも入らない。実測で効くことを確認済み。
                for ctrl, val in ((101, 0), (100, 0), (6, artic.BEND_RANGE), (38, 0)):
                    tr.append(mido.Message("control_change", channel=use_ch,
                                           control=ctrl, value=val, time=0))
                done_rpn.add(use_ch)
        if use_ch not in done_vol:                         # チャンネル音量は全開に（パートの音量は CC11 で出す）
            tr.append(mido.Message("control_change", channel=use_ch, control=7, value=_CC7, time=0))
            done_vol.add(use_ch)
        pending.append((tr, events, use_ch, part.gain,
                        _bend_msgs(bends, events, w, part.role) if bends else None, hand, part.role))
    if lead is not None:
        pending.append(_lead_track(mid, lead, for_halion, w))
    _write_all(pending, w)

    if include_vocal and score.vocal_notes:
        # ★`ch_i` が未定義で、歌を入れると `NameError` で落ちていた（2026-09-27 修正）。楽器の後ろのチャンネルを使う
        #   （主旋律の楽器が先頭のチャンネルを使っていれば、そのぶん後ろへ）
        ch_i = len(ch_of) + (1 if free is not _CHANNELS else 0)
        vpend: list[tuple] = []
        for title, notes, gain in [("vocal", score.vocal_notes, 1.0)] + [
                (v.title, v.notes, 10 ** (v.gain_db / 20)) for v in score.extra_voices]:
            tr = mido.MidiTrack()
            mid.tracks.append(tr)
            tr.append(mido.MetaMessage("track_name", name=_ascii(title) or "voice", time=0))
            vch = _CHANNELS[ch_i % len(_CHANNELS)]
            tr.append(mido.Message("program_change", channel=vch, program=53, time=0))  # Voice Oohs
            tr.append(mido.Message("control_change", channel=vch, control=7, value=_CC7, time=0))
            vpend.append((tr, [(n.start_tick, n.key, n.duration_tick, 100) for n in notes], vch, gain,
                          None, [], title))
            ch_i += 1
        _write_all(vpend, w)

    if path is None:                                       # `assembly_warnings`（所見だけ集める）
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))
    return path


#: 主旋律の楽器のトラック名（伴奏のパートの役 `lead` と取り違えないよう、別の名前にする）
LEAD_TRACK = "main_melody"
#: 主旋律の楽器の強さ（velocity）。旋律の工程は強さを書かないので、歌を MIDI に入れるとき（`include_vocal`）と同じ一定の値で弾く
#: （組み立ての決まり。音量は強さではなくパートの音量＝伴奏の `lead_gain` を CC11 で出す）
LEAD_VELOCITY = 100


def _lead_track(mid: mido.MidiFile, lead, for_halion: bool, w: list[str]) -> tuple:
    """**主旋律の楽器のトラック**（2026-10-03・§5-2）。伴奏の MIDI の1トラックとして書く（書き出しは `score_to_midi` の1か所）。

    音は `SongScore.lead.notes`（旋律の音そのまま・書いた区間の音だけ）、音色は `lead.timbre`、音量は `lead.gain`（CC11）。
    チャンネルは専用（`_CHANNELS[0]`・伴奏はその残り）。奏法は `lead.articulations`（伴奏の奏法と同じ書き出し・`_bend_msgs`）。
    戻り値は `_write_all` に渡す1行。
    """
    events = [(int(n.start_tick), int(n.key), int(n.duration_tick), LEAD_VELOCITY) for n in lead.notes]
    prog = gm_program(lead.timbre, LEAD_TRACK)
    bends: list[tuple[int, int, list, int]] = []
    at = {s: (d, k) for s, k, d, _v in events}
    for a in lead.articulations or []:
        hit = at.get(int(a.tick))
        if hit is None:                                    # 奏法の工程が表の音を指す（`artic_plan._check`）ので来ない
            w.append(f"{LEAD_TRACK}: {a.tick}tick に音が無いので『{a.name}』を掛けなかった")
            continue
        _depth_written(a, LEAD_TRACK)
        pts = artic.shape(a.name, prog, a.depth_st)
        if pts is None:
            why = artic.why_not(a.name, prog, a.depth_st)
            if f"{LEAD_TRACK}: {why}" not in w:
                w.append(f"{LEAD_TRACK}: {why}")
            continue
        bends.append((int(a.tick), hit[0], pts, hit[1]))
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("track_name", name=LEAD_TRACK, time=0))
    if for_halion:
        use_ch = halion_channel(lead.timbre, LEAD_TRACK) - 1
    else:
        use_ch = _CHANNELS[0]
        bank, _ = parse_gm_bank(lead.timbre)
        if bank:
            tr.append(mido.Message("control_change", channel=use_ch, control=0, value=bank, time=0))
        tr.append(mido.Message("program_change", channel=use_ch, program=prog, time=0))
        if bends:
            for ctrl, val in ((101, 0), (100, 0), (6, artic.BEND_RANGE), (38, 0)):
                tr.append(mido.Message("control_change", channel=use_ch, control=ctrl, value=val, time=0))
    tr.append(mido.Message("control_change", channel=use_ch, control=7, value=_CC7, time=0))
    return (tr, events, use_ch, lead.gain, _bend_msgs(bends, events, w, LEAD_TRACK) if bends else None, [], LEAD_TRACK)


def _raw(st: float) -> int:
    """半音 → `pitchwheel` の生値。**幅は RPN で広げた `BEND_RANGE` 前提**。"""
    return int(max(-8192, min(8191, round(st / artic.BEND_RANGE * 8192))))


#: 曲げたまま音が終わったとき、**0 へ戻すのをどれだけ待つか**（tick）。
#: ★`note_off` と同時に戻すと**リリースの余韻が跳ね上がる**（2026-09-23 に実測で踏んだ）。
#: 次の音がこれより近ければ、その直前で戻す。
_RESET_DELAY = Q // 2


def _bend_msgs(bends: list[tuple[int, int, list, int]],
               events: list[tuple[int, int, int, int]],
               warns: list[str], role: str) -> list[tuple[int, int]]:
    """奏法 → **(tick, 生値)** の列。

    ★**同時に他の音が鳴っている所では掛けない**。ベンドはチャンネル単位なので、
      巻き添えになる。和音パートは既に弾いてあるが、**単音パート同士が重なる**ことはある
      （同じ音色の `lead` が2本など）。ここが最後の砦。

    ★**次の音の高さ**は `events` から引く（`to_next` の行き先）。
    """
    starts = sorted({s for s, _k, _d, _v in events})
    out: list[tuple[int, int]] = []
    skipped = 0
    too_far = 0
    for start, dur, pts, key in bends:
        # 同時に鳴っている他の音（自分以外）があれば掛けない
        if any(s < start + dur and s + d > start and not (s == start and k == key)
               for s, k, d, _v in events):
            skipped += 1
            continue
        nxt = next((s for s in starts if s > start), None)
        next_st = None
        if nxt is not None:
            nk = next((k for s, k, _d, _v in events if s == nxt), None)
            if nk is not None:
                next_st = float(nk - key)
        # ★**次の音への滑り（`to_next`）の行き先がベンドの幅を超えたら掛けない**（2026-09-28・組み立ては補正しない）。
        #   以前は生値が ±8192 に張り付き、次の音ではなく 12半音先で止まって鳴った。検査は `artic_plan._check`
        if next_st is not None and abs(next_st) > artic.MAX_DEPTH and any(p.to_next for p in pts):
            too_far += 1
            continue
        for tick, st in artic.expand(pts, dur, next_st=next_st):
            out.append((start + tick, _raw(st)))
        if artic.ends_bent(pts):
            # ★曲がったまま終わる奏法は、**次の音の直前**まで待ってから戻す
            back = start + dur + _RESET_DELAY
            if nxt is not None:
                back = min(back, nxt - 1)
            out.append((max(start + dur, back), 0))
        else:
            out.append((start + dur, 0))
    if skipped:
        warns.append(f"{role}: {skipped}箇所の奏法を掛けなかった"
                     f"（同じ時に他の音が鳴っていて、ベンドが巻き添えになるため）")
    if too_far:
        warns.append(f"{role}: {too_far}箇所の次の音へ滑る奏法を掛けなかった"
                     f"（次の音まで{artic.MAX_DEPTH:g}半音を超え、ベンドの幅に入らない）")
    merged: dict[int, int] = {}
    for tick, raw in sorted(out):
        merged[tick] = raw                                 # 同じ tick は後勝ち
    return sorted(merged.items())


# ============================ パートの音量（2026-09-27）============================
#
# ★**音量は強さ（velocity）と別に出す**（目録の 音量-3）。以前は `vel * gain / 0.7` を強さに掛けていたので、
#   gain が 0.81 を超えると `X`（110）が127に張り付いて**アクセントの差が消え**、GM では強さで音色も変わった。
#   いまは**強さは書いたとおり**、パートの音量は**チャンネルの CC11（エクスプレッション）**で出す。
#   チャンネル音量（CC7）は全開（127）にして、CC11 = gain × 100/0.7 とする——gain 0.7 で
#   以前の「CC7 既定 100 × 強さそのまま」と同じ大きさ（GM の音量カーブは CC7・CC11・強さとも同じ形）。
#   gain がそれを超える曲だけ、**比を保ったまま**いちばん大きいパートが 127 になるよう全体を縮める。
#
#   CC11 はチャンネルに1つなので、**同じチャンネルで音量の違うパートが重なる**と、重なっている間は
#   後から入ったパートの音量で鳴る。役ごとにチャンネルを分けて避け（`score_to_midi`）、それでも起きたら所見に出す。
_CC7 = 127
_EXPR_PER_GAIN = 100 / 0.7


def _expr_scale(gains: list[float]) -> float:
    """gain → CC11 の倍率。いちばん大きい gain が 127 を超えるときだけ、比を保って縮める。"""
    top = max((g for g in gains if g > 0), default=0.0)
    return min(_EXPR_PER_GAIN, 127 / top) if top > 0 else _EXPR_PER_GAIN


def _expr(gain: float, scale: float) -> int:
    return max(0, min(127, int(round(max(0.0, gain) * scale))))


def _expression_events(pending: list[tuple], warns: list[str]) -> list[list[tuple[int, int]]]:
    """トラックごとの CC11 の置き場 `[(tick, 値), …]`。**チャンネルごとに**、音量の違う音が
    鳴り始める所でだけ値を変える（トラックは別でも CC はチャンネル単位で効くので、チャンネルで数える）。"""
    scale = _expr_scale([p[3] for p in pending])
    out: list[list[tuple[int, int]]] = [[] for _ in pending]
    by_ch: dict[int, list[tuple[int, int, int, int]]] = {}  # ch → (開始, 終わり, トラック番号, 値)
    for i, (_tr, events, ch, gain, _b, hand, _role) in enumerate(pending):
        v = _expr(gain, scale)
        for start, _k, dur, _v in list(events) + list(hand or []):
            by_ch.setdefault(ch, []).append((int(start), int(start) + int(dur), i, v))
    seen: set[tuple[str, str]] = set()
    for ch, notes in by_ch.items():
        notes.sort(key=lambda n: (n[0], n[2]))
        cur = None
        for s, _e, i, v in notes:
            if v == cur:
                continue
            out[i].append((s, v))
            if cur is not None:                            # 別の音量の音がまだ鳴っていれば、巻き添えになる
                for s2, e2, j, v2 in notes:
                    if s2 > s:
                        break
                    # 同じ役どうし（区間ごとの分割・フェード）は、フェーダーを動かしたのと同じなので所見にしない
                    if j != i and v2 != v and s2 <= s < e2 and pending[j][6] != pending[i][6]:
                        pair = tuple(sorted((str(pending[j][6]), str(pending[i][6]))))
                        if pair not in seen:
                            seen.add(pair)
                            warns.append(f"{pair[0]} と {pair[1]} は同じチャンネル（ch{ch + 1}）で重なって鳴り、"
                                         f"音量が違う。音量（CC11）はチャンネルに1つなので、重なっている間は"
                                         f"後から入ったほうの音量で鳴る")
                        break
            cur = v
    return out


def _write_all(pending: list[tuple], warns: list[str]) -> None:
    """集めたトラックを、CC11（パートの音量）を差し込んで書き込む。"""
    ccs = _expression_events(pending, warns)
    for (tr, events, ch, _gain, bends, hand, _role), cc in zip(pending, ccs):
        _write(tr, events, ch, bends=bends, raw=hand, expr=cc)


def _write(tr: mido.MidiTrack, events: list[tuple[int, int, int, int]], ch: int,
           bends: list[tuple[int, int]] | None = None,
           raw: list[tuple[int, int, int, int]] | None = None,
           expr: list[tuple[int, int]] | None = None) -> None:
    """(start,key,dur,vel) の配列を delta-time の on/off 列に変換して書き込む。

    ★**強さは書いたとおり**（2026-09-27）。パートの音量は `expr`（tick, CC11 の値）で別に出す。
    `bends` は (tick, `pitchwheel` の生値)。**音符と同じ時間軸に混ぜて並べる**。
    """
    msgs: list[tuple[int, int, str, int, int]] = []         # (tick, 優先, 種別, key, vel)
    for start, key, dur, vel in events:
        msgs.append((start, 1, "note_on", key, max(1, min(127, int(vel)))))
        msgs.append((start + dur, 0, "note_off", key, 0))   # 同時刻は off を先に
    # 人が手で書いた音も、強さをそのまま書く（2026-09-26）
    for start, key, dur, vel in (raw or []):
        msgs.append((start, 1, "note_on", key, max(1, min(127, int(vel)))))
        msgs.append((start + dur, 0, "note_off", key, 0))
    for tick, raw_v in (bends or []):
        # ★**ベンドは note_on より先**（優先 0 より小さい）。音が鳴り始めた後に
        #   最初の値が来ると、頭の一瞬だけ素の高さで鳴る。
        msgs.append((tick, -1, "pitchwheel", raw_v, 0))
    for tick, val in (expr or []):
        msgs.append((tick, -2, "expression", val, 0))      # 音量は鳴り始めより先に
    msgs.sort(key=lambda m: (m[0], m[1]))
    prev = 0
    for tick, _, kind, key, vel in msgs:
        dt = max(0, tick - prev)
        if kind == "pitchwheel":
            tr.append(mido.Message("pitchwheel", channel=ch, pitch=key, time=dt))
        elif kind == "expression":
            tr.append(mido.Message("control_change", channel=ch, control=11, value=key, time=dt))
        else:
            tr.append(mido.Message(kind, channel=ch, note=key, velocity=vel, time=dt))
        prev = tick
