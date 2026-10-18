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

"""既存スコアを参照（音高＋音価）に整合させる（決定論的・UC-2 の参照一致）。

各音節に参照の音高と音価（リズム）を移す。音符数が一致すれば 1:1、ズレれば DTW で対応付け。
参照に複数音が対応する音節（メリスマ）は音価を合算して伸ばす。音高は自作キーへ移調して合わせる。
→ 「歌詞と音符の対応がズレる」を構造的に解消（音高だけでなくリズムも本家どおりにする）。
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent

import song_llm
from lyric_slots import parts_of                            # 1つの音に当てたかなの読み方（新しい曲と同じ1か所）
from lyric_slots import kana_text_chars, kana_text_lines    # カバーの歌詞のかなの字と行（改行＝元の歌詞の行の切れ目・2026-10-03）
from machine_eval import best_match
from song_schema import TICKS_PER_QUARTER as Q
from song_schema import SongNote, SongScore
from song_table import spell_pitch                          # LLM に見せる音名は国際式・表と同じ綴り（2026-10-01）
from song_checks import JsonBool, JsonFloat, JsonInt  # noqa: E402（LLM が書く数と真偽の欄の型・2026-10-02 判断5）
from song_tuning import apply_by_gap   # 調声（カバーにも当てる）


def load_reference(p: Path | str) -> dict:
    """参照 JSON → {notes:[(pitch,dur)], starts:[beat]|None, breaks:[idx], beats_per_bar:int}。"""
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    raw = d.get("notes") or [[m, 1.0] for m in d["midi"]]
    notes = [(int(e[0]), float(e[1])) for e in raw]
    starts = [float(e[2]) for e in raw] if raw and len(raw[0]) > 2 else None
    return {"notes": notes, "starts": starts,
            "breaks": d.get("phrase_breaks", [0]), "beats_per_bar": d.get("beats_per_bar", 4)}


# ※参照の音高と音価を 1:1／DTW で移す `align_to_reference`（と、その入力を読む `load_reference_notes`）は、
#   道具から外した（2026-09-27・目録の道具-3）あと呼ばれていなかったので、2026-09-28 に消した。
#   カバーで歌詞を参照の音符へ当てるのは `align_to_reference_jp`。


def _bar_and_at(score: SongScore):
    """楽譜の1小節の tick（曲の拍子から）と、tick → 位置 `小節:拍` の書き手（道具の返事と同じ `song_edits.at_text`）。

    ★2026-10-02: 以前は所見の位置が楽譜の頭からの拍（`@12.5拍`）で、小節線も 4/4 決め打ち（`beats_per_bar=4`）で見ていた。
      所見は `evaluate_song`・`fix_song_timing`・`check_consistency` の返事に出て、音楽担当が `小節:拍` で書く直す道具へ
      書き写すので、表と同じ `小節:拍` で出す（仕様書 §16「道具の返事」・§17 ず-18 の残り）。
    """
    from song_edits import at_text, score_bar_ticks        # noqa: E402（song_design を読むので呼ぶ時に）
    bar = score_bar_ticks(score)
    return bar, (lambda t: at_text(t, bar))


def check_timing(score: SongScore, *, grid: float = 0.25, bar_head: bool = False) -> list[str]:
    """スコアの timing 自己評価（決定論）。問題点を文章で返す（空＝問題なし）。位置は `小節:拍`、小節線は曲の拍子で見る。

    ★**歌い出しが小節線からずれているかは、既定では見ない**（2026-09-26・本人判断「意図して置いたなら、
      機械的な移動はスキップした方がいい」）。いまのオリジナル曲は、音の位置を**設計の時間軸と16分の格子**から
      機械が置くので、小節線からずれた歌い出しは**設計の意図**（拍の裏から入る・弱起）。これを「伴奏と食い違う」と
      NG にし、`fix_timing` が歌全体を半拍前へ動かしていた——『明日への足音』は16.5拍目から入る設計なのに、
      v1・v3 で**歌全体が伴奏より半拍前にずれた**（伴奏は動かないので全部の音がずれる）。
      この検査が要ったのは、LLM が書く休みの長さの誤差で位置が累積してずれていたころ（2026-09-07 以前）。
      `bar_head=True` でだけ見る（格子を通らない古い作り方の曲のため）。
    """
    issues, notes = [], score.vocal_notes
    if not notes:
        return ["ボーカルが空"]
    bar, at = _bar_and_at(score)
    # ★**曲頭の「間」は問題にしない**（2026-09-01）。0.1拍でも空いていたら指摘していたが、
    #   **歌い出しの前に少し置くのは意図した表現**——「本当に0秒目から声が始まるのはどうだろう、
    #   0.2〜0.3秒空けてもいい」という指摘を受けて、作曲側が1拍ほど置くようにしてある。
    #   それを毎回「問題」と見なし、`fix_song_timing` が潰していた（**そのとき調声も消えた**）。
    #
    # ★**さらに 2026-09-07 に「長すぎる空き」の判定も外した**。
    #   「2小節を超えたら遅い」としていたが、**イントロが8小節ある曲では正常な位置が必ず引っかかる**。
    #   そして `fix_timing` がそれを潰し、**設計したイントロが消えていた**
    #   （45項目点検で「イントロ 0.0秒」として検出）。
    #   **小節線に乗っているなら、それは意図した歌い出し**（イントロぶん待っている）。
    #   直すべきなのは**小節線から半端にずれている**ときだけ——そのずれは伴奏との食い違いになる。
    head_off = notes[0].start_tick % bar
    if bar_head and head_off and min(head_off, bar - head_off) > grid * Q:
        issues.append(f"歌い出しが小節線からずれている（@{at(notes[0].start_tick)}）——伴奏と食い違う")
    prev_end = 0
    for i, n in enumerate(notes):
        if abs(n.start_tick / (grid * Q) - round(n.start_tick / (grid * Q))) > 0.05:
            issues.append(f"note{i}「{n.lyric}」@{at(n.start_tick)} が{grid}拍グリッド外")
        if i > 0 and n.start_tick < notes[i - 1].start_tick + notes[i - 1].duration_tick:
            ov = notes[i - 1].start_tick + notes[i - 1].duration_tick - n.start_tick
            issues.append(f"note{i}「{n.lyric}」@{at(n.start_tick)} が前の音（note{i - 1}「{notes[i - 1].lyric}」"
                          f"@{at(notes[i - 1].start_tick)}）と{ov / Q:g}拍重なり")
        prev_end = n.start_tick + n.duration_tick
    return issues  # fix_timing で必ず解消できる項目に限定（ループ収束保証）


def _retime(notes: list, off: int, g: int, lo: int | None, hi: int | None,
            moved: list[str] | None, who: str, at) -> list:
    """`check_timing` が挙げる2つだけを直す（2026-09-27・目録の道具-2）: **格子の外の開始**と**重なり**。

    - 格子の外で始まる音 … 開始と終わりを格子へ寄せる（終わりが開始以下になるなら長さはそのまま）
    - 重なり … **前の音を次の音の開始までに縮める**（同じ位置に並んだときだけ後ろの音を1格子後ろへ）
    - それ以外の音（格子の上の音）は**長さも位置も触らない**。以前は全音符の長さを16分へ丸めていた
    `lo`〜`hi`（tick）の外の音は触らない（範囲の外の音を縮めたときは、そう書く）。
    動かした音の位置は `at`（tick → `小節:拍`・`_bar_and_at`）で書く（2026-10-02。以前は楽譜の頭からの拍）。長さは拍のまま。
    """
    def inside(t: int) -> bool:
        return (lo is None or t >= lo) and (hi is None or t < hi)

    def say(msg: str) -> None:
        if moved is not None:
            moved.append(f"{who}{msg}")

    pos: list[list[int]] = []                              # [開始, 長さ]（off を引いたあと）
    for i, n in enumerate(notes):
        st, du = n.start_tick - off, n.duration_tick
        if inside(n.start_tick) and abs(st / g - round(st / g)) > 0.05:
            nst, nend = int(round(st / g)) * g, int(round((st + du) / g)) * g
            ndu = nend - nst if nend > nst else du
            say(f"note{i}「{n.lyric}」を {at(st)}→{at(nst)} へ（格子の外）"
                + (f"・長さ {du / Q:g}→{ndu / Q:g}拍（終わりも格子へ）" if ndu != du else ""))
            st, du = nst, ndu
        pos.append([st, du])
    for i in range(1, len(pos)):
        a, b = pos[i - 1], pos[i]
        if b[0] >= a[0] + a[1] or not (inside(notes[i].start_tick) or inside(notes[i - 1].start_tick)):
            continue
        out = "" if inside(notes[i - 1].start_tick) else "（範囲の外）"
        if b[0] > a[0]:
            say(f"note{i - 1}「{notes[i - 1].lyric}」{out}を {a[1] / Q:g}→{(b[0] - a[0]) / Q:g}拍に縮めた"
                f"（note{i} と重なるため）")
            a[1] = b[0] - a[0]
        else:                                              # 同じ位置（か前）に並んだ＝後ろの音を1格子後ろへ
            say(f"note{i}「{notes[i].lyric}」を {at(b[0])}→{at(a[0] + g)} へ、"
                f"note{i - 1}{out}を {g / Q:g}拍に縮めた（同じ位置に並んだため）")
            a[1], b[0] = g, a[0] + g
    return [n.model_copy(update={"start_tick": st, "duration_tick": du})
            for n, (st, du) in zip(notes, pos)]


def fix_timing(score: SongScore, *, grid: float = 0.25,
               bar_head: bool = False, start_beat: float | None = None, end_beat: float | None = None,
               moved: list[str] | None = None) -> SongScore:
    """`check_timing` が見つけた問題を決定論的に直す。

    直すのは **①拍グリッド外で始まる音 ②音の重なり**。**③小節線からの半端なずれは `bar_head=True` のときだけ**
    （2026-09-26。既定では歌全体を動かさない——理由は `check_timing` の説明。設計で拍の裏から入る歌を
    半拍前へずらし、伴奏と食い違わせていた）。
    ★**曲頭を 0 にはしない**（2026-09-07 変更）——イントロは設計が決めたものなので消さない。
    ★**格子の上の音の長さは丸めない・重なりは前の音を縮める**（2026-09-27・目録の道具-2）。
      以前は**全音符の長さを16分へ丸め**、重なりは後ろの音を押し出していた（スタッカートが伸び、弱起がずれた）。
      `start_beat`〜`end_beat` を渡すとその範囲だけ直す（スウィングなど意図して外した区間を避けるため）。
      `moved` を渡すと、**動かした音を1件ずつ**書く（道具がコーディネーターへ返す）。
    """
    g = int(grid * Q)
    notes = [n.model_copy() for n in score.vocal_notes]
    if not notes:
        return score
    lo = None if start_beat is None else int(round(start_beat * Q))
    hi = None if end_beat is None else int(round(end_beat * Q))
    # ★**0 へ引き寄せない**（2026-09-07 修正）。それまで `off = notes[0].start_tick` として
    #   歌い出しを tick 0 へ移していたが、これは**設計したイントロを消す**動きだった
    #   （イントロ8小節の曲で「イントロ 0.0秒」になった）。
    #   さらに移動量が小節の整数倍でないと（実測 32.5拍・16.5拍＝8.125・4.125小節）、
    #   **歌全体が伴奏の小節線に対して半端にずれる**。
    #   直すのは**小節線からの半端なずれだけ**——イントロぶんの待ちはそのまま残す。
    bar_t, at = _bar_and_at(score)                         # 小節線は曲の拍子で（2026-10-02。以前は 4/4 決め打ち）
    off = notes[0].start_tick % bar_t if bar_head else 0   # 小節線からのはみ出しぶんだけ詰める（指定時のみ）
    if off > bar_t / 2:                                    # 半分を超えているなら次の小節線へ送る
        off -= bar_t
    if off and moved is not None:
        moved.append(f"歌とハモリの全体を {-off / Q:+g}拍 動かした（歌い出しを小節線へ・bar_head）")
    # ★★**作り直さず、位置と長さだけ差し替える**（2026-09-23）。
    #   ここは**同じ穴に2回落ちた場所**——2026-09-01 に `tuning` を写し忘れて
    #   427/462 音の表情が消え、2026-09-23 に**新しく足した `artic`（歌の崩し）が
    #   同じように消えた**（工程は「12か所決めた」と言っているのに楽譜は 0 か所）。
    #   **欄を足すたびに構築箇所を直す作りだと、必ずまた漏れる。**
    #   `model_copy` なら**何が増えても勝手に運ばれる**。直すのはタイミングだけ（`_retime`）。
    fixed = _retime(notes, off, g, lo, hi, moved, "歌: ", at)

    # ★**ハモリも同じだけ動かす**（2026-09-07 修正）。
    #   ここは `vocal_notes` だけを `off` ぶん前へ寄せていたので、
    #   **ハモリが元の位置に取り残され、主旋律から `off` ぶんずれて鳴っていた**。
    #   実害: 主旋律が 37.8秒で終わったあと、**ハモリだけが 47.9秒まで鳴り続け**、
    #   主旋律より 6dB 低いぶん「小さいぼそぼそ声」として聴こえた（本人指摘・2026-09-06）。
    #   ずれ幅は補正量そのもの——BPM200 で 32.5拍＝9.75秒に対し、実測 +10.1秒で一致した。
    #   **ハモリは主旋律に重ねるもの**なので、動かすなら必ず一緒に動かす。
    voices = []
    for v in getattr(score, "extra_voices", None) or []:
        vn = _retime([n.model_copy() for n in v.notes], off, g, lo, hi, moved, f"{v.title}: ", at)
        drop = [n for n in vn if n.start_tick < 0]         # 補正で頭が負になる音は落とす（落としたと書く）
        if drop and moved is not None:
            moved.append(f"{v.title}: 曲頭より前へ出た {len(drop)}音を落とした（bar_head で全体を動かしたため）")
        voices.append(v.model_copy(update={"notes": [n for n in vn if n.start_tick >= 0]}))

    return score.model_copy(update={"vocal_notes": fixed,
                                    "extra_voices": voices,
                                    "lyrics_full": "".join(n.lyric for n in fixed)})


class _SylMap(BaseModel):
    lyric: str = Field(description="音節（確認用）")
    # ★2026-10-04（組N2）: pitch_advance・breath_after も既定（1・false）をやめて音節ごとに必ず書かせる
    pitch_advance: JsonInt = Field(ge=1, le=5,
        description="この音節の音高を得るため、前の音節から参照の音高を何音進めるか（**音節ごとに必ず書く**）。"
                    "1=次の音。日本語では原曲の装飾音/経過音を歌わず1音にまとめる事がある→飛ばす分だけ2以上")
    beats: JsonFloat = Field(gt=0, description="この音節の長さ（拍）。日本語の実際の歌い回しで決める")
    breath_after: JsonBool = Field(description="この音節の後に息継ぎ（無音）を入れるか（**音節ごとに必ず書く**。true / false）")


class _JpAlignOut(BaseModel):
    reasoning: str = Field(description="日本語の歌い回し・装飾音の扱い・伸ばし/息継ぎの方針")
    syllables: list[_SylMap]


_jp_agent: Agent[None, _JpAlignOut] = Agent(
    song_llm.pick_model("ALIGN", "flash"), output_type=_JpAlignOut,
    system_prompt=(
        "あなたは日本語の歌唱指導の専門家です。既知の曲を『原曲の音高を保ちつつ、日本語の実際の歌い回し』で"
        "初音ミクに歌わせます。\n"
        "与えるのは歌詞（音節順）と原曲の音高列（装飾音/経過音を含むインスト由来・／＝フレーズ境界）。\n"
        "各音節について決める:\n"
        "・pitch_advance: 前の音節から参照の音高を何音進めてこの音節の音高にするか。"
        "**日本語版は原曲の装飾音/経過音を歌わず1音にまとめる事が多い**→歌わない音はその分 advance を増やして飛ばす。\n"
        "・beats: この音節を何拍歌うか（**日本語『曲名』の実際の歌い回し**で。フレーズ末は長く伸ばす）。\n"
        "・breath_after: フレーズの大きな切れ目だけ true（伸ばして繋ぐ所は false）。\n"
        "音高は原曲に忠実、リズム（拍と区切り）は日本語の歌唱に合わせる。syllables は歌詞と同数・同順。"
    ),
)


def align_to_reference_jp(score: SongScore, ref_notes: list[tuple[int, float]], *,
                          song_title: str = "", phrase_breaks: list[int] | None = None,
                          starts: list[float] | None = None, transpose: bool = True,
                          findings: list[str] | None = None) -> SongScore:
    """参照=音高＋“実際の音符位置（精密な1文字ごとのタイミング）”。LLM=日本語で歌わない装飾音のスキップ選択のみ。

    transpose=False で参照の音高をそのまま使う（faithful-copy 等、参照＝歌うメロディの時）。
    `findings` を渡すと、LLM の割り当ての過不足を機械が埋めた／捨てたことを書く。
    """
    ours = [n.key for n in score.vocal_notes]
    shift = best_match(ours, [p for p, _ in ref_notes])[0] if transpose else 0
    ref_pitch = [p + shift for p, _ in ref_notes]
    ref_dur = [d for _, d in ref_notes]
    brk = set(phrase_breaks or [])
    lyrics = [n.lyric for n in score.vocal_notes]

    # 歌う長さ＝onset間隔(IOI)。starts があればそれ（スタッカート音価より正確）、無ければ音価。長い音＝伸ばす音。
    if starts and len(starts) == len(ref_pitch):
        sung = [round(starts[i + 1] - starts[i], 2) for i in range(len(ref_pitch) - 1)] + [round(ref_dur[-1], 2)]
    else:
        sung = [round(d, 2) for d in ref_dur]
    parts = []
    for i, p in enumerate(ref_pitch):
        if i in brk and i != 0:
            parts.append("／")
        parts.append(f"{spell_pitch(p)}:{sung[i]}")                        # 音名:歌う長さ(拍)（2026-10-01 まで階名）
    prompt = (f"曲名: {song_title or '（不明）'}\n"
              f"歌詞（音節順・{len(lyrics)}個）: {' '.join(lyrics)}\n"
              f"原曲の音列（{len(ref_pitch)}音・各音「音名:歌う長さ(拍)」・音名は国際式（`C4`＝中央のド）・／=フレーズ境界）:\n{' '.join(parts)}\n"
              f"各音節について pitch_advance（この音節が乗る参照音。日本語で歌わない装飾音/経過音はその分飛ばす）"
              f"と breath_after を決めて。最終タイミングは原曲の音符位置を使うが、"
              f"**割り当ては各音の『歌う長さ』を必ず考慮**すること：長い音（伸ばす音）には伸ばして自然なモーラ"
              f"（語尾や開いた母音）を当て、子音中心のモーラや語頭を長い音に置かない。短い音の連続は1音1モーラを基本に。"
              f"歌詞のフレーズと参照の／を対応させ、{len(lyrics)}音節ぴったりに。")
    from song_usage import run_counted                      # noqa: E402（費用に入れる・2026-10-08）
    out = run_counted(_jp_agent, "カバーの歌詞の割り当て", prompt).output

    syls = list(out.syllables)[:len(lyrics)]
    # ★**足りない・余った割り当てを黙って埋めない／捨てない**（2026-09-27・目録の道具-11）。
    #   直しはするが（音節の数は歌詞で決まっている）、何をしたかを `findings` に書いて返す。
    if len(out.syllables) != len(lyrics) and findings is not None:
        findings.append(
            f"割り当ての数が歌詞と合わない（歌詞{len(lyrics)}音節・割り当て{len(out.syllables)}）→ "
            + (f"足りない {len(lyrics) - len(out.syllables)}音節（{''.join(lyrics[len(out.syllables):])[:40]}）は"
               "機械が「次の参照音へ1つ進む・1拍」で置いた" if len(out.syllables) < len(lyrics)
               else f"余った {len(out.syllables) - len(lyrics)}件を捨てた"))
    while len(syls) < len(lyrics):
        syls.append(_SylMap(lyric=lyrics[len(syls)], pitch_advance=1, beats=1.0, breath_after=False))
    # 装飾音スキップで「歌う参照ノート」を選択
    sel, cur = [], -1
    for syl in syls:
        cur = min(len(ref_pitch) - 1, cur + max(1, syl.pitch_advance))
        sel.append(cur)
    # 各音節の onset = 参照の実位置（精密）。無ければ音価の累積で代用
    g = Q // 4
    if starts and len(starts) == len(ref_pitch):
        pos = [int(round(starts[i] * Q)) for i in sel]
    else:
        pos, t = [], 0
        for i in sel:
            pos.append(t)
            t += max(g, int(round(ref_dur[i] * Q)))
    off = pos[0]
    notes = []
    for k, (ly, i) in enumerate(zip(lyrics, sel)):
        st = pos[k] - off
        if notes and st < notes[-1].start_tick + g:
            st = notes[-1].start_tick + g
        if k < len(sel) - 1 and not syls[k].breath_after:
            dur = max(g, (pos[k + 1] - off) - st)          # 次の音まで伸ばす（スキップした装飾音ぶんも保持）
        else:
            dur = max(g, int(round(ref_dur[i] * Q)))       # 息継ぎ or 最終音は自分の音価
        notes.append(SongNote(lyric=ly, key=ref_pitch[i], start_tick=st, duration_tick=dur))
    # ★**MIDI 参照のカバーはここを通る**（2026-09-02 追加）。`align_cover.py` の3経路だけ
    #   調声を足して、**この関数を見落としていた**——実機で『きらきら星』を作って 0/84 音で発覚。
    #   「同じことをする場所が複数ある」型を、私自身が半分だけ直していた。
    apply_by_gap(notes, tempo=score.tempo)
    return score.model_copy(update={"vocal_notes": notes,
                                    "lyrics_full": "".join(n.lyric for n in notes)})


#: **1つの音符に書いたかなの読まれ方**（2026-09-29）。読み方は `lyric_slots.parts_of` の1か所で、新しい曲の詞と同じ。
#: カバーの割り当て（ここと `align_cover` の音声の割り当て）の指示に同じ文を入れる
CELL_RULE = (
    "1つの音符に2〜3かなを書くと、その音符を等分して1かなずつ歌い分けます。"
    "ただし**長音**（同じ母音が続く『かあ』『にい』と、オ段＋う『きょう』）は前の音のまま1つで伸ばします"
    "（言い直しません）。長音に見える2つの音節（『言い』の『い』『い』）を1つの音符で歌い分けるときは"
    "`い~い` のように `~` でつなぎます。拗音『きゃ』は2字で1音です。")

#: **歌詞の行の切れ目は材料**（2026-10-03 本人決定・仕様書 §2「カバーの歌詞のかな」）。カバーの割り当て（ここと `align_cover` の
#: 音声の割り当て）の指示に同じ文を入れる。機械は行の切れ目で割り当てを強制しない
LYRIC_LINES_RULE = (
    "歌詞は**元の歌詞の行ごと**に『歌詞の行N:』を付けて示します（行の番号と改行は歌詞の字ではありません。割り当てには書かない）。"
    "**歌詞の行の切れ目と、旋律のフレーズの切れ目（音列の／・音の間）をなるべく合わせてください**——行の切れ目は判断の材料で、"
    "合わないところ（1行を2フレーズで歌う・2行を続けて歌う）は実際の歌い方を優先します。")


class _NoteAssign(BaseModel):
    reasoning: str = Field(description="フレーズと歌詞の対応、どこを1音符2かな（速い箇所）/1かな/伸ばしにしたかの方針")
    mora_per_note: list[str] = Field(
        description="各音符で歌うかな（音符数ぴったり）。1音符に1〜3かな。順に全歌詞を使い切る。"
                    "長い音符（伸ばす音）は1かな（伸ばして自然な母音）。日本語が音符より多い箇所は1音符に2かな。"
                    "日本語で歌わない装飾音/経過音は空文字 \"\"。" + CELL_RULE)


def assign_count_type(base: type[BaseModel], n: int) -> type[BaseModel]:
    """**頼んだ音符の数を知っている、歌詞の割り当ての型**（2026-10-02・洗い出し H9 と中。前例は `align_cover.corrected_type`）。
    `mora_per_note` の数が音符の数 `n` と違えば型で差し戻す（pydantic_ai が LLM に書き直させ、直らなければ呼ぶ側が
    `StepUnfixed`「カバーの歌詞の割り当て」で止める）。記号整列（`_NoteAssign`）と音声での割り当て（`align_cover._AudioAlign`）が使う。

    ★以前は数が違うと、余った割り当てを捨て・足りない音符を空（歌わない）で埋めていた（記号整列は所見に書いて、
    音声での割り当ては何も残さずに）。"""
    from pydantic import model_validator

    class _ForCount(base):                                  # type: ignore[valid-type,misc]
        @model_validator(mode="after")
        def _same_count(self):
            got = len(self.mora_per_note)
            if got != n:
                raise ValueError(f"mora_per_note: {got}個ある——音符は {n}個。1つの音符に1つ（歌わない音符は空文字 \"\"）で、"
                                 f"同じ{n}個・時間順に返す（{'余り' if got > n else '足りない分'}を出さない）")
            # ★**1音ぶんのかなに、かなと `~` 以外の字を書かない**（2026-10-02 本人決定・案B）。以前は読み手（`parts_of`）が
            #   空白・`_` `*` を黙って落とし、ほかの字（`！`・漢字など）はそのまま VOCALOID の歌詞にしていた
            from song_lyrics import kana_cell_problems     # noqa: E402（字の範囲は詞と同じ1つ）
            bad = kana_cell_problems(self.mora_per_note)
            if bad:
                raise ValueError("mora_per_note: " + "／".join(bad[:8]))
            return self

    _ForCount.__name__ = base.__name__                      # LLM に見える型の名前は変えない
    return _ForCount


def assign_or_stop(agent, prompt, base: type[BaseModel], n: int, label: str):
    """割り当ての LLM を数を知っている型で呼ぶ。型の差し戻しが直らなければ `StepUnfixed`（`label`）で止める。"""
    try:
        from song_usage import run_counted                  # noqa: E402（費用に入れる・2026-10-08）
        return run_counted(agent, label, prompt, output_type=assign_count_type(base, n)).output
    except Exception as e:
        if song_llm.output_unfixed(e):                     # 形式の所見（数）が直らない → 止める（整列の道具が所見つきで返す）
            raise song_llm.to_step_unfixed(e, label, step=label) from e
        raise


_assign_agent: "Agent[None, _NoteAssign]" = Agent(
    song_llm.pick_model("ALIGN", "flash"), output_type=_NoteAssign,
    system_prompt=(
        "あなたは日本語の歌唱指導の専門家です。既知の曲の歌詞（全文・かな）を、原曲メロディの音符に割り当てます。\n"
        "重要: 日本語は英語より音数が多く、原曲が英語だと『1音符に2かな』が普通に起きます"
        "（例: 3音符の旋律に6かなの語＝各音符2かな）。逆に長い音符（伸ばす音）は1かなを伸ばして歌います。\n"
        "各音符に、その音符で歌うかな（1〜3字、伸ばす音は1字、歌わない装飾音は空文字）を割り当て、"
        "**歌詞全文を順番どおり過不足なく使い切る**こと。\n"
        + LYRIC_LINES_RULE + "\n"
        + CELL_RULE + "\n"
        "出力 mora_per_note の長さは音符数ぴったり。"
    ),
)


def lyric_lines_text(lyrics_kana: str) -> str:
    """割り当ての LLM に見せる**歌詞の行**（2026-10-03 本人決定）。元の歌詞の行の切れ目（改行1つ・仕様書 §2）を、
    旋律のフレーズ境界 `／` と混ざらないよう**行ごとに `歌詞の行N:` を付けて**見せる。行の切れ目は材料で、機械は割り当てを強制しない。"""
    return "\n".join(f"歌詞の行{i}: {ln}" for i, ln in enumerate(kana_text_lines(lyrics_kana), 1))


def assign_findings(assign: list[str], kana: list[str]) -> list[str]:
    """**割り当てたかなが歌詞と一致するか**（落ち・重なり）の所見（2026-09-27・目録の道具-11。中身の照合なので型にはしない）。
    記号整列（`align_full_lyrics_jp`）と音声での割り当て（`align_cover.align_full_lyrics_audio`）が同じここで照らす
    （2026-10-03・再レビュー。以前は音声の経路が照らしていなかった）。`kana` は行の切れ目を除いた歌詞の字（`kana_text_chars`）。"""
    used = "".join(c for cell in assign for c in cell if c != "~")   # `~`（詰めの印）は歌詞の字ではない（空白は型が差し戻す）
    if used == "".join(kana):
        return []
    return [f"割り当てたかな（{len(used)}字）が歌詞（{len(kana)}字）と一致しない——"
            "歌詞の一部が落ちた／重なった可能性（歌詞どおりに歌われていない）"]


def note_assign_findings(assign: list[str], kana: list[str], *, log: list | None, findings: list[str] | None,
                         tag: str) -> list[str]:
    """`assign_findings` の所見を `log`（★つき）と `findings`（`align_vocal` の返事に出る `assign_findings`）へ書く。"""
    found = assign_findings(assign, kana)
    for f in found:
        if log is not None:
            log.append(f"{tag} ★{f}")
        if findings is not None:
            findings.append(f)
    return found


def split_cell(cell: str) -> list[str]:
    """1つの音符に当てたかな → その音符を等分して鳴らし分けるもの（`lyric_slots.parts_of`・長音は等分しない）。"""
    return list(parts_of(cell))


def align_full_lyrics_jp(ref_pitch: list[int], sung: list[float], breaks: list[int],
                         lyrics_kana: str, *, song_title: str = "",
                         starts: list[float] | None = None, tempo: float = 120.0,
                         log: list | None = None, findings: list[str] | None = None) -> SongScore:
    """全歌詞（かな）を音符に割り当てる。1音符に複数かな（音符分割）可＝日本語の詰め込みに対応。

    starts（各音符の開始拍）で精密タイミング。長い音符=伸ばし、速い箇所=1音符2かな。
    log を渡すと割当の思考・mora_per_note を追記（追跡用）。
    `findings` を渡すと、割り当ての過不足（音符が余った＝歌わない・歌詞が余った＝落ちた）を書く。
    """
    # ★**歌詞のかなは書いたとおり読む**（2026-10-02 本人決定・案B）。かな以外（空白・句読点・漢字・記号）は歌詞を書く LLM の型
    #   （`align_cover` の `_SecKana` ほか・`song_lyrics.kana_text_problems`）が差し戻す。以前はここで空白・改行・`、。「」` を黙って落とし、
    #   ほかの字はそのまま音符に当てていた。
    # ★**改行は定義された区切り＝元の歌詞の行の切れ目**（2026-10-03 本人決定・仕様書 §2）。字を数えるときは除き（`kana_text_chars`）、
    #   LLM には行ごとに見せる（`lyric_lines_text`・材料。行の切れ目で割り当てを強制しない）
    kana = kana_text_chars(lyrics_kana)
    brk = set(breaks or [])
    parts = []
    for i, p in enumerate(ref_pitch):
        if i in brk and i != 0:
            parts.append("／")
        parts.append(f"{spell_pitch(p)}:{round(sung[i], 2)}")                # 音名（国際式・2026-10-01 まで階名）
    prompt = (f"曲名: {song_title or '（不明）'}\n"
              f"歌詞（かな・全{len(kana)}字・順番厳守。元の歌詞の行ごと）:\n{lyric_lines_text(lyrics_kana)}\n"
              f"原曲の音列（{len(ref_pitch)}音・各音「音名:歌う長さ(拍)」・音名は国際式（`C4`＝中央のド）・／=フレーズ境界）:\n{' '.join(parts)}\n"
              f"各音符に歌うかなを割り当てて（mora_per_note・長さ{len(ref_pitch)}）。"
              f"歌詞{len(kana)}字を順に過不足なく使い切る。長い音は1かな、速い連続は1音符2かな。")
    # ★**割り当ての数は型で差し戻す**（2026-10-02・洗い出しの中。`assign_count_type`）。以前（2026-09-27〜）は余った割り当てを捨て・
    #   足りない音符を空（歌わない）で埋め、そのことを所見に書いていた（直さずに機械が切り詰め・埋めていた）
    out = assign_or_stop(_assign_agent, prompt, _NoteAssign, len(ref_pitch), "カバーの歌詞の割り当て")
    if log is not None:
        log.append(f"[align] 入力かな({len(kana)}字): {''.join(kana)}")
        log.append(f"[align] 思考: {out.reasoning}")
        log.append(f"[align] mora_per_note({len(out.mora_per_note)}): {out.mora_per_note}")
    if len(out.mora_per_note) != len(ref_pitch):
        from song_fix import AssemblyError
        raise AssemblyError(f"カバーの歌詞の割り当てに、型を通ったはずの数の違う割り当てが来た"
                            f"（{len(out.mora_per_note)}個・音符は {len(ref_pitch)}個）")
    assign = list(out.mora_per_note)
    # 割り当てたかなが歌詞と一致しない（落ちた・重なった）ことは所見にして返す（2026-09-27・目録の道具-11。中身の照合なので型にはしない）
    note_assign_findings(assign, kana, log=log, findings=findings, tag="[align]")
    g = Q // 8
    pos = starts if (starts and len(starts) == len(ref_pitch)) else None
    notes = []
    for i, cell in enumerate(assign):
        # ★**1文字ずつではなく、鳴らし分けるものごとに割る**（2026-09-29）。以前は1文字ずつ等分していたので、
        #   「かあ」の「あ」を新しい音として言い直し、「きゃ」を「き」「ゃ」の2音に割っていた。読み方は新しい曲と同じ1か所
        ms = split_cell(cell)
        if not ms:
            continue
        on = pos[i] if pos else float(sum(sung[:i]))
        nxt = (pos[i + 1] if (pos and i + 1 < len(pos)) else on + sung[i])
        span = max((nxt - on) / len(ms), 0.05)
        for j, c in enumerate(ms):
            st = int(round((on + j * span) * Q))
            notes.append(SongNote(lyric=c, key=ref_pitch[i],
                                  start_tick=st, duration_tick=max(g, int(round(span * Q)))))
    off = notes[0].start_tick if notes else 0
    for n in notes:
        n.start_tick -= off
    return SongScore(title=song_title, tempo=tempo, time_signature="4/4",
                     lyrics_full="".join(n.lyric for n in notes), vocal_notes=notes, chords=[])


def align_score_lyrics(score: SongScore) -> SongScore:
    """既存スコアの歌詞を、自分の旋律（音高＋リズム）へ共通アライナで分割整列し直す。

    オリジナル/カバー共通で「1音符に複数かな（音符分割）」を効かせるための入口。
    歌詞数=音符数なら実質1:1（簡単曲は不変）、歌詞が多ければ速い箇所で詰める。
    """
    notes = score.vocal_notes
    if len(notes) < 2 or not score.lyrics_full:
        return score
    ref_pitch = [n.key for n in notes]
    starts = [n.start_tick / Q for n in notes]
    sung = [round(starts[i + 1] - starts[i], 2) for i in range(len(notes) - 1)] + \
           [round(notes[-1].duration_tick / Q, 2)]
    breaks = [0] + [i for i in range(1, len(notes))
                    if (notes[i].start_tick - (notes[i - 1].start_tick + notes[i - 1].duration_tick)) / Q >= 0.5]
    aligned = align_full_lyrics_jp(ref_pitch, sung, breaks, score.lyrics_full,
                                   song_title=score.title, starts=starts, tempo=score.tempo)
    apply_by_gap(aligned.vocal_notes, tempo=score.tempo)                          # ★調声
    return score.model_copy(update={"vocal_notes": aligned.vocal_notes,
                                    "lyrics_full": aligned.lyrics_full})


