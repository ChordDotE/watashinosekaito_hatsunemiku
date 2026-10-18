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

"""本番形プロトタイプ: compose パイプライン（validate→vpr→伴奏→仮歌→mix）。

本実装の音楽 coordinator（バックグラウンド非同期）に相当する手続き。
LLM 部分（song_plan_compose/melody_compose）は呼び出し側が SongScore を渡す形で代替。
"""
from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np

import song_files                                           # 曲フォルダの名前の決まり（完成品・ログ・印・形の持ち物）は1か所
from build_lipsync import write_lipsync
from harmony import MELODY_LO, SING_HI, SING_LO             # 音域の基準は harmony.py に1つだけ置く（主旋律の下端は別の値・2026-10-05）
from render_base import SR, load_wav, write_wav
from song_schema import TICKS_PER_QUARTER as Q
from song_schema import SongScore
from score_to_vpr import MIKU_V6_COMP_ID, score_to_vpr

# 曲の置き場は **`miku/data/songs/`**（2026-09-01 移植で移した・本人決定）。
# それまでは `song_service/output/`＝**ソースツリーの中**に成果物を書いていた。
# 実データ（既存の曲）の移動は別途。ここはこれから作る曲の行き先。
OUT_ROOT = Path(__file__).parents[3] / "data" / "songs"


# ---------- score_validate（機械検証）----------
def score_validate(score: SongScore) -> list[str]:
    # ★高さは国際式の音名で書く（`C4`＝60・曲の表と同じ綴り・2026-10-01）。以前は MIDI 番号（`key=79`・`推奨 58-86`）
    from song_table import spell_pitch                      # noqa: E402（LLM に見せる高さの綴りは1つ）
    # ★**主旋律と声部で下端が違う**（2026-10-05）。声部は D3（`SING_LO`）まで、主旋律は歌いやすい帯域の A#3（`MELODY_LO`）まで
    rng = f"{spell_pitch(SING_LO)}-{spell_pitch(SING_HI)}"
    mel_rng = f"{spell_pitch(MELODY_LO)}-{spell_pitch(SING_HI)}"
    issues = []
    notes = sorted(score.vocal_notes, key=lambda n: n.start_tick)
    for i, n in enumerate(notes):
        if not (MELODY_LO <= n.key <= SING_HI):
            issues.append(f"note[{i}] {n.lyric}: 音域外 {spell_pitch(n.key)}（推奨 {mel_rng}）")
        # 既知曲の実歌詞は「ん」「っ」を含むので許可。長音『ー』と単独小書き母音のみ警告
        if any(c in n.lyric for c in "ーぁぃぅぇぉ"):
            issues.append(f"note[{i}] {n.lyric}: 長音/小書き母音は別ノートにしない方が安全")
        if i > 0 and n.start_tick < notes[i - 1].start_tick + notes[i - 1].duration_tick:
            issues.append(f"note[{i}] {n.lyric}: 前ノートと時間的に重なり")

    # ★**ハモリ（extra_voices）も見る**（2026-09-06 追加）。それまで主旋律しか検証しておらず、
    #   **ハモリが 35音中32音も音域外なのに所見が1件も出ていなかった**
    #   （H48 の試聴で「サビ後半のハモリが聴こえない」と分かって発覚。実測で他区間より 20dB 低い）。
    #   ハモリは機械が主旋律の3度上へ機械的に置くので、**主旋律が高いと必ず外へ出る**。
    #   `harmony.build_harmony` が下ハモへ折り返すようにしたが、**検証も別に要る**
    #   ——折り返しても収まらない場合や、他の経路で作られた声部を取りこぼさないため。
    for v in score.extra_voices:
        over = [n for n in v.notes if not (SING_LO <= n.key <= SING_HI)]
        if over:
            ks = sorted(n.key for n in over)
            # ★事実だけを書く（2026-10-03 本人方針「機械は数えるだけ・判断は点検役」）。以前は8割を超えると「実質鳴らない」と断じていた
            issues.append(f"{v.title}: {len(over)}/{len(v.notes)}音が推奨音域（{rng}）の外"
                          f"（外の音は {spell_pitch(ks[0])}〜{spell_pitch(ks[-1])}）")

    issues += _check_in_key(score)
    return issues


# ---------- 伸ばす音は「落ち着く音」に置かれているか ----------
# ★本人の指摘（2026-09-09）:「**伸ばす長さはランダムではなくて、
#   落ち着きがよいところに配置しないとだよね**」
#
# 実測して裏を取った（RWC-P＝J-POP 100曲・35,588音・コード注釈つき）:
#
# | 音の長さ | 中央 | **鳴っている間コード構成音だった割合** |
# |---|---|---|
# | いちばん短い（〜p25）| 0.14秒 | 61.0% |
# | 短め | 0.23秒 | 62.2% |
# | 長め | 0.34秒 | 61.0% |
# | **いちばん長い（p75〜）**| **0.62秒** | **69.8%** |
#
# **いちばん長い音だけが跳ね上がる**（+8.7ポイント）。
# 短い〜長めは横並びなので、「伸ばすときだけ落ち着く音を選ぶ」という工夫が出ている。
#
# ★**測り方に注意**: 出だしのコードだけで判定すると**逆の結果**が出る（長い音ほど構成音でない）。
#   長い音は**複数のコードにまたがる**ので、**鳴っている時間のうち何割が構成音か**で測る。
#   出だしで非構成音・伸びた先で構成音になるのはサスペンション（かかり音）で、実在の技法。
#
# 判定は**相対**にする（絶対値は作風で動くため）——
# 「長い音のほうが、短い音より構成音率が高い」ことだけを見る。
# ★**2026-10-03 に楽譜の検証から外した**（本人了承「観点の整理」）。ここは「長い音のほうがコードの音の割合が低い」ときだけ
#   「落ち着いていない（J-POP は +8.7）。伸ばす音はコード構成音に置く」と**帯と直し方つきで断じていた**。同じ数え方の事実は
#   旋律の点検役の材料（`song_melody_step._tone_share`・長い音と短い音がコードの音で鳴る時間の割合・伸ばす音ごとの場所）に毎回出し、
#   判断は点検役に委ねる。上の実測（RWC-P）は、その材料の裏づけとして残す。


# ---------- 歌が伴奏と同じ調にいるか ----------
# ★**「作った形」ではなく「出来た形」を測る**（2026-09-07 追加）。
#   実際に起きた事故: `adjust_song` が歌だけを −6半音動かし、
#   **歌 A♭メジャー／伴奏 D メジャー**という三全音違いの曲が出荷された。
#   142音中109音（77%）がコードと半音でぶつかっていたのに、
#   既存の検査（音域・歌詞・重なり）も `review_progress`（密度・音符長・尺）も**全部素通り**した。
#   本人には「リズムに乗ってない」と聞こえた——**音程の事故は、聴くまで誰も気づけない。**
#
#   測り方: **拍頭の音がコード構成音である割合**。ヒット曲（POP909）は 61〜79%。
#   調が合っていればどんな作風でも 40% は超える（偶然でも約40%）。
#   **下回るのは「合っていない」ときだけ**なので、作風を縛らずに事故だけを捕まえられる。
def _chord_tones(chord: str) -> set[int]:
    """コード名 → 構成音のピッチクラス集合（読めない綴りは空集合＝判定から除く）。分数コードは分子だけ見る。

    ★**読み方は和音の組み立て（`song_audio.chord_notes`）の1つ**（2026-10-01・判-K の続き）。以前はここに
      別の読み手があり、種類の中の変化記号・テンション（`C7b9`・`C7#11`・`Cm7b5`・`C6`・`Cadd9`・`C+`）を読まなかった——
      鳴る和音には ♭9 があるのに、曲の表の「和音での役割」と調の検査は三和音＋7度だけで見ていた（読み手が2本で片方だけ直る形）。
    """
    from song_audio import chord_notes                      # 和音の読み手は1つ（組み立てと同じ）
    return {k % 12 for k in chord_notes(chord)}


def _check_in_key(score: SongScore) -> list[str]:
    spans = sorted(score.chords, key=lambda c: c.start_tick)
    if not spans or not score.vocal_notes:
        return []

    def tones_at(tick: int) -> set[int]:
        for c in spans:
            if c.start_tick <= tick < c.start_tick + c.duration_tick:
                return _chord_tones(c.chord)
        return set()

    on = hit = 0
    for n in score.vocal_notes:
        if n.start_tick % Q:                               # 拍頭の音だけを見る
            continue
        t = tones_at(n.start_tick)
        if not t:
            continue
        on += 1
        hit += n.key % 12 in t
    if on < 8:                                             # 標本が少なすぎる＝判定しない
        return []
    rate = hit / on

    # **12通り全部を測って、いまの調が最良かを見る。**
    # 割合の絶対値だけで判定すると、**5度ずれ（±7半音）を見逃す**——5度は共通音が多く、
    # 完全にずれていても 40% を超えてしまう（実測: C-Am-F-G の上で +7 して 44%）。
    # 「合っているか」ではなく**「動かしたほうがよく合うか」**を訊けば、作風によらず事故だけが出る。
    # 調が正しい曲は、どちらへ動かしても良くならない。
    def rate_at(sh: int) -> float:
        return sum(1 for n in score.vocal_notes
                   if n.start_tick % Q == 0 and tones_at(n.start_tick)
                   and (n.key + sh) % 12 in tones_at(n.start_tick)) / on

    best, best_rate = 0, rate
    for sh in range(1, 12):
        r = rate_at(sh)
        if r > best_rate:
            best, best_rate = sh, r
    gain = best_rate - rate
    if rate >= 0.40 and gain < 0.25:
        return []
    where = (f"拍頭の音がコード構成音である割合 {rate * 100:.0f}%（{hit}/{on}・"
             f"ヒット曲 61〜79%・偶然でも約40%）")
    # **2種類を混ぜない。** 動かせば直るのが「調のずれ」、動かしても直らないのは「旋律の書き方」。
    # 取り違えると、直らない曲に移調を繰り返して**さらに壊す**（それが今回の事故の入口だった）。
    if best and gain >= 0.25:
        return [f"★歌が伴奏と**違う調**で鳴っている: {where}。"
                f"**歌を {best - 12 if best > 6 else best:+d}半音 動かすと {best_rate * 100:.0f}% になる**。"
                f"`adjust_song(transpose=)` でこの量だけ動かせば直る"]
    return [f"★旋律がコードに支えられていない: {where}。"
            f"**移調では直らない**（どう動かしても最良 {best_rate * 100:.0f}%）＝"
            f"拍の頭にコード構成音を置いて書き直す必要がある。移調を試さないこと"]


# ---------- 仮歌（正弦波）----------
#: 仮歌サイン波の振幅。**実歌声（V6）の音量に合わせてある**（2026-09-20）。
#:
#: ★なぜ 0.3 ではないか——仮歌は「聴くためのもの」ではなく、いまや
#: **`score_to_vpr._accomp_gain_db` が伴奏の初期音量を決める基準**になっている
#: （仮歌ミックスは 2026-09-06 に廃止され、仮歌を読むのは音量見積もり・口の動き・
#: タイミング評価の3つだけ。どれも人の耳には入らない）。
#: 0.3 は昔「聴こえればよい」で決めた任意の値で、**実歌声より 13.5dB 大きかった**。
#: そのぶん伴奏が持ち上がるので、書き出すたびに `v6_export._fix_balance` が音量を直して
#: **V6 をもう一度起こしていた**（実測 14回中14回・書き出しが 1.9分→3.7分）。
#: 本人:「サイン波の大きさをそもそも変えとけば、狙い通り率上がるんじゃない？」
#:
#: 下げても壊れないことは実測済み（同日・9曲＋1曲で確認）——
#: 口の動きは 7871 フレーム中**差が 0.01 を超えたもの 0 個**（95 パーセンタイルで
#: 正規化しているため）、タイミング評価は offset/drift とも**完全一致**
#: （相互相関の argmax は倍率に影響されない）。
#: 値の根拠: 9曲で「正解だったフェーダー値 − 見積もり」の中央値が **−13.6dB**。
#: 0.3 × 10^(−13.6/20) ≒ 0.063。
PLACEHOLDER_AMP = 0.063


def render_placeholder_vocal(score: SongScore, path: Path) -> None:
    """サイン波の仮歌を書く。**聴かせるためではない**——口の動き（`lipsync.json`）の材料と、
    vpr の伴奏音量の見積もり（仮歌と伴奏の音量比）に使う（聴かせる仮歌ミックスは 2026-09-06 に廃止）。

    ★**numpy でまとめて計算する**（2026-09-26）。以前は音を1点ずつ Python の数値として並べ、
      1点ずつ `struct.pack` してつないでいたので、200秒の曲で**80秒・最大1.4GB**かかり、
      メモリが混んだ時間帯に `MemoryError` で落ちた（『明日への足音』v4 の上ハモの当て直し）。
      式は同じ（音ごとに立ち上がり 20ms・消え際 50ms のサイン波を足し、±1 で切って 16bit にする）。
    """
    spt = 60.0 / score.tempo / Q
    total = int((score.duration_ticks() * spt + 0.5) * SR) + 1
    buf = np.zeros(total, dtype=np.float64)

    def _render(notes, gain: float = 1.0) -> None:
        for n in notes:
            freq = 440.0 * 2 ** ((n.key - 69) / 12)
            i0 = int(n.start_tick * spt * SR)
            ln = int(n.duration_tick * spt * SR)
            m = min(ln, total - i0)                       # 曲の終わりを越える分は捨てる
            if m <= 0:
                continue
            i = np.arange(m, dtype=np.float64)
            t = i / SR
            env = np.minimum(1.0, t / 0.02) * np.minimum(1.0, (ln - i) / SR / 0.05)
            buf[i0:i0 + m] += gain * PLACEHOLDER_AMP * env * np.sin(2 * math.pi * freq * t)

    _render(score.vocal_notes, 1.0)                       # メイン
    for v in score.extra_voices:                          # 追加声部（ハモリ/輪唱/重ね/コーラス）
        _render(v.notes, 10 ** (v.gain_db / 20))
    pcm = (np.clip(buf, -1.0, 1.0) * 32767).astype("<i2")  # int() と同じく 0 へ向けて切り捨て
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ---------- mix ----------

def _pick_vocal(song_dir: Path, vocal_filename: str | None) -> Path | None:
    """V6 の生の書き出しを選ぶ: **明示指定 > 最新の書き出し**。無ければ None。

    **使ってよいのは `adopt_real_vocal_as_final()` だけ**（2026-09-02）。
    ミックスの経路から呼ぶと、V6 の完成ミックス（伴奏同梱）を「歌のトラック」として拾って
    **伴奏が二重に鳴る**。混ぜる側は仮歌（`placeholder.wav`）だけを見ること。

    候補は V6 が vpr と同じ名前で出す生の書き出し（`vocal.wav`・`vocal_v{N}.wav`＝`song_files.v6_raw`）。
    「最新」は**更新時刻**で比較（人間がどの版の vpr から書き出しても拾う・2026-07-11）。同時刻は固定名 `vocal.wav` 優先。
    ★完成品（`master_…`）は候補にしない（2026-10-03・F14①。仕上げたあと付け替えるので、生の書き出しは残らない）。"""
    if vocal_filename:
        p = song_dir / vocal_filename
        return p if p.exists() else None
    raw_fixed = song_files.split_version(song_files.v6_raw(1))[0]   # V6 が作業台の vpr（固定名）から出す名前（`vocal.wav`）
    cands = [p for p in (song_dir.iterdir() if song_dir.is_dir() else [])
             if p.is_file() and ((song_files.split_version(p.name) or ("",))[0] == raw_fixed or p.name == raw_fixed)]
    if cands:
        return max(cands, key=lambda p: (p.stat().st_mtime, p.name == raw_fixed))
    return None


def set_mix_setting(song_dir: Path, **kw) -> None:
    """mix.json に音量設定を保存（None は無視）。adjust_song 等で指定された値を後続mixが踏襲する。"""
    p = song_dir / "mix.json"
    d = {}
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            d = {}
    d.update({k: v for k, v in kw.items() if v is not None})
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")



# ---------- 仮歌ミックス（draft_mix）は 2026-09-06 に廃止 ----------
# **存在理由が消えたため**。もとは「人が V6 を開くまで、完成形の代わりに聴くもの」だったが、
# `export_vocal` が同じセッションの中で **90秒で実歌声を出す**ようになったので、待つ時間が無い。
# しかも仮歌はサイン波なので、それでバランスを判断しても**本番の音とは違う**（判断材料として不正確）。
#
# ボカロ制作の実務でも「伴奏を作る間だけ仮の音でメロディを鳴らす」のは標準だが、
# それに当たるのは `placeholder.wav` のほうで、**そちらは残す**（口の動きの材料でもある）。
#
# 一緒に消えたもの:
#   ・`_apply_region_gains`（区間ごとの音量増減）… draft_mix にしか効いていなかった
#   ・`auto_accomp_gain` / `_rms` / `_high_shelf` … mix 専用
#   ・**伴奏を歌の長さに切り揃えて書き戻す副作用**。これは `write_wav` が書くたびに
#     音量を正規化するため、**compose 直後の1回目だけ結果が変わる**という性質を生んでいた
#     （board H54）。関数ごと消えたので、その論点も同時に解消した。
#     vpr に同梱する伴奏は `score_to_vpr` が作る時点のものを使う（切り揃えの前後で変わらない）
#
# 音量バランスは **vpr に焼き込む**経路が正（`score_to_vpr._accomp_gain_db` が
# `placeholder.wav` と `accompaniment.wav` の RMS 比＋ `mix.json` の `vocal_lead_db` から算出）。
# 微調整は V6 のフェーダーが最も簡単、というのが 2026-08-10 からの方針。

def adopt_real_vocal_as_final(song_dir: Path, vocal_filename: str | None = None,
                              version: int | None = None) -> Path | None:
    """**手で書き出した V6 の歌声を仕上げて完成品にする**（`master_<形>_v{N}.wav`・2026-10-03・§6-2・F14①）。混ぜ直さない。

    書き出しは `export_vocal` が自動でやる（2026-09-05〜）。V6 が無い・自動の書き出しが失敗したときはマスターが手で。
    vpr は伴奏同梱（歌もの）なので、V6 のミックスダウンは歌＋伴奏がすでに揃った完成品。ここで伴奏を足すと二重になる。
    V6 は vpr と拡張子違いの同名で出す（`vocal_vN.vpr` → `vocal_vN.wav`）ので、マスターは既定の名前で保存すればよい。
    自動の書き出しと同じく**仕上げ（末尾の無音・−14 LUFS）を当ててから完成品の名前へ付け替え**、生の書き出しは残さない
    （2026-10-03 まではそのまま `final.wav` へ写していた。`final.wav` は作らない）。

    版は書き出しの名前（`vocal_v{N}.wav`）から読む。名前に版が無い（`vocal.wav`・明示した別の名前）ときは `version` で渡す。
    形は楽譜（`score_v{N}.json`／作業台の `score.json`）から読む。**インストは歌声が無いので受け付けない**（`ValueError`）。

    Returns:
        完成品のパス。V6 の書き出しがまだ無ければ None（何もしない）。
    """
    from v6_export import finish_as_master, score_form
    raw = _pick_vocal(song_dir, vocal_filename)
    if raw is None:
        return None
    n = song_files.version_of(raw.name) or version
    if not n:
        raise ValueError(f"書き出し『{raw.name}』の版が分からない（名前に _v{{N}} が無い）。版を渡して呼ぶ")
    form = score_form(song_dir, n)
    if form is None:
        raise ValueError(f"v{n} の楽譜が見つからないので、曲の形（完成品の名前）が分からない")
    if form == "instrumental":
        raise ValueError("インストの曲には歌声の書き出しが無い（完成品は render_final が伴奏から作る）")
    dst, _note = finish_as_master(raw, song_dir, form, n)
    return dst


def _render_accomp(score: SongScore, out_wav: Path) -> None:
    """伴奏を **SoundFont（実音）** で鳴らす。**鳴らせなければ止めて原因を返す**（例外をそのまま上げる）。

    ★2026-09-29（本人方針「組み立ては補正しない」・代わりのもので鳴らさない）: 以前は SoundFont の書き出しが
      **どんな理由で失敗しても**黙って numpy の合成音（7種の近似）へ落としていた。合成音では**ピアノを指定しても
      鉄琴のような音**で鳴り（『永遠に変わる晩夏』は全編がそれ）、タムはキック・シンバルはハイハットで鳴った。
      しかも落とし先へ行くと元の失敗（`N.C.` で落ちた等）がログの1行に埋もれ、別の音の曲が出来上がっていた。
      いまは FluidSynth・SoundFont が無い・MIDI が作れない・書き出しに失敗した、のどれでも例外が上がる
      （`render_sf.render` → `sf2_render.render_midi`）。呼び手（曲づくり・編集の道具・試聴）がそれを失敗として返す。
    """
    from render_sf import render as render_soundfont
    art_warns: list[str] = []
    render_soundfont(score, out_wav, warns=art_warns)
    for line in art_warns:                                 # 奏法を掛けられなかった所見
        print(f"  [奏法] {line}")


# ※作業中の版を読む `working_version` は 2026-10-03（再レビュー）に外した。版は session の1か所（`_working_version`・
#   `ensure_working_version`・`open_version_for_form`）で決める（読み手が2本あると片方だけ直る）。


def clear_other_form(song_dir: Path, form: str) -> list[str]:
    """**作業台から、その形の持ち物に無いファイルを消す**（2026-10-03・§4-4）。消した名前を返す（無ければ空）。

    版は作業台方式で、`session.end_session` は固定名にあるものを全部次の版へ写す。形を変えた版で作らなかったファイル
    （歌もの→アカペラの伴奏 wav・歌もの→インストの vpr と口の動き）が前の版から広げられたまま残ると、vpr に同梱され・
    音量の測定に使われ・次の版に写る。**ファイルの有無ではなく形で決める**——形の持ち物（`song_files.FORM_FILES`）に無いものを消す。
    消すのは作業台の固定名だけ（どれも楽譜から作る生成物）。前の版の `*_vN` は触らない。消したことは作業記録（`log_build.txt`）に残す。
    """
    keep = set(song_files.FORM_FILES[song_files.as_form(form)])
    gone = []
    for name in song_files.WORKBENCH:
        p = song_dir / name
        if name not in keep and p.is_file():
            p.unlink()
            gone.append(name)
    if gone:
        from datetime import datetime
        with (song_dir / song_files.BUILD_LOG).open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%H:%M:%S}] compose.form_files  form={form}  "
                    f"removed={','.join(gone)}（{song_files.FORM_LABEL[form]}の持ち物に無い前の形のファイル）\n")
    return gone


# ---------- compose（coordinator 相当）----------
def compose(song_id: str, score: SongScore, *, voice_comp_id: str = MIKU_V6_COMP_ID,
            version: int | None = None) -> dict:
    """楽譜から書き出す（伴奏・vpr・仮歌・口の動き・状態）。**形で分かれる**（2026-10-03・§5-3）:

    | | 歌もの | アカペラ | インスト |
    |---|---|---|---|
    | 最初 | 作業台からその形の持ち物に無いファイルを消す（`clear_other_form`） | 同じ | 同じ |
    | 伴奏 wav・mid | 作る | **作らない** | 作る（主旋律の楽器を含む） |
    | vpr | 伴奏を同梱 | 伴奏なし（`embed_accomp=False`） | **作らない** |
    | 仮歌・口の動き | 作る | 作る | **作らない**（F8） |
    | 状態 | `awaiting_vocal_export` | `awaiting_vocal_export` | `finished`（完成品 `master_inst_v{N}.wav` をここで書く） |

    **版**（`version` を省いたとき・2026-10-03・再レビュー）:
    - 作業中の版に**別の形の完成品**がもうあれば（完成のあと続けて形を変えた）、その版を確定して次の版を開く
      （`session.open_version_for_form`・同じ版に形の違う完成品を2つ作らない・前の版は残る）。全部の形
    - インストの完成品の版 N は**作業中の版の印**（`_WORKING_v{N}.txt`）から引く。印が無ければ `render_final` と同じ決め方
      （`session.ensure_working_version`＝現在版+1）でここで置く。★以前は印が無いと止めていた——印を置くのは `render_final` と
      `begin_session` だけで、作り直し（`remake_from_step`・`change_form`・`rearrange_song`・途中の曲の続き）は置かずに書き出すので、
      LLM の費用を払って作り直したインストが書き出しで例外になり、結果が全部失われた。歌もの・アカペラは印を置かない（今までどおり）
    """
    song_dir = OUT_ROOT / song_id
    song_dir.mkdir(parents=True, exist_ok=True)
    form = song_files.as_form(getattr(score, "form", "song"))
    if not version:
        # 版を決めるのは session（作業台方式の1か所）。engine から読むのはここだけ（循環を避けて中で読む）
        from session import ensure_working_version, open_version_for_form
        working = open_version_for_form(song_dir, form)
        if form == "instrumental":
            version = working or ensure_working_version(song_dir)
    removed = clear_other_form(song_dir, form)            # ★前の形のファイルを新しい版に混ぜない（§4-4）

    issues = score_validate(score)
    (song_dir / song_files.SCORE).write_text(score.model_dump_json(indent=2), encoding="utf-8")

    # **伴奏を先に書く**。score_to_vpr は「フォルダに accompaniment.wav があれば同梱」する実装なので、
    # 順序が逆だと**新規曲の vpr に伴奏が入らない**（＝V6 で開いても歌だけ。「ミックスダウン一発で
    # 完成ミックス」という設計の前提が初回だけ崩れていた）。既存曲は編集で vpr が作り直される際に
    # 前回の伴奏が残っていて入るため、v2 以降だけ伴奏ありという分かりにくい形で潜伏していた（2026-08-20 修正）
    # ★アカペラは伴奏を作らない（空の MIDI を FluidSynth に渡さない・§5-3）
    if form != "acappella":
        _render_accomp(score, song_dir / song_files.ACCOMP_WAV)
    master = ""
    if form == "instrumental":
        # ★**インストは V6 を使わない**（歌が無い）。伴奏 wav（主旋律の楽器を含む）に仕上げを当てた写しが完成品（§5-3・§6-2）。
        #   伴奏 wav は途中の物として残す（手直し・試聴の材料）
        from v6_export import finish_as_master
        dst, note = finish_as_master(song_dir / song_files.ACCOMP_WAV, song_dir, form, version, copy=True)
        master = dst.name + note
        status = "finished"
    else:
        # アカペラは伴奏を同梱しないことを**明示して**渡す（フォルダに前の形の伴奏 wav があるかで決めない・§4-4）
        score_to_vpr(score, song_dir / song_files.VPR, is_ai_singer=True, voice_comp_id=voice_comp_id,
                     embed_accomp=form == "song")
        render_placeholder_vocal(score, song_dir / song_files.GUIDE)
        write_lipsync(score, song_dir)                    # 口の動き（score＋仮歌が揃った時点で確定）
        status = "awaiting_vocal_export"                  # V6 の書き出し（→ 完成品 master_<形>_v{N}.wav）待ち
    meta = {"song_id": song_id, "title": score.title, "form": form, "status": status,
            "bpm": score.tempo, "validate_issues": issues, "removed": removed, "master": master}
    (song_dir / song_files.META).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
