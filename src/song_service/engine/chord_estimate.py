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

"""**小節ごとのコードの推定を1か所に**（伴奏 MIDI から・分離していない原曲の音声から。2026-10-02・組「カバーのコード」）。

推定は**材料**で、曲のコードにはしない（カバーのコードは LLM が書く・`song_cover_chords`）。どちらの推定も
**同じ24の型**（長三和音・短三和音 × 12 の根音）と**同じ出力の形**（`BarGuess`＝小節ごとの上位3と確からしさ・鳴っている音）にする。

| 入口 | 何から | 照合のしかた |
|---|---|---|
| `from_midi` / `estimate_bar_chords` | 伴奏 MIDI（`accomp_edit.load_accomp` のトラック） | 小節の中で鳴っている長さ＋低音の重み−経過音の罰（2026-10-02 まで `accomp_edit.estimate_bar_chords` にあったものをそのまま移した） |
| `from_audio` | 分離していない原曲の音声（CPU・`librosa` だけ） | CQT → CQT の上で打楽器を弱める（HPSS）→ 小節の平均のクロマ → 型との照合（`match_chroma`） |

確からしさは、型の点の差を温度 `TEMPERATURE` で割合にしたもの（目安。帯で縛らず、LLM に材料として見せるだけ）。
測った結果と前提（SoundFont の音では伴奏 MIDI からと同じくらい当たる・本物の録音では未確認）は
`docs/plan/song_cover_chords_llm.md` §5。

同じ照合を持っていた `reconstruct_accompaniment._chord_events`（呼ばれていない DSP 版の下書き）は 2026-10-03 に消した（本人承認）——3つ目の実装を作らない。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: 和音の取れない小節に書くコード名（記法の仕様書 §3 のノーコード。尺の変更 `song_resize.NO_CHORD_MARK` と同じ綴り）
NO_CHORD_MARK = "N.C."
#: 根音の綴り（推定は調を知らないので ♯ で綴る。見せる側が C 基準へ移すときに綴り直す）
NAMES = "C C# D D# E F F# G G# A A# B".split()
#: **24の型**（名前, 構成音のピッチクラス）。並びは 根音 C→B、それぞれ長→短（同点なら先のものを採る）
TRIADS: list[tuple[str, tuple[int, int, int]]] = [
    (NAMES[r] + kind, (r, (r + third) % 12, (r + 7) % 12)) for r in range(12) for kind, third in (("", 4), ("m", 3))]
#: 確からしさの温度（型の点の差をこの値で割って割合にする。点は 0〜1 前後に揃えてある）
TEMPERATURE = 0.05


@dataclass
class BarGuess:
    """1小節の推定。`top` は上位3つ (コード名, 確からしさ)・`tones` は鳴っている音（ピッチクラス・多い順）。
    鳴っている音がほとんど無い小節は `top` も `tones` も空（＝取れない。`N.C.` とは区別する——LLM の答えに見えないように）。"""
    top: list[tuple[str, float]] = field(default_factory=list)
    tones: list[int] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.top[0][0] if self.top else NO_CHORD_MARK


def _ranked(scores: np.ndarray) -> list[tuple[str, float]]:
    """24の型の点 → 上位3つ (名前, 確からしさ)。同点は型の並び（`TRIADS`）の先を採る。"""
    e = np.exp((scores - scores.max()) / TEMPERATURE)
    p = e / e.sum()
    order = sorted(range(len(TRIADS)), key=lambda i: (-scores[i], i))[:3]
    return [(TRIADS[i][0], round(float(p[i]), 2)) for i in order]


def _tones(weights: np.ndarray, *, floor: float = 0.5, most: int = 5) -> list[int]:
    """ピッチクラスの重み → 鳴っている音（多い順・いちばん大きい音の `floor` 倍以上・`most` 個まで）。"""
    top = float(weights.max()) if len(weights) else 0.0
    if top <= 0:
        return []
    return [int(i) for i in sorted(range(12), key=lambda i: -weights[i]) if weights[i] >= floor * top][:most]


# ---- 伴奏 MIDI から ---------------------------------------------------------------------------

def from_midi(tracks: list, tempo_bpm: float, bar0: int, bar1: int, beats_per_bar: float = 4.0) -> list[BarGuess]:
    """小節ごとの推定（トライアド・低音ルート重視）。`beats_per_bar` は1小節の4分音符の数（`song_design.beats_per_bar`。既定は 4/4）
    ——曲の拍子と違う数で切ると、推定した和音が表の小節とずれる。打楽器のトラックは数えない。"""
    spb = 60.0 / tempo_bpm
    out: list[BarGuess] = []
    for b in range(bar0, bar1):
        w0, w1 = b * beats_per_bar * spb, (b + 1) * beats_per_bar * spb
        pcw, bass = np.zeros(12), np.zeros(12)
        for tr in tracks:
            if tr.is_drums:
                continue
            for n in tr.notes:
                ov = min(w1, n.start_sec + n.dur_sec) - max(w0, n.start_sec)
                if ov <= 0:
                    continue
                pcw[n.key % 12] += ov
                if n.key < 52:
                    bass[n.key % 12] += ov * 2
        if pcw.sum() < 0.15:                               # 鳴っている音がほとんど無い＝取れない
            out.append(BarGuess())
            continue
        sc = np.array([pcw[list(iv)].sum() + bass[iv[0]] - 0.3 * sum(pcw[(iv[0] + i) % 12] for i in (1, 2, 6))
                       for _n, iv in TRIADS])
        out.append(BarGuess(top=_ranked(sc / (pcw.sum() + bass.sum())), tones=_tones(pcw)))
    return out


def estimate_bar_chords(tracks: list, tempo_bpm: float, bar0: int, bar1: int,
                        beats_per_bar: float = 4.0) -> list[str]:
    """小節ごとのいちばん確からしいコード名（アレンジ時のリードシート代わり・曲の表・伴奏の観察が使う）。
    ★**和音の取れない小節（鳴っている音がほとんど無い）は `N.C.`**（2026-10-02 本人「おすすめで」・記法の仕様書 §3）。
      以前は記法に無い `-` を書いていた。"""
    return [g.name for g in from_midi(tracks, tempo_bpm, bar0, bar1, beats_per_bar)]


# ---- 分離していない原曲の音声から（CPU）---------------------------------------------------------

_TPL = np.array([[1.0 if pc in iv else 0.0 for pc in range(12)] for _n, iv in TRIADS]) / np.sqrt(3.0)


def match_chroma(chroma: np.ndarray, bass: np.ndarray | None = None, bass_w: float = 0.0) -> np.ndarray:
    """12次元のクロマ（と低音のクロマ）→ 24の型の点（クロマとの余弦＋低音の重み×根音の低音の割合）。"""
    f = np.asarray(chroma, dtype=float)
    f = f / (np.linalg.norm(f) + 1e-9)
    sc = _TPL @ f
    if bass is not None and bass_w:
        bs = np.asarray(bass, dtype=float)
        bs = bs / (bs.sum() + 1e-9)
        sc = sc + bass_w * np.array([bs[iv[0]] for _n, iv in TRIADS])
    return sc


def audio_features(y: np.ndarray, sr: int, hop: int = 512):
    """音声 → (各フレームの時刻, 全体のクロマ, 低音のクロマ, 音量)。CQT（C1 から7オクターブ）の上で打楽器を弱める
    （STFT で分けるやり方はこの機械ではメモリが足りず落ちた・設計資料 §5-1）。"""
    import librosa
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C1"),
                           n_bins=84, bins_per_octave=12)).astype(np.float32)
    C = librosa.decompose.hpss(C, margin=2.0)[0]
    C = np.maximum(librosa.amplitude_to_db(C, ref=np.max) + 60.0, 0.0)   # 60dB 下までを残す
    full = np.zeros((12, C.shape[1]))
    bass = np.zeros((12, C.shape[1]))
    for k in range(C.shape[0]):
        midi = 24 + k                                       # C1=24
        if 36 <= midi < 96:
            full[midi % 12] += C[k]
        if 28 <= midi < 52:
            bass[midi % 12] += C[k] * (1.0 if midi < 48 else 0.5)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=hop)
    return times, full, bass, rms


def from_audio(y: np.ndarray, sr: int, bar_sec: float, n_bars: int, *, bass_w: float = 0.0) -> list[BarGuess]:
    """音声（曲の頭＝楽譜の tick 0 から）→ 小節ごとの推定。小節の音量が曲のいちばん大きい所（上位5%）の 5% 未満なら取れない。
    低音の重みの既定 0 は、SoundFont で鳴らした音でいちばん当たった値（本物の録音では 1.5 のほうが一致が高かった・設計資料 §5-2）。
    ★**鳴っている音（`tones`）は出さない**——このクロマは隣の半音へのにじみが大きく（合成した C の和音で B・C# が C と同じくらい出る）、
    「鳴っている音」として見せると LLM を誤らせる。上位3つと確からしさだけを材料にする（2026-10-02 に合成音で確かめた）。"""
    times, full, bass, rms = audio_features(y, sr)
    loud = float(np.percentile(rms, 95)) if len(rms) else 1.0
    n = min(len(times), len(rms))
    times, rms = times[:n], rms[:n]
    out: list[BarGuess] = []
    for b in range(n_bars):
        sel = (times >= b * bar_sec) & (times < (b + 1) * bar_sec)
        if not sel.any() or rms[sel].mean() < 0.05 * loud:
            out.append(BarGuess())
            continue
        f = full[:, :n][:, sel].mean(axis=1)
        out.append(BarGuess(top=_ranked(match_chroma(f, bass[:, :n][:, sel].mean(axis=1), bass_w))))
    return out
