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

"""スコア駆動リップシンク: 歌詞と音符から「口の開き」の時系列を作る（Phase 2 再生用）。

音量だけで口を開閉すると伴奏の打楽器にも反応してしまう。ここでは**歌詞を知っている口**を作る:
  - 開き量 = 仮歌の音量エンベロープ × **母音の開き係数**（あ1.0 / え0.75 / お0.7 / い0.4 / う0.35）
  - **ん**はほぼ閉じ(0.15)・**っ**は閉じ(0)
  - **ま行・ば行・ぱ行の頭で約80ms唇を閉じる**（人間は m/b/p で必ず閉じる）
  - 音符が無い区間（間奏・ブレス）は 0＝閉じる

エンベロープ源は **placeholder.wav（サイン波の仮歌）**。実歌声（VOCALOID6 の書き出し）ではなく
仮歌を使うのは、①score から必ず生成される ②歌声のみで伴奏が混ざらない ③実歌声とエンベロープが
ほぼ変わらない、の3点による（2026-08-12 ユーザー判断）。これにより **final.wav に依存せず**、
曲を確定した時点（render_final）で口の動きまで決まる。

出力 `lipsync.json`: {"fps": 60, "values": [0.0〜1.0, ...]}（配列の先頭 = 曲の t=0）。
再生側は `final.wav` の再生位置からこの配列を引くだけでよい（音声解析をしない）。

ラボ（test/song_generation_poc/live2d_video/build_lipsync.py）からの移設・2026-08-14。
ラボ版は vpr を読んでいたが、本番では **score（SoT）を直接受ける**（vpr を読み直す必要がない）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from song_schema import TICKS_PER_QUARTER as Q
from song_schema import SongScore

FPS = 60
VOWEL_OPEN = {"a": 1.0, "e": 0.75, "o": 0.7, "i": 0.4, "u": 0.35}
CLOSE_HEADS = set("まみむめもばびぶべぼぱぴぷぺぽ")   # m/b/p＝唇を閉じる子音
_DEFAULT_OPEN = 0.7                                   # 母音を判定できなかったかな（記号等）の既定


def _vowel_factor(kana: str, cache: dict[str, float]) -> float:
    """かな1音の開き係数。ローマ字末尾の母音で決める（拗音「しゃ」等もモーラ全体で判定）。"""
    if kana in cache:
        return cache[kana]
    if kana == "ん":
        cache[kana] = 0.15
    elif kana == "っ":
        cache[kana] = 0.0
    else:
        import pykakasi
        rom = "".join(x.get("hepburn", "") for x in pykakasi.kakasi().convert(kana)).lower()
        cache[kana] = next((VOWEL_OPEN[c] for c in reversed(rom) if c in VOWEL_OPEN), _DEFAULT_OPEN)
    return cache[kana]


def build_lipsync(score: SongScore, placeholder_wav: Path) -> dict:
    """score（歌詞・音符）と仮歌 wav から口の開き値（60fps）を作って返す。

    音符は**メイン旋律のみ**（extra_voices＝ハモリは口の動きを作らない。口は1つなので、
    主旋律に追従させる）。
    """
    y, sr = sf.read(placeholder_wav, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    n_frames = int(len(y) / sr * FPS) + 1
    hop = max(1, int(sr / FPS))
    # フレームごとの RMS（末尾の端数フレームは空になりうるので 0 で埋める）
    pad = np.zeros(n_frames * hop, dtype="float32")
    pad[:len(y)] = y[:len(pad)]
    env = np.sqrt(np.mean(np.square(pad.reshape(n_frames, hop)), axis=1))
    env = np.convolve(env, np.ones(3) / 3, mode="same")        # 軽い平滑（3フレーム≈50ms）
    loud = env[env > 0.001]
    ref = float(np.percentile(loud, 95)) if loud.size else 1.0
    env = np.clip(env / max(ref, 1e-6), 0.0, 1.0)

    spt = 60.0 / float(score.tempo) / Q                        # 1tick の秒数
    notes = sorted(score.vocal_notes, key=lambda n: n.start_tick)
    cache: dict[str, float] = {}
    segs = []                                                  # (開始f, 終了f, 開き係数, 頭で閉じるか)
    prev_fc = _DEFAULT_OPEN
    for i, n in enumerate(notes):
        t0 = n.start_tick * spt
        # 次の音符の頭まで開けたままにする（レガートに歌う人の口。最後の音は自分の長さ分だけ）
        t1 = (notes[i + 1].start_tick * spt) if i + 1 < len(notes) else t0 + n.duration_tick * spt
        # ★歌詞「-」は**前の音のタイ**（2026-09-13・伸ばしをこれで書くようにした）。
        #   口は前の母音のまま。頭で閉じもしない（新しい音ではないため）。
        fc = prev_fc if (n.lyric or "") == "-" else _vowel_factor(n.lyric or "あ", cache)
        segs.append((int(t0 * FPS), int(t1 * FPS), fc,
                     (n.lyric or "")[:1] in CLOSE_HEADS and n.lyric != "-"))
        prev_fc = fc

    fac = np.zeros(n_frames)                                   # 音符区間の母音係数（区間外=0→口閉じ）
    for f0, f1, fc, _ in segs:
        fac[max(0, f0):min(n_frames, max(f0 + 1, f1))] = fc
    values = env * fac
    for f0, _f1, _fc, close_head in segs:                      # m/b/p の頭: 直前30ms〜頭50msを閉じる
        if close_head:
            values[max(0, f0 - 2):min(n_frames, f0 + 3)] = 0.0
    values = np.convolve(values, np.ones(2) / 2, mode="same")  # 最終平滑（閉じの角を少しだけ丸める）

    on = values[values > 0.05]
    return {
        "fps": FPS,
        "values": [round(float(v), 3) for v in values],
        "meta": {"frames": n_frames, "seconds": round(n_frames / FPS, 2), "notes": len(notes),
                 "voiced_ratio": round(float(on.size) / max(n_frames, 1), 3),
                 "mean_open": round(float(on.mean()), 3) if on.size else 0.0,
                 "closes": sum(1 for s in segs if s[3])},
    }


def write_lipsync(score: SongScore, song_dir: Path) -> Path | None:
    """曲フォルダに `lipsync.json` を書く。仮歌が無ければ何もしない（None を返す）。

    **placeholder.wav を書き出した直後に呼ぶ**（score と仮歌が揃った時点＝口の動きが確定する時点）。
    失敗しても曲づくりは止めない（歌えないだけで、成果物としての曲は無事）。
    ★**インストは口の動きを作らない**（2026-10-03・F8。歌わずに流す）——楽譜の形がインストなら何も書かず None。
    アカペラは主旋律だけで作る（いまのまま・声部は口に出さない）。
    """
    import song_files                                       # 名前の決まりは1か所
    if getattr(score, "form", "song") == "instrumental":
        return None
    ph = song_dir / song_files.GUIDE
    if not ph.exists():
        return None
    try:
        data = build_lipsync(score, ph)
        out = song_dir / song_files.LIPSYNC
        out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return out
    except Exception as e:                                     # pykakasi 不在・壊れた wav 等
        print(f"[lipsync] 生成に失敗（歌わせる時に作り直せる）: {type(e).__name__}: {e}")
        return None
