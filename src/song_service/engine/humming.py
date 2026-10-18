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

"""**マスターが口ずさんだ音**（鼻歌・口笛・肉声）を読み取り、旋律工程へ渡す**1枚のテキスト**にする。

## なぜテキストで渡すのか

音符の一覧を構造体で渡すと、**全部が同じ確からしさに見える**。実際は
「この音は確か・ここは怪しい・ここはしゃくり」と差があり、その差こそが下流の判断材料になる。
本人:「**鼻歌とか実際のアカペラって精度もそんなに高くないし、機械的な読み取りがさらにそこに入る。
参照結果を全部テキストみたいな感じで渡せばいいのかなとも思った**」——そのとおりで、
**前工程の結果を全部テキストで渡している**いまの作りにそのまま乗る（`melody_prompt` と同じ形）。

## 測り方は「その録音から較正する」

歌い方は人によって・曲によって変わるので、**固定の数値を置かない**
（本人:「**人に配るときはさらにいろんな歌い方があるだろうからロバストな測定方法にしてね**」）。
音域・ビブラート・移動の境目・動きを見る幅を、**どれもその録音自身から決める**。

## 音が新しく始まるのは3つのときだけ

鼻歌は口を閉じて出すので**子音が無い**。振幅の立ち上がりを数えるだけでは足りず、逆に数えすぎる。

  (a) 声が途切れて、また出る  (b) 同じ高さのまま押し直す  (c) 高さが変わって落ち着く

(c) の**移動中（渡り・しゃくり）はどの音でもない**。音符にすれば「余計な音」になり、
高さの計算に混ぜれば音程がずれる。**装飾は捨てずに印を付けて残す**（潰すかは下流の判断）。

## 測った精度（2026-09-23・`docs/plan/humming_input.md` §4）

正解のあるキラキラ星で **鼻歌 F=0.835 / 肉声 F=0.846（歌詞つき）/ 口笛 F=0.660**。
他人の歌40本（vocadito）で **F=0.442**——ただし**人間の注釈者どうしの一致が 0.740** なので、
**1.0 を目指す意味はない**。検証の道具は `test/humming/`。**GPU は要らない**（CPU で実時間の 0.3〜1.0 倍）。
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

HOP = 256
GATE_REL_DB = 30.0          # 最大音量からこれだけ下は声でない（暗騒音の絶対 dB は部屋で変わる）
DRIFT_CLAMP = (0.25, 1.5)   # 較正が暴れたときの常識的な範囲（半音）
VIBRATO_MIN_CENT = 50.0     # これ未満は「自然な揺らぎ」＝均さない（声のビブラートは ±30〜100cent）
REATTACK_DB = 4.0           # 押し直しと認める音量の落ち込みの下限
MIN_NOTE = 0.06

_ABC = {0: "C", 1: "^C", 2: "D", 3: "^D", 4: "E", 5: "F",
        6: "^F", 7: "G", 8: "^G", 9: "A", 10: "^A", 11: "B"}


def _otsu(vals: np.ndarray, lo: float, hi: float) -> float:
    """1次元の値を2つの山に割る境目を、分布そのものから決める（大津の方法）。"""
    v = vals[np.isfinite(vals)]
    if v.size < 20:
        return float(np.clip(np.median(v) * 2 if v.size else lo, lo, hi))
    hist, edges = np.histogram(v, bins=64, range=(float(v.min()), float(v.max())))
    centers = (edges[:-1] + edges[1:]) / 2
    total, total_c = hist.sum(), float((hist * centers).sum())
    best, best_t, w0, c0 = -1.0, None, 0.0, 0.0
    for i in range(len(hist) - 1):
        w0 += hist[i]
        c0 += hist[i] * centers[i]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        between = w0 * w1 * ((c0 / w0) - ((total_c - c0) / w1)) ** 2
        if between > best:
            best, best_t = between, centers[i + 1]
    return float(np.clip(best_t if best_t is not None else np.median(v), lo, hi))


def _abc_pitch(midi: int) -> str:
    """MIDI ノート番号 → ABC の音名（`C`＝中央ハ・`,`で下・小文字で上）。"""
    name = _ABC[midi % 12]
    octv = midi // 12 - 1                                  # 60 → 4（中央ハのオクターブ）
    if octv >= 5:
        name = name.lower() + "'" * (octv - 5)
    elif octv < 4:
        name = name + "," * (4 - octv)
    return name


def transcribe(path: str | Path) -> dict:
    """録音を音符の列にする。戻り値は `notes` / `calib` / `kind` などの入った dict。

    `notes` の各要素: `start` `end` `midi` `cent_dev`（半音からのずれ）`hold`（保った時間）
    `conf`（確度 0..1）`ornament`（同じ高さへ戻る短い寄り道＝装飾候補）`db`。
    """
    src = str(path)
    y, sr = librosa.load(src, sr=22050, mono=True)
    fps = sr / HOP

    # ── 較正0: 探す音域を、その録音から決める ─────────────────
    # ★固定で 65〜1000Hz にすると**口笛が丸ごと外れる**（口笛は 1〜2kHz に出る）。
    #   実際、既存の採譜は口笛を**1オクターブ低く**採っていた。
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    sfreq = librosa.fft_frequencies(sr=sr, n_fft=2048)
    fdb = 20 * np.log10(S.max(axis=0) + 1e-9)
    strong = fdb > fdb.max() - GATE_REL_DB
    f_med = float(np.median(sfreq[np.argmax(S[:, strong], axis=0)])) if strong.any() else 200.0
    fmin = float(np.clip(f_med / 3.0, 55.0, 1500.0))
    fmax = float(np.clip(f_med * 3.0, 200.0, 4000.0))

    f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=HOP)
    times = librosa.times_like(f0, sr=sr, hop_length=HOP)
    midi_raw = librosa.hz_to_midi(np.where(voiced, f0, np.nan))
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0][:len(times)]
    db = 20 * np.log10(rms + 1e-9)
    gate = float(db.max()) - GATE_REL_DB
    n = len(times)

    # ── 較正1: ビブラートの周期を測り、1周期ぶんで均す ──────────
    trend = np.full(n, np.nan)
    w = int(round(0.25 * fps))
    for i in range(n):
        seg = midi_raw[max(0, i - w // 2):min(n, i + w // 2 + 1)]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            trend[i] = float(np.median(seg))
    rv = (midi_raw - trend)[np.isfinite(midi_raw - trend)]
    vib_hz = vib_depth = 0.0
    if rv.size > int(fps):
        spec = np.abs(np.fft.rfft(rv - rv.mean()))
        freqs = np.fft.rfftfreq(rv.size, d=1.0 / fps)
        band = (freqs >= 3.0) & (freqs <= 9.0)
        if band.any():
            vib_hz = float(freqs[band][np.argmax(spec[band])])
            vib_depth = float(np.sqrt(np.mean(rv ** 2)))
    # ★深さの下限を置かないと、**自然な揺らぎをビブラートと誤って 200ms も均し**、音が潰れる
    has_vibrato = vib_hz > 0 and vib_depth >= VIBRATO_MIN_CENT / 100.0
    sm = max(3, int(round(fps / vib_hz))) if has_vibrato else 3
    smooth = np.full(n, np.nan)
    for i in range(n):
        seg = midi_raw[max(0, i - sm // 2):min(n, i + sm // 2 + 1)]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            smooth[i] = float(np.median(seg))

    # ── 較正2: どれだけの幅で「動いた」を見るか ────────────────
    # ★均す幅（ビブラート由来）と動きを見る幅（音の速さ由来）は**別物**。
    #   同じにすると 35ms しか見ず、渡りがほとんど検出されない。
    ons = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, backtrack=True, units="time")
    ioi = np.diff([t for t in ons if db[min(int(t * fps), n - 1)] > gate])
    ioi_med = float(np.median(ioi)) if ioi.size else 0.2
    win = int(round(np.clip(ioi_med * 0.5, 0.05, 0.25) * fps))

    drift = np.full(n, np.nan)
    for i in range(n):
        if db[i] <= gate or np.isnan(smooth[i]):
            continue
        back = smooth[max(0, i - win):i + 1]
        fwd = smooth[i:min(n, i + win + 1)]
        back, fwd = back[~np.isnan(back)], fwd[~np.isnan(fwd)]
        if back.size and fwd.size:
            drift[i] = float(np.median(fwd)) - float(np.median(back))
    drift_th = _otsu(np.abs(drift), *DRIFT_CLAMP)

    kind = np.full(n, "無声", dtype=object)
    for i in range(n):
        if db[i] > gate and not np.isnan(smooth[i]):
            kind[i] = "移動中" if abs(drift[i]) >= drift_th else "安定"

    # ── 安定した高さの区間をまとめる ───────────────────────────
    levels: list[list] = []
    i = 0
    while i < n:
        if kind[i] != "安定":
            i += 1
            continue
        lv, j = int(round(smooth[i])), i
        while j < n and kind[j] == "安定" and abs(smooth[j] - lv) < 0.7:
            j += 1
        if j > i:
            levels.append([i, j, lv])
        i = j
    min_hold = max(int(round(fps / vib_hz)) if has_vibrato else 0, int(round(0.06 * fps)))

    # ── 安定区間の**内部**の押し直しで割る ─────────────────────
    # ★これが無いと「ドド」「ソソ」のような繰り返しが**1音に潰れる**（取りこぼしの主因だった）。
    #   谷の深さは固定せず、その録音の音量の振れ幅から決める。
    voiced_db = db[db > gate]
    dip_th = (max(REATTACK_DB, float(np.percentile(voiced_db, 90) - np.percentile(voiced_db, 10)) * 0.25)
              if voiced_db.size else REATTACK_DB)
    split: list[list] = []
    for a, b, lv in levels:
        seg, cuts, k = db[a:b], [a], 1
        while seg.size >= 3 * min(win, 9) and k < seg.size - 1:
            if seg[k] < seg[k - 1] and seg[k] <= seg[k + 1]:
                left = float(seg[max(0, k - win):k + 1].max())
                right = float(seg[k:min(seg.size, k + win + 1)].max())
                if min(left, right) - float(seg[k]) >= dip_th and a + k - cuts[-1] >= min_hold:
                    cuts.append(a + k)
                    k += min_hold
                    continue
            k += 1
        cuts.append(b)
        split += [[u, v, lv] for u, v in zip(cuts[:-1], cuts[1:]) if v - u >= min_hold]
    levels = split

    # ── 渡りの途中の踊り場を落とす（数値でなく「向き」で見る）──
    # ★**通過点は「声が続いたまま」のときにしか存在しない。**
    #   ここを見ずに向きと長さだけで判定していたら、**順次進行（ドレミ）の真ん中が消えた**——
    #   ド→レ→ミ は向きが同じなので「レ は通過点」と誤判定される。
    #   だが音と音のあいだに**声の切れ目**があれば、それは滑っているのではなく歌い直している。
    #   さらに長さは「その録音の中央値より短い」ではなく**前後どちらよりも短い**を要求する
    #   （同じ長さの音が並ぶと、中央値より短いものが半分できてしまう）。
    med_len = float(np.median([b - a for a, b, _ in levels])) if levels else 0.0
    kept = []
    for k, (a, b, lv) in enumerate(levels):
        before = drift[max(0, a - win):a][np.isfinite(drift[max(0, a - win):a])]
        after = drift[b:min(n, b + win)][np.isfinite(drift[b:min(n, b + win)])]
        legato = (0 < k < len(levels) - 1
                  and not any(kind[g] == "無声" for g in range(levels[k - 1][1], a))
                  and not any(kind[g] == "無声" for g in range(b, levels[k + 1][0])))
        shorter = (0 < k < len(levels) - 1
                   and (b - a) < (levels[k - 1][1] - levels[k - 1][0])
                   and (b - a) < (levels[k + 1][1] - levels[k + 1][0]))
        passing = (legato and shorter and before.size and after.size
                   and np.sign(np.median(before)) == np.sign(np.median(after))
                   and abs(np.median(before)) >= drift_th * 0.8
                   and abs(np.median(after)) >= drift_th * 0.8)
        if not passing:
            kept.append([a, b, lv])
    levels = kept

    # ── 同じ高さへ戻る短い寄り道＝装飾候補（落とさず印を付ける）──
    orn = set()
    for k in range(1, len(levels) - 1):
        a, b, lv = levels[k]
        pv, nv = levels[k - 1][2], levels[k + 1][2]
        short = (b - a) < (levels[k - 1][1] - levels[k - 1][0]) and (b - a) < (levels[k + 1][1] - levels[k + 1][0])
        if pv == nv and abs(lv - pv) <= 2 and short:
            orn.add(k)
    marked = [[a, b, lv, k in orn] for k, (a, b, lv) in enumerate(levels)]

    # ── 同じ高さの連続は、押し直しがあるときだけ割る ───────────
    notes: list[list] = []
    for a, b, lv, isorn in marked:
        if notes and notes[-1][2] == lv and not any(kind[g] == "無声" for g in range(notes[-1][1], a)):
            trough = float(db[notes[-1][1]:a].min()) if a > notes[-1][1] else float(db[a])
            after = db[a:min(n, a + int(0.15 * fps))]
            if not (after.size and float(after.max()) - trough >= REATTACK_DB):
                notes[-1][1], notes[-1][3] = b, notes[-1][3] and isorn
                continue
        notes.append([a, b, lv, isorn])

    # ── 余韻を落とさない: 切れ目を渡りの真ん中まで広げる ────────
    bounds = [[a, b] for a, b, _, _ in notes]
    for k in range(len(bounds) - 1):
        gap = range(bounds[k][1], bounds[k + 1][0])
        if not len(gap):
            continue
        silent = [g for g in gap if kind[g] == "無声"]
        if silent:
            bounds[k][1] = max(silent[0], bounds[k][1])
        else:
            mid = (bounds[k][1] + bounds[k + 1][0]) // 2
            bounds[k][1], bounds[k + 1][0] = mid, mid
    if bounds:
        tail = [i for i in range(bounds[-1][1], n) if kind[i] == "無声"]
        bounds[-1][1] = tail[0] if tail else n - 1

    # ── 開始を実際の立ち上がりへ寄せる（負けていたのは主に開始時刻）──
    snap_w = max(0.03, ioi_med * 0.25)
    onset_t = np.array([t for t in ons if db[min(int(t * fps), n - 1)] > gate])

    out: list[dict] = []
    for (s_i, e_i), (a, b, lv, isorn) in zip(bounds, notes):
        sel = [smooth[k] for k in range(a, b) if kind[k] == "安定"]
        if not sel:
            continue
        med = float(np.median(sel))
        s = float(times[s_i])
        if onset_t.size:
            near = onset_t[int(np.argmin(np.abs(onset_t - s)))]
            if abs(near - s) <= snap_w and (not out or near > out[-1]["start"] + 1e-3):
                s = float(near)
        e = float(times[min(e_i, n - 1)])
        if out:
            out[-1]["end"] = max(out[-1]["start"] + MIN_NOTE, min(out[-1]["end"], s))
        if e - s < MIN_NOTE:
            continue
        stable = float(times[b - 1] - times[a])
        out.append({"start": round(s, 3), "end": round(e, 3), "midi": int(round(med)),
                    "cent_dev": round((med - round(med)) * 100, 1), "hold": round(stable, 3),
                    "conf": round(min(1.0, stable / max(0.12, med_len / fps)), 2),
                    "ornament": bool(isorn), "db": round(float(np.median(db[s_i:e_i])), 1)})

    return {"audio": Path(src).name, "duration": round(len(y) / sr, 2), "notes": out,
            "voice": _voice_kind(y, sr, out, f_med),
            "calib": {"f_med_hz": round(f_med, 1), "range_hz": [round(fmin), round(fmax)],
                      "vib_hz": round(vib_hz, 2), "vib_depth_cent": round(vib_depth * 100, 1),
                      "drift_th": round(drift_th, 2), "ioi_med": round(ioi_med, 3)}}


def _voice_kind(y, sr, notes, f_med: float) -> dict:
    """鼻歌か口笛か肉声か。**口笛はほぼ正弦波で倍音が無い**——そこで分かれる。

    確実に言えるのは「口笛かどうか」まで。**鼻歌と肉声の切り分けは、歌詞が採れるかで決める**
    （肉声なら歌詞がどのみち要るので二度手間にならない）。境目は4本で測った暫定値
    （口笛 0.01 / 鼻歌 0.11・0.34 / 肉声 0.63）。
    """
    if not notes:
        return {"kind": "分からない", "h2h1": 0.0, "f_med_hz": round(f_med, 1)}
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    ratios = []
    for nt in notes:
        f = float(librosa.midi_to_hz(nt["midi"]))
        if f * 2 > freqs[-1]:
            continue
        a = int((nt["start"] + 0.02) * sr / 512)
        b = max(a + 1, int((nt["end"] - 0.02) * sr / 512))
        seg = S[:, a:b]
        if not seg.size:
            continue

        def band(center: float) -> float:
            lo, hi = np.searchsorted(freqs, center * 0.94), np.searchsorted(freqs, center * 1.06)
            return float(seg[lo:max(hi, lo + 1)].max()) if hi > lo else 0.0

        h1 = band(f)
        if h1 > 0:
            ratios.append(band(f * 2) / h1)
    h2h1 = float(np.median(ratios)) if ratios else 0.0
    kind = ("口笛" if (f_med > 700 or h2h1 < 0.05)
            else "鼻歌" if h2h1 < 0.45 else "肉声かもしれない")
    return {"kind": kind, "h2h1": round(h2h1, 2), "f_med_hz": round(f_med, 1)}


def to_sheet(res: dict, *, lyrics: str = "") -> str:
    """採譜の結果を、旋律工程へ渡す**1枚のテキスト**にする。

    ★**小節線は入れない。** 単独の鼻歌から拍は読めない（実測で、どの BPM を当てても
      でたらめと大差ない当たりしか出なかった）。**曲側の拍に合わせにいくほうが確実**なので、
      長さは「この録音のいちばん短い音＝1単位」の相対で書き、読み替えてもらう。
    """
    notes = res["notes"]
    if not notes:
        return "【マスターから渡された音】読み取れる音がなかった（無音か、声として拾えなかった）。"
    durs = [nt["end"] - nt["start"] for nt in notes]
    unit = max(0.08, float(np.percentile(durs, 10)))
    toks: list[str] = []
    for i, nt in enumerate(notes):
        ln = int(np.clip(round((nt["end"] - nt["start"]) / unit), 1, 16))
        toks.append(f"{_abc_pitch(nt['midi'])}{ln if ln > 1 else ''}")
        if i + 1 < len(notes):
            rest = int(round((notes[i + 1]["start"] - nt["end"]) / unit))
            if rest >= 1:
                rest = min(rest, 16)
                toks.append(f"z{rest if rest > 1 else ''}")   # 1単位は数字を書かない（記法の約束）
    weak = [i + 1 for i, nt in enumerate(notes) if nt["conf"] < 0.5]
    orn = [i + 1 for i, nt in enumerate(notes) if nt["ornament"]]
    cents = [abs(nt["cent_dev"]) for nt in notes]
    v = res["voice"]

    lines = [f"【マスターが口ずさんだ音】{v['kind']}・{res['duration']}秒・{len(notes)}音",
             "",
             "L:1/16",
             " ".join(toks),
             "",
             f"・**1単位＝この録音でいちばん短い音**（約{unit*1000:.0f}ms）。"
             f"**曲の拍に読み替えてよい**——単独の口ずさみから拍は読めないので、小節線は入れていない",
             "・**調も曲に合わせて移してよい**（絶対の高さで書いてあるが、そこは意図ではない）",
             f"・高さのばらつき: 半音からのずれが中央 {np.median(cents):.0f}cent"
             f"（50cent を超えるとどちらの音か決まらない）"]
    if weak:
        lines.append(f"・**確かでない音**（短くて高さが決まりきらない）: {weak} 番目")
    if orn:
        lines.append(f"・**しゃくり・ため**（同じ高さへ戻る短い寄り道。音符ではなく歌い方）: {orn} 番目")
    if lyrics:
        lines.append(f"・**歌っていた言葉**: {lyrics}")
    lines += ["",
              "**これは「だいたいこう歌いたい」の実演であって、正確な楽譜ではありません。**",
              "口ずさみ自体がずれていることもあり、読み取りの誤りも混じります。",
              "**音楽として不自然になるところは変えてください。**"
              "そのまま使える所は活かし、**変えたところは reason にどこをなぜ変えたかを書いてください**"
              "（マスターが「そこはそのままでよかった」と言えるように）。"]
    return "\n".join(lines)
