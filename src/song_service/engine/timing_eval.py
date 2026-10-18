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

"""作成物のタイミングを客観評価（エージェントの自己評価用）。

- 歌声 vs 伴奏 のオンセット包絡を窓ごとに相互相関 → 頭ずれ(global offset)と進行ずれ(drift)を測る。
- （参照ボーカルstemがあれば）歌声 vs 実ボーカルを chroma DTW で対応付け → 音色非依存で時間ずれを測る。
音楽はタイミング一致が要なので、機械評価(check_timing/similarity)に「実音の時間整合」を補う。
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

SR = 22050
HOP = 512


def _onset_env(path: Path, offset: float | None = None, dur: float | None = None) -> np.ndarray:
    if offset is None:
        y, _ = librosa.load(str(path), sr=SR, mono=True)
    else:
        y, _ = librosa.load(str(path), sr=SR, mono=True, offset=offset, duration=dur)
    return librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)


def cross_corr_drift(a: np.ndarray, b: np.ndarray, win_s: float = 4.0, max_lag_s: float = 0.5) -> dict:
    """a を基準に b の窓ごとのラグ(秒)を相互相関で測る。global offset（中央値）と drift（標準偏差）を返す。"""
    fpw, ml = int(win_s * SR / HOP), int(max_lag_s * SR / HOP)
    n = min(len(a), len(b))
    lags = []
    for st in range(0, n - fpw, fpw):
        aw = a[st:st + fpw] - a[st:st + fpw].mean()
        bw = b[st:st + fpw] - b[st:st + fpw].mean()
        if aw.std() < 1e-6 or bw.std() < 1e-6:
            continue
        c = np.correlate(aw, bw, "full")
        ctr = len(bw) - 1
        seg = c[ctr - ml:ctr + ml + 1]
        lags.append((np.argmax(seg) - ml) * HOP / SR)
    if not lags:
        return {"offset_sec": None, "drift_sec": None, "n": 0}
    lags = np.array(lags)
    return {"offset_sec": round(float(np.median(lags)), 3),
            "drift_sec": round(float(lags.std()), 3),
            "max_abs_sec": round(float(np.max(np.abs(lags))), 3), "n": len(lags)}


def chroma_dtw_offset(a: Path, b: Path, a_off=None, a_dur=None, b_off=None, b_dur=None) -> dict:
    """a,b の chroma を DTW で対応付け、経路上の時間差(b-a)の中央値/範囲を返す（音色非依存）。"""
    ya = librosa.load(str(a), sr=SR, mono=True, offset=a_off, duration=a_dur)[0] if a_off is not None \
        else librosa.load(str(a), sr=SR, mono=True)[0]
    yb = librosa.load(str(b), sr=SR, mono=True, offset=b_off, duration=b_dur)[0] if b_off is not None \
        else librosa.load(str(b), sr=SR, mono=True)[0]
    ca = librosa.feature.chroma_cqt(y=ya, sr=SR, hop_length=HOP)
    cb = librosa.feature.chroma_cqt(y=yb, sr=SR, hop_length=HOP)
    _, wp = librosa.sequence.dtw(X=ca, Y=cb, metric="cosine")
    wp = wp[::-1]
    off = (wp[:, 1] - wp[:, 0]) * HOP / SR
    p10, p90 = np.percentile(off, [10, 90])                # 端の伸ばし音/無音でDTWが暴れる外れ値を除いた“主部”範囲
    return {"offset_median_sec": round(float(np.median(off)), 3),
            "offset_p10_90_sec": [round(float(p10), 2), round(float(p90), 2)],
            "offset_range_sec": [round(float(off.min()), 2), round(float(off.max()), 2)],  # 参考（外れ値込み）
            "offset_std_sec": round(float(off.std()), 3)}


#: ★**形ごとの対象外**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §7-2）。歌と伴奏の時間の合い方を測る検査なので、
#:   インスト（歌が無い）は測らない。アカペラ（伴奏が無い）も歌↔伴奏は測らない——前の形の伴奏が曲フォルダに残っていても混ぜない
NOT_APPLICABLE = {"instrumental": "インストなので歌が無い（歌と伴奏の時間の合い方は測らない）",
                  "acappella": "アカペラなので伴奏が無い（歌と伴奏の時間の合い方は測らない）"}


def analyze_timing(song_dir: Path, form: str = "song") -> dict:
    """曲フォルダの timing を評価。歌声(vocal.wav or placeholder)↔伴奏、（あれば）歌声↔実ボーカルstem。
    `form`（曲の形）がインスト・アカペラなら測らずに対象外を返す（`{"skipped": 理由}`）。"""
    song_dir = Path(song_dir)
    if form in NOT_APPLICABLE:
        return {"song": song_dir.name, "skipped": NOT_APPLICABLE[form]}
    voc = song_dir / "vocal.wav"
    is_placeholder = not voc.exists()                      # 本物vocal.wavが無い＝仮歌(サイン)段階＝計測は参考値
    if is_placeholder:
        voc = song_dir / "placeholder.wav"
    acc = song_dir / "accompaniment.wav"
    out: dict = {"song": song_dir.name, "is_placeholder": is_placeholder}
    if voc.exists() and acc.exists():
        out["vocal_vs_accomp"] = cross_corr_drift(_onset_env(acc), _onset_env(voc))
    # 参照ボーカルstem（source/*.vocals.wav）があれば、**区間秒数で切って** DTW（フル尺と比べると無意味）
    import json as _json
    refs = list((song_dir / "source").glob("*.vocals.wav")) if (song_dir / "source").exists() else []
    ref_idx = song_dir / "source" / "reference.json"
    if voc.exists() and refs and ref_idx.exists():
        try:
            r = _json.loads(ref_idx.read_text(encoding="utf-8"))
            s, e = r.get("section_start_sec"), r.get("section_end_sec")
            if s is not None and e is not None:
                out["vocal_vs_reference"] = chroma_dtw_offset(voc, refs[0], b_off=float(s), b_dur=float(e) - float(s))
                voc_len = librosa.get_duration(path=str(voc))   # 尺の整合: 歌声長 vs 区間長（chroma範囲より頑健な破綻検知）
                out["length_mismatch_sec"] = round(abs(voc_len - (float(e) - float(s))), 2)
            else:
                out["vocal_vs_reference"] = {"skipped": "区間秒数(section_start/end_sec)が未保存→比較不可"}
        except Exception as ex:
            out["vocal_vs_reference"] = {"error": str(ex)}
    return out


def verdict(report: dict) -> str:
    """評価結果を短い文に。**仮歌段階は『参考値』**（サイン仮歌 vs 実声は計測ノイズが大きく、
    タイミングは align_vocal の再整列で確定済み＝ここで“要修正”に見えても再整列し直さない）。
    確定判定は本物 vocal.wav 書き出し後に行う。しきい値はオンセット相互相関のノイズ床(~0.27s)/頑健化を考慮。"""
    if report.get("skipped"):                              # 形で対象外（測っていない・「問題なし」ではない）
        return f"対象外——{report['skipped']}"
    placeholder = report.get("is_placeholder", False)
    lines = []
    va = report.get("vocal_vs_accomp")
    if va and va.get("offset_sec") is not None:            # 別content相互相関＝進行ぶれはノイジー。頭ずれのみ判定
        ok = abs(va["offset_sec"]) < 0.15
        lines.append(f"歌声↔伴奏: 頭ずれ{va['offset_sec']:+.2f}s（進行ぶれ{va['drift_sec']:.2f}s=参考）"
                     f" → {'OK' if ok else '頭ずれ注意'}")
    # 構造破綻は **尺の不一致（歌声長 vs 区間長）** で頑健に判定（chroma範囲は伸ばし音/端で自然に広がり誤検出する）
    lm = report.get("length_mismatch_sec")
    gross = lm is not None and lm > 3.0                    # 3s超のズレ＝尺崩壊（記号整列で膨張等）
    vr = report.get("vocal_vs_reference")
    if vr and "offset_median_sec" in vr:                   # 同content chroma-DTW。中央ずれ＋主部(p10-90)で追従を見る（参考）
        sp = vr.get("offset_p10_90_sec") or vr.get("offset_range_sec")
        ok = abs(vr["offset_median_sec"]) < 0.2 and (sp[1] - sp[0]) < 0.6
        lines.append(f"歌声↔実ボーカル: 中央ずれ{vr['offset_median_sec']:+.2f}s 主部範囲{sp}"
                     f"{f' 尺差{lm}s' if lm is not None else ''} → "
                     f"{'OK(追従)' if ok else '要確認(整列・参考)'}")
    body = " / ".join(lines) if lines else "タイミング評価対象なし"
    if gross:                                              # 尺崩壊は仮歌でも『参考』で隠さず必ず修正対象に
        return f"★【重大破綻・要修正】歌声長と区間長が{lm}sズレ＝尺崩壊。{body}（記号整列等で壊れた可能性→align_vocal をやり直す）"
    if placeholder:                                        # 仮歌＝参考。微小残差では修正ループに入らせない
        return (f"【仮歌・参考値／再整列で確定済み・微小残差では再alignしない】{body}"
                f"（確定評価は本物 vocal.wav 書き出し後）")
    return body
