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

"""マスターの声の登録・照合（話者照合）。

speechbrain ECAPA-TDNN（192次元・CPU 動作）でマスターの声の埋め込みを管理する。
顔（face_engine）と同じく **本人判定はせず一致度だけを返し、解釈は LLM に委ねる**。

## モデル選定と数値の根拠（実測 2026-08-09・test/perception_latency/results_speaker.md）

同性・同帯域の実話者24人（FLEURS ja_jp から F0 100〜160Hz を抽出）とマスターの実録音で
3案を比較し、本人の最低スコアと他人の最高スコアの差（マージン）で選んだ:

| 案 | マージン | 判定 |
|---|---|---|
| **speechbrain ECAPA-TDNN（採用）** | **+0.118** | 唯一実用に足りた |
| ONNX wespeaker | +0.021 | 際どい |
| resemblyzer GE2E | +0.002 | 実質分離しない |

合成音声だけで評価すると speechbrain が +0.502 と好成績に見えたが、実際の肉声を登録に使うと
resemblyzer は +0.002 まで落ちた。**評価素材を実声・同性・同帯域に揃えて初めて差が出る**。

## 一致度の読み方（同・実測）

音声の長さでスコアが大きく動くため、固定しきい値では判定できない:

| 音声長 | 本人 | 同帯域の他人 |
|---|---|---|
| 0.5秒 | +0.64 | -0.02 |
| 1.0秒 | +0.65 | -0.05 |
| 3.0秒 | +0.86 | +0.36 |
| 7.9秒 | — | +0.48 |

マスターの 1.0秒未満の発話は +0.19〜+0.62 に散らばり、**他人の長い発話（+0.48）を下回る**。
このため照合側は長さとセットで数値を出し、判断は LLM に任せる。
"""
from __future__ import annotations

import io
import pickle
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

_JST = timezone(timedelta(hours=9))

SR = 16000                  # 照合に使うサンプリングレート（ECAPA の学習条件）
EMBEDDING_DIM = 192
MIN_ENROLL_SEC = 5.0        # これ未満の音声では登録させない
RECOMMEND_ENROLL_SEC = 10.0 # 案内する推奨の長さ（実測では 10.9秒で登録した）
MIN_SCORE_SEC = 1.0         # これ未満は単独で照合しても数値が信用できない

# 区間分割（音響的な無音で切る）のパラメータ。実測 2026-08-09 で決めた値
_TOP_DB = 32.0              # 最大音量からこれだけ下がったら無音とみなす
_JOIN_SEC = 0.35            # これ未満の間は同じ発話として繋ぐ（言い淀みで切らない）
_MIN_SEG_SEC = 0.4          # これ未満の区間は捨てる（物音・リップノイズ）
# 短い区間を隣と連結して測るときの上限。これ以上離れた区間とは繋がない
_JOIN_LIMIT_SEC = 1.0
# 前借りと後借りの値がこれ以上食い違ったら、隣で話者が変わっているとみて数値を出さない
_BORROW_AGREE_DIFF = 0.15

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "voice"
_EMBEDDING_PATH = _DATA_DIR / "master_voice_embedding.pkl"
_SAMPLE_PATH = _DATA_DIR / "master_voice_sample.wav"   # 登録時の音声（確認用・16kHz mono）

_model_lock = threading.Lock()
_model = None


class VoiceError(Exception):
    """呼び出し側がユーザーへそのまま見せられる理由つきエラー。"""


# ── モデル ────────────────────────────────────────────────────

def defuse_lazy_modules() -> list[str]:
    """speechbrain が sys.modules に置く遅延ロード用ダミーを無害化する（戻り値は処理した名前）。

    **これを外すと顔照合がプロセスごと恒久的に壊れる**（2026-08-11 実測）。機序:

    1. speechbrain は k2 / spacy / flair など「入っていなくてよい」オプション依存を、
       属性アクセスで実 import を試みるダミーとして sys.modules に登録する（実測7個）
    2. torch は `register_fake` の登録元を記録するため `inspect.getframeinfo()` を呼び、
       その中の `inspect.getmodule()` が **sys.modules を全走査して `hasattr(m, '__file__')`**
       を確かめる。ここでダミーが発火し、未インストールの k2 で ImportError になる
    3. これが torchvision の Meta カーネル登録の**途中**で起きるため、登録済みの roi_align は
       torch のグローバル dispatcher（プロセス単位のシングルトン）に残ったまま、
       sys.modules だけが巻き戻る
    4. 以後 torchvision を再 import するたび「roi_align は登録済み」で必ず失敗し、
       顔照合（facenet-pytorch）がプロセスを再起動するまで直らない

    顔と声は同じ Recognition サービス、しかも Gateway と同一プロセス（内部スレッド起動）に
    同居するため、この衝突を避けられない。ロード順の固定では解決しない
    ——どちらが先に呼ばれるかはマスターの操作次第だから（実際 voice_enroll が先に走って壊れた）。

    判定はクラス名ではなく「`__file__` を持たないか」で行う（speechbrain の更新で
    クラス名が変わっても効き続けるため）。`hasattr` は**それ自体が発火させる**ので使わない。
    """
    defused = []
    for name, mod in list(sys.modules.items()):
        if not name.startswith("speechbrain"):
            continue
        d = getattr(mod, "__dict__", None)      # __dict__ は型スロット。__getattr__ を通らない
        if d is None or "__file__" in d:
            continue
        d["__file__"] = None
        defused.append(name)
    return defused


def _get_model():
    """ECAPA-TDNN を遅延初期化して返す（初回のみ数秒）。

    CPU 固定: 顔と同じく低頻度・短時間で足りるため VRAM は STT/TTS へ譲る。
    重み（約80MB）はセットアップ時に事前ダウンロード済み（保存先は
    HF_HOME=miku\\runtime\\models\\hf。setup/system_start が設定・フォルダ内隔離方針）。
    """
    global _model
    with _model_lock:
        if _model is None:
            from speechbrain.inference.speaker import EncoderClassifier
            # import した直後に無害化する（この後どのライブラリが inspect を使っても踏まないように）
            defused = defuse_lazy_modules()
            if not defused:
                # speechbrain 側の実装が変わって効かなくなった可能性。顔照合が壊れる予兆なので残す
                print("[声] speechbrain の遅延ダミーを無害化できませんでした"
                      "（顔照合が壊れる恐れ。voice_engine.defuse_lazy_modules を確認）")
            _model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(Path(__file__).resolve().parents[2] / "runtime" / "models"
                            / "speechbrain_ecapa"),
                run_opts={"device": "cpu"},
            )
    return _model


# ── 音声の読み込み ────────────────────────────────────────────

def decode_audio(data: bytes) -> tuple[np.ndarray, float]:
    """音声バイト列を 16kHz mono float32 にして返す。戻り値は (波形, 秒数)。

    ブラウザ側で 16kHz mono WAV へ変換してから送る運用だが、メンテナンス画面からは
    任意の音声が来るためここでもリサンプルする。libsndfile が読める形式
    （wav / mp3 / flac / ogg 等）に対応。m4a・aac は読めない（ffmpeg 非依存のため）。
    """
    import soundfile as sf

    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32")
    except Exception as e:
        raise VoiceError(
            "音声ファイルを読み取れませんでした（対応形式: wav / mp3 / flac / ogg）"
        ) from e

    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g)
    y = np.asarray(y, dtype=np.float32)
    return y, len(y) / SR


def to_wav_bytes(pcm: np.ndarray) -> bytes:
    """16kHz mono float32 を WAV バイト列にする（登録音声の保存用）。"""
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


# ── 埋め込み ──────────────────────────────────────────────────

def embed(pcm: np.ndarray) -> np.ndarray:
    """波形から話者埋め込み（192次元）を計算する。"""
    import torch
    with torch.no_grad():
        return _get_model().encode_batch(
            torch.from_numpy(np.asarray(pcm, dtype=np.float32)).unsqueeze(0)
        ).squeeze().numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ── 登録データの読み書き ──────────────────────────────────────

def load_embedding() -> tuple[np.ndarray | None, str | None]:
    """保存済みの埋め込みと登録日時を返す。未登録なら (None, None)。"""
    if not _EMBEDDING_PATH.exists():
        return None, None
    try:
        with _EMBEDDING_PATH.open("rb") as f:
            data = pickle.load(f)
        return data["embedding"], data.get("enrolled_at")
    except Exception:
        return None, None


def load_embedding_for_verify() -> np.ndarray | None:
    """照合用の埋め込みを返す。次元が合わなければ未登録扱い（作り直しが要る）。"""
    emb, _ = load_embedding()
    if emb is None:
        return None
    return emb if getattr(emb, "shape", (0,))[-1] == EMBEDDING_DIM else None


def save_embedding(emb: np.ndarray, pcm: np.ndarray, seconds: float) -> str:
    """埋め込みと登録音声を保存し、登録日時を返す。"""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    enrolled_at = datetime.now(_JST).isoformat()
    with _EMBEDDING_PATH.open("wb") as f:
        pickle.dump(
            {
                "embedding": emb,
                "enrolled_at": enrolled_at,
                "seconds": round(seconds, 2),
                "model": "speechbrain-spkrec-ecapa-voxceleb",
            },
            f,
        )
    _SAMPLE_PATH.write_bytes(to_wav_bytes(pcm))
    return enrolled_at


def delete_enrollment() -> None:
    for p in (_EMBEDDING_PATH, _SAMPLE_PATH):
        if p.exists():
            p.unlink()


def sample_path() -> Path:
    return _SAMPLE_PATH


# ── 公開処理 ──────────────────────────────────────────────────

def enroll(data: bytes) -> dict:
    """音声バイト列からマスターの声を登録する。

    短すぎる音声は VoiceError（登録は行わない）。実測では 10.9秒の音声で登録した。
    """
    pcm, seconds = decode_audio(data)
    if seconds < MIN_ENROLL_SEC:
        raise VoiceError(
            f"音声が短すぎます（{seconds:.1f}秒）。"
            f"{RECOMMEND_ENROLL_SEC:.0f}秒以上の音声で登録してください"
        )
    emb = embed(pcm)
    enrolled_at = save_embedding(emb, pcm, seconds)
    return {
        "status": "enrolled",
        "seconds": round(seconds, 2),
        "enrolled_at": enrolled_at,
        "short_warning": seconds < RECOMMEND_ENROLL_SEC,
    }


def acoustic_segments(pcm: np.ndarray) -> list[tuple[float, float]]:
    """RMS の無音で発話区間を切る（秒で返す）。

    **区間の境目は音声側で決める**のが方式の要。STT のタイムスタンプで切ると、
    話者交代の無音を1単語が飲み込んだときに切れない（実測 2026-08-09: 「うん」が
    18.40-19.28s の0.88秒語になり、相手とマスターが1区間に混ざった）。
    """
    frame, hop = 1024, 256
    if len(pcm) < frame:
        return [(0.0, len(pcm) / SR)]
    n = 1 + (len(pcm) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    r = np.sqrt(np.mean(pcm[idx] ** 2, axis=1))
    voiced = r > r.max() * (10 ** (-_TOP_DB / 20.0))

    spans, start = [], None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append([start * hop, min(len(pcm), i * hop + frame)])
            start = None
    if start is not None:
        spans.append([start * hop, len(pcm)])

    merged: list[list[int]] = []
    for s, e in spans:
        if merged and (s - merged[-1][1]) < _JOIN_SEC * SR:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s / SR, e / SR) for s, e in merged if (e - s) / SR >= _MIN_SEG_SEC]


def _score_segments(pcm: np.ndarray, spans: list[tuple[float, float]],
                    known: np.ndarray) -> list[dict]:
    """各区間のスコアを求める。単独で短すぎる区間は隣と連結して測る。

    連結の規則（実測 2026-08-09）:
      - MIN_SCORE_SEC 未満の区間だけが対象
      - 前後の**両方向**で測り、値が _BORROW_AGREE_DIFF 以上食い違ったら数値を出さない
        （話者が変わっていなければどちらから借りても同じ値になるはず。食い違う＝
        借りた声が混ざっている。素朴に近い方と繋ぐと嘘の数値が出る）
      - 連結長は両方向でぴったり MIN_SCORE_SEC に揃える。揃えないと「長いほど
        スコアが高い」性質のせいで、話者が変わっていないのに食い違って見える
      - 連結する音声は発話区間だけを繋ぐ（間の無音は入れない）
    """
    need = int(MIN_SCORE_SEC * SR)

    def cut(i: int) -> np.ndarray:
        s, e = spans[i]
        return pcm[int(s * SR): int(e * SR)]

    def extend(i: int, direction: int) -> np.ndarray | None:
        parts, total, j = [cut(i)], spans[i][1] - spans[i][0], i
        while total < MIN_SCORE_SEC:
            k = j + direction
            if not (0 <= k < len(spans)):
                return None
            gap = (spans[j][0] - spans[k][1]) if direction < 0 else (spans[k][0] - spans[j][1])
            if gap > _JOIN_LIMIT_SEC:
                return None
            parts.insert(0, cut(k)) if direction < 0 else parts.append(cut(k))
            total += spans[k][1] - spans[k][0]
            j = k
        joined = np.concatenate(parts)
        # 対象区間を残す側から need サンプルだけ取る（前借りなら末尾・後借りなら先頭）
        return joined[-need:] if direction < 0 else joined[:need]

    out: list[dict] = []
    for i, (s, e) in enumerate(spans):
        row = {"start": round(s, 2), "end": round(e, 2),
               "borrowed": False, "scored": True, "score": 0.0}
        if e - s >= MIN_SCORE_SEC:
            row["score"] = round(cosine(known, embed(cut(i))), 4)
            out.append(row)
            continue

        cands = [c for c in (extend(i, -1), extend(i, +1)) if c is not None]
        if not cands:
            row["scored"] = False          # 孤立した短い発話。前後にも頼れない
        else:
            row["borrowed"] = True
            scores = [cosine(known, embed(c)) for c in cands]
            if len(scores) == 2 and abs(scores[0] - scores[1]) > _BORROW_AGREE_DIFF:
                row["scored"] = False      # 前後で食い違う＝どちらの声か決められない
            else:
                row["score"] = round(sum(scores) / len(scores), 4)
        out.append(row)
    return out


def analyze(data: bytes) -> dict:
    """1ターン分の音声を区間に切り、区間ごとの一致度を返す（LLM 投入前の前処理）。

    テキストの割り当ては呼び出し側（STT）が STT のタイムスタンプで行う。
    ここは音声だけで完結する処理に閉じる。
    """
    known = load_embedding_for_verify()
    if known is None:
        return {"enrolled": False}
    pcm, seconds = decode_audio(data)
    spans = acoustic_segments(pcm)
    return {
        "enrolled": True,
        "seconds": round(seconds, 2),
        "segments": _score_segments(pcm, spans, known),
    }


def verify(data: bytes) -> dict:
    """音声バイト列とマスターの登録音声との一致度を返す。

    本人かどうかの判定はしない（顔と同じ方針）。長さでスコアが動くため
    `seconds` を必ず添え、`reliable` で「単独で測るに足る長さか」を示す。
    """
    known = load_embedding_for_verify()
    if known is None:
        return {"enrolled": False}
    pcm, seconds = decode_audio(data)
    score = cosine(known, embed(pcm))
    return {
        "enrolled": True,
        "score": round(score, 4),
        "seconds": round(seconds, 2),
        "reliable": seconds >= MIN_SCORE_SEC,
    }
