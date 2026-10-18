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

"""
stt_smart_turn.py - Smart Turn v3 による意味的終話判定（P1）

VAD の無音検知は「音が止まった」しか分からないため、従来は固定タイマー
（stt.auto_send.delay_seconds = 5 秒）で「続きを言うかもしれない猶予」を置いていた。
本モジュールは無音検知の瞬間に発話音声の波形そのものから
「言い終わった話し方か（COMPLETE）／続きがある話し方か（INCOMPLETE）」を判定し、
final メッセージに `turn_complete` フラグを付与できるようにする。
ブラウザ（sttController.js）はフラグが true のとき固定タイマーなしで即送信する。

モデル: pipecat-ai/smart-turn-v3（BSD 2-Clause・ONNX・約8MB・日本語含む23言語）
  - 入力: 16kHz mono float32 の末尾8秒（不足分は先頭ゼロ詰め）→ Whisper log-mel 特徴量
  - 出力: sigmoid 確率（>threshold で COMPLETE）
  - CPU 推論 十数ms（PoC 実測。GPU 不要）

使い方（STT サービスから）:
    analyzer = SmartTurnAnalyzer.from_settings()   # サービス起動時に1回。無効なら None
    session = analyzer.new_session()               # WebSocket 接続ごとに1つ
    ...
    result = session.predict_pcm16(utterance_pcm_bytes)  # -> SmartTurnResult

継続発話に備え、session は判定に使った音声の末尾を保持し、次回の判定入力へ自動で連結する
（人間が「間を置いた続き」を一続きに聞くのと同じ扱い）。**COMPLETE でもリセットしない**
（2026-08-11。文脈なしの単独判定による誤 COMPLETE を防ぐため。詳細は predict_f32 のコメント）。
ONNX セッション（重い）は analyzer が共有し、継続バッファ（接続ごとの状態）は session が持つ
 — 複数ブラウザ接続でも混ざらない。

設定（settings.json）:
    stt.smart_turn.enabled    : true で有効（既定 false = 従来挙動）
    stt.smart_turn.threshold  : COMPLETE 判定の確率閾値（既定 0.5）
    stt.smart_turn.model_path : ONNX パス（既定 miku/data/models/smart_turn/smart-turn-v3.2-cpu.onnx）

エラー方針（サイレント失敗禁止）:
    enabled=true なのに onnxruntime 未導入・モデル欠損の場合、from_settings() は
    RuntimeError を送出する（STT サービス起動時に明示的に落とす）。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = _SRC_DIR.parent / "data" / "models" / "smart_turn" / "smart-turn-v3.2-cpu.onnx"

MODEL_SAMPLE_RATE = 16000
_WINDOW_SEC = 8  # モデル入力は末尾8秒


@dataclass
class SmartTurnResult:
    """1回の終話判定の結果。"""

    complete: bool
    probability: float
    elapsed_ms: float


def load_smart_turn_config(settings_path: Path | None = None) -> dict:
    """settings.json から stt.smart_turn セクションを読み、既定値とマージして返す。"""
    config = {
        "enabled": False,
        "threshold": 0.5,
        "model_path": str(_DEFAULT_MODEL_PATH),
    }
    path = settings_path or (_SRC_DIR / "settings.json")
    try:
        with open(path, encoding="utf-8-sig") as f:   # BOM 付きでも読む
            settings = json.load(f)
        section = settings.get("stt", {}).get("smart_turn", {})
        config.update({k: v for k, v in section.items() if k in config})
    except Exception as e:
        print(f"[SmartTurn] settings.json 読み込みエラー（無効として継続）: {e}")
        config["enabled"] = False
    return config


class SmartTurnAnalyzer:
    """Smart Turn v3 ONNX の判定器（スレッドセーフ）。

    ONNX セッションはロードに数百ms かかるため、サービス起動時に1回生成して
    接続をまたいで使い回す。推論はロックで直列化する（十数ms なので十分）。
    """

    def __init__(self, model_path: str | Path, threshold: float = 0.5):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime が見つかりません。`pip install onnxruntime` を実行するか、"
                "settings.json の stt.smart_turn.enabled を false にしてください"
            ) from e

        model_path = Path(model_path)
        if not model_path.exists():
            raise RuntimeError(
                f"Smart Turn モデルが見つかりません: {model_path}\n"
                "  インストーラをもう一度実行してモデルを取得するか、"
                "settings.json の stt.smart_turn.enabled を false にしてください"
            )

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(model_path), sess_options=so)

        self.threshold = float(threshold)
        self._lock = threading.Lock()  # ONNX 推論の直列化（複数接続からの同時呼び出し対策）
        print(f"[SmartTurn] モデルロード完了: {model_path.name} (threshold={self.threshold})")

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings_path: Path | None = None) -> "SmartTurnAnalyzer | None":
        """settings.json に従って生成する。無効なら None（呼び出し側は従来挙動）。"""
        config = load_smart_turn_config(settings_path)
        if not config["enabled"]:
            return None
        return cls(config["model_path"], config["threshold"])

    def new_session(self) -> "SmartTurnSession":
        """接続ごとの判定セッション（継続バッファ持ち）を作る。"""
        return SmartTurnSession(self)

    # ------------------------------------------------------------------
    # 推論（stateless・SmartTurnSession から呼ばれる）
    # ------------------------------------------------------------------

    def _infer(self, window: np.ndarray) -> float:
        """8秒ぶんの float32 波形に対する COMPLETE 確率を返す。"""
        from stt_whisper_features import compute_whisper_log_mel_features

        log_mel = compute_whisper_log_mel_features(window, do_normalize=True)
        input_features = np.expand_dims(log_mel, axis=0)
        with self._lock:
            outputs = self._session.run(None, {"input_features": input_features})
        return float(outputs[0][0].item())


class SmartTurnSession:
    """接続 1 本ぶんの終話判定セッション。

    INCOMPLETE 判定後の継続発話を直前の音声と繋げて再判定するため、
    判定に使った波形の末尾（最大8秒）を保持する。接続をまたいで共有しない。
    """

    def __init__(self, analyzer: SmartTurnAnalyzer):
        self._analyzer = analyzer
        self._tail = np.zeros(0, dtype=np.float32)

    def predict_pcm16(self, pcm: bytes) -> SmartTurnResult:
        """PCM Int16 (16kHz mono) の発話音声で終話判定する。"""
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return self.predict_f32(audio)

    def predict_f32(self, audio: np.ndarray) -> SmartTurnResult:
        """float32 (-1.0〜1.0, 16kHz mono) の発話音声で終話判定する。"""
        t0 = time.perf_counter()
        max_samples = _WINDOW_SEC * MODEL_SAMPLE_RATE

        combined = np.concatenate([self._tail, audio.astype(np.float32)])
        if len(combined) > max_samples:
            combined = combined[-max_samples:]
        window = combined
        if len(window) < max_samples:
            window = np.pad(window, (max_samples - len(window), 0),
                            mode="constant", constant_values=0)

        probability = self._analyzer._infer(window)
        complete = probability > self._analyzer.threshold

        # COMPLETE / INCOMPLETE のどちらでも波形を持ち越す（2026-08-11）。
        # 以前は COMPLETE でリセットしていたが、直後の発話が**文脈なしで単独判定**され、
        # 語尾が下がる短文を高確信度で誤って COMPLETE にしていた
        # （実会話で「の機能にさ。」を p=0.969 と誤判定 → ミクが話の途中で返事を始めた）。
        # A/B 実測（test/smart_turn_buffer_ab/）で誤 COMPLETE が3件とも解消し、悪化はゼロ。
        # 窓は末尾8秒固定なので推論コスト・メモリは変わらない。
        self._tail = combined

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return SmartTurnResult(complete=complete, probability=probability, elapsed_ms=elapsed_ms)

    def reset(self) -> None:
        """継続バッファを破棄する（接続終了時などに呼ぶ）。"""
        self._tail = np.zeros(0, dtype=np.float32)
