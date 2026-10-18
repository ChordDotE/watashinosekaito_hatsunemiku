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

"""声のトーン — LLM が選んだ感情語を、Fish Audio の PCM への加工に翻訳する。

Fish から届いた PCM を**届いたそばから**加工し、話す声の調子を変える。
Fish へのリクエストは一切変えない（感情タグも `prosody` も使わない）。

- **言葉8つ、音6つ**。1対1にしない。区別できない音を8種類持っても複雑さが増えるだけで、
  一方で言葉を減らすと LLM が自分の感情を見失う
- 動かすのは**話速・低域/高域・音量**の3種。声を分解して組み直す処理は通さないので、
  作り直しに由来する劣化（ざらつき・語尾切れ・継ぎ目）が起きない
- 3段とも**過去だけを見る**処理なので、刻み方で音が変わらない

`ToneHolder` は**あとからトーンを差し替えられる**。Fish のストリーマは
テキストが届く前に起動される（事前起動で -640ms）ため、`stream()` の引数では間に合わない。

仕様・数値の根拠・実測値は docs/plan/voice_tone_design.md が正。
見送った F0（声の高さ）版は docs/plan/voice_tone_f0_design.md。
"""
from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal

# --- 言葉（LLM が選ぶ）→ 音 -------------------------------------------------
# 「驚き」と「怒り」、「悲しい」と「申し訳ない」は、ラベルを伏せた出題で
# 互いに取り違えられた＝音として区別されていないため、それぞれ同じ音へ寄せる
TONE_TO_SOUND: Dict[str, str] = {
    "そのまま": "neutral",
    "嬉しい": "bright",
    "驚き": "strong",
    "怒り": "strong",
    "穏やか": "soft",
    "悲しい": "low",
    "申し訳ない": "low",
    "そっと伝えたい": "whisper",
}
VOICE_TONE_VALUES = list(TONE_TO_SOUND.keys())

# --- 音の設定（話速, 低域dB, 高域dB, 音量dB）--------------------------------
# 話速・低域・高域・音量はブラインドで有効を確認した値。
# 怒りの低域がマイナスなのは誤記ではない（感情音声では怒り・喜び・恐れで
# 低域が相対的に減る。プラスにすると「音量が大きいだけ」に聞こえる）
#
# 話速だけ一旦 1.00 に統一（2026-09-21・本人が実際の会話で速度変化に違和感を報告）。
# 低域/高域/音量は変えていない。ブラインドでは有効と出ていた値なので、
# 検証前の値を消さずコメントで残す（再検証して戻すか、正式に外すかは未判断）。
# 検証前の話速: bright 1.10 / strong 1.12 / soft 0.93 / low 0.89 / whisper 0.95
SOUNDS: Dict[str, Tuple[float, float, float, float]] = {
    "neutral": (1.00, 0.0, 0.0, 0.0),
    "bright": (1.00, -1.0, 4.0, 1.0),
    "strong": (1.00, -2.0, 5.0, 2.5),
    "soft": (1.00, 0.0, -1.5, -1.0),
    "low": (1.00, -1.5, -3.0, -3.5),
    "whisper": (1.00, -8.0, 1.0, -3.5),
}
DEFAULT_SOUND = "neutral"

#: 全ての音に共通で掛ける減衰（dB）。**上げる音がそのままだと必ず潰れる**ため。
#:
#: Fish の素の出力は実測で **-0.4 dBFS**（ほぼフルスケール）。そこへ `bright` の +1.0 dB や
#: `strong` の +2.5 dB を掛けると 1.0 を超え、`_to_bytes` の `np.clip` で頭が切り取られる。
#: 実測（2026-09-02・VM で同じ発話を録り比べ）: **トーン入りで 224 サンプルが 0 dBFS に張り付き、
#: 切ると 0 サンプル**。潰れた PCM がそのままブラウザへ届くので、再生側の音量では戻せない。
#:
#: リアルタイムなので「全体を見て正規化」はできない（先の音量が分からず、チャンクごとに
#: 音量が波打つ）。**固定の減衰**なら音量の揺れが起きず、音同士の相対関係もそのまま残る。
#: 小さく出るぶんは再生側で上げられる（2026-09-02 ユーザー判断）。
#: 0.0 にすれば従来どおりの動作に戻る。
HEADROOM_DB = -3.0
_HEADROOM_SCALE = 10 ** (HEADROOM_DB / 20)

#: ★**ヘッドルームだけでは足りない音がある**（2026-10-18 実測）。`strong`（驚き・怒り）は高域 +5dB と音量 +2.5dB が重なり、
#:   -3dB のヘッドルームを越えて **6本中 203 標本が 0 dBFS に張り付いた**（`test/fish_drama3_ab`・本人「たまにノイズが乗る」）。
#:   値（ブラインドで有効を確かめた音）は変えず、**出口でなめらかに頭を丸める**（ソフトリミット）。
#:   1標本ずつで完結する計算なので、0.25秒の刻み方で音が変わらず、継ぎ目も出ない（リアルタイムで先を見なくてよい）。
#:   `_KNEE` より小さい音は1ビットも変えない。それを超える頭だけを、1.0 に届かないように曲げる（硬く切るとプツッと鳴る）
_KNEE = 0.89                                               # -1 dBFS


def soft_limit(y: np.ndarray) -> np.ndarray:
    """`_KNEE` を超える所だけ tanh で曲げ、|y| < 1.0 に収める（境目で値も傾きもつながる）。"""
    a = np.abs(y)
    over = a > _KNEE
    if not over.any():
        return y
    out = y.copy()
    room = 1.0 - _KNEE
    out[over] = np.sign(y[over]) * (_KNEE + room * np.tanh((a[over] - _KNEE) / room))
    return out

_LOW_HZ = 300.0
_HIGH_HZ = 3000.0
_WIN_MS = 30.0
_SEARCH_MS = 10.0


def sound_of(tone: Optional[str]) -> str:
    """感情語を音の名前に直す。未知の値・None は neutral に落とす。"""
    if not tone:
        return DEFAULT_SOUND
    return TONE_TO_SOUND.get(tone.strip(), DEFAULT_SOUND)


class ToneHolder:
    """いま鳴らすべきトーンを保持する箱。**あとから差し替えられる**。

    Fish のストリーマは事前起動されるので、起動時点ではトーンが決まっていない。
    箱を先に渡しておき、`voice_tone` が届いた時点で `set()` する。
    最後まで届かなければ neutral のまま＝加工なしと同じ音になる。
    """

    def __init__(self, tone: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._sound = sound_of(tone)

    def set(self, tone: Optional[str]) -> str:
        with self._lock:
            self._sound = sound_of(tone)
            return self._sound

    @property
    def sound(self) -> str:
        with self._lock:
            return self._sound


class ToneProcessor:
    """0.25 秒ずつ渡される PCM を、状態を持ち越しながら加工する。

    話速（WSOLA）→ 低域/高域（IIR）→ 音量 の順。
    IIR はフィルタの内部状態を持ち越すのでチャンクの継ぎ目が出ない。
    """

    def __init__(self, fs: int, sound: str = DEFAULT_SOUND) -> None:
        rate, low_db, high_db, gain_db = SOUNDS.get(sound, SOUNDS[DEFAULT_SOUND])
        self.fs = fs
        self.rate = rate
        # 話速もフィルタも音量も変えない音（＝ neutral）。
        # **ヘッドルームは別に掛かる**ので、この場合でも音量だけは下がる（ToneStream.feed）
        self.identity = (rate == 1.0 and low_db == 0.0
                         and high_db == 0.0 and gain_db == 0.0)
        self.N = int(fs * _WIN_MS / 1000) // 2 * 2
        self.Hs = self.N // 2
        self.Ha = self.Hs * rate
        self.delta = int(fs * _SEARCH_MS / 1000)
        self.w = np.hanning(self.N)
        self.buf = np.zeros(0)
        self.origin = 0        # buf[0] が入力全体の何サンプル目か
        self.x_ideal = 0.0     # 理想の読み出し位置（探索結果を累積させない）
        self.prev_tail = None  # 直前に選んだ区間の後半。次の探索の照合先
        self.acc = np.zeros(self.N)
        self.accw = np.zeros(self.N)
        self.gain = 10 ** ((gain_db + HEADROOM_DB) / 20)
        self.sos_low = signal.butter(1, _LOW_HZ / (fs / 2), btype="low", output="sos")
        self.sos_high = signal.butter(1, _HIGH_HZ / (fs / 2), btype="high", output="sos")
        self.zi_low = signal.sosfilt_zi(self.sos_low) * 0
        self.zi_high = signal.sosfilt_zi(self.sos_high) * 0
        self.low_k = 10 ** (low_db / 20) - 1.0
        self.high_k = 10 ** (high_db / 20) - 1.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """1 次元配列を受け取り、加工して返す（まだ出せないときは空を返す）。"""
        if self.identity:
            return np.asarray(x, dtype=np.float64)
        self.buf = np.concatenate([self.buf, np.asarray(x, dtype=np.float64)])
        return self._color(self._stretch())

    def flush(self) -> np.ndarray:
        """発話の終わり。重ね合わせの途中を1回ぶんだけ吐き出す。

        ループが出しているのと同じ形（acc の先頭 Hs）を1つ出すだけ。
        これが無いと末尾が最大 15ms 切れる。
        """
        if self.identity:
            return np.zeros(0)
        y = self.acc[:self.Hs] / np.maximum(self.accw[:self.Hs], 1e-6)
        self.acc = np.zeros(self.N)
        self.accw = np.zeros(self.N)
        return self._color(y)

    def _stretch(self) -> np.ndarray:
        """WSOLA。波形を重ねて貼り直すだけなので音を作り変えない。

        踏んだ落とし穴が2つあるので、書き換えるときは注意すること:

        - **探索結果を読み出し位置に累積させない**。理想の位置（x_ideal）を別に持つ。
          累積させると遅くする側が効かなくなる
        - **右側の探索範囲を切らないだけの先読みを待つ**。以前は
          `hi` が `len(buf) - N - Hs` に丸められて探索が左に偏り、
          同じ場所を繰り返す（どもる）音になっていた。追加で待つのは Hs（15ms）ぶん
        """
        out = []
        while True:
            xp = int(self.x_ideal) - self.origin
            lo = max(0, xp - self.delta)
            hi = min(len(self.buf) - self.N - self.Hs, xp + self.delta)
            if hi < lo or xp + self.N + self.delta + self.Hs > len(self.buf):
                break
            if self.prev_tail is None:
                best = max(lo, min(xp, hi))
            else:
                cand = np.arange(lo, hi + 1)
                win = np.lib.stride_tricks.sliding_window_view(
                    self.buf[lo:hi + self.Hs], self.Hs)[:len(cand)]
                den = np.linalg.norm(win, axis=1) * np.linalg.norm(self.prev_tail) + 1e-9
                best = int(cand[int(np.argmax((win @ self.prev_tail) / den))])
            self.acc[:self.N] += self.buf[best:best + self.N] * self.w
            self.accw[:self.N] += self.w
            out.append(self.acc[:self.Hs] / np.maximum(self.accw[:self.Hs], 1e-6))
            self.acc = np.concatenate([self.acc[self.Hs:], np.zeros(self.Hs)])
            self.accw = np.concatenate([self.accw[self.Hs:], np.zeros(self.Hs)])
            self.prev_tail = self.buf[best + self.Hs:best + 2 * self.Hs]
            self.x_ideal += self.Ha
            keep = int(self.x_ideal) - self.origin - self.delta - self.N
            if keep > 0:
                self.buf = self.buf[keep:]
                self.origin += keep
        return np.concatenate(out) if out else np.zeros(0)

    def _color(self, y: np.ndarray) -> np.ndarray:
        """低域・高域を足し引きし、音量を掛ける。フィルタの状態は持ち越す。"""
        if len(y) == 0:
            return y
        if abs(self.low_k) > 1e-9:
            low, self.zi_low = signal.sosfilt(self.sos_low, y, zi=self.zi_low)
            y = y + low * self.low_k
        if abs(self.high_k) > 1e-9:
            high, self.zi_high = signal.sosfilt(self.sos_high, y, zi=self.zi_high)
            y = y + high * self.high_k
        return y * self.gain


class ToneStream:
    """PCM の bytes を受け取り、加工した bytes を返す（Int16 mono 前提）。

    トーンは箱から毎回読むので、発話の途中で差し替わっても追随する。
    ただし加工の状態（フィルタ・重ね合わせ）は音が変わると作り直す必要があるため、
    **差し替えは実際に音が変わったときだけ**反映する。
    """

    def __init__(self, fs: int, holder: ToneHolder) -> None:
        self.fs = fs
        self.holder = holder
        self._sound = holder.sound
        self._proc = ToneProcessor(fs, self._sound)

    def feed(self, pcm: bytes) -> bytes:
        self._sync()
        if self._proc.identity:
            # 話速もフィルタも要らない音（neutral）。**音量を下げるだけ**にして、
            # WSOLA も IIR も通さない。Int16 のまま掛けるので変換の往復も起きない
            if _HEADROOM_SCALE == 1.0:
                return pcm
            a = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
            return (a * _HEADROOM_SCALE).astype("<i2").tobytes()
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
        return self._to_bytes(self._proc.process(x))

    def flush(self) -> bytes:
        return self._to_bytes(self._proc.flush())

    def _sync(self) -> None:
        current = self.holder.sound
        if current != self._sound:
            self._sound = current
            self._proc = ToneProcessor(self.fs, current)

    @staticmethod
    def _to_bytes(y: np.ndarray) -> bytes:
        if not len(y):
            return b""
        return (np.clip(soft_limit(y), -1.0, 1.0) * 32767).astype("<i2").tobytes()
