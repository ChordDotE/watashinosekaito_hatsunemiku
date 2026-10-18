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

"""MIDI（記譜）→ 参照階名列。採譜不要・誤差ゼロの正解参照を作る。

mu-tech の原曲 MIDI など、メロディが特定チャンネルにある曲から主旋律を抽出して
<midi>.solfa.json（transcription サイドカーと同形式）を書く。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mido

_INV = {0: "ド", 1: "ド#", 2: "レ", 3: "レ#", 4: "ミ", 5: "ファ",
        6: "ファ#", 7: "ソ", 8: "ソ#", 9: "ラ", 10: "ラ#", 11: "シ"}


def midi_to_solfa(k: int) -> str:
    return f"{_INV[k % 12]}{k // 12 - 1}"


def extract_melody(path: Path, channel: int = 0) -> tuple[list[tuple[int, int, int]], int, int]:
    """(start_tick, end_tick, pitch) の列、ticks_per_beat、拍子の分子(1小節の拍数) を返す。"""
    mid = mido.MidiFile(str(path))
    notes, active, t, num = [], {}, 0, 4
    for msg in mido.merge_tracks(mid.tracks):
        t += msg.time
        if msg.type == "time_signature":
            num = msg.numerator
        if msg.type == "note_on" and msg.velocity > 0 and getattr(msg, "channel", -1) == channel:
            active[msg.note] = t
        elif msg.type in ("note_off", "note_on") and getattr(msg, "channel", -1) == channel and msg.note in active:
            notes.append((active.pop(msg.note), t, msg.note))
    notes.sort()
    return notes, mid.ticks_per_beat, num


def _quantize(beats: float, grid: float = 0.5) -> float:
    """音価を grid（既定0.5拍）に量子化（最小 grid）。"""
    return max(grid, round(beats / grid) * grid)


def build_reference(midi_path: Path | str, channel: int = 0, *,
                    min_dur: float = 0.12, quantize: float = 0.5) -> dict:
    """MIDI から主旋律（音高＋音価）を抽出し <midi>.solfa.json を書く。情報を返す。

    min_dur 未満の極小ノート（グレース/重なり由来）は除去。音価は quantize 拍に量子化。
    """
    path = Path(midi_path)
    raw, tpb, bpb = extract_melody(path, channel)
    raw = [(s, e, m) for s, e, m in raw if (e - s) / tpb >= min_dur]   # 極小音除去
    midis = [m for _, _, m in raw]
    solfa = [midi_to_solfa(m) for m in midis]
    durs = [_quantize((e - s) / tpb, quantize) for s, e, _ in raw]     # 音価(拍)→量子化
    grid = 0.25
    starts = [round((s / tpb) / grid) * grid for s, _, _ in raw]       # 開始位置も拍グリッドへスナップ（人間ズレ除去）
    notes = [[m, d, st] for m, d, st in zip(midis, durs, starts)]      # [pitch, dur_beats, start_beat]
    # 休符（前音の終わり→この音の始まり）でフレーズ境界を検出
    breaks = [0]
    for i in range(1, len(raw)):
        if (raw[i][0] - raw[i - 1][1]) / tpb >= 0.75:                  # 0.75拍以上空いたら切れ目
            breaks.append(i)
    out = path.with_suffix(".solfa.json")
    out.write_text(json.dumps({"audio": path.name, "channel": channel, "ticks_per_beat": tpb,
                               "beats_per_bar": bpb, "solfa": solfa, "midi": midis,
                               "notes": notes, "phrase_breaks": breaks},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    return {"reference_json": str(out), "note_count": len(notes), "beats_per_bar": bpb,
            "solfa": " ".join(solfa), "durations": durs, "phrase_breaks": breaks}


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "reference/momotaro_guitar.mid")
    ch = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    info = build_reference(path, ch)
    print(f"ch{ch} 主旋律 {info['note_count']}音 → {info['reference_json']}")
    print("階名:", info["solfa"])
    print("音価(拍):", " ".join(str(d) for d in info["durations"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
