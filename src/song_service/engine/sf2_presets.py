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

"""**使う音源から音色一覧を読む**（LLM に見せるヒントを手書きしない）。

## なぜ必要か

音色の対応表を手で書いていたところ、**1つ多くて105番以降が全部ずれた**
（「琴を選んだのにカリンバが鳴る」状態。2026-08-22）。エラーは出ないので気づけない。

SF2 はファイル内部に**プリセット（音色）一覧**を持っているので、**音源から読めばずれない**。
さらに、ユーザーが `MIKU_SOUNDFONT` で音源を差し替えても、
**その音源に実在する音色だけ**を提示できる（「選んだのに鳴らない」が構造的に起きない）。

## 前提

| # | 前提 | 崩れると |
|---|---|---|
| 1 | 音源は SF2 形式（`sf2utils` で読める）| .sf3（圧縮）は読めないことがある → GM 既定表に落ちる |
| 2 | bank 0 が GM 準拠の128音色 | 並びが違う音源では番号と楽器がずれる |
| 3 | bank 128 がドラムキット（GM 慣習）| 無い音源ではドラムの選択肢が出ない |
| 4 | 読み取りは**起動時に1回**でよい（音源は変わらない）| 実行中に差し替えると古い一覧のまま |
"""
from __future__ import annotations

import functools
from pathlib import Path

# 読めなかったときの保険（GM の標準的な並び。timbre_map の手書き表を使う）
_FALLBACK_NOTE = "（音源から読めなかったため GM の標準的な並びを使用）"


@functools.lru_cache(maxsize=4)
def read_presets(sf2_path: str) -> dict[int, dict[int, str]]:
    """SF2 → {bank: {program: 音色名}}。読めなければ空 dict。"""
    p = Path(sf2_path)
    if not p.exists() or p.suffix.lower() != ".sf2":
        return {}
    try:
        from sf2utils.sf2parse import Sf2File
        out: dict[int, dict[int, str]] = {}
        with p.open("rb") as f:
            for pr in Sf2File(f).presets:
                name = getattr(pr, "name", None)
                if not name or name == "EOP":
                    continue
                out.setdefault(pr.bank, {})[pr.preset] = name
        return out
    except Exception:
        return {}


@functools.lru_cache(maxsize=16)
def drum_layout(sf2_path: str, program: int) -> dict[int, str]:
    """ドラムキット（bank 128・`program`）の**鍵盤ごとのサンプル名** {鍵盤: 名前}。読めなければ空 dict（2026-09-27）。

    ★**キットごとに鍵盤の中身が違う**。GeneralUser-GS のオーケストラ用キットは 41〜53 番がティンパニで、
      GM ではハイハット・シンバルの番号。**手で表を書くと音源を差し替えたときにずれる**ので、ここから読む
      （`timbre_map.kit_drums` がこの中身を見て、そのキットで鳴らせる打楽器を決める。別の番号へは運ばない・2026-09-29）。
    """
    p = Path(sf2_path)
    if not p.exists() or p.suffix.lower() != ".sf2":
        return {}
    try:
        from sf2utils.sf2parse import Sf2File
        out: dict[int, str] = {}
        with p.open("rb") as f:
            for pr in Sf2File(f).presets:
                if getattr(pr, "bank", None) != 128 or getattr(pr, "preset", None) != program:
                    continue
                for pb in pr.bags:
                    if pb.instrument is None:              # 全体の設定だけの区画
                        continue
                    plo, phi = pb.key_range or (0, 127)
                    for ib in pb.instrument.bags:
                        if ib.sample is None:
                            continue
                        lo, hi = ib.key_range or (0, 127)
                        for k in range(max(lo, plo), min(hi, phi) + 1):
                            out.setdefault(k, ib.sample.name)
                break
        return out
    except Exception:
        return {}


def melodic_menu(sf2_path: str, per_line: int = 4) -> str:
    """音程のある音色の一覧（bank 0）を、GM のカテゴリごとに並べた文字列。"""
    banks = read_presets(sf2_path)
    gm = banks.get(0) or {}
    if not gm:
        return ""
    cats = [(0, 7, "ピアノ"), (8, 15, "鍵盤打楽器・ベル"), (16, 23, "オルガン・アコーディオン"),
            (24, 31, "ギター"), (32, 39, "ベース"), (40, 47, "弦楽器・ハープ・ティンパニ"),
            (48, 55, "アンサンブル・合唱"), (56, 63, "金管"), (64, 71, "サックス・木管"),
            (72, 79, "笛・尺八"), (80, 87, "シンセリード"), (88, 95, "シンセパッド"),
            (96, 103, "シンセ効果音"), (104, 111, "民族楽器"), (112, 119, "打楽器・太鼓"),
            (120, 127, "効果音")]
    lines = []
    for lo, hi, label in cats:
        items = [f"{i + 1}:{gm[i]}" for i in range(lo, hi + 1) if i in gm]
        if items:
            lines.append(f"  【{label}】" + " / ".join(items))
    return "\n".join(lines)


def variation_menu(sf2_path: str, limit: int = 40) -> str:
    """bank 0 以外の**バリエーション音色**（音源固有の追加音色）。無ければ空文字。"""
    banks = read_presets(sf2_path)
    extras = []
    for b in sorted(banks):
        if b in (0, 128):
            continue
        for prog in sorted(banks[b]):
            extras.append(f"{b}-{prog + 1}:{banks[b][prog]}")
    if not extras:
        return ""
    shown = extras[:limit]
    more = f" …ほか{len(extras) - len(shown)}種" if len(extras) > len(shown) else ""
    return ("  【この音源だけの追加音色】**bank-番号** の形で指定します\n    "
            + " / ".join(shown) + more)


def drum_kits(sf2_path: str) -> dict[int, str]:
    """ドラムキット一覧（bank 128）。{program: 名前}。"""
    return read_presets(sf2_path).get(128, {})


def describe_source(sf2_path: str) -> str:
    """音源の素性（LLM に「何で鳴らすか」を伝える1行）。"""
    banks = read_presets(sf2_path)
    if not banks:
        return _FALLBACK_NOTE
    n = sum(len(v) for v in banks.values())
    return f"（音源: {Path(sf2_path).name}・{n}音色・{len(banks)}バンク）"


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import song_design                                     # noqa: F401（bootstrap）
    from sf2_render import soundfont

    sf = sys.argv[1] if len(sys.argv) > 1 else soundfont()
    print(f"■ {sf}\n{describe_source(sf)}\n")
    print(melodic_menu(sf))
    v = variation_menu(sf)
    if v:
        print("\n" + v)
    kits = drum_kits(sf)
    if kits:
        print(f"\n  【ドラムキット】{' / '.join(f'{k + 1}:{v}' for k, v in sorted(kits.items()))}")
