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

"""SOFA（歌用強制アライナ・日本語モデルJPN_Test2）アダプタ。

かな歌詞＋区間音声 → 各モーラの実測時刻。発話用wav2vec2 FAと違い**歌声で学習**されており、
伸ばし音・フレーズ末の頭を正しく取ることを期待（残ズレ4箇所の本命対策）。

- 実行系: sofa_venv(py3.10)の infer.py を subprocess で呼ぶ（poc統合venvとは隔離）
- 入力: <tmp>/seg0.wav + seg0.lab（辞書キー=ローマ字モーラの空白区切り）
- 出力: <tmp>/TextGrid/seg0.TextGrid の words 層 → モーラごとの (start,end)
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

SERVICE = Path(__file__).parents[1]                        # song_service/ 直下（venv・モデル等のデータ基準）
SOFA_PY = SERVICE / "sofa_venv" / "Scripts" / "python.exe"
SOFA_REPO = SERVICE / "sofa_repo"
CKPT = SERVICE / "sofa_model" / "step.100000.ckpt"
DICT = SERVICE / "sofa_model" / "japanese-extension-sofa.txt"
_SIL = {"", "SP", "AP", "EP", "sil", "pau", "br", "R"}


def kana_to_moras(kana: str) -> list[str]:
    """かな文字列→モーラ列（拗音/小書きは直前へ結合。1モーラ=VOCALOID 1ノート）。
    ★「ゎ」はつながない（2026-10-02・組み立ての `lyric_slots._SMALL_KANA` と同じ扱い・仕様書 §2）
    ★空白・句読点を落とさない（2026-10-02・案B）。かな以外は歌詞のかなの型（`align_cover._kana_written`）が差し戻す
    ★改行は**定義された区切り**＝元の歌詞の行の切れ目（2026-10-03・仕様書 §2）なのでモーラに数えない（`kana_text_chars`）"""
    from lyric_slots import _SMALL_KANA, kana_text_chars    # 小書きの一覧は組み立てと同じ1つ
    small = set(_SMALL_KANA)
    out: list[str] = []
    for c in kana_text_chars(kana):
        if c in small and out:
            out[-1] += c
        else:
            out.append(c)
    return out


def moras_to_keys(moras: list[str]) -> list[str]:
    """モーラ→SOFA辞書キー（ローマ字）。ん→N・っ→cl・ー→直前母音。キーは辞書実在を検証。"""
    import pykakasi
    kks = pykakasi.kakasi()
    keys = {line.split("\t")[0].split()[0] for line in DICT.read_text(encoding="utf-8").splitlines() if line.strip()}
    res: list[str] = []
    for m in moras:
        if m == "ん":
            res.append("N")
            continue
        if m == "っ":
            res.append("cl")
            continue
        if m == "ー":
            res.append(res[-1][-1] if res and res[-1][-1] in "aiueo" else "a")
            continue
        r = re.sub(r"[^a-z]", "", "".join(x.get("hepburn", "") for x in kks.convert(m)).lower())
        if r in keys:
            res.append(r)
        elif r == "n":
            res.append("N")
        elif r and r[-1] in "aiueo" and r[-1] in keys:          # 未知キーは母音へ縮退（1:1対応は維持）
            res.append(r[-1])
        else:
            res.append("a")
    return res


def _parse_textgrid_tier(path: Path, tier: str) -> list[tuple[float, float, str]]:
    txt = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(rf'name = "{tier}"(.*?)(?:item \[|\Z)', txt, re.S)
    if not m:
        return []
    ivs = re.findall(r'xmin = ([\d.eE+-]+)\s*xmax = ([\d.eE+-]+)\s*text = "([^"]*)"', m.group(1))
    return [(float(a), float(b), t) for a, b, t in ivs]


_VOWELS = {"a", "i", "u", "e", "o", "N"}


def _vowel_starts(words: list[tuple[float, float, str]], phones: list[tuple[float, float, str]]) -> list[float]:
    """各語の note開始時刻＝**語内の最初の母音音素の開始**（子音は拍前に出る歌唱/VOCALOID慣習に合わせる）。
    母音が見つからない語は語頭のまま。"""
    out = []
    for a, b, _ in words:
        vs = a
        for pa, pb, pt in phones:
            if pa >= a - 1e-4 and pa < b and pt in _VOWELS:
                vs = pa
                break
        out.append(vs)
    return out




def sofa_align(vocal_wav: str, kana: str, clip_t0: float, end_sec: float,
               log: list | None = None, vowel_start: bool = False) -> list | None:
    """区間[clip_t0,end_sec]をSOFAで整列。返り値: [{char(モーラ), start, end(clip_t0基準秒), midi:None}] / 失敗None。"""
    import soundfile as sf
    moras = kana_to_moras(kana)
    keys = moras_to_keys(moras)
    tmp = SERVICE / "_sofa_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    y, sr = sf.read(vocal_wav, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    sf.write(tmp / "seg0.wav", y[int(clip_t0 * sr):int(end_sec * sr)], sr)
    (tmp / "seg0.lab").write_text(" ".join(keys), encoding="utf-8")
    r = subprocess.run([str(SOFA_PY), "infer.py", "-c", str(CKPT), "-f", str(tmp),
                        "-d", str(DICT), "-of", "textgrid"],
                       cwd=str(SOFA_REPO), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=3600)
    tg = tmp / "TextGrid" / "seg0.TextGrid"
    if not tg.exists():
        if log is not None:
            log.append(f"[SOFA] 失敗 rc={r.returncode} stderr末尾: {(r.stderr or '')[-300:]}")
        return None
    ivs = [(a, b, t) for a, b, t in _parse_textgrid_tier(tg, "words") if t not in _SIL]
    if vowel_start and ivs:                                    # note開始=母音開始（子音は拍前・VOCALOID/UST慣習）
        phones = [(a, b, t) for a, b, t in _parse_textgrid_tier(tg, "phones") if t not in _SIL]
        vstarts = _vowel_starts(ivs, phones)
        ivs = [(vs, b, t) for (a, b, t), vs in zip(ivs, vstarts)]
    if len(ivs) != len(moras):                                 # 語数不一致→順序difflibで対応（多少の欠落に耐える）
        import difflib
        smk = difflib.SequenceMatcher(None, keys, [t for _, _, t in ivs], autojunk=False)
        pairs = {}
        for a, b, sz in smk.get_matching_blocks():
            for k in range(sz):
                pairs.setdefault(a + k, b + k)
        if len(pairs) < len(moras) * 0.8:
            if log is not None:
                log.append(f"[SOFA] 語数不一致が大きい: 入力{len(moras)} 出力{len(ivs)} 対応{len(pairs)} → 不採用")
            return None
        chars = []
        for i, m in enumerate(moras):
            if i in pairs:
                a, b, _ = ivs[pairs[i]]
                chars.append({"char": m, "start": a, "end": b, "midi": None})
        if log is not None:
            log.append(f"[SOFA] 整列 {len(chars)}/{len(moras)}音（difflib対応・{len(moras) - len(chars)}音欠落）")
        return chars or None
    chars = [{"char": m, "start": a, "end": b, "midi": None} for m, (a, b, _) in zip(moras, ivs)]
    if log is not None:
        log.append(f"[SOFA] 整列 {len(chars)}音（歌用アライナ・語数完全一致）")
    return chars
