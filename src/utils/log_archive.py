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

"""WF ログ（`miku/data/logs/`）の月次アーカイブ。

## なぜ要るのか

ログは WF 起動1回につき1ファイル、平均 935KB で積み上がる（[agent_logger.py] が書く）。
2026-08-29 の実測で **654.6MB / 717ファイル / 4か月**、2026-08 は29日で 223.8MB ＝
**年 1.6〜2.4GB ペース**。縮める仕組みも消す仕組みも無く、**利用者の環境でも同じ速度で増える**。

一方でログは A/B 検証と障害調査の材料なので、**捨てたくない**。
そこで「消す」のではなく「畳む」。

## なぜ xz なのか（zip ではなく）

**同じ会話本文が1つのログの中に何度も書かれている**。実測で 2.0MB のログ1本に
「会話の本文」が9回出る（`input.user_prompt` にも `output.context` にも
`output.recent_conversation` にも、ループごとに載るため）。

  zip / gzip / tar.gz   654.6MB → 182.5MB（27.9%）  ← 窓が 32KB しかなく重複を見つけられない
  月ごと xz             654.6MB →   1.8MB（0.27%）  ← 辞書が大きいので畳める

`preset=3` / 辞書 `64MB` を使う。preset=6 との差は 124MB あたり 0.13MB しかないのに
**所要は 7分の1**（16.0s → 2.2s）で、これは**利用者の PC で毎月走る**処理だから。
辞書を 64MB に抑えるのは、**展開時のメモリが辞書サイズにほぼ比例する**ため。

## 消す前に必ず検証する

実データの削除を伴うので、順番を固定する:

  ① `YYYY-MM.tar.xz.tmp` を作る
  ② 開き直して検証（**エントリ名の集合とサイズが元と一致するか**）
  ③ 通ったら正式名へ差し替える
  ④ **そこで初めて**元の `*.log` を消す

どこかで失敗したら `.tmp` を捨てて**元ファイルはそのまま残す**（次回に再挑戦）。
既存アーカイブに追記するときは**同名エントリを二重に入れない** —— ③と④の間で落ちると
次回また同じファイルを足そうとするため、名前で弾く。

**ログのアーカイブがシステムを止めてよい理由は無い**ので、例外は呼び出し側で握る。
"""
from __future__ import annotations

import lzma
import os
import re
import tarfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

# agent_logger.py と同じ規約でパスを決める（settings.json は見ない）
LOGS_DIR = Path(__file__).resolve().parents[2] / "data" / "logs"
ARCHIVE_DIR = LOGS_DIR / "archive"

DEFAULT_KEEP_DAYS = 30      # 生で残す日数（settings.json: logs.keep_days）
_PRESET = 3                 # 圧縮の強さ（実測で速度と率のつり合いが最良）
_DICT_SIZE = 64 * 1024 * 1024   # 辞書サイズ。展開時のメモリもこれに比例する

# ミクメイン応答_20260829_214829_732_mode2.log → 20260829
_STAMP = re.compile(r"_(\d{8})_\d{6}_")


def _log_date(path: Path) -> datetime:
    """ログの日付。ファイル名から取り、取れなければ更新時刻で代用する。"""
    m = _STAMP.search(path.name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d")
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def collect_targets(keep_days: int = DEFAULT_KEEP_DAYS,
                    now: datetime | None = None,
                    logs_dir: Path | None = None) -> dict[str, list[Path]]:
    """`keep_days` より古い `*.log` を YYYY-MM ごとにまとめて返す。

    新しいものを触らないのは容量の都合ではなく、**いま書き込み中のファイルに
    手を出さないため**でもある（ログは WF 開始時に開き、終了まで開きっぱなし）。
    """
    logs_dir = logs_dir or LOGS_DIR
    now = now or datetime.now()
    cutoff = now - timedelta(days=keep_days)

    groups: dict[str, list[Path]] = {}
    if not logs_dir.is_dir():
        return groups
    for path in sorted(logs_dir.glob("*.log")):
        if not path.is_file():
            continue
        stamp = _log_date(path)
        if stamp >= cutoff:
            continue
        groups.setdefault(stamp.strftime("%Y-%m"), []).append(path)
    return groups


def _open_read(path: Path) -> tarfile.TarFile:
    """アーカイブを読む。展開側は xz のヘッダから設定を読むので指定は要らない。"""
    return tarfile.open(path, "r:xz")


@contextmanager
def _open_write(path: Path):
    """アーカイブを書く。

    `tarfile.open(mode="w:xz")` は `preset` しか受け取らず **辞書サイズを指定できない**。
    ここは辞書サイズが圧縮率の要（ファイルをまたぐ重複を畳むため）なので、
    `lzma.LZMAFile` を自前で開いて tar を流し込む。両方を確実に閉じる必要があるため
    コンテキストマネージャにしてある。
    """
    raw = lzma.LZMAFile(
        path, "wb", format=lzma.FORMAT_XZ,
        filters=[{"id": lzma.FILTER_LZMA2, "preset": _PRESET, "dict_size": _DICT_SIZE}],
    )
    try:
        with tarfile.open(fileobj=raw, mode="w|") as tar:
            yield tar
    finally:
        raw.close()


def archive_month(ym: str, files: list[Path],
                  archive_dir: Path | None = None) -> tuple[int, int]:
    """1か月ぶんをアーカイブし、検証を通ったら元ファイルを消す。

    Returns:
        (畳んだファイル数, 減ったバイト数)。何もしなかった場合は (0, 0)。

    Raises:
        RuntimeError: 検証に通らなかったとき（元ファイルは消さずに残す）。
    """
    archive_dir = archive_dir or ARCHIVE_DIR
    archive_dir.mkdir(parents=True, exist_ok=True)
    final = archive_dir / f"{ym}.tar.xz"
    tmp = archive_dir / f"{ym}.tar.xz.tmp"

    # 既存アーカイブに入っている名前は足さない（③と④の間で落ちた場合の二重登録を防ぐ）
    existing: dict[str, int] = {}
    if final.exists():
        with _open_read(final) as tar:
            existing = {m.name: m.size for m in tar.getmembers() if m.isfile()}

    adding = [p for p in files if p.name not in existing]
    if not adding and final.exists():
        # 既に全部入っている＝前回の④が終わる前に落ちた。元を消して整合させる
        freed = sum(p.stat().st_size for p in files)
        for p in files:
            p.unlink()
        return len(files), freed
    if not adding:
        return 0, 0

    expected = dict(existing)
    for p in adding:
        expected[p.name] = p.stat().st_size

    # ① 作る
    if tmp.exists():
        tmp.unlink()
    try:
        with _open_write(tmp) as out:
            if final.exists():
                with _open_read(final) as old:
                    for member in old.getmembers():
                        if not member.isfile():
                            continue
                        src = old.extractfile(member)
                        if src is None:
                            continue
                        out.addfile(member, src)
            for p in adding:
                out.add(p, arcname=p.name)

        # ② 開き直して検証する
        with _open_read(tmp) as check:
            actual = {m.name: m.size for m in check.getmembers() if m.isfile()}
        if actual != expected:
            missing = set(expected) - set(actual)
            odd = {k for k in set(expected) & set(actual) if expected[k] != actual[k]}
            raise RuntimeError(
                f"{ym}: アーカイブの検証に失敗（欠落 {len(missing)}件・"
                f"サイズ不一致 {len(odd)}件）。元ファイルは残します"
            )
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    # ③ 差し替える（os.replace は同一ボリュームで原子的）
    os.replace(tmp, final)

    # ④ ここで初めて元を消す
    freed = sum(p.stat().st_size for p in adding)
    for p in adding:
        p.unlink()
    return len(adding), freed


def run(keep_days: int = DEFAULT_KEEP_DAYS,
        now: datetime | None = None,
        logs_dir: Path | None = None,
        archive_dir: Path | None = None) -> dict:
    """アーカイブを1回実行する。月ごとに独立して処理し、失敗しても他の月は続ける。

    Returns:
        {"months": 処理した月数, "files": 畳んだ件数, "freed": 減ったバイト数,
         "errors": [エラーメッセージ, ...]}
    """
    groups = collect_targets(keep_days, now, logs_dir)
    result = {"months": 0, "files": 0, "freed": 0, "errors": []}
    for ym in sorted(groups):
        try:
            count, freed = archive_month(ym, groups[ym], archive_dir)
        except Exception as exc:
            result["errors"].append(f"{ym}: {exc}")
            continue
        if count:
            result["months"] += 1
            result["files"] += count
            result["freed"] += freed
    return result
