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

"""参照取得ツール: エージェントが自動実行できる DL 関数（手動 curl の代替）。

`download_reference` は URL を urllib で取得して reference/ に保存・検証する（バイナリ可）。
`fetch_midi_reference` は DL＋MIDI記譜化（音高＋音価の階名JSON）まで一括。
本番では reference_fetch サブエージェントが「検索でURL特定→このツールでDL」する。
"""
from __future__ import annotations

import urllib.parse
import urllib.request
from pathlib import Path

from midi_to_reference import build_reference

REF_DIR = Path(__file__).parents[1] / "reference"      # service直下（旧固定素材置き場・存在しない環境では未使用）
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}


def download_reference(url: str, filename: str, *, timeout: int = 30, out_dir=None) -> dict:
    """URL を DL して <out_dir or reference>/<filename> に保存。検証情報を返す。"""
    d = Path(out_dir) if out_dir else REF_DIR              # 曲ごとの source/ を渡せる
    d.mkdir(parents=True, exist_ok=True)
    dest = d / filename
    p = urllib.parse.urlparse(url)
    headers = {**_HEADERS, "Accept": "*/*",                # ホットリンク保護(403)対策に Referer 等を付与
               "Referer": f"{p.scheme}://{p.netloc}/"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    dest.write_bytes(data)
    head = data[:4]
    looks_html = data.lstrip()[:1] == b"<"
    return {"path": str(dest), "size": len(data), "head_hex": head.hex(),
            "is_midi": head == b"MThd", "looks_html": looks_html,
            "ok": len(data) > 0 and not looks_html}


def fetch_midi_reference(url: str, filename: str = "reference.mid", *, channel: int = 0) -> dict:
    """MIDI 参照を DL → 主旋律（音高＋音価）の階名JSONを作成して返す。"""
    dl = download_reference(url, filename)
    if not dl["is_midi"]:
        return {"ok": False, "reason": f"MIDIでない（head={dl['head_hex']} html={dl['looks_html']}）", **dl}
    ref = build_reference(dl["path"], channel)
    return {"ok": True, **dl, **ref}


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(fetch_midi_reference(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "reference.mid"),
                     ensure_ascii=False, indent=2))
