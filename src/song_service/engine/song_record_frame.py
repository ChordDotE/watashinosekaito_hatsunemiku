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

"""**オリジナル曲の尺の変更が読む区間・調・コード**を、工程の記録からそのまま載せる（2026-10-03 本人決定・組「カバーのコード」）。

それまでオリジナル曲の尺の変更は、工程の記録を使わず、カバーと同じ道で区間を LLM に起こし直し（カバーの構成推定）、
コードを伴奏 MIDI から機械で推定し直していた（`song_cover_load.add_sections`・`add_machine_chords`）——コードの工程が書いた
コードがあるのに、伴奏から推定したもので置き換わっていた。いまは記録の**時間軸の区間**（設計の区間・歌の実長で決まった小節数）・
**区間の調**（転調を含む）・**コードの工程が書いたコード**（C 基準）を読み、組み立てと同じく区間の調へ移して実音で持つ。

記録が使えないとき（記録が無い古い曲・記録の楽譜といまの楽譜の音符の位置が違う＝前に尺を変えた曲など）だけ、呼び手（`resize_song`）が
いままでの推定へ戻る。戻ったことは返事と記録に出す（黙って推定しない）。手直しの移調（台帳の `組み立て.移調`）は記録の調に足す
（記録の調は手直し前・ハモリ追加 `editing._added_voice_basis` と同じ数え方）。
"""
from __future__ import annotations

from song_design import CHORD_BAR, Section, bar_ticks, transpose_chord
from song_model import Song


def from_record(song: Song, record) -> str:
    """工程の記録（`record`・オリジナル曲）の区間・調・コードを `song`（いまの楽譜を読んだもの）に載せる。
    載せたら空文字、使えなければ理由（呼び手が推定へ戻り、この理由を返事と記録に出す）。"""
    from song_cover_chords import moved_key, transposed
    from song_cover_load import place_sections
    from song_place import keys_shift
    from song_timeline import LEAD_IN
    if record is None:
        return "工程の記録が無い（古い曲）"
    tl, d, ch = getattr(record, "timeline", None), getattr(record, "design", None), getattr(record, "chords", None)
    d = d.model_dump() if hasattr(d, "model_dump") else d
    ch = ch.model_dump() if hasattr(ch, "model_dump") else ch
    if tl is None or not isinstance(d, dict) or not isinstance(ch, dict) or not ch.get("sections"):
        return "工程の記録に時間軸・設計・コードのどれかが無い（古い曲）"
    rs = getattr(record, "score", None)
    if rs is None or [n.start_tick for n in rs.vocal_notes] != [n.start_tick for n in song.score.vocal_notes]:
        return "工程の記録の楽譜と、いまの楽譜の音符の位置が違う（前に尺を変えた・音符を置き直した曲）"
    if tl.bar_ticks != bar_ticks(song.score.time_signature or "4/4"):
        return "工程の記録の拍子が、いまの楽譜と違う"
    spans = [s for s in tl.spans if s.kind != LEAD_IN]
    lead = sum(s.bars for s in tl.spans if s.kind == LEAD_IN)
    cells = {s["kind"]: list(s["chords_per_bar"]) for s in ch["sections"]}
    if any(len(cells.get(s.kind, [])) != s.bars for s in spans):
        return "工程の記録のコードの小節数が、時間軸の区間と合わない"
    off = transposed(getattr(record, "requests", None))
    key = moved_key(d.get("key") or "", off)
    # pitch_center は時間軸に無い（カバーの区間と同じく機械が以前の型の既定 mid を明示する・2026-10-04 組N2）
    sections = [Section(kind=s.kind, has_vocal=s.has_vocal, intensity=s.intensity, bars=s.bars, pitch_center="mid") for s in spans]
    place_sections(song, key, sections, lead)
    rk = getattr(record, "key_of", None) or {}
    song.key_of = {s.kind: moved_key(rk.get(s.kind) or d.get("key") or "", off) for s in spans}   # 転調した区間はその調
    shift_of = keys_shift(song.key_of, key)
    song.chords_of = {s.kind: [" ".join(transpose_chord(c, shift_of(s.kind), song.key_of[s.kind]) for c in CHORD_BAR.read(cell))
                               for cell in cells[s.kind]] for s in spans}
    return ""
