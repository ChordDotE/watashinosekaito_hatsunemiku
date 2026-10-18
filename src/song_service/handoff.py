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

"""完了申し送りの生成: 作業ログを読んで「ミク本人への短いメモ」を書く。

song_runner が完了通知（Gateway → mode=3）に載せる本文。最終報告はマスター向けの長文なので、
そのままだと申し送りとして冗長（先頭で切れる）。ここで作業記録（`log_build.txt`）の要点と併せて要約し直す。
**LLM 呼び出しに失敗しても通知は止めない**（最終報告の先頭へフォールバック）。
"""
from __future__ import annotations

from pathlib import Path

import song_files

SVC = Path(__file__).resolve().parent


def _log_digest(song_id: str, limit: int = 3000) -> str:
    """作業記録から申し送りの材料（ツールの入出力・警告・検問所見）を末尾中心に拾う。"""
    p = SVC.parents[1] / "data" / "songs" / song_id / song_files.BUILD_LOG
    if not p.exists():
        return ""
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    keep = [ln.strip() for ln in lines
            if any(k in ln for k in (".in ", ".out ", "所見:", "★", "セッション開始", "[版]"))]
    return "\n".join(keep)[-limit:]


def master_state(song_id: str) -> str:
    """**いまの版の完成品ができているか**を、形を添えてファイルの実在で1行にする（申し送りの材料）。

    ★2026-09-26: 書き出しは済んでいたのに、申し送りが「マスターに VOCALOID6 で書き出してもらう」と書き、
      ミクがそのままメールした（『明日への足音』v1）。指示の例文が手作業の時代のままだったうえ、
      **済んだかどうかをログの抜粋から推し量らせていた**。事実を渡せば推し量る必要が無い。
    ★2026-10-03: 完成の判定は名前の決まり（`song_files`・完成品 `master_<形>_v{N}.wav`）の1か所から引く。
      **インストには歌声の書き出しが無い**ので、V6 の話を材料に入れない（完成品は `render_final` が書く）。
      形を変えた版（例: インストに歌を足した）なら、前の版の形も添える。
    """
    import re

    from session import score_form
    d = SVC.parents[1] / "data" / "songs" / song_id
    vers = [int(m.group(1)) for p in d.glob("_CURRENT_v*.txt")
            if (m := re.fullmatch(r"_CURRENT_v(\d+)\.txt", p.name))] if d.exists() else []
    if not vers:
        return "完成品: 版が確定していない（曲が途中までか、失敗で止まった）"
    n = max(vers)
    got = song_files.master_of(d, n)
    form = got.form if got else score_form(d, n)
    label = song_files.FORM_LABEL[form]
    changed = ""
    if n > 1 and (prev := score_form(d, n - 1)) != form and (d / song_files.versioned(song_files.SCORE, n - 1)).exists():
        changed = f"・v{n - 1} の{song_files.FORM_LABEL[prev]}から{label}に形を変えた版"
    if got:
        how = ("演奏だけの曲なので歌声の書き出しは無い" if form == "instrumental"
               else "マスターに書き出しを頼まない")
        return f"完成品: **済み**（{label}{changed}・{got.path.name}。もう聴ける＝{how}）"
    if form == "instrumental":
        return (f"完成品: **まだ**（{label}{changed}・{song_files.master(form, n)} が無い。"
                "演奏の仕上げが失敗した。歌の入らない曲なので VOCALOID は関係ない）")
    return (f"完成品: **まだ**（{label}{changed}・{song_files.master(form, n)} が無い。"
            f"自動の歌声の書き出しが失敗した。手で頼むなら {song_files.versioned(song_files.VPR, n)} を開いて書き出してもらう）")


def compose_handoff(song_id: str, final_report: str, instruction: str = "") -> str:
    """ミク本人への申し送り（200字程度）を生成。失敗時は最終報告の先頭を返す。"""
    fallback = " ".join(str(final_report).split())[:300]
    try:
        from pydantic_ai import Agent

        from song_llm import pick_model
        agent = Agent(pick_model("HANDOFF", "flash"),
                      system_prompt=(SVC / "prompts" / "handoff.md").read_text(encoding="utf-8"))
        prompt = (f"依頼: {instruction[:300]}\n\n"
                  f"{master_state(song_id)}\n\n"
                  f"作業ログ（要点）:\n{_log_digest(song_id)}\n\n"
                  f"最終報告:\n{str(final_report)[:2000]}\n\n"
                  f"曲フォルダ: output/{song_id}/")
        from song_usage import run_counted                  # noqa: E402（費用に入れる・2026-10-08）
        text = " ".join(str(run_counted(agent, "申し送り", prompt).output).split())
        return text[:400] or fallback
    except Exception as e:
        print(f"[handoff] 生成に失敗（最終報告で代用）: {type(e).__name__}: {e}")
        return fallback
