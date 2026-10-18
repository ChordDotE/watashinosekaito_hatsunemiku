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

"""**点検役に渡す共通の文脈**（2026-10-04・`docs/plan/song_final_reviewer.md`「あわせて入れるもの」・組P1）。

## 前提・背景

点検役（旋律・詞・伴奏・後ろの工程＝仕上げ）は、それぞれ自分の工程の本文と機械の所見だけを読んでいた。届いていなかったのは:
自分の受け持ちの範囲／前の点検役の指摘（やり直させた理由・残した弱い所）／作曲の頼みの原文（仕上げとカバーの編曲にだけ届いていた）／
やり直しの2回目以降に、前回の自分の指示。前に何を言ったか知らずに判定し、前の点検役の受け持ちまで深追いしていた。

## 決めたこと

点検役の呼び出し口（`song_gate.judge`）が**1か所で**依頼文に足す（点検役ごとに手で足すと漏れる）。対象は `GATE_SCOPE` にある点検役だけで、
カバー（`Song.origin == "cover"`）は足さない（カバーは工程の点検役を通らない。編曲の点検役は材料に頼みを持つ）。
ただしカバーの点検役（コード・伴奏）の**やり直したあと**は、再判断の材料（下の `rejudge_text`・上からの指摘を除く）だけ足す（2026-10-04 組F4）。
- 頭: 受け持ち（`scope_text`）・頼み（`request_block`）・前の点検役の判断の履歴（`gate_history_text`）
- 選べる判断の手前: **再判断の材料**（`rejudge_text`・2026-10-04 組P2a）——同じ点検役の前回の判断（記録の指示 `guidance`・
  弱い所 `notes` から・`previous_own_text`）、作業者の指摘への対応（工程の出力の `response`）、前の稿から変わった所
  （`song_draft_diff`）、上からの指摘（ほかの点検役がこの受け持ちの工程へ差し戻した指摘・`upper_text`）

**最終の点検役**（2段目・組P2）は `gate_history` / `gate_history_text` を単独で呼べる（曲の記録 `Song.history` だけを読む）。
"""
from __future__ import annotations

import re
import sys

from song_gate import _ALL_STEPS, FORM_STEPS

#: **点検役 → 受け持ちの範囲（最初の工程, 最後の工程）**。並び＝点検役が呼ばれる順（`song_compose._compose_once_body`）。
#: 範囲は作る順（テーマ＋`STEP_ORDER`）で読み、その形で走る工程だけを取る（`scope_steps`）。
#: ★**範囲が重なる工程は、表で先の点検役の受け持ち**（後ろの工程の点検役の範囲 曲名〜奏法 には編成・伴奏が入るが、
#:   伴奏の点検役がいる形ではそちら）。点検役がいない形（アカペラの伴奏）では、その工程は次に範囲を持つ点検役へ移る。
#: ★**点検役を足したら、ここに1行足す**（ここに無い点検役には何も足さない）。`tests/song_service/test_gate_brief.py` が
#:   どの形でも「走る工程が1つずつ、ちょうど1つの点検役の受け持ちに入る」ことを確かめる。
#: ★**値が文の点検役は2段目**（工程の範囲を持たず、1段目の点検役を束ねる・2026-10-04 組P2b）。最終の点検役（`song_final_gate`）は
#:   曲全体を見る。工程の受け持ち（`scope_steps`）は空で、受け持ちの文（`scope_text`）にこの文を書く。再判断の材料は `final_rejudge_text`
GATE_SCOPE: dict[str, tuple[str, str] | str] = {
    "旋律": ("テーマ", "旋律"),
    "詞": ("詞の設計", "詞"),
    "伴奏": ("編成", "伴奏"),
    "仕上げ": ("曲名", "奏法"),
    "最終": "曲全体。頼みどおりか・1段目の点検役の指摘が直ったか",
}


def is_second_tier(gate: str) -> bool:
    """2段目の点検役か（`GATE_SCOPE` の値が文＝工程の範囲を持たない）。"""
    return isinstance(GATE_SCOPE.get(gate), str)

#: 判断の呼び名（記録の `request` → 文）
ACTION_WORD = {"accept": "通した", "redo": "やり直させた・差し戻した", "stop": "所見つきで進めた（止めた）"}
#: 前の点検役の判断の履歴の上限（字・全体）。超えたら切ったことを書く
#: ★2026-10-04 に 5000 → 12000（指示を「どこ・何が（数字）・どう直す」の1件1行で書く決まりにしたので、指示が長くなる。
#:   最終の点検役は1段目の指示が直ったかを照らすので、指示を削ると照らせない）
GATE_HISTORY_LIMIT = 12000
#: 1つの判断の理由・弱い所の上限（字）
REASON_SHOWN = 240
NOTES_SHOWN = 500
#: 1つの判断の指示（guidance）の上限（字）。理由より長く取る（修正役へ渡った中身そのもので、直ったかを照らす物差しになる）
GUIDANCE_SHOWN = 1500
#: 頼みの原文の上限（字・頼みの台帳の1行は `REQUEST_ROW`）。超えたら切ったことを書く
REQUEST_SHOWN = 600
REQUEST_ROW = 200
#: 前回の自分の判断のうち、理由と指示（`guidance`）の上限（字）。超えたら切ったことを書く（`_one_line`）。
#: ★2026-10-04（組F2・§2‑3）: 600字で切っていた。指示は場所と数字を1件1行で書く決まりにしたので長くなる
PREV_SHOWN = 2000


def gate_runs(gate: str, form: str) -> bool:
    """その形でこの点検役が呼ばれるか（工程名の点検役は、その工程が走る形だけ。仕上げはどの形でも）。"""
    return gate not in _ALL_STEPS or gate in FORM_STEPS[form]


def scope_steps(gate: str, form: str) -> tuple[str, ...]:
    """**その点検役の受け持ちの工程**（作る順・その形で走る工程だけ）。表に無い点検役・その形で呼ばれない点検役は空。"""
    if gate not in GATE_SCOPE or is_second_tier(gate) or not gate_runs(gate, form):
        return ()
    taken: set[str] = set()
    for g, rng in GATE_SCOPE.items():
        if isinstance(rng, str) or not gate_runs(g, form):
            continue
        a, z = rng
        span = [s for s in _ALL_STEPS[_ALL_STEPS.index(a):_ALL_STEPS.index(z) + 1]
                if s in FORM_STEPS[form] and s not in taken]
        if g == gate:
            return tuple(span)
        taken.update(span)
    return ()


def _span_text(steps: tuple[str, ...]) -> str:
    if not steps:
        return "（なし）"
    return steps[0] if len(steps) == 1 else f"{steps[0]}〜{steps[-1]}（{'・'.join(steps)}）"


def scope_text(gate: str, form: str) -> str:
    """【あなたの受け持ち】の節（表に無い点検役は空）。前の点検役・後の点検役の受け持ちも並べる。
    2段目の点検役（最終）には、受け持ちの文と1段目の点検役の受け持ちを書く。1段目には、最後に最終の点検役が曲全体を見ることも書く。"""
    if is_second_tier(gate):
        first = [g for g in GATE_SCOPE if scope_steps(g, form)]
        return (f"【あなたの受け持ち】{GATE_SCOPE[gate]}。\n"
                "  工程ごとの中身は、1段目の点検役が見ました: "
                + "／".join(f"{g}の点検役＝{_span_text(scope_steps(g, form))}" for g in first))
    mine = scope_steps(gate, form)
    if not mine:
        return ""
    order = [g for g in GATE_SCOPE if gate_runs(g, form)]
    before = [g for g in order[:order.index(gate)] if scope_steps(g, form)]
    after = [g for g in order[order.index(gate) + 1:] if scope_steps(g, form)]
    lines = [f"【あなたの受け持ち】あなたが見るのは {_span_text(mine)} の工程です。"]
    if before:
        lines.append("  それより前の工程は、前の点検役が見ました: "
                     + "／".join(f"{g}の点検役＝{_span_text(scope_steps(g, form))}" for g in before))
    else:
        lines.append("  あなたが最初の点検役です。")
    if after:
        lines.append("  後の工程は、後の点検役が見ます: "
                     + "／".join(f"{g}の点検役＝{_span_text(scope_steps(g, form))}" for g in after))
    if (top := [g for g in GATE_SCOPE if is_second_tier(g)]):
        lines.append(f"  最後に、{'・'.join(top)}の点検役が曲全体（頼みどおりか・各点検役の指摘が直ったか）を見ます。")
    return "\n".join(lines)


# ---- 頼みの原文 ----------------------------------------------------------------------------------

def request_block(song, mark=()) -> str:
    """**マスターの頼み**: 作曲を頼まれたときの文（`Song.instruction`）・頼まれごとの台帳の生きている行・
    形を変えるときの頼み（`song_context.FORM_REQUEST`）。どれも無ければ空。

    2026-10-03（組G4）に仕上げの点検役の材料の頭に置いたものを、2026-10-04 にここへ移し、全部の点検役へ渡す。
    `mark` … 台帳の行の id のうち「この編集で積んだ行」（編集のあとの最終の点検・2026-10-04・案C）。頭に【この編集】を付ける
    （台帳は全期間の行を混ぜて出すので、印が無いと、どれがいま頼まれたことか分からない）。
    """
    from song_requests import active, standing
    out = []
    ins = " ".join((getattr(song, "instruction", "") or "").split())
    if ins:
        out.append("【マスターの頼み（作曲を頼まれたときの文）】" + ins[:REQUEST_SHOWN]
                   + (f"…（以下略・全{len(ins)}字）" if len(ins) > REQUEST_SHOWN else ""))
    # ★**続いている頼みを先に・別の見出しで**（2026-10-14・`tools/editing.keep_request`）。一度通った頼みが後の手直しで黙って崩れた
    keep = standing(list(getattr(song, "requests", None) or []))
    if keep:
        out.append("【続いている頼み（マスターがこの先も守ってほしいと言ったこと。いまの曲がこれに反していれば NG）】" + "".join(
            f"\n   ・{r.target}{'（' + r.scope + '）' if r.scope else ''}: {_one_line(r.words, REQUEST_ROW)}" for r in keep))
    rows = [r for r in active(list(getattr(song, "requests", None) or []))
            if (r.words or "").strip() and r not in keep]
    if rows:
        out.append("【頼まれごと（作り直しで頼まれたこと）】" + "".join(
            f"\n   ・{'【この編集】' if r.id in mark else ''}{r.target}{'（' + r.scope + '）' if r.scope else ''}: "
            f"{_one_line(r.words, REQUEST_ROW)}" for r in rows))
    sc = sys.modules.get("song_context")
    form_request = getattr(sc, "FORM_REQUEST", "") if sc is not None else ""
    if form_request:
        out.append(f"【形を変えるときのマスターの頼み】{form_request}")
    return "\n".join(out)


# ---- 前の点検役の判断の履歴 ------------------------------------------------------------------------

def gate_record_name(gate: str) -> str:
    """点検役 → 曲の記録の名前（`melody_gate`・`lyrics_gate`・`arrange_gate`・`finish_gate`）。"""
    from song_flow import usage_record_name                 # noqa: E402（循環を避けてここで）
    return usage_record_name(f"関門({gate})")


def _one_line(text: str, limit: int) -> str:
    t = " ".join((text or "").split())
    t = t.removeprefix("【保存から戻した判断】")
    # ★切ったら全体の字数まで書く（2026-10-04・組F2・§2‑3。以前は「…」だけで、どれだけ切れたか分からなかった）
    from song_fix import cut_note                           # noqa: E402
    return cut_note(t, limit)


def gate_history(song, gates=None) -> list[dict]:
    """**点検役ごとの判断の履歴**（曲の記録 `Song.history` の `*_gate` から・単独で呼べる）。

    `gates` … 見る点検役（作る順で並べ直す）。省略時は `GATE_SCOPE` の全部。記録の無い点検役は入れない。
    戻り値は点検役ごとの dict:
      `gate`（点検役の名前）・`steps`（受け持ちの工程・曲の形で）・
      `decisions`（判断の流れ＝ `[{"action", "reason", "guidance", "notes"}]`、古い順。action は accept / redo / stop。
      guidance はやり直す側へ渡した指示＝記録の `guidance`・2026-10-04 組P2a。欄の無い古い記録は空）・
      `last_action`（最後の判断）・`guidance`（最後の判断の指示）・`notes`（最後の判断で残した弱い所。空もある）
    """
    form = getattr(song, "form", "song") or "song"
    names = [g for g in GATE_SCOPE if gates is None or g in gates]
    names += [g for g in (gates or []) if g not in GATE_SCOPE]
    out = []
    for g in names:
        rec = gate_record_name(g)
        decisions = [{"action": r.request, "reason": r.reason or "", "guidance": getattr(r, "guidance", "") or "",
                      "notes": r.notes or ""}
                     for r in (getattr(song, "history", None) or []) if rec and r.step == rec]
        if not decisions:
            continue
        out.append({"gate": g, "steps": scope_steps(g, form), "decisions": decisions,
                    "last_action": decisions[-1]["action"], "guidance": decisions[-1]["guidance"],
                    "notes": decisions[-1]["notes"]})
    return out


def redo_rounds(song) -> list[str]:
    """**この曲の差し戻しの経過**（前の周で、どの点検役がどこから作り直させたか・記録の `redo`）。理由の文の並び。"""
    return [r.reason for r in (getattr(song, "history", None) or []) if r.step == "redo" and r.reason]


def gate_history_text(song, gates=None, *, limit: int = GATE_HISTORY_LIMIT, rounds: bool = True,
                      heading: str = "【前の点検役の判断】") -> str:
    """`gate_history` を依頼文の1節にする（無ければ空）。`rounds` なら差し戻しの経過（`redo_rounds`）も付ける。
    1つの理由は `REASON_SHOWN`・弱い所は `NOTES_SHOWN` 字まで、全体は `limit` 字まで（切ったら書く）。"""
    lines: list[str] = []
    for h in gate_history(song, gates):
        span = GATE_SCOPE[h["gate"]] if is_second_tier(h["gate"]) else _span_text(h["steps"])
        lines.append(f"■{h['gate']}の点検役（受け持ち: {span}）——最後は「{ACTION_WORD.get(h['last_action'], h['last_action'])}」")
        for i, d in enumerate(h["decisions"], 1):
            lines.append(f"  {i}. {ACTION_WORD.get(d['action'], d['action'] or '?')}: {_one_line(d['reason'], REASON_SHOWN)}"
                         + (f"／指示: {_one_line(d['guidance'], GUIDANCE_SHOWN)}" if d["guidance"] else ""))
        lines.append("  最後に残した弱い所: " + (_one_line(h["notes"], NOTES_SHOWN) or "（書いていない）"))
    back = redo_rounds(song) if rounds else []
    if back:
        lines.append("■この曲の差し戻し（前の周）")
        lines += [f"  ・{_one_line(r, REASON_SHOWN)}" for r in back]
    if not lines:
        return ""
    body = "\n".join(lines)
    if len(body) > limit:
        body = body[:limit] + f"…（以下略・全{len(body)}字を上限 {limit}字で切った）"
    return heading + "\n" + body


# ---- 前回の自分の判断（記録から・2026-10-04 組P2a）------------------------------------------------------
#
# ★2026-10-04（組P1）は点検役の判断を作業中に覚えていた（`remember`）。記録（`StepRecord`）に指示（`guidance`）の欄が無く、
#   続きから作る・作り直すと前回の自分の判断が消えた。組P2a で記録に `guidance` を足し、**記録から読む**形にした:
#   - やり直しの2回目以降（`attempt` > 0）… いまの曲の記録の、同じ点検役の最後の判断（やり直させた判断）
#   - 差し戻しの後の周・作り直し・続きから作る（`attempt` == 0）… 戻している記録（`song_replay` の保存＝前の周・前の版）の、
#     同じ点検役の最後の判断。新しく作る曲の1周目には無い

def _as_rec(r) -> dict:
    return dict(r) if isinstance(r, dict) else dict(vars(r))


def _saved_history() -> list[dict]:
    """戻している記録（差し戻しの前の周・作り直す前の版・続きから作る前）の `history`。戻していなければ空。"""
    rp = sys.modules.get("song_replay")
    st = getattr(rp, "_state", None) if rp is not None else None
    return list(getattr(st, "history", None) or [])


def previous_own(gate: str, song=None, attempt: int = 0) -> dict | None:
    """**同じ点検役の前回の判断**（記録の dict に `when` を足したもの・無ければ None）。`when` は retry（やり直しの前）／
    round（前の周・前の版）。単独で呼べる（最終の点検役の2周目も同じ形で読む）。"""
    song = current_song() if song is None else song
    rec = gate_record_name(gate)
    if not rec:
        return None
    if attempt:
        cur = [_as_rec(r) for r in (getattr(song, "history", None) or []) if r.step == rec]
        return {**cur[-1], "when": "retry"} if cur else None
    saved = [h for h in _saved_history() if isinstance(h, dict) and h.get("step") == rec]
    return {**saved[-1], "when": "round"} if saved else None


def _redo_target(song, guidance: str) -> str:
    """差し戻しの記録（`redo`）のうち、この指示で差し戻した先の工程（無ければ空）。"""
    if not guidance:
        return ""
    for r in reversed(getattr(song, "history", None) or []):
        if r.step == "redo" and (getattr(r, "guidance", "") or "") == guidance:
            return r.request or ""
    return ""


def previous_own_text(gate: str, attempt: int = 0, song=None) -> str:
    """【前回のあなたの判断】の節（判断・理由・指示・弱い所・そのときの機械の所見の数）。前回が無ければ空。"""
    song = current_song() if song is None else song
    p = previous_own(gate, song, attempt)
    if p is None:
        return ""
    when = (f"やり直しの前（{attempt} 回目のやり直しの前）" if p["when"] == "retry"
            else "前の周（差し戻し・作り直し・続きから作る前の記録）")
    action = p.get("request") or ""
    back = _redo_target(song, p.get("guidance") or "")
    lines = [f"【前回のあなたの判断】{when}に、あなたはこう判断しました。",
             f"  判断: {ACTION_WORD.get(action, action or '?')}" + (f"（{back}から作り直させた）" if back else ""),
             f"  理由: {_one_line(p.get('reason') or '', PREV_SHOWN)}"]
    if p.get("guidance"):
        lines.append(f"  指示: {_one_line(p['guidance'], PREV_SHOWN)}")
    if p.get("notes"):
        lines.append(f"  弱い所: {_one_line(p['notes'], NOTES_SHOWN)}")
    lines.append(f"  そのときの機械の所見: {len(p.get('warnings') or [])}件")
    return "\n".join(lines)


# ---- 作業者の対応・変わった所・上からの指摘（再判断の材料・2026-10-04 組P2a）-------------------------------

#: 作業者の対応（`response`）の上限（字・1工程）
RESPONSE_SHOWN = 1200
#: 上からの指摘の上限（字・1件）
UPPER_SHOWN = 600

#: **指摘を受けて直す呼び出しの依頼文に足す1文**（自工程のやり直し・差し戻し・作り直しの頼み）。工程の出力の共通の欄 `response`
#: （`song_model.StepStatus`）に指摘ごとの対応を書かせる。指摘の無い呼び出しには足さない（欄の説明が「空のまま」と言う）
#: ★**迷ったときの扱い**も同じ1か所に（2026-10-04・テーラリング。`prompts.md`「作業・直しを頼むときに渡すもの」）。点検役の指示に
#:   【必須】／【できれば】・変えない所が付くようになったので、それをどう扱うかと、ぶつかったとき・直せないときの扱いを共通の文で渡す
RESPOND_ASK = ("★**指摘の扱い方**: 【必須】の指摘から直し、【できれば】のために【必須】や「変えない所」を崩さない。"
               "指摘どうし・指摘とこの工程の書き方の決まり（記法・個数・長さ）がぶつかったら、**決まりを守り**、どちらを取ったかを"
               " `response` に書く。直せないと分かった指摘は、無理に直さず「直さなかった（なぜ）」と書く（無理に直して別の所を壊さない）。\n"
               "★**`response` に、上の指摘ごとの対応を書いてください**——指摘1つにつき1行で「直した（どこをどう）」か"
               "「直さなかった（なぜ）」。点検役は、この対応と前の稿からの実際の変化を見比べて判断します。")


def redo_ask(points: str) -> str:
    """自工程のやり直しの依頼文に足す【前回の指摘】と `response` の頼み。"""
    return f"【前回の指摘】\n{points}\n{RESPOND_ASK}"


def carry_note(step: str, who: str, guidance: str) -> str:
    """**差し戻しの申し送り**（`song_context.CARRY` の1行）。差し戻された工程と、差し戻した点検役を書く。"""
    return f"【{step}のやり直し" + (f"（{who}の点検役の指摘）" if who else "") + f"】{guidance}"


_CARRY_RE = re.compile(r"^【(?P<step>.+?)のやり直し(?:（(?P<who>.+?)の点検役の指摘）)?】(?P<body>.*)$", re.S)


def parse_carry(note: str) -> tuple[str, str, str] | None:
    """`carry_note` の行 → (差し戻された工程, 差し戻した点検役, 指摘)。差し戻しの行でなければ None。"""
    m = _CARRY_RE.match(note or "")
    return (m["step"], m["who"] or "", m["body"]) if m else None


def upper_notes(gate: str, form: str) -> list[tuple[str, str, str]]:
    """**上からの指摘**: この周の差し戻しのうち、**この点検役の受け持ちの工程へ、ほかの点検役が**戻したもの
    （(工程, 差し戻した点検役, 指摘) の並び）。自分が出した差し戻しは「前回のあなたの判断」で見るので入れない。
    ★**受け持ちの工程が巻き込まれた差し戻しも入れる**（2026-10-04 組F1・§2‑2）——差し戻された工程が受け持ちの工程の祖先なら
      （`song_gate.is_ancestor`）、受け持ちは作り直しに巻き込まれた。以前は差し戻された工程そのものが受け持ちにあるときだけで、
      最終の点検役が詞へ戻したとき、作り直した曲名〜奏法を見る仕上げの点検役には届かなかった。"""
    from song_gate import is_ancestor                      # noqa: E402
    sc = sys.modules.get("song_context")
    notes = list(getattr(sc, "CARRY", None) or []) if sc is not None else []
    mine = scope_steps(gate, form)
    out = []
    for note in notes:
        got = parse_carry(note)
        if got and got[1] != gate and (got[0] in mine or any(is_ancestor(got[0], s) for s in mine)):
            out.append(got)
    return out


def upper_text(gate: str, form: str) -> str:
    """【上からの指摘】の節（無ければ空）。受け持ちが巻き込まれただけの差し戻しは、そう書く。"""
    rows = upper_notes(gate, form)
    if not rows:
        return ""
    mine = set(scope_steps(gate, form))
    return ("【上からの指摘】あなたの受け持ちの工程へ（または、その上流へ）ほかの点検役が差し戻しました"
            "（この周は、その指摘を受けて作り直した稿です）。\n"
            + "\n".join(f"  ・{who or '（不明）'}の点検役 → {step}"
                        + ("" if step in mine else "（あなたの受け持ちは、この作り直しに巻き込まれて作り直した）")
                        + f": {_one_line(body, UPPER_SHOWN)}" for step, who, body in rows))


def _output(song, step: str, own: str, after):
    from song_context import step_output_key            # noqa: E402（循環を避けてここで）
    if step == own and after is not None:
        return after
    key = step_output_key(step)
    return getattr(song, key, None) if key else None


def _prev_outputs() -> dict:
    sc = sys.modules.get("song_context")
    return dict(getattr(sc, "PREV_OUTPUTS", None) or {}) if sc is not None else {}


def drafts(gate: str, song, attempt: int = 0, *, before=None, after=None) -> list[tuple[str, object, object]]:
    """**比べる稿**（(工程, 前の稿, 新しい稿) の並び・作る順）。
    やり直しの2回目以降は、この点検役の工程だけ（呼び手が渡す前の回の出力 `before` と新しい出力 `after`）。
    1回目（差し戻しの後の周・作り直し）は、受け持ちの工程のうち前回の出力（`song_context.PREV_OUTPUTS`）があるものを、いまの出力と。"""
    from song_draft_diff import as_dict                    # noqa: E402
    form = getattr(song, "form", "song") or "song"
    # カバーの点検役（コード・伴奏）は受け持ちの表の外。自分の工程の稿（やり直しの前後）だけを比べる（組F4）
    own = gate if gate in scope_steps(gate, form) or (is_cover(song) and gate in _ALL_STEPS) else ""
    if attempt:
        return [(own, before, after)] if own and as_dict(before) is not None and as_dict(after) is not None else []
    prev = _prev_outputs()
    out = []
    for s in scope_steps(gate, form):
        b = before if (s == own and before is not None) else prev.get(s)
        a = _output(song, s, own, after)
        if as_dict(b) is not None and as_dict(a) is not None:
            out.append((s, b, a))
    return out


def responses(gate: str, song, attempt: int = 0, *, before=None, after=None) -> list[tuple[str, str]]:
    """**作業者の対応**（(工程, `response`) の並び）。比べる稿（`drafts`）の新しい稿のうち、対応を書いたものだけ。"""
    from song_draft_diff import as_dict                    # noqa: E402
    out = []
    for s, _b, a in drafts(gate, song, attempt, before=before, after=after):
        r = " ".join(((as_dict(a) or {}).get("response") or "").split())
        if r:
            out.append((s, r))
    return out


def changes_text(gate: str, song, attempt: int = 0, *, before=None, after=None) -> str:
    """【前の稿から変わった所】の節（機械が並べた事実・無ければ空）。"""
    from song_context import step_output_key            # noqa: E402
    from song_draft_diff import diff_text                  # noqa: E402
    blocks = [diff_text(s, step_output_key(s), b, a) for s, b, a in drafts(gate, song, attempt, before=before, after=after)]
    blocks = [b for b in blocks if b]
    if not blocks:
        return ""
    return "【前の稿から変わった所（機械が前の稿と比べて並べた事実。良し悪しは書いていない）】\n" + "\n".join(blocks)


def rejudge_text(gate: str, attempt: int = 0, song=None, *, before=None, after=None) -> str:
    """**再判断の材料**: 前回の自分の判断・作業者の対応・変わった所・上からの指摘（どれも無ければ空）。"""
    song = current_song() if song is None else song
    form = getattr(song, "form", "song") or "song"
    parts = [previous_own_text(gate, attempt, song)]
    resp = responses(gate, song, attempt, before=before, after=after)
    if resp:
        parts.append("【作業者の対応（指摘ごとに、直した／直さなかった理由）】\n"
                     + "\n".join(f"  ■{s}: {_one_line(r, RESPONSE_SHOWN)}" for s, r in resp))
    elif drafts(gate, song, attempt, before=before, after=after) and (parts[0] or upper_notes(gate, form)):
        parts.append("【作業者の対応】（作業者は書いていない）")
    parts += [changes_text(gate, song, attempt, before=before, after=after),
              "" if is_cover(song) else upper_text(gate, form)]   # カバーにはほかの点検役が無い（組F4）
    return "\n\n".join(p for p in parts if p)


# ---- 依頼文へ足す（呼び出し口から1か所）-------------------------------------------------------------

def current_song():
    sc = sys.modules.get("song_context")
    return getattr(sc, "CURRENT_SONG", None) if sc is not None else None


def is_cover(song) -> bool:
    """カバーの曲か（`Song.origin == "cover"`）。"""
    return song is not None and getattr(song, "origin", "compose") == "cover"


def applies(gate: str, song) -> bool:
    """この点検役の依頼文に共通の文脈を足すか（表にある点検役・新しく作る曲と作り直し。カバーは足さない）。
    ★カバーでも、やり直したあとの再判断の材料だけは足す（`gate_brief` のカバーの枝・組F4）。"""
    return song is not None and gate in GATE_SCOPE and not is_cover(song)


def gate_brief(gate: str, attempt: int = 0, song=None, *, before=None, after=None) -> tuple[str, str]:
    """点検役 `gate` の依頼文に足す (頭, 選べる判断の手前)。足さないときは ("", "")。`song` を省くといま作っている曲。
    `before` / `after` … この点検役の工程の前の回の稿と新しい稿（呼び手が渡す。やり直しの2回目以降の比べる相手・新しい稿の対応）。
    選べる判断の手前には、再判断の材料（`rejudge_text`＝前回の自分の判断・作業者の対応・変わった所・上からの指摘）を置く。"""
    song = current_song() if song is None else song
    # ★**編集のあとの最終の点検**（2026-10-04・案C）は、呼んでいる間だけ置かれる材料（`EDIT`）から組む（新しい曲の流れとは別の節）
    if EDIT is not None and gate == EDIT.get("gate"):
        return edit_brief(song)
    # ★**カバーの点検役（コード・伴奏）のやり直したあと**は、再判断の材料（前回の自分の判断・作業者の対応・変わった所）だけを足す
    #   （2026-10-04 組F4・R2 #7）。以前はカバーを丸ごと外していて、2回目の点検役は自分が何を直せと言ったかを知らずに判定した。
    #   受け持ち・前の点検役の判断・上からの指摘は足さない（カバーは工程の点検役を通らない。頼みは材料にある）。
    #   1回目は前回が無い（記録を戻す仕組みにも乗らない）ので何も足さない
    if is_cover(song):
        return "", (rejudge_text(gate, attempt, song, before=before, after=after) if attempt else "")
    if not applies(gate, song):
        return "", ""
    form = getattr(song, "form", "song") or "song"
    order = [g for g in GATE_SCOPE if gate_runs(g, form)]
    earlier = order[:order.index(gate)] if gate in order else []
    if is_second_tier(gate):                               # 2段目は1段目の点検役だけを並べる（2段目どうしは並べない）
        earlier = [g for g in earlier if not is_second_tier(g)]
    head = [t for t in (scope_text(gate, form), request_block(song),
                        gate_history_text(song, earlier, heading=("【1段目の点検役の判断】" if is_second_tier(gate)
                                                                  else "【前の点検役の判断】"))) if t]
    rejudge = (final_rejudge_text(gate, song) if is_second_tier(gate)
               else rejudge_text(gate, attempt, song, before=before, after=after))
    return "\n\n".join(head), rejudge


# ---- 2段目（最終の点検役）の再判断の材料（2026-10-04・組P2b）---------------------------------------------
#
# 人の監督者が、自分が差し戻した作業の出し直しを見るときに要るもの（`docs/plan/song_final_reviewer.md`「追記」）:
#   ①自分の前回の NG（どの工程へ・何を・どう直せ）②差し戻した工程群それぞれの「指摘への対応」と、前の稿から変わった所
#   ③その間に1段目の点検役がどう判断し直したか。
# 前回の NG は曲の記録から読む——差し戻しの周回（`song_compose._compose_song_rounds`）が、最終の点検役の判断の写しを次の周の記録の頭に
# 入れる（周をまたいで消えない）。続きから作る・作り直すとき（保存から戻した記録）は、戻した記録の最後の判断（`previous_own`）。
# 比べる前の稿は、最終の点検役が差し戻したときの稿（`song_context.FINAL_PREV`）。無ければ作り直す工程の前回の出力（`PREV_OUTPUTS`）。

#: 2段目の再判断の材料の上限（字・全体）。超えたら切ったことを書く
FINAL_REJUDGE_LIMIT = 12000


def _final_prev_outputs() -> dict:
    sc = sys.modules.get("song_context")
    got = dict(getattr(sc, "FINAL_PREV", None) or {}) if sc is not None else {}
    return got or _prev_outputs()


def final_own_decisions(gate: str, song) -> list[dict]:
    """**最終の点検役のこれまでの判断**（古い順・記録の dict に `when` を足したもの）。この曲の記録にあれば（差し戻しの周回の写し）
    それを、無ければ戻している記録の最後の判断（続きから作る・作り直す前の版）を1つ。どちらも無ければ空。"""
    rec = gate_record_name(gate)
    cur = [{**_as_rec(r), "when": "round"} for r in (getattr(song, "history", None) or []) if rec and r.step == rec]
    if cur:
        return cur
    p = previous_own(gate, song, 0)
    return [{**p, "when": "saved"}] if p else []


def final_rejudge_text(gate: str, song=None) -> str:
    """**2段目の再判断の材料**: 前回の自分の NG・差し戻した工程群の対応と変わった所・その間の1段目の点検役の判断（無ければ空）。"""
    from song_context import step_output_key            # noqa: E402（循環を避けてここで）
    from song_draft_diff import as_dict, diff_lines, diff_text   # noqa: E402
    song = current_song() if song is None else song
    form = getattr(song, "form", "song") or "song"
    own = final_own_decisions(gate, song)
    if not own:
        return ""
    parts: list[str] = []
    rows = []
    for i, d in enumerate(own, 1):
        action = d.get("request") or ""
        back = _redo_target(song, d.get("guidance") or "")
        rows.append(f"  {i}. {ACTION_WORD.get(action, action or '?')}" + (f"（{back}から作り直させた）" if back else "")
                    + f": {_one_line(d.get('reason') or '', PREV_SHOWN)}"
                    + (f"\n     指示: {_one_line(d['guidance'], PREV_SHOWN)}" if d.get("guidance") else "")
                    + (f"\n     弱い所: {_one_line(d['notes'], NOTES_SHOWN)}" if d.get("notes") else ""))
    when = ("この曲の前の周（あなたが差し戻す前）" if own[-1]["when"] == "round"
            else "前の版・前の周（続きから作る・作り直す前の記録）")
    parts.append(f"【前回までのあなたの判断】{when}に、あなたはこう判断しました（古い順）。\n" + "\n".join(rows))
    prev = {s: o for s, o in _final_prev_outputs().items() if s in FORM_STEPS[form]}
    # ★**差し戻しの範囲と、それ以外で変わった工程を並べる**（2026-10-04 組F1・1‑4）。`FINAL_PREV` は差し戻したときの全工程の写しで、
    #   差し戻しの範囲は `FINAL_REDONE`。範囲の外でも、その間の1段目の差し戻しで変わった工程（例: サビの詞・編成・伴奏）は並べる。
    #   以前は範囲の工程しか比べず、変わったことも、その工程を受け持つ点検役が判断し直したことも見えなかった
    sc = sys.modules.get("song_context")
    final_range = set(getattr(sc, "FINAL_REDONE", None) or ()) if sc is not None else set()
    remade_now = set(_prev_outputs())                      # この周に作り直した工程（1段目の差し戻しの宛先と、それを読む工程）
    resp, diffs = [], []
    for s in [x for x in _ALL_STEPS if x in prev]:
        key = step_output_key(s)
        now = getattr(song, key, None) if key else None
        if as_dict(prev[s]) is None or as_dict(now) is None:
            continue
        if final_range and s not in final_range and s not in remade_now and not diff_lines(key, prev[s], now)[0]:
            del prev[s]                                    # 範囲の外で、この周に作り直してもおらず変わってもいない工程は並べない
            continue
        r = " ".join(((as_dict(now) or {}).get("response") or "").split())
        resp.append(f"  ■{s}: " + (_one_line(r, RESPONSE_SHOWN) if r else "（書いていない）"))
        if (t := diff_text(s, key, prev[s], now)):
            diffs.append(t)
    if resp:
        parts.append("【差し戻した工程の対応（指摘ごとに、直した／直さなかった理由。作り直しに巻き込まれた工程も並べる）】\n"
                     + "\n".join(resp))
    if diffs:
        parts.append("【前の稿から変わった所（機械が前の稿と比べて並べた事実。良し悪しは書いていない）】\n" + "\n".join(diffs))
    redone = set(prev)
    again = []
    for h in gate_history(song, [g for g in GATE_SCOPE if not is_second_tier(g)]):
        if not redone & set(h["steps"]):
            continue
        live = [d for d in h["decisions"] if not (d["reason"] or "").startswith("【保存から戻した判断】")]
        if not live:
            again.append(f"  ■{h['gate']}の点検役: 判断し直していない（前の周の判断を保存から戻した）")
            continue
        again.append(f"  ■{h['gate']}の点検役（受け持ち: {_span_text(h['steps'])}）: "
                     + " → ".join(f"{ACTION_WORD.get(d['action'], d['action'] or '?')}（{_one_line(d['reason'], REASON_SHOWN)}）"
                                  for d in live)
                     + (f"\n     最後に残した弱い所: {_one_line(h['notes'], NOTES_SHOWN)}" if h["notes"] else ""))
    if again:
        parts.append("【差し戻しのあと、1段目の点検役の判断（作り直した工程を受け持つ点検役）】\n" + "\n".join(again))
    body = "\n\n".join(parts)
    if len(body) > FINAL_REJUDGE_LIMIT:
        body = body[:FINAL_REJUDGE_LIMIT] + f"…（以下略・全{len(body)}字を上限 {FINAL_REJUDGE_LIMIT}字で切った）"
    return body


# ---- 編集のあとの最終の点検（2026-10-04・案C・`song_final_gate.judge_edit`）---------------------------------------
#
# 追加編集（移調・ハモリ・歌の置き換え・伴奏の書き換え）のあと、書き出しの入口でまとめて1回、最終の点検役が見る。
# 新しい曲の材料（頼み・1段目の判断・曲全体の表）に、次を足す（設計は `docs/plan/song_final_reviewer.md`「追記: 編集のあとの点検」）:
#   【この編集で頼まれたこと】マスターの言葉と、この編集で台帳に積んだ行 ／【この編集でしたこと】点検待ちの束の各件 ／
#   【前の版から変わった所】楽譜の区間ごとの違い（`song_draft_diff.score_changes`）／【前に通したときのあなたの判断】最後の `final_gate`
# 1段目の点検役の判断は「編集の前の曲への判断」と見出しで分ける（編集した音は1段目の点検役を通っていない）。
# ★受け持ちの表（`GATE_SCOPE`）には足さない——あの表は新しい曲の流れの点検役の並びで、1段目の受け持ちの文（「最後に◯◯の点検役が
#   曲全体を見ます」）にも出る。編集のあとの点検はその流れの外にいる。記録の名前は `song_gate.GATE_RECORD`（`final_gate_edit`）

#: 呼んでいる間だけ置く材料（`song_final_gate.judge_edit` が入れて、終わったら None に戻す）。
#: キー: gate（点検役の名前）・instruction（この実行のマスターの言葉）・ids（この編集で台帳に積んだ行の id）・
#: did（この編集でしたことの行の並び）・changes（前の版から変わった所の文）・earlier（この頼みの中で前に出した判断の文）
EDIT: dict | None = None
#: 【この編集でしたこと】の上限（字）
EDIT_SHOWN = 4000

EDIT_SCOPE = ("【あなたの受け持ち】この曲は、前に最終の点検役（あなた）を通ったあと、マスターの頼みで追加の編集をしました。"
              "見るのは**この編集の結果**です——頼んだ変化になっているか・編集した所が周りと噛み合うか。"
              "編集していない所の弱さは、前に通したときの判断（下の【前に通したときのあなたの判断】）に従い、蒸し返さないでください。")


def last_final(song) -> dict | None:
    """この曲の最後の最終の点検役の判断（記録 `final_gate`。無ければ None）。"""
    rec = gate_record_name("最終")
    got = [_as_rec(r) for r in (getattr(song, "history", None) or []) if rec and r.step == rec]
    return got[-1] if got else None


def _cut(body: str, limit: int) -> str:
    return body if len(body) <= limit else body[:limit] + f"…（以下略・全{len(body)}字を上限 {limit}字で切った）"


def edit_brief(song) -> tuple[str, str]:
    """編集のあとの最終の点検の (頭, 選べる判断の手前)。材料は `EDIT`。"""
    e = EDIT or {}
    form = getattr(song, "form", "song") or "song"
    ids = set(e.get("ids") or ())
    first = [g for g in GATE_SCOPE if not is_second_tier(g) and gate_runs(g, form)]
    head = [EDIT_SCOPE, request_block(song, mark=tuple(ids)),
            gate_history_text(song, first, heading="【1段目の点検役の判断（編集の前の曲への判断。"
                                                   "この編集で書いた音は、1段目の点検役を通っていません）】")]
    ins = " ".join((e.get("instruction") or "").split())
    asked = [f"  マスターの言葉: {_one_line(ins, REQUEST_SHOWN) or '（記録に無い）'}"]
    rows = [r for r in (getattr(song, "requests", None) or []) if r.id in ids]
    if rows:
        asked.append("  台帳に積んだ行: " + "／".join(
            f"{r.target}{'（' + r.scope + '）' if r.scope else ''}: {_one_line(r.words, REQUEST_ROW)}" for r in rows))
    parts = ["【この編集で頼まれたこと】\n" + "\n".join(asked)]
    body = "\n".join(f"  ・{d}" for d in (e.get("did") or [])) or "  （束が空）"
    parts.append("【この編集でしたこと】（道具・大きさ・中身。大＝誰かが音を新しく書いた／曲全体の土台を動かした、"
                 "中＝引き算・高さの移しだけ）\n" + _cut(body, EDIT_SHOWN))
    parts.append("【前の版から変わった所（機械が前の版の楽譜と比べて並べた事実。良し悪しは書いていない）】\n"
                 + (e.get("changes") or "（比べられなかった）"))
    p = last_final(song)
    if p is None:
        parts.append("【前に通したときのあなたの判断】（記録に無い——この曲は最終の点検役ができる前に作った）")
    else:
        action = p.get("request") or ""
        lines = ["【前に通したときのあなたの判断】（この曲を作った・作り直したときの最後の判断）",
                 f"  判断: {ACTION_WORD.get(action, action or '?')}",
                 f"  理由: {_one_line(p.get('reason') or '', PREV_SHOWN)}"]
        if p.get("notes"):
            lines.append(f"  弱い所: {_one_line(p['notes'], NOTES_SHOWN)}")
        parts.append("\n".join(lines))
    if e.get("earlier"):
        parts.append("【この頼みの中で、前に出したあなたの判断】（コーディネーターはこのあと直した。直ったかを見てください）\n"
                     + "\n".join(f"  {i}. {_one_line(x, PREV_SHOWN)}" for i, x in enumerate(e["earlier"], 1)))
    return "\n\n".join(t for t in head if t), "\n\n".join(parts)
