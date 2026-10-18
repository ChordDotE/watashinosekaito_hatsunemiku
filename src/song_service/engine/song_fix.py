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

"""**機械は LLM の決めた中身を黙って変えない**——検査して所見を返し、その工程に差分で直させる（2026-09-27）。

## なぜ

2026-09-27 の棚卸しで、機械が LLM の決めた中身（高さ・長さ・区間名・詞…）を書き換える箇所が **76件**あった
（`docs/plan/song_machine_corrections.md`）。大半は事故のたびに症状の出た場所へ足した局所パッチで、
LLM が「機械が直す」ことまで知っていたのは約10件、半数以上は記録にも残らなかった。本人「極論全部なくしたい」。
規約は `.claude/skills/miku-dev/details/coding/song_service_editing.md`、計画は `docs/plan/song_no_silent_corrections.md`。

## 流れ（どの工程も同じ）

    工程の LLM が書く → 機械が検査（所見を作る。直さない）
      → 所見があれば、同じ工程に所見を返し **差分だけ** 書き直させる（`ask`）→ 機械は差分をそのまま当てる（`apply`）
      → もう一度検査（`MAX_FIX_ROUNDS` 回まで）
      → まだ所見があれば、機械は書き換えず、**その工程で曲づくりを止める**（`StepUnfixed`・2026-09-30）

工程ごとに違うのは **検査・差分の問い方・差分の当て方** の3つだけ。流れはここ1か所。

## 直らなかったら止める（2026-09-30 本人方針）

本人「形式の所見が直らなかったら止めるのはいいと思う。その場合は、人間が追加で依頼したら、また LLM がやり直す感じかな」。
ここで見るのは**形式**（答えが1つに決まる検査）だけなので、直らないまま先へ進んでも、組み立ては鳴らせない所を鳴らさずに
進むしかなかった（書いたものが黙って欠ける）。止めて、どの工程で何が直らなかったかをコーディネーターからマスターへ伝え、
マスターが頼めば `remake_from_step` でその工程から続きを作る（前の工程は途中保存から費用なしで戻る）。
**内容**（音楽として・言葉として）は点検役が判断するので止めない（`song_gate`）。
組み立て（記号→音）は、検査が保証している形しか来ない前提で書く。それでも来たら不具合なので `AssemblyError` で知らせる。

## どうしても機械が直すとき

`record_change` を必ず通す。「どの工程の何を、なぜ、どう直したか」を曲の記録（`StepRecord`・`<出力名>_machine`）に残し、
`song_context.upstream_intents` がその工程を読む後の工程へ「機械がした変更」として見せる。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

#: 差分で書き直させる回数の上限。1回ごとに LLM を1回呼ぶ（出力は差分だけなので小さい）
MAX_FIX_ROUNDS = 2


class StepUnfixed(Exception):
    """**形式の所見が、差分で直させても残った**——その工程で曲づくりを止める合図（2026-09-30 本人方針）。

    投げるのは `StepFixer.run` だけ（保存から戻した出力＝再生では投げない）。投げる前に `<key>_fix` の記録
    （`status="failed"`・残った所見）を曲に積む。前の工程は途中保存（`song_flow._rec`）で残っていて、
    受け手（`song_compose._compose_once`）がこの記録ごともう一度保存する。
    `step` は `remake_from_step` に渡せる工程名（`song_gate.STEP_ORDER` の表記＝工程の `label`）。テーマで止まったときだけ
    `テーマ`（テーマからは作り直せないので、文は「もう一度頼む」になる）。

    **型の差し戻しが直らなかったときも同じ合図**（2026-09-30・判-19）。出力の型（選べる値・`field_validator` など）が受け付けず、
    pydantic_ai が書き直させても通らなかったら、`song_llm.run_sync` がこれに変えて投げる（`via="type"`）。
    findings は最後に LLM へ返した検証エラー。

    **最終の点検役の NG が差し戻しを使い切っても残ったとき**は、子の `FinalUnpassed` で止める（2026-10-04・組P2b）。
    受け手（`except StepUnfixed`・`STOPS`）はそのまま効き、文（`message`）と止めた理由（`why`）だけが違う。
    """

    #: 止めた理由の短い言い方（止めた印 `_progress.json` の `stopped.why`・完了通知・続きの案内が読む）
    why = "書いたものの形式が直らなかった"
    #: 直しの差分が差し戻され続けて止まったとき、`findings` の先頭の何件が元の所見か（`StepFixer._with_findings`・2026-10-04 組F3・4‑2）
    n_original = 0

    def __init__(self, step: str, findings: list[str], *, song: Any = None, rounds: int = 0, last_note: str = "",
                 via: str = "diff"):
        self.step, self.findings, self.song, self.rounds = step, list(findings), song, rounds
        #: 最後の直しの回に当てたことの一行（`apply` の戻り）。直しの担当が「直さなかった理由」を書いていればそこに入る
        self.last_note = last_note
        #: どこで直らなかったか: `diff`＝形式の検査を差分で直させた（`StepFixer.run`）／`type`＝出力の型の差し戻し（`song_llm.run_sync`）
        self.via = via
        super().__init__(f"{step}の形式の所見が直らなかった（{len(self.findings)}件）: "
                         + " / ".join(f[:80] for f in self.findings[:3]))

    def message(self, *, restart: str = "", edit: bool = False, how: str = "") -> str:
        """コーディネーターへ出す文（どの工程で・何が直らなかったか・どう続けるか）。

        - 新しい曲（`edit=False`）: 前の工程は途中保存に残っているので、`remake_from_step` に止まった工程を渡せば続きを作れる
        - 直しの途中（`edit=True`・作り直し／アレンジ／尺の変更）: 元の版はそのまま残り、直しは反映していない。
          途中まで作った工程は曲の記録に入っていないので、`restart`（直しを始めた工程）からもう一度頼む。
          `how` を渡すと、所見のあとの「どうなったか・どう頼み直すか」をその文に差し替える（道具ごとの言い方）
        """
        restart = restart or self.step
        rows = "\n".join(f"  ・{cut_note(f, 300)}" for f in self.findings[:10])   # 切ったら切ったと書く（2026-10-04 組F3）
        more = f"\n  ほか{len(self.findings) - 10}件" if len(self.findings) > 10 else ""
        how_tried = (f"直しの差分そのものが型・差分の検証で差し戻され（{self.rounds}回書き直させても通らなかった）、"
                     f"直せないまま残った（先頭の{self.n_original}件が直らなかった所見、後ろが差分を当てられなかった理由）"
                     if self.n_original else
                     f"出力の型（選べる値・書き方）で差し戻して{self.rounds}回書き直させても通らなかった"
                     if self.via == "type" else f"差分で{self.rounds}回直させても残った")
        head = (f"★**『{self.step}』の工程で止めた**——書いたものの形式（記法・個数・長さ・名前など、"
                f"答えが1つに決まる所）の所見が、{how_tried}:\n{rows}{more}\n"
                + (f"（最後の直しの回: {cut_note(self.last_note, 300)}）\n" if self.last_note else ""))
        if self.step == "テーマ":
            # テーマは曲の最初の工程で、`remake_from_step` はテーマから作り直せない（保存した工程もまだ無い）
            return (head + "**まだ曲はできていない**（最初の工程で止まったので、保存した工程も無い）。\n"
                    "続けるときは **draft_melody** でもう一度作る（上の所見が直るように、という一言を頼みに足してよい）。"
                    "\n**最終報告でマスターに伝える**: テーマの工程で何が直らなかったか、もう一度頼めること。")
        saved = edit or _autosaved()
        if edit:
            body = how or ("**この直しは曲に反映していない**（元の版はそのまま残っている）。\n"
                           f"続けるときは **remake_from_step(step='{restart}', request=同じ頼み)** でもう一度"
                           "（上の所見が直るように、という一言を頼みに足してよい）。")
        elif not saved:
            # ★**保存できなかったときに「保存してある」と言わない**（2026-10-03・再レビュー）。途中保存が書けなかったら
            #   `song_flow.AUTOSAVE_FAILED` が立つ。続きからは作れないので、最初から作り直す道を伝える
            body = ("**まだ曲はできていない**（楽譜も音も無い）。**ここまでの工程を保存できなかった**ので、続きからは作れない。\n"
                    "続けるときは **draft_melody** でもう一度作る（上の所見が直るように、という一言を頼みに足してよい）。")
        else:
            body = ("**まだ曲はできていない**（楽譜も音も無い）。ここまでの工程は保存してある。\n"
                    f"続けるときは **remake_from_step(step='{restart}', request=マスターの頼み)**——"
                    f"済んだ工程は費用なしで戻し、『{restart}』の工程から書き直す"
                    "（特に頼みが無ければ、上の所見が直るように書き直してほしい、と頼む）。")
        # 結びも保存の成否で分ける（2026-10-03・保存できなかったのに「続きから作れる」と伝えない）
        return (head + body + "\n**最終報告でマスターに伝える**: どの工程で・何が直らなかったか、"
                + ("直しは入っていないこと・もう一度頼めること。" if edit
                   else "続きから作れること。" if saved else "保存できなかったので最初から作り直すこと。"))


#: 止めたときに返す、最終の点検役の NG の1行の上限（字）。超えたら切ったことを書く（2026-10-04 組F1・4‑1）
FINAL_ROW_SHOWN = 3000


def cut_note(text: str, limit: int) -> str:
    """`text` を `limit` 字までにする。切ったら「…（以下略・全N字）」を付ける（黙って切らない）。"""
    t = text or ""
    return t if len(t) <= limit else t[:limit] + f"…（以下略・全{len(t)}字）"


class FinalUnpassed(StepUnfixed):
    """**最終の点検役の NG が、差し戻しを使い切っても残った**——完成扱いにせず止める合図（2026-10-04・組P2b・
    `docs/plan/song_final_reviewer.md` 決まったこと1）。

    投げるのは `song_final_gate.run_final_gate` だけ。`step` は**ここから続けられる**工程（最終の点検役が最後に原因と見た工程。
    分からなければ空）、`findings` は残った NG（どこがどう・どう直すか）。
    ★**新しい例外にせず `StepUnfixed` の子にした**: 止める・途中保存する・止めた印を書く・続きを案内する受け手（`song_compose._compose_once`・
      `tools/creation._stopped_message`・`tools/editing._stopped_edit`・工程の `except NOT_SWALLOW`）が、どれも `StepUnfixed` で
      受けている。別の例外にすると、その全部に1行ずつ足すことになり、足し漏れた所で「途中の曲を黙って完成扱い」になる。
      違うのは文（`message`）と止めた理由（`why`）だけなので、そこだけ上書きする。
    """

    why = "最終の点検役の NG が残った（差し戻しても直らなかった）"

    def __init__(self, step: str, findings: list[str], *, song: Any = None, rounds: int = 0):
        super().__init__(step, findings, song=song, rounds=rounds, via="final")
        Exception.__init__(self, "最終の点検役の NG が残ったので止めた"
                           + (f"（{step}から続けられる）" if step else "") + ": " + " / ".join(f[:80] for f in self.findings[:2]))

    def message(self, *, restart: str = "", edit: bool = False, how: str = "") -> str:
        """コーディネーターへ出す文（残った NG・ここから続けられる・どう頼むか）。引数は `StepUnfixed.message` と同じ。"""
        from song_gate import STEP_ORDER                    # noqa: E402（循環を避けてここで）
        # ★**NG は「どう直すか」まで全文を出す**（2026-10-04 組F1・4‑1）。以前は1行 400字で印なしに切れ、【確かめ方】や
        #   「ラスサビも同じ直しを」が消えた。長すぎるときだけ切り、切ったことを書く（頭から `FINAL_ROW_SHOWN` 字は残る）
        rows = "\n".join(f"  ・{cut_note(f, FINAL_ROW_SHOWN)}" for f in self.findings[:8])
        how_far = (f"{self.rounds}回差し戻して作り直しても NG と判断した" if self.rounds
                   else "NG と判断したが、差し戻せる工程を選べなかった")
        head = f"★**完成扱いにせず止めた**——曲全体を見る最終の点検役が、{how_far}（残った NG）:\n{rows}\n"
        step = self.step if self.step in STEP_ORDER else ""
        if edit:
            # 直しの途中: 元の版はそのまま。直しを始めた工程と、最終の点検役が原因と見た工程の、前のほうから頼み直す
            start = min((s for s in (restart, step) if s in STEP_ORDER), key=STEP_ORDER.index, default=restart or step)
            body = how or ("**この直しは曲に反映していない**（元の版はそのまま残っている）。\n"
                           f"続けるときは **remake_from_step(step='{start}', request=同じ頼み＋上の NG を直すように)** でもう一度。")
        elif not _autosaved():
            body = ("**まだ曲は完成していない**（楽譜・音は書き出していない）。**ここまでの工程を保存できなかった**ので、続きからは作れない。\n"
                    "続けるときは **draft_melody** でもう一度作る（上の NG が出ないように、という一言を頼みに足してよい）。")
        elif step:
            body = ("**まだ曲は完成していない**（楽譜・音は書き出していない）。ここまでの工程は保存してある。\n"
                    f"**ここから続けられる**: **remake_from_step(step='{step}', request=上の NG を直すように＋マスターの頼み)**——"
                    f"済んだ工程は費用なしで戻し、『{step}』（最終の点検役が原因と見た工程）から作り直す。")
        else:
            body = ("**まだ曲は完成していない**（楽譜・音は書き出していない）。ここまでの工程は保存してある。\n"
                    "続けるときは、上の NG を読んで原因の工程を選び **remake_from_step(step=その工程, request=上の NG を直すように)**。")
        return (head + body + "\n**最終報告でマスターに伝える**: 曲はまだ完成していないこと・残った NG（どこがどう）・"
                + ("もう一度頼めること。" if edit else "どこから続けられるか。"))


def _autosaved() -> bool:
    """いまの曲の途中保存が**失敗していない**か（`song_flow.AUTOSAVE_FAILED`）。song_flow を読めない場面（単体の検査）では書けている扱い。"""
    try:
        import song_flow
    except Exception:
        return True
    return not getattr(song_flow, "AUTOSAVE_FAILED", False)


class AssemblyError(Exception):
    """**組み立て（記号→音）に、形式の検査が保証しているはずの形が来た**——不具合の知らせ（2026-09-30）。

    形式の所見が直らなければ曲づくりは止まる（`StepUnfixed`）ので、組み立てが「鳴らせない所は鳴らさずに記録して進む」
    分岐へは構造上届かない。それでも来たら、検査と組み立ての読み方がずれている不具合なので、黙って鳴らさず・黙って進まず、ここで止める。
    検査が保証していない所（V6 や音源の限界など、組み立てでしか分からない所）はこれを使わず、今までどおり鳴らせなかったことを記録する。
    """


#: 工程の中の「失敗しても曲全体は落とさない」`except Exception` が握ってはいけない例外。そこでは `except STOPS: raise` を先に書く
STOPS = (StepUnfixed, AssemblyError)


@dataclass
class FixResult:
    """検査と書き直しの結果。`remaining` が空でなければ、直らなかった所見（`StepFixer.run` はここで止める）。"""
    value: Any
    remaining: list[str] = field(default_factory=list)
    rounds: list[dict] = field(default_factory=list)        # [{"findings": [...], "patched": "…"}]

    def summary(self) -> str:
        if not self.rounds:
            return ""
        parts = [f"{i}回目: 所見{len(r['findings'])}件 → {r.get('patched') or '直しなし'}"
                 for i, r in enumerate(self.rounds, 1)]
        tail = f"／残った所見 {len(self.remaining)}件" if self.remaining else "／所見なし"
        return "機械の検査で所見を返し、差分で書き直させた: " + "／".join(parts) + tail


def fix_loop(value: Any, check: Callable[[Any], list[str]],
             ask: Callable[[Any, list[str], int], Any],
             apply: Callable[[Any, Any], tuple[Any, str]],
             *, max_rounds: int = MAX_FIX_ROUNDS, verbose: bool = False, what: str = "",
             with_prev: bool = False) -> FixResult:
    """検査 → 所見を返す → 差分を当てる → 再検査。**値そのものを機械が書き換えることはしない**（当てるのは LLM の差分だけ）。

    - `check(value)` → 所見の文（空なら合格）
    - `ask(value, findings, round)` → LLM が返した差分（その工程の差分の型）。`with_prev=True` なら `prev=` に前の回
      （`{"findings", "patched", "sent"}`・1回目は None）も渡す（2026-10-04 組F3・3‑3。`StepFixer` が2回目の依頼文に前回の直しを出す）
    - `apply(value, patch)` → (差分を当てた新しい値, 何を当てたかの一行)
    """
    res = FixResult(value=value)
    findings = check(value)
    for rnd in range(1, max_rounds + 1):
        if not findings:
            break
        if verbose:
            print(f"      [検査] {what} 所見{len(findings)}件 → 差分で書き直し（{rnd}回目）")
            for f in findings[:5]:
                print(f"         ・{f}")
        patch = (ask(res.value, findings, rnd, prev=res.rounds[-1] if res.rounds else None) if with_prev
                 else ask(res.value, findings, rnd))
        res.value, note = apply(res.value, patch)
        res.rounds.append({"findings": list(findings), "patched": note, "sent": _patch_text(patch)})
        findings = check(res.value)
    res.remaining = list(findings)
    return res


#: 2回目の依頼文に出す、前回当てた差分の上限（字）。超えたら切ったことを書く（2026-10-04 組F3・3‑3）
PREV_PATCH_SHOWN = 1500


def _patch_text(patch: Any) -> str:
    """当てた差分を、次の回の依頼文に見せる形（JSON）にする。読めなければ空。"""
    if patch is None:
        return ""
    try:
        return patch.model_dump_json(exclude_defaults=True) if hasattr(patch, "model_dump_json") else str(patch)
    except Exception:                                       # noqa: BLE001（見せられなくても直しは進める）
        return ""


def prev_fix_text(prev: dict | None, rnd: int) -> str:
    """**2回目以降の依頼文の【前回の直し】**（2026-10-04 組F3・3‑3）。直しの呼び出しは毎回新しい会話なので、前回に何を当て・
    なぜ直さなかったかを書かないと、直しの LLM は同じ直し方を繰り返す（R3 で「前回」の語が依頼文に0件だった）。
    点検役の「前回の自分の判断」と同じ考え方。下の【機械の検査の所見】が、前回の直しのあとにまだ残った所見になる。"""
    if not prev:
        return ""
    rows = "\n".join(f"  ・{f}" for f in prev.get("findings") or [])
    sent = prev.get("sent") or ""
    return (f"\n\n【前回の直し（{rnd - 1}回目）】あなたは前回、次の所見に差分を返した:\n{rows}\n"
            + (f"  返した差分: {cut_note(sent, PREV_PATCH_SHOWN)}\n" if sent else "")
            + f"  機械が当てた結果: {prev.get('patched') or '直しなし'}\n"
            "  → 当てたあとにもう一度検査した結果が、下の【機械の検査の所見】（まだ残った所見）。"
            "前回と同じ直し方では直らない——所見の数字（いまの値とあるべき値）に合わせて直し方を変える。")


def record_change(song: Any, key: str, what: str, why: str = "") -> None:
    """**機械がどうしても直したこと**を曲の記録に残す（後の工程・作り直しの LLM に見える）。

    `key` は工程の出力名（`melody`・`lyrics`…）。記録は `StepRecord(step=f"{key}_machine")` に積み、
    `song_context.upstream_intents` が「■<工程> に機械がした変更」として依頼文へ足す。
    """
    if song is None or not what:
        return
    from song_model import StepRecord                      # noqa: E402（循環を避けてここで）
    reason = what + (f"（理由: {why}）" if why else "")
    # 同じ変更を二度積まない（作り直しで前回の記録を引き継いだあと、同じ処理がもう一度走ることがある）
    if any(r.step == f"{key}_machine" and r.reason == reason for r in (getattr(song, "history", []) or [])):
        return
    song.record(StepRecord(step=f"{key}_machine", reason=reason, status="ok", warnings=[what]))


def carry_records(song: Any, key: str, history: list[dict]) -> int:
    """**作り直しで前回の出力を戻した工程**について、前回の「機械がした変更」「差分で直させた結果」を引き継ぐ
    （2026-09-27 監査）。戻した出力は直したあとの値なので、処理をもう一度走らせても記録は出ない——
    引き継がないと、後の工程の LLM は「直したあとの値」を理由なしで前段の決定として読む。戻り値は引き継いだ数。
    """
    if song is None or not history:
        return 0
    from song_model import StepRecord                      # noqa: E402（循環を避けてここで）
    have = {(r.step, r.reason) for r in (getattr(song, "history", []) or [])}
    n = 0
    for h in history:
        st = h.get("step", "")
        if st in (f"{key}_machine", f"{key}_fix") and (st, h.get("reason", "")) not in have:
            song.record(StepRecord(**{k: v for k, v in h.items() if k in StepRecord.__dataclass_fields__}))
            n += 1
    return n


def machine_changes(song: Any, key: str) -> list[str]:
    """その工程の出力に機械がした変更（記録の順）。"""
    name = f"{key}_machine"
    return [r.reason for r in (getattr(song, "history", []) or []) if r.step == name]


# ---- 差分の検証（2026-10-02・洗い出し。どの工程の差分も同じ1つ）-----------------------------------------------------
#
# ★以前は各工程の `apply` が、範囲外の番号・同じ番号の2つ目・drop も中身も無い直しを**黙って捨てていた**（次の検査で同じ所見が
#   出るので音は変わらないが、LLM は直したつもりのまま、なぜ直らないかを知らずに回を重ねて止まる）。番号の範囲は曲ごとに違うので
#   差分の型には書けない——直しを頼んでいる間だけ工程が「差分 → 所見」を置き（`StepFixer.ask`）、直しの LLM の出力の検証
#   （`guard_patch`）が差し戻す。2026-10-02 に3つの組が別々に同じ形を作っていた（編成 `instrumentation._fix_no_in_range`・
#   伴奏と彩り `song_arrange._patch_in_range`・声部ほか `harmony.guard_patch_numbers`）のを、ここへ1つにまとめた（組「洗い出し-型と残り」）

from contextvars import ContextVar                          # noqa: E402

#: 直しを頼んでいる工程の「差分 → 当てられない書き方の所見」（`StepFixer.patch_problem`）。`StepFixer.ask` が頼む間だけ置く
PATCH_GUARD: ContextVar = ContextVar("PATCH_GUARD", default=None)


def guard_patch(agent):
    """直しの LLM の出力に、差分の検証を足す（`StepFixer.ask` が頼んでいる間だけ効く）。pydantic_ai が所見の文をその LLM へ返して
    書き直させ、直らなければ `song_llm.run_sync` がその工程で止める（`StepUnfixed`）。"""
    @agent.output_validator
    def _patch_problem(out):
        problem = PATCH_GUARD.get()
        if problem is not None and (why := problem(out)):
            from pydantic_ai import ModelRetry               # noqa: E402
            raise ModelRetry(why)
        return out
    return agent


def patch_no_problem(items: list[tuple[int, bool]], n: int, what: str) -> str:
    """差分の番号の所見（無ければ空）。`items` は (番号, drop も直す欄も無いか)。範囲は 1〜`n`。"""
    nos = [no for no, _e in items]
    out: list[str] = []
    if bad := sorted({no for no in nos if not 1 <= no <= n}):
        out.append(f"番号 {'・'.join(map(str, bad))} は【いまの{what}】に無い——{what}は{n}つなので、番号は 1〜{n}"
                   "（【いまの…】の番号をそのまま書く）")
    if dup := sorted({no for no in nos if nos.count(no) > 1}):
        out.append(f"番号 {'・'.join(map(str, dup))} を2回以上書いた——1つの番号に直しは1つ（直す欄を1つの項目にまとめる）")
    if empty := sorted({no for no, e in items if e and 1 <= no <= n}):
        out.append(f"番号 {'・'.join(map(str, empty))} は drop も直す欄も無い——直すなら直す欄を書き、直さないならその番号を書かない")
    return ("差分の番号: " + "／".join(out)) if out else ""


class PatchNumbers:
    """**差分の番号の決まりが同じ工程の部品**（声部・ハモリの計画・奏法・歌の崩し・歌い方。`StepFixer` の前に継ぐ）。

    工程は `patch_count(value)`（番号の数）と `patch_items(patch)`（差分 → [(番号, 中身が無いか)]）を持ち、直しの LLM を
    `guard_patch` に通しておく。`apply` は型を通らずに来た差分（保存から戻したものなど）だけ当てずに記録へ書く。
    `patch_what` は所見の「【いまの◯】」の◯（空なら工程名 `label`）。
    """
    patch_what: str = ""

    def patch_count(self, value) -> int:
        raise NotImplementedError

    def patch_items(self, patch) -> list[tuple[int, bool]]:
        raise NotImplementedError

    def patch_problem(self, value, patch) -> str:
        """差分の番号の所見（無ければ空）。番号の決まりが違う工程（歌い方の取り消し）だけ上書きする。"""
        return patch_no_problem(self.patch_items(patch), self.patch_count(value), self.patch_what or self.label)


# ---- 工程の形（2026-09-28・`docs/plan/song_step_objects.md`）----------------------------------------------

#: **形式の直しの依頼文の共通の頭**（所見の並びの前・2026-10-04 テーラリング。`prompts.md`「作業・直しを頼むときに渡すもの」の
#: 「形式の直し」の行）。前提（どの決まりか）と修正範囲（所見の所だけ・ほかは変えない）と優先（全部直す）を全工程に同じ文で渡す。
#: 完了の条件は「検査が通ること」で自明なので書かない。どう直すか（手順と例）は工程ごとの指示書（`*_fix.md`）と `return_hint` が持つ
FIX_SCOPE_NOTE = ("（どれも、この工程の書き方の決まり——指示書と依頼文に書いた記法・個数・長さ・枠・設計の区間——に照らして、"
                  "合っていない所です。各行に場所と、いまの値（分かるものはあるべき値）を書いています。"
                  "所見の出た所だけを直し、所見の無い所——言葉・音・狙い・ほかの番号の項目——は変えないでください。"
                  "所見は全部直します。1つでも残ると、この工程で曲づくりが止まります"
                  "（直すと狙いが壊れるときの扱いは、指示書の「直せないとき」のとおり））")

class StepFixer:
    """**工程が自分の出力を検査し、差分で直させる**ための同じ形（2026-09-28）。

    工程ごとにこれを1つ継いで、次だけを持つ。流れ（検査→所見を返す→差分を当てる→再検査→記録）は `run` の1か所。

    - `key` … 出力名（記録の `<key>_fix`・`<key>_machine`）／`label` … 工程名（LLM のラベル `<label>(直し{n})`・表示）
    - `agent` … 差分を返す LLM（その工程の差分の型を出力にする）
    - `check(value)` … **形式の検査**だけ（`song_checks` の部品を並べる）。音楽としての良し悪し（内容）は入れない。
      **記法（読めるか）はここに書かず `written` で宣言する**（記法1つ＝型1つ・`song_checks.Notation`・2026-09-29）
    - `show(value)` … いまの出力を LLM に見せる文／`return_hint` … 差分の返し方の一言
    - `apply(value, patch)` … 差分をそのまま当てる（元の値は書き換えない）→ (新しい値, 何を当てたかの一行)
    - `finish(value, res)` … 直したあとに値を作り直す必要がある工程だけ（例: 旋律の ABC → 音符）

    直させても所見が残ったら `StepUnfixed` で止める（2026-09-30。以前は `settle` で記録して点検役へ渡し、組み立てが
    鳴らせない所を鳴らさずに進んでいた）。保存から戻した出力（再生）は検査も直しもしない——作り直しで動かさず、
    再生で LLM を呼ばないため（保存された出力は、作ったときに検査を通っている）。
    """
    key: str = ""
    label: str = ""
    agent: Any = None
    #: 設計の区間（`kind`・`bars`）。渡すと、`design_items` が申告した区間名と小節数を親の決まり（`song_checks.against_design`）で見る
    design: Any = None
    need_all_sections: bool = False
    ordered_sections: bool = False                          # 区間を設計の順に続けて書く工程（旋律）
    return_hint: str = "**直す所だけ**を差分に入れてください。直さない所は書かないでください。"

    # -- 工程が持つもの --
    def check(self, value: Any) -> list[str]:
        raise NotImplementedError

    def show(self, value: Any) -> str:
        raise NotImplementedError

    def apply(self, value: Any, patch: Any) -> tuple[Any, str]:
        raise NotImplementedError

    def finish(self, value: Any, res: "FixResult") -> Any:
        return value

    def written(self, value: Any) -> Iterable[tuple[str, Any, str]]:
        """この工程が**出力のどこに・どの記法の文字列を書いたか** (場所, `song_checks.Notation`, 文字列)（2026-09-29）。

        記法の検査はこの宣言だけで効く（照合は `findings` の1か所）。`check` に記法の判定を書かない。記法を持たない工程は空のまま。
        """
        return ()

    def design_items(self, value: Any) -> list[tuple[str, str, int | None]]:
        """この工程が書いた区間名と小節数 (場所, 区間名, 小節数 or None)。設計との照合は親の決まりで見る"""
        return []

    def rename_sections(self, value: Any, mapping: dict[str, str]) -> Any:
        """書いた区間名を設計の綴りへそろえた**写し**を返す（`mapping` は 書いた名前→設計の名前）。区間名を持つ工程だけが持つ"""
        return value

    def patch_problem(self, value: Any, patch: Any) -> str:
        """**差分の当てられない書き方**の所見（無ければ空）——範囲外の番号・同じ番号の2つ目・中身の無い直しなど、いまの出力
        （`value`）を知らないと型に書けない決まり。直しを頼む間（`ask`）だけ、直しの LLM の出力の検証（`guard_patch`）が読んで
        差し戻す。番号の決まりが同じ工程は `PatchNumbers` を継ぐ。持たない工程は空のまま（検証は何もしない）。"""
        return ""

    def canonize(self, value: Any, song: Any = None) -> Any:
        """区間名の**前後の空白を取り除いた**写しを返す（組み立て・2026-09-28 新設・2026-10-02 から前後の空白だけ）。

        区間名の前後の空白は全角も取り除く決まり（区間名は半角だけの記法ではない・仕様書 §0-6 の表）で、設計との照合（`song_checks.resolve_section`）も取り除いて照らす。
        出力の名前も取り除いた形（＝設計の綴り）にしておけば、後の組み立て（コード表・旋律の枠・時間軸・転調）は完全一致で引ける。
        取り除いたことは記録する。設計に無い名前は触らない（設計との照合が所見にし、差分で直させる）。
        ★2026-10-01 までは大文字小文字・途中の全角空白の違いも設計の綴りへ書き換えていた。その観点（後の組み立てが1つの引き方で
          区間を引けること）はいま、設計との照合が完全一致でない名前を所見にして工程に直させることで保っている（判断6）。
        """
        if self.design is None:
            return value
        from song_checks import resolve_section              # noqa: E402（照らし方は1か所）
        names = [s.kind for s in self.design]
        mapping = {n: r for _w, n, _b in self.design_items(value) if (r := resolve_section(n, names)) and r != n}
        if not mapping:
            return value
        new = self.rename_sections(value, mapping)
        if new is not value and song is not None:
            record_change(song, self.key, "区間名の表記を設計にそろえた: " + "、".join(f"{a}→{b}" for a, b in mapping.items()),
                          "前後の空白だけの違い（前後の空白は取り除く決まり・仕様書 §0-6。後の組み立てが区間名を完全一致で引くため）")
        return new

    def findings(self, value: Any) -> list[str]:
        """形式の検査。並びは 設計との照合（親の決まり）→ 記法（`written` を親の `notation_findings` で）→ 工程の `check`
        （個数・長さ・範囲・名前）。"""
        from song_checks import notation_findings           # noqa: E402
        out = notation_findings(self.written(value)) + self.check(value)
        if self.design is not None:
            from song_checks import against_design              # noqa: E402
            out = against_design(self.design_items(value), self.design, need_all=self.need_all_sections,
                                 what=self.label, ordered=self.ordered_sections) + out
        return out

    def replayed(self, value: Any) -> bool:
        """保存から戻した出力か（印 `_replayed`、または再生中でこの工程の出力が保存にある）。"""
        if getattr(value, "__dict__", {}).get("_replayed"):
            return True
        try:
            import song_replay as rp                         # noqa: E402（循環を避けてここで）
        except Exception:
            return False
        st = getattr(rp, "_state", None)
        return st is not None and (st.raw or {}).get(self.key) is not None

    # -- 共通の流れ --
    def ask(self, value: Any, findings: list[str], rnd: int, prev: dict | None = None) -> Any:
        # 2回目以降は【前回の直し】（何を当て・なぜ直さなかったか）を所見の前に出す（2026-10-04 組F3・3‑3）
        text = (self.show(value) + prev_fix_text(prev, rnd)
                + "\n\n【機械の検査の所見】" + ("（前回の直しのあとにまだ残った所見）" if prev else "") + "\n"
                + FIX_SCOPE_NOTE + "\n"
                + "\n".join(f"・{f}" for f in findings) + "\n\n" + self.return_hint)
        token = PATCH_GUARD.set(lambda patch: self.patch_problem(value, patch))   # 頼む間だけ（差分の検証 `guard_patch` が読む）
        try:
            return self.call(text, rnd)
        except StepUnfixed as e:
            raise self._with_findings(e, findings) from e
        finally:
            PATCH_GUARD.reset(token)

    def _with_findings(self, e: StepUnfixed, findings: list[str]) -> StepUnfixed:
        """**直しの差分そのものが型・差分の検証で差し戻され続けて止まった**とき、元の所見を残して投げ直す（2026-10-04 組F3・4‑2）。

        ★以前は `song_llm.to_step_unfixed` が検証エラー（「差分の番号: 行 13 は【いまの旋律】に無い」）だけを所見にしたので、
          本当に直らなかった所見（「3行目の長さが36単位」）が、返事・止めた印（`stopped.findings`）・`<key>_fix` の記録から消えた。
          元の所見を先に、差し戻しの文を「直しの差分が当てられなかった理由」として後ろに置く。`via="type"` は保つ。
        別の工程・点検役の止まり（`step` が違う）と最終の点検役（`FinalUnpassed`）はそのまま通す。"""
        if isinstance(e, FinalUnpassed) or e.via != "type" or e.step != self.label or getattr(e, "n_original", 0):
            return e
        why = [f"直しの差分が当てられなかった理由（型・差分の検証の差し戻し）: {f}" for f in e.findings]
        new = StepUnfixed(e.step, list(findings) + why, song=e.song, rounds=e.rounds, last_note=e.last_note, via="type")
        new.n_original = len(findings)
        # 止める前に積まれた `<key>_fix`（`to_step_unfixed`・status=failed）にも元の所見を入れる（記録から消さない）
        for r in reversed(getattr(e.song, "history", None) or []):
            if r.step == f"{self.key}_fix" and getattr(r, "status", "") == "failed" and r.reason == e.last_note:
                r.warnings = list(new.findings)
                break
        return new

    def call(self, text: str, rnd: int) -> Any:
        from song_llm import run_sync                        # noqa: E402（循環を避けてここで）
        from song_usage import record as rec_usage          # noqa: E402
        label = f"{self.label}(直し{rnd})"
        r = run_sync(self.agent, text, label=label)
        rec_usage(self.agent, r, label)
        return r.output

    def run(self, value: Any, song: Any = None, *, verbose: bool = False) -> tuple[Any, list[str]]:
        """検査して、所見があれば差分で直させる。戻り値は (値, 残った所見)——**残った所見はいつも空**
        （直らなければ `StepUnfixed` で止める。戻り値の形は呼び側のために残してある）。

        直した経過（`FixResult`）は `self.result` に残る（呼び側が「◯回目: 所見◯件 → …」の一行を所見に添えるとき）。
        ★保存から戻した出力（再生）は**検査しない**ので止まらない（2026-09-30）。区間名の表記だけ設計の綴りへそろえる。
        """
        self.result = FixResult(value=value)
        if self.replayed(value):
            return self.canonize(value, song), []
        res = self.result = fix_loop(value, self.findings, self.ask, self.apply, verbose=verbose, what=self.label,
                                     with_prev=True)        # 2回目の依頼文に前回の直し（2026-10-04 組F3・3‑3）
        value = self.finish(res.value, res) if res.rounds else res.value
        if song is not None and res.rounds:
            from song_model import StepRecord                    # noqa: E402
            song.record(StepRecord(step=f"{self.key}_fix", reason=res.summary(), warnings=list(res.remaining),
                                   status="ok" if not res.remaining else "failed"))
        if res.remaining:
            # ★**形式の所見が直らなかったら、この工程で止める**（2026-09-30 本人方針）。前の工程は途中保存に残っている
            raise StepUnfixed(self.label, res.remaining, song=song, rounds=len(res.rounds),
                              last_note=(res.rounds[-1].get("patched") or "") if res.rounds else "")
        return self.canonize(value, song), []
