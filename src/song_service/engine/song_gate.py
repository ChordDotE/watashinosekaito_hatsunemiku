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

"""**工程ごとの関門**——機械が見つけたことを、別視点の LLM が判断する。

## 本番の「工程検問」との関係（**重複させないための整理**）

本番には既に同じ思想の仕組みがある。**作り直しではない**:

| | 本番 `tools/diagnosis.py: review_progress` | ここ（新パイプライン用）|
|---|---|---|
| いつ | **節目3回**（材料検問・整列検問・最終検問）| **工程ごと**（旋律・詞・伴奏。位置合わせは 2026-09-29 に外した）|
| 誰が判断 | 別視点の LLM 検査官（`prompts/reviewer.md`）| 同じ思想の関門（このファイル）|
| 何を渡す | 機械計測の事実 | 機械検査の所見（`Finding`）|
| 結果 | 健全 / 破綻あり / 要確認 ＋ 推奨アクション | accept / redo / stop ＋ 指示 |

本番の検問は**組み上がったあとの総点検**なので、「どの工程が原因か」を推定する必要がある。
こちらは**工程の直後**に置くので、原因が確定していて**やり直しが安い**。
両者は競合しないが、**本番へ移植するときは `review_progress` と役割を突き合わせること**
（節目検問を残しつつ、工程関門を前段に置く形を想定）。

本番の `reviewer.md` が置いている原則は**ここでも守る**——
「破綻あり は複数の独立した指標が同じ破綻を指す時だけ」「単独の★警告だけで破綻としない」。

## 前提・背景

各工程は `StepStatus` で自己申告（ok / compromised / failed）するようになったが、
**それを読んで何かを決める処理が無かった**。実際『夜明けと心音』（2026-08-22）は

    詞の警告 7件（音数と詞が合わない）＋ 伴奏の警告 1件（区間が薄すぎる）

を出しながら、**全工程が `ok` を自己申告して完成扱い**になった。
自己申告だけを信じると、こうして黙って抜ける。

## 論点: 機械で落としてよいか

「機械的な合否を入れて、否なら落とす」だと**意図した表現を殺す**。
たとえば**あえて無音にする**、あえて言葉を削ぎ落とす、といった作りは
検査から見れば「薄すぎ」「詞が足りない」に見える（2026-08-22 ユーザー指摘）。

## 決めたこと

**機械検査は証拠を集めるだけで、判定しない。** 判定はコーディネーター（LLM）が行う。

    各工程 → 自己申告 ＋ 機械検査の所見（Finding）
                    ↓
        コーディネーター: 「曲として破綻しているか？」
                    ↓
        accept（意図的表現かもしれない・このまま進む）
        redo  （この工程をやり直す・指示付き）
        stop  （直せないので、やり直さずに所見つきで進む。完成後に人へ報告する）

やり直しは**1工程あたり2回まで**。超えたら止めて人に報告する（費用が読めなくなるため）。
★**ここで判断するのは内容だけ**（2026-09-30）。形式の所見（記法・個数・長さ・名前——答えが1つに決まるもの）は
  工程の中で差分で直させ、直らなければ**その工程で曲づくりを止める**（`song_fix.StepUnfixed`）ので、点検役へは来ない。
  以前は直らなかった形式の所見も「（◯◯の検査で直らなかった）」として点検役へ渡し、`stop`（所見つきで進む）なら
  組み立てがその所を鳴らさずに最後まで作っていた。`stop`・差し戻しの打ち止め（`song_flow.NO_MORE_REDO`）の
  「止めずに最後まで作る」は、内容の判断についてだけのこと。
★**選べる判断は関門ごとに違う**（2026-09-27）。`judge(..., redo_from_options=, can_redo=)` で
  実際に動く行き先だけを見せ、選べない答えは一度聞き直す（`options_text`・`unavailable`）。
  結果は `Song.summary()` でコーディネーターへ届く。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, get_args

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from pydantic_ai import Agent

sys.path.insert(0, str(Path(__file__).resolve().parent))
import song_design as compose_v2                                          # noqa: E402,F401（bootstrap・API キー）
from song_llm import pick_model
from song_llm import run_sync as _llm_run
from song_usage import record as _rec_usage         # 費用の集計                            # noqa: E402
from song_prompts import prompt                          # プロンプトの正は prompts/*.md
from song_schema import Form                             # 曲の形の綴り（`FORM_STEPS`・`form_change`）
from song_fix import cut_note                            # 切ったら「…（以下略・全N字）」（黙って切らない）

MAX_REDO = 2                                               # 1工程あたりのやり直し上限
#: 点検役の依頼文に載せる**工程の申告（reason）**の上限（字）。超えたら切ったことを書く（`cut_note`）。
#: ★2026-10-04（組F2・§2‑3）: 以前は600字で黙って切っていた（伴奏の申告は実測で662字・930字あった）
REASON_LIMIT = 2000

# **前工程へ戻れる範囲**（作る順。左が早い）。`Verdict.redo_from` はここから選ぶ。
# ★**工程を足したら、ここと `song_context._STEP_KEY` の両方に足す**（2026-09-15）。
#   ここに無い工程名は `_maybe_redo_from` が**黙って無視する**——関門が
#   「詞の設計から作り直して」と言っても差し戻しが効かない。
#   触る場所の一覧は `docs/components/song_service.md`「工程を足すときに触る6か所」。
STEP_ORDER = ("設計", "コード", "グルーヴ", "旋律",
              "詞の設計", "サビの詞", "詞", "曲名", "歌い方", "歌の崩し",
              "編成", "伴奏", "ハモリ", "声部", "彩り", "奏法")
# ★**曲名は詞のすぐあと**（2026-10-03・本人了承）。テーマは仮の曲名を決め、詞の設計〜詞はそれを受けて書き、
#   詞ができてから最後の曲名を決める（仮の曲名をそのまま採ってもよい）。どの形でも走る（インストでは旋律のあと）。
# ★**声部 → 彩り の順**（2026-09-25）。彩りは声部の入った所を避けて置くので、
#   声部を作り直したら彩りも作り直す（逆は要らない）。

#: **工程の依存の木**（2026-09-25・`docs/plan/song_revision_design.md` §4）。
#: 値は**その工程が直接読んでいるもの**（コードで洗い出した）。`brief()` は
#: テーマ・設計・コード・旋律の意図を含むので、それを受け取る工程はこの4つに依存する。
#: ★**作り直しはこの木を辿る**——それまでは `STEP_ORDER` の**列の後ろ全部**を作り直していて、
#:   歌い方を直すと編曲が丸ごと（6工程）、詞を直すと詞を読まない編成・伴奏まで作り直していた。
#:   人の制作でも、歌を録り直してバンドを録り直すことはない。
#: ★**迷ったら依存ありに倒す。** 書き漏らすと、上流が変わっても古い結果が黙って残る。
#:   多めに書くと作り直しが増えるだけ（費用が増えるが、壊れない）。
#: ※曲名（工程『曲名』が決める・2026-10-03）は編成・伴奏・彩り・仕上げの点検役の見出しにも使うが、**音楽の入力ではない**ので数えない
#:   ——「曲名」を読む工程は無い。曲名だけを作り直すと、ほかは何も作り直さない（楽譜の曲名は組み立てが毎回写し直す）。
DEPS: dict[str, tuple[str, ...]] = {
    "設計":     ("テーマ",),
    "コード":   ("テーマ", "設計"),
    "グルーヴ": ("テーマ", "設計", "コード"),
    "旋律":     ("テーマ", "設計", "コード", "グルーヴ"),
    "詞の設計": ("テーマ", "設計", "コード", "グルーヴ", "旋律"),       # 旋律の譜割りを読む
    "サビの詞": ("テーマ", "設計", "コード", "グルーヴ", "旋律", "詞の設計"),
    "詞":       ("テーマ", "設計", "コード", "グルーヴ", "旋律", "詞の設計", "サビの詞"),
    # ★曲名（2026-10-03）は仮の曲名・何を歌うか（テーマ）・設計の要点・旋律の狙い・詞の設計（hook・視点）・サビの詞・詞の本文を読む。
    #   インストでは詞の系統が走らない（空の出力は前段として渡らない・`song_context.upstream_intents`）
    "曲名":     ("テーマ", "設計", "旋律", "詞の設計", "サビの詞", "詞"),
    "歌い方":  ("テーマ", "設計", "コード", "旋律", "詞"),             # 区間の盛り上がりは設計から
    # ★歌の崩しは歌い方を読む（2026-09-30・出-12 の前提）。以前は読まず、歌い方の「しゃくる」と崩しの scoop が同じ音に二重に掛かった
    "歌の崩し": ("テーマ", "設計", "コード", "旋律", "詞", "歌い方"),
    "編成":     ("テーマ", "設計", "コード", "旋律"),
    "伴奏":     ("テーマ", "設計", "コード", "グルーヴ", "旋律", "編成"),
    "ハモリ":   ("テーマ", "設計", "コード", "旋律", "編成"),
    # ★声部は設計を読む（2026-10-03・再レビュー）。区間の並び・歌の有無（アカペラの歌の無い区間の表 `harmony.quiet_sections_text`）を
    #   設計から直接読むのに、依頼文には設計が届いていなかった（作り直しの範囲は変わらない＝設計はコードを通して既に祖先）
    "声部":     ("設計", "コード", "旋律", "詞", "ハモリ"),             # 合いの手の言葉は詞に合わせる
    # ★彩りはコードを読む（2026-09-30・出-7）。以前は読まず、対旋律の同じ形を G/B・G・Em7・G7 のどの小節にも置いていた
    #   （依頼文の本文にもコード表と拍子を出す・`song_arrange.add_colour`）
    "彩り":     ("設計", "コード", "旋律", "編成", "伴奏", "声部"),     # 声部の所を避ける
    "奏法":     ("テーマ", "設計", "コード", "グルーヴ", "旋律", "編成", "伴奏", "彩り"),
}


def affected(step: str) -> set[str]:
    """**その工程を作り直したら、作り直しが要る工程**（自分と、自分を読んでいる工程の全部）。"""
    out, todo = {step}, [step]
    while todo:
        cur = todo.pop()
        for s, deps in DEPS.items():
            if cur in deps and s not in out:
                out.add(s)
                todo.append(s)
    return out


def is_ancestor(a: str, b: str) -> bool:
    """`a` を作り直すと `b` も作り直されるか（`a` が `b` の祖先か）。"""
    return a != b and b in affected(a)


# ---- 曲の形ごとに走る工程（2026-10-03・`docs/plan/song_acappella_instrumental.md` §3・§4-2）----------------
#
# ★`DEPS` は変えない。形ごとの「走る工程」の集合を足し、流れ（`_compose_once_body`）・作り直しの `affected`・関門の `redo_targets`・
#   コーディネーターに見せる一覧・`remake_from_step` の受け付けが、この集合と重ねて読む。走らない工程の出力は、同じ型で
#   中身が空・状態 skipped（`song_model.skipped_output`・`song_flow.fill_skipped`・2026-10-03 F1。以前は `None`）。
#   アカペラ: 伴奏・彩り・奏法が無い（ドラム・ベースの声・ボイパも作らない・F3）。インスト: 詞の設計〜詞・歌い方・歌の崩し・ハモリ・声部が無い。
#   組み立て（伴奏の組み立て・点検役・楽譜）は工程ではないのでここに無い（形で分かれるのは組み立ての担当）。

_ALL_STEPS = ("テーマ",) + STEP_ORDER
_NO_VOCAL = ("詞の設計", "サビの詞", "詞", "歌い方", "歌の崩し", "ハモリ", "声部")
_NO_ACCOMP = ("伴奏", "彩り", "奏法")

#: **形 → その形で走る工程**（テーマを含む・作る順）。形の綴りは `song_schema.Form`
FORM_STEPS: dict[str, tuple[str, ...]] = {
    "song": _ALL_STEPS,
    "acappella": tuple(s for s in _ALL_STEPS if s not in _NO_ACCOMP),
    "instrumental": tuple(s for s in _ALL_STEPS if s not in _NO_VOCAL),
}


class FormChange(BaseModel):
    """形を変えたときに、どの工程をどう扱うか（`form_change` の答え・§4-2）。どれも作る順の工程名。

    - `add` … 足す工程（新しい形にあって前の形に無い。前回の出力が無いので一から書く）
    - `remove` … 外す工程（新しい版の記録では中身が空・状態 skipped（F1）。前の版の記録には残る）
    - `remake` … 作り直す工程（`add` は含まない）。足す工程・外す工程・所見の出た工程を読んでいる工程と、所見の出た工程そのもの
    - `keep` … 前の版の出力をそのまま使う工程（形式の検査を通った・何も読み直さない）
    - `findings` … 残す工程の出力に新しい形の形式の検査を当てて出た所見（工程名 → 所見の文の並び）。所見の出た工程は `remake` に入る
    """
    old: str
    new: str
    add: tuple[str, ...] = ()
    remove: tuple[str, ...] = ()
    remake: tuple[str, ...] = ()
    keep: tuple[str, ...] = ()
    findings: dict[str, list[str]] = Field(default_factory=dict)


#: 形を変えたときに編成を作り直す理由（`FormChange.findings["編成"]` の頭に入る・機械の決まり）
FORM_REMAKE_REASON = "形が変わったので編成は作り直す（楽器・主旋律の楽器・声とハモリの方針の決まりが形ごとに違う）"


def form_change(old: str, new: str,
                check: Callable[[str], Iterable[str]] | None = None) -> FormChange:
    """**形を `old` から `new` へ変えたときの、足す・外す・作り直す・残す工程**（§4-2 の1〜4・機械が決める）。

    1. 足す工程＝`FORM_STEPS[new] − FORM_STEPS[old]`
    2. 外す工程＝`FORM_STEPS[old] − FORM_STEPS[new]`
    3. 残す工程（両方の形にある工程）に `check(工程名)` を当てる——**新しい形の形式の検査**（担当B の検査を呼ぶ口。
       戻り値は所見の文の並び・空なら通った）。所見が出た工程も作り直す。`check` を渡さなければ 3 は当てない。
       ★**編成は検査に頼らず、形が変われば必ず作り直す**（所見の頭に `FORM_REMAKE_REASON`・2026-10-03 親の判断）
    4. 作り直す工程＝ 1・2・3 の各工程の `affected`（依存の木 `DEPS` の子孫）を `FORM_STEPS[new]` と重ねたもの（足す工程は除く）。
       外す工程を読んでいた工程も入る（読む前段が無くなるため）

    `old == new` は断らない（全部が `keep` になる）——同じ形への変更を断るのは入口（`change_form`）の仕事。
    """
    forms = get_args(Form)
    for f in (old, new):
        if f not in forms:
            raise ValueError(f"形『{f}』は読めない（{' / '.join(forms)}）")
    before, after = FORM_STEPS[old], FORM_STEPS[new]
    add = [s for s in after if s not in before]
    remove = [s for s in before if s not in after]
    findings: dict[str, list[str]] = {}
    # ★**形が変わったら編成は必ず作り直す**（2026-10-03・§4-2「編成は形ごとに決まりが違うので、どの変更でも作り直しに入る」）。
    #   楽器の本数・主旋律の楽器 `lead`・声の種類とハモリの方針の決まりが形ごとに違うので、形式の検査が所見を出すかに頼らない
    #   （インスト→歌もので検査が通ると、ハモリ none・声を決めないままの編成が残る）
    if old != new and "編成" in before and "編成" in after:
        findings["編成"] = [FORM_REMAKE_REASON]
    if check is not None:
        for s in after:
            if s in before and (got := list(check(s) or [])):
                findings[s] = findings.get(s, []) + got
    touched: set[str] = set()
    for s in (*add, *remove, *findings):
        touched |= affected(s)
    remake = [s for s in after if s in touched and s not in add]
    keep = [s for s in after if s not in add and s not in remake]
    return FormChange(old=old, new=new, add=tuple(add), remove=tuple(remove),
                      remake=tuple(remake), keep=tuple(keep), findings=findings)

# **1曲あたりの差し戻し上限**（回数だけ）。
# ※費用の上限（`COST_LIMIT_FACTOR`＝最初に差し戻した時点の費用の1.5倍）は 2026-09-30 に外した（出-1。基準が早い工程での
#   1回目の費用なので、2回目はほぼ必ず上限に達していた。やめる条件は `song_compose._compose_song_rounds` の回数だけ）
MAX_REDO_SONG = 2


class Finding(BaseModel):
    """機械検査が見つけたこと。**良し悪しの判定は書かない**（それは関門の仕事）。"""
    what: str = Field(description="何を見つけたか（事実だけ）")
    evidence: str = Field(default="", description="数値・具体例（判断の材料）")


#: **stop が実際に起こすこと**（2026-09-27・目録の関門-2）。以前のスキーマは「人に報告して止める」と説明していたが、
#: 曲づくりはどの関門でも止まらずに最後まで作る（`song_flow.NO_MORE_REDO` の説明——途中の曲を返すと後段が壊れる）。
#: 説明を実際の動きに合わせ、`Song.summary` でコーディネーターへ届ける（そこから人へ報告される）。
STOP_MEANS = ("やり直さずに、**所見を記録に残して先へ進む**（曲は最後まで作り、完成後にコーディネーターと"
              "マスターへ報告される）。**壊れているが、やり直しても直らない／判断できない**とき")

#: 関門の名前 → 差し戻しの起点にする工程（`song_flow._GATE_STEP` と同じ。工程名そのものの関門は書かない）
GATE_STEP = {"仕上がりの抑揚": "伴奏"}

#: **工程名でない関門の記録の名前**（関門の名前 → 記録の `step`・2026-10-03・判-21）。工程の関門の記録は `<出力の名前>_gate`
#: （`song_flow.usage_record_name`）。仕上げの点検役（`song_finish_gate`）は後ろの工程をまとめて見るので工程名を持たない。
#: 費用の割り当て（`usage_record_name`）・保存した判断を戻す（`song_replay.Replay.judge`）・差し戻しの記録（`song_compose`）がここを引く
#: ★**最終の点検役**（2段目・`song_final_gate`・2026-10-04 組P2b）も工程名を持たない。点検役を足したら、ここと
#:   `song_gate_brief.GATE_SCOPE`（受け持ち）にも足す
#: ★**編集のあとの最終の点検**（`song_final_gate.judge_edit`・2026-10-04・案C）は記録を分ける（`final_gate_edit`）。`final_gate` に混ぜると、
#:   作り直しの「前の版で残った NG」の申し送り（`song_remake.final_ng_note`）に、マスターが「そのままでいい」と言った編集の NG まで届く。
#:   名前の括弧は全角（費用のラベル `関門(…)` の読み手 `song_flow.usage_record_name` が半角の `)` を落とすため）
GATE_RECORD = {"仕上げ": "finish_gate", "最終": "final_gate", "最終（編集のあと）": "final_gate_edit"}


def gate_of_record(record_step: str) -> str:
    """関門の記録の名前 → 関門の名前（`GATE_RECORD` にあるものだけ。無ければ空）。"""
    return next((g for g, r in GATE_RECORD.items() if r == record_step), "")

#: **その関門で「この工程だけやり直す」が実際に動くか**（2026-09-27・目録の関門-2）。
#: `song_compose` で自工程のやり直しの輪がある関門。呼ぶ側が `can_redo=` で渡せばそちらが正。
#: ★**旋律を足した**（2026-09-30）。以前は旋律の関門に自工程のやり直しが無く、点検役が「破綻」と判定しても
#:   `redo_from` が空の redo は**何もせずに先へ進んでいた**（旋律を1案にし、点検役を毎回呼ぶようにしたのと同時に作った）
#: ★**設計を外した**（同日）。設計の関門は区間名の重複の所見だけで開いていたが、重複は設計の型が差し戻すようにした
#:   （`song_design.Design._unique_kinds`）ので、設計の関門ごと無くなった
SELF_REDO = frozenset({"旋律", "詞", "伴奏"})


def redo_targets(step: str) -> list[str]:
    """`step` の関門から**戻って作り直せる工程**（作る順）。依存の木で `step` の祖先に当たる工程だけ。

    ★`song_flow._maybe_redo_from`（当時は `song_compose`）は、知らない工程・後ろの工程・祖先でない工程を**黙って無視**していた。
      選べない行き先を初めから見せない（祖先でない工程を作り直しても、`step` は同じものが出てくる）。
    """
    here = GATE_STEP.get(step, step)
    if here not in STEP_ORDER:
        return []
    return [s for s in STEP_ORDER if is_ancestor(s, here)]


class Verdict(BaseModel):
    """関門の判断。"""
    reason: str = Field(description=(
        "なぜそう判断したか。**曲として破綻しているか**を軸に書く。"
        "意図的な表現の可能性を検討したなら、それも書く"))
    # ★型で縛る（2026-09-27・目録の関門-3）。以前は `str` で受け、知らない語は**黙って accept** にしていた。
    action: Literal["accept", "redo", "stop"] = Field(description=(
        "accept … このまま進む（問題ない／意図的な表現とみなせる）\n"
        "redo   … やり直す（**破綻していて、やり直せば直る**とき）。**依頼文の【選べる判断】で"
        "選べると書かれているときだけ**\n"
        "stop   … " + STOP_MEANS))
    # ★修正役に届くのは指示だけ（2026-10-04・組F2・`docs/plan/song_feedback_review_20261004.md` §2‑1）。点検役が見た数字
    #   （区間の重心・重なった長さ・表の行）と理由は届かないのに、点検役はそれを知らされず、書き方の決まりも無かった
    # ★**依頼の項目をそろえる**（2026-10-04・テーラリング・`.claude/skills/miku-dev/details/coding/prompts.md`「作業・直しを頼むときに
    #   渡すもの」）。それまでの決まりは「どこ・何が・どうなっていて・どう直す」だけで、前提（何に照らしたか）・目的・変えない所・
    #   直ったと見る所・優先・迷ったときの扱いが無かった。**型は `str` のまま**（記録・再生・変換・再判断の材料が文字列で読んでいる）、
    #   見出しの語で形をそろえる
    # ★2026-10-04（組N2）: guidance・notes・redo_from も既定（空）をやめて必ず書かせる。「無し」は空の文字 `""` を明示で
    guidance: str = Field(description=(
        "`redo` のとき、**やり直す側へ渡す指示**（**必ず書く**。`redo` 以外は空の文字 `\"\"`）。\n"
        "★**やり直す側（修正役）は、あなたが見た材料（機械の事実・区間の表・パターンの表）も理由（`reason`）も見られません。"
        "見られるのはこの指示だけです。**\n"
        "次の見出しで、1つにつき1行以上で書く:\n"
        "  前提: 何に照らして NG としたか（頼み・設計の方針・テーマ・記法の決まり。照らした文を短く写す）\n"
        "  目的: 直した結果、聴く人にどう聞こえてほしいか\n"
        "  【必須】／【できれば】: 指摘。**指摘は1件1行**で、頭に優先度を付け、「**どこ**（区間・行・小節・パート）・"
        "**何が**（数字）・**どうなっていて**・**どう直す**（方向と理由。細かい手順は任せてよい）」を書く。"
        "**根拠の数字は指示に写す**（「表のとおり」「上の所見」では伝わりません）\n"
        "  変えない所: 直す所のほかで、崩してほしくない所（よく出来ている所・前の指摘で直った所）\n"
        "  直ったと見る所: 次の回に何を見て直ったとするか。数えられるものは数字で、聴かないと分からないものは観点で書く"
        "（数字の帯で決めつけない）。書けないときは省いてよいが、指摘の「どう直す」は必ず書く\n"
        "  迷ったら: 指摘どうし・指摘と決まりがぶつかりうるとき、どちらを取るか\n"
        "この工程だけのやり直しでは、前提・目的・指摘・変えない所は必ず、直ったと見る所・迷ったらは要るときに書く。"
        "**前の工程へ戻すとき（`redo_from`）は全部書く**（戻した工程から後ろを全部作り直すので、やり直しが重い）。\n"
        "  ○「前提: 設計の全体方針『サビで気持ちが開く』\n"
        "    目的: サビの頭で声が張り、Aメロから開けたと聞こえるように\n"
        "    【必須】chorus の重心 E4 は verse（E4）と同じ。chorus の1・3行目の頭を A4 以上へ上げ、最高音を chorus に置く"
        "（音域が上へ開くと、声が張って聞こえる）\n"
        "    【できれば】chorus の4行目は1行目と同じ形。終わりを伸ばして、サビの区切りを付ける\n"
        "    変えない所: verse・bridge の旋律と、各行の長さ（グルーヴが決めた枠）\n"
        "    直ったと見る所: chorus の重心と最高音が verse より上か。聴いて、サビの頭で開けたと感じるか\n"
        "    迷ったら: 気持ちよく鳴る上限（D6）を越えるなら上限を守り、上げ幅を小さくする」\n"
        "  ✕「サビをもっと開いて」\n"
        "「もう一度やって」では同じ結果になります"))
    notes: str = Field(description=(
        "**壊れてはいないが、弱いと感じたところ**（**必ず書く**。2026-09-16 追加）。\n"
        "★**`accept` のときこそ書いてください。** やり直すほどではないが"
        "「ここは物足りない」と思ったことを、**具体的に**書く"
        "（例:「verse1 の8行が1行ずつ別のことを言っていて、まとまりが無い」"
        "「間奏16小節が同じ型の繰り返しで、変化が無い」）。\n"
        "**これは曲を止めません**——記録に残り、**この工程の出力を読む後の工程にも届きます**"
        "（補えるところは後の工程が補う）。人が読むときの材料にもなります。"
        "後の工程と作曲担当も、あなたの材料は見られないので、**どこの・どの数字の話か**を書いてください。"
        "『特に無し』なら空の文字 `\"\"` と書きますが、**読んで何も感じないことは稀**です"))
    # ★**前の工程まで戻れる**（2026-09-01 追加）。人は致命的な破綻があれば前工程から
    #   作り直す。それまでは自工程しかやり直せず、伴奏の関門は「やり直すと編成から
    #   組み直しになる」という理由で**やり直しを諦めていた**。
    redo_from: str = Field(description=(
        "`redo` のとき、**どの工程まで戻ってやり直すか**（**必ず書く**）。その工程だけをやり直すとき・`redo` 以外は空の文字 `\"\"`。\n"
        "**戻れる先は関門ごとに違う**——依頼文の【選べる判断】に並んだ工程だけを書く"
        "（並んでいない工程を書いても差し戻せない）。\n"
        "**その工程をやり直しても直らないときだけ**前へ戻す——たとえば\n"
        "・詞が音符に収まらないのは、**旋律のフレーズが極端**（1音だけ／30音）だから\n"
        "・伴奏が薄いのは、**編成が2本しか選んでいない**から\n"
        "**戻るほど費用と時間がかかります**（戻った工程から後ろを全部作り直す）。"
        "その工程のやり直しで直るなら、空のままにしてください"))
    #: ★**機械がこの判断に対してしたこと**（LLM は書かない・スキーマに出さない）。LLM を呼ばずに止めた／
    #:   選べない行き先を選んだので止めた、など。`reason` の頭にも同じことを書く（記録に残るのは `reason`）。
    machine: SkipJsonSchema[str] = ""


_judge: Agent[None, Verdict] = Agent(
    pick_model("COORD", "flash"), output_type=Verdict, system_prompt=(
        prompt("song_gate_judge")
    ),
)


# ※複数案からの選抜（`pick_best`・`Pick`・指示書 `song_gate_pick.md`）は 2026-09-30 に外した（本人「一旦 A にしようか」）。
#   旋律とサビの詞を3案作って選んでいたが、選抜役は旋律の本文を読まず数字の札だけで選び、3案が工程費の約45〜49%だった。
#   どちらも1案にし、「方針と違う案」は点検役（`judge`）が本文を読んで見る


# **その工程で前に何を指摘したか**（無進展の検出に使う）。工程名 → 所見の指紋の並び。
# 曲をまたいで残らないよう `reset_progress()` で消す。
_SEEN: dict[str, list[frozenset[str]]] = {}

#: **関門ごとの、この曲で差し戻し（前の工程から作り直す）を投げた回数**（2026-10-03・判-21）。差し戻しの周をまたいで数える
#: （1周ごとに曲を作り直すので、曲の記録では数えられない）。仕上げの点検役の「1曲で1回まで」が読む。`reset_progress()` で消す
_REDO_SENT: dict[str, int] = {}


def note_redo_sent(gate: str) -> None:
    """その関門が差し戻しを投げたことを数える。"""
    _REDO_SENT[gate] = _REDO_SENT.get(gate, 0) + 1


def redo_sent(gate: str) -> int:
    """その関門がこの曲で差し戻しを投げた回数。"""
    return _REDO_SENT.get(gate, 0)


def reset_progress() -> None:
    """無進展の記録と差し戻しの回数を消す（**1曲＝1回**。曲をまたいで持ち越さない。`compose_song`・作り直しの入口が呼ぶ）。"""
    _SEEN.clear()
    _REDO_SENT.clear()


def _no_progress(step: str, findings: list[Finding]) -> str:
    """**同じ指摘が繰り返されていないか**を見る。戻り値は理由（空なら進んでいる）。

    ★やり直しの止め方が「回数」と「費用」しか無かった（2026-09-02 追加）。
      どちらも**天井**であって、**堂々巡りそのもの**は見ていない——
      同じ所見を2回出しても、上限までは回り続ける。**回った結果 何も変わっていない**なら、
      3回目も変わらない。**そこで止めて人に返すほうが速いし安い**。

    見るのは**所見の顔ぶれ**（`what` の集合）。文言の細かな揺れではなく
    「**同じ種類の問題が残っているか**」で判定する。
    """
    now = frozenset(f.what for f in findings)
    prev = _SEEN.setdefault(step, [])
    same = sum(1 for x in prev if x == now)
    prev.append(now)
    if same >= 1:                                          # 前と同じ顔ぶれが再び出た
        return (f"やり直したのに**同じ指摘が {same + 1} 回目**"
                f"（{'／'.join(sorted(now)) or '所見なし'}）——**進んでいない**")
    return ""


def options_line(can_redo: bool, redo_from_options: list[str], allow_stop: bool = True) -> str:
    """選べる判断の1行（記録・聞き直し用）。"""
    parts = ["accept"] + (["stop"] if allow_stop else [])
    if can_redo:
        parts.append("redo（この工程だけ）")
    if redo_from_options:
        parts.append("redo_from=" + "／".join(redo_from_options))
    return "選べるのは " + "・".join(parts)


def options_text(can_redo: bool, redo_from_options: list[str], allow_stop: bool = True) -> str:
    """点検役へ見せる【選べる判断】。**実際に動くものだけ**を並べる（2026-09-27・目録の関門-2）。
    `allow_stop=False` … stop（所見つきで進む）の無い関門（最終の点検役・2026-10-04 組P2b。妥協して出さない）。"""
    lines = ["【選べる判断】（この関門で実際に動くものだけ）",
             "  ・accept … このまま進む（問題ない／意図した表現とみなせる）"]
    if allow_stop:
        lines.append(f"  ・stop … {STOP_MEANS}")
    if can_redo:
        lines.append("  ・redo（`redo_from` は空）… **この工程だけ**をやり直す")
    if redo_from_options:
        lines.append("  ・redo（`redo_from` に工程名）… その工程まで戻って作り直す。戻れる先: "
                     + " / ".join(redo_from_options))
    if not can_redo and not redo_from_options:
        lines.append("  ※この関門では**やり直し（redo）は選べない**。" + ("壊れていて直せないなら stop" if allow_stop else ""))
    elif not can_redo:
        lines.append("  ※この関門では**この工程だけのやり直しは動かない**。redo なら必ず `redo_from` を書く")
    if not allow_stop:
        lines.append("  ※この関門には**stop（所見つきで進む）は無い**。通すなら accept、NG なら redo で戻す工程を書く")
    return "\n".join(lines)


def unavailable(v: Verdict, step: str, can_redo: bool, redo_from_options: list[str], allow_stop: bool = True) -> str:
    """その判断がこの関門で**選べない**なら、何を選んだかの説明（選べるなら空）。"""
    if v.action == "stop" and not allow_stop:
        return "「所見つきで進む（stop）」"
    if v.action != "redo":
        return ""
    back = (v.redo_from or "").strip()
    if back in ("", step, GATE_STEP.get(step, step)):     # 空・自分の名前＝この工程だけのやり直し
        return "" if can_redo else "「この工程だけやり直す」"
    return "" if back in redo_from_options else f"「{back}から作り直す」"


def judge(step: str, reason: str, status: str, trouble: str,
          findings: list[Finding], *, attempt: int = 0,
          context: str = "", material: str = "", verbose: bool = True,
          redo_from_options: list[str] | tuple[str, ...] | None = None,
          can_redo: bool | None = None, agent: Agent | None = None, stop_step: str = "",
          before: Any = None, after: Any = None, allow_stop: bool = True, ask: str = "") -> Verdict:
    """関門を1回通す。**上限を超えたか、進んでいないなら LLM を呼ばずに止める**（止めたことは `reason` に書く）。

    `agent` … 点検役の LLM（省略時は工程の点検役 `_judge`＝指示書 `song_gate_judge.md`）。仕上げの点検役は自分の指示書の agent を渡す
    `stop_step` … 点検役の答えが型を通らなかったとき止める工程（省略時は `GATE_STEP` で引いた工程＝関門の名前）。
    工程名でない関門（仕上げ）は、その形で走る最後の工程を渡す（2026-10-03・判-21）

    ★**選べる判断は、実際に動くものだけ見せる**（2026-09-27・目録の関門-2）。スキーマは全工程を戻り先として並べ、
      `stop` を「人に報告して止める」と説明していたが、曲づくりはそのどちらもしていなかった
      （知らない・後ろの・祖先でない戻り先は黙って無視、旋律と位置合わせの関門の redo は何も起きない、stop は進む）。
      - `redo_from_options` … この関門から戻れる工程。**省略時は依存の木の祖先**（`redo_targets`）。
        差し戻しが打ち止め（`song_flow.NO_MORE_REDO`）なら、呼ぶ側が `[]` を渡す
      - `can_redo` … 「この工程だけやり直す」が動くか。省略時は `SELF_REDO`
      選べない答えが来たら**一度だけ聞き直し**、それでも選べないなら `stop`（所見つきで進む）にして、
      そのことを `reason` と `machine` に残す（黙って読み替えない）。

    依頼文の頭の受け持ち・マスターの頼み・前の点検役の判断と、選べる判断の手前の再判断の材料（前回の自分の判断・作業者の対応・
    前の稿から変わった所・上からの指摘）は、ここが `song_gate_brief.gate_brief` で足す（2026-10-04・組P1／P2a。呼ぶ側は渡さない）。
    `before` / `after` … この工程の前の回の稿と新しい稿（2026-10-04・組P2a）。自工程のやり直しのある関門（旋律・詞・伴奏）が渡す。
    前の回の判断は記録（`*_gate` の `guidance`・`notes`）から読むので、呼ぶ側が記録を書いてから次の回を呼ぶ。

    **最終の点検役**（2段目・`song_final_gate`・2026-10-04 組P2b）のための引数:
      - `allow_stop=False` … stop（所見つきで進む）を見せず、選んだら選べない答えとして聞き直す（妥協して出さない）
      - `ask` … 依頼文の結びの問い（省略時は工程の点検役の問い）
      - `findings=None` … 機械検査の無い点検役（【機械検査の所見】の節を出さない）。`status`・`reason` が空なら、その行も出さない

    `material` は**この関門が自分の目で読むもの**（詞なら歌詞の本文、伴奏ならパターンの表）。
    ★**2026-09-16 に追加**——それまで関門は**機械検査の所見しか見ていなかった**ので、
    「歌詞を一度も読まずに『曲として破綻していますか』と聞かれている」状態だった。
    機械が測っていない観点（**言葉のまとまり・展開の単調さ**）は、入口にすら届かなかった。
    本人の指摘「**1番の中で歌詞のまとまりがない**」「**間奏が単調**」がどちらもこれ。

    ※**全部を渡さない。** 工程の記録は 186KB（≒63k トークン）あり、うち7割は旋律の枠で、
      詞や伴奏の良し悪しには使えない。**混ぜると肝心の本文が埋もれる**
      （このプロジェクトには「200行超のプロンプトでは埋もれる」の実測がある——
      同じ指示を短く渡したら拍頭率が 0.520 → 0.194 になった）。
      **その関門が読むべきものだけ**を渡すこと。
    """
    # ★**LLM を呼ばずに止めるとき（合成した stop）**も、機械が止めたことを残す（目録の関門-3・(d) 残す）
    # ★**やり直しの最後の稿（`attempt == MAX_REDO`）も点検役が見る**（2026-10-04 組F1・本人決定「見せる」＝案A）。以前は点検役を呼ばずに
    #   機械が止め、費用を払った2回目の稿を誰も点検せず、指摘が直ったかの照合も無かった。最後の稿では**自工程のやり直しだけ選べない**
    #   （通す／前の工程へ差し戻す〔回数が残っていれば〕／弱い所を書いて進む）。上限を超えた呼び出しは来ないはずだが、来たら機械が止める
    if attempt > MAX_REDO:
        note = f"やり直しが上限（{MAX_REDO}回）を超えたので、点検役を呼ばずに機械が止めた（所見つきで進む）"
        return Verdict(reason=f"【機械】{note}", action="stop", guidance="", notes="", redo_from="", machine=note)
    last_try = attempt >= MAX_REDO
    # ★**出し直しの稿は、機械の所見の顔ぶれが前と同じでも点検役に見せる**（2026-10-04・案A「見せる」の徹底・組F4 の指摘）。
    #   以前は1回目のやり直しのあとで顔ぶれが同じ（所見なしが2回も含む）だと点検役を呼ばずに機械が止めていた。内容の所見は
    #   事実を並べるだけなので顔ぶれが同じになりやすく、出し直した稿を誰も点検しないまま後ろへ流れていた。回り続けることは
    #   やり直しの上限（`MAX_REDO`）が止めるので、同じだったことは依頼文に書いて点検役に判断させる
    stuck = _no_progress(step, findings) if attempt else ""

    opts = list(redo_targets(step) if redo_from_options is None else redo_from_options)
    self_ok = ((step in SELF_REDO) if can_redo is None else bool(can_redo)) and not last_try
    # ★**受け持ち・頼み・前の点検役の判断・前回の自分の判断を、ここ1か所で足す**（2026-10-04・組P1・`song_gate_brief`）。
    #   点検役ごとに手で足すと漏れる。表（`GATE_SCOPE`）に無い点検役とカバーには足さない
    import song_gate_brief as gb                           # noqa: E402（循環を避けてここで）
    import song_gate_checks as gc                          # noqa: E402
    head, prev_own = gb.gate_brief(step, attempt, before=before, after=after)
    # ★**【この工程で機械が確かめたこと】もここ1か所で足す**（2026-10-04・`song_gate_checks`）。①形式の検査で保証済みのこと
    #   （受け持ちの工程の分）②機械が数えた事実の項目 ③機械が見ていないこと。表に無い点検役には足さない。カバーの点検役にも足す。
    #   所見が無いとき「見た結果無かった」と分かるよう、この節があれば所見なしの文もそう書く
    checks = gc.checks_brief(step)
    body ="\n".join(f"  ・{f.what}" + (f"（{f.evidence}）" if f.evidence else "")
                     for f in (findings or [])) or (gc.NO_FINDINGS if checks else "  （機械検査は何も見つけていない）")
    prompt = (
        (f"{head}\n\n" if head else "")
        + (f"{context}\n\n" if context else "")
        + f"【工程】{step}\n"
        + (f"【この工程の自己申告】{status}\n" if status else "")
        + (f"【本人が困ったと言っていること】{trouble}\n" if trouble else "")
        + (f"【この工程が何をしたか】{cut_note(reason, REASON_LIMIT)}\n" if reason else "") + "\n"
        + (f"{checks}\n\n" if checks else "")
        + (f"【出来たもの——**自分の目で読んでください**】\n{material}\n\n" if material else "")
        + (f"【機械検査の所見】\n{body}\n\n" if findings is not None else "")
        + (f"※これは {attempt} 回目のやり直し後の結果です。\n" if attempt else "")
        + (f"※機械の所見の顔ぶれは前の回と同じです（{stuck}）。\n" if stuck else "")
        + (f"※**この工程のやり直しは上限（{MAX_REDO}回）を使い切りました。この稿が最後です**——直っていれば accept。"
           + ("まだ NG で原因が前の工程にあるなら redo（`redo_from` に戻る工程）。" if opts else "")
           + ("それ以外は、残った弱い所を `notes` に書いて stop（所見つきで進む。最終の点検役が曲全体を見ます）。\n"
              if allow_stop else "\n")
           if last_try else "")
        + (f"{prev_own}\n\n" if prev_own else "")
        + options_text(self_ok, opts, allow_stop) + "\n\n"
        + (ask or ("**①曲として破綻していますか**（`action`）。"
                   "**②壊れてはいないが弱いところはどこですか**（`notes`）。"
                   if material else "曲として破綻していますか。どうしますか。")))
    # 点検役の答えが型（`Verdict`）を通らず書き直しでも直らなければ、点検している工程で止める（`stop_step`・2026-09-30）
    who = agent if agent is not None else _judge
    stop_at = stop_step or GATE_STEP.get(step, step)
    _r = _llm_run(who, prompt, label='関門', stop_step=stop_at)
    _rec_usage(who, _r, f"関門({step})")                   # 費用集計から漏れていた
    v = _r.output
    if (bad := unavailable(v, step, self_ok, opts, allow_stop)):
        # ★一度だけ聞き直す（選べる判断を示して）。それでも選べないなら所見つきで進み、そう記録する
        _r = _llm_run(who, prompt + f"\n\n★前の答え（{bad}）はこの関門では選べません。"
                                    "【選べる判断】の中から選び直してください。", label='関門',
                      stop_step=stop_at)
        _rec_usage(who, _r, f"関門({step}・聞き直し)")
        v = _r.output
        if (bad2 := unavailable(v, step, self_ok, opts, allow_stop)):
            # ★stop の無い関門（最終の点検役）でも機械は stop として返す——扱いはその関門の呼び手が決める（最終の点検役は曲を止める）
            note = (f"点検役は{bad2}を選んだが、この関門では選べない（{options_line(self_ok, opts, allow_stop)}）。"
                    + ("機械がやり直さずに所見つきで進めた（stop として扱う）" if allow_stop
                       else "機械は差し戻せない（stop として返す。呼び手が曲を止める）"))
            v = v.model_copy(update={"action": "stop", "machine": note,
                                     "reason": f"【機械】{note}。点検役の理由: {v.reason}"})
    if verbose:
        mark = {"accept": "通す", "redo": "やり直し", "stop": "所見つきで進む"}[v.action]
        print(f"  [関門] {step}: **{mark}** — {v.reason[:130]}")
        if v.guidance:
            print(f"         指示: {v.guidance[:160]}")
        if v.notes:                                        # 通しても、弱いところは必ず出す
            print(f"         ▲気になったところ: {v.notes[:200]}")
    return v
