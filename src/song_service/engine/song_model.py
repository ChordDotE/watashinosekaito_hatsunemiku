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

"""**曲そのものを1つの入れ物にまとめる**（D9）。

## なぜ要るか

いままで曲の状態は `compose_v3()` のローカル変数に散っていた。そのため:

  ・**編集で工程を呼び直せない**（前工程の成果が関数の中にしか無い）
  ・**どう作ったかが残らない**（steps.json に理由の文字列だけ）
  ・**失敗が上に伝わらない**（工程が妥協しても、それらしい出力が返るだけ）
  ・カバーと新規作成が別のフローに見える（**実際は同じ曲を埋めているだけ**）

`SongScore`（楽譜＝音の実体）は既に集約オブジェクトになっているので、
**その上の層**——工程の成果・作成の経緯・カバーの参照——をまとめる。

## これで3経路が同じ形になる

    カバー:     Song を作る → 採譜で melody を埋める → 歌詞を取得 → 伴奏を採譜
    新規作成:   Song を作る → LLM で theme → design → … を埋める
    編集:       既存の Song を読む → **指定の工程だけ埋め直す**
    エラー復帰: 同上（きっかけが人か工程かの違いだけ）

## この設計が置いている前提

| # | 前提 | 崩れると |
|---|---|---|
| 1 | 工程は **Song を受け取り、担当パートだけ埋める** | 他の工程の成果を壊す |
| 2 | 工程は必ず `StepRecord` を返す | 履歴が欠け、編集時に前回の判断を渡せない |
| 3 | `history` は**追記のみ**（消さない）| 「なぜそうしたか」が失われる |
| 4 | 工程ごとの出力クラスは**別のまま**（共通の親を作らない）| 型が緩くなり、Pydantic の検査が効かなくなる |
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, Field, model_validator

from song_schema import Form, SongScore                 # noqa: F401（`Form` は曲の形・ここからも引ける）

# 工程の名前（順序どおり）。**この並びが「曲の作り方」**
STEPS = ["theme", "design", "chords", "groove", "melody", "lyrics", "tuning",
         "instrumentation", "arrange", "harmony"]

STEP_LABEL = {
    "theme": "テーマ", "design": "設計", "chords": "コード", "groove": "グルーヴ",
    "melody": "旋律",
    "lyrics": "詞", "naming": "曲名", "tuning": "調声", "instrumentation": "編成",
    "arrange": "伴奏", "harmony": "ハモリ",
}


# ※記録の形の版（`RECORD_FORMAT`）と、伴奏に彩りが混ざった 2026-09-25 までの記録を読み替える `normalize_record` は
#   2026-09-30 に外した（リリース前なので古い記録は読まない）。伴奏は骨組みだけ、彩りは `colour` にだけ持つ。


#: **工程の自己申告の選択肢**（2026-10-02・組「洗い出し-型と残り」）。以前は `str` で受け、記録するとき（`song_flow._rec`）に
#: `strip().lower()` で読んでいた（`OK`・` Failed ` も通り、知らない語は表示で `?` になるだけ）。いまは型が3つの綴りだけを受け、
#: ほかは差し戻す。保存済みの記録（工程 508・履歴 780）はどれもこの3つ（2026-10-02）。
#: ★**`skipped`＝走らなかった**（2026-10-03・F1 本人決定「インストやアカペラでも、データ形式は変えないようにしたい。空配列とかで」）。
#:   その曲の形で走らない工程（インストの詞・アカペラの伴奏など・`song_gate.FORM_STEPS`）の出力は、ほかの工程と同じ型で
#:   **中身が空・状態 skipped**（`skipped_output`）。以前は出力が None で、記録の形が曲の形で変わっていた。
#:   **機械だけが書く**——工程の LLM に見せる型の選択肢（JSON スキーマ）は3つのまま（`SELF_REPORT`・`StepStatus.status`）
Status = Literal["ok", "compromised", "failed", "skipped"]
#: 工程の LLM が申告できる3つ（`skipped` は機械が書く）
SELF_REPORT = ("ok", "compromised", "failed")
SKIPPED = "skipped"


def _llm_choices(schema: dict) -> None:
    """工程の LLM に見せる状態の選択肢を3つにする（`skipped` は見せない・機械だけが書く）。"""
    schema["enum"] = list(SELF_REPORT)


class PatchStatus(BaseModel):
    """**自己申告（状態・困りごと）だけ**の土台。差分の直し（`*Patch`・`ArticFix`）の型はこれを継承する
    （差分には「指摘への対応」`response` を持たせない・2026-10-04 組P2a）。工程の出力の型は `StepStatus` を継承する。

    以下は 2026-08-22（D8）からの説明:

    「失敗が失敗として上がってこない」ために、V6 で開くまで気づけない不具合が4件続いた
    （歌詞の動詞が削られた／編成の指示を伴奏が無視した／ハモリが1音も作られなかった／
    音量が既定値のままだった）。**妥協したなら妥協したと言わせる**のがこの欄の目的。

    追加のトークンは数十で、費用はほぼゼロ。
    """
    status: Status = Field(default="ok", description=(
        "この工程がどうだったか。**正直に申告してください**（次の3つのどれかを、この綴りの小文字で）:\n"
        "  ok          … 指示どおりにできた\n"
        "  compromised … **何かを妥協した**（例: 音符数に合わせて言葉を削った・"
        "音域に収めるため旋律の山を削った・指定の楽器では表現できず別の音を使った）\n"
        "  failed      … できなかった"), json_schema_extra=_llm_choices)
    trouble: str = Field(default="", description=(
        "`ok` 以外のとき、**何がどう困ったか**を具体的に。"
        "**どこが・なぜ・何を諦めたか**が分かるように書く"
        "（例:「フレーズ7は音符11個だが、言いたいことに14モーラ必要だった。"
        "『踏み出す』を削って辻褄を合わせたので、意味が通らなくなっている」）。"
        "これを読んだディレクターが**やり直すか、条件を変えるか、そのまま進むか**を判断します"))

    @model_validator(mode="after")
    def _skipped_is_empty(self):
        """`skipped`（走らなかった）は機械が書く中身の空の出力だけ。理由・困りごとが書いてあれば工程が書いたもの＝差し戻す。"""
        if self.status == SKIPPED and (getattr(self, "reason", "") or self.trouble or getattr(self, "response", "")):
            raise ValueError("status の skipped は機械だけが書く（走らなかった工程の印）。ok / compromised / failed のどれかを書く")
        return self


#: **指摘への対応**の欄の説明（LLM に見せる・2026-10-04 組P2a）。指摘が無い呼び出しでは空のままと分かるように書く
RESPONSE_TEXT = (
    "**指摘への対応**。点検役・最終の点検役・後ろの工程からの指摘を受けて直したとき（依頼文に【前回の指摘】や"
    "差し戻し・作り直しの頼みがあるとき）**だけ**、指摘ごとに1行で「直した（どこをどう）」か「直さなかった（なぜ）」を書く。"
    "点検役は、この対応と前の稿からの実際の変化を見比べて判断します。"
    "**指摘が無い呼び出しでは空のまま**にしてください（理由や工夫は reason に書く）")


class StepStatus(PatchStatus):
    """**全工程の出力に付ける自己申告**（D8）。各工程の出力型がこれを継承する（差分の型は `PatchStatus`）。

    ★**指摘への対応 `response`**（2026-10-04・組P2a・`docs/plan/song_final_reviewer.md`「追記: 監督者を人として見たとき」の2）。
      点検役がやり直させた・差し戻した稿を見るとき、作業者がどの指摘にどう応えたかが分からなかった。指摘を受けて直す呼び出し
      （自工程のやり直し・前の工程への差し戻し・作り直しの頼み）だけが書き、点検役の再判断の材料になる
      （`song_gate_brief.gate_brief`）。欄の無い古い記録・走らなかった工程の出力は空。
    """
    response: str = Field(default="", description=RESPONSE_TEXT)


def _empty_of(ann: Any) -> Any:
    """型の**空**（文字 ""・並び []・辞書 {}・真偽 false・数 0・省ける欄 None）。走らなかった工程の出力の必須の欄に入れる。"""
    origin = get_origin(ann)
    if ann is str:
        return ""
    if ann in (list, tuple, set) or origin in (list, tuple, set):
        return []
    if ann is dict or origin is dict:
        return {}
    if ann is bool:
        return False
    if ann in (int, float):
        return 0
    if origin in (Union, UnionType) and type(None) in get_args(ann):
        return None
    raise TypeError(f"走らなかった工程の出力: 必須の欄の型 {ann!r} の空を決めていない（`song_model._empty_of` に足す）")


def skipped_output(model_cls: type[StepStatus]) -> StepStatus:
    """**走らなかった工程の出力**（2026-10-03・F1）。その工程の出力の型のまま、中身が空で状態が `skipped`。

    必須の欄は型の空（`_empty_of`・理由の文も空——機械は中身を作らない）、既定のある欄は既定のまま。型の検査は通す
    （詞の設計の `hook_line` は 0＝置く行が無い。差分の `LyricPlanPatch.hook_line` の 0＝「いまのまま」と同じ読み）。"""
    data = {n: _empty_of(f.annotation) for n, f in model_cls.model_fields.items() if f.is_required()}
    return model_cls.model_validate({**data, "status": SKIPPED})


def is_skipped(out: Any) -> bool:
    """工程の出力が「走らなかった」か（型でも、保存から読んだ dict でも）。"""
    st = out.get("status") if isinstance(out, dict) else getattr(out, "status", None)
    return st == SKIPPED


def made(out: Any) -> bool:
    """工程の出力が**ある**か——作った（None でない）うえで、走らなかった印（`skipped`）でもない。
    「その工程はもう作ったか」「前段の出力を渡すか」「前回の出力として見せるか」はこれで見る（`is not None` で見ない）。"""
    return out is not None and not is_skipped(out)


#: ★**工程の記録の所見（`StepRecord.warnings`）がどこへ届くか**（2026-10-03・判-22）。記録の名前（`StepRecord.step`）→ 届け先。
#:   以前は `Song.summary` が点検役の判断・差分の直し・機械の変更・ok でない工程しか拾わず、**状態 ok で所見つきの記録**
#:   （声部の推奨音域・彩りの組み立て・届け先の無い頼み など）は誰にも届かなかった（「記録だけ」＝「だれも読まない」）。
#:   所見を書く所（`StepRecord(step=…, warnings=…)`・`song_flow._rec(song, …, warnings=…)`）をコードで洗い出した表。
#:   **所見を残す記録を足したらここにも足す**——`tests/song_service/test_findings_route.py` が書く所を拾い、表に無ければ落とす。
#:   表に無い名前は `TO_SUMMARY` として扱う（迷ったら届ける）。`*_gate`・`*_fix`・`*_machine` は要約にそれぞれの見出しがあるので載せない。
#:   - 値が `*_gate` … 点検役が材料として読む（旋律・詞・伴奏だけ・`song_gate.SELF_REDO`）。その点検役の記録が**後に無い**道
#:     （尺の変更の詞・`song_resize`）では記録だけなので、要約に出す
#:   - `TO_SUMMARY` … 点検役のいない工程＝記録だけ。要約の【点検役のいない工程の所見】に出す
#:   - `NOT_PASSED` … 要約にも出さない（帯で測る数字は点検役にもコーディネーターにも渡さない決まり・2026-09-29）
#:   ※最終の点検役（`final_gate`・2段目・2026-10-04 組P2b）を届け先にする記録は無い。工程の記録の所見は1段目の点検役が読み、
#:     最終の点検役は1段目の判断（`*_gate` の理由・弱い所）と区間ごとの表を読む。`final_gate` 自身は `*_gate` なので要約の【点検役の判断】に出る
TO_SUMMARY = "summary"
NOT_PASSED = "none"
FINDINGS_ROUTE: dict[str, str] = {
    # 点検役が読む（`song_lyrics`・`song_band.run_arrange_gate`・`song_arrange_cover._arrange_gate`）
    "lyrics": "lyrics_gate",                               # 尺の変更（`song_resize`）の詞は点検役の記録が続かない＝記録だけ
    "arrange": "arrange_gate",
    # 仕上げの点検役が読む（`song_finish_gate`・2026-10-03 判-21）。後に `finish_gate` の記録が続かない道（カバー・尺の変更など）は要約に出る
    "expression": "finish_gate",                           # 歌い方（`vocal_expression`）
    "vocal_artic": "finish_gate",                          # 歌の崩し（`vocal_artic`）
    "harmony": "finish_gate",                              # ハモリの計画（`song_band`）・カバーの声部と奏法（`song_arrange_cover`）
    "voices": "finish_gate",                               # 声部（`song_band.run_voices`）——推奨音域・9割（`harmony.check_voices`）
    "colour": "finish_gate",                               # 彩り（`song_band.run_colour`）——組み立ての所見
    "articulation": "finish_gate",                         # 奏法（`artic_plan`）
    "naming": "finish_gate",                               # 曲名（`song_naming`・2026-10-03。曲名と理由を仕上げの点検役が読む）
    # 点検役のいない工程（記録だけ）
    "design": TO_SUMMARY,                                  # カバーの区間の推定（`song_cover_load`）・尺の変更（`song_resize`）
    "chords": TO_SUMMARY,                                  # カバーのコード（`song_cover_load`・`song_cover_chords`）
    "form": TO_SUMMARY,                                    # 曲の形を変えた（`song_remake`）——届け先の無い頼み
    # 渡さない
    "melody_rhythm": NOT_PASSED,                           # 旋律のリズムの数字（`song_melody_step`）——帯で測る記録
    # ★編集のあとの最終の点検（`song_final_gate.judge_edit`・2026-10-04・案C）。所見の欄は使わない（判断は要約の【点検役の判断】に
    #   「最終の点検役（編集のあと）」として出る）。ここに書くのは、記録だけの所見の行に二重に出さないため
    "final_gate_edit": NOT_PASSED,
}


def form_of_record(d: dict) -> Form:
    """記録（`record_v{N}.json`）の形。**欄が無ければ song**（2026-10-03 より前の記録）。知らない綴りは不具合として止める
    （記録は機械が書くので、読めない綴りは読み替えない）。"""
    v = d.get("form") or "song"
    if v not in get_args(Form):
        raise ValueError(f"曲の記録の形『{v}』は読めない（{' / '.join(get_args(Form))}）")
    return v


@dataclass
class StepRecord:
    """**1工程ぶんの記録**。全工程がこの形で返す。

    `status` は工程自身の申告（2026-08-22 に追加）:
      ok           … 指示どおりにできた
      compromised  … **何かを妥協した**（例: 音符数に合わせて言葉を削った）
      failed       … できなかった

    「失敗が失敗として上がってこない」ために、V6 で開くまで気づけない不具合が
    4件続いた。**妥協したなら妥協したと言わせる**のがこの欄の目的。
    """
    step: str
    reason: str = ""
    status: Status = "ok"
    trouble: str = ""
    warnings: list[str] = field(default_factory=list)
    request: str = ""                                      # 編集・やり直しの指示（新規作成では空）
    #: ★**点検役（関門）が「壊れてはいないが弱い」と書いたこと**（2026-09-27）。以前は画面に出すだけで記録にも
    #:   残らず、後の工程にも届かなかった（「詞が平易すぎる」「ドラムが意図と食い違う」と的確に書いても効かなかった）。
    #:   記録に残し、`song_context.upstream_intents` がその工程を読む後の工程へ渡す。
    notes: str = ""
    #: ★**点検役の指示**（2026-10-04・組P2a）。`*_gate` の記録で、点検役がやり直す側へ渡した「どう直すか」（`Verdict.guidance`）。
    #:   以前は理由（`reason`）にも残らず、作業中に覚えるだけで、続きから作ると前回の自分の指示が消えた。
    #:   点検役の再判断（`song_gate_brief.previous_own`）と最終の点検役（`gate_history`）が読む。差し戻しの記録（`redo`）では
    #:   差し戻しの指示（`RedoFrom.guidance`）。ほかの記録・欄の無い古い記録は空
    guidance: str = ""
    usd: float = 0.0                                       # この工程にかかった費用

    def is_clean(self) -> bool:
        return self.status == "ok" and not self.warnings

    def summary(self) -> str:
        mark = {"ok": "", "compromised": "△", "failed": "×", SKIPPED: "－"}.get(self.status, "?")
        head = f"{mark}[{STEP_LABEL.get(self.step, self.step)}]"
        body = self.reason.replace("\n", " ")[:110]
        tail = f"  困りごと: {self.trouble}" if self.trouble else ""
        warn = "".join(f"\n      ★{w}" for w in self.warnings)
        return f"{head} {body}{tail}{warn}"


@dataclass
class Song:
    """**曲そのもの**。工程はこれを受け取り、担当パートを埋める。"""
    instruction: str = ""                                  # 最初の依頼文
    score: SongScore | None = None                         # 楽譜（音の実体）
    #: ★**曲の形**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §2）。song＝歌もの・acappella＝アカペラ・
    #:   instrumental＝インスト。決めるのはコーディネーター。走る工程は `song_gate.FORM_STEPS[form]`。欄の無い古い記録は song
    form: Form = "song"

    # ---- 工程の成果（型はそれぞれの工程が定義したもののまま）----
    #   None＝まだ作っていない。**その曲の形で走らない工程は、同じ型で中身が空・状態 skipped**（`skipped_output`・
    #   `song_flow.fill_skipped` が曲を作り始めるときに入れる・2026-10-03 F1）。「作ったか」は `made` で見る
    theme: Any = None
    design: Any = None
    chords: Any = None
    groove: Any = None                                     # グルーヴ（2026-09-07 新設）
    melody: Any = None
    lyric_plan: Any = None                                 # 詞の設計（hook・置き場・区間ごとの一文）2026-09-13
    lyric_chorus: Any = None                               # サビの詞（先に書いて確定）2026-09-13
    lyrics: Any = None
    naming: Any = None                                     # 曲名（詞のあと・どの形でも）2026-10-03
    expression: Any = None                                 # 歌い方（5.7）2026-09-23
    vocal_artic: Any = None                                # 歌の崩し（5.8）2026-09-23
    instrumentation: Any = None
    arrangement: Any = None                                # ArrangePlan（伴奏の計画）
    harmony: Any = None
    colour: Any = None                                     # 彩り（対旋律・リフ）2026-09-17
    voices: Any = None                                     # 声部（合いの手・追っかけ）2026-09-23
    articulation: Any = None                               # 伴奏の奏法（8.7）2026-09-23

    # ---- 工程間で持ち回る構造化データ ----
    # ★**区間がどこにあるか**を知るのは `timeline` だけ（2026-08-23）。
    # `bars_of` と `sec_of_note` は**そこからの導出**で、単独では正ではない。
    # 以前は3箇所が独立に計算しており、片方だけ触るとずれた（実測で最大4.1小節）。
    # 詳細は docs/plan/song_timeline.md。
    timeline: Any = None                                   # SectionTimeline（区間の位置の正）
    chords_of: dict = field(default_factory=dict)          # セクション → コード列
    bars_of: dict = field(default_factory=dict)            # 導出: セクション → (開始小節, 終了小節)
    sec_of_note: list = field(default_factory=list)        # 導出: 音符 i → セクション名
    key_of: dict = field(default_factory=dict)             # セクション → キー
    instruments: list = field(default_factory=list)        # [(bank, 番号), ...]
    allow_drums: bool = True

    # ---- 記録 ----
    history: list[StepRecord] = field(default_factory=list)
    #: **頼まれごとの台帳**（2026-09-25・段階2）。`song_requests.Request` の並び。
    #: 作り直しても消えず、頼まれた工程とその下流にだけ届く（`song_requests.deliver`）。
    requests: list = field(default_factory=list)
    source: dict | None = None                             # カバーなら参照曲の情報

    # ---- 版管理 ----
    # 本番の**作業台方式**（session.py）に合わせる: 作業中は固定名、確定したら `_v{N}` を切る。
    # **元の版は消さない**——「さっきのほうが良かった」に応えられなくなるため。
    version: int = 1
    parent: int | None = None                              # どの版を編集して作ったか
    origin: str = "compose"                                # compose / cover / edit

    # ---- 参照系 --------------------------------------------------------------
    def record(self, rec: StepRecord) -> StepRecord:
        """**追記のみ**（消さない）。同じ工程を2度やっても両方残る＝経緯が追える。"""
        self.history.append(rec)
        return rec

    def last(self, step: str) -> StepRecord | None:
        for r in reversed(self.history):
            if r.step == step:
                return r
        return None

    def reason_of(self, step: str) -> str:
        r = self.last(step)
        return r.reason if r else ""

    def troubles(self) -> list[StepRecord]:
        """**妥協・失敗した工程**（コーディネーターが判断するための材料）。"""
        return [r for r in self.history if r.status != "ok" or r.warnings]

    def cost(self) -> float:
        return sum(r.usd for r in self.history)

    def title(self) -> str:
        """曲名。楽譜 → **曲名の工程** → **テーマの仮の曲名** → 「無題」の順に引く（工程の側は `step_title`）。

        ★**最後の曲名は工程『曲名』（`naming`）が決める**（2026-10-03・本人了承）。テーマは仮の曲名（`working_title`）を持ち、
          詞の設計・サビの詞・詞はそれを受けて書く。曲名の工程はどの形でも走る（詞のあと・インストでは旋律のあと）。
          それまで（組F2）はテーマの `title` だけが持ち 楽譜 → テーマ、判-11 では 楽譜 → 詞の設計 → （古い記録の）詞 → テーマ、
          その前は 楽譜 → 詞 → テーマ。古い記録は比較用の変換（`convert_record._title_to_theme`・`_naming_step`）がいまの形へ写す。
          楽譜の曲名はカバー（テーマ・曲名の工程が無い）と、組み立てが曲名の工程の値を写したもの（`song_place.make_score`）。"""
        if self.score is not None and self.score.title:
            return self.score.title
        return self.step_title() or "無題"

    def step_title(self) -> str:
        """**工程が決めた曲名**——曲名の工程（`naming.title`）→ テーマの仮の曲名（`theme.working_title`）。無ければ空。

        楽譜を見ない（組み立てが楽譜へ写す元・`song_place.make_score`／編成の見出し `song_band.run_instrumentation`）。
        曲名の工程が走らなかった・まだの曲（作っている途中・空の出力 skipped）は仮の曲名。
        工程の出力は型（属性）でも辞書（`Song.load` が読んだまま）でも読む。"""
        for src, field_name in ((self.naming, "title"), (self.theme, "working_title")):
            if not made(src):
                continue
            t = src.get(field_name) if isinstance(src, dict) else getattr(src, field_name, None)
            if isinstance(t, str) and t.strip():
                return t.strip()
        return ""

    # ---- 記録の書き出し ------------------------------------------------------
    def build_log(self) -> str:
        """**どう作ったか**を人にも LLM にも読める形で。編集時にそのまま渡せる。"""
        out = [f"■『{self.title()}』の作り方", f"  依頼: {self.instruction}"]
        for i, r in enumerate(self.history, 1):
            out.append(f"  {i}. {r.summary()}")
            if r.request:
                out.append(f"      （指示）{r.request}")
        if bad := self.troubles():
            out.append(f"  ※**うまくいかなかった工程**: "
                       + ", ".join(STEP_LABEL.get(r.step, r.step) for r in bad))
        return "\n".join(out)

    def summary(self) -> str:
        """**コーディネーターへ出す、曲づくりの中で起きたこと**（2026-09-27・目録の関門-2）。

        ★`tools/creation.py` はこれを呼んでいたのに**実装が無く**、`hasattr` で空文字になっていた——
          点検役（関門）の判断も、機械の検査で直させたことも、機械がした変更も、コーディネーターには届かなかった。
          点検役が `stop`（直せないので所見つきで進む）を選んでも、人へ報告する道が無かった。
        出すもの: ①点検役の判断（通した／やり直した／所見つきで進んだ・理由・気になったところ）
        ②機械の検査で差分を直させた結果（`*_fix`）③機械がどうしても直したこと（`*_machine`）
        ④点検役のいない工程の所見（記録だけの所見・2026-10-03・判-22。届け先は `FINDINGS_ROUTE`）
        ⑤妥協・失敗と申告した工程 ⑥「前回のまま」にした工程。何も無ければ空文字。
        """
        def label(step: str) -> str:
            base = step.rsplit("_", 1)[0] if step.endswith(("_gate", "_fix", "_machine")) else step
            return {"lyric_plan": "詞の設計", "lyric_chorus": "サビの詞", "naming": "曲名", "expression": "歌い方",
                    "vocal_artic": "歌の崩し", "voices": "声部", "colour": "彩り", "articulation": "奏法",
                    "arrangement": "伴奏", "form": "曲の形", "finish": "仕上げ",
                    "final": "最終の点検役",                      # 2段目・曲全体（2026-10-04 組P2b）
                    "final_gate_edit": "最終の点検役（編集のあと）"}.get(   # 編集のあと（2026-10-04・案C）
                base, STEP_LABEL.get(base, base))

        mark = {"accept": "通した", "redo": "やり直しにした", "stop": "★直せないので所見つきで進んだ"}
        # 最終の点検役には stop が無い（止めるときは曲づくりごと止まる）。redo は「NG で差し戻した」
        final_mark = {"accept": "通した", "redo": "NG で差し戻した", "stop": "★NG のまま差し戻せなかった"}
        # 編集のあとの最終の点検は差し戻さない（所見を返すだけ・NG でも書き出す・2026-10-04・案C）
        edit_mark = {"accept": "通した", "redo": "★NG（直す案は提案として返した・書き出しは続けた）",
                     "stop": "★NG（直す工程を選べなかった・書き出しは続けた）"}
        gates, fixes, machine, bad, kept = [], [], [], [], []
        for r in self.history:
            s = r.step
            if s.endswith("_gate") or s == "final_gate_edit":
                m = final_mark if s == "final_gate" else edit_mark if s == "final_gate_edit" else mark
                line = f"  ・{label(s)}: {m.get(r.request, r.request or '?')} — {r.reason[:200]}"
                if r.warnings:
                    line += "\n      所見: " + " / ".join(w[:80] for w in r.warnings[:5]) \
                            + (f" ほか{len(r.warnings) - 5}件" if len(r.warnings) > 5 else "")
                if r.notes:
                    line += f"\n      ▲気になったところ: {r.notes[:200]}"
                gates.append(line)
            elif s.endswith("_fix"):
                fixes.append(f"  ・{label(s)}: {r.reason[:200]}"
                             + (f"\n      残った所見: " + " / ".join(w[:80] for w in r.warnings[:5])
                                if r.warnings else ""))
            elif s.endswith("_machine"):
                machine.append(f"  ・{label(s)}: {r.reason[:600]}")
            elif r.request == "keep":
                kept.append(f"  ・{label(s)}: {r.reason[:120]}")
            elif r.status != "ok":
                bad.append(f"  ・{label(s)}（{r.status}）: {(r.trouble or r.reason)[:200]}")
        out = []
        for head, rows in (("【点検役の判断】", gates),
                           ("【機械の検査で所見を返し、差分で直させた】", fixes),
                           ("【機械がどうしても直したこと（理由つき）】", machine),
                           ("【点検役のいない工程の所見】（記録に残るだけで、点検役は読んでいない）", self._record_only_rows(label)),
                           ("【妥協・失敗と申告した工程】", bad),
                           ("【前回のままにした工程】", kept)):
            if rows:
                out.append(head + "\n" + "\n".join(rows))
        if not out:
            return ""
        stops = sum(1 for r in self.history
                    if r.step.endswith("_gate") and r.step != "final_gate" and r.request == "stop")
        tail = ("\n★**所見つきで進んだ関門が{}件**ある——直せなかったことを最終報告でマスターに伝える".format(stops)
                if stops else "")
        return "■曲づくりの中で起きたこと\n" + "\n".join(out) + tail

    def _record_only_rows(self, label) -> list[str]:
        """**記録だけの所見**の行（`summary` の④・2026-10-03・判-22）。届け先は `FINDINGS_ROUTE`。

        工程ごとに**最後の記録だけ**を見る（作り直した前の所見は古い）。点検役が読む工程は、その点検役の記録が後に続けば出さない。
        差分の直しの経過（`*_fix` の reason と同じ文）を所見に添えている工程があるので、それは【差分で直させた】に任せて外す。
        """
        fixed = {r.reason for r in self.history if r.step.endswith("_fix")}
        last: dict[str, int] = {}
        for i, r in enumerate(self.history):
            if not r.step.endswith(("_gate", "_gate_edit", "_fix", "_machine", "_discarded")):
                last[r.step] = i
        rows: list[str] = []
        for step, i in sorted(last.items(), key=lambda x: x[1]):
            route = FINDINGS_ROUTE.get(step, TO_SUMMARY)
            if route == NOT_PASSED:
                continue
            if route != TO_SUMMARY and any(r.step == route for r in self.history[i + 1:]):
                continue                                   # 点検役が読んだ
            ws = [" ".join(w.split()) for w in self.history[i].warnings if w not in fixed]
            if not ws:
                continue
            rows.append(f"  ・{label(step)}: " + " / ".join(w[:120] for w in ws[:5])
                        + (f" ほか{len(ws) - 5}件" if len(ws) > 5 else ""))
        return rows

    def to_json(self) -> dict:
        """steps.json 相当。**工程の成果も構造化データも記録も1つに**。"""
        def dump(x):
            return x.model_dump() if hasattr(x, "model_dump") else x
        return {
            "instruction": self.instruction,
            "title": self.title(),
            "version": self.version,
            "parent": self.parent,                         # **どの版から派生したか**
            "origin": self.origin,                         # compose / cover / edit
            "form": self.form,                             # 曲の形（2026-10-03）
            "history": [vars(r) for r in self.history],
            "requests": [r.to_dict() if hasattr(r, "to_dict") else r for r in self.requests],
            # ★**並びは工程の順番**（再生・差し戻しの向きと揃える）。
            #   足したら `song_replay.LABEL_TO_STEP`/`_types()`・`song_gate.STEP_ORDER`・
            #   `song_context._STEP_KEY`・`song_remake.STEP_ROLES`（工程名は `STEP_ORDER` から引くので別名の表は無い）・検証用の写しにも足す
            #   ——`tests/song_service/test_step_registry.py`（`STEP_ROLES` は `test_step_roles.py`）が落として教える。
            "steps": {k: dump(getattr(self, k)) for k in
                      ("theme", "design", "chords", "groove", "melody",
                       "lyric_plan", "lyric_chorus", "lyrics", "naming",
                       "expression", "vocal_artic",
                       "instrumentation", "arrangement", "harmony",
                       "voices", "colour", "articulation")},
            "context": {"chords_of": self.chords_of, "bars_of": self.bars_of,
                        "sec_of_note": self.sec_of_note, "key_of": self.key_of,
                        "instruments": self.instruments, "allow_drums": self.allow_drums,
                        # **正はこれ**。bars_of / sec_of_note は互換のための写し
                        "timeline": self.timeline.to_json() if self.timeline else None},
            "source": self.source,
        }

    def save(self, path: Path) -> None:
        """`<曲>.json`（楽譜）と `<曲>.steps.json`（作り方）を書く。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.score is not None:
            path.write_text(self.score.model_dump_json(indent=2), encoding="utf-8")
        path.with_suffix(".steps.json").write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")

    # ※版管理の入口（`versions` / `path_of` / `save_as_new_version` / `load_version`）は、本番の版は作業台方式
    #   （session.py）が持っていて誰も呼んでいなかったので、2026-09-28 に消した。

    # ---- 曲フォルダの版ごとの記録（2026-09-25・段階3）--------------------------
    # `_wip/<曲名>` は**曲名で引く1本**なので、同じ曲名の別の曲が取り合い、版も区別できなかった。
    # 曲フォルダに `record.json`（作業台）を置き、`session._VERSIONED` に載せて**楽譜と一緒に版を切る**。
    # 楽譜も中に持つ（1ファイルで完結させる——版の中で記録と楽譜がずれないように）。

    def save_record(self, path: Path) -> None:
        d = self.to_json()
        d["score"] = self.score.model_dump() if self.score is not None else None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load_record(cls, path: Path) -> "Song":
        d = json.loads(path.read_text(encoding="utf-8"))
        song = cls()
        if d.get("score"):
            song.score = SongScore.model_validate(d["score"])
        return song._fill(d)

    @classmethod
    def load(cls, path: Path) -> "Song":
        """保存した曲を読み戻す（**編集の入口**）。

        工程の成果は dict のまま持つ（型に戻すのは、その工程を再実行するときだけでよい）。
        """
        song = cls()
        if path.exists():
            song.score = SongScore.model_validate(json.loads(path.read_text(encoding="utf-8")))
        sp = path.with_suffix(".steps.json")
        if not sp.exists():
            return song
        return song._fill(json.loads(sp.read_text(encoding="utf-8")))

    def _fill(self, d: dict) -> "Song":
        """記録の dict から工程の成果・台帳・文脈を読み込む（`load` / `load_record` の共通部分）。"""
        song = self
        song.instruction = d.get("instruction", "")
        song.version = int(d.get("version", 1))
        song.parent = d.get("parent")
        song.origin = d.get("origin", "compose")
        song.form = form_of_record(d)
        song.history = [StepRecord(**r) for r in d.get("history", [])]
        from song_requests import Request
        song.requests = [Request.from_dict(x) for x in (d.get("requests") or [])]
        song.source = d.get("source")
        for k, v in (d.get("steps") or {}).items():
            setattr(song, k, v)
        ctx = d.get("context") or {}
        song.chords_of = ctx.get("chords_of") or {}
        if ctx.get("timeline"):                            # **正**があればそこから導出し直す
            from song_timeline import SectionTimeline
            song.timeline = SectionTimeline.from_json(ctx["timeline"])
            song.bars_of = song.timeline.bars_of()
        else:                                              # 時間軸を持たない曲（カバーは timeline を作らない）
            song.bars_of = {k: tuple(v) for k, v in (ctx.get("bars_of") or {}).items()}
        song.sec_of_note = ctx.get("sec_of_note") or []
        song.key_of = ctx.get("key_of") or {}
        song.instruments = [tuple(x) for x in (ctx.get("instruments") or [])]
        song.allow_drums = bool(ctx.get("allow_drums", True))
        return song
