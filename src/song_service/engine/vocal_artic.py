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

"""**歌の奏法を音ごとに決める工程**（2026-09-23 新設・工程 5.8）。

本人:
> 「**実際の記憶は、多分、伸ばしの途中でしゃくるとか、この1音だけ変化させるとかあると思う。
>   切り出してくれる**」

## なぜ歌い方（5.7）から切り出したか

一度は歌い方の工程に「フレーズの終わりを落とすか」として持たせた。**足りなかった。**

**フレーズ単位では、フレーズの中に届かない。** 実際の歌にあるのは:

  ・**伸ばしている音の途中**でしゃくり上げる（こぶし・R&B のメリスマ）
  ・**この1音だけ**を落とす（言い切らない語尾・投げ出し）
  ・同じフレーズでも**2音目は素で、5音目だけ**動かす

★**伴奏で切り出したのと同じ理由**——「初めの1音だけ」が上の単位では書けない。
  ただし**伴奏とは事情が違った**: 伴奏は**1小節のパターンを繰り返す**ので構造的に届かなかったが、
  歌は音が独立して並んでいるので**届く**。それでも**フレーズ単位の判断では拾えない**ので、
  **音を1つずつ見せる工程**にする。

## 歌は伴奏より浅い

楽器のフォールは7半音（ブラス）だが、**それをそのまま歌に当てると崩れて聞こえる**。
ここでは **5半音**を上限にする（**出力の型で差し戻す**。2026-09-27 までは黙って丸めていた・目録 表情-5）。

## 次の音へ滑る系は使わない

V6 は放っておいても **100ms で次の音へ滑る**（2026-09-23 実測）。
そこへ `gliss` / `portamento` を重ねると**二重に滑り、戻しの位置で音程が跳ねる**
（実測で +2.2 → +0.7 のつまずきが出た）。**渡りは `pitch_transition_*` の仕事**。
"""
from __future__ import annotations

import sys
from pathlib import Path

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_ai import Agent

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import articulation as artic                               # noqa: E402
from song_llm import pick_model                            # noqa: E402
from song_llm import run_sync as _llm_run                  # noqa: E402
from song_model import StepRecord, PatchStatus, StepStatus              # noqa: E402
from song_checks import JsonBool, JsonFloat, JsonInt  # noqa: E402（LLM が書く数と真偽の欄の型・2026-10-02 判断5）
from song_checks import edge_ws_problem, strip_ascii_ws     # noqa: E402（半角の名前の前後の空白・§0-6・2026-10-03）


def _strip_name(v):
    """前後の半角の空白とタブだけ落とす（仕様書 §0-6・伴奏の奏法名 `artic_plan._strip` と同じ扱い）。
    前後の全角の空白・改行などは型で差し戻す。2026-10-03 まで歌の崩しの名前だけ前後の空白を一切取らず、半角の空白も差し戻していた"""
    if isinstance(v, str):
        if why := edge_ws_problem(v):
            raise ValueError(f"奏法の名前{why}")
        return strip_ascii_ws(v)
    return v
from song_prompts import prompt                            # noqa: E402
from song_schema import SongNote                           # noqa: E402
from song_tuning import phrases_by_gap                     # noqa: E402
from song_usage import record as _rec_usage                # noqa: E402

Q = 480

#: 歌で使える奏法。**次の音へ滑る系は入れない**（上の注記）。
#: `articulation.py` のカタログから、歌に合うものだけを選んである。
VOCAL_NAMES = ["scoop", "fall", "doit", "plop", "bend_up", "bend_release"]
#: ★**選べる名前は型で縛る**（2026-09-30 本人「選択肢は enum にしよう」・判-1）。以前は形式の検査（`apply_plan` の
#:   「歌では使わない奏法」）が見て差分で直させていた。いまは型が LLM へ差し戻し、直らなければ歌の崩しの工程で止まる
#:   （`song_llm.run_sync`）。同じ観点を2か所で見ないよう、検査のほうは外した
VocalName = Literal[tuple(VOCAL_NAMES)]                     # type: ignore[valid-type]
VocalNameOrKeep = Literal[tuple(["", *VOCAL_NAMES])]        # type: ignore[valid-type]  # 差分: 空＝変えない

#: 歌の深さの**目安**（半音）。**伴奏より浅い**——楽器の7半音を歌に当てると崩れて聞こえる。依頼文の説明と標準の深さの上限に使う。
#: ★2026-10-03（`docs/plan/song_viewpoints_review.md` §2 d1）: 以前はこれを型の上限にしていた（5 を超えると差し戻し）。
#:   5 の根拠は音の良し悪し（内容の判断）なので、型は描ける幅（`VOCAL_MAX_DEPTH`）まで受け、5 を超えた崩しは
#:   仕上げの点検役の材料に事実で出す（`deep_bends`・`song_finish_gate`）
VOCAL_GUIDE_DEPTH = 5.0
#: 歌の深さの**上限**（半音）＝ V6 の歌のベンド幅（`score_to_vpr.VOCAL_BEND_RANGE`＝12）。超えると描けない（ベンドが端で潰れる）
VOCAL_MAX_DEPTH = 12.0

#: **歌の標準の深さ**（半音）。カタログの基準の形（`articulation.base_depth`）を歌の上限で抑えたもの:
#: scoop 2・plop 4・fall 5・doit 5・bend_up 2・bend_release 2（仕様書 §15）。依頼文の表の頭に出す（`standard_text`）。
#: ★2026-10-02（洗い出し H1）: 以前は空の深さをカタログの基準の形のまま V6 へ描き（`score_to_vpr` → `articulation.shape_of`）、
#:   フォールが歌の上限 5 半音を超える 7 半音で鳴った。10-02〜10-04 は組み立て（`apply_plan`）が空の深さをこの値で埋めた。
#: ★2026-10-04（本人「深さを書かないのは良くない。書き忘れたのか標準にしたのかが曖昧になる。全部書いてもらったほうがいい」）:
#:   深さは LLM が必ず書く（`VocalArticNote.depth_st`）。この値は**選ぶための目安の1つ**として依頼文に出すだけで、組み立ては空を埋めない。
#:   古い記録の空の深さは、変換（`dev_tools/song_legacy/convert_record.py`）がこの値を書き込む（10-02〜10-04 の鳴り方と同じ）
VOCAL_STANDARD: dict[str, float] = {n: min(artic.base_depth(n) or 0.0, VOCAL_GUIDE_DEPTH) for n in VOCAL_NAMES}


#: **崩しごとの形と標準の使いどころ**（型の説明に出す短い形。詳しい使いどころ・深さの目安・使いすぎの目安は指示書 `song_vocal_artic.md` の表）。
#: ★2026-10-04（本人の実機確認「31秒あたりの音程が外れている」・`20261004_132741_水たまりの青空`）: ポップスの伸ばしの真ん中に
#:   `bend_release`（1半音）が掛かり、外れて聞こえた。選択肢と型の範囲は変えず、標準の使いどころと意図を教える（本人判断）。
#:   `bend_*` は伸ばしの途中で高さが変わる（`articulation._BASE`: 15%→35% で上げ、bend_release は 65% で戻る）ので、
#:   歌では節回しの様式（R&B・ブルース・ソウルなど）で浅く使い、ポップス・バラードの伸ばしではふつう使わない
#: ★2026-10-04（本人の実機確認『水たまりのプリズム』・約78秒で10か所）: 「崩しはここぞという1ポイントで入れるもの。A メロ・B メロは
#:   特殊な意図がある場合以外は使わないのが自然」。標準を「ポイントに絞る・曲全体で3〜5か所」に直した（`count_text`・指示書の表）。
#:   跳躍のあとの伸ばしに `fall` を重ねて不自然に聞こえた（22秒の「だ」・9半音跳んで2拍・fall 2）ので、fall の使いどころに足した
VOCAL_USES: dict[str, str] = {
    "scoop": "頭を下から入る（語の頭・感情の入りに浅く）",
    "fall": "終わりの2割で落とす（フレーズ末の抜き。跳躍のあとの伸ばしには掛けない）",
    "doit": "終わりで上へ跳ね上げる（明るい曲の語尾に軽く少し）",
    "plop": "頭を上からかぶせて入る（まれに）",
    "bend_up": "途中で上げて上がったまま終わる（R&B・ブルースなど節回しの様式で浅く。ポップス・バラードの伸ばしではふつう使わない）",
    "bend_release": "途中で上げて戻す（同上。伸ばしの真ん中で高さが変わり、外れて聞こえることがある）",
}


def standard_text() -> str:
    """依頼文に出す歌の標準の深さ（`VOCAL_STANDARD`）。深さは必ず書く欄で、これは目安（2026-10-04）。"""
    return ("標準の深さ（半音・深さは必ず書く。目安の1つ）: " + " ".join(f"{n}={VOCAL_STANDARD[n]:g}" for n in VOCAL_NAMES))


#: 歌の崩しの**標準の数**（曲全体・ポップス／バラード）。ポイントに絞る（2026-10-04 本人判断）。指示書の表と `count_text` に同じ数
VOCAL_COUNT_SONG = (3, 5)


def count_text() -> str:
    """依頼文に出す歌の崩しの標準の数と置き場所（指示書の「ここぞという所に絞る」の表と同じ数）。"""
    lo, hi = VOCAL_COUNT_SONG
    return (f"標準の数（外すなら reason に何を・なぜ）: 曲全体で{lo}〜{hi}か所。サビの頭・盛り上がりの頂点・イントロ・アウトロ・"
            "間奏の入りなどポイントに絞る。A メロ・B メロは0（意図があるときだけ1か所まで）")

#: 表に出す音の上限。多すぎると読み飛ばされる。
MAX_ROWS = 400

# ★**声を楽器で代用しない**（2026-09-23）。最初はサックス（GM 65）で引いていたが、
#   **サックスの持ち物に無いものが、声でもできないことになってしまう**
#   ——`plop`（上から落として入る）は人の歌にあるのに、サックスの持ち物には無い。
#   **人の声はこの表の楽器ではない**ので、縛るのは上の `VOCAL_NAMES` にして、
#   カタログからは**形だけ**引く（`articulation.shape_of`）。


class VocalArticNote(BaseModel):
    """1つの音に掛ける奏法。**位置は表と同じ書き方**で。"""
    at: str = Field(description="鳴り始める位置。**表と同じ書き方**（`12:1.50`。拍は小数2桁）。**拍は 0 から数えます**")
    name: VocalName = Field(description="奏法（**この中から1つ**）: " + " / ".join(f"{n}＝{VOCAL_USES[n]}" for n in VOCAL_NAMES))
    #: ★**上限は型で差し戻す**（2026-09-27・目録 表情-5）。それまでは 5半音を超えた値を**黙って 5 に丸め**、
    #:   LLM は知らなかった。いまは型の検証で LLM に返り、書き直してもらう。
    #: ★**0 より大きい数だけ**（2026-10-02・洗い出し H1 と判断3「1通りに書く」・伴奏の `ArticNote.depth_st` と同じ `gt=0`）。
    #:   以前は −5〜5 を受けて負を大きさとして読み（`2` と `-2` が同じ＝2通り）、0 は「カタログの基準の形」で描いて
    #:   フォールが 7 半音で鳴った（歌の上限を超える）。動かさないなら崩しを書かない。記録に負・0 の深さは0件
    #: ★**必ず書く**（2026-10-04・本人「深さを書かないのは良くない。書き忘れたのか標準にしたのかが曖昧になる」）。以前は空＝表の頭の
    #:   標準（`VOCAL_STANDARD`）で、書き忘れと「標準を選んだ」が区別できなかった。書き忘れは型が差し戻す
    depth_st: JsonFloat = Field(gt=0, le=VOCAL_MAX_DEPTH, description=(
        f"何半音動かすか。**必ず書く**（空・省略は書き直しを頼まれます）。**0 より大きく {VOCAL_MAX_DEPTH:.0f} 半音まで**。"
        "選ぶ目安は奏法ごとに指示書の表（scoop 1〜2・fall 2〜5・doit 2〜3・plop 2〜4・bend_up 1〜2・bend_release 0.5〜1）と、"
        f"表の頭の標準の深さ。**歌は {VOCAL_GUIDE_DEPTH:.0f} 半音までが目安**（深く崩すと崩れて聞こえやすい）。"
        "目安から外すなら、何を外したか・なぜかを reason に書く。\n"
        "**符号は要りません**（落とす系は下、上げる系は上、と奏法が決めます。0 や負の数は書けません）"))
    reason: str = Field(description="なぜこの音でそうするのか（歌詞・フレーズの中の位置から）。"
                                    "標準の使いどころ・深さから外すなら、何を外したか・なぜか")

    @field_validator("name", mode="before")
    @classmethod
    def _name(cls, v):
        return _strip_name(v)


class VocalArticPlan(StepStatus):
    reason: str = Field(description="この曲で歌の崩しをどう使ったか（**使わない判断ならその理由**）")
    # ★必ず書く（2026-10-04・既定値をやめる）。崩さないなら空の並び `[]` を書く（書き忘れと「崩さない」を分ける）
    notes: list[VocalArticNote] = Field(description=(
        "崩す音の並び。**必ず書く**——崩さない曲は空の並び `[]` を書き、理由を reason に。"
        f"標準は曲全体で{VOCAL_COUNT_SONG[0]}〜{VOCAL_COUNT_SONG[1]}か所（サビの頭・頂点・イントロ・アウトロ・間奏の入りなどポイントに絞り、"
        "A メロ・B メロは0）。外すなら reason に何を・なぜ"))


_vocal_articulator: Agent[None, VocalArticPlan] = Agent(
    pick_model("HARMONY", "flash"), output_type=VocalArticPlan,
    system_prompt=prompt("song_vocal_artic"),
)


class VocalArticFix(BaseModel):
    """歌の崩しの差分1項目（2026-09-27・目録の表情-6）。"""
    # ※`why` は読み手なし（記録にも依頼文にも出ない）。直す前に考えを書かせる欄として残す（2026-09-30・出-13）
    no: JsonInt = Field(ge=1, description="直す項目の番号（所見の『◯番』。1から数える）")
    drop: JsonBool = Field(default=False, description="true ならその項目を消す（崩さないことにする）。`at`・`name`・`depth_st` と一緒に書かない")
    at: str = Field(default="", description="新しい位置（表と同じ書き方 `12:1.50`・拍は 0 からの小数2桁）。変えないなら空")
    name: VocalNameOrKeep = Field(default="", description=f"新しい奏法（{' / '.join(VOCAL_NAMES)} のどれか）。変えないなら空。"
                                                          "**替えるときは `depth_st` も必ず書く**（深さの目安は奏法ごとに違う）")
    # ★**深さも差分で直せる**（2026-10-04・深さを必須にしたとき）。以前は差分に欄が無く、直しで深さは変えられなかった。
    #   空＝いまのまま（差分の決まり）。奏法を替えるときは、前の奏法の深さを黙って引き継がないよう必ず書かせる（`_drop_alone`）
    depth_st: JsonFloat | None = Field(default=None, gt=0, le=VOCAL_MAX_DEPTH, description=(
        f"新しい深さ（半音・0 より大きく {VOCAL_MAX_DEPTH:.0f} まで。目安は指示書の表）。変えないなら空。奏法（name）を替えるときは必ず書く"))
    why: str = Field(default="", description="どの所見にどう応えたか（一言）")

    @field_validator("name", mode="before")
    @classmethod
    def _name(cls, v):
        return _strip_name(v)

    # ★**消すと直す欄を一緒に書いた項目は型が差し戻す**（2026-10-03・記B・仕様書 §15）。以前は drop が黙って勝ち、
    #   書いた位置・名前は読まれなかった（LLM は直したつもり）。歌い方・伴奏の差分と同じ扱い
    @model_validator(mode="after")
    def _drop_alone(self):
        if self.drop and (both := [k for k in ("at", "name", "depth_st") if getattr(self, k) not in ("", None)]):
            raise ValueError(f"{self.no}番: drop と {'・'.join(both)} を一緒に書いた——やめるなら drop: true だけ、"
                             f"直すなら {'・'.join(both)} だけ")
        if self.name and self.depth_st is None:
            raise ValueError(f"{self.no}番: 奏法を {self.name} に替えるのに深さ（depth_st）が無い——奏法を替えるときは深さも書く"
                             "（目安は指示書の表の奏法ごとの深さ）")
        return self


class VocalArticPatch(PatchStatus):
    # ※`reason` は読み手なし（記録にも依頼文にも出ない）。直す前に考えを書かせる欄として残す（2026-09-30・出-13）。
    #   `trouble` は直しの記録に入る（`VocalArticStep.apply`）
    reason: str = Field(description="何をどう直したか（全体で一言）")
    fixes: list[VocalArticFix] = Field(default_factory=list, description="直す項目だけ。直さない項目は書かない")


from song_fix import PatchNumbers, guard_patch           # noqa: E402（差分の検証は工程の形の1つ・2026-10-02）

_vocal_artic_fixer: Agent[None, VocalArticPatch] = guard_patch(Agent(
    pick_model("HARMONY", "flash"), output_type=VocalArticPatch,
    system_prompt=prompt("song_vocal_artic_fix"),
))


from song_fix import AssemblyError, StepFixer               # noqa: E402


class VocalArticStep(PatchNumbers, StepFixer):
    """歌の崩し: 位置が読めるか・拍が1小節に収まるか（記法＝`written`・`song_checks.position_notation`・2026-09-30）・その位置に歌の音があるか・
    1つの音に1つか（形式の検査）。2026-09-27・目録の表情-6。歌で使える奏法かは型（`VocalName`）が見る（2026-09-30）。

    以前は所見を記録に積むだけで、書いた崩しは黙って落ちた。いまは所見を返し、差分（`VocalArticPatch`＝番号で
    位置／奏法を直すか消す）で直させる。

    ★**検査は組み立て（`apply_plan`）を写しに当てて取る**（音は変えない）。検査と組み立てで同じ判定を2本書くと、
      片方だけ直って「検査は通ったのに鳴らない」ができる。
    ★直らずに残ったら、歌の崩しの工程で曲づくりを止める（`song_fix.StepUnfixed`・2026-09-30）。検査＝組み立ての判定なので、
      止まらずに組み立てまで来た崩しは全部掛かる（以前は残った所見の崩しを鳴らさずに落とし、記録して進んでいた）。
    """
    key = "vocal_artic"
    label = "歌の崩し"
    agent = _vocal_artic_fixer
    return_hint = "**直す項目だけ**を fixes に入れてください。"

    def __init__(self, notes: list[SongNote], bar_ticks: int, entries: dict[int, str] | None = None,
                 keys: list[tuple[int, str]] | None = None):
        self.notes, self.bar_ticks, self.entries, self.keys = notes, bar_ticks, entries or {}, keys

    def written(self, p: VocalArticPlan):
        # ★拍子を知っている位置の記法（2026-09-30）。以前は拍子を知らない `POSITION` のままで、3/4 の `12:3.00` が読めて
        #   「歌の音が無い」の所見になった（なぜ無いのかが分からない）。奏法・声部と同じく、拍が1小節をはみ出す所見を出す
        pos = _position(self.bar_ticks)
        return ((f"{i}番", pos, a.at) for i, a in enumerate(p.notes, 1))

    def check(self, p: VocalArticPlan) -> list[str]:
        # ★二重のしゃくり（`しゃくる` の入りに `scoop`）は 2026-10-03 に形式から外した（§2 d2）。どちらも描けて、二重に掛けるのが
        #   狙いのこともある＝音の判断。仕上げの点検役の材料に事実で出す（`double_scoop_notes`・`song_finish_gate`）
        _, warns = apply_plan([n.model_copy() for n in self.notes], p, self.bar_ticks)
        return warns

    def show(self, p: VocalArticPlan) -> str:
        rows = "\n".join(f"  {i}番: at={a.at} {a.name} 深さ={a.depth_st:g}" for i, a in enumerate(p.notes, 1))
        return (f"{TABLE_HEAD}\n{standard_text()}\n" + table(self.notes, self.bar_ticks, entries=self.entries, keys=self.keys)
                + "\n\n【いまの歌の崩し】\n" + rows)

    def patch_count(self, p: VocalArticPlan) -> int:
        return len(p.notes)

    def patch_items(self, patch: VocalArticPatch) -> list[tuple[int, bool]]:
        return [(f.no, not (f.drop or f.at or f.name or f.depth_st is not None)) for f in patch.fixes or []]

    def apply(self, p: VocalArticPlan, patch: VocalArticPatch) -> tuple[VocalArticPlan, str]:
        new = p.model_copy(deep=True)
        done, drop = [], set()
        # ★範囲外・重なった番号・中身の無い直しは差分の検証（`song_fix.guard_patch`）が差し戻すので来ない（2026-10-02・洗い出し）。
        #   型を通らずに来たら当てずに、当てなかったことを直しの記録に書く（以前は黙って捨てた）
        if why := self.patch_problem(p, patch):
            done.append(f"当てなかった: {why}")
        seen: set[int] = set()
        for f in getattr(patch, "fixes", None) or []:
            if not 1 <= f.no <= len(new.notes) or f.no in seen or not (f.drop or f.at or f.name or f.depth_st is not None):
                continue
            seen.add(f.no)
            if f.drop:
                drop.add(f.no)
            else:
                a = new.notes[f.no - 1]
                new.notes[f.no - 1] = a.model_copy(update={k: v for k, v in (("at", f.at), ("name", f.name),
                                                                             ("depth_st", f.depth_st)) if v not in ("", None)})
            done.append(f"{f.no}番" + ("を消した" if f.drop else ""))
        new.notes = [a for i, a in enumerate(new.notes, 1) if i not in drop]
        note = "・".join(done) if done else "直しなし"
        if getattr(patch, "trouble", ""):                  # ほかの工程の差分と同じく、直さなかった理由を直しの記録へ（2026-09-30・出-13）
            note += f"／直さなかった理由: {patch.trouble}"
        return new, note


# ★**位置は表の書き方どうしで照らす**（2026-09-29・伴奏の奏法と同じ部品）。以前は書かれた拍を tick へ直して
#   完全一致で照らしていたので、`~` で3等分した音（160tick＝表では `0.33`）が 158tick になり、表に載っている音を
#   書き写しても「歌の音が無い」になった。表（`table`）と照合（`apply_plan`）で同じ `tick_key` を使う
from artic_plan import at_key, near_note_text, pitch_name, tick_key   # noqa: E402（位置の書き方・音高の綴り・近い音の示し方は奏法と同じ1つ）
from song_checks import position_notation                   # noqa: E402（拍子を知っている位置の記法）
from song_table import _hold_shown                           # noqa: E402（歌詞の欄の伸ばしは `_`・曲の表と同じ1つ・ず-11）


def _position(bar_ticks: int):
    """この曲の拍子を知っている位置の記法（`song_checks.position_notation`・2026-09-30）。拍が1小節の長さ以上なら所見。"""
    return position_notation(bar_ticks / Q)


#: 表の見出し（最初の依頼と直しの依頼で同じもの）
TABLE_HEAD = "【歌の音】（小節:拍|歌詞|音高|長さ|フレーズの中の位置（頭の音には歌い方が決めた入り方））"


def table(notes: list[SongNote], bar_ticks: int, *, entries: dict[int, str] | None = None,
          keys: list[tuple[int, str]] | None = None) -> str:
    """LLM に見せる表。**1音ずつ**（フレーズ単位では届かない所を決めるため）。

    ★**フレーズの中のどこか**（頭・中・終わり）を添える——「終わりだけ落とす」のような
      判断は位置で決まるので、**数えさせずに渡す**。
    ★**歌詞も出す**。言葉の意味で決まるものなので、これが無いとこの工程の意味が無い。伸ばしの音は LLM が書く伸ばしの印 `_`
      で出す（`song_table._hold_shown`・2026-10-01・ず-11。以前は楽譜の V6 の歌詞 `-` のままで、曲の表と印が違った）。
    ★`entries`（音の番号 → 歌い方の入り方・`vocal_expression.entries_of_heads`）を渡すと、頭の音に
      `頭（入り:しゃくる）` のように添える（2026-09-30・出-12）。以前は歌い方の入り方が見えず、`しゃくる` の頭へ `scoop` を重ねていた。
    ★`keys`（`[(開始tick, 調名)]`・`song_table.key_regions`）を渡すと、音高をその区間の調の綴り（♯／♭）で出す
      （`artic_plan.pitch_name`・2026-09-30）。以前は ♯ だけで、F の曲の B♭ が `ラ#` と出ていた。
    """
    head, tail = phrases_by_gap(notes)
    ent = entries or {}
    lines = []
    for i, n in enumerate(notes[:MAX_ROWS]):
        where = "頭" if i in head else ("終" if i in tail else "中")
        if where == "頭" and i in ent:
            where += f"（入り:{ent[i]}）"
        long_ = "★伸ばし" if n.duration_tick >= Q else ""
        lines.append(f"  {tick_key(n.start_tick, bar_ticks)}|{_hold_shown(n.lyric)}|{pitch_name(n.key, n.start_tick, keys)}"
                     f"|{n.duration_tick / Q:.2f}拍|{where}{long_}")
    if len(notes) > MAX_ROWS:
        lines.append("  …（多いので以下省略）")
    return "\n".join(lines)


#: 歌い方の入り方のうち、**下から入る**もの（`vocal_expression.ENTRIES`）。そこへ `scoop` を重ねると同じ入りに二重に掛かる
SCOOP_ENTRIES = ("しゃくる",)


def double_scoop_notes(notes: list[SongNote], entries: dict[int, str]) -> list[int]:
    """**歌い方が `しゃくる` にした頭の音に `scoop` が掛かった音**の番号（組み立てた歌の音 `artic` で数える・事実だけ）。

    どちらも「下から入る」を同じ音に掛けることになる（V6 の渡りを伸ばしたうえに、ピッチベンドで下から持ち上げる）。
    『孤独のまま』で 21・57・81小節の頭に二重に掛かった。
    ★2026-10-03（`docs/plan/song_viewpoints_review.md` §2 d2）: 以前（2026-09-30・出-12）は歌の崩しの形式の所見（`double_scoops`）で、
      「答えは1つ」として直させていた。どちらも描けて、二重に掛けるのが狙いのこともある＝音の判断なので形式から外し、
      仕上げの点検役の材料に「二重に掛かっている音が◯か所（どこ）」と出す（`song_finish_gate`）。組み立ては両方を掛ける（以前から）。
    """
    return [i for i, n in enumerate(notes)
            if (getattr(n, "artic", "") or "") == "scoop" and entries.get(i) in SCOOP_ENTRIES]


def deep_bend_notes(notes: list[SongNote]) -> list[int]:
    """**深さが目安（`VOCAL_GUIDE_DEPTH`＝5半音）を超える歌の崩し**が掛かった音の番号（組み立てた歌の音 `artic_depth` で数える・事実だけ）。
    ★2026-10-03（§2 d1）: 以前は型の上限 5 で差し戻していた。描ける幅は 12 半音なので、型は 12 まで受け、ここで材料に出す。"""
    def _depth(n) -> float:                                # 深さの無い古い曲の音は、書き出しと同じくカタログの基準の形の深さで描かれる
        d = getattr(n, "artic_depth", None)
        return float(d) if d is not None else float(artic.base_depth(n.artic) or 0.0)
    return [i for i, n in enumerate(notes)
            if (getattr(n, "artic", "") or "") and _depth(n) > VOCAL_GUIDE_DEPTH]


def apply_plan(notes: list[SongNote], plan: VocalArticPlan,
               bar_ticks: int) -> tuple[int, list[str]]:
    """書かれたものを音へ落とす。**その場で書き換え**、(付けた音数, 所見) を返す。

    ★**表に無い位置は落として所見に出す**——読めないものを黙って 0 にしない
      （曲頭に奏法が湧く）。声部で3回踏んだ型。**読めない位置**（`12:1.50` の形でない）の所見は、2026-09-29 から
      記法の型（`song_checks.position_notation`＝拍子を知っている形・2026-09-30）が出す（`VocalArticStep.written`・`plan` の所見）。
      ここは落とすだけ（拍が1小節をはみ出す位置も同じ。「歌の音が無い」を重ねない）。
    ★検査（`VocalArticStep.check`）はこれを写しに当てて所見を取る。曲づくりの組み立て（`plan`）では、所見が出たら
      不具合として止める（2026-09-30。検査が直らなければその前に工程で止まっている）。
    ★**位置は表の書き方どうしで照らす**（`tick_key`・2026-09-29）。3等分した音（`0.33`）も指せる。
    ★**1つの音に崩しは1つ**（2026-09-29・伴奏の奏法と同じ）。以前は同じ音に2つ書くと**後に書いたほうが黙って勝った**。
      いまは2つ目以降を所見にして、掛けるのは最初に書いたものだけ（直しても残ったとき）。
    """
    at_of = {tick_key(n.start_tick, bar_ticks): i for i, n in enumerate(notes)}
    pos = _position(bar_ticks)
    warns: list[str] = []
    n_set = 0
    done: dict[int, int] = {}                              # 音の番号 → その音に崩しを書いた項目の番号
    for i, a in enumerate(plan.notes, 1):
        key = at_key(a.at)
        # 読めない位置・拍が1小節をはみ出す位置は落とす（所見は記法の型＝工程の `written`。ここで「歌の音が無い」を重ねない・2026-09-30）
        if key is None or pos.problems(a.at):
            continue
        j = at_of.get(key)
        if j is None:
            # ★近い歌の音・前奏などであることを添える（2026-10-04 組F3・3‑4。以前は位置だけだった）
            near = near_note_text(a.at, [(n.start_tick, n.start_tick + n.duration_tick, f"「{_hold_shown(n.lyric)}」")
                                         for n in notes], bar_ticks, what="歌の音")
            warns.append(f"{i}番: {a.at} に歌の音が無い{near}（表に載っている位置をそのまま書き写す）")
            continue
        # ※「歌では使わない奏法」の検査は 2026-09-30 に外した: 名前は型（`VocalName`）が選べる6つに閉じている
        #   （外れたら LLM へ差し戻し、直らなければ歌の崩しの工程で止まる）。同じ観点を2か所で見ない
        name = a.name
        if j in done:
            warns.append(f"{i}番: {a.at} には {done[j]}番でもう崩しを書いた（1つの音に1つ）")
            continue
        done[j] = i
        # ※「カタログにあるか」の検査は 2026-09-28 に外した: `VOCAL_NAMES` はすべてカタログの奏法（上の検査と同じことを2回見ていた）
        # 深さは**書かれた大きさのまま**（範囲は型が 5半音で差し戻す・目録 表情-5。以前はここで黙って丸めていた）。
        # 符号は奏法が決める（指示書に書いてある）。深さは型が 0 より大きい数だけにしている（2026-10-02 まで負を `abs` で大きさとして読んだ）。
        # ★深さは型が必ず書かせる（2026-10-04）ので空は来ない。来たら不具合として止める（10-02〜10-04 は空を歌の標準
        #   `VOCAL_STANDARD` で埋めていた。古い記録の空は変換が同じ値を書き込む・`convert_record.py`）
        if a.depth_st is None:
            raise AssemblyError(f"歌の崩しの {i}番（{a.at} {name}）に深さが無い——型が必ず書かせているはずの形でない")
        depth = float(a.depth_st)
        notes[j] = notes[j].model_copy(update={"artic": name, "artic_depth": depth})
        n_set += 1
    return n_set, warns


def plan(notes: list[SongNote], brief_text: str, bar_ticks: int,
         *, verbose: bool = True, song=None,
         entries: dict[int, str] | None = None,
         keys: list[tuple[int, str]] | None = None) -> tuple[int, list[str], VocalArticPlan | None]:
    """**この工程の入口。** 音を1つずつ見せて、どこをどう崩すかを決めてもらう。

    戻り値の3つ目は**LLM が書いたもの（直させた後）**——呼び側が `Song.vocal_artic` へ
    しまうと保存され、再生と「続きから」が使える（2026-09-24）。
    所見があれば差分で直させる（`VocalArticStep`・表情-6）。`song` を渡すと `vocal_artic_fix` を記録する。
    直らなければ止まる（`StepUnfixed`）。組み立てで落ちる崩しが出たら不具合として `AssemblyError`（2026-09-30）。
    保存から戻した出力（再生）は直させない——再生で LLM を呼ばないため。
    `entries`（頭の音 → 歌い方の入り方・`vocal_expression.entries_of_heads`）を渡すと表に出し、`しゃくる` の頭への
    `scoop` を形式の所見にする（2026-09-30・出-12）。
    `keys`（区間の調・`song_table.key_regions`）を渡すと、表の音高をその調の綴りで出す（2026-09-30）。
    """
    if not notes:
        return 0, [], None
    ask = (f"{brief_text}\n\n{TABLE_HEAD}\n{standard_text()}\n{count_text()}\n"
           f"{table(notes, bar_ticks, entries=entries, keys=keys)}\n\nどの音を、どう崩しますか。")
    r = _llm_run(_vocal_articulator, ask, label="歌の崩し")
    _rec_usage(_vocal_articulator, r, "歌の崩し")
    step = VocalArticStep(notes, bar_ticks, entries, keys)
    out, _ = step.run(r.output, song, verbose=verbose)      # 直らなければ止まる（`StepUnfixed`）。保存から戻した崩しは直させない
    warns: list[str] = [step.result.summary()] if step.result.rounds else []
    # ★検査は組み立てを写しに当てて取っている（`VocalArticStep.check`）ので、直らなければ上で止まり、ここで落ちる崩しは無い。
    #   来たら不具合として止める（2026-09-30。以前は落とした崩しを記録して進んでいた）。読めない位置は記法の型が見ている
    #   （読めない位置・拍が1小節をはみ出す位置＝拍子を知っている位置の記法の所見）
    pos = _position(bar_ticks)
    unread = [f"{i}番: {why}" for i, a in enumerate(out.notes, 1) for why in pos.problems(a.at)]
    n_set, left = apply_plan(notes, out, bar_ticks)
    if unread or left:
        raise AssemblyError("歌の崩しの組み立てに、検査を通ったはずの表が来た: " + "／".join(unread + left))
    if verbose:
        print(f"  [歌の崩し] {n_set}か所（{out.reason[:70]}）")
        for line in warns[:5]:
            print(f"      {line}")
    return n_set, warns, out


def run_vocal_artic(st) -> None:
    """**曲づくりの工程 5.8（歌の崩し）**——新しい曲の流れ（`song_compose._compose_once_body`）から呼ぶ（2026-09-30 に本体から切り出した）。

    読む: 歌の組み立て（`st.notes`・`st.bar`）・`st.brief_text`。書く: `song.vocal_artic`（音符の崩しも）。
    """
    from song_flow import NOT_SWALLOW, _rec                 # noqa: E402（止める・途中で見せる・差し戻しは握らない／記録の形）
    from vocal_expression import _self_report, entries_of_heads   # noqa: E402（歌い方の入り方・記録の形）
    song, verbose, log = st.song, st.verbose, st.log
    # ---- 5.8 歌の崩し（音ごと・2026-09-23 新設）----
    #   ★5.7（歌い方）から**切り出した**。本人「**実際の記憶は、伸ばしの途中でしゃくるとか、
    #     この1音だけ変化させるとかあると思う。切り出してくれる**」——
    #     フレーズ単位では**フレーズの中**に届かない。
    #   ★歌の音は独立して並んでいるので、**1音ずつ見せて決めさせる**（伴奏の 8.7 と同じ形）。
    try:
        # ★**歌い方が決めた入り方を表に出す**（2026-09-30・出-12）。歌い方の工程は先に走り、`song.expression` にある
        _entries = entries_of_heads(st.notes, getattr(song, "expression", None))
        # 表の音高は区間の調で綴る（2026-09-30）。区間の調と時間軸は曲の記録に入る前（`song_compose._finish`）なので、組み立ての値から引く
        from song_table import key_regions                  # noqa: E402
        _n, _vw, song.vocal_artic = plan(st.notes, st.brief_text, st.bar, verbose=verbose, song=song,
                                         entries=_entries, keys=key_regions(st))
        # ※鳴らずに落ちた崩しの記録は 2026-09-30 に外した（直らなければ `plan` の中で止まるので、落ちる崩しは無い）
        log["vocal_artic"] = f"{_n}か所" + ("／" + "／".join(_vw) if _vw else "")
        # ★**LLM の reason・status・trouble を記録に入れる**（2026-09-30・出-13）。以前は「14か所」の件数だけだった
        if song.vocal_artic is not None:
            _rec(song, "vocal_artic", _self_report(song.vocal_artic, f"{_n}か所"), warnings=_vw, verbose=verbose)
        else:                                               # 歌の音が無い（歌の崩しの LLM を呼んでいない）
            song.record(StepRecord(step="vocal_artic", reason=log["vocal_artic"], warnings=_vw))
    except NOT_SWALLOW:                                         # 形式の所見が直らない・組み立ての不具合は止める（握らない・2026-09-30）
        raise
    except Exception as e:                                  # 崩しで曲全体を落とさない
        import traceback as _tb
        log["vocal_artic"] = f"（失敗: {type(e).__name__}: {e}）"
        song.record(StepRecord(step="vocal_artic", reason="", status="compromised",
                               warnings=[f"★歌の崩しを書けなかった: {type(e).__name__}: {e}",
                                         _tb.format_exc()[-600:]]))
        if verbose:
            print(f"  [5.8 歌の崩し] ★失敗: {type(e).__name__}: {e}")


def record_unrendered(st) -> None:
    """**V6 へ描かない歌の奏法を記録する**（曲づくりの最後・奏法のあと。2026-09-30 に `song_compose._compose_once_body` から切り出した）。"""
    song, verbose = st.song, st.verbose
    # ★**V6 へ描かない歌の奏法を記録する**（2026-09-27・目録の表情-8）。gliss／portamento は V6 が自前で滑るので
    #   描かない（組み立ての決まり）。以前は描かなかった所見が誰にも読まれなかった
    try:
        from score_to_vpr import vocal_artic_findings
        from song_fix import record_change
        _vf = vocal_artic_findings(st.score)
        if _vf:
            record_change(song, "vocal_artic", "vpr へ描かなかった歌の奏法: " + "／".join(_vf),
                          "V6 が自前で次の音へ滑るので二重になる／カタログに無い奏法")
    except Exception as e:                                 # noqa: BLE001（記録の失敗で曲を落とさない）
        if verbose:
            print(f"  [歌の奏法の所見] 取れなかった: {type(e).__name__}: {e}")
