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

"""**詞の工程**——詞の型・詞の記法（`LYRIC_LINE`）・形式の検査と差分の直し（`LyricsStep`）・点検役の材料・詞の工程（`run_lyrics`）。

2026-09-30 に `song_compose` から分けた。詞の工程は「詞の設計（hook・置き場）→ サビの詞 → 詞 → 検査 → 点検役」の順（`run_lyrics`）。
詞を音符へ当てる組み立て（`fit_phrase`・促音）は、組み立ての1か所（`song_place`）にある。
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent

import lyric_display                                       # 見せる歌詞↔歌う歌詞の突き合わせ
import song_checks as ck                                   # 形式の検査の部品
from song_checks import JsonBool, JsonInt  # noqa: E402（LLM が書く数と真偽の欄の型・2026-10-02 判断5）
import song_context as ctx                                 # LLM の呼び出し口（`ctx._call`・`ctx.judge`）・申し送り
from song_fix import PatchNumbers, StepFixer, cut_note, guard_patch  # 工程の形・2026-09-28／差分の検証・2026-10-03／切ったら書く・2026-10-04
from song_flow import _maybe_redo_from, _rec, _redo_options
from song_gate import MAX_REDO, Finding
from song_gate_brief import redo_ask                       # 前回の指摘と「指摘への対応」の頼み（2026-10-04・組P2a）
from song_llm import pick_model, solfa_to_midi
from song_model import StepRecord, PatchStatus, StepStatus
from song_prompts import prompt                          # プロンプトの正は prompts/*.md

# モーラの数え方と詞の譜割り（印）の読み方は `lyric_slots` の1か所（2026-09-29 に移した）
from lyric_slots import _BLANK                              # noqa: E402
import lyric_slots as _ls                                   # noqa: E402


# ---- ② 旋律に詞を当てる -------------------------------------------------------

class PhraseLyric(BaseModel):
    index: JsonInt = Field(description="フレーズ番号（1始まり・提示された順）")
    text: str = Field(description="そのフレーズの**歌う歌詞**。**ひらがなを音ごとに空白で区切る**（カタカナ語もひらがなで）"
                                  "（1つの区切り＝旋律の1音・区切りの数＝音符数ちょうど）。"
                                  "`にい`＝長音を1つの音で（区切らない）／`け~っ`＝促音・撥音を前の音に収める（詰めの標準はこれだけ）／"
                                  "`_`＝前の音節を伸ばす／`*`＝その音は歌わない。"
                                  "**タイでつないだ音は、つないだ音の数だけ印を書き、続きの音は `_`**（`か _`・ABC の歌詞行と同じ数え方）。"
                                  "例『け~っ て は し る』。漢字と英字は歌えない。"
                                  "「ゎ」「ゔ」「ゐ」「ゑ」は書かない（「わ」「ぶ」「い」「え」で書く）")
    # ★**標準外の詰めの理由**（2026-10-06 本人決定）。詰めの標準は促音・撥音を前の音に収めるときだけで、ふつうの2音節（`あ~す`）を
    #   1つの音に詰めたフレーズは、ここに理由が無いと形式の検査（`LyricsStep.check`・`lyric_slots.plain_joins`）が書き直しを求める。
    #   理由があれば通り、詞の点検役が理由ごと読んで決める（「標準を示し、外すなら理由」。項目ごとの理由は奏法の `reason` と同じ形）
    join_reason: str = Field(default="", description=(
        "**ふつうの2音節を1つの音に詰めたとき（`あ~す`＝明日 など）だけ**、どの詰めを・なぜ詰めたかを書く。"
        "詰めの標準は促音「っ」・撥音「ん」を前の音に収めるときだけで、ふつうの2音節は詰めずに言い回しを変えて1音1音節にする。"
        "詰めなければ空のまま"))
    display: str = Field(description=(
        "そのフレーズの**見せる歌詞**（字幕・投稿・歌詞表示に使う）。"
        "漢字・カタカナ・英字・記号を使ってよい。**text と同じ言葉を、読める表記で書く**"
        "（text『よ か ぜ に ま か せ て』→ display『夜風に任せて』。譜割りの印（空白・`~` `_` `*`）は入れない）。"
        "**読みが text と一致すること**——表示のほうを削って詰めない。"
        "**空にしない**（漢字の無い言葉ならかなのまま書く: text『ら ら ら』→ display『ららら』。"
        "空にしてよいのは、歌う音の無いフレーズ＝印が全部 `*` のときだけ）"))

    @model_validator(mode="after")
    def _display_written(self):
        """**見せる詞が空なら型で差し戻す**（2026-10-02 本人決定・案2）。以前は空を通し、保存する見せる歌詞（`lyric_display.full_text`・
        2番の詞の `song_resize.write_new_lyrics`）が歌うかなで埋め、所見（「表示用の歌詞が無い」）を記録していた。
        歌う音の無いフレーズ（印が全部 `*`・歌うかなが無い）だけは見せるものが無いので空を通す。"""
        if not (self.display or "").strip() and _ls.sung_kana(self.text or ""):
            raise ValueError(f"display: フレーズ{self.index}の見せる詞が空——text と同じ言葉を読める表記で書く"
                             "（漢字の無い言葉ならかなのまま: text『ら ら ら』→ display『ららら』）")
        return self


class MelodyLyrics(StepStatus):
    reason: str = Field(description="詞の運び（どこで何を言うか）")
    # ★**曲名の欄は持たない**（2026-10-03・判-11 本人了承）。仮の曲名はテーマ（`Theme.working_title`）、最後の曲名は詞のあとの工程
    #   『曲名』（`song_naming.Naming`）が持つ（2026-10-03。組F2 ではテーマの `title` だけが持った——曲の形でデータの形や置き場を変えない）。
    #   以前は詞にも曲名を書かせ、サビのある曲だけ詞の設計の曲名で黙って上書きし、サビの無い曲では詞の曲名が残っていた
    #   （指示書も「曲名も付けて」「曲名は変えない」の2文が並んでいた）。保存済みの記録の `title` は型が読み捨てる
    #   （比較用の変換 `convert_record._title_to_theme` がいまの曲名をテーマへ写す）
    phrases: list[PhraseLyric]


# ★★**詞の工程を分解した**（2026-09-13・本人「歌詞の作り方が大まかすぎる」）★★
#   旋律が「土台 → 3案 → 選抜 → 関門」なのに、詞は1回の呼び出しで全部を決めていた。
#   hook（決め台詞）・曲名・置き場が**成果物として出てこない**ので確かめられなかった。
#   プロの手順（タイトルを先に → サビ＝決め台詞 → A・B → 音数の調整）に揃える:
#   ① `LyricPlan`（詞の設計）→ ② `ChorusLyrics`（サビを先に。3案→選抜は 2026-09-30 に1案へ）→ ③ 残り → ④ 検査。
#   出典と突き合わせは `docs/dev/song_quality_premises.md` §4-C-2。

class SectionSays(BaseModel):
    kind: str = Field(description="区間名（示されたものをそのまま）")
    says: str = Field(description="その区間で**何を言うか**を一文で（心の動きのどこに当たるか）")


class LyricSymbol(BaseModel):
    """詞の設計の象徴1つ（`LyricPlan.symbols`・2026-10-05）。"""
    thing: str = Field(description="象徴にする**物・言葉**（詞に実際に出す語。例: トランク・合鍵・「おかえり」）")
    change: str = Field(description=(
        "その物・言葉の**意味が曲の中でどう変わるか**を、区間名を添えて順に"
        "（例: verse: 旅立ちの荷物 → verse2: 捨てられない過去 → chorus_last: 一緒に運んでいく覚悟）。"
        "同じ意味のまま置き続けるなら、そう書いてその理由を書く"))


#: 「裏は無い」とだけ書いた答え（理由が無い）。前後の空白と句読点を外して比べる（2026-10-05・`subtext_problems`）
_BARE_NO = {"無い", "ない", "無し", "なし", "特になし", "特に無し", "特にない", "特に無い", "裏は無い", "裏はない",
            "裏は無し", "裏はなし", "裏なし", "裏無し", "none", "n/a", "-"}
_PUNCT = " 　\t\n。、.,!！「」『』（）()…"


def says_no_subtext(text: str) -> bool:
    """`subtext` が「裏は無い」と答えたものか（理由つきでも）。事実として点検役の材料に書く（`plan_layer_facts`）。"""
    t = (text or "").strip().strip(_PUNCT)
    return t.startswith(("裏は無", "裏はな", "裏な", "裏無", "無い", "ない", "なし", "無し", "特にな", "特に無"))


def subtext_problems(subtext, symbols) -> list[str]:
    """**詞の設計の裏の層と象徴の形式**（2026-10-05）。裏の層が空でない（『無い』だけでなく理由がある）・象徴が1つ以上・
    物と意味の変化が空でない。裏がある曲か素直な曲かの良し悪しは見ない（内容＝詞の点検役が `plan_layer_facts` を読む）。"""
    out = []
    t = (subtext or "").strip()
    if not t:
        out.append("subtext（裏で本当に言っていること）が空——表で言うことと別の層を書く。無いなら『裏は無い』と書き、"
                   "なぜ表だけで足りる曲なのかを続けて書く")
    elif t.strip(_PUNCT).lower() in _BARE_NO:
        out.append(f"subtext が『{t}』だけ——裏が無いなら、なぜ表だけで足りる曲なのかを続けて書く")
    syms = list(symbols or [])
    if not syms:
        out.append("symbols（象徴になる物・言葉）が無い——詞に出す物・言葉を1つ以上選び、その意味が曲の中でどう変わるかを書く")
    for i, s in enumerate(syms, 1):
        if not (s.thing or "").strip() or not (s.change or "").strip():
            out.append(f"symbols の{i}番: 物・言葉（thing）と意味の変化（change）の両方を書く（空は書けない）")
    return out


class LyricPlan(StepStatus):
    reason: str = Field(description="なぜこの hook・この運びにしたか")
    phrase_list: list[str] = Field(description="この曲に入れたい言葉・言い回し（5〜10個）")
    # ★**裏の層と象徴**（2026-10-05・本人「選択肢がいろいろある中で、あえて直球を選ぶならよいが、検討せずになんとなく作るのは
    #   クオリティ不足」・『朝光のトランク』の詞が素直すぎた）。以前は区間ごとの says（表で言うこと）だけで、暗示・象徴・
    #   「表で言うことと裏で本当に言っていること」を考える欄が無かった。形式（空でない・『無い』だけでない・象徴が1つ以上）は型が差し戻す
    #   （`_layers`）。中身の良し悪しは詞の点検役（`plan_layer_facts`）。サビの詞・詞の依頼文の【詞の設計】に出す
    subtext: str = Field(description=(
        "**裏で本当に言っていること**（**必ず書く**）——区間ごとの says（表で言うこと）の下に流れている、言葉では言い切らない層を一〜二文で。"
        "テーマの「曲の方向」を詞でどう実現するかがここに出ます。"
        "素直な曲で裏が無いなら、空にせず『裏は無い』と書き、**なぜ表だけで足りる曲なのか**を続けて書く"))
    symbols: list[LyricSymbol] = Field(description=(
        "**象徴になる物・言葉**（**必ず1つ以上**）と、その意味が曲の中でどう変わるか（1番→2番→落ちサビ→ラスサビ のように区間名で）。"
        "同じ物・言葉が戻ってくるたびに意味が変わると、聴き手は「戻ってきた」と感じます"))

    @model_validator(mode="after")
    def _layers(self):
        """裏の層と象徴の形式（`subtext_problems`）を型で差し戻す。走らなかった出力（インストの `skipped`）は見ない。"""
        if self.status == "skipped":
            return self
        if bad := subtext_problems(self.subtext, self.symbols):
            raise ValueError("／".join(bad))
        return self
    # ※`hook_candidates` は読み手なし（機械も後の工程の本文も読まない。丸ごとの JSON でサビの詞・詞に届くだけ）。
    #   候補を並べてから1つ選ぶ、という考える手順のための欄として残す（2026-09-30・出-13。消すと選ぶ手順が崩れる）
    hook_candidates: list[str] = Field(description="決め台詞の候補（3つ）")
    hook: str = Field(description="選んだ決め台詞（サビの中心の一言）")
    # ★**曲名の欄は持たない**（2026-10-03・組F2 本人決定）。仮の曲名はテーマ（`Theme.working_title`）が決め、詞の設計は依頼文で
    #   受け取るだけ（最後の曲名は詞のあとの工程『曲名』が決める・2026-10-03）。
    #   以前（判-11〜組F2）は詞の設計が曲名を持ち、インストだけテーマが持っていた＝曲の形で置き場が変わっていた
    hook_section: str = Field(description=(
        "hook を置く区間（サビの区間名を**1つ**。【区間と行数】の名前をそのまま書く。複数並べない——"
        "サビの詞を先に書くのはこの1区間で、ほかのサビは詞の工程が書く）"))
    hook_line: JsonInt = Field(description="その区間の何行目に置くか（1始まり・推しメロのある行）")
    viewpoint: str = Field(description="視点と時制（誰が・誰に・いまの話か回想か）")
    repeat_policy: str = Field(description="サビの繰り返し方針（同じ／最後だけ変える、と理由）")
    sections: list[SectionSays] = Field(description="区間ごとに「何を言うか」一文")


class SectionFix(BaseModel):
    """詞の設計の `sections` の差分1つぶん（番号で指して区間名を直すか消す）。"""
    no: JsonInt = Field(ge=1, description="直す項目の番号（【いまの詞の設計】の sections の番号・1始まり）")
    kind: str = Field(default="", description="新しい区間名（設計の区間名をそのまま）。消すなら空のまま drop")
    drop: JsonBool = Field(default=False, description="true ならこの項目を消す（`kind` と一緒に書かない）")

    # ★**消すのは drop: true の1通り**（2026-10-03・記B・仕様書 §15）。以前は `kind` を空にしても消え（`no` だけの項目も消えた）、
    #   drop と `kind` を一緒に書くと drop が黙って勝った。併記は型が差し戻し、`kind` も drop も無い項目は差分の番号の検証
    #   （`LyricPlanStep.patch_items`→`PatchNumbers`）が「中身の無い直し」として差し戻す
    @model_validator(mode="after")
    def _drop_alone(self):
        if self.drop and (self.kind or "").strip():
            raise ValueError(f"sections の{self.no}番: drop と kind を一緒に書いた——消すなら drop: true だけ、区間名を直すなら kind だけ")
        return self


class LyricPlanPatch(PatchStatus):
    """詞の設計の直し（差分・2026-09-30・出-14②／判-5）。**直す所だけ**を書く——書かなかった所はいまのまま残る。"""
    # ※`reason` は読み手なし（記録にも依頼文にも出ない）。直す前に考えを書かせる欄として残す。`trouble` は直しの記録に入る
    reason: str = Field(description="何をどう直したか（全体で一言）")
    hook_section: str = Field(default="", description="決め台詞を置く区間を直すときだけ、その区間名を**1つ**。空ならいまのまま")
    hook_line: JsonInt = Field(default=0, ge=0, description="決め台詞を置く行を直すときだけ（その区間の何行目か・1始まり）。0 ならいまのまま")
    sections: list[SectionFix] = Field(default_factory=list, description="区間名を直す／消す項目だけ。直さない項目は書かない")


class ChorusLyrics(StepStatus):
    reason: str = Field(description="どの音節を推しメロ（最高音・伸ばし）に当てたか")
    phrases: list[PhraseLyric] = Field(description="サビの各フレーズの歌詞（番号は提示どおり）")


# 曲種ごとの**目安表**。単一の数値を指示すると、守らせるたびに別の何かが壊れた——
# 「配分を守れ」でフレーズの型が固定され、「幅を21以内に」で最高音を削られた（2026-08-20 実測）。
# 複数の曲種を並べて見せ、**曲想に合う列を選ばせる**ほうが破綻しない。
# 中間（ミドルのポップス）を必ず入れるのは、両端しか無いと極端に寄るため。
# かな → 母音（音符が詞より多いときの埋めに使う）
_VOWEL = {}
for _row, _v in zip(("あかさたなはまやらわがざだばぱゃ", "いきしちにひみりぎじぢびぴ",
                     "うくすつぬふむゆるぐずづぶぷゅ", "えけせてねへめれげぜでべぺ",
                     "おこそとのほもよろをごぞどぼぽょ"), "あいうえお"):
    for _c in _row:
        _VOWEL[_c] = _v
#: 小書きの母音 → 母音（「ヴァー」「フィー」の長音を開くときに使う・`to_sung_kana`）
_SMALL_VOWEL = dict(zip("ぁぃぅぇぉ", "あいうえお"))


_lyrics_for_melody = Agent(pick_model("LYRIC", "flash"), output_type=MelodyLyrics, system_prompt=(
    prompt("song_lyrics")
))


class LyricsPatch(PatchStatus):
    """詞の差分（2026-09-28）。直すフレーズだけを番号で丸ごと書き直す。"""
    # ※`reason` は読み手なし（記録にも依頼文にも出ない）。直す前に考えを書かせる欄として残す（2026-09-30・出-13）。
    #   `trouble` は直しの記録に入る（`LyricsStep.apply`・ほかの工程の差分と同じ）
    reason: str = Field(description="何をどう直したか（全体で一言）")
    phrases: list[PhraseLyric] = Field(default_factory=list,
                                       description="直すフレーズだけ（番号は提示どおり）。直さないフレーズは書かない。"
                                                   "詞の無いフレーズは、その番号で書けば足される")
    drop: list[JsonInt] = Field(default_factory=list,
                            description="**消すフレーズの番号**（旋律に無い番号の詞など）。番号を付け替えるときは、"
                                        "正しい番号で phrases に書き、間違った番号をここに書く。phrases に書いた番号はここに書かない")

    # ★**同じ番号を書き直しと消すの両方に書いた差分は型が差し戻す**（2026-10-03・記B・仕様書 §15）。以前は消してから書くので
    #   phrases が黙って勝った。歌い方の差分（書き直しと取り消しの両方）と同じ扱い
    @model_validator(mode="after")
    def _drop_or_write(self):
        if both := sorted({p.index for p in self.phrases} & set(self.drop)):
            raise ValueError(f"番号 {'・'.join(map(str, both))} を phrases と drop の両方に書いた——書き直すなら phrases だけ、"
                             "消すなら drop だけ")
        return self


# ★当てられない差分（無い番号・同じ番号2つ・空の詞）は差し戻す（2026-10-03 再レビュー・`LyricsStep.patch_problem`）
_lyrics_fixer = guard_patch(Agent(pick_model("LYRIC", "flash"), output_type=LyricsPatch, system_prompt=prompt("song_lyrics_fix")))


class LyricsStep(StepFixer):
    """**詞の工程の検査と直し**（2026-09-28・工程が自分の検査を持つ形）。値は詞の工程が書いた `MelodyLyrics`。

    形式の検査だけ:
    - 記法（`written`＝各フレーズの詞・`LYRIC_LINE`）: 歌えない文字（ひらがな以外）・助詞「を」・譜割りの書式（`lyric_slots.problems`）
    - 工程の検査: フレーズ番号（旋律に無い番号・重複）・詞の無いフレーズ・印の数と音の数のずれ
      （組み立ては補正しないので、ここで直させる・2026-09-28）・**理由の無い標準外の詰め**（ふつうの2音節を1つの音に詰めたのに
      `join_reason` が空・2026-10-06。サビの詞・2番の詞も同じこの検査）
    """
    key = "lyrics"
    label = "詞"
    agent = _lyrics_fixer
    return_hint = "**直すフレーズだけ**を phrases に入れてください（そのフレーズの text と display を丸ごと書き直す）。"

    def __init__(self, melody, indices: list[int] | None = None, *, bpm: float | None = None):
        """`indices` は書くフレーズの番号（1始まり）。無ければ旋律の全フレーズ（詞の工程）。サビの詞はサビの番号だけ。
        `bpm` は譜割りの「ー伸ばし」の秒の下限（詞を書く依頼文の `fu_wari` と同じ値を渡す・2026-10-04 組F3）。"""
        self.melody, self.bpm = melody, bpm
        self._indices = list(indices) if indices is not None else None

    @property
    def indices(self) -> list[int]:
        return self._indices if self._indices is not None else list(range(1, len(self.melody.phrases) + 1))

    def written(self, words):
        # ★書いたフレーズを全部見る（旋律に無い番号・重複も。2026-09-29 に記法の型へまとめる前は、譜割りの書式だけ
        #   旋律にある番号の最後の1件しか見ていなかった）
        return ((f"フレーズ{p.index}「{p.text}」", LYRIC_LINE, p.text) for p in words.phrases)

    def _outside(self, x: int) -> str:
        n = len(self.melody.phrases)
        return f"フレーズ番号 {x} は旋律に無い（旋律は {n}フレーズ）——この詞は歌われない"

    def check(self, words):
        want = set(self.indices)
        idx = [p.index for p in words.phrases]
        out = [self._outside(x) for x in sorted({x for x in idx if x not in want})]
        out += [f"フレーズ番号 {x} が2回書かれている——後に書いたほうだけが歌われる"
                for x in sorted({x for x in idx if idx.count(x) > 1})]
        # 歌えない文字・譜割りの書式は記法（`written`）が見る
        # ★**詞の無いフレーズ・モーラ数と音の数のずれ**も形式の検査（2026-09-28）。組み立ては補正しない（詞の無い音は鳴らさず、
        #   音の無いモーラは歌わない）ので、ここで直させる。以前は組み立てが伸ばし・音の分割・末尾の切り捨て・「あ」で埋めていた
        by = {p.index: p.text for p in words.phrases}
        for i in self.indices:
            ph = self.melody.phrases[i - 1]
            t = by.get(i, "")
            if not (t or "").strip():
                out.append(f"フレーズ{i}（{_ls.slot_count(ph.notes)}音）に詞が無い——この旋律は鳴らない")
                continue
            # ★**数えるのは印**（2026-09-29・譜割りつきの詞）。空白で区切った1つが1音。組み立てと同じ読み方
            # ★**音は ABC で書いた音の数**（2026-10-01・判-H）。タイでつないだ音は書いた音ごとに印を1つ（続きの音は `_`）
            ss = LYRIC_LINE.read(t)
            got = ck.count_is(f"フレーズ{i}「{t}」", len(ss), _ls.slot_count(ph.notes), "音の印の数",
                              then="空白で区切った1つの印が旋律の1音（タイでつないだ音は続きの音も1つ・続きは `_`）。"
                                   "多い印は歌われず、足りない音は鳴らない")
            if got:
                # ★**どの音が足りない・余るかを言う**（2026-10-04 組F3・3‑4）。以前は「6（7であるべき）」だけで、どの音に印が
                #   無いのかを直す LLM が数え直すしかなかった。前から順に当てた結果を出す（どこで1つ抜けたかは機械には分からない）
                _rows, short, extra = self._pairs(i, ss)
                marks = self._marks_of(i)
                where = ("前から順に当てると、" + "・".join(f"{j}音目（{marks[j - 1]}）" for j in short[:4])
                         + ("ほか" if len(short) > 4 else "") + "に印が足りない" if short
                         else f"前から順に当てると、{_ls.slot_count(ph.notes) + 1}個目からの印「{' '.join(extra)}」が余る"
                         f"（{len(marks)}音目 {marks[-1] if marks else ''} で旋律が終わる）")
                got = [got[0] + f"（{where}。抜けた・増えた所が途中なら、そこから後ろの音の当たり方もずれる——"
                       "依頼文の譜割り（音ごとに当たっている印）で、どの音にどの印を置くかを決め直す）"]
            out += got
            if not got:
                out += [f"フレーズ{i}「{t}」: {w}" for w in _ls.tie_problems(ss, ph.notes)]
            # ★**ふつうの2音節の詰めは理由が要る**（2026-10-06 本人決定）。詰めの標準は促音・撥音を前の音に収めるときだけ。
            #   『火花になって飛べ』（BPM200）は行末などで42か所を詰め、「何小節かに一回最後の音が詰められている」と聞こえた。
            #   詞の点検役で見つけるのでは遅い（旋律からやり直すと詞の設計・サビの詞・詞を捨てる）ので、書いたその場で返す。
            #   理由（`join_reason`）が書いてあれば通し、詞の点検役が理由ごと読む（`join_frames`）
            why = next((getattr(p, "join_reason", "") or "" for p in words.phrases if p.index == i), "")
            if (pj := _ls.plain_joins(t)) and not why.strip():
                out.append(f"フレーズ{i}「{t}」: ふつうの2音節を1つの音に詰めた所 " + "・".join(f"「{x}」" for x in pj)
                           + "——詰めの標準は、促音「っ」・撥音「ん」を前の音に収めるときだけ（速い曲でも、ふつうの音節は1音1音節で歌う）。"
                             "言い回しを変える・短い語にして1音1音節に収める（旋律の音の数は変えられない）。"
                             "長音なら `~` を付けずに続けて書く（`よお`・`にい`）。"
                             "どうしても詰めるなら、そのフレーズの join_reason に、どの詰めを・なぜ詰めるかを書く")
        return out

    # ---- 譜割り（2026-10-04 組F3・3‑2）: 直しの依頼文に、詞を書くときと同じ音の印（`note_marks`）を、いまの印と並べて出す ----
    #: 直しの依頼文に出す譜割りの上限（字）。超えたら、出さなかったフレーズの番号を書く（黙って切らない）
    FRAME_LIMIT = 6000

    def _marks_of(self, i: int) -> list[str]:
        """フレーズ `i` の音ごとの印（`E4:0.5ー伸ばし` など）。高さの読めない旋律（尺の変更の `_PhraseCounts`）は拍とタイだけ。"""
        if not hasattr(self, "_mk"):
            try:
                self._mk = note_marks(self.melody, getattr(self, "bpm", None))
            except Exception:                               # noqa: BLE001（音の高さを持たない旋律の形）
                self._mk = None
        notes = self.melody.phrases[i - 1].notes
        if self._mk is not None and i - 1 < len(self._mk) and len(self._mk[i - 1]) == len(notes):
            return self._mk[i - 1]
        out = []
        for n in notes:
            b = getattr(n, "beats", None)
            m = f"{b:g}拍" if isinstance(b, (int, float)) else "音"
            if t := _ls.ties_of(n):
                m += f"⌒タイ{1 + t}音（印{1 + t}つ・続きは {' '.join(['_'] * t)}）"
            out.append(m)
        return out

    def _pairs(self, i: int, ss) -> tuple[list[str], list[int], list[str]]:
        """フレーズ `i` の音に、いまの印 `ss` を前から順に当てる → (音ごとの行, 印が足りない音の番号, 余る印)。
        当て方は組み立て（`lyric_slots.by_note`）と同じ（タイでつないだ音は印を 1＋続きの数）。"""
        notes, marks = self.melody.phrases[i - 1].notes, self._marks_of(i)
        rows, short, pos = [], [], 0
        for j, (n, m) in enumerate(zip(notes, marks), 1):
            need = 1 + _ls.ties_of(n)
            got = [s.token for s in ss[pos:pos + need]]
            pos += need
            if len(got) < need:
                short.append(j)
            rows.append(f"{j}音目 {m}←" + (f"「{' '.join(got)}」" if got else "（印なし）")
                        + (f"（印が{need - len(got)}つ足りない）" if got and len(got) < need else ""))
        return rows, short, [s.token for s in ss[pos:]]

    def show(self, words):
        by = {p.index: p for p in words.phrases}
        rows = "\n".join(f"フレーズ{i}（{_ls.slot_count(self.melody.phrases[i - 1].notes)}音）「{getattr(by.get(i), 'text', '')}」"
                         + (f"　［表示］{by[i].display}" if by.get(i) and by[i].display else "")
                         + (f"　［join_reason］{by[i].join_reason}" if by.get(i) and (getattr(by[i], "join_reason", "") or "").strip()
                            else "")
                         for i in self.indices)
        return (f"【いまの{self.label}】（番号・その旋律の音の数・歌う詞＝空白で区切った印・見せる詞）\n" + rows
                + self._frames(words))

    def _frames(self, words) -> str:
        """**所見の出たフレーズの譜割り**（音ごとの高さ:拍・▲最高音・ー伸ばし・⌒タイ・休みと、いま当たっている印）。
        ★以前は音の数しか出ず、印を足すときにどの音がタイの続き（`_`）か・どこが伸ばしかが直す LLM に分からなかった（R3 の欠け3）"""
        import re
        try:
            named = {int(x) for f in self.findings(words) for x in re.findall(r"フレーズ(\d+)「", f)}
        except Exception:                                   # noqa: BLE001（検査が落ちても依頼文は出す）
            named = set()
        want = [i for i in self.indices if i in named] or list(self.indices)
        by = {p.index: p.text for p in words.phrases}
        blocks, used, left = [], 0, []
        for i in want:
            ss = LYRIC_LINE.read(by.get(i, "") or "")
            rows, _short, extra = self._pairs(i, ss)
            b = (f"フレーズ{i}（{self.melody.phrases[i - 1].section if hasattr(self.melody.phrases[i - 1], 'section') else ''}）: "
                 + " ／ ".join(rows) + (f" ／ 余る印: 「{' '.join(extra)}」" if extra else ""))
            if used + len(b) > self.FRAME_LIMIT and blocks:
                left.append(i)
                continue
            blocks.append(b)
            used += len(b)
        if not blocks:
            return ""
        note = (f"\n（譜割りは字数の上限 {self.FRAME_LIMIT}字で切った——出していないフレーズ: "
                f"{'・'.join(map(str, left))}）" if left else "")
        return ("\n\n【譜割り】（所見の出たフレーズ。音ごとに 高さ:拍・▲最高音・ー伸ばし・⌒タイ（続きの音の印は `_`）・｜＝小節線・(休◯)＝音のあとの休み"
                "。← の右が、いまの印を前から順に当てたもの）\n" + "\n".join(blocks) + note)

    def _not_here(self, xs: list[int]) -> str:
        """書く番号の外（`phrases` に書けない番号）の所見の文。サビの詞はサビの番号だけ（`ChorusLyricsStep` が上書き）。"""
        return f"phrases の番号 {'・'.join(map(str, xs))} は旋律に無い（旋律は {len(self.melody.phrases)}フレーズ・番号は 1〜{len(self.melody.phrases)}）"

    def patch_problem(self, words, patch) -> str:
        """**差分の番号の所見**（無ければ空・2026-10-03 再レビュー）。直しを頼む間だけ、差分の検証（`song_fix.guard_patch`）が読んで差し戻す。

        - `phrases`: 番号は書くフレーズの番号（詞＝旋律の全フレーズ・サビの詞＝サビの番号）・同じ番号は1回だけ・text が空の直しは書けない
        - `drop`: 番号は【いまの…】にある番号（旋律に無い番号の詞を消すため）・同じ番号は1回だけ
        ★以前は同じ番号の2つ目が黙って勝ち（後のほうだけ残る）、無い番号の drop は黙って何もしなかった（編成・伴奏・声部と同じ形に）。
        ※同じ番号を `phrases` と `drop` の両方に書いた差分は型（`LyricsPatch._drop_or_write`）が差し戻す（2026-10-03・記B）ので、ここには来ない
        """
        want = set(self.indices)
        idx = [p.index for p in getattr(patch, "phrases", None) or []]
        drops = list(getattr(patch, "drop", None) or [])
        have = {p.index for p in words.phrases}
        out: list[str] = []
        if bad := sorted({x for x in idx if x not in want}):
            out.append(self._not_here(bad))
        if dup := sorted({x for x in idx if idx.count(x) > 1}):
            out.append(f"phrases に番号 {'・'.join(map(str, dup))} を2回以上書いた——1つの番号に直しは1つ")
        if empty := sorted({p.index for p in getattr(patch, "phrases", None) or [] if not (p.text or "").strip()}):
            out.append(f"phrases の番号 {'・'.join(map(str, empty))} の text が空——直すなら詞を書き、直さないならその番号を書かない")
        if badd := sorted({x for x in drops if x not in have}):
            out.append(f"drop の番号 {'・'.join(map(str, badd))} は【いまの{self.label}】に無い——消せるのはいま書いてある番号だけ")
        if dupd := sorted({x for x in drops if drops.count(x) > 1}):
            out.append(f"drop に番号 {'・'.join(map(str, dupd))} を2回以上書いた——1回だけ")
        return ("差分の番号: " + "／".join(out)) if out else ""

    def apply(self, words, patch):
        # ★**足す・消す・番号の付け替えもできる**（2026-09-29）。以前は既にある番号の最初の1件を上書きするだけで、
        #   「詞の無いフレーズ」「旋律に無い番号」「番号の重複」の所見は直せなかった（重複は歌われない側を書き換えていた）
        new = words.model_copy(deep=True)
        done: list[str] = []
        # ★無い番号・同じ番号の2つ目・空の詞は、差分の検証（`patch_problem`・`song_fix.guard_patch`）が差し戻すので来ない（2026-10-03）。
        #   型を通らずに来たら当てずに、当てなかったことを直しの記録に書く（以前は2つ目が黙って勝った）
        if why := self.patch_problem(words, patch):
            done.append(f"当てなかった: {why}")
        want, seen = set(self.indices), set()
        for k in getattr(patch, "drop", None) or []:
            if any(x.index == k for x in new.phrases):
                new.phrases = [x for x in new.phrases if x.index != k]
                done.append(f"{k}を消した")
        for p in getattr(patch, "phrases", None) or []:
            if p.index not in want or p.index in seen or not (p.text or "").strip():
                continue
            seen.add(p.index)
            old = next((x for x in new.phrases if x.index == p.index), None)
            # 同じ番号が2つあったら1つにまとめる（重複の所見の直し）
            # ★見せる詞も差分に書いたとおり（2026-10-02・案2）。直しは text と display を丸ごと書く決まりで、空は型が差し戻す。
            #   以前は差分の display が空ならいまの display を残していた（歌う詞だけ変わり、見せる詞が古いまま残りえた）
            new.phrases = [x for x in new.phrases if x.index != p.index]
            new.phrases.append(PhraseLyric(index=p.index, text=p.text, display=p.display,
                                           join_reason=getattr(p, "join_reason", "") or ""))   # 標準外の詰めの理由も書いたとおり（2026-10-06）
            done.append(str(p.index) if old is not None else f"{p.index}を足した")
        new.phrases.sort(key=lambda x: x.index)
        note = f"直した（{'・'.join(done)}）" if done else "直しなし"
        if getattr(patch, "trouble", ""):                  # 直さなかった理由を直しの記録へ（2026-09-30・出-13。以前は誰も読まなかった）
            note += f"／直さなかった理由: {patch.trouble}"
        return new, note
# 詞の設計（①）とサビの詞（②）。2026-09-13 に分解（`LyricPlan` の説明を見ること）
_lyric_plan = Agent(pick_model("LYRIC", "flash"), output_type=LyricPlan, system_prompt=(
    prompt("song_lyric_plan")
))
_lyric_plan_fixer = guard_patch(Agent(pick_model("LYRIC", "flash"), output_type=LyricPlanPatch,
                                      system_prompt=prompt("song_lyric_plan_fix")))


class LyricPlanStep(PatchNumbers, StepFixer):
    """**詞の設計の形式の検査と差分の直し**（2026-09-30・出-14②／判-5）。値は詞の設計が書いた `LyricPlan`。

    - `hook_section`・`sections[].kind` が設計の区間名か（親の決まり `song_checks.against_design`。前後の空白だけ
      取り除いて設計の綴りにする＝`canonize`。大文字小文字・途中の空白の違いは所見）
    - `hook_section` が**歌のある区間**か（旋律がフレーズを書いた区間。無い区間ではサビの詞が書けない）
    - `hook_line` がその区間の行の範囲（1〜その区間のフレーズ数）か（判-5。決め台詞が**その行に**乗ったかは内容＝点検役）
    ★以前は検査が無く、区間名にならない `hook_section`（1周目の「chorus1, chorus2, chorus_last」）は**黙って最初のサビ
      （chorus で始まる区間）に落ち**、それも無ければサビの詞の工程が記録なしで飛んだ。`sections` の綴りが違うと、サビの詞・
      詞の点検役の〈狙い〉が完全一致で引けずに黙って空になった。当て推量は外し、所見にして差分で直させる（直らなければ止まる）。
    """
    key = "lyric_plan"
    label = "詞の設計"
    agent = _lyric_plan_fixer
    return_hint = "**直す所だけ**を書いてください（hook_section は直すときだけ・sections は番号で指して直す／消す）。"

    def __init__(self, design_sections, sung: list[str], lines: dict[str, int] | None = None):
        """`sung` は歌のある区間名（旋律の順）、`lines` は区間ごとのフレーズ数（無ければ行の範囲は見ない）。"""
        self.design, self.sung, self.lines = list(design_sections), list(sung), dict(lines or {})

    def design_items(self, plan):
        return ([("hook_section（決め台詞を置く区間）", plan.hook_section, None)]
                + [(f"sections の{i}番", s.kind, None) for i, s in enumerate(plan.sections, 1)])

    def rename_sections(self, plan, mapping):
        return plan.model_copy(update={
            "hook_section": mapping.get(plan.hook_section, plan.hook_section),
            "sections": [s.model_copy(update={"kind": mapping.get(s.kind, s.kind)}) for s in plan.sections]})

    def check(self, plan) -> list[str]:
        names = [s.kind for s in self.design]
        hs = ck.resolve_section(plan.hook_section, names)
        if hs is not None and ck.resolve_section(hs, self.sung) is None:     # 設計の区間名だが歌が無い
            return [f"hook_section（決め台詞を置く区間）: 『{plan.hook_section}』は歌の無い区間——"
                    f"サビの詞を書く区間は歌のある区間から1つ（{' / '.join(self.sung)}）"]
        sec = ck.resolve_section(plan.hook_section, self.sung) if hs is not None else None
        n = self.lines.get(sec or "", 0)
        if n and not 1 <= plan.hook_line <= n:
            return [f"hook_line（決め台詞を置く行）: {plan.hook_line}行目は『{sec}』の範囲に無い（この区間は {n}行・1〜{n}）"]
        return []

    def show(self, plan) -> str:
        rows = "\n".join(f"  {i}番: {s.kind}: {s.says[:60]}" for i, s in enumerate(plan.sections, 1))
        return (f"【いまの詞の設計】\nhook: 「{plan.hook}」／hook_section: {plan.hook_section}／hook_line: {plan.hook_line}\n"
                f"sections:\n{rows}\n\n【設計の区間名】{' / '.join(s.kind for s in self.design)}"
                f"\n【歌のある区間（行数）】{' / '.join(f'{k}（{self.lines[k]}行）' if k in self.lines else k for k in self.sung)}")

    # ★**差分の番号の検証**（2026-10-03・再レビュー・`song_fix.PatchNumbers`）: `sections[].no` は 1〜いまの項目の数・同じ番号は1回だけ。
    #   以前は範囲外を黙って捨て、同じ番号の2つ目が勝った。
    #   ★`kind` も drop も無い項目は中身の無い直し（2026-10-03・記B。以前は `kind` が空なら消していた＝消し方が2通り）
    patch_what = "詞の設計の sections"

    def patch_count(self, plan) -> int:
        return len(plan.sections)

    def patch_items(self, patch) -> list[tuple[int, bool]]:
        return [(f.no, not f.drop and not (f.kind or "").strip()) for f in getattr(patch, "sections", None) or []]

    def apply(self, plan, patch):
        new = plan.model_copy(deep=True)
        done: list[str] = []
        # 範囲外・同じ番号の2つ目は差分の検証（`guard_patch`）が差し戻すので来ない。型を通らずに来たら当てずに記録へ（2026-10-03）
        if why := self.patch_problem(plan, patch):
            done.append(f"当てなかった: {why}")
        seen: set[int] = set()
        if (patch.hook_section or "").strip():
            new.hook_section = patch.hook_section.strip()
            done.append(f"hook_section を『{new.hook_section}』に")
        if patch.hook_line:
            new.hook_line = patch.hook_line
            done.append(f"hook_line を{new.hook_line}行目に")
        drop = set()
        for f in patch.sections or []:
            if not 1 <= f.no <= len(new.sections) or f.no in seen or not (f.drop or f.kind.strip()):
                continue
            seen.add(f.no)
            if f.drop:
                drop.add(f.no)
                done.append(f"sections の{f.no}番を消した")
            else:
                new.sections[f.no - 1] = new.sections[f.no - 1].model_copy(update={"kind": f.kind.strip()})
                done.append(f"sections の{f.no}番を『{f.kind.strip()}』に")
        new.sections = [s for i, s in enumerate(new.sections, 1) if i not in drop]
        note = "・".join(done) if done else "直しなし"
        if getattr(patch, "trouble", ""):
            note += f"／直さなかった理由: {patch.trouble}"
        return new, note


# サビの詞も1案で作る（2026-09-30・旋律と同じ判断）。決め台詞・行頭の音・伸ばす母音は、選抜の札で見ていたものを
# 詞の点検役の材料へ移した（`chorus_facts`）
_lyric_chorus = Agent(pick_model("LYRIC", "flash"), output_type=ChorusLyrics, system_prompt=prompt("song_lyric_chorus"))
_lyric_chorus_fixer = guard_patch(Agent(pick_model("LYRIC", "flash"), output_type=LyricsPatch,
                                        system_prompt=prompt("song_lyric_chorus_fix")))   # 差分の検証（2026-10-03・詞と同じ）


class ChorusLyricsStep(LyricsStep):
    """**サビの詞の形式の検査と差分の直し**（2026-09-30・判-7 本人承認）。値はサビの詞が書いた `ChorusLyrics`。

    詞の工程（`LyricsStep`）と**同じ検査を、サビのフレーズ番号に絞って**掛ける: 記法（歌えない文字・「を」・譜割りの書式＝`LYRIC_LINE`）・
    番号（サビに無い番号・重複）・詞の無いフレーズ・印の数＝音の数。
    ★以前は検査が無く、間違ったサビも「確定・そのまま書き写す」として詞へ渡り、詞の検査で初めて所見になった——**書き写せという指示と
      直せという所見がぶつかり**、間違ったサビは `lyric_chorus` として曲に残って作り直しの「前回の出力」と後の工程へ渡っていた。
    """
    key = "lyric_chorus"
    label = "サビの詞"
    agent = _lyric_chorus_fixer

    def _outside(self, x: int) -> str:
        return (f"フレーズ番号 {x} はサビのフレーズではない（書くのは {', '.join(map(str, self.indices))}）——"
                "サビの詞はこの番号だけ（この詞は歌われない）")

    def _not_here(self, xs: list[int]) -> str:
        return (f"phrases の番号 {'・'.join(map(str, xs))} はサビのフレーズではない——サビの詞が書くのは "
                f"{', '.join(map(str, self.indices))} だけ")


def fu_wari(melody, *, top: int | None = None, bpm: float | None = None, ts: str | None = None) -> str:
    """**譜割り**——各フレーズの音数・音価に加えて、**最高音と伸ばしの位置**を印で示す（2026-09-13）。

    プロは「推しメロ（サビで最も耳に飛び込む音）から言葉をつける」。それには
    **どの音が最高音で、どの音を伸ばすか**が見えていなければならない。以前は音数と音価しか
    渡していなかったので、最高音（86）に何の言葉が乗るかを誰も決めていなかった。

    ★**休みの位置と音の高さも出す**（2026-09-30・出-15）。以前は音数・音価と印だけで、詞はフレーズの前の休み・音のあいだの
      小休み（言葉の切れ目を置きやすい所）と旋律の上がり下がりを、丸ごと渡る旋律の JSON（約2万字）から拾うしかなかった。
      1音は `音名:拍`（C 基準・移調前）、フレーズの前の休みは `前に休み◯拍`、音のあとの休みは `(休◯)`。
      1行は1フレーズのまま（サビの詞の依頼文が行番号で切り出す）。
    ★**高さは国際式の音名**（`G4:1`・`C5:1`。2026-10-01・本人「楽譜（国際式の音名）に寄せる」）。以前は階名（`ソ4:1`）。
      綴りは表と同じ `song_table.spell_pitch`（C 基準なので調を渡さない＝黒鍵は ♯）。
    ★**`▲最高音` は区間の中でいちばん高い音**（2026-10-03・組G2・`_section_tops`）。以前は曲全体の最高音だけで、Cメロのほうが
      高い曲ではサビに ▲ が付かなかった。`ー伸ばし` は `_is_hold`（`bpm` を渡すと秒の下限も見る）。
    """
    lines = []
    for i, (p, marks) in enumerate(zip(melody.phrases, note_marks(melody, bpm, ts)), 1):
        rest = float(getattr(p, "rest_before", 0.0) or 0.0)
        lines.append(f"  フレーズ{i}（{p.section}）: **{_ls.slot_count(p.notes)}音** ／"
                     + (f"前に休み{rest:g}拍 ／" if rest > 0 else "") + " ".join(marks))
    return "\n".join(lines)


def _bar_of(ts: str | None) -> tuple[float, list[float]]:
    """拍子 → (1小節の拍数（4分音符）, 拍の頭の位置)。読めない拍子（古い記録）は 4/4 として数える。"""
    from song_design import beat_heads, beats_per_bar      # 拍子から数を出す1か所
    try:
        ts = ts or ctx.METER
        return beats_per_bar(ts), [(h - 1) / 4 for h in beat_heads(ts)]
    except Exception:                                      # noqa: BLE001
        return 4.0, [0.0, 1.0, 2.0, 3.0]


def _bars_in(lo: float, hi: float, bar: float) -> bool:
    """`lo` より後ろ・`hi` まで（両端は 16分に丸めた位置）に小節線があるか。"""
    nxt = (int(lo / bar + 1e-6) + 1) * bar
    return nxt <= hi + 1e-6


def note_marks(melody, bpm: float | None = None, ts: str | None = None) -> list[list[str]]:
    """**譜割りの音の印**（フレーズごと・音ごと。`E4:0.5`・`A4:2.5ー伸ばし⌒タイ2音（…）`・`(休2)`）。詞を書く依頼文（`fu_wari`）と
    詞の直しの依頼文・所見（`LyricsStep`・2026-10-04 組F3・3‑2）が同じ1つを読む（見せ方を2通りにしない）。

    ★**小節線を出す**（2026-10-04・本人「単語の切れ目と小節の切れ目が一致していない」）。小節の頭で始まる音の前に `｜`、
      タイで小節線をまたぐ音には「途中に小節線｜」。以前は小節線がどこにも見えず、詞はタイの続き `_` のあとに語の続きを置いた
      （『水たまりの青空』の Aメロ「あす _ ｜ふぁると」）。位置は区間の頭からの足し算（`_onsets`・組み立てと同じ）。"""
    from song_table import spell_pitch                      # LLM に見せる高さの綴りは1つ
    out: list[list[str]] = []
    keys = [[solfa_to_midi(n.solfa, n.octave) for n in p.notes] for p in melody.phrases]
    # ★印を付けるのは**実際にいちばん高い音**だけ（`top` は上限であって閾値ではない。
    #   閾値に使うとサビの全音に ▲ が付いた——2026-09-13 の自己検査で発覚）。
    tops = _section_tops(melody)
    bar, _heads = _bar_of(ts)
    for p, ks, ons in zip(melody.phrases, keys, _onsets(melody)):
        marks = []
        # ★「伸ばし」は**その行の中で相対的に長い音**（中央値の2倍以上）。拍で固定すると
        #   速い曲では普通の音まで伸ばし扱いになる（BPM185 で 1拍＝0.32秒・98個に印が付いた）。
        for n, k, at in zip(p.notes, ks, ons):
            m = ("｜" if abs(at % bar) < 1e-6 or abs(at % bar - bar) < 1e-6 else "") + f"{spell_pitch(k)}:{n.beats:g}"
            if k >= tops.get(p.section, k):
                m += "▲最高音"
            if _is_hold(n, p.notes, bpm):
                m += "ー伸ばし"
            t = _ls.ties_of(n)
            cross = _bars_in(at, at + float(n.beats) - 0.25, bar) if n.beats > 0.25 else False
            if t:                                          # ★タイでつないだ音は印も書いた音の数だけ（続きは `_`・2026-10-01 判-H）
                m += (f"⌒タイ{1 + t}音（印{1 + t}つ・続きは {' '.join(['_'] * t)}"
                      + ("・途中に小節線｜" if cross else "") + "）")
            elif cross:
                m += "（途中に小節線｜）"
            gap = float(getattr(n, "gap_after", 0.0) or 0.0)
            if gap > 0:                                    # 行末の音の後ろの空き（区間の終わりの休み）もここに出る
                m += f" (休{gap:g})"
            marks.append(m)
        out.append(marks)
    return out


_ROW_OF = {"あ": "あかさたなはまやらわがざだばぱゃ", "い": "いきしちにひみりぎじぢびぴ",
           "う": "うくすつぬふむゆるぐずづぶぷゅ", "え": "えけせてねへめれげぜでべぺ",
           "お": "おこそとのほもよろをごぞどぼぽょ"}


#: 濁点・半濁点を落とす対応表（**読み変換が連濁を落とすことがある**ので、比べる前に揃える）。
_DAKUTEN = lyric_display._DAKUTEN                           # 表は1つ（2026-10-04 に語の切れ目の推定と共有）


def _no_dakuten(s: str) -> str:
    """濁点を落とした形。**表記ゆれを吸収して同じ語かを見る**ためだけに使う。"""
    return s.translate(_DAKUTEN)


def to_sung_kana(text: str) -> str:
    """**歌う表記**（ひらがな・を→お・へ→え）に直す。hook（設計は漢字まじり）と詞（ひらがな）を
    比べるため（2026-09-13）。「風を切って」→「かぜおきって」。"""
    try:
        import pykakasi
        s = "".join(x.get("hira", "") for x in pykakasi.kakasi().convert(text or ""))
    except Exception:
        s = text or ""
    s = s.replace("を", "お").replace("へ", "え")
    # ★**オ段の長音は「お」**——詞はそう書く決まり（「衝動」→「しょおどお」）。ここも揃えないと
    #   hook「この衝動」が実体「このしょおどお」に対して × と出る（2026-09-13 の実走で発覚）。
    # ★**長音「ー」は直前の母音に開く**（2026-09-30）。以前は「ー」を消していたので、hook「スニーカー」が「すにか」になり、
    #   詞「す にい かあ」（詞は長音を母音で書く決まり）と一致せず「決め台詞がサビに無い」と誤って出た（『駆け抜けろスニーカー』）
    out: list[str] = []
    for c in s:
        prev = out[-1] if out else ""
        if c == "う" and prev and prev in _O_ROW:
            out.append("お")
        elif c == "ー":
            v = _VOWEL.get(prev) or _SMALL_VOWEL.get(prev, "")
            if v:                                          # 「ん」「っ」のあとの「ー」は開く母音が無いので数えない
                out.append(v)
        else:
            out.append(c)
    return "".join(c for c in out if c in _KANA_OK)


#: **伸ばしとみなす長さの秒の下限**（2026-10-03・組G2）。拍の基準（行の中央値の2倍以上かつ1拍以上）だけだと、速い曲では
#: 1拍（BPM168 で 0.36秒）の音まで伸ばしになる。BPM が分かるときだけ、この秒にも届く音を伸ばしとする（BPM120 の1拍＝0.5秒）
HOLD_MIN_SEC = 0.5


def _is_hold(note, notes, bpm: float | None = None) -> bool:
    """その行の中で**相対的に長い**音か（中央値の2倍以上かつ1拍以上・`fu_wari` と同じ基準）。
    `bpm` を渡したときは `HOLD_MIN_SEC` 秒以上も条件にする（速い曲で短い音を伸ばしと数えない・2026-10-03）。"""
    bs = sorted(n.beats for n in notes)
    med = bs[len(bs) // 2] if bs else 1.0
    if note.beats < max(1.0, med * 2.0):
        return False
    return not bpm or note.beats * 60.0 / float(bpm) >= HOLD_MIN_SEC


def _section_tops(melody) -> dict[str, int]:
    """区間 → **その区間の最高音**（C 基準）。譜割りの `▲最高音` はこれ（2026-10-03・組G2）。
    ★以前は曲全体の最高音だけに印を付けていた。Cメロや転調前の区間のほうが高い曲では、サビの行に ▲ が付かず、
      サビで耳に飛び込む音（推しメロ）が見えなかった。詞の設計が決め台詞の区間を決める前に譜割りを見せるので、
      どの区間がサビかは機械には分からない——区間ごとに付け、サビの行の ▲ がサビの中の最高音になる。"""
    out: dict[str, int] = {}
    for p in melody.phrases:
        for n in p.notes:
            k = solfa_to_midi(n.solfa, n.octave)
            out[p.section] = max(out.get(p.section, k), k)
    return out


def _onsets(melody) -> list[list[float]]:
    """各フレーズの各音の**区間の頭からの位置**（拍＝4分音符・16分に丸める）。組み立て（`song_place._lay`）と同じ足し方:
    区間の1フレーズ目は区間の頭（小節の頭）から、`rest_before` を足して始まり、音ごとに長さと `gap_after` だけ進む。"""
    cur: dict[str, float] = {}
    out: list[list[float]] = []
    for p in melody.phrases:
        at = cur.get(p.section, 0.0) + float(getattr(p, "rest_before", 0.0) or 0.0)
        row = []
        for n in p.notes:
            row.append(round(at * 4) / 4)
            at += float(n.beats) + float(getattr(n, "gap_after", 0.0) or 0.0)
        out.append(row)
        cur[p.section] = at
    return out


def hook_in_chorus(hook: str, chorus_text: str) -> bool:
    """決め台詞がサビの詞に入っているか。**歌う表記に直し、濁点の違いを吸収して**比べる（`to_sung_kana`・`_no_dakuten`）。

    ★読み変換（pykakasi）は連濁を落とすことがある（「立ち止まろう」→『たちとまろお』・2026-09-16）。長音は母音に開く（2026-09-30）。
    ★**両方を同じ決まりで畳む**（2026-10-07・VM の『水たまりのプリズム』）。`to_sung_kana` は「オ段のあとの う → お」を
      語の切れ目に関係なく当てるので、hook「空を映した」は『そらおおつした』になる。詞は区切り（空白）つきで渡るため、
      `お う` の間の空白で畳みが効かず『そらおうつしたよ』のまま——**同じ言葉なのに「サビに入っていない」と鳴った**。
      空白を除いてから同じ変換に通せば、両方が『…おお…』に揃う。行をまたいだ一致は、呼び手がサビの行をつないで渡すので「入っている」と読む
      （どの行に乗ったかは `hook_lines` が別に出す）。語が変わっていれば（『さあ火花になって／未練ごと飛べよ』）一致しない。
    """
    hk = _hook_key(hook)
    return bool(hk) and hk in _hook_key(chorus_text)


def _hook_key(text: str) -> str:
    """決め台詞の照合に使う形。区切りの空白を除き、歌う表記に直して濁点を落とす（hook と詞を同じ決まりで畳む）。"""
    return _no_dakuten(to_sung_kana("".join((text or "").split())))


def hook_lines(hook: str, lines: list[str]) -> list[tuple[int, int]]:
    """決め台詞が**どの行に**乗ったか（1始まりの行の範囲の並び・2026-10-03・判-6）。1行の中に入っている行があればそれを全部、
    無ければ行をまたいで入っている**いちばん短い**範囲を全部（`(2, 3)`＝2〜3行目にまたがる）。どこにも無ければ空。
    比べ方は `hook_in_chorus`（歌う表記に直し、濁点の違いを吸収する）と同じ。"""
    n = len(lines)
    for span in range(1, n + 1):
        got = [(a + 1, a + span) for a in range(n - span + 1) if hook_in_chorus(hook, "".join(lines[a:a + span]))]
        if got:
            return got
    return []


def o_then_u(text: str) -> list[str]:
    """**オ段の音のあとに、別の音として置いた「う」**（`し ん ご う` の「ご う」・2026-10-03・判-9 の残り）。詞の点検役の材料。

    「信号」は「しんごお」と歌うのが自然（V6 は「う」を u の音で歌う）が、「想う・追う」のように「う」と言う語なら正しい——
    答えは1つに決まらない（内容）ので、断じずに事実として並べる（9/27 に外した機械の所見は、断じて誤報ばかりだった）。
    `_`（前の音節を伸ばす）は飛ばして前の音節を見る。`*`（歌わない）をはさんだら続きとみなさない。
    1つの区切りの中の `ごう`（長音）と `~` でつないだ `ご~う` は1つの音の中なので、ここでは見ない（後者は `long_vowel_joins`）。"""
    out: list[str] = []
    prev, run = None, []                                   # 直前に歌った印と、そこからの印の並び（`_` を含む）
    for s in _ls.slots(text):
        if s.kind == "skip":
            prev, run = None, []
            continue
        if s.kind == "hold":
            if prev is not None:
                run.append(s.token)
            continue
        if prev is not None and prev.moras and s.moras and s.moras[0] == "う" and prev.moras[-1][-1:] in _O_ROW:
            out.append(" ".join(run + [s.token]))
        prev, run = s, [s.token]
    return out


def o_then_u_inside(text: str) -> list[str]:
    """**1つの区切りの中で、オ段の音のあとに続く「う」**（`ごう`・`きょう`・`もう`・2026-10-03・組G2）。詞の点検役の材料。
    `o_then_u`（別の音として置いた「う」）と同じく事実として並べる——「想う」を1つの音に詰めた `もう` もありうる。
    `~` でつないだ `ご~う` は別の音節なのでここでは見ない（`long_vowel_joins` が拾う）。"""
    out: list[str] = []
    for s in _ls.slots(text):
        if s.kind != "sing":
            continue
        for syl in s.token.split(_ls.JOIN):
            ms = _ls.moras(syl)
            if any(b == "う" and a[-1:] in _O_ROW for a, b in zip(ms, ms[1:])):
                out.append(s.token)
                break
    return out


#: **単独で1つの音に乗せると言いにくい音**（2026-10-03・組G2）。促音・撥音・母音だけの音（`lyric_slots.parts_of` で1つ）
_AWKWARD_ALONE = {"っ": "「っ」", "ん": "「ん」", **{v: "母音だけの音" for v in "あいうえお"}}


def awkward_spots(by: dict[int, str], melody, ts: str | None = None, bpm: float | None = None,
                  shift_of=None) -> list[str]:
    """**言いにくい音の置き場所**（詞の点検役の材料・2026-10-03・組G2）。プロソディ（言葉と旋律の一致）の目安
    「小さい『っ』『ん』『ー』と母音だけの音は、強拍や長い音に単独で置かない」を、**場所つきの事実**で並べる。
    本人の実例は「つめた｜い」——「い」が小節の頭に来て遅れて鳴った。

    - 数える音: 1つの音に**それだけ**が乗った「っ」「ん」・母音だけの音（あ・い・う・え・お）と、`_`（前の音節の伸ばし＝「ー」）。
      `~` で詰めた中の「ん」（`ん~で`）や、1つの区切りの中の「ん」（`しん`）は単独ではないので数えない。
    - 強拍: 音の頭が**小節の頭**か**拍の頭**（拍子の `beat_heads`・6/8 は付点4分の頭）。位置は区間の頭からの足し算（`_onsets`）。
    - 長い音: `_is_hold`（行の中央値の2倍以上かつ1拍以上・`bpm` があれば `HOLD_MIN_SEC` 秒以上）＝譜割りの `ー伸ばし`。
    - `_` は**強拍だけ**数える（長い音に `_` を置くのは伸ばしそのものなので数えない）。
    - ★**単独の「っ」は場所を問わず全部出し、長さ（拍）と高さを添える**（2026-10-04・組Q2）。本人の実例は『水たまりのプリズム』の
      「け っ て」——「っ」が G5・1拍の音に単独で乗り、V6 では前の音の母音を保つ（`song_place._sustain_sokuon`）が、高い音で
      小さく聞こえた。詞の標準は「`け~っ て` と前の音に収める」なので、単独の「っ」は強拍でなくても点検役が読めるように並べる。
      高さは `shift_of`（区間 → 移調量・`song_place.section_keys`）があれば**鳴る高さ**、無ければ C 基準（移調前）と書く。
    戻り値は「フレーズ3の5音目「い」（小節の頭・2拍）」の形の並び。良し悪しは書かない——意図して置くこともある。
    """
    from song_table import spell_pitch                      # LLM に見せる高さの綴りは1つ
    from song_design import beat_heads, beats_per_bar      # 拍子から数を出す1か所
    ts = ts or ctx.METER
    try:
        bar = beats_per_bar(ts)
        heads = [(h - 1) / 4 for h in beat_heads(ts)]
    except Exception:                                      # 読めない拍子（古い記録）は 4/4 として数える
        bar, heads = 4.0, [0.0, 1.0, 2.0, 3.0]
    out: list[str] = []
    for i, (p, ons) in enumerate(zip(melody.phrases, _onsets(melody)), 1):
        ss = _ls.slots(by.get(i, ""))
        for j, (n, at, (s, _c)) in enumerate(zip(p.notes, ons, _ls.by_note(ss, p.notes)), 1):
            if s is None:
                break
            if s.kind == "hold":
                what = "`_`（前の音節の伸ばし）"
            elif s.kind == "sing" and len(s.parts) == 1 and s.parts[0] in _AWKWARD_ALONE and _ls.JOIN not in s.token:
                what = f"「{s.token}」"
            else:
                continue
            pos = at % bar
            where = []
            if abs(pos) < 1e-6 or abs(pos - bar) < 1e-6:
                where.append("小節の頭")
            elif any(abs(pos - h) < 1e-6 for h in heads):
                where.append("拍の頭")
            if s.kind != "hold" and _is_hold(n, p.notes, bpm):
                where.append(f"{n.beats:g}拍の伸ばし")
            if s.kind == "sing" and s.token == "っ":           # 単独の「っ」は場所を問わず、長さと高さを添えて出す
                k = solfa_to_midi(n.solfa, n.octave)
                if shift_of is not None:
                    try:
                        k, tag = k + int(shift_of(p.section)), "鳴る高さ"
                    except Exception:                      # noqa: BLE001 — 区間の移調量が引けなければ移調前で書く
                        tag = "高さ（C 基準・移調前）"
                else:
                    tag = "高さ（C 基準・移調前）"
                if not any(w.endswith("拍の伸ばし") for w in where):
                    where.append(f"{n.beats:g}拍")
                where.append(f"{tag} {spell_pitch(k)}")
            if where:
                out.append(f"フレーズ{i}の{j}音目{what}（{'・'.join(where)}）")
    return out


def word_breaks(by: dict[int, str], disp: dict[int, str], melody, ts: str | None = None,
                bpm: float | None = None) -> tuple[list[str], list[int]]:
    """**語の途中に伸ばし・小節線が入った所の候補**（詞の点検役の材料・2026-10-04）→ (候補の並び, 語の切れ目を推せなかったフレーズ)。

    本人の実例（『水たまりの青空』の Aメロ）: 「ぬ れ た あ す(1.5) ｜ふぁ る と」——語の途中で伸ばしてから小節をまたぎ、
    言葉がちぎれて聞こえた。実務の目安「語を言い切ってから伸ばす（伸ばしは語末側）」。
    - 語の切れ目は**推定**（`lyric_display.word_spans`＝見せる歌詞の字の種類の境目と助詞。読みは推さない）。だから候補
    - 伸ばし: `_is_hold`（譜割りの `ー伸ばし`）・タイでつないだ音・あとに `_` を置いた音。小節線: 次に歌う音の頭までに小節線がある
    - 前の音の母音をそのまま伸ばす長音（`しんご|お`）は語の途中でも数えない（伸ばしそのもの）
    - ひらがなだけの語（切れ目を推せない所）は数えない
    """
    bar, _heads = _bar_of(ts)
    out: list[str] = []
    lost: list[int] = []
    for i, (p, ons) in enumerate(zip(melody.phrases, _onsets(melody)), 1):
        text = by.get(i, "")
        kana = _ls.sung_kana(text)
        if not kana:
            continue
        spans = lyric_display.word_spans(disp.get(i, ""), kana)
        if spans is None:
            lost.append(i)
            continue
        sung: list[dict] = []                               # 歌う音ごと: 頭の位置・かなの位置・伸ばし
        pos = 0
        for n, at, (s, conts) in zip(p.notes, ons, _ls.by_note(_ls.slots(text), p.notes)):
            if s is None:
                break
            if s.kind == "sing" and s.moras:
                k = len("".join(s.moras))
                sung.append(dict(at=at, a=pos, b=pos + k, s=s, beats=float(n.beats),
                                 hold=_is_hold(n, p.notes, bpm) or bool(_ls.ties_of(n)), rest=False))
                pos += k
            elif sung and s.kind == "hold":
                sung[-1]["beats"] += float(n.beats)
                sung[-1]["hold"] = True
            elif sung and s.kind == "skip":
                sung[-1]["rest"] = True

        def word_of(c: int):
            return next((w for w in spans if w[0] <= c < w[1]), None)

        line, hits = [], []
        for x, y in zip(sung, sung[1:] + [None]):
            line.append(x["s"].token + (f"({x['beats']:g})" if x["hold"] else ""))
            if y is None:
                break
            bar_ = _bars_in(x["at"], y["at"], bar)
            if bar_:
                line.append("｜")
            w = word_of(x["b"] - 1)
            if w is None or w is not word_of(y["a"]) or not w[3]:
                continue
            if _ls._is_long(x["s"].moras[-1], y["s"].moras[0]):
                continue
            why = ([f"{x['beats']:g}拍の伸ばし"] if x["hold"] else []) + (["小節線"] if bar_ else []) \
                + (["歌わない音（*）"] if x["rest"] else [])
            if why:
                hits.append(f"「{w[2]}」の{kana[w[0]:x['b']]}｜{kana[y['a']:w[1]]}（{'・'.join(why)}）")
        if hits:
            out.append(f"フレーズ{i}「{' '.join(line).replace(' ｜ ', ' ｜')}」: " + "／".join(hits))
    return out, lost


#: **短い詰め枠の目安**（2026-10-05・`join_frames`）。`~` で詰めた音は組み立てが等分して1音1音節で鳴らす（`song_place.fit_phrase`）。
#: 1音節あたりがこれより短い、または枠全体がこれより短い詰めを「短い詰め枠」として先に並べる。
#: ★**V6 で測った限界ではなく目安**——速い曲の16分（BPM150 で 0.1秒）を1音節で言い切れる下限の見当と、2音節を言い分けるのに
#:   要る長さの見当。実例は『朝光のトランク』（BPM200）のサビ「あ~す」＝0.5拍（0.15秒）を2つに割って1音節 0.075秒、
#:   「あっそう」とつぶれて聞こえた。良し悪しは点検役が詞と旋律を読んで決める（機械は数えた事実を出すだけ）
JOIN_SHORT_SYL_SEC = 0.1
JOIN_SHORT_FRAME_SEC = 0.2


def join_frames(by: dict[int, str], melody, bpm: float | None = None,
                disp: dict[int, str] | None = None, why: dict[int, str] | None = None) -> tuple[list[str], list[str]]:
    """**標準外の詰め**＝ふつうの2音節以上を1つの音（枠）に詰めた所（詞の点検役の材料・2026-10-05・2026-10-06 に標準外だけへ）
    → (短い詰め枠, そのほかの標準外の詰め)。

    詰めた枠が2音節を歌える長さかは誰も見ていなかった（旋律は詞の音数を知らずに音の長さを決め、詞は「足りなければ詰める」で収まれば
    済ませる）。枠の長さ（拍・秒）と詰めた音節の数、1音節あたりの秒を、場所と見せる歌詞と**書いた理由**（`why`＝`PhraseLyric.join_reason`）つきで並べる。
    ★**2026-10-06（本人決定）から、詰めの標準は促音・撥音を前の音に収めるときだけ**（`lyric_slots.STD_JOIN_MORAS`）。ふつうの2音節の詰めは
      理由が無いと詞の形式の検査（`LyricsStep`）が書き直しを求めるので、ここへ来るのは理由を書いた詰め。点検役は理由ごと読んで、
      よほどの理由が無ければ差し戻す（指示書）。
    - 対象: `lyric_slots.plain_joins` と同じ（`~` でつないだものも、1つの区切りに続けて書いた `せい` も）。促音・撥音を前の音に収めた詰め
      （`け~っ`・`し~ん`・`しん`）と長音（`にい`）は対象外
    - 枠: その音の長さ（タイでまとめた続きの音を含む・`gap_after` の休みは含まない）。組み立てはこれを鳴らし分けるものの数で等分する
    - 音節: `lyric_slots.squeezed_syllables`（長音は前の音節に含む・「っ」「ん」は数えない）。「っ」「ん」を含む詰めは、
      等分の数（「っ」「ん」を含む）で1つあたりを出し、そのことを添える
    - 短い: `bpm` があるときだけ秒で見る（`JOIN_SHORT_SYL_SEC`・`JOIN_SHORT_FRAME_SEC`）。`bpm` が無ければ全部「そのほか」に拍だけで並べる
    戻り値は「フレーズ7（chorus）「あ~す」: 枠0.5拍＝0.15秒に2音節・1音節0.075秒［表示］切り開くよ明日を（理由: …）」の形。良し悪しは書かない。
    """
    short: list[str] = []
    other: list[str] = []
    disp, why = disp or {}, why or {}
    for i, p in enumerate(melody.phrases, 1):
        for n, (s, _c) in zip(p.notes, _ls.by_note(_ls.slots(by.get(i, "")), p.notes)):
            if s is None:
                break
            syl = _ls.squeezed_syllables(s)
            if len(syl) < 2:                                # 単独の音節・長音・促音や撥音を前の音に収めた詰め（標準）は対象外
                continue
            ps = s.parts or tuple(s.moras)
            beats = float(n.beats)
            tag = f"{len(syl)}音節" + (f"（「っ」「ん」を含めて{len(ps)}つに等分）" if len(ps) != len(syl) else "")
            head = f"フレーズ{i}（{p.section}）「{s.token}」: 枠{beats:g}拍"
            d = disp.get(i, "")
            w = (why.get(i, "") or "").strip()
            tail = (f"［表示］{d}" if d else "") + (f"（理由: {w}）" if w else "（理由なし）")
            if not bpm:
                other.append(f"{head}に{tag}・1つあたり{beats / len(ps):g}拍{tail}")
                continue
            sec = beats * 60.0 / float(bpm)
            per = sec / len(ps)
            row = f"{head}＝{sec:.3g}秒に{tag}・1音節{per:.3g}秒{tail}"
            (short if per < JOIN_SHORT_SYL_SEC or sec < JOIN_SHORT_FRAME_SEC else other).append(row)
    return short, other


def _oshimero(by: dict[int, str], melody, chorus_idx: list[int], bpm: float | None = None) -> list[str]:
    """サビの各行で、**推しメロ**（`▲最高音`・`ー伸ばし`＝譜割り `fu_wari` と同じ印）の音に乗った音節（2026-10-03・判-26）。
    ▲は**その区間（サビ）の中の最高音**（2026-10-03・組G2・`_section_tops`。以前は曲全体の最高音で、サビより高い区間があると付かなかった）。"""
    tops = _section_tops(melody)
    rows = []
    for k, i in enumerate(chorus_idx, 1):
        notes = melody.phrases[i - 1].notes
        got, last = [], ""
        for n_, (s_, _c) in zip(notes, _ls.by_note(_ls.slots(by.get(i, "")), notes)):
            if s_ is None:
                break
            word = (f"「{s_.token}」" if s_.kind == "sing" else
                    f"「{last}」の伸ばし" if s_.kind == "hold" and last else "歌わない")
            if s_.kind == "sing":
                last = s_.token
            top = tops.get(melody.phrases[i - 1].section, 0)
            marks = ("▲" if solfa_to_midi(n_.solfa, n_.octave) >= top else "") + ("ー" if _is_hold(n_, notes, bpm) else "")
            if marks:
                got.append(f"{marks}{word}")
        if got:
            rows.append(f"{k}行目 " + " ".join(got))
    return rows


def plan_layer_text(plan) -> str:
    """詞の設計の**裏の層と象徴**を依頼文・点検役の材料に見せる文（2026-10-05）。欄の無い型なら空。"""
    sub = (getattr(plan, "subtext", "") or "").strip()
    syms = list(getattr(plan, "symbols", None) or [])
    if not sub and not syms:
        return ""
    return (f"裏で本当に言っていること: {sub}\n"
            + "象徴（物・言葉と、曲の中での意味の変化）: "
            + ("／".join(f"「{s.thing}」{s.change}" for s in syms) if syms else "（無し）"))


def plan_layer_facts(plan, section_names: list[str]) -> str:
    """**詞の設計の裏の層と象徴の事実**（詞の点検役の材料・2026-10-05）。裏の層の字数と『裏は無い』と答えたか・象徴の数と、
    意味の変化に名前が挙がった区間（設計の区間名が文の中に出るか）。良し悪し（素直な曲か凝った曲か）は書かない。"""
    sub = (getattr(plan, "subtext", "") or "").strip()
    syms = list(getattr(plan, "symbols", None) or [])
    if not sub and not syms:
        return ""
    rows = [f"  裏で言っていること: {len(sub)}字" + ("（『裏は無い』と答えた）" if says_no_subtext(sub) else "")]
    names = sorted({k for k in section_names if k}, key=len, reverse=True)
    for s in syms:
        seen, text = [], s.change or ""
        for k in names:                                    # 長い名前から当てる（chorus_last の中の chorus を数えない）
            if k in text:
                seen.append(k)
                text = text.replace(k, "")
        order = [k for k in section_names if k in seen]
        rows.append(f"  象徴「{s.thing}」: 意味の変化に名前が挙がった区間 "
                    + ("・".join(order) if order else "なし（区間名では書いていない）"))
    return ("【詞の設計の裏の層と象徴の事実（機械が数えたもの・良し悪しの判定ではない）】（象徴 "
            f"{len(syms)}つ）\n" + "\n".join(rows))


def chorus_facts(by: dict[int, str], melody, chorus_idx: list[int], plan, chosen=None, *, bpm: float | None = None) -> str:
    """サビの詞の**事実**（詞の点検役の材料・2026-09-30）。

    サビの詞を3案→選抜で作っていたとき、選抜の札（`_chorus_card`）が見ていた観点を、1案にしたので点検役の材料へ移した
    （規約「外す・まとめるときに観点を捨てない」）: ①決め台詞が入っているか ②各行の頭の音（あ/い/お段＋か/さ行が強い・
    う段は弱い）③伸ばす音節の母音（あ・お・え が開く）。札の「印の数が音数に合うか」は詞の工程の形式の検査（`LyricsStep`）が見る。
    良し悪しは断じない——決まりは指示書（`song_lyric_chorus.md`）にあり、意図して外すこともある。

    2026-10-03（束A・本人了承）に足したもの。どれも事実だけで、良し悪しは点検役が決める:
    - 判-6: 決め台詞が**乗った行**と、詞の設計が指定した行（以前は「サビ全体に入っているか」の ○× だけ・`hook_lines`）
    - 判-8: 詞が確定したサビ（`chosen`＝サビの詞の出力）を**書き写さなかった行**（違うときだけ）
    - 判-26: **推しメロに乗った音節**（`_oshimero`）と、詞の設計の**サビの繰り返し方針**（`repeat_policy`）

    2026-10-03（組G2）: 推しメロの ▲ を**サビの中の最高音**に（`_section_tops`）。伸ばす音節の母音は `_`（前の音節を伸ばす）で
    書いた伸ばしも**前の音節の母音**で数える（以前は `_` に音節が無いので抜けていた）。`bpm` を渡すと伸ばしの秒の下限も見る。
    """
    heads, long_v = [], []
    for i in chorus_idx:
        t = by.get(i, "")
        ss = _ls.slots(t)                                  # 旋律の1音ごとの印（2026-09-29）
        sk = _ls.sung_kana(t)
        if sk:
            h = sk[0]
            row = next((v for v, s in _ROW_OF.items() if h in s), "?")
            heads.append(f"{h}({row}段{'・か/さ行' if h in 'かきくけこさしすせそ' else ''})")
            _notes = melody.phrases[i - 1].notes
            last = ""                                      # 直前に歌った音節の最後のモーラ（`_` はこれを伸ばす）
            for n_, (s_, _c) in zip(_notes, _ls.by_note(ss, _notes)):   # 印は音ごとに分けて読む（タイの続き・判-H）
                if s_ is None:
                    break
                if s_.kind == "skip":
                    last = ""
                    continue
                if s_.kind == "sing":
                    last = s_.moras[-1] if s_.moras else ""
                mo = last                                  # `_` は前の音節の母音で数える（2026-10-03・組G2）
                if _is_hold(n_, _notes, bpm) and mo and mo[-1] not in "っん":
                    long_v.append(next((v for v, s in _ROW_OF.items() if mo[-1] in s), "?"))
    hook = getattr(plan, "hook", "") or ""
    lines = [f"【サビの詞の事実（機械が数えたもの・良し悪しの判定ではない）】（フレーズ {', '.join(map(str, chorus_idx))}）"]
    if hook:                                               # 設計は漢字まじり・詞はひらがな（歌う表記に直して比べる）
        # ★**乗った行**を出す（2026-10-03・判-6）。以前は「サビ全体に入っているか」の ○× だけで、行をまたいでいても ○ だった。
        #   詞の設計は「推しメロのある行に置く」、サビの詞は「指定の行の推しメロに置く」と頼んでいるので、指定の行と並べる
        spans = hook_lines(hook, [by.get(i, "") for i in chorus_idx])
        where = "・".join(f"{a}行目（フレーズ{chorus_idx[a - 1]}）" if a == b else
                         f"{a}〜{b}行目にまたがる（フレーズ{chorus_idx[a - 1]}〜{chorus_idx[b - 1]}）" for a, b in spans)
        lines.append(f"  決め台詞「{hook}」が乗った行: {where or 'サビの詞に入っていない'}"
                     f"／詞の設計が指定した行: {getattr(plan, 'hook_section', '')} の {getattr(plan, 'hook_line', '')}行目")
    if chosen is not None:                                 # ★確定したサビを書き写さなかった行（違うときだけ・2026-10-03・判-8）
        fixed = {p.index: p.text for p in getattr(chosen, "phrases", None) or []}
        diff = [f"フレーズ{i}: 確定『{fixed[i]}』→ 詞『{by.get(i, '')}』" for i in chorus_idx
                if i in fixed and (fixed[i] or "").split() != (by.get(i, "") or "").split()]
        if diff:
            lines.append("  確定したサビ（サビの詞）を書き写さなかった行: " + "／".join(diff))
    lines.append(f"  各行の頭の音: {' '.join(heads)}（あ/い/お段＋か/さ行が強い・う段は弱い）")
    if long_v:
        lines.append(f"  伸ばす音節の母音: {' '.join(long_v)}（あ・お・え が開く／い・う は痩せる）")
    # ★**推しメロに乗った音節**（2026-10-03・判-26）。以前は伸ばす音の母音しか見えず、最高音に何の字が乗ったかは誰も見ていなかった
    om = _oshimero(by, melody, chorus_idx, bpm)
    lines.append("  推しメロ（▲最高音・ー伸ばし）に乗った音節: "
                 + ("／".join(om) if om else "このサビには ▲最高音・ー伸ばし の音が無い"))
    # ★**サビの繰り返し方針**（詞の設計の一文・2026-10-03・判-26）。2番・最後のサビが方針どおりかを見る物差し（本文は上にある）
    if (rp := (getattr(plan, "repeat_policy", "") or "").strip()):
        lines.append(f"  サビの繰り返し方針（詞の設計）: {rp}")
    return "\n".join(lines)


# ---- 詞の記法（歌える文字・譜割り）----------------------------------------------

_KANA_OK = set("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽゃゅょっぁぃぅぇぉ")
#: **ひらがなだが書けない字**と、代わりの書き方（2026-10-01・仕様書 §2・§17 ず-16 本人決定）。モーラの数え方（`lyric_slots.moras`）も
#: 「ゎ」を小書きとして扱わない。★「ゐ」「ゑ」も同じ扱い（2026-10-02 本人「標準に寄せて1つにそろえる」。以前は「ひらがな以外」の
#: 所見になり、なぜ・どう書くかが分からなかった。歌い手は「い」「え」と同じ音で読む）
_UNWRITABLE = {"ゎ": "わ", "ゔ": "ぶ", "ゐ": "い", "ゑ": "え"}


#: **オ段のあとに『う』が来る**と、V6 は `M`（u の音）を発音してしまう。
#: 歌の表記は「きょお」「もお」「とおい」が正しい（2026-09-07 に vpr の音素で実測）。
#: ★**拗音の『ょ』も含める**（2026-09-07）。「きょう」は直前が『ょ』なので、
#: オ段のかなだけ見ていると取りこぼす（実際に素通りした）。
_O_ROW = "おこそとのほもよろをごぞどぼぽょ"


# ※促音の所見（`sokuon_findings`・2026-09-09〜）とオ段＋『う』の所見（`check_o_long`・2026-09-07〜）は
#   2026-09-27 に外した（本人判断）。促音は最近の曲で0件、9/15 以降の機械の促音直し29件は全部誤り
#   （聞いた→切った・前へと→まえっと）。オ段＋う は 9/8 以降に正しい当たり0件・最近は全部誤報（想う・目で追う）。
#   どちらも決まりは指示書（song_lyrics.md）にあり、詞の点検役が本文を読む


def check_kana(text: str) -> list[str]:
    """ひらがな以外・歌えない表記を洗い出す（プロンプトだけでは守られないため機械で見る）。

    空白（音の区切り）と譜割りの印（`_` `*` `~`）は指摘しない（数と並びは `lyric_slots.problems` が見る）。
    古い区切り記号（`・` `|` `/`）は読まなくなったので、ひらがな以外として指摘する（2026-09-30）。
    """
    # 前後の空白（どれも）は取り除いて読む（詞は日本語の欄・仕様書 §0-6・判断4＝2026-10-03 本人決定）。途中の区切りは定義の空白だけ（§2 の `sep`・2026-10-02）
    text = (text or "").strip()
    bad = sorted({c for c in text if c not in _KANA_OK and c not in _BLANK
                  and c not in (_ls.HOLD, _ls.SKIP, _ls.JOIN) and c not in _UNWRITABLE})
    odd_space = [c for c in bad if c.isspace() or c in "\u200b\u2060\ufeff"]
    bad = [c for c in bad if c not in odd_space]
    ng = [f"ひらがな以外: {' '.join(bad)}"] if bad else []
    if odd_space:
        ng.append("音の区切りの空白は半角・全角の空白・タブ・改行だけ（NBSP・細い空白・幅のない空白などは書けない）——"
                  "半角の空白で区切る")
    # ★**ひらがなだが書けない字**（2026-10-01・仕様書 §17 ず-16 本人決定。ゐ ゑ は 2026-10-02）。「ゎ」「ゔ」は以前「ひらがな以外」の所見になり、
    #   なぜ書けないか・どう書くかが分からなかった。書き方を添えて別の所見にする
    odd = [c for c in _UNWRITABLE if c in text]
    if odd:
        ng.append("「" + "」「".join(odd) + "」は書けない字——"
                  + "・".join(f"「{c}」は「{_UNWRITABLE[c]}」" for c in odd) + "で書く")
    # ★**ABC の歌詞行（`w:`）では意味のある記号**（2026-10-01・標準とのずれの点検）。`-`＝語の中の音節の区切り（次の音へ）・
    #   `|`＝小節線まで飛ばす・`\-`＝字としてのハイフン。ここでは読まない（区切りは空白だけ）ので、ひらがな以外の所見に
    #   標準との違いと書き方を添える（以前は「ひらがな以外: -」だけで、なぜ書けないかが分からなかった）
    if "-" in text or "‐" in text:
        ng.append("`-` は書かない——ABC の歌詞行では語の中の音節の区切りだが、ここでは音の区切りは空白だけ"
                  "（`か-ぜ` → `か ぜ`。伸ばすなら `_`）")
    if "|" in text:
        ng.append("`|` は書かない——ABC の歌詞行では「次の小節まで飛ばす」だが、ここでは使えない"
                  "（歌わない音は `*` を音の数だけ並べる）")
    if "を" in text:
        ng.append("助詞『を』が残っている（ボカロは wo と歌うので『お』にする）")
    return ng


#: **カバーの歌詞のかな**（譜割りの印の無い、区間の歌詞の全文）の書き方。LLM に見せる指示・型の説明に同じ文を入れる（2026-10-02・案B）
KANA_TEXT_RULE = ("**ひらがなだけ**で書く（カタカナ語・英語もひらがなで・漢字・記号・句読点・「ー」は書かない）。"
                  "**改行は元の歌詞の行の切れ目にだけ1つ**書く（空行・続けての改行は書かない）。**空白は書かない**"
                  "（行の中は続けて書く）。助詞は発音どおり（は→わ／へ→え／を→お）、"
                  "長音は母音で開く（ウ段は『う』・オ段は『お』）。「ゎ」「ゔ」「ゐ」「ゑ」は「わ」「ぶ」「い」「え」で書く")


def kana_text_problems(text: str) -> list[str]:
    """**カバーの歌詞のかな**（`align_cover` の `_SecKana`・`_HeardLyrics`・`_Lyrics` の `lyrics_kana`）の書けない書き方（2026-10-02 本人決定・案B）。

    字の範囲は詞の1フレーズ（仕様書 §2 の `kana`・`_KANA_OK`）と同じ。譜割りの印（`_` `*` `~`）と空白は書けない。
    **改行1つ＝元の歌詞の行の切れ目**（2026-10-03 本人決定）。空行・続けての改行・`\\r` は書けない。行の切れ目は割り当ての LLM に
    見せる材料（`align_reference.lyric_lines_text`）で、字を数えるときは定義された区切りとして除く（`lyric_slots.kana_text_chars`）。
    前後の空白（改行も）は取り除いて読む（§0-6）。空は「この取得元では取れなかった」（`align_cover._cover_lyrics` が次の取得元へ）なので通す。
    ★以前は読み手（割り当て・強制整列・SOFA）が空白・改行・`、。「」` を黙って落とし、それ以外の字（`！`・漢字・カタカナ）は
      そのまま音符に当てて VOCALOID の歌詞に入れていた。"""
    t = (text or "").strip()
    brk = _ls.KANA_LINE_BREAK
    bad = sorted({c for c in t if c not in _KANA_OK and c not in _BLANK
                  and c not in (_ls.HOLD, _ls.SKIP, _ls.JOIN) and c not in _UNWRITABLE})
    ng = [f"ひらがな以外: {' '.join(repr(c) if c.isspace() else c for c in bad)}"] if bad else []
    if any(c in _BLANK and c != brk for c in t):
        ng.append("空白は書かない——行の中は続けて書く（元の歌詞の行の切れ目は改行1つ）")
    if brk * 2 in t:
        ng.append("空行・続けての改行は書かない——元の歌詞の行の切れ目は改行1つ")
    marks = [c for c in (_ls.HOLD, _ls.SKIP, _ls.JOIN) if c in t]
    if marks:
        ng.append(f"譜割りの印（{' '.join(marks)}）は書かない——音への割り当ては後で別に頼む")
    odd = [c for c in _UNWRITABLE if c in t]
    if odd:
        ng.append("「" + "」「".join(odd) + "」は書けない字——"
                  + "・".join(f"「{c}」は「{_UNWRITABLE[c]}」" for c in odd) + "で書く")
    if "を" in t:
        ng.append("助詞『を』が残っている（ボカロは wo と歌うので『お』にする）")
    return ng


def kana_cell_problems(cells: list[str]) -> list[str]:
    """カバーの歌詞の割り当て（`mora_per_note`）の1音ぶんのかなに、**かなと `~` 以外の字**があれば所見（2026-10-02・案B）。
    1音ぶんの読み方は詞の1つの印と同じ（`lyric_slots.parts_of`・`CELL_RULE`）。空＝歌わない音。
    ★以前は読み手（`parts_of`）が空白・`_` `*` を黙って落とし、ほかの字はそのまま VOCALOID の歌詞にしていた。
    ★**`~` の置き方も詞と同じ決まりで見る**（片側だけ・2つ続けて・`lyric_slots.join_problem`・仕様書 §2 の文法）。以前は `parts_of` が
      空の側を黙って捨てていた（2026-10-02・組「歌詞のかなと見せる詞」の続き）。"""
    out = []
    for i, c in enumerate(cells):
        bad = sorted({ch for ch in c if ch not in _KANA_OK and ch != _ls.JOIN})
        if bad:
            out.append(f"{i}番目『{c}』: かなと `~` 以外は書かない（{' '.join(repr(ch) if ch.isspace() else ch for ch in bad)}）")
        elif w := _ls.join_problem(c):
            out.append(f"{i}番目{w}")
    return out


# ★**詞の1フレーズ（譜割りつき）の記法＝型1つ**（2026-09-29・`song_checks.Notation`）。所見は歌えない文字（`check_kana`）と
#   譜割りの書式（`lyric_slots.problems`）、読み方は `lyric_slots.slots`（旋律の1音ごとの印）——組み立て（`song_place.place_vocal` の
#   1音ごとの印・`song_resize`）も印の数の検査（`LyricsStep`）も `slots` で読む。詞の工程（`LyricsStep.written`）が宣言する
LYRIC_LINE = ck.Notation("詞の1フレーズ（譜割りつき）", lambda t: check_kana(t) + _ls.problems(t), _ls.slots)


# ※詞の所見（`inspect_lyrics`・モーラ数と音の数のずれ）は 2026-09-28 に詞の工程の形式の検査（`LyricsStep`）へ移した


def _listed(pairs: list[tuple[int, str]], cap: int = 12) -> str:
    """`フレーズ3「ご う」・フレーズ5「…」 ほか◯か所` の形（所見に並べる場所）。"""
    return ("・".join(f"フレーズ{i}「{t}」" for i, t in pairs[:cap])
            + (f" ほか{len(pairs) - cap}か所" if len(pairs) > cap else ""))


def lyric_findings(by_index: dict[int, str], disp_index: dict[int, str], melody, *,
                   ts: str | None = None, bpm: float | None = None,
                   shift_of=None, why_index: dict[int, str] | None = None) -> tuple[list[Finding], list[str]]:
    """**詞の点検役へ渡す機械の所見**（事実だけ）と、記録の warnings に積む短い文（2026-10-03・組G2 で `run_lyrics` から1か所へ）。

    どれも止めない・断じない。良し悪しと直し方は点検役が詞を読んで決める（規約「機械は数えた事実を並べる」）。
    `why_index` はフレーズ番号 → 標準外の詰めの理由（`PhraseLyric.join_reason`・2026-10-06）。
    """
    findings: list[Finding] = []
    issues: list[str] = []
    # ★**見せる歌詞と歌う歌詞を突き合わせる**（2026-09-21）。詞は「ひらがなのみ」で書くので**漢字版はどこにも存在しなかった**。
    #   両方書かせて保存する。**止めない**——伸ばして合わせるのは正当なので所見に残す。
    # ★**突き合わせの結果も点検役の材料に渡す**（2026-09-30 本人「付き合わせ結果渡して」）。
    # ★添え書きは**機械が実際に見ている2つ**だけ（2026-10-03・組G2）。以前は「語がフレーズをまたいだ疑い」と書いていたが、
    #   機械は1フレーズの中しか見ないので、両側を同じ所で切っていれば出ない——出ていない行は大丈夫と読ませていた
    _disp = lyric_display.check_all(by_index, disp_index)
    issues += _disp
    findings += [Finding(what=f"見せる歌詞と歌う歌詞の突き合わせ: {m}",
                         evidence="機械が見たのは1フレーズの中の2つだけ——表示のかな（送りがな・助詞）が歌う詞にこの順で出るか／"
                                  "表示の字数（どの字も1モーラ以上）が歌う詞のモーラ数を超えるか。漢字の読みと、"
                                  "語がフレーズをまたいだかは見ていない（［表示］を読む）")
                 for m in _disp]
    # ★**`~` の後ろが前の音節の長音に見える所**（2026-09-30・記-7）。長音なら1つの区切り（`にい`）に書く決まりだが、
    #   「言い」（`い~い`）のように別の音節のこともあり、答えが1つに決まらない（内容）ので点検役の材料にする
    _joins = [(i, t) for i in sorted(by_index) for t in _ls.long_vowel_joins(by_index[i])]
    if _joins:
        findings.append(Finding(
            what="`~` の後ろが前の音節と同じ母音（長音の書き違いの疑い）: " + _listed(_joins),
            evidence="長音（スニーカーの『にい』）は1つの区切りに書く——`~` でつなぐと後ろの母音が別の音として立ち上がる。"
                     "別の音節が同じ母音で続く語（言い＝い~い・場合＝ば~あ~い）なら長音の書き違いではない"
                     "（ただし1つの音に2音節を詰めた標準外の詰めなので、下の標準外の詰めの所見にも出る）"))
        issues.append(f"`~` の後ろが前の音節と同じ母音: {'・'.join(t for _i, t in _joins[:12])}")
    # ★**オ段の音のあとの「う」**（判-9 の残り・2026-10-03）。別の音として置いた `し ん ご う` と、1つの区切りの中の `ごう`
    #   （組G2 で足した）を並べる。9/27 に外した機械の所見は、断じて誤報ばかりだった（想う・目で追う）ので、事実だけ
    _ou = [(i, t) for i in sorted(by_index) for t in o_then_u(by_index[i])]
    _oi = [(i, t) for i in sorted(by_index) for t in o_then_u_inside(by_index[i])]
    if _ou or _oi:
        findings.append(Finding(
            what="オ段の音のあとの「う」: "
                 + "／".join(x for x in ((f"別の音として置いた {_listed(_ou)}" if _ou else ""),
                                         (f"1つの区切りの中 {_listed(_oi)}" if _oi else "")) if x),
            evidence="詞の決まりではオ段の長音は『お』で書く（信号＝しんごお）。別の音に置いた『う』は u の音で歌われる"
                     "（2026-09-07 に V6 の音素で確かめた）。1つの区切りの中の『ごう』を V6 がどう歌うかは確かめていない。"
                     "「想う・追う」のように『う』と言う語もある"))
        issues.append("オ段のあとの「う」: " + "・".join(t for _i, t in (_ou + _oi)[:12]))
    # ★**言いにくい音の置き場所**（2026-10-03・組G2・`awkward_spots`）。「っ」「ん」・母音だけの音・`_` が強拍・長い音に単独で乗った所
    _aw = awkward_spots(by_index, melody, ts, bpm, shift_of)
    if _aw:
        findings.append(Finding(
            what="「っ」「ん」・母音だけの音・`_`（伸ばし）が、1つの音に単独で強拍（小節・拍の頭）や長い音に乗った所"
                 "（単独の「っ」は場所を問わず全部・長さと高さつき）: "
                 + "・".join(_aw[:16]) + (f" ほか{len(_aw) - 16}か所" if len(_aw) > 16 else ""),
            evidence="プロソディ（言葉と旋律の一致）の目安で数えた位置（作詞の目安・意図して置くこともある）。"
                     "長い音は譜割りの『ー伸ばし』と同じ基準。`~` で詰めた音や、1つの区切りの中の「ん」（しん）は数えていない。"
                     "単独の「っ」は V6 では前の音の母音を保って終わり際を切る形で鳴る（独立した音にはならない）——"
                     "詞の標準は前の音に収める書き方（`け~っ て`）"))
        issues.append(f"言いにくい音の置き場所: {len(_aw)}か所")
    # ★**語の途中の伸ばし・小節線**（2026-10-04・本人「単語の切れ目と小節の切れ目が一致していない。特に Aメロ」・`word_breaks`）。
    #   語の切れ目は機械の推定なので候補として渡す（断じない）。推せなかった行も書く（出ていない行を大丈夫と読ませない）
    _wb, _lost = word_breaks(by_index, disp_index, melody, ts, bpm)
    if _wb or _lost:
        findings.append(Finding(
            what=("語の途中に伸ばし・小節線が入った候補（｜は小節線・( ) は伸ばしの拍）: " + "・".join(_wb[:16])
                  + (f" ほか{len(_wb) - 16}行" if len(_wb) > 16 else "") if _wb else "語の途中の伸ばし・小節線: 候補なし")
                 + (f"（語の切れ目を推せず見ていない行: フレーズ{'・'.join(map(str, _lost))}）" if _lost else ""),
            evidence="語の切れ目は機械の推定（見せる歌詞の字の種類の境目と助詞から推した。読みは推していない）——外れうる。"
                     "ひらがなだけの所と、前の母音をそのまま伸ばす長音は数えていない。"
                     "伸ばしは譜割りの『ー伸ばし』・タイ・`_` と同じ"))
        if _wb:
            issues.append(f"語の途中の伸ばし・小節線の候補: {len(_wb)}行")
    # ★**標準外の詰め**（2026-10-05・`join_frames`。2026-10-06 に標準外だけへ）。『朝光のトランク』のサビ「あ~す」が 0.15秒に2音節で
    #   「あっそう」とつぶれ、『火花になって飛べ』は詰めが42か所あった。詰めの標準は促音・撥音を前の音に収めるときだけ（本人決定）なので、
    #   ふつうの2音節の詰めを、書いた理由・枠の秒つきで並べ、短いものを先に出す（断じない。差し戻すかは点検役が理由を読んで決める）
    _js, _jo = join_frames(by_index, melody, bpm, disp_index, why_index)
    if _js or _jo:
        _cap = 16
        _jp = []
        if _js:
            _jp.append(f"短い詰め枠（1音節{JOIN_SHORT_SYL_SEC:g}秒未満か、枠{JOIN_SHORT_FRAME_SEC:g}秒未満）{len(_js)}か所: "
                         + "・".join(_js[:_cap]) + (f" ほか{len(_js) - _cap}か所" if len(_js) > _cap else ""))
        if _jo:
            _jp.append(f"そのほか {len(_jo)}か所: " + "・".join(_jo[:_cap])
                         + (f" ほか{len(_jo) - _cap}か所" if len(_jo) > _cap else ""))
        findings.append(Finding(
            what=f"標準外の詰め（ふつうの2音節以上を1つの音（枠）に詰めた所）{len(_js) + len(_jo)}か所: " + "／".join(_jp),
            evidence="詰めの標準は促音「っ」・撥音「ん」を前の音に収めるときだけ（2026-10-06 本人決定）——促音・撥音を前の音に収めた詰め"
                     "（`け~っ`・`し~ん`）と長音（`にい`）は数えていない。理由は詞の工程が join_reason に書いたもの"
                     "（理由の無い標準外の詰めは形式の検査が書き直しを求める）。"
                     "枠はその音の長さ（タイの続きを含む・音のあとの休みは含まない）。組み立てはこれを等分して1音1音節で鳴らす"
                     + (f"。秒は BPM{float(bpm):g} で換算" if bpm else "。BPM が分からないので秒にしていない（短いかも見ていない）")
                     + f"。短いの目安（1音節{JOIN_SHORT_SYL_SEC:g}秒・枠{JOIN_SHORT_FRAME_SEC:g}秒）は V6 で測った限界ではない"))
        issues.append(f"標準外の詰め: {len(_js) + len(_jo)}か所" + (f"（うち短い詰め枠 {len(_js)}か所）" if _js else ""))
    return findings, issues


def run_lyrics(st) -> None:
    """**詞の工程**（2026-09-30 に `song_compose._compose_once_body` から切り出した）。

    詞の設計（hook・置き場）→ サビの詞 → 詞（形式の検査と差分の直し）→ 点検役（毎回・redo なら詞だけ作り直す）。
    曲名はテーマの**仮の曲名**（`theme.working_title`）を依頼文で見せるだけ（この3段は曲名を書かない。最後の曲名は詞のあとの
    工程『曲名』が決める・2026-10-03。それまで組F2 ではテーマの曲名がそのまま最後の曲名だった）。
    読む: `song.theme`・`song.design`・`song.chords`・`song.groove`・`song.melody`・`st.top_c`。
    書く: `song.lyric_plan`・`song.lyric_chorus`・`song.lyrics`・`st.by_index`（フレーズ番号 → 歌う詞。組み立てが読む）。
    """
    from song_design import BREAKS, groove_number          # 区切りの強さの読み手（循環を避けてここで）
    song, verbose = st.song, st.verbose
    theme, design, chords, melody = song.theme, song.design, song.chords, song.melody
    # ② その旋律に詞を当てる
    # ★**譜割り**（最高音・伸ばしの印つき）を渡す（2026-09-13）。音数と音価だけでは
    #   「最高音に何の言葉が乗るか」を誰も決められなかった。
    listing = fu_wari(melody, top=st.top_c, bpm=getattr(design, "bpm", None),   # 伸ばしの秒の下限（2026-10-03・組G2）
                      ts=getattr(design, "time_signature", None))              # 小節線の位置（2026-10-04）

    # ---- ①詞の設計（hook・置き場・区間ごとの一文）----
    #   ★詞の工程を分解した（2026-09-13）。旋律が「土台→3案→選抜→関門」なのに詞は1回だった（旋律は 2026-09-30 に1案へ）。
    #   プロの手順（タイトル先決め → サビ＝決め台詞 → A・B → 音数の調整）に揃える。タイトルはテーマが仮の曲名として先に決めている
    #   （2026-10-03。最後の曲名は詞のあとの工程『曲名』が決める）
    _plan_prompt = (ctx.brief(theme, design, chords, melody)
                    + f"\n\n仮の曲名『{theme.working_title}』（テーマで決まった仮の曲名。この工程では書かない）\n"
                    + f"使いたい言葉: {', '.join(theme.keywords)}\n"
                    + "\n【区間と行数（グルーヴで決めたもの）】\n"
                    + "\n".join(f"・{g.kind}: {g.lines}行 × 1行 {g.moras_per_line}モーラ"
                                # 区切りは記法の読み手（`BREAKS.read`）を通して見せる（2026-10-02。以前は生の文字列を1字ずつ並べていた）
                                + (f"／区切り {' '.join(BREAKS.read(g.breaks))}" if g.breaks else "")
                                for g in song.groove.sections if g.lines)
                    + "\n\n【譜割り】\n" + listing)
    plan = ctx._call(_lyric_plan, _plan_prompt, "詞の設計")
    # ★**区間名を形式で見て、差分で直させる**（2026-09-30・出-14②／判-5）。直らなければ詞の設計の工程で止まる（`StepUnfixed`）。
    #   保存から戻した設計（再生）は検査しない（綴りだけ設計へそろえる）
    _sung = list(dict.fromkeys(p.section for p in melody.phrases))
    _lines = {k: sum(1 for p in melody.phrases if p.section == k) for k in _sung}
    plan, _ = LyricPlanStep(design.sections, _sung, _lines).run(plan, song, verbose=verbose)
    song.lyric_plan = plan
    _rec(song, "lyric_plan", plan, verbose=verbose)
    if verbose:
        print(f"  [詞の設計] hook「{plan.hook}」→ {plan.hook_section} {plan.hook_line}行目 ／ 仮の曲名『{theme.working_title}』")

    # ---- ②サビを先に書く ----
    # ★**当て推量で最初のサビに落とさない**（2026-09-30・出-14②）。以前は区間名にならない hook_section を「chorus で始まる区間」へ
    #   黙って落としていた。いまは上の検査（`LyricPlanStep`）が歌のある区間名を保証する。保存から戻した古い設計で解決できなければ、
    #   サビの詞を書かずに進み、そのことを記録に残す（黙って飛ばさない）
    _chorus_kind = ck.resolve_section(plan.hook_section, _sung)
    if _chorus_kind is None:
        from song_fix import record_change                 # noqa: E402
        record_change(song, "lyric_plan", f"hook_section『{plan.hook_section}』が歌のある区間名にならないので、サビの詞を先に書かなかった",
                      "保存から戻した詞の設計（検査の前の記録）。区間を当て推量で選ばない")
    chorus_idx = [i for i, p in enumerate(melody.phrases, 1) if p.section == _chorus_kind]
    chosen: "ChorusLyrics | None" = None
    if chorus_idx:
        _c_prompt = (ctx.brief(theme, design, chords, melody)
                     + "\n\n【詞の設計】\n"
                     + f"hook: 「{plan.hook}」／仮の曲名（テーマ）: 『{theme.working_title}』／hook の置き場: {plan.hook_section} の {plan.hook_line}行目\n"
                     + f"視点: {plan.viewpoint}\n"
                     # ★裏の層と象徴（2026-10-05）。サビ（確定）の詞にも象徴が出るように、サビの詞の依頼にも出す
                     + (f"{plan_layer_text(plan)}\n" if plan_layer_text(plan) else "")
                     + "この区間で言うこと: " + next((s.says for s in plan.sections if s.kind == _chorus_kind), "")
                     + "\n入れたい言葉: " + " / ".join(plan.phrase_list)
                     + f"\n\n【書くフレーズ（{_chorus_kind} だけ）】\n"
                     + "\n".join(listing.splitlines()[i - 1] for i in chorus_idx))
        # ★**1案で作る**（2026-09-30）。以前は3案→選抜。選抜の札が見ていた観点（決め台詞・行頭の音・伸ばす母音）は、
        #   詞の点検役の材料へ移した（下の `chorus_facts`）。サビの詞は詞の工程が書き写すので、点検役は詞の関門で読む
        chosen = ctx._call(_lyric_chorus, _c_prompt, "サビの詞")
        # ★**詞と同じ形式の検査を掛け、差分で直させる**（2026-09-30・判-7）。直らなければサビの詞の工程で止まる（`StepUnfixed`）。
        #   保存から戻したサビの詞（再生）は検査しない
        chosen, _ = ChorusLyricsStep(melody, chorus_idx, bpm=getattr(design, "bpm", None)).run(chosen, song, verbose=verbose)
        song.lyric_chorus = chosen
        _rec(song, "lyric_chorus", chosen, verbose=verbose)
    lyric_prompt = (
        ctx.brief(theme, design, chords, melody) +
        f"\n\n**曲の種類={ctx._GENRE_NAME[theme.genre]}**（目安表のこの列を軸に）\n"
        f"使いたい言葉: {', '.join(theme.keywords)}\n\n"
        f"**旋律がどんな意図で書かれたか**を踏まえて、それに合う言葉を選んでください"
        f"（切迫感を狙った旋律なら短く鋭い語を、伸びやかな旋律なら母音の開く語を）。\n\n"
        f"この旋律に詞を乗せてください:\n{listing}"
        # ★詞の設計と、確定したサビを渡す（2026-09-13）。機械の据え直しは 2026-09-27 に外した（書き写すのは詞の工程）。
        + "\n\n【詞の設計】\n"
        + f"仮の曲名（テーマ）: 『{theme.working_title}』／hook: 「{plan.hook}」（{plan.hook_section} {plan.hook_line}行目）\n"
        + f"視点: {plan.viewpoint}／サビの繰り返し: {plan.repeat_policy}\n"
        + (f"{plan_layer_text(plan)}\n" if plan_layer_text(plan) else "")   # 裏の層と象徴（2026-10-05）
        + "\n".join(f"・{s.kind}: {s.says}" for s in plan.sections)
        # ★標準外の詰めの理由（`join_reason`）も書き写させる（2026-10-06）。写し損ねると詞の形式の検査が理由を求める
        + (("\n\n【サビ（確定）——そのまま書き写す】\n"
            + "\n".join(f"フレーズ{p.index}「{p.text}」"
                        + (f"（join_reason: {p.join_reason}）" if (getattr(p, "join_reason", "") or "").strip() else "")
                        for p in chosen.phrases))
           if chosen else "")
        # ★**対句の型を伝える**（2026-09-07）。グルーヴ工程が「区間ごとに何行×何モーラ」を
        #   決めているのに、それが詞の工程へ届いていなかった。
        #   **同じ区間の行は長さが揃っている**と分かっていれば、対句として書ける。
        + (("\n\n【この曲の対句の型（グルーヴで決めたもの）】\n"
            + "\n".join(f"・{g.kind}: **{g.lines}行 × 1行 {g.moras_per_line}モーラ**"
                        # ★**区切りの強さを詞へも渡す**（2026-09-09 新設）。
                        #   旋律は「S の行を長く伸ばす」ために使い、詞は「S の行で文を終える」
                        #   ために使う。**同じ宣言を両方が読む**ので、噛み合いが運任せでなくなる。
                        + (f"／**区切り {' '.join(BREAKS.read(g.breaks))}**"
                           f"（S=ここで文を終える／M=息を入れる／C=次の行へ続ける）"
                           if g.breaks else "")
                        for g in song.groove.sections if g.lines)
            + "\n**同じ区間の行は長さが揃っています。**対句として——"
              "1行目と2行目で言い方を対応させる、同じ語を返す、問いと答えにする——"
              "書くと、聴き手が型として掴めます。")
           if getattr(song, "groove", None) else ""))

    # **関門つきで詞を作る**。機械検査は所見を出すだけで、直すかどうかは関門が決める
    # （意図した省略・伸ばしを機械で落とさないため。2026-08-22 ユーザー指示）
    words, issues, by_index, redo_note = None, [], {}, ""
    for attempt in range(MAX_REDO + 1):
        retry = "" if attempt == 0 else (
            f"\n\n★**前回の詞は作り直しになりました。**\n{redo_ask(redo_note)}\n"
            # ★**前回の詞は全部見せる**（2026-09-30・判-12）。以前は先頭12フレーズだけで、フルサイズ（30フレーズ超）の曲では
            #   後半が見えないまま書き直していた
            # ★見せる詞（［表示］）も渡す（2026-10-04 組F1・1‑5）。点検役は表示の切れ目で「語がフレーズをまたぐ」を判断するので、
            #   修正役も前回の表示を見て直す
            f"前回の詞: " + " / ".join(f"P{p.index}「{p.text}」"
                                      + (f"［表示］{p.display}" if getattr(p, "display", "") else "") for p in words.phrases))
        _before = words                                    # 前の回の稿（点検役に変わった所を並べる・組P2a）
        words = ctx._call(_lyrics_for_melody, lyric_prompt + retry, "詞")
        # 形式の検査（歌えない文字・「を」・フレーズ番号）は詞の工程に差分で直させる（2026-09-28・`LyricsStep`）。
        # ★直らなければ詞の工程で曲づくりを止める（`StepUnfixed`・2026-09-30）。以前は残った所見を点検役へ渡して進んでいた
        words, _ = LyricsStep(melody, bpm=getattr(design, "bpm", None)).run(words, song, verbose=verbose)
        # ※機械の促音直し（2026-09-12〜）・促音の所見・サビの据え直し（2026-09-13〜・詞-2）は 2026-09-27 に外した（本人判断）。
        #   サビの据え直しは本来の目的（詞の工程が書き写し損ねる）で働いた例が無く（10回中10回そのまま書き写した）、
        #   働いた1回は点検役が頼んだサビの直しを打ち消して、作り直しを空振りさせた。詞の工程と点検役が本文を読む。
        # ※詞はテーマの仮の曲名を受けて書く（「タイトルが決まっていない状態で書き始めない」・2026-10-03）。詞は曲名を書かない（判-11。
        #   以前はサビのある曲だけ、詞の書いた曲名を詞の設計の曲名で黙って上書きしていた）。最後の曲名は詞のあとの工程『曲名』が決める
        by_index = {pl.index: pl.text for pl in words.phrases}
        disp_index = {pl.index: getattr(pl, "display", "") or "" for pl in words.phrases}
        issues: list[str] = []
        findings: list[Finding] = []
        # ※差分で直させても残った形式の所見は、2026-09-30 から点検役へ渡さない（詞の工程で止まるので、ここへは来ない）
        # ★**見せる歌詞の突き合わせ・`~` の長音の疑い・オ段のあとの「う」・言いにくい音の置き場所**は `lyric_findings` の1か所（2026-10-03・組G2）。
        #   どれも事実として点検役へ渡す（止めない・断じない）
        try:                                               # 単独の「っ」の鳴る高さ（組Q2）。移調量の計算は `song_place` の1か所
            from song_place import section_keys
            _, _shift_of = section_keys(design)
        except Exception:                                  # noqa: BLE001 — 引けなければ移調前の高さで書く（材料なので止めない）
            _shift_of = None
        _f, _w = lyric_findings(by_index, disp_index, melody, ts=getattr(design, "time_signature", None) or ctx.METER,
                                bpm=getattr(design, "bpm", None), shift_of=_shift_of,
                                why_index={pl.index: getattr(pl, "join_reason", "") or "" for pl in words.phrases})   # 標準外の詰めの理由（2026-10-06）
        findings += _f
        issues += _w
        # ④ hook がサビの実体に残っているか（検査。無ければ所見に残す）
        # ★**濁点の違いを吸収して比べる**（2026-09-16）。読み変換（pykakasi）は
        #   **連濁を落とす**ことがあり、「立ち止まろう」を『たち**と**まろお』と読む。
        #   実体は正しく『たち**ど**まろお』なのに「hook がサビに入っていない」と鳴った
        #   ——**無い欠陥を追いかけるところだった**（同じ型で今日3件目）。
        if chosen and plan.hook and not hook_in_chorus(plan.hook, "".join(by_index.get(i, "") for i in chorus_idx)):
            issues.append(f"★hook「{plan.hook}」がサビの歌詞に入っていない（設計と実体の食い違い）")
            if not chorus_idx:                             # サビがあれば同じ照合の結果はサビの詞の事実（`chorus_facts`）で渡っている
                findings.append(Finding(what=f"決め台詞「{plan.hook}」がサビの歌詞に入っていない（詞の設計と実体の食い違い）"))
        rec = _rec(song, "lyrics", words, warnings=issues,
                   request=redo_note if attempt else "", verbose=verbose)
        if verbose:
            for i, p in enumerate(melody.phrases, 1):
                print(f"      P{i:<2} {_ls.slot_count(p.notes)}音 「{by_index.get(i, '')}」")
        # ★**関門に歌詞の本文を読ませる**（2026-09-16）。それまで所見しか渡しておらず、
        #   **所見がゼロなら関門を呼びもしなかった**（`if not findings: break`）。
        #   機械が測っていない観点——**言葉のまとまり**——は入口に届かず、
        #   本人に「**1番の中で歌詞の部分的なまとまりがない**」と言われた。
        #   ※渡すのは**詞の関門が読むべきものだけ**（本文＋設計）。工程の記録は 186KB あり、
        #     うち7割は旋律の枠で、詞の良し悪しには使えない（混ぜると本文が埋もれる）。
        _lines: list[str] = []
        _cur = ""
        for i, p in enumerate(melody.phrases, 1):
            if p.section != _cur:
                _cur = p.section
                _g = next((g for g in song.groove.sections if g.kind == _cur), None) \
                    if getattr(song, "groove", None) else None
                _lines.append(f"\n■{_cur}"
                              + (f"（{_g.lines}行・行末の空き"
                                 f"{groove_number(_g, 'tail_rest'):.0%}"
                                 f"・区切り={' '.join(BREAKS.read(_g.breaks or ''))}）" if _g else ""))
                if _says := next((s.says for s in (plan.sections or []) if s.kind == _cur), ""):
                    _lines.append(f"   〈狙い〉{_says}")
            # ★**見せる歌詞も関門に見せる**（2026-09-21）。語がフレーズをまたいでいないかは、
            #   読みを推定する機械より**言葉を読む LLM のほうが確かに判断できる**
            #   （辞書は `南風`→「はえ」のように外す）。機械は証拠を並べ、判定はここへ渡す。
            _d = disp_index.get(i, "")
            _lines.append(f"   {by_index.get(i, '')}" + (f"　［表示］{_d}" if _d else ""))
        _material = (f"【決め台詞】「{plan.hook}」→ {plan.hook_section} {plan.hook_line}行目"
                     f"／仮の曲名『{theme.working_title}』\n【視点】{plan.viewpoint}\n"
                     # ★裏の層と象徴と、その事実（2026-10-05）。詞に効いているか・『裏は無い』の理由が曲に合うかは点検役が読む
                     + (f"【裏の層と象徴（詞の設計）】\n{plan_layer_text(plan)}\n" if plan_layer_text(plan) else "")
                     + (f"{_lf}\n" if (_lf := plan_layer_facts(plan, [s.kind for s in design.sections])) else "")
                     + "\n".join(_lines)
                     # ★サビの詞の事実（決め台詞が入っているか・行頭の音・伸ばす母音）——サビの詞の選抜の札が見ていた観点（2026-09-30）。
                     #   2026-10-03（束A）に、決め台詞の行・書き写さなかった行・推しメロの音節・繰り返し方針を足した
                     + (("\n\n" + chorus_facts(by_index, melody, chorus_idx, plan, chosen, bpm=getattr(design, "bpm", None))) if chorus_idx else ""))
        # ★見出しと材料の曲名はテーマの仮の曲名（2026-10-03。組F2 ではテーマの曲名、判-11 では詞の設計のもの。それより前は
        #   1行目がテーマの曲名・材料が詞の設計の曲名で、2つ並んでいた）。最後の曲名は詞のあとの工程『曲名』が決める
        v = ctx.judge("詞", words.reason, rec.status, rec.trouble, findings,
                      redo_from_options=_redo_options("詞"), before=_before, after=words,
                      attempt=attempt, verbose=verbose, material=_material,
                      context=f"曲『{theme.working_title}』（仮の曲名・{ctx._GENRE_NAME[theme.genre]}）"
                              f"／狙い: {cut_note(theme.reason, 600)}")
        song.record(StepRecord(step="lyrics_gate", reason=v.reason, notes=v.notes or "", guidance=v.guidance or "",
                               status="ok" if v.action == "accept" else "compromised",
                               warnings=[f.what for f in findings], request=v.action))
        # 原因が旋律なら旋律から作り直す。退けた詞を曲へ載せてから投げる（前回の出力として渡る・2026-10-04 組F1）
        _maybe_redo_from(v, "詞", song, rejected={"lyrics": words})
        if v.action == "accept":
            break
        if v.action == "stop":
            if verbose:
                print("  [関門] ★詞が直せないので、このまま進めて記録に残す")
            break
        redo_note = v.guidance or "日本語として成立する詞に直してください"
    st.log["lyrics"] = words.reason
    st.log["lyric_issues"] = "\n".join(issues) if issues else "（なし）"
    song.lyrics = words
    st.by_index = by_index
