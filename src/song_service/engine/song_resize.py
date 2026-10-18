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

"""**曲の尺を変える**——2番を足す／短くする（TVサイズ）。

## なぜ別立てか

編集の中でも**構成そのものが変わる**ので、他の工程と性質が違う:

  ・区間の並びが変わる（増える・減る・順が変わる）
  ・**音符の位置が全部ずれる**（tick の再配置）
  ・新しい区間には**詞が要る**（2番の歌詞）

## 設計の要点: **旋律は作り直さない**

2番の旋律は1番と同じでよい（実際の曲もそう）。**コピーして tick をずらす**だけなら
音楽的な破綻が起きず、費用もかからない。**新しく要るのは詞だけ**。

短縮も同じ仕組み——**残す区間を選んで並べ直す**。旋律も詞も既にあるものを使う。

## この実装が置いている前提

| # | 前提 | 崩れると |
|---|---|---|
| 1 | `sec_of_note` が音符と同じ長さ | 区間ごとに音符を切り出せない |
| 2 | `design.sections` に区間の並びがある | 何をコピー/削除するか決められない |
| 3 | コピー元の区間は**まるごと**使う（部分コピーはしない）| フレーズが途中で切れる |
| 4 | 新しい詞は**元の区間と同じ音符数**の印（譜割り）で書く | 詞の検査が差分で直させる。直らなければ詞の工程で止まり（`StepUnfixed`）、尺は変えずに返す |
| 5 | コピー元の無い区間のコードは **C/Am 基準**で書く | 機械が区間の調へ移す（コピーした区間のコードは伴奏 MIDI から取った実際の高さ）。移さないと足した間奏だけ C で鳴る |
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_ai import Agent

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lyric_display as _display                            # noqa: E402（見せる歌詞の突き合わせ）
import lyric_slots as _ls                                    # noqa: E402（詞の譜割りの読み方・新しい曲と同じ1か所）
from song_lyrics import LyricsStep, PhraseLyric            # noqa: E402（詞の検査と直しを新しい曲と共用）
from song_place import fit_phrase                          # noqa: E402（詞を音へ当てる組み立てを新しい曲と共用）
from song_design import CHORD_BAR, CHORD_WRITING, Section, bar_ticks, chord_readable, key_shift, transpose_chord   # noqa: E402
from song_checks import JsonBool, JsonInt  # noqa: E402（LLM が書く数と真偽の欄の型・2026-10-02 判断5）
from song_checks import strip_ascii_ws  # noqa: E402（半角の記法の前後の空白・§0-6・2026-10-03）
from song_llm import pick_model                             # noqa: E402
from song_model import Song, StepRecord, StepStatus         # noqa: E402
from song_schema import ChordSpan, SongNote                 # noqa: E402
from song_schema import TICKS_PER_QUARTER as Q              # noqa: E402

#: 和音を鳴らさない小節のコード名（`song_design.NO_CHORD` の書き方の1つ）
NO_CHORD_MARK = "N.C."


# ---- 曲の形（2026-10-03・`docs/plan/song_acappella_instrumental.md` §7-2）--------------------------------
#: ★**第1段は歌もの（`song`）だけ尺を変える**。アカペラ・インストは費用をかける前に断る（黙って作らない・足さない）。
#:   アカペラ: オリジナル曲のコードは伴奏 MIDI から推定し直す経路（`add_machine_chords`）があり、伴奏 MIDI が無いと全小節が和音なしになる。
#:   インスト: 並べ直しが歌の音（`vocal_notes`）と詞を前提にしている（歌0音で落ちる・新しい区間の詞が書けない）。対応を広げるのは第2段
_FORM_NOT_YET = {
    "acappella": "アカペラは伴奏が無く、尺を変えたあとのコードを起こす材料（伴奏 MIDI）が無い",
    "instrumental": "インストは歌が無く、尺の変更（歌の音を区間ごとに並べ直し、新しい区間に詞を書く）が使えない",
}


def form_refusal(form: str) -> str:
    """その形の曲の尺を変えないなら、断る文（**尺は変えていない**ことと理由・行き先）。変えられるなら空。"""
    why = _FORM_NOT_YET.get(form or "song", "")
    if not why:
        return ""
    return (f"尺は変えなかった: {why}（いまは歌ものだけ尺を変えられる）。"
            "構成から変えたいなら remake_from_step('設計', 頼み)（設計から作り直す・以降の工程も作り直し）。"
            "その旨をマスターに伝える。")


class NewSection(BaseModel):
    """変更後の構成の1区間。**既存の区間からコピーする**（新しい旋律は作らない）。"""
    source: str = Field(description=(
        "コピー元の区間名（**いまある区間の名前をそのまま**・【いまの構成】の名前と1字も違えない）。"
        "歌の無い区間（間奏等）を新しく足すなら空"))
    kind: str = Field(description=(
        "新しい区間名。**短い識別子**（verse2 / chorus2 など・説明やかっこ書きを入れない）。"
        "**構成の中で1つずつ**（同じ区間をもう一度使うときも、名前は verse1 / verse2 のように分ける）"))
    new_lyrics: JsonBool = Field(default=False, description=(
        "**この区間に新しい詞を書くか**。2番なら true（旋律は同じ・言葉だけ変える）。"
        "1番をそのまま使うなら false"))
    intensity: JsonInt = Field(default=0, ge=0, le=5, description=(
        "**この区間の盛り上がり**（1=最も静か 〜 5=最も熱い）。コピー元のある区間は **0 ならコピー元と同じ**。"
        "**コピー元の無い区間は 1〜5 を必ず書く**（0 は書けない）。\n"
        "★**曲全体の起伏をここで設計する**——同じサビでも、1番より2番、2番よりラスサビが\n"
        "熱くなるのが普通。Cメロや落ちサビで**一度落とす**と、最後の解放が効く"))
    note: str = Field(default="", description="この区間をどうしたいか（詞を書くときの手がかり）")
    # ★**コピー元の無い区間は、小節数・コード・調を案に書く**（2026-09-29 本人判断「a」）。以前は欄が無く、
    #   機械が 4小節・コード C・調 C で置いていた（書かせていないものを機械が決めていた）
    # ★上限 32 は外した（2026-10-03・§2 c2。設計の `Section.bars` と同じく読み手に上限が無い）
    bars: JsonInt = Field(default=0, ge=0, description=(
        "**コピー元の無い区間（`source` が空）だけ**: この区間の小節数。"
        "コピー元のある区間には書かない（0 のまま。小節数はコピー元と同じになる）"))
    chords: list[str] = Field(default_factory=list, description=(
        "**コピー元の無い区間だけ**: この区間のコード（**1要素=1小節**・`bars` と同じ個数）。"
        "Cメジャー / Aマイナー基準で書く（機械が区間の調へ移す）。1小節に2つ以上なら `Am F` のように空けて並べる。"
        "和音を鳴らさない小節は `N.C.`。コピー元のある区間には書かない（空のまま。コードはコピー元を写す）。\n"
        # ★2026-10-02（組「洗い出し-型と残り」）: コードの工程と同じ書き方の文（`song_design.CHORD_WRITING`）を入れる。
        #   以前はここに書き方が無く、検査（`CHORD_BAR`）はコードの工程と同じ決まりで差し戻すのに LLM は知らなかった
        + CHORD_WRITING))
    key: str = Field(default="", description=(
        "**コピー元の無い区間だけ**: この区間の調（C / Am / Eb / F#m など）。空ならこの曲のキー。"
        "長調は何も付けない・短調は `m` だけ（`E major`・`E minor` とは書かない）。"
        "変化記号は `#` と `b` を1つだけ（記号の ♯ ♭ と、2つ重ねた Ebb・C## は書けない）。"
        "途中に空白を入れない（`A m` ではなく `Am`）。"
        "コピー元のある区間には書かない（空のまま。調はコピー元と同じ——尺の変更では転調できない）"))

    # ★**前後の空白は取り除いて読む**（2026-10-02・洗い出し H4 と中の「空白だけの source」）。仕様書 §0-6・§4「空白だけも空と同じ」。
    #   以前は型の検証だけが strip して見て、適用（`apply_resize`）は生の値を読んでいた（読み手が2本）——調 `" "` は型を通り、
    #   適用で `" " or 曲のキー` が `" "` のまま `key_shift(" ")`＝0 になり**足した区間が C で鳴った**。空白だけの `source` は型が
    #   「コピー元なし」、適用が「名前あり（無い区間）」と読んでいた。ここで1度だけ読み、適用はこの値を読む
    # ★**調は半角の記法**（2026-10-03・判断4 本人決定・§0-6）: 前後は半角の空白とタブだけ取り除き、前後の全角の空白・改行などは
    #   残して下の検証（`key_problem`）が所見にする。区間名（`source`）は設計の綴りに照らす名前なので、これまでどおり前後の空白を全部取り除く
    @field_validator("source")
    @classmethod
    def _trimmed(cls, v: str) -> str:
        return (v or "").strip()

    @field_validator("key")
    @classmethod
    def _trimmed_key(cls, v: str) -> str:
        return strip_ascii_ws(v)

    @model_validator(mode="after")
    def _new_section_is_written(self):
        """コピー元の無い区間の小節数・コード・盛り上がりが書かれているか、コピー元のある区間に書けない欄を書いていないか
        （型で差し戻す。読めないコード名・調も）。"""
        if self.source:
            # ★**コピー元のある区間の小節数・コード・調は書けない**（2026-10-02・洗い出し H6）。以前は型が見ず、適用が黙って捨てた
            #   （小節数はコピー元のまま・書いたコードと調は使われない）——LLM はラスサビの転調や小節数の変更が効いたと思っていた
            wrote = [f"`{name}`" for name, v in (("bars", self.bars), ("chords", self.chords), ("key", self.key)) if v]
            if wrote:
                raise ValueError(f"区間『{self.kind}』はコピー元『{self.source}』があるので、{'・'.join(wrote)} は書けない"
                                 "（小節数・コード・調はコピー元と同じになる。尺の変更では転調もできない）——"
                                 "消すか、コピー元を空にして歌の無い新しい区間として書く")
            return self
        from song_design import CHORD_BAR                   # noqa: E402（循環回避。コードの記法はこの型1つ・2026-09-29）
        if self.bars < 1:
            raise ValueError(f"区間『{self.kind}』はコピー元が無いので、`bars`（小節数）を書く")
        if len(self.chords) != self.bars:
            raise ValueError(f"区間『{self.kind}』の `chords` が{len(self.chords)}小節ぶん（`bars` は{self.bars}）——"
                             "1要素=1小節で、小節数と同じ個数にする")
        bad = [why for cell in self.chords for why in CHORD_BAR.problems(cell)]
        if bad:
            raise ValueError(bad[0])
        from song_design import key_problem, parse_key       # noqa: E402（読めない調は移せない＝型で差し戻す・2026-09-30）
        if self.key and parse_key(self.key) is None:
            raise ValueError(key_problem(self.key))
        # ★**コピー元の無い区間は盛り上がりも書く**（2026-10-02・洗い出しの中）。以前は 0（既定）だと機械が 2 を入れていた
        if self.intensity < 1:
            raise ValueError(f"区間『{self.kind}』はコピー元が無いので、`intensity`（1〜5）を書く"
                             "（0＝コピー元と同じ、はコピー元のある区間だけ）")
        return self


class ResizePlan(StepStatus):
    reason: str = Field(description="指示をどう解釈し、どんな構成にしたか")
    target: str = Field(description="目指す尺（**必ず書く**。例「90秒」「フルコーラス」）")
    sections: list[NewSection] = Field(description=(
        "**変更後の構成を、曲頭から順に**並べる。"
        "同じコピー元を2回使ってよい（2番＝1番の旋律をもう一度使う）が、区間名 `kind` は1つずつ（verse1 / verse2）"))

    # ★**新しい区間名の重複は型で差し戻す**（2026-10-02・洗い出し H5。設計の `song_design.Design._unique_kinds` と同じ形）。
    #   以前は見ず、区間の表（`bars_of`・`chords_of`）は後の区間が勝ち・調（`key_of`）は先の案を読み、`new_lyrics=true` の区間名が
    #   別の区間と同じだと2番の詞の書き直しが両方に及んだ
    @field_validator("sections")
    @classmethod
    def _unique_kinds(cls, v: list[NewSection]) -> list[NewSection]:
        seen: dict[str, int] = {}
        for s in v:
            seen[s.kind] = seen.get(s.kind, 0) + 1
        dup = [f"区間名『{k}』が{n}回ある" for k, n in seen.items() if n > 1]
        if dup:
            raise ValueError("／".join(dup) + "——新しい区間名 `kind` は構成の中で1つずつ"
                             "（同じコピー元をもう一度使うときは verse1 / verse2 のように別の名前にする）")
        return v


def resize_plan_type(names: list[str]) -> type[ResizePlan]:
    """**いまの区間名を知っている尺の変更の型**（2026-10-02・洗い出し H7。前例は `align_cover.corrected_type`）。
    コピー元 `source` が【いまの構成】の区間名に無ければ型で差し戻す（pydantic_ai が LLM に書き直させ、直らなければ
    `plan_resize` が `StepUnfixed`「尺の変更の構成」で止める）。照合は適用（`apply_resize`）と同じ完全一致。

    ★以前は無いコピー元の区間を、適用が警告に書いて構成から外して進んでいた（`Verse`／`verse` の違いでも外れた）。"""
    from pydantic import model_validator
    known = list(names)

    class _ResizePlanForSong(ResizePlan):
        @model_validator(mode="after")
        def _sources_exist(self):
            bad = [f"区間『{s.kind}』のコピー元『{s.source}』" for s in self.sections if s.source and s.source not in known]
            if bad:
                raise ValueError("・".join(bad) + " はいまの構成に無い——コピー元はいまある区間の名前をそのまま書く"
                                 f"（いまの区間名: {' / '.join(known) or '（無し）'}）。歌の無い区間を新しく足すならコピー元を空にする")
            return self

    _ResizePlanForSong.__name__ = "ResizePlan"              # LLM に見える型の名前は変えない
    return _ResizePlanForSong


_resizer: Agent[None, ResizePlan] = Agent(
    pick_model("RESIZE", "flash"), output_type=ResizePlan,
    system_prompt=(
        "あなたは音楽制作のディレクターです。**既にある曲の構成を組み替えて**、尺を変えます。\n"
        "\n"
        "**旋律は作り直しません。** いまある区間をコピーして並べ替えます。\n"
        "・**2番を作る** … verse → pre → chorus をもう一度並べ、`new_lyrics=true` にする（区間名 `kind` は verse2 のように1つずつ分ける）\n"
        "  （旋律は1番と同じ・**言葉だけ新しく書く**。実際の曲もそうです）\n"
        "・**短くする** … 残す区間だけを並べる（2番を落とす・間奏を削る・Cメロを飛ばす）\n"
        "・**間奏を足す** … `source` を空にして歌の無い区間を挟む。**そのときは `bars`（小節数）・`chords`（1小節ずつ・"
        "C/Am 基準）・`intensity`（1〜5）・必要なら `key` を書く**（書かないと差し戻されます。機械は小節数もコードも強さも決めません）\n"
        "  コードの書き方（コードの工程と同じ・同じ和音は1通りの綴り）:\n  " + CHORD_WRITING + "\n"
        "・コピー元のある区間には `bars`・`chords`・`key` を書かない（コピー元と同じになる。書くと差し戻されます）\n"
        "\n"
        "**尺の目安**（BPM と小節数から概算されます）:\n"
        "  90秒 … アニメの TV サイズ（**1番だけで完結させる**）\n"
        "  2分台 … 2020年代のポップス\n"
        "  3分台 … 一般的。フルコーラス（1番→2番→Cメロ→ラスサビ）\n"
        "\n"
        "★**「フルバージョン」「フルコーラス」と言われたら、2番までで終わらせないこと。**\n"
        "  日本のポップスの完全形は **1番 → 2番 → Cメロ（ブリッジ）→ ラスサビ** です。\n"
        "  ・**Cメロ**は1番2番と違う展開。**一度静かに落とす**か、逆に叫ぶ\n"
        "  ・**落ちサビ**（伴奏を薄くしたサビ）を挟むと、ラスサビの解放が際立つ\n"
        "  ・**ラスサビ**が曲の最高点。`intensity` で持ち上げる（転調は尺の変更ではできない——コピーした区間の調はコピー元のまま）\n"
        "  Cメロの旋律は既存の区間（pre や verse）を流用してよい（`source` に指定）。\n"
        "\n"
        "★**曲全体の起伏を `intensity` で設計してください。**\n"
        "  コピーしただけでは**1番と2番が同じ厚み**になり、のっぺりします。\n"
        "  よくある形: intro=1 → verse1=2 → chorus1=4 → verse2=3 → chorus2=4 →\n"
        "  **Cメロ=2（落とす）** → **ラスサビ=5（最高点）** → outro=1\n"
        "  **この形に縛られません**——静かに終わる曲、Cメロが山の曲もあります。\n"
        "  大事なのは**どこかで落として、どこかで最高点を作る**こと。\n"
        "\n"
        "**曲として成立させること。** サビで終わる・落ち着いて終わる、どちらでもよいが、\n"
        "**途中で切れた印象にしない**。短くするなら、削ったぶん繋ぎが自然かを考えてください。"
    ),
)


def _sections_of(song: Song) -> list[Section]:
    d = song.design or {}
    raw = d.get("sections") if isinstance(d, dict) else getattr(d, "sections", [])
    out = []
    for s in raw or []:
        out.append(Section.model_validate(s) if isinstance(s, dict) else s)
    return out


def describe_now(song: Song) -> str:
    """いまの構成（区間・小節・秒・音数）を LLM に見せる。"""
    score = song.score
    bar = bar_ticks(score.time_signature or "4/4")
    spb = 60.0 / score.tempo
    lines = []
    for s in _sections_of(song):
        n = sum(1 for x in song.sec_of_note if x == s.kind)
        sec = s.bars * bar / Q * spb
        lines.append(f"  {s.kind}: {s.bars}小節（{sec:.0f}秒）{n}音 強さ{s.intensity}"
                     + ("" if s.has_vocal else " **歌なし**"))
    total = max(n.start_tick + n.duration_tick for n in score.vocal_notes) / Q * spb
    return (f"【いまの構成】『{score.title}』BPM{score.tempo:.0f} 全体{total:.0f}秒\n"
            + "\n".join(lines))


def plan_resize(song: Song, request: str) -> ResizePlan:
    """構成の案（LLM）。型（`resize_plan_type`）が差し戻した形式が直らなければ `StepUnfixed`「尺の変更の構成」で止める
    （`resize_song` が尺を変えずに所見を返す・2026-10-02）。"""
    import song_llm                                         # noqa: E402
    names = [s.kind for s in _sections_of(song)]
    try:
        from song_usage import run_counted                  # noqa: E402（費用に入れる・2026-10-08）
        return run_counted(_resizer, "構成の変更",
            f"{describe_now(song)}\n\n{song.build_log()}\n\n"
            f"【マスターの指示】{request}\n\n変更後の構成を決めてください。",
            output_type=resize_plan_type(names)).output
    except Exception as e:
        if song_llm.output_unfixed(e):                     # 型の差し戻しが直らなかった＝形式の所見が直らない → 止める
            raise song_llm.to_step_unfixed(e, "尺の変更の構成", step="尺の変更の構成") from e
        raise


def melody_lines(record, score) -> tuple[list[int], int] | None:
    """**いまの楽譜の歌の音ごとに、旋律の何行目か**（1始まり・旋律の工程のフレーズの順＝見せる歌詞の行の順）と、行の数。
    分からなければ None（記録が無い・旋律か時間軸が無い・記録の楽譜といまの楽譜の音の位置が違う＝カバー・前に尺を変えた曲）。

    ★2026-10-03・再レビュー: 2番の詞は以前、フレーズを**休符（240 tick の隙間）で割り直していた**ので、旋律の1行の中の
      長い休みで行が2つに割れ・短い間で続く2行が1つになり、旋律の行と番号がずれた。見せる歌詞は旋律の1行＝1行なので、
      行の数が合わないと2番の字幕が1番のまま残った。行の頭は歌・主旋律の楽器と同じ置き方（`song_place._lay`）で旋律から引く。
    """
    if record is None or score is None:
        return None
    mel, tl, rs = getattr(record, "melody", None), getattr(record, "timeline", None), getattr(record, "score", None)
    if mel is None or tl is None or rs is None:
        return None
    if [n.start_tick for n in rs.vocal_notes] != [n.start_tick for n in score.vocal_notes]:
        return None
    try:
        from song_melody import BareMelody
        from song_place import _lay, _melody_pieces
        from song_rhythm import GRID
        phrases = (mel if hasattr(mel, "phrases") else BareMelody.model_validate(mel)).phrases
        placed, _ = _lay(phrases, tl.tick_from, lambda _kind: 0, GRID, _melody_pieces)
    except Exception:                                      # noqa: BLE001（旋律から行が起こせない＝分からない。呼び手が休符で割る）
        return None
    starts = [x[3] for x in placed if x[6]]                # 各行の頭（旋律の行の順）
    if len(starts) != len(phrases) or starts != sorted(starts):
        return None
    from bisect import bisect_right
    out = [bisect_right(starts, n.start_tick) for n in score.vocal_notes]
    if any(k < 1 for k in out):                            # 1行目の頭より前の歌の音＝旋律と楽譜が合っていない
        return None
    return out, len(phrases)


def _gap_lines(score) -> tuple[list[int], int]:
    """旋律の行が分からない曲（カバー・記録の無い曲）の行: **休符で割ったフレーズ**（`_split_phrases`）の番号と、その数。"""
    out = [0] * len(score.vocal_notes)
    groups = _split_phrases(score, range(len(score.vocal_notes)))
    for k, idxs in enumerate(groups, 1):
        for i in idxs:
            out[i] = k
    return out, len(groups)


def apply_resize(song: Song, plan: ResizePlan, *, verbose: bool = True, record=None) -> list[str]:
    """構成の変更を**機械で**適用する（旋律・詞をコピーして並べ直す）。

    戻り値は警告。`new_lyrics=True` の区間は**詞が未記入のまま**返るので、
    呼び出し側が詞工程を回すこと（この関数は LLM を呼ばない）。

    `record` は曲の工程の記録（オリジナル曲）。渡すと、歌の音ごとに**旋律の何行目か**を引いて並べ直しても持ち回る
    （`song._line_of_note`＝(新しい区間名, 元の行の番号)・2番の詞のフレーズと見せる歌詞の行に使う・2026-10-03）。
    渡さない・旋律の行が分からない曲（カバー）は、休符で割ったフレーズを行とする（`_gap_lines`）。
    """
    score = song.score
    bar = bar_ticks(score.time_signature or "4/4")
    warns: list[str] = []
    #: ★**機械が捨てた・埋めた・決めたこと**（2026-09-27・目録の道具-6）。以前は★警告の先頭3件しか返らず、
    #:   記録にも残らなかった。全部ここに集め、`dropped` として返し、曲の記録（`design_machine`）に残す。
    dropped: list[str] = []
    # 歌の音ごとの行（旋律の行。分からなければ休符で割ったフレーズ）と、元の行の数（見せる歌詞の行と照らす）
    _ml = melody_lines(record, score)
    old_line, n_old_lines = _ml if _ml is not None else _gap_lines(score)

    # 区間 → その区間の音符（相対 tick に直す）
    # ★**区間の頭は「最初の音」ではなく「最初の音がある小節の頭」**（2026-09-27）。最初の音を 0 にすると、
    #   拍の裏から入る歌（弱起・食い）が小節の頭へ動いていた（書かれた位置を機械が変える）。
    by_sec: dict[str, list[SongNote]] = {}
    idx_sec: dict[str, list[int]] = {}                     # 区間 → 元の音の番号（`by_sec` と同じ並び・行を持ち回るため）
    for i, (n, s) in enumerate(zip(score.vocal_notes, song.sec_of_note)):
        by_sec.setdefault(s, []).append(n)
        idx_sec.setdefault(s, []).append(i)
    old_bar0: dict[str, int] = {}                          # 区間 → 前の構成で始まる小節（0始まり）。伴奏の照合に使う
    for s, ns in by_sec.items():
        base = (min(x.start_tick for x in ns) // bar) * bar
        old_bar0[s] = base // bar
        by_sec[s] = [x.model_copy(update={"start_tick": x.start_tick - base}) for x in ns]
    for s, (a, _b) in (song.bars_of or {}).items():        # 歌の無い区間は区間の表から
        old_bar0.setdefault(s, a - 1)
    if score.vocal_notes:                                  # 歌い出しの前の小節（前奏）はどの区間にも入っていない
        lead = min(n.start_tick for n in score.vocal_notes) // bar
        if lead and not any(not p.source for p in plan.sections[:1]):
            dropped.append(f"歌い出しの前の {lead}小節（前奏）は新しい構成に入っていない（消えた）")
    if score.extra_voices:
        dropped.append("ハモリ・声部を全部外した（" + " / ".join(
            f"{v.title} {len(v.notes)}音" for v in score.extra_voices) + "）——位置が変わるため。要るなら付け直す")

    old = {s.kind: s for s in _sections_of(song)}
    _song_key = (d0.get("key") if isinstance((d0 := song.design), dict) else getattr(d0, "key", "")) or ""
    new_notes: list[SongNote] = []
    new_sec: list[str] = []
    new_line: list[tuple[str, int]] = []                   # 音ごとの (新しい区間名, 元の行の番号)
    new_sections: list[Section] = []
    new_chords: list[str] = []
    fresh: list[str] = []                                  # 詞を書き直す区間
    placed: list[tuple[str, str, int, int, int | None]] = []   # (新しい区間, コピー元, 新しい頭の小節0始まり, 小節数, 元の頭)
    at = 0                                                 # いまの小節位置

    from song_fix import AssemblyError
    _kinds = [ns.kind for ns in plan.sections]
    if len(set(_kinds)) != len(_kinds):                    # 型（`ResizePlan._unique_kinds`）が差し戻すので来ない。来たら不具合
        raise AssemblyError(f"尺の変更の構成に、型を通ったはずの重なった区間名が来た: {' / '.join(_kinds)}")
    for ns in plan.sections:
        src = old.get(ns.source)
        # ★**無いコピー元・盛り上がりの無い新しい区間は型が差し戻す**（2026-10-02・洗い出し H7・中）。以前は無いコピー元の区間を
        #   構成から外して進み、コピー元の無い区間の強さ 0 には機械が 2 を入れていた。来たら不具合
        if ns.source and src is None:
            raise AssemblyError(f"尺の変更の構成に、型を通ったはずの無いコピー元が来た: 区間『{ns.kind}』のコピー元『{ns.source}』")
        if not ns.source and ns.intensity < 1:
            raise AssemblyError(f"尺の変更の構成に、型を通ったはずの強さの無い新しい区間が来た: 区間『{ns.kind}』")
        bars = src.bars if src else ns.bars               # コピー元の無い区間は案に書いた小節数（型で必須）
        notes = by_sec.get(ns.source, []) if ns.source else []
        for n, j in zip(notes, idx_sec.get(ns.source, []) if ns.source else []):
            new_notes.append(n.model_copy(update={"start_tick": n.start_tick + at * bar}))
            new_sec.append(ns.kind)
            new_line.append((ns.kind, old_line[j]))
        # **抑揚は指定があればそれを使う**（0=コピー元のまま。コピー元の無い区間は型で 1〜5 が必須）。
        # コピーしただけだと1番と2番が同じ厚みになり、フルコーラスがのっぺりする
        inten = ns.intensity or src.intensity
        # 音域の狙いは尺の変更では決めない: コピー元があればそのまま、無ければ以前の型の既定 mid を明示（2026-10-04 組N2）
        new_sections.append(Section(kind=ns.kind, has_vocal=bool(notes), intensity=inten, bars=bars,
                                    pitch_center=src.pitch_center if src else "mid"))
        if src:
            # ★**コピー元のコードを1小節ずつそのまま写す**（2026-09-29・組み立ては一切補正しない）。以前はコードが無ければ
            #   全小節を C で埋め、小節より少なければ頭から繰り返していた（当て推量・繰り返しの補正）。
            #   書かれていない小節は `N.C.`（和音を鳴らさない）にし、そう記録する
            cs = list(song.chords_of.get(ns.source) or [])
            got = cs[:bars]
            if not cs:
                dropped.append(f"{ns.kind}: コピー元『{ns.source}』のコードが分からないので、{bars}小節とも"
                               f"和音を鳴らさない（{NO_CHORD_MARK}）")
            elif len(cs) < bars:
                dropped.append(f"{ns.kind}: コピー元『{ns.source}』のコードは {len(cs)}小節ぶんしか無いので、"
                               f"残りの {bars - len(cs)}小節は和音を鳴らさない（{NO_CHORD_MARK}）")
            elif len(cs) > bars:
                dropped.append(f"{ns.kind}: コピー元『{ns.source}』のコードが {len(cs)}小節ぶんあり、"
                               f"区間の {bars}小節を超える「{' '.join(cs[bars:])}」は使わなかった")
            new_chords.extend(got + [NO_CHORD_MARK] * (bars - len(got)))
        else:
            # コピー元の無い区間（間奏を足した所）は、案に書いたコードを置く（2026-09-29。以前は機械が 4小節・C）。
            # ★**C/Am 基準で書かせているので、区間の調へ移す**（2026-09-30）。コピーした区間のコード（`song.chords_of`）は
            #   実際の高さ（カバーはコードの段 `song_cover_chords`・オリジナルは工程の記録 `song_record_frame` が移したもの）なので、移さないと C 以外の曲で足した間奏だけ C で鳴っていた（説明には「機械が移す」と書いてあった）
            _sh = key_shift(ns.key or _song_key)
            for _j, cell in enumerate(ns.chords):
                names = CHORD_BAR.read(cell)
                # ★**空の要素・読めないコード名は移調量に関わらず止める**（2026-10-02・新しい曲の `song_place.place_chords` と同じ）。
                #   以前は空の要素を黙って空のコード名にし、移調しない区間では読めないコード名もそのまま置いていた（移調する区間だけ
                #   `transpose_chord` の ValueError で落ちた）。どちらも型（`NewSection` の `CHORD_BAR`）が差し戻すので来ない。来たら不具合
                bad = [c for c in names if not chord_readable(c)]
                if not names or bad:
                    from song_fix import AssemblyError
                    raise AssemblyError(f"尺の変更のコードの組み立てに、型を通ったはずのコードが来た: 区間『{ns.kind}』{_j + 1}小節目"
                                        + (f"『{'・'.join(bad)}』（読めないコード名）" if bad else f"が空（『{cell}』。和音なしは `N.C.`）"))
                new_chords.append(" ".join(transpose_chord(c, _sh, ns.key or _song_key) for c in names))   # 区間の調で綴る（2026-10-01・判-I）
        if ns.new_lyrics and notes:
            fresh.append(ns.kind)
        placed.append((ns.kind, ns.source, at, bars, old_bar0.get(ns.source) if ns.source else None))
        at += bars

    if not new_notes:
        warns.append("**音符が1つも残らなかった**（構成の指定を確認）")
        return warns + dropped

    # score を組み直す。1小節に2つ以上書いたコードは、書いた数で小節を等分する（新しい曲の組み立て `song_place` と同じ読み方）
    spans: list[ChordSpan] = []
    for i, cell in enumerate(new_chords):
        cs = CHORD_BAR.read(cell)
        # ★**空の要素・読めないコード名は止める**（2026-10-02）。以前は `or [cell]` で空のコード名 `""` を黙って置いていた（和音が鳴らない）。
        #   ここへ来るのはコピー元のコード（新しい曲は検査済み・カバーは伴奏 MIDI からの推定）・`N.C.`・上で確かめた案のコードだけで、
        #   どれも読めるコード名しか作らない。来たら不具合。※カバーの推定は和音の取れない小節に以前 `-` を書き、ここだけ「LLM が
        #   書いたものではない」として読めないまま通していた。推定が `N.C.` を書くようにして特例を外した（2026-10-02 本人「おすすめで」）
        bad = [c for c in cs if not chord_readable(c)]
        if not cs or bad:
            from song_fix import AssemblyError
            raise AssemblyError("尺の変更のコードの組み立てに、" + (f"読めないコード名『{'・'.join(bad)}』" if bad else "空の小節")
                                + f"が来た: {i + 1}小節目（『{cell}』。和音なしは `N.C.`）")
        spans += [ChordSpan(chord=c, start_tick=i * bar + h * bar // len(cs),
                            duration_tick=(h + 1) * bar // len(cs) - h * bar // len(cs)) for h, c in enumerate(cs)]
    song.score = score.model_copy(update={
        "vocal_notes": new_notes,
        "lyrics_full": "".join(n.lyric for n in new_notes),
        "extra_voices": [],                                # ハモリは付け直す（位置が変わるため）
        "chords": spans,
    })
    song.sec_of_note = new_sec
    song._line_of_note = new_line                          # 2番の詞のフレーズ・見せる歌詞の行（`write_new_lyrics`）
    song._old_line_count = n_old_lines                     # 元の行の数（見せる歌詞の行と1対1か照らす）
    bars_of, pos = {}, 0
    for s in new_sections:
        bars_of[s.kind] = (pos + 1, pos + s.bars)
        pos += s.bars
    song.bars_of = bars_of
    song.chords_of = {s.kind: new_chords[bars_of[s.kind][0] - 1:bars_of[s.kind][1]]
                      for s in new_sections}
    _key_of = {}
    for s in new_sections:
        _p = next((p for p in plan.sections if p.kind == s.kind), None)
        _src = _p.source if _p else ""
        if _src:
            if _src not in song.key_of:
                dropped.append(f"{s.kind}: コピー元『{_src}』の調が分からないので、機械が C にした")
            _key_of[s.kind] = song.key_of.get(_src, "C")
        else:
            # コピー元の無い区間は案に書いた調（空ならこの曲のキー——`NewSection.key` の説明で LLM に知らせてある）
            _key_of[s.kind] = (_p.key if _p else "") or _song_key or "C"
            if not ((_p.key if _p else "") or _song_key):
                dropped.append(f"{s.kind}: 調が書かれておらず曲のキーも分からないので、機械が C にした")
    song.key_of = _key_of
    kept = {p.source for p in plan.sections if p.source}
    # 構成から外した区間は**LLM の決定**なので機械の変更としては記録しないが、返事には載せる
    song._resize_omitted = [f"区間『{k}』は新しい構成に入れなかった（構成の案で外した）"
                            for k in old if k not in kept]
    d = dict(song.design) if isinstance(song.design, dict) else {}
    d["sections"] = [s.model_dump() for s in new_sections]
    song.design = d

    # **起伏があるか**を見る（コピーしただけだと全区間が同じ厚みになる）
    intens = [s.intensity for s in new_sections if s.has_vocal]
    if intens and (max(intens) - min(intens)) < 2:
        warns.append(f"**歌の区間の抑揚がほぼ平坦**（強さ {min(intens)}〜{max(intens)}）"
                     "——盛り上がりと静かな場所の差が付いていない")

    if verbose:
        spb = 60.0 / song.score.tempo
        total = max(n.start_tick + n.duration_tick for n in new_notes) / Q * spb
        print(f"  [構成] " + " → ".join(f"{s.kind}({s.bars}/強さ{s.intensity})"
                                        for s in new_sections)
              + f"  = {total:.0f}秒 / {len(new_notes)}音")
        for w in warns:
            print(f"      ★{w}")
        if fresh:
            print(f"  [詞] 書き直す区間: {', '.join(fresh)}")
    song.record(StepRecord(step="design", reason=plan.reason, status=plan.status,
                           trouble=plan.trouble, warnings=warns, request=plan.target))
    from song_fix import record_change
    for d in dropped:                                      # 機械がしたことは全部、記録に残す（道具-6）
        record_change(song, "design", d, "尺を変える（resize_song）で構成を機械が並べ直した")
    # ★**伴奏は新しい構成に合わせていない**（2026-09-27・監査）。楽譜の伴奏（`arrangement`）は
    #   パートごとに**前の構成の小節番号**（`bar_filter`）で鳴る所を持つので、並べ直した構成では
    #   別の区間で鳴る・鳴らない。機械は黙って合わせず、食い違う所だけを返事と記録で知らせる。
    arr_notes = arrangement_mismatch(score, song.score, placed)
    for d in arr_notes:
        record_change(song, "arrangement", d,
                      "尺を変えても伴奏の小節指定は前の構成のまま（機械は合わせていない）")
    song._resize_arrangement = arr_notes                   # 道具が返事に書く
    song._resize_placed = placed                           # カバーの伴奏 MIDI を同じ対応表で並べ替えるため（道具が使う）
    song._fresh_lyric_sections = fresh                     # 詞工程へ渡す
    song._resize_dropped = dropped                         # 道具が全部返す
    return warns + dropped


def arrangement_mismatch(old_score, new_score, placed) -> list[str]:
    """尺を変えたあと、**伴奏が新しい構成と食い違う所**（文の一覧・空なら食い違い無し）。直さない。

    `placed` は `(新しい区間名, コピー元の区間名, 新しい頭の小節（0始まり）, 小節数, コピー元の頭の小節（0始まり）|None)`。

    見るもの（2026-09-27・監査で実物を確かめた食い違い）:
      ・パートの**小節指定**（`PartSpec.bar_filter`）は前の構成の**小節番号**。区間名では持っていない
        （`song_arrange.to_spec` が区間名を小節番号へ直して保存する）ので、並べ替えると
        「コピー元の区間では鳴っていたのに、新しい位置では鳴らない／その逆」が起きる
      ・人が手で書いた小節（`hand_bars`）と音ごとの奏法（`articulations`）も、前の構成の小節・tick のまま
    ※伴奏が楽譜に無い曲（カバー＝伴奏 MIDI）は見ない（空を返す）。そちらは `rearrange_accomp` が
      音符と同じ対応表で伴奏 MIDI を並べ替え、捨てた・切ったことを返す（`resize_song` が使う）。
    """
    from song_audio import bars_of_filter, song_bars, to_bar_filter
    arr = old_score.arrangement
    if arr.is_empty():                                     # 伴奏が楽譜に無い（パートが0本・F1）
        return []
    old_n, new_n = song_bars(old_score), song_bars(new_score)
    # 新しい小節（1始まり）→ (区間, コピー元, コピー元で対応する前の小節（1始まり）|None)
    where: dict[int, tuple[str, str, int | None]] = {}
    for kind, src, at0, bars, old0 in placed:
        for j in range(bars):
            where[at0 + j + 1] = (kind, src, (old0 + j + 1) if old0 is not None else None)
    out: list[str] = []
    for i, p in enumerate(arr.parts, 1):
        before = bars_of_filter(p.bar_filter, old_n)       # 前の構成で鳴っていた小節
        now = bars_of_filter(p.bar_filter, new_n)          # 同じ指定が新しい構成で鳴らす小節
        silent: dict[tuple[str, str], set[int]] = {}       # コピー元では鳴っていたのに鳴らない
        extra: dict[tuple[str, str], set[int]] = {}        # コピー元では鳴っていなかったのに鳴る
        for b in range(1, new_n + 1):
            kind, src, ob = where.get(b, ("（区間の外）", "", None))
            was = ob is not None and ob in before
            if was and b not in now:
                silent.setdefault((kind, src), set()).add(b)
            elif b in now and not was:
                extra.setdefault((kind, src), set()).add(b)
        if not (silent or extra):
            continue
        name = f"伴奏のパート{i}『{p.role}{('・' + p.timbre) if p.timbre else ''}』（小節指定 {p.bar_filter}）"
        bits = [f"{k} の {to_bar_filter(bs)}小節で鳴らない（コピー元『{s}』では鳴っていた）"
                for (k, s), bs in silent.items()]
        bits += [f"{k} の {to_bar_filter(bs)}小節で鳴る（"
                 + (f"コピー元『{s}』では鳴っていなかった" if s else "コピー元の無い区間") + "）"
                 for (k, s), bs in extra.items()]
        out.append(name + ": " + "／".join(bits))
    def _same(b: int) -> bool:                             # 新しい b 小節目が、前の b 小節目と同じ中身か
        w = where.get(b)
        return w is not None and w[2] == b

    hb = [h for h in (getattr(arr, "hand_bars", None) or [])
          if not all(_same(b) for b in range(int(h.get("start_bar", 0)) + 1, int(h.get("end_bar", 0)) + 1))]
    if hb:
        out.append(f"人が手で書いた伴奏の小節（{len(hb)}件: "
                   + "、".join(f"{h.get('role')} {int(h.get('start_bar', 0)) + 1}〜{int(h.get('end_bar', 0))}小節"
                              for h in hb[:6]) + (" ほか" if len(hb) > 6 else "")
                   + "）は、前の構成の小節番号のまま鳴る——並べ替えたので、その小節には別の区間が来ている")
    bt = bar_ticks(new_score.time_signature or "4/4")
    ar = [a for a in (getattr(arr, "articulations", None) or []) if not _same(a.tick // bt + 1)]
    if ar:
        out.append(f"伴奏の音ごとの奏法（{len(ar)}件: "
                   + "、".join(f"パート{a.part + 1}の{a.tick // bt + 1}小節目 {a.name}" for a in ar[:6])
                   + (" ほか" if len(ar) > 6 else "")
                   + "）は、前の構成の位置（tick）のまま掛かる——並べ替えた所では別の音に掛かるか、音が無くて掛からない")
    return out


def rearrange_accomp(tracks: list, tempo: float, bar: int, placed) -> tuple[list, list[str]]:
    """**カバーの伴奏 MIDI を、音符と同じ対応表で並べ替える**（2026-09-27・監査。組み立て）。

    `tracks` は `accomp_edit.load_accomp` の結果（トラック別の音・絶対秒）。伴奏 MIDI の 0秒は楽譜の tick 0 と同じ
    （カバーのコードの材料＝原曲の伴奏 MIDI からの推定 `song_cover_chords.midi_guesses` もこの前提で小節を切る。実物の『シャイニングスター』で伴奏 0.02〜31.15秒・
    歌 0.00〜31.22秒と揃っていることを確かめた）。`placed` は `apply_resize` が作った対応表
    `(新しい区間, コピー元, 新しい頭の小節（0始まり）, 小節数, コピー元の頭の小節（0始まり）|None)`。

    ・コピー元の区間の小節に**始まる音**を、新しい区間の位置へ丸ごと写す（歌の音符と同じ——区間の途中で切らない）
    ・区間の終わりを越えて伸びる音は、区間の終わりで切る（次の区間には別の中身が来るため）
    ・どの区間にも写らなかった音（前奏・後奏・構成から外した区間）は捨てる
    ・コピー元の無い区間（歌の無い区間を新しく足した所）は伴奏も鳴らない
    **捨てた・切った・鳴らない所は全部、文にして返す**（呼び側が返事と記録に出す）。戻り値は (新しいトラック, 文)。
    """
    import copy

    spb = 60.0 / float(tempo)
    bar_sec = bar / Q * spb
    tol = spb / 8                                          # 32分音符ぶん早く弾かれた音は、次の小節の頭の音とみなす
    msgs: list[str] = []
    used: set[int] = set()                                 # 写した前の小節（0始まり）
    cut = 0
    out = []
    for tr in tracks:
        t2 = copy.copy(tr)
        t2.notes = []
        out.append(t2)
    for kind, _src, at0, bars, old0 in placed:
        if old0 is None:
            msgs.append(f"{kind}（{at0 + 1}〜{at0 + bars}小節）: コピー元の無い区間なので、伴奏も鳴らない（{bars}小節の無音）")
            continue
        used.update(range(old0, old0 + bars))
        w0, w1 = old0 * bar_sec, (old0 + bars) * bar_sec
        shift = (at0 - old0) * bar_sec
        for tr, t2 in zip(tracks, out):
            for n in tr.notes:
                if not (w0 - tol <= n.start_sec < w1 - tol):
                    continue
                end = min(n.start_sec + n.dur_sec, w1)
                if end < n.start_sec + n.dur_sec - 1e-6:
                    cut += 1
                start = max(0.0, n.start_sec + shift)
                t2.notes.append(type(n)(n.key, start, max(0.03, end + shift - start), n.vel))
    for t2 in out:
        t2.notes.sort(key=lambda n: n.start_sec)
    # どの区間にも写らなかった音（前の小節ごとにまとめる）
    lost: dict[int, int] = {}
    for tr in tracks:
        for n in tr.notes:
            b = int((n.start_sec + tol) // bar_sec)
            if b not in used:
                lost[b] = lost.get(b, 0) + 1
    if lost:
        from song_audio import to_bar_filter
        msgs.append(f"前の構成の {to_bar_filter({b + 1 for b in lost})}小節の伴奏（{sum(lost.values())}音）は"
                    "新しい構成のどの区間にも入っていない（捨てた。前奏・後奏・構成から外した区間）")
    if cut:
        msgs.append(f"区間の終わりを越えて伸びていた伴奏の音 {cut}音を、区間の終わりで切った（次に別の区間が来るため）")
    return out, msgs


# ---- 新しい区間の詞を書く -------------------------------------------------------

#: 2番の詞の1フレーズは**新しい曲の詞と同じ形**（歌う詞＝譜割りつき・見せる詞）。検査と直し（`LyricsStep`）も同じものを使う
NewPhrase = PhraseLyric


class NewLyrics(StepStatus):
    reason: str = Field(description="2番（または新しい区間）で何を言うか。**1番との違い**を書く")
    phrases: list[PhraseLyric]


#: 2番の詞の LLM 呼び出しのラベル（`song_llm.run_sync`。工程名は「詞」＝頼まれごと・止める工程・`song_context.step_of_label`）
NEW_LYRICS_LABEL = "詞(2番)"

_lyricist = Agent(
    pick_model("LYRIC", "flash"), output_type=NewLyrics,
    system_prompt=(
        "あなたは作詞家です。**1番と同じ旋律**に、**2番の歌詞**を書きます。\n"
        "\n"
        "★**1番の繰り返しにしない。** 曲の心の動き（arc）に沿って、\n"
        "  **語り手を先へ進めてください**。同じ情景を言い換えるだけでは 2番の意味がありません。\n"
        "  ・1番が「気づいていなかった」なら、2番は「気づく」\n"
        "  ・1番が「情景」なら、2番は「その中にいる自分」\n"
        "  ・**季節・時間・距離**を動かすと変化が伝わる（夕方→夜、ぬるい風→冷たい風）\n"
        "\n"
        "**歌詞は2つ書きます**: `text`＝歌う歌詞（ひらがな・譜割りつき）／`display`＝見せる歌詞"
        "（漢字・カタカナ・英字で読める形。字幕・投稿で使う。譜割りの印は入れない）。\n"
        "  `text`『よ か ぜ に ま か せ て』→ `display`『夜風に任せて』。"
        "**display の読みが text と一致すること**（表示を削って詰めない）。"
        "**display は空にしない**——漢字の無い言葉ならかなのまま書く（`text`『ら ら ら』→ `display`『ららら』。"
        "空にしてよいのは印が全部 `*` の、歌う音の無いフレーズだけ）。\n"
        "\n"
        # ↓ 譜割りの書き方は新しい曲の詞の指示書（prompts/song_lyrics.md「譜割りの書き方」）と同じ内容（2026-09-29）
        "**譜割りの書き方**（ABC 記法の歌詞行と同じ）——`text` は**音ごとに空白で区切って**書きます。"
        "**1つの区切り＝旋律の1音**です。\n"
        "・`さ` … その音で1モーラ歌う（`そ ら に`＝3音）\n"
        "・`にい` … **長音を1つの音で**歌う（前の母音のまま伸ばす。区切らずに続けて書く。`す にい かあ`＝3音）\n"
        "・`け~っ` … **促音・撥音を前の音に収める**（音を等分して、それぞれ歌う。`け~っ て`＝「蹴って」2音・`ほ~ん と`＝「ほんと」2音）\n"
        "・★**詰めの標準は、促音「っ」・撥音「ん」を前の音に収めるときだけ**（2026-10-06）。ふつうの2音節（`あ~す`＝明日）を1つの音に"
        "詰めない——詰めた音は等分されて言葉がつぶれる。言い回しを変える・短い語にして1音1音節に収める。"
        "どうしても詰めるなら、そのフレーズの `join_reason` に、どの詰めを・なぜ詰めるかを書く（書かずに詰めると書き直しを頼まれる）\n"
        "・`_` … **前の音節をもう1音伸ばす**（`そ ら _ _ に`＝5音）。歌った音のあとにだけ置ける（フレーズの頭に置かない）\n"
        "・`*` … **その音は歌わない**（`き み * の な ま え`＝7音）\n"
        "・**タイでつないだ音**（音価に `(+タイ1)` と付いた音）は、つないだ音の数だけ印を書き、**続きの音は `_`**"
        "（`か _`＝「か」を2つの音にわたって伸ばす。ABC の歌詞行と同じ数え方。音の数にも入っています）\n"
        "・**区切りの数が、そのフレーズの音符数とちょうど同じ**であること。機械は書かれたとおりに置くだけで、"
        "数が合わなければ、**余った区切りは歌われず、足りない音は鳴りません**（後から埋めたり詰めたりしません）\n"
        "・長音には `~` を付けない。1つの区切りの中で前の母音のまま伸ばすのは、同じ母音が続く長音（にい・かあ）と"
        "オ段＋う（きょう）だけ。それ以外（`せい`・`しん`）を1つの区切りに入れると、音を等分して歌い分けます\n"
        "・1つの区切りに入れるのは**3モーラまで**\n"
        "・「っ」も1つの区切りとして書く（`ま っ て`）。前の音を保って終わり際を切って歌うので、"
        "**フレーズの頭と、`*`（歌わない音）の直後には置けない**（`* っ て` は保つ音が無い）\n"
        "・拗音は前の字と同じ区切りに書く（`きょ`。`き ょ` と分けると「ょ」が1つの音になる）\n"
        "・ABC で使う `-`（語の中の音節の区切り）と `|`（小節まで飛ばす）は**使わない**——区切りは空白だけ、歌わない音は `*`\n"
        "\n"
        "**表記の決まり（守らないと歌えません）**——以下は `text` の話です:\n"
        "・**ひらがなだけ**。漢字とアルファベットは歌えない（音素が付かない）。**カタカナも使わない**"
        "（譜割りはひらがなで読む。外来語は `display`『サイダー』・`text`『さ い だあ』）。記号は譜割りの印（空白・`~` `_` `*`）だけ\n"
        "・「ゎ」「ゔ」「ゐ」「ゑ」は書かない（「わ」「ぶ」「い」「え」で書く）\n"
        "・拗音「きゃ」は2文字で1モーラ（`きゃ` で1つの区切り）。「ん」「っ」は1モーラ\n"
        "・長音「ー」と単独の小書き母音は使わない。母音で開く——ウ段は『う』（ゆうき）、**オ段は『お』**（きょお・とおい）\n"
        "・促音『っ』は『っ』のまま書く（○ たちどまって × たちどまあて）\n"
        "・★**発音どおりに書く**: 助詞『は』→『わ』、『へ』→『え』、『を』→『お』\n"
        "\n"
        "各フレーズの**音符数と音価**が示されます。\n"
        "★**数を合わせるために日本語を壊さないでください**——音符が余れば**伸ばす**（`_`）、"
        "足りなければ**言い回しを変える・短い語にする**（ふつうの2音節は詰めない）。**助詞で終わらせない**。\n"
        "\n"
        "**サビは1番と同じ詞でよい**場合が多い（そう指示されたら変えない）。"
    ),
)


def _split_phrases(score, indices) -> list[list[int]]:
    """音符の番号を**休符でフレーズに割る**（間が16分2つぶん以上空いたら切る／飛んでいても切る）。

    ★**見せる歌詞の行と揃えるために切り出した**（2026-09-21）。同じ規則で2回割らないと、
    書き直した区間がどの行に当たるか分からない。
    ★2026-10-03 から、行は**旋律の行**（`melody_lines`）で決める。ここは旋律の行が分からない曲（カバー・記録の無い曲）の行
    （`_gap_lines`）にだけ使う——カバーの旋律は原曲の採譜で、行の区切りは休符しか無い（`song_cover_load.phrases_of` と同じ考え）。
    """
    out: list[list[int]] = []
    cur: list[int] = []
    prev = prev_i = None
    for i in indices:
        n = score.vocal_notes[i]
        if prev is not None and (n.start_tick - prev >= 240 or i != prev_i + 1) and cur:
            out.append(cur)
            cur = []
        cur.append(i)
        prev, prev_i = n.start_tick + n.duration_tick, i
    if cur:
        out.append(cur)
    return out


def _group_lines(keys: list, indices) -> list[list[int]]:
    """音の番号を**行ごとに**まとめる（`keys[i]`＝その音の行・並びは曲の順）。行が変わるか、番号が飛んだら切る。"""
    out: list[list[int]] = []
    cur: list[int] = []
    prev = None
    for i in indices:
        if cur and (keys[i] != keys[prev] or i != prev + 1):
            out.append(cur)
            cur = []
        cur.append(i)
        prev = i
    if cur:
        out.append(cur)
    return out


def _beat_cell(n) -> str:
    """2番の依頼文の音価の欄。タイでまとめた音は `2(+タイ1)`——印はタイの続きの音のぶんも書く（続きは `_`・判-H）。"""
    t = _ls.ties_of(n)
    return f"{n.duration_tick / Q:g}" + (f"(+タイ{t})" if t else "")


def write_new_lyrics(song: Song, *, verbose: bool = True) -> StepRecord | None:
    """`apply_resize` が印を付けた区間に**新しい詞**を書く（旋律はそのまま）。"""
    fresh = getattr(song, "_fresh_lyric_sections", [])
    if not fresh:
        return None
    score = song.score

    # ★**対象区間の音符を、旋律の行でフレーズに割る**（2026-10-03・再レビュー）。行は `apply_resize` が持ち回った
    #   (区間名, 元の行の番号)。以前は休符（240 tick の隙間）で割り直していて、旋律の行と番号がずれた。
    #   `apply_resize` を通っていない呼び方（行が無い）だけ、休符で割ったフレーズを行とする（カバーと同じ）
    keys = list(getattr(song, "_line_of_note", None) or [])
    n_old_lines = getattr(song, "_old_line_count", None)
    if len(keys) != len(score.vocal_notes):
        _gl, n_old_lines = _gap_lines(score)
        keys = [(None, k) for k in _gl]
    targets = [i for i, s in enumerate(song.sec_of_note) if s in fresh]
    phrases = _group_lines(keys, targets)
    _all = _group_lines(keys, range(len(score.vocal_notes)))   # 曲全体の行（見せる歌詞を組み直す・依頼文に見せる）

    # タイでまとめた音は印を 1＋ties 個（続きの音は `_`）。音価の欄は `2(+タイ1)` のように見せる（2026-10-01・判-H）
    # ★1番の詞の伸ばしは、LLM が書く伸ばしの印 `_` で見せる（2026-10-01・仕様書 §16「伸ばしの印は `_`」）。以前は楽譜の歌詞の `-`
    #   （V6 の伸ばし）のまま見せていて、詞の記法では書けない `-`（§2）を見本にしていた。表と同じ `song_table._hold_shown`
    from song_table import _hold_shown                      # noqa: E402（循環を避けてここで）
    listing = "\n".join(
        f"  フレーズ{k}（{song.sec_of_note[idxs[0]]}）: **{_ls.slot_count([score.vocal_notes[j] for j in idxs])}音**"
        f"／音価（拍） {' '.join(_beat_cell(score.vocal_notes[j]) for j in idxs)}"
        f"／1番の詞（音ごと・『_』は前の音を伸ばした音とタイの続きの音）"
        f"「{' '.join(_hold_shown(score.vocal_notes[j].lyric) + ' _' * _ls.ties_of(score.vocal_notes[j]) for j in idxs)}」"
        for k, idxs in enumerate(phrases, 1))
    # ★**いまの歌詞は切らずに、伸ばしは `_` で、旋律の1行ずつ見せる**（2026-10-03・再レビュー）。以前は `lyrics_full[:200]`
    #   （楽譜の伸ばし `-` 入り・200字で切れる）で、詞の記法では書けない `-` を見本にし、長い曲では後半が見えなかった
    sung_now = "\n".join(
        f"  {song.sec_of_note[g[0]]}: {''.join(_hold_shown(score.vocal_notes[j].lyric) for j in g)}"
        for g in _all if song.sec_of_note[g[0]] not in fresh)
    import song_llm                                         # noqa: E402
    from song_usage import record as record_usage           # noqa: E402
    # ★**ラベル付きの共通の入口（`song_llm.run_sync`）を通す**（2026-10-03・再レビュー）。以前は `_lyricist.run_sync` を直に呼び、
    #   書き方の決まり（`EDGE_RULE`）・頼まれごと・曲の形の見出しが付かず、費用（`song_usage`）と依頼文の記録にも残らなかった。
    #   型の差し戻し（見せる詞が空・2026-10-02 案2 ほか）が直らなければ、入口が詞の工程で止める（`StepUnfixed`）
    r = song_llm.run_sync(
        _lyricist,
        f"{song.build_log()}\n\n"
        + (f"【いまの歌詞】（書き直す区間の外・1行＝旋律の1行・『_』は前の音を伸ばした音）\n{sung_now}\n\n" if sung_now else "")
        + f"【書き直す区間】{', '.join(fresh)}\n{listing}\n\n"
        "この旋律に、**1番から先へ進んだ**詞を書いてください。",
        label=NEW_LYRICS_LABEL, stop_step="詞")
    record_usage(_lyricist, r, NEW_LYRICS_LABEL)
    out = r.output
    # ★**新しい曲の詞と同じ検査で、差分で直させる**（2026-09-29）。印の数と音の数のずれ・譜割りの書式・歌えない文字・
    #   フレーズ番号の誤り。以前は機械が足りない音符を「ー」で埋め、余った言葉を捨てていた（組み立ては一切補正しない）
    #   直らなければ詞の工程で止まる（`song_fix.StepUnfixed`・2026-09-30）ので、ここへ来るのは所見の無い詞だけ
    out, _ = LyricsStep(_PhraseCounts(phrases, score.vocal_notes)).run(out, song, verbose=verbose)

    by_index = {p.index: p.text for p in out.phrases}
    disp_index = {p.index: getattr(p, "display", "") or "" for p in out.phrases}
    warns: list[str] = [f"P{k}: {m}" for k, m in
              ((k, _display.check_phrase(by_index.get(k, ""), disp_index.get(k, "")))
               for k in sorted(by_index)) if m]
    # 書き直す区間の音符を、印のとおりに置き直す（詰め＝等分・`_`＝「-」・`*`＝鳴らさない）。区間の対応（`sec_of_note`）も作り直す
    head_of = {idxs[0]: k for k, idxs in enumerate(phrases, 1)}
    rewritten = {i for idxs in phrases for i in idxs}
    notes: list[SongNote] = []
    sec_of: list[str] = []
    line_of: list = []                                     # 音ごとの行（`sec_of` と同じく置き直した音に合わせる）
    fit_log: list[str] = []
    for i, n in enumerate(score.vocal_notes):
        if i in head_of:
            k = head_of[i]
            got = place_phrase([score.vocal_notes[j] for j in phrases[k - 1]], by_index.get(k, ""),
                               fit_log, f"フレーズ{k}（{song.sec_of_note[i]}）: ")
            notes += got
            sec_of += [song.sec_of_note[i]] * len(got)
            line_of += [keys[i]] * len(got)
        elif i not in rewritten:
            notes.append(n)
            sec_of.append(song.sec_of_note[i])
            line_of.append(keys[i])
    if fit_log:
        # ★**来たら不具合**（2026-09-30）。印の数と音の数は詞の検査が保証していて、直らなければ詞の工程で止まる。
        #   新しい曲の組み立て（`song_place.place_vocal`）と同じく、合う分だけ置いて黙って進まない
        from song_fix import AssemblyError
        raise AssemblyError("2番の詞と音の数が合わない（詞の検査を通ったはず）: " + "／".join(fit_log))
    # ★**見せる歌詞も書き直す**（2026-09-21）。歌う歌詞だけ差し替えると、字幕と投稿が
    #   **1番のまま**になる（派生物は正を直しても黙って古いまま残る）。
    # ★**行は旋律の行で組み直す**（2026-10-03・再レビュー）。見せる歌詞は元の曲の旋律の1行＝1行なので、新しい構成の行を
    #   曲の順に並べ、書き直した行は新しい見せる詞、コピーしただけの行は元の行（`(区間名, 元の行の番号)` の番号）を置く。
    #   以前は休符で割ったフレーズの数と元の見せる歌詞の行数を比べていて、2番を足すと（行が増えるので）ほぼ必ず合わず、
    #   字幕は1番のまま残っていた。元の見せる歌詞の行数が元の行の数と違う曲（並びが分からない）だけ、黙って捨てずに警告する
    _old = (score.lyrics_display or "").splitlines()
    if _old and n_old_lines == len(_old):
        # ★**見せる詞は書いたとおり**（2026-10-02 本人決定・案2）。空は詞の型（`PhraseLyric`）が差し戻す。歌う音の無いフレーズ
        #   （印が全部 `*`）だけ空の行——1番の行を残すと、2番の所に1番の字幕が出る。
        #   以前は見せる詞の無いフレーズを歌う詞（印を外したかな）で埋め、空の行は1番の行のまま残していた
        _new_line = {tuple(g): disp_index.get(k, "") for k, g in enumerate(phrases, 1)}
        _lines = [(_new_line[tuple(g)] if tuple(g) in _new_line else _old[keys[g[0]][1] - 1]) for g in _all]
        _disp = "\n".join(_lines)
    else:                                                  # 元の行と見せる歌詞が1対1でない＝並びが分からない。黙って捨てない
        _disp = score.lyrics_display
        if _old:
            warns.append(f"見せる歌詞の行数（{len(_old)}）と元の曲の旋律の行の数（{n_old_lines}）が合わず、"
                         f"書き直した区間を反映できなかった（字幕は1番のまま）")
    song.score = score.model_copy(update={
        "vocal_notes": notes, "lyrics_full": "".join(n.lyric for n in notes),
        "lyrics_display": _disp})
    song.sec_of_note = sec_of                              # 音符と同じ長さ（詰めで増え・`*` と詞の無い音で減る）
    song._line_of_note = line_of

    rec = StepRecord(step="lyrics", reason=out.reason, status=out.status,
                     trouble=out.trouble, warnings=warns)
    song.record(rec)
    if verbose:
        print(f"  [詞] {out.reason[:130]}")
        for k, idxs in enumerate(phrases, 1):
            print(f"      P{k:<2} {len(idxs)}音 「{by_index.get(k, '')}」")
        for w in warns:
            print(f"      ★{w}")
    return rec


class _PhraseCounts:
    """詞の検査（`song_lyrics.LyricsStep`）へ渡す**旋律の形**。検査が読むのは `phrases[i].notes` の数だけ。

    2番の旋律は1番の音符のコピーなので、旋律の行で割ったフレーズ（`_group_lines`）の音符をそのまま並べる。
    """
    def __init__(self, phrases: list[list[int]], notes: list | None = None):
        # タイでまとめた音（`SongNote.ties`）は印を 1＋ties 個数える（2026-10-01・判-H）。検査は音の `ties` を読む
        self.phrases = [SimpleNamespace(notes=[notes[i] if notes is not None else SimpleNamespace(ties=0) for i in idxs])
                        for idxs in phrases]


@dataclass
class _Frame:
    """`song_place.fit_phrase` が読む1音の形（`BareNote` の長さの欄だけ）。高さは使わない（置いたあと元の音符から取る）。"""
    beats: float                                           # 鳴る長さ（拍）
    gap_after: float                                       # 次の音までの休み（拍）
    solfa: str = ""
    octave: int = 0
    ties: int = 0                                          # タイでまとめた続きの音の数（詞の印は 1＋これ・2026-10-01 判-H）


def place_phrase(src: list[SongNote], text: str, log: list[str] | None = None, label: str = "") -> list[SongNote]:
    """2番の1フレーズの詞を、**コピーした音符へ印のとおりに置く**（2026-09-29）。

    組み立ては新しい曲と同じ1か所（`song_place.fit_phrase`）を通す——2本あると片方だけ直る:
      ・`~` で詰めた別々の音節（`い~い`）… その音を**等分**して1音1モーラにする（V6 は1音に2モーラを歌えない）
      ・`~` の無い区切りの長音（`にい`）… 前のモーラのまま1音で伸ばす
      ・`_` … 歌詞「-」（前の音を再発音せずに伸ばす）／「っ」… 前の音を保って終わり際を切る（`_sustain_sokuon`）
      ・`*` … その音を置かない（**時間は変わらない**。後ろの音は元の位置のまま）
      ・印と音の数が合わなければ、合う分だけ置く——**詞の無い音は鳴らさず、音の無い印は歌わない**（`log` に書く）
    音の高さ・調声はコピー元の音符のまま。詰めて分けた2つ目以降の音には、音の頭に付けた奏法（`artic`）を写さない。
    """
    if not src:
        return []
    frames: list[_Frame] = []
    for j, n in enumerate(src):
        nxt = src[j + 1].start_tick if j + 1 < len(src) else None
        sound = n.duration_tick if nxt is None else max(1, min(n.duration_tick, nxt - n.start_tick))
        gap = 0 if nxt is None else max(0, nxt - (n.start_tick + sound))
        frames.append(_Frame(beats=sound / Q, gap_after=gap / Q, ties=int(getattr(n, "ties", 0) or 0)))
    # (歌詞, _, _, 枠, 鳴る長さ, 音の中の何番目か, タイの続きの数)。タイの続きの `_` は1番と同じく前の音に含める（判-H）
    pieces = fit_phrase(_ls.slots(text), frames, log, label)
    placed: list[tuple[str, int, int, int, int, int]] = []    # (歌詞, 元の音の番号, 何番目か, 位置, 長さ, タイの続きの数)
    j, at = -1, 0.0
    for lyric, _s, _o, frame, sound, sub, ties in pieces:
        if sub == 0:                                       # 1つの音の頭（分けた音は sub≥1）
            j += 1
        placed.append((lyric, j, sub, src[0].start_tick + int(round(at * Q)), max(1, int(round(sound * Q))), ties))
        at += frame
    out: list[SongNote] = []
    for k, (lyric, j, sub, st, du, ties) in enumerate(placed):
        if not lyric:                                      # `*`（歌わない）
            continue
        if k + 1 < len(placed):                            # 次の音に食い込まない（丸めの1tick）
            du = max(1, min(du, placed[k + 1][3] - st))
        upd: dict = {"lyric": lyric, "start_tick": st, "duration_tick": du, "ties": ties}
        if sub:
            upd.update(artic="", artic_depth=None)
        out.append(src[j].model_copy(update=upd))
    return out
