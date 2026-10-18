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

"""**ハモリ**（extra_voices）を付ける。

楽譜スキーマ（`SongScore.extra_voices`）も本番の生成機能も前からあったのに、
新パイプラインが繋いでいなかったため**ずっと単声**だった（2026-08-21 のセルフチェックで判明）。

設計の要点は**役割を分けること**（2026-09-23 に2段目を差し替えた）:

  ・**どこに・何のために重ねるか** → LLM（1段目 `plan_harmony`。セクション単位の判断だけ）
  ・**実際に鳴る音と、その位置・長さ** → LLM（2段目 `voices_for`。表を見て書く）
  ・**出来た声部の内容の所見**     → 機械（`check_voices`＝推奨音域の外。**直さず所見に出す**。
                                    仕上げの点検役の材料は `voice_content_facts(song)`・2026-10-03 判-15）
  ・**声部の形式の検査**           → 機械（`VoicesStep`。1音1モーラのひらがな＝記法の型 `VOICE_WORD`・狙いの番号・
                                    狙いに1音も無い・歌える範囲の外・1つの狙いの中で重ならないか＝`voice_findings`。
                                    所見を返し、直す音だけ差分で直させる・2026-09-29。後ろの2つを足したのは 2026-10-03 判-15）

★**もとは2段目が機械だった**（主旋律をスケール上で ±2 ずらす `build_harmony`）。
  音は一意に決まって安上がりだったが、**主旋律に無い音・無い言葉を1つも作れない**ので、
  合いの手も追っかけも原理的に届かなかった。**こちらが位置を計算すると格子から外れる**
  （22/38 音）のも分かったため、**位置と長さごと LLM に書かせる**形にした。
  `build_harmony` とその補助（`INTERVALS` / `_move` / `_scale_pcs`）は
  **2026-09-23 に削除**（本人承認済み。どこからも呼ばれていなかった）。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from pydantic_ai import Agent

from song_model import PatchStatus, StepStatus
from song_checks import JsonBool, JsonFloat, JsonInt  # noqa: E402（LLM が書く数と真偽の欄の型・2026-10-02 判断5）
from song_checks import strip_ascii_ws  # noqa: E402（半角の名前の前後の空白・§0-6・2026-10-03）
from song_llm import pick_model
from song_llm import run_sync as _llm_run
from song_usage import record as _rec_usage           # 費用の集計
from song_schema import SongNote, VoiceTrack
from song_prompts import prompt                          # プロンプトの正は prompts/*.md
from voice_style import STYLES as _VS_STYLES             # 声部の歌い方の選択肢（V6 のスタイル）
from voice_style import style_problem as _style_problem

_VS_NAMES = list(_VS_STYLES)

#: **声の推奨音域**（D3〜D6・鳴る高さ）。`song_pipeline.score_validate` もこれを使う（**基準は1つ**）。
#: 物理的に声が出る範囲（36-96）とは別。
#:
#: ★**2026-09-07 に 60-76 から広げた。** 元の 76 という上限は**誤診の名残**だった——
#: 2026-09-06 に「ハモリが 20dB 以上低い」のを音域のせいだと診断して置いた値だが、
#: 真因は `fix_timing` が `extra_voices` を動かさず**ハモリだけ 10.1秒ずれていた**ことで、
#: 音域は関係なかった（`tests/song_service/test_fix_timing_voices.py`）。
#:
#: 実測（`dev_tools/scratch/s5065b362_range_audio.py`・人が V6 で書き出したボーカル単体 70音）:
#: **65〜86 のどの音高でも、基準（ド4〜ド5 の中央 −32.1dB）から 6dB 以上落ちない**
#: ——最も低い ド6〜レ6 でも −36.8dB＝**−4.7dB**。つまり 86 まで実用になる。
#:
#: 76 のままだと、**プロンプトが求めるサビの最高音（ラ5〜シ♭5＝81〜82）が必ず所見になる**。
#: 実際、調の合った曲で **84件の所見**が出て、本物の所見（調のずれ）を埋もれさせた。
#: **鳴らない音を止めるための検査が、良い曲を止めていた。**
#:
#: ★**下端は 2026-10-05 に A#3（58）→ D3（50）へ広げた**（本人決定・案A）。58 には実測の根拠が無かった。
#: 実測（`dev_tools/scratch/s5065b362_lowrange.py`・母音「あ」1.5秒・V6）: 高さは C2 まで正しく出る（ずれ数cent）が、
#: 音量は A#3〜G3 −30〜−33dBFS・F#3〜D3 −35〜−42・C#3〜C3 −43〜−45・B2 以下 −47〜−51 で頭打ち（声として鳴らない）。
#: 本人が試聴して **D3 まで使える**と判断。A#3〜D3 は主旋律より最大 −12dB 小さいので、声部の指示書が音量で補う標準を書く
#: （`prompts/song_voices.md`）。記録は `docs/plan/song_acappella_instrumental.md` 末尾。
#: ★**主旋律の下端は別の値 `MELODY_LO`（58）のまま**——主旋律は歌いやすい帯域を保つ（旋律の枠・`score_validate` の主旋律）。
SING_LO, SING_HI = 50, 86
#: **主旋律の下端**（鳴る高さ・`A#3`）。旋律の枠（`song_melody_step` の `floor`）と `score_validate` の主旋律の所見が使う。
#: 声部の推奨音域（`SING_LO`）を D3 へ広げたとき、旋律の枠を動かさないために分けた（2026-10-05）
MELODY_LO = 58
#: **歌える範囲**（`C2`〜`C7`・鳴る高さ）。外は V6 で鳴らない。声部（`check_voices` の所見）とハモリ追加（型の差し戻し・
#: `song_llm.added_voice_type`）が同じ値を使う（2026-10-02・ず-12③。以前ハモリ追加だけ `C2`〜`B6`）
VOICE_LO, VOICE_HI = 36, 96
#: **声部の音量の範囲**（主旋律 0dB に対する dB）。vpr の声部フェーダーが受け付ける範囲（`v6_export._FADER_MIN/MAX`＝
#: −60〜+6dB・1/10dB 単位）と同じ（2026-10-03・§2 c10）。工程のハモリ・ハモリ追加・`adjust_voice` が同じ値を使う
VOICE_GAIN_DB = (-60.0, 6.0)


class HarmonyPart(BaseModel):
    """**この区間に、何のために、どれくらい声を重ねるか**（2026-09-23 に広げた）。

    ★もとは `interval`（3度上・6度上…）しか持っていなかった。これは
      「**機械が主旋律を ±2 ずらす**」前提の指示書で、**合いの手も追っかけも表せない**
      （主旋律に無い音・無い言葉を作れないため）。度数の固定をやめ、**狙い**にした。
      経緯と実測は [vocal_parts.md](../../../../docs/plan/vocal_parts.md)。
    """
    sections: list[str] = Field(description="声を重ねるセクション（設計の kind と同じ名前）")
    # ★既定値をやめて必ず書かせる（2026-10-02・洗い出し）。以前は省くと「厚みを出す」が入り、声部の LLM には
    #   前の工程が決めた狙いとして見えていた（決めていない狙いを決めたことにしていた）。空も型が差し戻す
    aim: str = Field(description=(
        "この区間で**何をしたいか**。決まった型はありません。よくあるのは:\n"
        "  厚みを出す   主旋律と同時に、同じ言葉で重ねる\n"
        "  掛け合い     主旋律が休んでいる隙間で、別の言葉を返す\n"
        "  追う         主旋律の言葉を、少し遅れて繰り返す\n"
        "  空気を足す   言葉でない声（あー・うー）を伸ばして敷く\n"
        "**これは例であって、選択肢ではありません。** その区間に合うものを書いてください"))
    # ★選択肢は型（`Literal`）で閉じる（2026-09-30 本人「選択肢は enum にしよう」・判-1）。以前はただの文字列で検査も無かった
    # ★2026-10-04（組N2）: density・gain_db・voice_style も既定をやめて必ず書かせる（以前は ふつう・-6.0・空）
    density: Literal["薄く", "ふつう", "厚く"] = Field(description=(
        "どれくらい入れるか（**必ず書く**。**次の3つのどれか**）: 薄く / ふつう / 厚く。\n"
        "**Aメロは薄く、サビで厚く**が基本。**同じ濃さで並べると平坦になります**"))
    # ★範囲は書き出しの限界（vpr の声部フェーダー −60〜+6dB・`VOICE_GAIN_DB`）に合わせた（2026-10-03・
    #   `docs/plan/song_viewpoints_review.md` §2 c10・d3）。以前は −18〜0。主旋律より大きい（0 超）・ごく小さい音量は
    #   音の判断なので型では弾かず、0 超は仕上げの点検役の材料に事実で出す（`song_finish_gate`・`loud_voices`）
    gain_db: JsonFloat = Field(ge=VOICE_GAIN_DB[0], le=VOICE_GAIN_DB[1], description=(
        "主旋律に対する音量（**必ず書く**。dB・−60〜+6）。**-4〜-8 が自然**。0 にすると主旋律と対等になり歌が濁る"
        "（0 より大きくすると主旋律より前に出る——狙いがあるときだけ）。"
        "**低く置く声部は上げて補う**（ミクの声は低いほど小さく鳴る。G#3〜C4 付近は 0〜+3、D3〜G3 付近は +3〜+6・2026-10-05 実測）"))
    voice_style: str = Field(description=(
        "この声部の歌い方（**必ず書く**。例 `MIKU Default`）。ハモリ全体の歌い方（編成の `harmony_style`）のままでよければ空の文字 `\"\"` と書く。**使えるのは次の名前だけ**: "
        + " / ".join(_VS_NAMES) + "\n"
        "**掛け合いは声を変えると「別の人が返している」ように聞こえます**"))
    # ※度数で書いていた頃の欄（`interval`）は 2026-09-30 に外した（古い記録は読まない）。LLM に見せるスキーマからも消える。
    #   古い記録に残っていても、pydantic は知らない欄を無視するので読み込みでは落ちない

    @field_validator("aim")
    @classmethod
    def _aim_written(cls, v: str) -> str:
        if not (v or "").strip():
            raise ValueError("狙い（aim）が空——この区間で何をしたいかを書く（例: 厚みを出す・掛け合い・追う・空気を足す）")
        return v

    @field_validator("voice_style")
    @classmethod
    def _known_style(cls, v: str) -> str:
        """**知らない歌い方は型で差し戻す**（2026-09-27・目録 表情-10）。以前は書き出しで当てずに据え置き、
        所見は画面に出るだけでこの工程に届かなかった。いまは LLM に「使える名前」つきで返る。"""
        why = _style_problem(v)                             # 前後の全角の空白などは所見（半角の名前・§0-6・2026-10-03）
        if why:
            raise ValueError(why)
        return strip_ascii_ws(v)


class HarmonyPlan(StepStatus):
    reason: str = Field(description="どこになぜ重ねたか（付けない判断ならその理由）")
    parts: list[HarmonyPart] = Field(description=(
        "（**必ず書く**。ハモリを付けないなら空の並び `[]`＝付けないを選んだ。理由は reason に）**必要な場所にだけ**。曲中ずっと重ねると平坦になり、主旋律が埋もれる"))


_harmonizer: Agent[None, HarmonyPlan] = Agent(
    pick_model("HARMONY", "flash"), output_type=HarmonyPlan,
    system_prompt=(
        prompt("song_harmony")
    ),
)


def check_voices(notes: list[SongNote], tracks: list[VoiceTrack], form: str = "song") -> list[str]:
    """出来た声部の**内容の所見**（2026-09-23 に切り出した・2026-10-03 に内容だけにした＝判-15）。**作り方によらず掛ける**。

    ★もとは `build_harmony`（＝主旋律を±2ずらす作り方）の中に埋まっていた。
      作り方を差し替えると**守りごと消える**ので、**検査だけ独立させた**。
      どれも実際の失敗から入ったもので、**直さずに所見として出す**
      （判断は点検役と人。機械は証拠を並べる側。仕上げの点検役の材料は `voice_content_facts`）。

    | 見るもの | いつ入ったか |
    |---|---|
    | 推奨音域（`SING_LO`〜`SING_HI`＝50〜86・2026-10-05 までは 58〜86）の外 | 2026-09-06。H48 の試聴で、外の音が他区間より 20dB 低く聴こえなかった |
    | ~~曲の9割以上に乗っている~~ | 2026-08-22 に入れ、2026-10-03 に外した（下の★） |

    ★**答えが1つに決まる2つは形式の検査へ移した**（2026-10-03・判-15）。「歌える範囲（36〜96）の外＝V6 で鳴らない」と
      「計画した狙いに1音も無い」（2026-08-21。**計画したハモリが黙って消えていた**）は、声部の工程の検査（`voice_findings`）が
      所見を返して差分で直させ、直らなければ声部の工程で止まる。以前はここで記録に残すだけで、しかも「1音も無い」は
      `voices_to_tracks` が音の無い声部を作らないので、ここへ来る形では一度も出なかった。
      ハモリ追加（`song_llm.generate_added_voice`）の歌える範囲は型（`added_voice_type`）が差し戻す。
    ★**音域は鳴る高さ（実音）で見る**（2026-09-30・記-3）。声部は C 基準で書き、組み立て（`voice_to_track`）が区間の調へ移すので、
      ここへ来る `tracks` は移したあとの声部。書いた音名（C 基準）で音域を見ないこと。
    ★**「曲の9割以上に乗っている」は外した**（2026-10-03・本人了承「観点の整理」＝`docs/plan/song_viewpoints_review.md` §1-5）。
      9割は閾値で、発端（サビ1区間の全98音に3度上）は区間の単調さなのに曲全体で数えていた（ハモリはサビだけのことが多く、
      曲全体では9割に届かない）。乗る割合は仕上げの点検役が**区間ごと**に事実で読む（`song_finish_facts.voice_coverage`）。
      `notes`・`form` は呼び出しの形を変えないために残している（以前はアカペラだけ9割を見なかった・F7）。
    """
    return voice_range_facts(tracks)


def voice_range_facts(tracks: list[VoiceTrack]) -> list[str]:
    """声部の**推奨音域の外**の所見だけ（`check_voices` の1つ目・2026-10-03 に切り出した＝組G4）。
    仕上げの点検役は「声部が乗る割合」を曲全体でなく区間ごとに材料へ出す（`song_finish_facts.voice_coverage`）ので、こちらだけ使う。"""
    warns: list[str] = []
    for t in tracks:
        out = sorted(n.key for n in t.notes if not SING_LO <= n.key <= SING_HI)
        # 音域は音名で見せる（点検役の材料・2026-10-01・判-D。以前は MIDI の番号 `36-96`・`58-86`）
        # ★**事実だけを書く**（2026-10-03・本人「機械が検査して事実を出し、直し方は点検役に委ねる」）。以前は「V6 で実質鳴らない。
        #   下ハモや別の度数を検討するか、主旋律の高さ自体を見直す」と断じて直し方まで書いていた。判断は仕上げの点検役
        if out:
            warns.append(f"{t.title}: {len(out)}/{len(t.notes)}音が**推奨音域（{spell_pitch_name(SING_LO)}-{spell_pitch_name(SING_HI)}）の外**"
                         f"（区間の調へ移した鳴る高さで・外の音は {spell_pitch_name(out[0])}〜{spell_pitch_name(out[-1])}）")
    return warns


def voice_content_facts(song, *, whole_song_share: bool = False) -> list[str]:
    """**声部の内容の所見**（推奨音域の外）を曲から取り出す（2026-10-03・判-15→判-21）。
    仕上げの点検役（後ろの工程の内容を見る）の材料。声部（楽譜の `extra_voices`・ハモリ追加の声部も含む）が無ければ空。

    見方は `check_voices` と同じ（移したあとの鳴る高さ）。形式の所見（歌える範囲の外・狙いに1音も無い）は
    声部の工程が直させて止めるので、ここへは来ない。曲全体の「9割に乗る」は 2026-10-03 に外した（`check_voices`）。
    `whole_song_share` は呼び出しの形を変えないために残している（どちらでも同じ）。
    """
    score = getattr(song, "score", None)
    tracks = list(getattr(score, "extra_voices", None) or []) if score is not None else []
    return voice_range_facts(tracks) if tracks else []


def plan_harmony(title: str, sections, brief_text: str, *, song=None, verbose: bool = False) -> HarmonyPlan:
    """セクション一覧を見せて、どこに重ねるかだけ決めさせる。

    ★書いた区間名を設計と照らし、差分で直させる（2026-09-30・判-14①・`HarmonyPlanStep`）。直らなければハモリの工程で止まる
      （`StepUnfixed`）。`song` を渡すと直しの記録（`harmony_fix`）が残る。再生はこの関数ごと差し替えるので検査しない。
    """
    lines = []
    acap = (getattr(song, "form", "song") or "song") == "acappella"
    for s in sections:
        has_vocal = s.has_vocal
        lines.append(f"  {s.kind}: {s.bars}小節 / 強さ{getattr(s, 'intensity', 3)}"
                     + ("" if has_vocal else (" / **歌なし**（アカペラなので声を置ける）" if acap else " / **歌なし**")))
    prompt = (f"曲名『{title}』\n\n{brief_text}\n\n【セクション】\n" + "\n".join(lines)
              + "\n\nどこにハモリを重ねますか（**入れない判断もあり**）。")
    r = _llm_run(_harmonizer, prompt, label='ハモリ')
    _rec_usage(_harmonizer, r, "ハモリ")                    # 費用集計から漏れていた（2026-08-23）
    # 曲の形（2026-10-03）: アカペラだけ「歌の無い区間に付けられない」を外す（F7）。曲を渡さない呼び方（カバー）は歌もの
    plan, _ = HarmonyPlanStep(sections, getattr(song, "form", "song") or "song").run(r.output, song, verbose=verbose)
    return plan


# ============================ 2段目: 声部を書く（2026-09-23）============================
#
# ★**`build_harmony` を置き換える作り方**。主旋律を ±2 ずらすのではなく、
#   **LLM が位置も長さも書く**。こちらは拍を tick に直すだけ。
#
#   もとは「隙間を音数で均等割り」してこちらが位置と長さを決めていたが、
#   **16分の格子から 22/38 音が外れ**、本人に「入れ方が不自然」と言われた。
#   書かせたら**格子外 0/55**。本人「**かなり自然になった。問題は解消された**」。
#   ※これは 2026-09-20 に旋律で一度到達していた形
#     （`abc_sketch_to_melody`「音価は LLM が書いたものをそのまま使う」）。
#   実測と捨てた案は [vocal_parts.md](../../../../docs/plan/vocal_parts.md)。

from song_checks import POSITION, Notation, position_notation   # noqa: E402（記法の型・2026-09-29）

_Q = 480                                                   # 4分音符の tick（`song_schema.TICKS_PER_QUARTER`）


class VoiceNote(BaseModel):
    """声部の音1つ。**`inspect_song` の表と同じ言葉**で書いてもらう。"""
    at: str = Field(description="鳴らし始める位置。**表と同じ書き方**（`62:3.00` ＝62小節目・拍3.00。小節は楽譜の頭から1・"
                                "拍は**小数2桁**・半角の数字）。"
                                "**拍は 0 から数えます**（小節の頭が 0.00）。拍は1小節の長さより小さく"
                                "（4/4 なら 3.75 まで）、次の小節の頭は小節を進めて書きます（`63:0.00`）")
    beats: JsonFloat = Field(gt=0, description="長さ（拍）。表の『長さ』の欄と同じ単位（0.5 / 1.0 / 2.0 …）")
    # ★**表の音高と同じ物差し**（2026-09-30・記-3）。新しい曲の表は C メジャー／A マイナー基準（移調前）で、書いた音は組み立て
    #   （`voice_to_track`）が区間の調へ移す——旋律・コード・伴奏の音と同じ。以前は表も声部も実音で、同じ `ミ5` が工程で違う高さを指した
    # ★**国際式の音名だけ**（2026-10-01・判-B）。以前は英語（`C5`）と階名（`ミ5`）の両方を受けていた（1つの欄に2通り）
    note: str = Field(description=(
        "高さ。国際式の音名（`C4`＝中央のド・`E5`・`F#4`・`Bb3`。変化記号は `#` と `b`、オクターブの数字は必ず付ける）。"
        "**表の音高の欄と同じ書き方・同じ物差し**——表は **C メジャー／A マイナー基準（移調前）**で、"
        "書いた音は機械が区間の調へ移して鳴らす（旋律・コードと同じ。カバーも同じ。表の頭に書いてある）"))
    # ★1音1モーラ・ひらがなだけは、ここに書いてあるだけで指示書にも検査にも無かった（2026-09-29）。
    #   指示書（song_voices.md）に書き、形式の検査（`voice_findings`・`VoicesStep`）で所見にして差分で直させる
    word: str = Field(description=(
        "歌う**1モーラ**（**ひらがなだけ**。「きゃ」のような拗音は2文字で1モーラ）。前の音を伸ばすなら `_` だけを書く（詞と同じ印）。"
        "2モーラ以上・カタカナ・漢字・英字・`か_` のような印つき・`~` `*`・途中の空白・小書きだけ（`ゃ`）は検査で所見として返します"
        "（音を分けて1モーラずつ書く）"))
    #: ★**どの狙いの音か**（2026-09-27・目録 声部-2／声部-3）。狙いごとに音量（`gain_db`）と声（`voice_style`）が
    #:   違うのに、書かれた音は1本の声部にまとめられ、**1つ目の狙いの音量しか使われず、声は全部同じ**になっていた。
    #:   番号を書いてもらい、**狙いごとに別の声部**として書き出す。0＝書かれていない——狙いが1つなら省けて、その狙いの音になる
    #:   （`voices_to_tracks`）。狙いが2つ以上なら、書くときの型（`voice_part_type`）と声部の検査が番号を求める。
    # ★2026-10-04（組N2）: 既定 0（＝書かれていない・狙いが1つならその狙い）をやめて必ず書かせる。狙いが1つでも 1 と書く
    aim: JsonInt = Field(ge=1, description=(
        "この音が【前の工程が決めた狙い】の**何番**のためのものか（**全部の音に必ず書く**。1から。狙いが1つなら全部 1）。"
        "狙いごとに音量と声が違うので、番号ごとに別の声部として鳴らします"))

    # ★**読めない位置・音名は型で差し戻す**（2026-09-27・監査）。以前は `voice_to_track` が**黙って捨て**、
    #   所見は `StepRecord(step="harmony")` の warnings に入るだけで、書いた工程にも後の工程にも届かなかった。
    #   いまは pydantic_ai がこの文を声部の LLM へ返して書き直させる。
    #   ※保存から戻すとき（`song_replay`）もこの検証は通る。2026-09-27 時点で保存済みの声部に該当は無い
    #     （`dev_tools/scratch/s5065b362_auditfix.py` で 445 ファイルを確認）。
    @field_validator("at")
    @classmethod
    def _readable_at(cls, v: str) -> str:
        # 読めるかは位置の記法の型（`POSITION`）1つ。所見の文も同じ（小数2桁でない拍には正しい書き方を添える・2026-10-01・ず-8）
        if bad := POSITION.problems(v):
            raise ValueError(bad[0])
        return v

    @field_validator("note")
    @classmethod
    def _readable_note(cls, v: str) -> str:
        # 読めるか・鳴らせる高さか（MIDI 0〜127）は音名の記法の型（`song_llm.PITCH`）1つで見る（組み立ても同じ読み手）
        from song_llm import PITCH                          # noqa: E402（循環回避）
        if bad := PITCH.problems(v):
            raise ValueError(bad[0])
        return v

    @field_validator("beats")
    @classmethod
    def _sounding_beats(cls, v: float) -> float:
        if why := beats_problem(v):
            raise ValueError(why)
        return v


def beats_problem(beats: float) -> str:
    """声部・ハモリ追加の長さ `beats` の所見（無ければ空）。**0 より大きくても tick にして 0 になる長さは書けない**
    （2026-10-02・仕様書 §14。以前は `0.001` を型が通し、長さ 0 tick の音ができた）。0 以下は型の `gt=0` が先に返す。"""
    if beats > 0 and beats_to_ticks(beats) <= 0:
        return (f"長さ {beats:g} 拍は短すぎて鳴らない（tick にすると 0）——長さは拍数（4分音符＝1.00・16分＝0.25）で、"
                "16分なら 0.25 と書く")
    return ""


class VoicePart(StepStatus):
    reason: str = Field(description="どこを厚くし、どこを空けたか（曲全体の方針）")
    title: str = Field(description="この声部の名前（**必ず書く**。harmony / call / canon など・表示用）")
    notes: list[VoiceNote] = Field(description="声部の音を、**鳴る順に**並べる（**必ず書く**）")


def at_to_tick(at: str, bar_ticks: int) -> int | None:
    """`62:3.00` → tick。**表とまったく同じ読み方をする**。

    ★**拍は 0 始まり**。`inspect_song` の表は `あ|A4|3:0.00|1.00|Dm|5th` のように
      **小節の頭を 0.00** と書く。1始まりと決めつけたせいで、
      **LLM が表のとおり書いた 55音のうち 23音を捨てていた**（2026-09-23）。
    ★読み方は位置の記法の型（`song_checks.POSITION`・2026-09-29。奏法・歌の崩しと同じ1つ）。以前はここに同じ正規表現を持ち、
      `1.2.3` のような拍は読めない（None）ではなく例外で落ちていた。
    ★前後の半角の空白とタブだけ除く（前後の全角の空白・改行などは 2026-10-03 から読めない・§0-6）。途中に空白のある位置（`12 : 1.50`）は 2026-10-02 から読めない（None。所見は位置の記法の型・§12）。
    """
    got = POSITION.read(at)
    if got is None:
        return None
    bar, beat = got
    return (bar - 1) * bar_ticks + beats_to_ticks(beat)


def beats_to_ticks(beats: float) -> int:
    """拍（表の書き方・小数2桁）→ tick。**表の書き方の逆**として読む（2026-09-29）。

    表は拍を小数2桁で見せるので、3等分した位置・長さ（160tick）は `0.33` と出る。以前は `round(0.33×480)`＝158tick で読み、
    表のとおり書いた声部の音が格子から外れた。小数2桁で3等分と読める数（`0.33`・`0.67`・`1.33` …）は3等分の tick に戻す。
    ★**3等分と読むのは、小数2桁で書いた数が3等分の数を2桁に丸めた形と一致するときだけ**（2026-10-02・洗い出し。仕様書 §0-3・§14）。
      以前は ±0.0067 拍の幅で読み、`1.005` のような2桁でない数も 1.00 と同じ 480tick に寄せていた（いまは 482tick＝書いたとおり）。
    ★**8分・16分の3等分も同じ決まりで読む**（2026-10-03・記N）。4分音符を12等分した数（40tick の倍数）を2桁に丸めた形と一致すれば
      その tick に戻す（`0.08`→40・`0.17`→80・`0.42`→200・`0.58`→280・`0.83`→400・`0.92`→440）。表（`artic_plan.tick_key`）は
      8分の3等分を `0.17`・`0.83` で見せるのに、以前は4分の3等分（`0.33`・`0.67`）しか戻さず、書き写すと 2tick ずれた（82・398tick）。
    """
    twelfths = round(beats * 12)
    hund = beats * 100
    if abs(hund - round(hund)) < 1e-6 and round(beats, 2) == round(twelfths / 12, 2):
        return int(twelfths) * _Q // 12
    return int(round(beats * _Q))


# ※声部の音名の読み手は `song_llm.pitch_name_to_midi`（国際式・伴奏と同じ1つ・2026-10-01）。以前はここに `note_name_to_midi`
#   （英語も階名も受ける）があった。表の音高の欄と同じ書き方で返してもらう決まり（2026-09-23 に、表の英語を読めず書かれた音を
#   全部捨てた事故から）は、表も声部も国際式の音名1つになったので、そのまま守られる。


def section_shift_at(shift_of, bars_of: dict | None, bar_ticks: int):
    """tick → その位置の区間の移調量（半音）。**声部の移調量はここで引く**（2026-09-30・記-3）。

    移調量そのものは旋律・コードと同じ `shift_of`（区間 → 移調量・`song_place.section_keys`）で、ここは tick がどの区間かを
    時間軸の区間（`bars_of`＝区間→(開始小節, 終了小節)・1始まり・両端を含む）で引くだけ——2本目の移調の計算を作らない。
    区間の外（あれば）は時間軸（`SectionTimeline.at`）と同じく近い端の区間。
    `shift_of` が無いなら None を返す——表は鳴る高さのまま、声部も移さない（曲づくりの呼び手はどれも渡す。カバーも
    2026-10-01 の判-C から推定した調の `shift_of` を渡す・`song_arrange_cover`）。
    """
    if shift_of is None:
        return None
    spans = sorted(((int(b0), int(b1), k) for k, (b0, b1) in (bars_of or {}).items()), key=lambda x: x[0])
    if not spans:
        raise ValueError("声部の移調量を引くのに区間の位置（bars_of）が無い")

    def at(tick: int) -> int:
        bar = int(tick) // bar_ticks + 1
        for b0, b1, k in spans:
            if b0 <= bar <= b1:
                return int(shift_of(k))
        return int(shift_of(spans[0][2] if bar < spans[0][0] else spans[-1][2]))
    return at


def voice_to_track(part: VoicePart, bar_ticks: int, *, gain_db: float = -6.0,
                   voice_style: str = "", tempo: float | None = None,
                   shift_at=None) -> tuple[VoiceTrack, list[str]]:
    """書かれたものを声部へ。**拍を tick に直すだけ**——位置も長さも触らない。

    ★**高さは区間の調へ移す**（2026-09-30・記-3）。声部は旋律・コードと同じ C メジャー／A マイナー基準で書くので、
      `shift_at`（tick → 移調量・`section_shift_at`）の分だけ上げる（音の頭の区間の量。旋律がフレーズの区間の量で移すのと同じ）。
      `shift_at` が無ければ移さない（区間を持たない呼び方。カバーも 2026-10-01 から移す・判-C）。

    ★**格子から外れていても直さない**。所見に出して、判断は関門の LLM と人に渡す
      （音域外を機械で直さないのと同じ作法・2026-09-06 本人判断）。
      吸わせてしまうと**合っているように見える壊れたもの**ができる。

    ★位置・音名（鳴らせる高さ）・長さ（0 より大きい）は `VoiceNote` の型が差し戻し、言葉（`VOICE_WORD`）と拍の上限
      （`position_notation`）は声部の検査（`VoicesStep`）が所見にして直させるので、ここへは検査を通った音しか来ない。
      来たら型・検査と組み立ての読み方がずれている不具合なので `song_fix.AssemblyError` で止める（2026-09-30。
      以前は読めずに捨てて記録して進んでいた）。
    ★言葉の伸ばす印 `_` は V6 の伸ばす歌詞 `-` に訳す（2026-09-30・記-4。詞の `_` と同じ訳し方＝組み立て）。
    """
    from song_fix import AssemblyError                     # noqa: E402（循環を避けてここで）
    from song_llm import PITCH                              # noqa: E402（音名の読み手＝型と同じ1つ）
    pos = position_notation(bar_ticks / _Q)
    out: list[SongNote] = []
    warns: list[str] = []
    for i, n in enumerate(part.notes, 1):
        t = at_to_tick(n.at, bar_ticks)
        d = beats_to_ticks(n.beats)
        key = PITCH.read(n.note)
        if key is not None and t is not None and shift_at is not None:
            key += int(shift_at(t))                          # C 基準 → 区間の調（記-3）
        bad = pos.problems(n.at) + VOICE_WORD.problems(n.word)
        if t is None or key is None or d <= 0 or not 0 <= key <= 127 or bad:
            raise AssemblyError(f"声部の組み立てに、型・検査を通ったはずの音が来た: {part.title} {i}番"
                                f"（at={n.at!r} note={n.note!r} beats={n.beats} word={n.word!r}）"
                                + (f"——{'／'.join(bad)}" if bad else ""))
        if t % (_Q // 4) or d % (_Q // 4):
            warns.append(f"{part.title} {i}番: **16分の格子に乗っていない**（{n.at} / {n.beats}拍）")
        # ★前後の空白を除いた言葉を置く（2026-10-02・洗い出し H3。仕様書 §0-6 の「前後の空白は取り除く」）。以前は伸ばしの印以外
        #   を書いたまま置き、`' か '` が空白入りの V6 の歌詞になった
        w = n.word.strip()
        out.append(SongNote(lyric=_V6_WORD.get(w, w), key=key, start_tick=t, duration_tick=d))
    # ★**声部にも調声を当てる**（2026-09-23）。差し替えたとき**ここが抜けていた**
    #   ——旧 `build_harmony` は主旋律の調声を写していたが、`voice_to_track` は
    #   `tuning` を渡しておらず、`aiExp` が全部 0.5（V6 の既定）になっていた。
    #   **主旋律の写しは使えない**——合いの手は位置も言葉も違うので、
    #   「フレーズの頭でしゃくる／語尾を抜く」の当たる場所がずれる。
    #   **同じ規則を、この声部自身の並びに掛け直す**。
    if out:
        from song_tuning import apply_by_gap                # noqa: E402（循環回避）
        # ★**この機械の表情は指示書に書いてある**（2026-09-27・目録 声部-1）。それまで `song_voices.md` は
        #   「あなたが書いたとおりに鳴ります」とだけ言っていた。位置・長さ・高さ・言葉は書いたとおり、
        #   表情（しゃくり・揺れ・音量・抜き）は主旋律と同じ規則で付く、と書き分けた。
        #   `tempo` を渡すと揺れは「後半 0.5秒以上の音」だけ（`song_tuning.vibrato_fits`・目録 表情-7）。
        apply_by_gap(out, tempo=tempo)
    extra: dict = {}
    if voice_style:
        if "voice_style" in VoiceTrack.model_fields:
            extra["voice_style"] = voice_style
        else:                                               # 楽譜に欄が無いと黙って消えるので、所見に出す
            warns.append(f"{part.title}: 声『{voice_style}』を楽譜に持てない（`VoiceTrack.voice_style` の欄が無い）"
                         f"——ハモリ全体の声（harmony_style）で鳴る")
    return VoiceTrack(title=part.title, notes=out, gain_db=gain_db, **extra), warns


def voices_to_tracks(got: VoicePart, plan: HarmonyPlan, bar_ticks: int,
                     *, tempo: float | None = None, shift_at=None) -> tuple[list[VoiceTrack], list[str]]:
    """書かれた音を**狙いごとの声部**にする（2026-09-27・目録 声部-2／声部-3）。`tempo` は揺れの条件に使う。
    `shift_at`（tick → 移調量）は `voice_to_track` へそのまま渡す（C 基準 → 区間の調・記-3）。

    狙い（`HarmonyPart`）ごとに `gain_db` と `voice_style` を持つのに、以前は全音を1本にまとめ、
    **1つ目の狙いの音量だけ**を使い、声は区別しなかった。いまは音の `aim`（LLM が書いた番号）で分け、
    **その狙いの音量と声のまま**別の声部にする。音の位置・長さ・高さ・言葉は `voice_to_track` のとおり触らない。

    - 狙いが1つなら全部その狙い（番号が無くても、狙いは1つしかないので迷わない）。2 以上の番号は声部の検査が返すので来ない（2026-10-02）
    - 狙いが2つ以上で、番号が無い・範囲外の音は、声部の検査（`voice_findings`）と書くときの型（`voice_part_type`）が
      差し戻し、直らなければ声部の工程で止まる（`StepUnfixed`）ので、ここへは来ない。来たら不具合として `AssemblyError`
      （2026-09-30。2026-09-28 からは鳴らさずに記録して進み、その前は1つ目の狙いの音量・声で鳴らしていた）
    - 声部が1本なら名前は LLM が書いたまま（以前と同じ）。2本以上なら `<名前>-<狙いの番号>`
    """
    # 計画の無い呼び方の受け皿（以前の型の既定の音量・声で1本。機械が組む受け皿なので値を明示する・2026-10-04 組N2）
    parts = plan.parts or [HarmonyPart(sections=[], aim="（計画なし）", density="ふつう", gain_db=-6.0, voice_style="")]
    groups: dict[int, list[VoiceNote]] = {}
    unknown = 0
    for n in got.notes:
        k = int(n.aim)                                     # 必須（2026-10-04 組N2。以前は 0＝省いた・狙いが1つならその狙い）
        if not 1 <= k <= len(parts):                       # どの狙いの音量・声で鳴らすか決まらない（範囲外の番号は検査が返す）
            unknown += 1
            continue
        groups.setdefault(k, []).append(n)
    if unknown:
        from song_fix import AssemblyError                 # noqa: E402（循環を避けてここで）
        raise AssemblyError(f"声部の組み立てに、検査を通ったはずの声部が来た: {got.title} の {unknown}音に"
                            f"どの狙いの音か（aim）が無い・範囲外（狙いは {len(parts)}個）")
    warns: list[str] = []
    tracks: list[VoiceTrack] = []
    for k in sorted(groups):
        p = parts[k - 1]
        title = got.title if len(groups) == 1 else f"{got.title}-{k}"
        sub = got.model_copy(update={"title": title, "notes": groups[k]})
        t, w = voice_to_track(sub, bar_ticks, gain_db=p.gain_db, voice_style=p.voice_style, tempo=tempo,
                              shift_at=shift_at)
        warns += w
        if t.notes:
            tracks.append(t)
    return tracks, warns


_voicer: Agent[None, VoicePart] = Agent(
    pick_model("HARMONY", "flash"), output_type=VoicePart,
    system_prompt=prompt("song_voices"),
)


def voice_part_type(n_aims: int) -> type[VoicePart]:
    """**狙いの数を知っている声部の型**（2026-09-27・監査）。狙いが2つ以上のとき、`aim` が無い（0）・
    範囲外の音を**型で差し戻す**（pydantic_ai がこの文を声部の LLM へ返して書き直させる）。

    以前は `voices_to_tracks` が黙って1つ目の狙いの音量・声で鳴らしていた（所見は `harmony` の記録に入るだけ）。
    型を通らずに残ったもの（保存から戻した古い記録）は、2026-09-28 から `voices_to_tracks` が鳴らさない。
    狙いの数は呼ぶたびに違うので、`VoicePart` 自体には書けない——書くときだけこの型を渡す。
    保存から戻すとき（`song_replay`）は `VoicePart` のまま読むので、古い記録は今までどおり読める。
    """
    if n_aims <= 1:
        return VoicePart

    class _VoicePartForPlan(VoicePart):
        @model_validator(mode="after")
        def _aims_in_range(self):
            why = _aim_problem(self.notes, n_aims)
            if why:
                raise ValueError(why)
            return self

    _VoicePartForPlan.__name__ = "VoicePart"               # LLM に見える型の名前は変えない
    return _VoicePartForPlan


def _aim_problem(notes: list[VoiceNote], n_aims: int) -> str:
    """`aim` の所見（無ければ空）。型（`voice_part_type`）と検査（`voice_findings`）で同じ文を使う。

    狙いが2つ以上なら、`aim` が無い・範囲外の音。狙いが1つなら、0（省いた）と 1 のほか（2026-10-02・洗い出し。以前は
    範囲外の番号も黙ってその1つの狙いと読んでいた）。
    """
    if n_aims <= 1:
        bad1 = [f"{i}番（{n.at} {n.note}「{n.word}」aim={n.aim}）"
                for i, n in enumerate(notes, 1) if int(n.aim) != 1]
        return ("どの狙いの音か（aim）が範囲外の音がある: " + " / ".join(bad1[:12])
                + (f" ほか{len(bad1) - 12}音" if len(bad1) > 12 else "")
                + "——狙いは1つなので、aim は全部の音に 1 と書く") if bad1 else ""
    bad = [f"{i}番（{n.at} {n.note}「{n.word}」aim={n.aim}）"
           for i, n in enumerate(notes, 1) if not 1 <= int(n.aim) <= n_aims]
    if not bad:
        return ""
    return (f"どの狙いの音か（aim）が無い・範囲外の音がある: {' / '.join(bad[:12])}"
            + (f" ほか{len(bad) - 12}音" if len(bad) > 12 else "")
            + f"——狙いは {n_aims}個あるので、全部の音に 1〜{n_aims} の番号を書く"
              "（狙いごとに音量と声が違うため、番号が無いと鳴らし方が決まらない）")


# ---- 声部の形式の検査と、差分での直し（2026-09-29）------------------------------------------
#
# ★それまで声部の形式は型（`VoiceNote`・`voice_part_type`）が見る「位置と音名が読めるか・狙いの番号」だけで、
#   **1音1モーラ**はスキーマの説明にしか無く、**ひらがな**（`song_lyrics.check_kana`）は詞にだけ掛かり、
#   **1つの狙い＝1本の声部の中で音が重なる**のは誰も見ていなかった（1本の声部は同時に1音しか歌えない）。
#   型で差し戻すと、1音の書き違いで声部が丸ごと書き直しになる（直らなければ声部が0本）ので、ほかの工程と同じ形
#   （`song_fix.StepFixer`: 検査→所見を返す→直す音だけ差分で置き換えさせる）にする。
#   残ったら声部の工程で曲づくりを止める（`song_fix.StepUnfixed`・2026-09-30。以前は書いたとおりに鳴らして所見に出していた）。

#: 前の音を伸ばす印（詞と同じ `_`）→ V6 の伸ばす歌詞 `-`（組み立ての訳し方・`voice_to_track`）
_V6_WORD = {"_": "-"}


def _voice_word_problems(word: str) -> list[str]:
    """声部の言葉1つの所見（1モーラのひらがな、または前の音を伸ばす印 `_`）。文法 `word = "_" | mora`（仕様書 §13・前後の空白は除く）。

    ★2026-10-02（洗い出し H3）: 印（`_` `~` `*`）と途中の空白・小書きで始まる言葉も所見にした。以前は詞の検査
      （`check_kana`＝譜割りの印と空白を指摘しない）とモーラの数え方（印と空白を数えない）だけで見ていたので、`か_`・`か~`・`か*`・
      `き ゃ`・`~か`・小書きだけの `ゃ` が通り、組み立てがそのまま V6 の歌詞にしていた。
    """
    from song_lyrics import check_kana                     # noqa: E402（詞と同じ決まり・循環を避けてここで）
    from lyric_slots import _SMALL_KANA, HOLD, JOIN, SKIP, moras   # noqa: E402（モーラの数え方・印は1か所）
    w = (word or "").strip()
    if w == HOLD:                                          # 前の音を伸ばす印（詞と同じ）
        return []
    if w == "-":
        # ★2026-09-30（記-4・本人「記法側で」）: 伸ばす印は詞と同じ `_` にそろえた。ABC の歌詞の決まりでは `_` が
        #   「前の音節を伸ばす」、`-` は「語を音節に分ける」で、声部の `-` は標準と逆だった。2通りを残さない
        return ["伸ばす印は `_`（`-` ではない。詞と同じ書き方）"]
    out: list[str] = []
    if HOLD in w:
        out.append(f"伸ばす印 `_` は1つの音に単独で書く（『{w}』ではなく、言葉の音と `_` の音に分ける）")
    if JOIN in w or SKIP in w:
        out.append(f"`~` `*` は詞の譜割りの印で、声部の言葉には書けない（『{w}』。1つの音に1モーラのひらがなだけ）")
    if any(c.isspace() for c in w):
        out.append(f"言葉の途中に空白を入れない（『{w}』→『{''.join(w.split())}』）")
    ng = check_kana(w)
    if ng:
        out.append("言葉は**ひらがなだけ**——" + "／".join(ng) + "（前の音を伸ばすなら「ー」ではなく `_`）")
    if out:
        return out
    if w[:1] and w[0] in _SMALL_KANA:
        return [f"小書きの『{w[0]}』で始まる——小書きは前の字とつなげて1モーラ（「きゃ」のように2文字で書く）"]
    ms = moras(w)
    if not ms:
        return ["言葉が空（歌う1モーラを書く。前の音を伸ばすなら `_`）"]
    if len(ms) > 1:
        return [f"1つの音に{len(ms)}モーラ（{'・'.join(ms)}）。**1つの音には1モーラ**——音を分けて1モーラずつ書く"]
    if len(w) > 2:                                         # 1モーラ＝ひらがな1字か、ひらがな＋小書き1つ（「かゃゃ」は数えると1つになる）
        return [f"『{w}』は1モーラの形ではない——ひらがな1字か、ひらがな＋小書き1つ（「か」「きゃ」）で書く"]
    return []


#: ★**声部の言葉の記法＝型1つ**（2026-09-29・`song_checks.Notation`）。詞の行（`song_lyrics.LYRIC_LINE`）と違い、
#:   1つの音に1モーラ（か伸ばす印 `_`）だけを書く。声部の工程（`VoicesStep.written`）が宣言する。
#:   組み立て（`voice_to_track`）は書いた言葉をそのまま歌詞に置き、伸ばす印 `_` だけ V6 の `-` に訳す（2026-09-30・記-4）
VOICE_WORD = Notation("声部の言葉（1モーラのひらがな・伸ばす印 _）", _voice_word_problems)


def _span_text(sections, bars_of: dict | None, *, name: bool = True) -> str:
    """区間名 → 「chorus は 26〜33小節目」（時間軸 `bars_of` に無い名前はそのまま）。`name=False` なら「・26〜33小節目」だけ（2026-10-04 組F3）。"""
    from song_checks import resolve_section                # noqa: E402
    out = []
    for s in sections or []:
        r = resolve_section(s, bars_of) if bars_of else None
        if r is None:
            if name:
                out.append(str(s))
            continue
        b0, b1 = bars_of[r]
        out.append(f"{r} は {b0}〜{b1}小節目" if name else f"・{b0}〜{b1}小節目")
    return ("／" if name else "").join(out)


def voice_findings(part: VoicePart, bar_ticks: int, n_aims: int, *, plan: HarmonyPlan | None = None,
                   bars_of: dict | None = None, shift_at=None) -> list[str]:
    """声部の**形式の検査**（所見の文だけ。直さない）: 狙いの番号・計画した狙いに1音も無い・歌える範囲の外・
    1つの狙いの中で音が重ならないか・狙いの区間の外に置いた音（`plan` と `bars_of`＝区間→(開始小節, 終了小節) を渡したとき）。

    言葉が1モーラのひらがなかと、位置の拍が1小節に収まるかは、記法の型（`VOICE_WORD`・`position_notation`）が見る
    （`VoicesStep.written`）。
    ★**歌える範囲（`C2`〜`C7`）は鳴る高さで見る**（2026-10-03・判-15）。`shift_at`（tick → 移調量・`section_shift_at`）を渡すと
      書いた高さ（C 基準）にその位置の量を足して見る。渡さなければ書いた高さのまま（区間を持たない呼び方）。
    """
    from song_llm import PITCH                              # noqa: E402（音名の読み手＝型と同じ1つ・循環回避）
    out: list[str] = []
    aim_why = _aim_problem(part.notes, n_aims)
    if aim_why:
        out.append(aim_why)
    groups: dict[int, list[tuple[int, int, int]]] = {}     # 狙いの番号 → [(開始, 終わり, 項目の番号)]
    pos = position_notation(bar_ticks / _Q)
    for i, n in enumerate(part.notes, 1):
        t = at_to_tick(n.at, bar_ticks)
        k = int(n.aim) if n_aims > 1 else 1
        # ★**歌える範囲の外＝V6 で鳴らない**（2026-10-03・判-15。以前は `check_voices` が記録に残すだけだった）。答えが1つ
        key = PITCH.read(n.note)
        if key is not None:
            sh = int(shift_at(t)) if (shift_at is not None and t is not None) else 0
            if not VOICE_LO <= key + sh <= VOICE_HI:
                out.append(f"{i}番（{n.at} {n.note}「{n.word}」）: 鳴る高さ {spell_pitch_name(key + sh)}"
                           + (f"（区間の調へ {sh:+d}半音移したあと）" if sh else "")
                           + f"が**歌える範囲（{spell_pitch_name(VOICE_LO)}〜{spell_pitch_name(VOICE_HI)}）の外**——V6 で鳴らない。"
                           f"この位置では {spell_pitch_name(VOICE_LO - sh)}〜{spell_pitch_name(VOICE_HI - sh)}"
                           "（表と同じ物差し）の中に書く（オクターブを移すなど）")
        if t is None or pos.problems(n.at) or not 1 <= k <= max(1, n_aims):
            continue
        groups.setdefault(k, []).append((t, t + beats_to_ticks(n.beats), i))
    # ★**計画した狙いに1音も無い**（2026-10-03・判-15。2026-08-21 に計画したハモリが黙って消えていた）。狙いの番号の所見が
    #   あるときは見ない（番号を直すのが先・同じ音に2つの所見を重ねない）
    if not aim_why:
        for k in range(1, n_aims + 1):
            if not any(n_aims == 1 or int(n.aim) == k for n in part.notes):   # 狙いが1つなら全部その狙い（番号の所見は上）
                what = (f"（{'／'.join(plan.parts[k - 1].sections)}: {plan.parts[k - 1].aim}）"
                        if plan is not None and k <= len(plan.parts) else "")
                # 区間の小節を添える（2026-10-04 組F3・3‑1。以前は「その区間に」だけで、何小節目へ足すかが依頼文から引けなかった）
                span = _span_text(plan.parts[k - 1].sections, bars_of) if plan is not None and k <= len(plan.parts) else ""
                out.append(f"狙い{k}{what}に音が1つも無い——前の工程が計画した狙いなので、その区間"
                           + (f"（{span}）" if bars_of and span else "") + f"に1音以上を`add` で足す（aim は {k}）")
    # ★**狙いの区間の外に置いた音**（2026-09-30・判-16②）。狙いごとに区間（`sections`）を決めてあるのに、以前は位置を見ず、
    #   狙い1＝chorus の音を verse に書いても、その狙いの音量・声で verse に鳴った。答えが1つに決まるので形式に置く。
    #   音が狙いの区間に少しでも掛かっていれば中とする（区間の頭へ食い込む音・区間をまたいで伸ばす音は通す）。
    #   区間名が設計に無い狙いはここでは見ない（ハモリの計画の工程が設計と照らして直させる＝`HarmonyPlanStep`）
    if plan is not None and bars_of:
        from song_checks import resolve_section            # noqa: E402
        for k, spans in sorted(groups.items()):
            if not 1 <= k <= len(plan.parts):
                continue
            want = [resolve_section(s, bars_of) for s in plan.parts[k - 1].sections]
            if not want or None in want:
                continue
            ranges = [((bars_of[s][0] - 1) * bar_ticks, bars_of[s][1] * bar_ticks) for s in want]
            for a0, a1, i in spans:
                if not any(a0 < r1 and a1 > r0 for r0, r1 in ranges):
                    n = part.notes[i - 1]
                    where = next((s for s, (b0, b1) in bars_of.items() if b0 <= a0 // bar_ticks + 1 <= b1), "区間の外")
                    # 区間の小節を添える（2026-10-04 組F3・3‑1。移し先の小節が所見から分かるように）
                    out.append(f"{i}番（{n.at} {n.note}「{n.word}」）: 狙い{k}の区間（{_span_text(want, bars_of)}）の外"
                               f"（{where}{_span_text([where], bars_of, name=False)}）に置いている——"
                               f"狙い{k}の区間の中に置くか、その区間を狙う別の番号があれば aim をその番号にするか、この音をやめる")
    for k, spans in sorted(groups.items()):
        spans.sort()
        for (a0, a1, ia), (b0, _b1, ib) in zip(spans, spans[1:]):
            if b0 < a1:
                out.append(f"{ia}番と{ib}番: 同じ狙い（{k}）の中で音が重なっている（{ia}番が終わる前に{ib}番が始まる）。"
                           f"1つの狙いは1本の声部で、**同時に1音しか歌えない**——前の音の長さか、次の音の位置を直す")
    return out


class VoiceNoteFix(BaseModel):
    """声部の差分1つぶん。番号で指した音を、書いた音（1つ以上）で置き換える。"""
    # ※`why` は読み手なし（記録にも依頼文にも出ない）。直す前に考えを書かせる欄として残す（2026-09-30・出-13）
    no: JsonInt = Field(ge=1, description="直す音の番号（依頼文の「いまの声部」の『◯番』・1始まり）")
    drop: JsonBool = Field(default=False, description="この音をやめるなら true（`notes` は空のまま・一緒に書かない）")
    notes: list[VoiceNote] = Field(default_factory=list, description=(
        "その音と**置き換える音**（1つ以上）。1つの音に2モーラ以上を書いていたなら、1モーラずつの音に分けて並べる"
        "（位置と長さも分ける）。直さない欄も含めて、音を丸ごと書く"))
    why: str = Field(default="", description="どの所見にどう応えたか（一言）")

    # ★**消すと置き換える音を一緒に書いた項目は型が差し戻す**（2026-10-03・記B・仕様書 §15）。以前は drop が黙って勝ち、
    #   書いた音は読まれなかった。歌い方・伴奏の差分と同じ扱い
    @model_validator(mode="after")
    def _drop_alone(self):
        if self.drop and self.notes:
            raise ValueError(f"{self.no}番: drop と notes を一緒に書いた——やめるなら drop: true だけ、置き換えるなら notes だけ")
        return self


class VoicePatch(PatchStatus):
    """声部の直し（差分）。**直さない音は書かない**——書かなかった音はいまのまま残る。"""
    # ※`reason` は読み手なし（記録にも依頼文にも出ない）。直す前に考えを書かせる欄として残す（2026-09-30・出-13）。
    #   `trouble` は直しの記録に入る（`VoicesStep.apply`）
    reason: str = Field(description="何をどう直したか（全体で一言）")
    fixes: list[VoiceNoteFix] = Field(default_factory=list, description="直す音だけ")
    # ★**足す音**（2026-10-03・判-15）。「計画した狙いに1音も無い」を形式の所見にしたので、番号で指せない（いまの声部に無い）
    #   音を書く欄が要る。足した音は位置の順に、いまの声部の間へ入る（書いてある音の並びと中身は変えない）
    add: list[VoiceNote] = Field(default_factory=list, description=(
        "**新しく足す音**（いまの声部に無い音）。計画した狙いに音が1つも無いと言われたら、その狙いの音をここに書く"
        "（位置・長さ・音名・言葉・狙いの番号を全部）。足さないなら空"))


_voice_fixer: Agent[None, VoicePatch] = Agent(
    pick_model("HARMONY", "flash"), output_type=VoicePatch,
    system_prompt=prompt("song_voices_fix"),
)


from song_fix import StepFixer                              # noqa: E402（工程の形）


# ---- 差分の番号の検証（2026-10-02・洗い出し）は工程の形 `song_fix`（`PatchNumbers`・`guard_patch`）へ移した。
#   声部・ハモリの計画・奏法・歌の崩し・歌い方と、編成・伴奏・彩りの差分の検証が同じ1つを使う（組「洗い出し-型と残り」）
from song_fix import PatchNumbers, guard_patch            # noqa: E402


# ---- ハモリの計画の形式の検査と、差分での直し（2026-09-30・判-14①）---------------------------------------
#
# ★以前は区間名（`HarmonyPart.sections`）を設計と照らしていなかった。名前がずれると、声部の LLM が表の中で狙いの区間を
#   探せず、声部の検査（狙いの区間の外の音・判-16②）も効かない。区間名は設計の照合（親の決まり `song_checks.against_design`。
#   前後の空白だけ取り除いて設計の綴りにする＝`canonize`。大文字小文字・途中の空白の違いは所見）、歌の無い区間と区間の無い狙いは工程の検査（指示書に
#   「歌の無い区間には付けられません」とある）。所見を返し、直す狙いだけ差分で直させる。直らなければハモリの工程で止まる。

class HarmonyPartFix(BaseModel):
    """ハモリの計画の差分1つぶん（番号で指した狙いの区間を書き直すか、狙いをやめる）。"""
    no: JsonInt = Field(ge=1, description="直す狙いの番号（【いまのハモリの計画】の『◯番』・1始まり）")
    sections: list[str] = Field(default_factory=list, description=(
        "その狙いを置く区間を**書き直した全体**（設計の区間名をそのまま・歌のある区間だけ）。やめるなら空で drop"))
    drop: JsonBool = Field(default=False, description="この狙いをやめるなら true（`sections` は空のまま・一緒に書かない）")

    # ★**消すと書き直す区間を一緒に書いた項目は型が差し戻す**（2026-10-03・記B・仕様書 §15）。以前は drop が黙って勝った
    @model_validator(mode="after")
    def _drop_alone(self):
        if self.drop and self.sections:
            raise ValueError(f"{self.no}番: drop と sections を一緒に書いた——やめるなら drop: true だけ、区間を書き直すなら sections だけ")
        return self


class HarmonyPlanPatch(PatchStatus):
    """ハモリの計画の直し（差分）。**直す狙いだけ**を書く——書かなかった狙いはいまのまま残る。"""
    # ※`reason` は読み手なし（記録にも依頼文にも出ない）。直す前に考えを書かせる欄として残す。`trouble` は直しの記録に入る
    reason: str = Field(description="何をどう直したか（全体で一言）")
    fixes: list[HarmonyPartFix] = Field(default_factory=list, description="直す狙いだけ")


_harmony_fixer: Agent[None, HarmonyPlanPatch] = guard_patch(Agent(
    pick_model("HARMONY", "flash"), output_type=HarmonyPlanPatch,
    system_prompt=prompt("song_harmony_fix"),
))
guard_patch(_voice_fixer)


class HarmonyPlanStep(PatchNumbers, StepFixer):
    """ハモリの計画の形式の検査と差分の直し（2026-09-30・判-14①）。値は `HarmonyPlan`。

    - 狙いの区間名が設計の区間名か（親の決まり `against_design`）
    - 狙いの区間が歌のある区間か・狙いに区間が1つはあるか（工程の `check`）
    ★**アカペラ（`form="acappella"`）は歌の無い区間（前奏・間奏・後奏）にも置ける**（2026-10-03・F7・仕様書 §15-4）。伴奏が無く、
      声部が和声とリズムを持つので、前奏から通しで敷く使い方を許す。検査を消すのではなく形で決まりを変える（歌もの・インストは今のまま）。
      形を変えたときに残す計画へ当てる検査（`song_form_checks.form_findings`）も同じ `check` を呼ぶ。
    """
    key = "harmony"
    label = "ハモリ"
    patch_what = "ハモリの計画"
    agent = _harmony_fixer
    return_hint = ("**直す狙いだけ**を fixes に入れてください（番号で指し、その狙いの区間を書き直した全体を sections に書く。"
                   "やめるなら drop）。直さない狙いは書かないでください。")

    def __init__(self, design_sections, form: str = "song"):
        # 設計の区間が渡されないとき（区間の無い呼び方）は照らさない（全部の名前が「設計に無い」になるため）
        self.design = [s for s in (design_sections or []) if getattr(s, "kind", None)] or None
        self.form = form or "song"

    def design_items(self, plan: HarmonyPlan):
        return [(f"狙い{i}の sections", s, None) for i, p in enumerate(plan.parts, 1) for s in p.sections]

    def rename_sections(self, plan: HarmonyPlan, mapping: dict[str, str]) -> HarmonyPlan:
        return plan.model_copy(update={"parts": [
            p.model_copy(update={"sections": [mapping.get(s, s) for s in p.sections]}) for p in plan.parts]})

    def check(self, plan: HarmonyPlan) -> list[str]:
        from song_checks import resolve_section             # noqa: E402
        if not self.design:
            return []
        names = [s.kind for s in self.design]
        # アカペラは歌の無い区間にも置ける（F7）——「歌の無い区間」を空にして、区間の決まりを形で変える
        quiet = set() if self.form == "acappella" else {s.kind for s in self.design if not s.has_vocal}
        sung = [k for k in names if k not in quiet]
        out: list[str] = []
        for i, p in enumerate(plan.parts, 1):
            if not p.sections:
                out.append(f"狙い{i}: 区間（sections）が無い——重ねる区間を1つ以上書く（"
                           + ("区間: " if self.form == "acappella" else "歌のある区間: ") + f"{' / '.join(sung)}）")
            for s in p.sections:
                if resolve_section(s, names) in quiet:
                    out.append(f"狙い{i}の sections: 『{s}』は歌の無い区間——声は歌のある区間にだけ重ねる"
                               f"（歌のある区間: {' / '.join(sung)}）")
        return out

    def show(self, plan: HarmonyPlan) -> str:
        rows = "\n".join(f"  {i}番: sections={p.sections} ／ aim: {p.aim}（濃さ: {p.density}）"
                         for i, p in enumerate(plan.parts, 1))
        quiet = {s.kind for s in self.design or [] if not s.has_vocal}
        mark = "（歌なし・アカペラなので置ける）" if self.form == "acappella" else "（歌なし）"
        return (f"【いまのハモリの計画】\n{rows}\n\n【設計の区間名】"
                + " / ".join(s.kind + (mark if s.kind in quiet else "") for s in self.design or []))

    def patch_count(self, plan: HarmonyPlan) -> int:
        return len(plan.parts)

    def patch_items(self, patch: HarmonyPlanPatch) -> list[tuple[int, bool]]:
        return [(f.no, not f.drop and not f.sections) for f in patch.fixes or []]

    def apply(self, plan: HarmonyPlan, patch: HarmonyPlanPatch) -> tuple[HarmonyPlan, str]:
        """差分をそのまま当てる。**書かれていない狙いは変えない**（元の計画は変えない）。
        範囲外・重なった番号・中身の無い直しは差分の検証（`song_fix.guard_patch`）が差し戻す（2026-10-02・洗い出し）。"""
        parts = list(plan.parts)
        done: list[str] = []
        drop: set[int] = set()
        fixes = list(getattr(patch, "fixes", None) or [])
        if why := self.patch_problem(plan, patch):
            done.append(f"当てなかった: {why}")              # 型を通らずに来た差分だけ（以前は黙って捨てた）
        seen: set[int] = set()
        for f in fixes:
            if not 1 <= f.no <= len(parts) or f.no in seen:
                continue
            seen.add(f.no)
            if f.drop:
                drop.add(f.no)
                done.append(f"狙い{f.no}をやめた")
            elif f.sections:
                parts[f.no - 1] = parts[f.no - 1].model_copy(update={"sections": [s.strip() for s in f.sections]})
                done.append(f"狙い{f.no}の区間を {'／'.join(parts[f.no - 1].sections)} に")
        parts = [p for i, p in enumerate(parts, 1) if i not in drop]
        note = "／".join(done) if done else "直しなし"
        if getattr(patch, "trouble", ""):
            note += f"／直さなかった理由: {patch.trouble}"
        return plan.model_copy(update={"parts": parts}), note


class VoicesStep(PatchNumbers, StepFixer):
    """声部の形式の検査と、差分での直し（2026-09-29）。記法は `written`（言葉・`VOICE_WORD`）、工程の検査は `voice_findings`
    （狙いの番号・狙いに1音も無い・歌える範囲の外・重なり・狙いの区間の外）、差分は `VoicePatch`（音を番号で置き換える・消す・
    番号の無い音を足す）。

    保存から戻した声部（再生）は直させない（親の `run`）。直らなければ声部の工程で止まる（`StepUnfixed`・2026-09-30）。
    """
    key = "voices"
    label = "声部"
    agent = _voice_fixer
    return_hint = ("**直す音だけ**を fixes に入れてください（番号で指し、置き換える音を1つ以上書く。やめるなら drop）。"
                   "直さない音は書かないでください。音が1つも無い狙いの音は add に書いて足してください。")

    def __init__(self, plan: HarmonyPlan, bar_ticks: int, bars_of: dict | None = None, shift_at=None, table: str = ""):
        """`bars_of` は区間→(開始小節, 終了小節)（時間軸）。渡すと、狙いの区間の外に置いた音を見る（判-16②）。
        `shift_at`（tick → 移調量・`section_shift_at`）を渡すと、歌える範囲を移したあとの鳴る高さで見る（判-15）。
        `table` は声部を書くときに見せた【この曲の中身】（歌とコードの表・歌える範囲の行・アカペラの歌の無い区間）。直しの依頼文の頭にも
        同じものを出す（2026-10-04 組F3・3‑1。以前は直しに表が無く、指示書の「表の音高と同じ物差し」が依頼文に無かった）。"""
        self.plan, self.bar_ticks, self.bars_of = plan, bar_ticks, dict(bars_of or {})
        self.shift_at, self.table = shift_at, table

    def written(self, part: VoicePart):
        # ※位置が読めるかと音名は型（`VoiceNote` の検証）が差し戻す（位置の読み方は `POSITION` の1つ）。拍が1小節の長さを
        #   超えていないかは拍子を知っている記法（`position_notation`・2026-09-30・記-15）でここに宣言する——型は拍子を知らないため
        pos = position_notation(self.bar_ticks / _Q)
        for i, n in enumerate(part.notes, 1):
            where = f"{i}番（{n.at} {n.note}「{n.word}」）"
            yield where, pos, n.at
            yield where, VOICE_WORD, n.word

    def check(self, part: VoicePart) -> list[str]:
        return voice_findings(part, self.bar_ticks, len(self.plan.parts), plan=self.plan, bars_of=self.bars_of,
                              shift_at=self.shift_at)

    def show(self, part: VoicePart) -> str:
        # 狙いの区間に小節を添え、声部を書いたときと同じ表を頭に出す（2026-10-04 組F3・3‑1）
        aims = "\n".join(f"  {i}. {'／'.join(p.sections)}: {p.aim}"
                         + (f"（{_span_text(p.sections, self.bars_of)}）" if self.bars_of else "")
                         for i, p in enumerate(self.plan.parts, 1))
        rows = "\n".join(f"  {i}番: at={n.at} beats={n.beats:g} note={n.note} word={n.word} aim={n.aim}"
                         for i, n in enumerate(part.notes, 1))
        head = f"【この曲の中身】（声部を書いたときと同じ表）\n{self.table}\n\n" if self.table else ""
        return f"{head}【前の工程が決めた狙い】\n{aims}\n\n【いまの声部】（{part.title}）\n{rows}"

    def patch_count(self, part: VoicePart) -> int:
        return len(part.notes)

    def patch_items(self, patch: VoicePatch) -> list[tuple[int, bool]]:
        return [(f.no, not f.drop and not f.notes) for f in patch.fixes or []]

    def apply(self, part: VoicePart, patch: VoicePatch) -> tuple[VoicePart, str]:
        """差分をそのまま当てる。**書かれていない音は変えない**（元の声部は変えない）。
        範囲外・重なった番号・中身の無い直しは差分の検証（`song_fix.guard_patch`）が差し戻す（2026-10-02・洗い出し）。"""
        repl: dict[int, list[VoiceNote]] = {}
        done: list[str] = []
        if why := self.patch_problem(part, patch):
            done.append(f"当てなかった: {why}")              # 型を通らずに来た差分だけ（以前は黙って捨てた）
        for f in getattr(patch, "fixes", None) or []:
            if not 1 <= f.no <= len(part.notes) or f.no in repl:
                continue
            if f.drop:
                repl[f.no] = []
                done.append(f"{f.no}番をやめた")
            elif f.notes:
                repl[f.no] = [n.model_copy(deep=True) for n in f.notes]
                done.append(f"{f.no}番を{len(f.notes)}音に書き直した" if len(f.notes) > 1 else f"{f.no}番を書き直した")
        notes = [x for i, n in enumerate(part.notes, 1) for x in (repl[i] if i in repl else [n])]
        # 足す音（判-15）: 位置の順に間へ入れる（いまの音の並びは変えない。読めない位置の音は末尾＝型が先に差し戻すので来ない）
        added = list(getattr(patch, "add", None) or [])
        for a in added:
            t = at_to_tick(a.at, self.bar_ticks)
            at = next((j for j, x in enumerate(notes)
                       if t is not None and (xt := at_to_tick(x.at, self.bar_ticks)) is not None and xt > t), len(notes))
            notes.insert(at, a.model_copy(deep=True))
        if added:
            done.append(f"{len(added)}音を足した")
        note = "／".join(done) if done else "直しなし"
        if getattr(patch, "trouble", ""):
            note += f"／直さなかった理由: {patch.trouble}"
        return part.model_copy(update={"notes": notes}), note


def voices_for(plan: HarmonyPlan, score,
               *, verbose: bool = True, song=None,
               bars_of: dict | None = None, shift_of=None) -> tuple[list[VoiceTrack], list[str], "VoicePart | None"]:
    """**楽譜を渡すだけ**で声部を書く（呼ぶ側に表の組み立てを書かせない）。

    ★呼び出しは3か所（新規作成・カバーの編曲・編集のやり直し）ある。
      表の組み立てを各所に書くと**必ずどれかがずれる**ので、ここ1か所に集める
      （工程を足すと触る所が7つある、と同じ理由）。
    `bars_of`（区間→(開始小節, 終了小節)）は狙いの区間の外の音を見るため（判-16②）。渡されなければ曲（`song.bars_of`）から引く。
    `shift_of`（区間 → 移調量・旋律とコードを移したのと同じ `song_place.section_keys` のもの）を渡すと、表を C メジャー／A マイナー
    基準（移調前）で見せ、書いた声部を区間の調へ移す（2026-09-30・記-3）。渡さなければ表の音高が鳴る高さ。
    ★カバーも推定した調の `shift_of` を渡す（2026-10-01・判-C。それまでカバーは渡さず、表も声部も実音だった）。

    ※以前（2026-09-27〜）は `changes` に「機械が捨てた音・どの狙いか分からず鳴らさなかった音」を入れて記録させていた。
      2026-09-30 から、形式の所見が直らなければ声部の工程で止まり（`StepUnfixed`）、組み立てで捨てる音は無い（来たら `AssemblyError`）。
    """
    from pathlib import Path as _P
    from song_table import key_regions, song_table          # noqa: E402（engine 内・循環しない）

    from song_design import bar_ticks as _meter_bar_ticks   # noqa: E402（拍子の読み手は1つ・2026-10-01 まで写しがあった）
    bar = _meter_bar_ticks(score.time_signature or "4/4")
    if bars_of is None:
        bars_of = getattr(song, "bars_of", None) or None
    # ★**表と書く高さの物差しを1つにする**（2026-09-30・記-3）。移調量は旋律・コードと同じ `shift_of` から tick ごとに引き
    #   （`section_shift_at`）、表の表示（移調前へ戻す）と組み立て（区間の調へ移す）の両方にこの1つを渡す
    shift_at = section_shift_at(shift_of, bars_of, bar)
    # 伴奏 MIDI はまだ書かれていないことがある。**無くても歌の表は組める**
    table, *_ = song_table(score, _P("_no_accomp_yet"), shift_at=shift_at,
                           keys=(key_regions(song) or None) if song is not None and shift_at else None)
    rng = sing_range_text(shift_at, bar, bars_of)
    if mel := melody_range_text(score.vocal_notes, shift_at):
        rng = f"{rng}\n{mel}"
    # ★アカペラ（2026-10-03・F7）は歌の無い区間（前奏・間奏・後奏）にも声部を置けるので、その区間の小節とコードを添える
    #   （表は主旋律の音の行で組むので、歌の無い区間はコードが出ず、最後の音より後ろ＝後奏は表から落ちる）
    if (getattr(song, "form", "song") or "song") == "acappella" and shift_at is not None:
        if quiet := quiet_sections_text(song, bars_of):
            table = f"{table}\n\n{quiet}"
    return write_voices(plan, f"{table}\n\n{rng}", score.vocal_notes, bar, verbose=verbose,
                        tempo=float(getattr(score, "tempo", 0) or 0) or None, song=song, bars_of=bars_of,
                        shift_at=shift_at)


def quiet_sections_text(song, bars_of: dict | None) -> str:
    """**歌の無い区間の小節とコード**（アカペラの声部に見せる・2026-10-03・F7）。無ければ空。

    コードはコードの工程が書いたまま（C 基準・移調前＝声部に見せる表と同じ物差し）。小節は時間軸（`bars_of`＝楽譜の頭から数えた小節）。
    """
    design, chords = getattr(song, "design", None), getattr(song, "chords", None)
    if design is None or not bars_of:
        return ""
    ch = {c.kind: list(c.chords_per_bar) for c in (getattr(chords, "sections", None) or [])}
    rows: list[str] = []
    for s in design.sections:
        if s.has_vocal or s.kind not in bars_of:
            continue
        b0, b1 = bars_of[s.kind]
        cells = ch.get(s.kind) or []
        body = " / ".join(f"{b0 + i}: {c}" for i, c in enumerate(cells[:b1 - b0 + 1]))
        rows.append(f"  {s.kind}（{b0}〜{b1}小節）" + (f" コード（小節: コード）{body}" if body else " コードなし"))
    if not rows:
        return ""
    return ("【歌の無い区間】（アカペラなので声部を置ける。表には主旋律の音の行しか無いので、ここで小節とコードを見る。"
            "コードは表と同じ C 基準）\n" + "\n".join(rows))


def sing_range_text(shift_at, bar_ticks: int, bars_of: dict | None) -> str:
    """声の推奨音域の1行。**表の物差しで見せる**——検査（`check_voices`）は移したあとの鳴る高さで `SING_LO`〜`SING_HI`（D3〜D6）を見るので、
    C 基準で書く LLM には区間ごとに移調量を引いた範囲を渡す（旋律に移調前の上限 `_top_c` を渡すのと同じ考え方）。
    声部（`voices_for`）とハモリ追加（`song_llm.generate_added_voice`）が同じ1つを使う（2026-10-02・ず-12③）。"""
    from song_table import shift_regions                    # noqa: E402
    lo, hi = spell_pitch_name(SING_LO), spell_pitch_name(SING_HI)
    if shift_at is None:
        return f"声の推奨音域: {lo}〜{hi}（この表の音高のまま鳴る）"
    n_bars = max((int(b1) for _b0, b1 in (bars_of or {}).values()), default=1)
    return (f"声の推奨音域は鳴る高さで {lo}〜{hi}。この表の物差し（C 基準）では: "
            + " ／ ".join(f"{b}小節〜 {spell_pitch_name(SING_LO - sh)}〜{spell_pitch_name(SING_HI - sh)}"
                          for b, sh in shift_regions(shift_at, bar_ticks, n_bars)))


def melody_range_text(notes: list[SongNote], shift_at=None) -> str:
    """主旋律の音域の1行（**表の物差し**で・2026-10-04）。声部を主旋律より下・同じ帯域・上のどこに置くかを決める材料
    （『夜のしじまに灯す声』で声部が主旋律とほぼ同じ帯域に集まった）。表の音高の欄から読めることを1行にまとめただけ。主旋律が無ければ空。"""
    ks = [n.key - (shift_at(n.start_tick) if shift_at else 0) for n in notes or []]
    if not ks:
        return ""
    return f"主旋律の音域（この表の物差しで）: 最低 {spell_pitch_name(min(ks))}〜最高 {spell_pitch_name(max(ks))}"


def spell_pitch_name(k: int) -> str:
    """MIDI → 表と同じ綴りの音名（`A#3`）。推奨音域の行に使う（表の綴りは `song_table.spell_pitch` の1つ）。"""
    from song_table import spell_pitch                      # noqa: E402
    return spell_pitch(k)


def write_voices(plan: HarmonyPlan, table: str, notes: list[SongNote], bar_ticks: int,
                 *, verbose: bool = True, tempo: float | None = None, song=None,
                 bars_of: dict | None = None, shift_at=None) -> tuple[list[VoiceTrack], list[str], "VoicePart | None"]:
    """**2段目**。プランの狙いを、実際に鳴る音にする（`build_harmony` の置き換え）。

    渡すのは `inspect_song` の表そのもの——`歌詞|音高|小節:拍|長さ|コード|和音での役割|ドラム|<声部>`。
    **位置も長さも音楽の言葉で書いてある**ので、**同じ言葉で返してもらう**。
    ★**自前の形式を作らない**（2026-09-22 本人指摘: 「**前にやった『この小節ではこの音が
      鳴ってる』みたいなデータをちゃんと参照として入れてるんだよね**」——入れていなかった）。

    `shift_at`（tick → 移調量・`section_shift_at`）は表を C 基準で組んだときに渡す。書いた音をその量だけ区間の調へ移し、
    音域の検査（`check_voices`）は移したあとの鳴る高さで見る（2026-09-30・記-3）。

    戻り値は (声部, 所見, LLM が書いたものそのもの)。**所見は直さずに出す**——判断は関門の LLM と人。
    3つ目は呼び側が `Song.voices` へしまうと保存され、再生と「続きから」が使える（2026-09-24）。
    ★所見に LLM の理由（`声部:…`）を入れない（2026-09-30・判-16①）。以前は先頭に足していたので、所見の欄が空にならず、
      声部を作った曲は毎回「うまくいかなかった工程」（`Song.troubles`）に並んだ。理由は記録の reason（`VoicePart.reason`）が持つ。
      新規の流れ（`song_band.run_voices`）は呼び側で外していたが、カバーの編曲（`song_arrange_cover`）には入ったままだった。
    """
    if not plan.parts:
        return [], [], None
    # ★**狙いに番号を振って見せる**（2026-09-27）。音ごとに `aim` で番号を返してもらい、狙いごとの
    #   音量・声のまま別の声部にする（`voices_to_tracks`）。
    aims = "\n".join(
        f"  {i}. {'／'.join(p.sections)}: {p.aim}（濃さ: {p.density}"
        + (f"・声 {p.voice_style}" if p.voice_style else "") + "）"
        for i, p in enumerate(plan.parts, 1))
    # ★狙いが2つ以上なら、`aim` の無い・範囲外の音を**型で差し戻す**（`voice_part_type`・2026-09-27・監査）
    r = _llm_run(_voicer, f"【この曲の中身】\n{table}\n\n"
                          f"【前の工程が決めた狙い】\n{aims}\n\n"
                          f"（{plan.reason[:200]}）", label="声部",
                 output_type=voice_part_type(len(plan.parts)))
    _rec_usage(_voicer, r, "声部")
    got: VoicePart = r.output
    if type(got) is not VoicePart and isinstance(got, VoicePart):   # 書くときだけの型 → 保存する形へ戻す
        got = VoicePart.model_validate(got.model_dump())
    # ★形式の検査（1音1モーラのひらがな・狙いの番号・1つの狙いの中で重ならない）で所見を返し、直す音だけ差分で直させる
    #   （2026-09-29・`VoicesStep`）。保存から戻した声部は直させない。直らなければ声部の工程で止まる（`StepUnfixed`・2026-09-30。
    #   以前は書いたとおりに鳴らして所見に出していた）
    step = VoicesStep(plan, bar_ticks, bars_of, shift_at=shift_at,   # 歌える範囲は移したあとの鳴る高さで（判-15）
                      table=table)                                   # 直しの依頼文にも同じ表（2026-10-04 組F3・3‑1）
    got, _ = step.run(got, song, verbose=verbose)          # 曲を渡すと直しの記録（`voices_fix`）がほかの工程と同じく残る
    fixed = [step.result.summary()] if step.result.rounds else []
    tracks, warns = voices_to_tracks(got, plan, bar_ticks, tempo=tempo, shift_at=shift_at)   # C 基準 → 区間の調（記-3）
    warns = fixed + warns
    # ★作り方によらず同じ検査を掛ける（移したあとの鳴る高さで見る）（推奨音域の外。曲の9割に乗るは 2026-10-03 に外した）
    warns += check_voices(notes, tracks, getattr(song, "form", "song") or "song")
    if verbose:
        print(f"  [声部] {sum(len(t.notes) for t in tracks)}音・{len(tracks)}本 — {got.reason[:110]}")
        for w in warns[:4]:
            print(f"      ★{w}")
    return tracks, warns, got
