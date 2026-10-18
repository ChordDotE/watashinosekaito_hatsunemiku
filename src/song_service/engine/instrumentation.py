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

"""**編成を決める**工程（どの楽器でこの曲を演奏するか）。

これまで音色は「編曲プロンプトに書いた24種」から選ばせていた。24種に絞っていたのは
**一覧を編曲プロンプトに毎回書く都合**であって、音源が持っていないからではない
（SoundFont には GM の128音色が入っている）。

工程を分けると3つ良いことがある（2026-08-22・本人の提案）:

  1. **128種すべてが対象**になる（琴・三味線・尺八・オーボエ・パイプオルガンも選べる）
  2. **編曲プロンプトが短くなる**（選ばれた数本だけ渡せばよい）
  3. **なぜその楽器かが記録に残る**——従来は音色選びの理由がどこにも残っていなかった

編曲側には**番号で指定させる**（勝手な音色名を書けないようにする）。
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from pydantic_ai import Agent

from sf2_presets import describe_source, melodic_menu, variation_menu
from song_usage import record as _rec_usage           # 費用の集計
from voice_style import menu as _vs_menu                   # 歌い方の一覧（一次情報つき）
from voice_style import style_problem as _style_problem    # 名前の検査（型で差し戻す・監査 2026-09-27）
from voice_style import voice_problem as _voice_problem
from voice_style import STYLES as _VS_STYLES               # 歌い方の名前（一覧はここから引く）

VOCAL_STYLE_MENU = _vs_menu(for_lead=True)
from sf2_render import soundfont
from song_llm import pick_model
from song_llm import run_sync as _llm_run
from song_model import PatchStatus, StepStatus
from song_checks import JsonBool, JsonInt  # noqa: E402（LLM が書く数と真偽の欄の型・2026-10-02 判断5）
from song_checks import edge_ws_problem, strip_ascii_ws  # noqa: E402（半角の名前の前後の空白・§0-6・2026-10-03）
from timbre_map import gm_menu, gm_name
from song_prompts import prompt                          # プロンプトの正は prompts/*.md

#: ★**楽器の本数の上限＝鳴らし分けられる数**（2026-10-03・`docs/plan/song_viewpoints_review.md` §2 c7）。書き出し（`midi_out.score_to_midi`）は
#:   音色ごとに旋律のチャンネルを1本ずつ割り当てる。旋律のチャンネルはドラム（ch10）を除く **15 本**（`midi_out._CHANNELS`）で、
#:   主旋律の楽器（`lead`）があるとそれが専用に1本使うので伴奏は **14 本**。これを超えるとチャンネルを使い回し、
#:   後から program change を出した音色で両方が鳴る（音色が入れ替わる＝読み手が壊れる）。以前は 10 本で、機械の限界より狭かった。
#:   本数の目安（弾き語り2本・バンド4〜5本…）は説明に残す。主旋律の楽器があるときの 14 本は編成の検査 `form_problems` が見る
MAX_INSTRUMENTS = 15
MAX_INSTRUMENTS_WITH_LEAD = 14


class Instrument(BaseModel):
    gm: JsonInt = Field(ge=1, le=128, description=(
        "音色番号（**一覧に載っている番号**から選ぶ）。追加音色は `bank` も指定する"))
    # ★必ず書く（2026-10-04・既定値をやめる）。以前は省くと 0 で、追加音色の番号だけ書くと別の音色になった
    bank: JsonInt = Field(ge=0, le=127, description=(
        "音色のバンク。**必ず書く**。一覧のふつうの音色は 0、**「この音源だけの追加音色」は `bank-番号` の bank**"
        "（例 `1-61:Solo French Horn` なら bank=1, gm=61）"))
    part: str = Field(description=(
        "この楽器の**担当**。伴奏の役割が分かる短い言葉で"
        "（例: 低音を支える / コードを刻む / 旋律に絡む対旋律 / 空間を埋める持続音）"))
    reason: str = Field(description="なぜこの曲にこの楽器か（音色の性格と曲想の結びつき）")


class Instrumentation(StepStatus):
    reason: str = Field(description="編成全体の狙い（何と何を組み合わせて、どんな響きを作るか）")
    instruments: list[Instrument] = Field(max_length=MAX_INSTRUMENTS, description=(
        # 上限は 2026-09-01 に 6 → 10 へ広げた（**仕組みの天井だけ**・推奨は据え置き）。
        # 打楽器は ch10 専用なので旋律楽器は MIDI の 15 本まで置ける。
        # ★アカペラは0本（2026-10-03・仕様書 §15-4。形ごとの決まりは編成の検査 `form_problems` が見る）
        # ★**本数は 0〜10 を型で・歌もの／インストの 1本以上は編成の検査で**（2026-10-03・再レビュー 記C・仕様書 §15）。
        #   以前は定義が無く、11本目以降は伴奏が指せず（`PartPlan.instrument` は 1〜10）、歌もので0本だと伴奏の組み立てで落ちた。
        #   型は形を知らないので、下限（形で 0 か 1）は `form_problems` が見る
        f"**1〜{MAX_INSTRUMENTS}本**（アカペラは0本・主旋律の楽器 `lead` を書くときは {MAX_INSTRUMENTS_WITH_LEAD}本まで）。"
        "多ければ良いものではありません——"
        "弾き語り調なら2本、バンドサウンドなら4〜5本、厚いポップスで6本程度。"
        "**それ以上は、管や弦のセクションを組むなど本数が要る編成のときだけ**。"
        "**アカペラの曲（依頼文の【曲の形】）は0本**（空の並び）"))
    # ★**主旋律を弾く楽器**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §3・F2・F10②・F19・F20。定義は仕様書 §15）。
    #   どの形でも同じ欄・同じ意味（書かない＝楽器では弾かない）。形ごとの決まり（インストは必ず書く・アカペラは書けない・
    #   インストの `lead_sections` は空だけ・番号と区間名の照らし方）は編成の検査が見る。鳴らすのは組み立て（`midi_out` の1か所）
    # ★2026-10-04: 既定値をやめて必ず書く欄にした（本人「書き忘れたのか標準にしたのかが曖昧になる」）。弾かないなら null を、
    #   全部の区間なら空の並びを、選んだことが分かる形で書く
    lead: JsonInt | None = Field(ge=1, le=MAX_INSTRUMENTS, description=(
        "**主旋律（旋律の工程の音）を弾く楽器**の番号（上の `instruments` の並びの番号・1始まり）。**必ず書く**——"
        "楽器では弾かないなら null と書く（たいていの歌ものは null）。歌ものでは歌と同じ旋律を同じ高さ・同じ長さで重ねて弾く。"
        "**インストの曲は番号を必ず書く**・**アカペラの曲とカバーは null**（依頼文の【曲の形】）"))
    lead_sections: list[str] = Field(description=(
        "`lead` の楽器が主旋律を弾く区間（設計の区間名を1字も違わずに・主旋律のある区間だけ）。**必ず書く**——"
        "主旋律のある区間の全部で弾くなら空の並び `[]`（`lead` が null のときも空の並び）。"
        "**インストの曲は空の並びだけ**（全部の区間で弾く）・アカペラの曲とカバーは空の並び"))
    # ★2026-10-02（判断6・本人「おすすめで」）: 書けるのは【ドラムキット】の一覧の名前を1字も違わずに1つだけ（`timbre_map.kit_names`。
    #   1つのキットに名前は1つ）。以前は大文字小文字を無視し、番号だけ・音源のキット名だけ・先頭5字の前方一致でも読んでいた
    #   （この説明が「electronic・tr808」と2つを並べていたので、記録に `electronic・tr808` と書いた曲が3曲あり、前方一致で electronic になっていた）
    # ★2026-10-04: 既定（standard）をやめて必ず書く欄にした。書き忘れは型が差し戻す（以前は黙って standard で鳴った）
    drum_kit: str = Field(description=(
        "ドラムの種類。**必ず書く**。**【ドラムキット】の一覧の名前を1字も違わずに1つだけ**（使わないなら none）。"
        "迷ったら万能の standard。"
        "よく使うもの: standard（万能）/ room（残響多め）/ power（太い）/ electronic（打ち込み系）/ tr808（打ち込み系・808）/ "
        "jazz（軽い）/ brush（ブラシ・静か）/ orchestra（オーケストラ）/ none（ドラムを使わない）。"
        "一覧で `27:Dance` のように番号つきで載っているキットは番号つきのまま書く（かっこの中は音源のキット名で、書かない）。"
        "大文字小文字・番号だけ・音源のキット名だけ（`49`・`Orchestral`・`Standard`）・2つ並べた書き方は読めない。"
        # ★目録の 打-6（2026-09-27）。以前は読めない名前（taiko など）を黙って標準のキットで鳴らしていた
        "**それ以外の名前は読めず、検査で所見として返します**（直さずに残るとドラムは鳴りません。"
        "標準のキットで代わりに鳴らすことはしません）。和太鼓などはドラムキットではなく、音程パートの楽器で選ぶ。"
        "**キットによって入っている打楽器が違います**（伴奏はそのキットにある打楽器だけで書きます。"
        "例えばオーケストラ用のキットにはハイハットやシンバルの刻みが無いことがあります）"))
    # ★**ハモリを入れるかをここで決める**（2026-08-23 ユーザー指摘）。
    # 以前はハモリ工程を**毎回呼んでいた**ので、どの曲にも必ずハモリが付いた。
    # 『空白の夜明け』（一人きりの独白のバラード）にも 60音のハモリが乗り、
    # 「過剰。この曲は無い方が良かった」という指摘を受けた。
    # **重ねるのが当たり前ではない**——独白の曲は1人で歌うから独白に聞こえる。
    # ★2026-10-04: 既定（none）をやめて必ず書く欄にした。省略と「使わない」を選んだことを分ける（以前は省略でハモリ工程が黙って走らなかった）
    harmony: str = Field(description=(
        "**ハモリ（重ねる声）を使うか**。**必ず書く**（次の3つのどれかを小文字で）。\n"
        "  none    … **使わない**。独白・弾き語り・静かなバラード・"
        "語りかける曲は、1人で歌うからこそ届きます\n"
        "  accent  … **ここぞという所だけ**重ねる（ラスサビだけ・サビの後半だけ 等）\n"
        "  full    … サビでしっかり重ねる（合唱・アイドル・応援歌・厚いポップス）\n"
        "**迷ったら none。** ハモリは「厚くする」ためではなく、"
        "**その瞬間だけ景色を変える**ために使う道具です"))
    harmony_reason: str = Field(description=(
        "`harmony` をそう決めた理由。**必ず書く**——none を選んだときも書く"
        "（「一人称の独白なので声を重ねない」等）"))
    # ---- 歌い方（2026-08-23 追加）----
    # ミクには声2種とスタイル14種があり、vpr に書けば V6 がそのとおりに歌う。
    # **仕組みは前からあったのに、パイプラインが一度も呼んでいなかった**
    # （全曲が `No Effect`＝加工なしで書き出されていた）。
    # ★2026-10-04: 声・歌い方・声部の歌い方は既定（original / MIKU Default / MIKU Whisper）をやめて必ず書く欄にした。
    #   以前は空は差し戻すのに省略は黙って既定で通り、省略と空で扱いが違った
    voice: str = Field(description=(
        "**どちらの声で歌うか**。**必ず書く**。original（芯のある標準の声）/ soft（やわらかく息の多い声）。"
        "激しい曲・言葉を立てたい曲は original、静かな曲・語りかける曲は soft。"
        "歌の無いインストでも書く（使われない。迷ったら original）"))
    vocal_style: str = Field(description=(
        "**歌い方**（V6 のスタイルプリセット）。**必ず書く**。加工しない素の歌い方は `MIKU Default`。"
        "曲想に合うものを1つ:\n" + VOCAL_STYLE_MENU))
    # ★説明を実体に合わせた（2026-09-30・出-13）。以前は「none 以外のときだけ使う」と書いていたが、楽譜の声の種類として
    #   いつも持ち回り（`voice_style.set_choice` → `score.voice_choice`）、あとで編集してハモリ・合いの手を足したときにも
    #   声部の既定の歌い方になる（`voice_style` の `part_styles` の空欄）
    harmony_style: str = Field(description=(
        "**ハモリ・声部の歌い方**（声部ごとに歌い方の指定が無いときに使う）。**必ず書く**。`harmony` が none の曲でも書く——"
        "**あとで編集してハモリや合いの手を足したとき**にこの歌い方で歌う。"
        "迷ったら `MIKU Default`。息を含んだ柔らかい声部は `MIKU Breath` か `MIKU Breath+`。"
        "`MIKU Whisper` は高さによらず声にならず息の音だけになる（AIR +100＝完全に無声。実測）ので、**音程のある声部には使わない**"
        "（息づかいの質感・効果音的な息の差し込み専用。頼みで「ウィスパー」と明示されたときだけ、その息の質感として使ってよい）。"
        "**使えるのは次の名前だけ**: " + " / ".join(_VS_STYLES)))
    voice_reason: str = Field(description="その声と歌い方を選んだ理由（曲想との結びつき）。**必ず書く**")

    # ★**知らない声・歌い方の名前は型で差し戻す**（2026-09-27・監査。`harmony.HarmonyPart.voice_style` と同じ形）。
    #   以前は検証が無く、知らない名前は `voice_style.apply_choice` が**当てずに据え置き**、所見は画面に出るだけで
    #   この工程には届かなかった（V6 は既定の声・歌い方で歌う）。使える名前は `voice_style` の実体
    #   （`VOICE_CHOICES` / `STYLES`）から引く——手書きの一覧は作らない。2026-10-02（判断6）から別名の表は無く、正式な名前を
    #   1字も違わずに（前後の半角の空白とタブだけ取り除く。全角の空白・改行などは所見・§0-6・2026-10-03）。**空も所見**——編成の声・歌い方には「空＝既定」の定義が無い。
    #   2026-10-04 から省略も型が差し戻す（欄が必須。以前は省略だけ型の既定で通った）。
    @field_validator("voice")
    @classmethod
    def _known_voice(cls, v: str) -> str:
        why = _voice_problem(v)
        v = strip_ascii_ws(v)
        if why:
            raise ValueError(why)
        return v

    @field_validator("vocal_style", "harmony_style")
    @classmethod
    def _known_style(cls, v: str) -> str:
        # ★空も差し戻す（2026-10-02・洗い出し）。以前は `required` なしで呼び、空 `""` が型を通って何も当てずに V6 の既定で歌った。
        #   声部の `HarmonyPart.voice_style` は「空＝ハモリ全体の歌い方」と定義しているので `required` なし（触らない）
        why = _style_problem(v, required=True)
        if why:
            raise ValueError(why)
        return strip_ascii_ws(v)


# 楽器の使いどころと編成の実例。**サンプルであって規則ではない**——
# 数値を1つ指示するたびに別の何かが壊れた経験（配分を守らせたらフレーズの型が固定、
# 音域の幅を締めたら最高音を削られた）から、**例を示して外れる自由を明記する**書き方に統一した。
# 例に無い組み合わせを作れないと、独自の表現も新しい響きも生まれない（2026-08-22・本人指示）。
_HINTS = """

【楽器の使いどころ】**こういう使い方もある、という例です。この通りにする必要はありません。**
  琴・三味線 … 分散和音と「間」で聴かせる。西洋和音のように刻み続けると持ち味が消える
  尺八・笛   … 歌の**合間**に入る対旋律。歌と同時に同じ高さで鳴らすと喧嘩する
  太鼓       … 拍の要を打つ。**洋楽のドラムセットの代わり**になる
  ピアノ     … 低音も和音も旋律も1台でこなす。1本で曲が成立する数少ない楽器
  アコギ     … 分散和音（アルペジオ）でも刻みでも。弾き語り調の要
  歪みギター … 和音を鳴らし続けると濁るので、5度中心か短く刻む
  ストリングス … 白玉で下を支える。動かしすぎると歌が埋もれる
  ブラス     … 短く鋭く差し込む。鳴らしっぱなしにしない
  ベース類   … **ほぼ必須**。低音が無いと曲が浮く
  ベル・鉄琴 … 数音だけ置いてきらめきを作る。全編に敷くと単調になる
  パッド     … 音の隙間を埋める空気。前に出す音ではない

【編成の例】**定番を並べただけです。ここに無い組み合わせを作って構いません。**
  和風・神楽     尺八 ＋ 琴 ＋ 太鼓（＋ 神楽鈴＝ティンクルベル）
  弾き語り       アコギ 1本（＋ ベース）
  ボカロロック   歪みギター ＋ ベース ＋ ドラム ＋ シンセリード
  バラード       ピアノ ＋ ストリングス ＋ ベース
  童謡           オルゴール ＋ グロッケン ＋ リコーダー
  ジャズ寄り     エレピ ＋ ウッドベース ＋ ジャズギター
  シティポップ   エレピ ＋ クリーンギター ＋ ピックベース ＋ ブラス

**※ 例から離れてよい場面**:
  ・マスターから「この楽器で」「こういう雰囲気で」と指示があったとき
  ・**あなた自身が試してみたいと思ったとき**——聴いたことのない組み合わせを探すのは良いことです
  例に無い音（シタール・スチールドラム・オーボエ・バグパイプ…）を選んで構いません。
  **理由が音楽的であれば、それが正解です。**
"""


_instrumenter: Agent[None, Instrumentation] = Agent(
    pick_model("INSTRUMENT", "flash"), output_type=Instrumentation,
    system_prompt=(
        prompt("song_instrumentation") + _HINTS
    ),
)


def available_menu() -> tuple[str, str, str]:
    """**実際に鳴らす音源から**音色一覧を作る（手書き表はずれるため・2026-08-22）。

    音源が読めなければ GM の標準表に落ちる。戻り値は (音程音色, 追加音色, ドラムキット)。
    """
    # ★ドラムキットは書ける名前（`timbre_map.kit_names`）の一覧を見せる（2026-10-02・判断6）。以前は音源の番号つきの名前
    #   （`49:Orchestral`）を見せ、型の説明は GM の名前（`orchestra`）を見せていた——同じキットに書き方が2通り。
    #   音源が読めないときは GM の名前だけ（以前の手書きの一覧は tr808 が抜けていた）
    from timbre_map import kit_menu                          # noqa: E402
    sf = soundfont() or ""
    mel = melodic_menu(sf)
    if not mel:                                            # .sf3 等で読めない → 手書きの GM 表へ
        return gm_menu(), "", kit_menu()
    return mel, variation_menu(sf), kit_menu()


def plan_instruments(title: str, genre: str, bpm: int, brief_text: str) -> Instrumentation:
    mel, extra, kits = available_menu()
    prompt = (f"曲名『{title}』 種類={genre} BPM{bpm}\n\n{brief_text}\n\n"
              f"【使える音色】{describe_source(soundfont() or '')}\n{mel}\n"
              + (f"\n{extra}\n" if extra else "")
              + f"\n【ドラムキット】{kits} / none（ドラムを使わない）\n\n"
              "この曲をどんな編成で演奏しますか。")
    r = _llm_run(_instrumenter, prompt, label='編成')
    _rec_usage(_instrumenter, r, "編成")                    # 費用集計から漏れていた（2026-08-23）
    out = r.output
    if type(r).__name__ == "_Replayed":                    # 保存から戻した編成は直させない（`InstrumentationStep`）
        try:
            out.__dict__["_replayed"] = True
        except Exception:                                   # noqa: BLE001
            pass
    return out


# ---- 編成の検査と直し（2026-09-27・`song_fix`・目録の 打-6）--------------------------------
#
# ★読めないドラムキット名（`taiko` など）は、書き出しで**音源の既定のキット**（standard）になっていた。黙って。
#   いまは所見を編成の工程へ返し、`drum_kit` だけを差分で直させる。
#   ★2026-09-29 から、残ったら**ドラムは鳴らさない**（代わりのキットで鳴らさない・`midi_out`）。
#   ★2026-09-30 から、残ったら編成の工程で曲づくりを止める（`song_fix.StepUnfixed`）。

#: ハモリの方針の選択肢（`Instrumentation.harmony` の説明と同じ）
HARMONY_CHOICES = ("none", "accent", "full")


class InstrumentationPatch(PatchStatus):
    """編成の直し（差分）。`drum_kit`・`harmony`・音色番号と、主旋律の楽器（`lead`・`lead_sections`）・外す楽器（アカペラ）。"""
    reason: str = Field(description="何をどう直したか（一言）")
    drum_kit: str = Field(default="", description=(
        "新しいドラムの種類（【ドラムキット】の一覧の名前を1字も違わずに1つ、か none）。直さないなら空"))
    harmony: str = Field(default="", description="新しいハモリの方針（none / accent / full のどれか）。直さないなら空")
    instruments: list["InstrumentFix"] = Field(default_factory=list, description=(
        "音色番号を直す楽器だけ（番号は【いまの編成】の楽器の番号）。直さないなら空"))
    # ★主旋律の楽器と、形ごとの決まりを直す欄（2026-10-03・`docs/plan/song_acappella_instrumental.md` §3・仕様書 §15-4）。
    #   以前の差分は drum_kit・harmony・音色番号だけで、形の所見（アカペラの楽器・インストの lead）を直せなかった
    lead: JsonInt | None = Field(default=None, ge=1, le=MAX_INSTRUMENTS, description=(
        "新しい主旋律の楽器の番号（【いまの編成】の楽器の番号）。直さないなら null"))
    drop_lead: JsonBool = Field(default=False, description="主旋律を楽器で弾かないことにする（lead を null に）なら true")
    lead_sections: list[str] = Field(default_factory=list, description=(
        "主旋律を弾く区間を**書き直した全体**（設計の区間名をそのまま・主旋律のある区間だけ）。直さないなら空"))
    clear_lead_sections: JsonBool = Field(default=False, description=(
        "lead_sections を空（＝主旋律のある区間の全部）にするなら true"))
    remove_instruments: list[JsonInt] = Field(default_factory=list, description=(
        "外す楽器の番号（【いまの編成】の楽器の番号）。アカペラの曲で楽器を0本にするときは全部の番号を書く。外さないなら空"))


class InstrumentFix(BaseModel):
    # ※1始まり（仕様書 §15「差分の no は1以上」・2026-10-01。以前は範囲が無く、0 以下は `apply` が黙って捨てていた）
    no: JsonInt = Field(ge=1, description="直す楽器の番号（【いまの編成】の番号・1始まり）")
    gm: JsonInt = Field(ge=1, le=128, description="新しい音色番号（一覧にある番号）")
    # ★**空＝いまのまま**（2026-10-04・差分の決まりにそろえた）。以前は既定 0 で、音色番号（gm）だけを直すと bank が 0 に上書きされた
    #   （追加音色 bank 1 の楽器の番号を直すと、別の音色に化けた）
    bank: JsonInt | None = Field(default=None, ge=0, le=127, description=(
        "新しいバンク（一覧のふつうの音色は 0・追加音色は `bank-番号` の bank）。変えないなら空（いまのバンクのまま）"))


InstrumentationPatch.model_rebuild()


def fix_no_problem(fixes: list[InstrumentFix], n_instruments: int) -> str:
    """差分の `instruments[].no` が【いまの編成】の楽器の数を超えている所見（無ければ空）。型（`patch_type`）と `apply` で同じ文を使う。"""
    nos = [f.no for f in fixes or []]
    bad = [x for x in nos if not 1 <= x <= n_instruments]
    out: list[str] = []
    if bad:
        out.append(f"instruments の楽器の番号 {'・'.join(map(str, bad))} は【いまの編成】に無い——"
                   f"楽器は{n_instruments}つなので、番号は 1〜{n_instruments}（【いまの編成】の番号をそのまま書く）")
    # ★**同じ番号は1回だけ**（2026-10-03・再レビュー）。以前は2つ目が黙って勝った（ほかの工程の差分と同じ決まり・`song_fix.patch_no_problem`）
    if dup := sorted({x for x in nos if nos.count(x) > 1 and 1 <= x <= n_instruments}):
        out.append(f"instruments に楽器の番号 {'・'.join(map(str, dup))} を2回以上書いた——1つの楽器に直しは1つ")
    return "／".join(out)


# ---- 曲の形ごとの決まりと主旋律の楽器（2026-10-03・`docs/plan/song_acappella_instrumental.md` §3・仕様書 §15・§15-4）----
#
# ★形はコーディネーターが決め、編成の LLM には依頼文の【曲の形】で知らせる。編成が楽器を書いたアカペラ・`lead` の無いインストは、
#   **機械が楽器を消したり主旋律の楽器を選んだりせず**、所見で編成の LLM に直させる（直らなければ編成の工程で止まる）。
#   同じ部品を、形を変えたときに残す編成へ当てる検査（`song_form_checks.form_findings`）も呼ぶ（検査を2本書かない）。

def form_problems(plan: Instrumentation, form: str = "song", sections=None, *, cover: bool = False) -> list[str]:
    """**編成の、曲の形ごとの決まりと主旋律の楽器の欄**の所見（答えが1つに決まるものだけ）。

    - アカペラ: 楽器は0本・ドラムは none・`lead`／`lead_sections` は書けない
    - インスト: `lead` は必ず書く・ハモリは none・`lead_sections` は空だけ
    - 歌もの・インスト: 楽器は1本以上（15本までは型・2026-10-03 記C・c7）
    - どの形も: 主旋律の楽器 `lead` があるときは楽器は14本まで（`MAX_INSTRUMENTS_WITH_LEAD`・2026-10-03 c7。主旋律の楽器が
      専用のチャンネルを1本使うので、伴奏に残る旋律のチャンネルは14本）
    - カバー（`cover`・形は歌もの）: `lead`／`lead_sections` は書けない（2026-10-03 記A。組み立てがカバーの主旋律の楽器を作らない）
    - どの形も: `lead` は【いまの編成】の楽器の番号の範囲・`lead_sections` は `lead` があるときだけ・
      主旋律のある区間（`has_vocal` が true）だけ・同じ区間は1回だけ
    区間名が設計に無いことは設計との照合（`song_checks.against_design`・`InstrumentationStep.design_items`）が見るので、ここでは見ない。
    `sections` は設計の区間（`kind`・`has_vocal`）。渡さなければ区間の決まりは見ない。
    """
    from song_checks import resolve_section                 # noqa: E402（照らし方は1か所）
    # 欄はどれも必須（2026-10-04・既定値をやめた）。欄の無い古い記録は変換（`convert_record.py`）が当時の既定を書き込んでから読む
    n = len(plan.instruments or [])
    lead = plan.lead
    secs = list(plan.lead_sections)
    kit = strip_ascii_ws(plan.drum_kit)
    out: list[str] = []
    if form == "acappella":
        if n:
            out.append(f"アカペラの曲なのに楽器が{n}本ある——アカペラは声だけで作る（伴奏の楽器は0本）。"
                       f"楽器を全部外す（差分の remove_instruments に 1〜{n} の番号を全部書く）")
        if kit != "none":
            out.append(f"アカペラの曲なのにドラムの種類が『{kit}』——アカペラはドラムを使わない。drum_kit を none にする")
        if lead is not None:
            out.append(f"アカペラの曲なのに主旋律の楽器（lead＝{lead}）を書いた——アカペラは楽器で弾かない（差分の drop_lead）")
        if secs:
            out.append("アカペラの曲なのに lead_sections を書いた——アカペラは楽器で弾かない（差分の clear_lead_sections）")
        return out
    # ★**歌もの・インストは楽器が1本以上**（2026-10-03・再レビュー 記C・仕様書 §15）。以前は定義が無く、歌もので0本だと
    #   伴奏の組み立て（`to_spec`）で落ちた。差分には楽器を足す欄が無いので、直らなければ編成の工程で止まる（作り直しで書き直す）
    if not n:
        out.append(f"{'インスト' if form == 'instrumental' else '歌もの'}の曲なのに楽器が0本——"
                   f"伴奏の楽器は1〜{MAX_INSTRUMENTS}本（楽器が0本で作れるのはアカペラだけ）")
    # ★**主旋律の楽器があるときは 14 本まで**（2026-10-03・§2 c7）。主旋律の楽器が専用のチャンネルを1本使い、伴奏は残りの 14 本で
    #   音色ごとに鳴らし分ける（`midi_out.score_to_midi`）。15 本目の音色はチャンネルを使い回して音色が入れ替わる
    if lead is not None and n > MAX_INSTRUMENTS_WITH_LEAD:
        out.append(f"主旋律の楽器（lead＝{lead}）があるのに楽器が{n}本——主旋律の楽器が専用のチャンネルを1本使うので、"
                   f"楽器は{MAX_INSTRUMENTS_WITH_LEAD}本まで（鳴らし分けられる数の限界）。"
                   "楽器を減らす（差分の remove_instruments）か、主旋律を楽器で弾かない（差分の drop_lead）")
    # ★**カバーは主旋律の楽器を書けない**（2026-10-03・再レビュー 記A・仕様書 §15-4）。カバーの組み立ては主旋律の楽器を作らないので、
    #   以前は書けても鳴らなかった（3本のレビューが別々に指摘）。重ねたくなったら組み立てを足してから定義を変える
    if cover:
        if lead is not None:
            out.append(f"カバーなのに主旋律の楽器（lead＝{lead}）を書いた——カバーは楽器で主旋律を重ねない（差分の drop_lead）")
        if secs:
            out.append("カバーなのに lead_sections を書いた——カバーは楽器で主旋律を重ねない（差分の clear_lead_sections）")
        return out
    if form == "instrumental":
        if lead is None:
            out.append("インストの曲なのに主旋律の楽器（lead）が無い——主旋律（旋律の工程の音）を弾く楽器を、"
                       f"【いまの編成】の番号で lead に書く（楽器は{n}本）")
        h = strip_ascii_ws(plan.harmony)
        if h in HARMONY_CHOICES and h != "none":          # 一覧に無い綴りは方針の所見（`InstrumentationStep.check`）が出す
            out.append(f"インストの曲なのにハモリの方針が『{h}』——インストは歌が無いのでハモリは none")
        if secs:
            out.append("インストの曲なのに lead_sections を書いた——インストは主旋律のある区間の全部で弾く（空だけ。"
                       "差分の clear_lead_sections）")
    if lead is not None and not 1 <= lead <= n:
        out.append(f"主旋律の楽器 lead＝{lead} は【いまの編成】に無い——楽器は{n}本なので、番号は 1〜{n}")
    if form == "song" and secs:
        if lead is None:
            out.append("lead_sections を書いたのに主旋律の楽器（lead）が無い——重ねないなら lead_sections を空に"
                       "（差分の clear_lead_sections）、重ねるなら lead に楽器の番号を書く")
        if dup := [s for s in dict.fromkeys(secs) if secs.count(s) > 1]:
            out.append("lead_sections に同じ区間を2回以上書いた: " + "・".join(f"『{s}』" for s in dup) + "——1つの区間は1回だけ")
        if sections:
            names = [s.kind for s in sections]
            quiet = {s.kind for s in sections if not getattr(s, "has_vocal", True)}
            sung = [k for k in names if k not in quiet]
            for s in dict.fromkeys(secs):
                if resolve_section(s, names) in quiet:
                    out.append(f"lead_sections の『{s}』は主旋律の無い区間——主旋律を弾けるのは主旋律のある区間だけ"
                               f"（{' / '.join(sung)}）")
    return out


def lead_patch_problem(plan: Instrumentation, patch: "InstrumentationPatch") -> str:
    """差分の主旋律の楽器と外す楽器の、当てられない書き方の所見（無ければ空）。差分の検証（`patch_problem`）と `apply` で同じ文を使う。"""
    n = len(plan.instruments or [])
    rm = list(getattr(patch, "remove_instruments", None) or [])
    out: list[str] = []
    if bad := sorted({x for x in rm if not 1 <= x <= n}):
        out.append(f"remove_instruments の番号 {'・'.join(map(str, bad))} は【いまの編成】に無い（楽器は{n}本）")
    if dup := sorted({x for x in rm if rm.count(x) > 1}):
        out.append(f"remove_instruments に番号 {'・'.join(map(str, dup))} を2回以上書いた——1回だけ")
    if patch.lead is not None and patch.drop_lead:
        out.append("lead と drop_lead を一緒に書いた——弾く楽器を変えるなら lead だけ、弾かないなら drop_lead だけ")
    if patch.lead_sections and patch.clear_lead_sections:
        out.append("lead_sections と clear_lead_sections を一緒に書いた——書き直すなら lead_sections だけ、空にするなら clear_lead_sections だけ")
    if patch.lead is not None and not 1 <= patch.lead <= n:
        out.append(f"lead の番号 {patch.lead} は【いまの編成】に無い（楽器は{n}本）")
    lead = None if patch.drop_lead else (patch.lead if patch.lead is not None else plan.lead)
    if lead is not None and lead in rm:
        out.append(f"主旋律の楽器（楽器{lead}）を外そうとしている——外すなら drop_lead も書く（弾く楽器を変えるなら lead に別の番号）")
    return "／".join(out)


_instrument_fixer: Agent[None, InstrumentationPatch] = Agent(
    pick_model("INSTRUMENT", "flash"), output_type=InstrumentationPatch,
    system_prompt=prompt("song_instrumentation"),
)


# ★**楽器の数を超える `instruments[].no` は差し戻す**（2026-10-02）。以前は `InstrumentationStep.apply` が範囲外の `no` を
#   **黙って捨てていた**——LLM は直したつもりで、所見（音源に無い音色）は残ったまま次の回へ進んだ。楽器の数は曲ごとに違うので
#   `InstrumentFix` の型には書けない——直しを頼む間だけ工程の `patch_problem` を置き、差分の検証（`song_fix.guard_patch`）が差し戻す
#   （同じ日に作った編成だけの検証 `_fix_no_in_range` と楽器の数の `_N_INSTRUMENTS` は、工程の形の1つへまとめた・組「洗い出し-型と残り」）
from song_fix import StepFixer, guard_patch                 # noqa: E402

guard_patch(_instrument_fixer)


class InstrumentationStep(StepFixer):
    """編成: ドラムの種類が音源で読めるか（形式の検査。`timbre_map.kit_finding`）・ハモリの方針・音色が音源にあるか・
    **曲の形ごとの決まりと主旋律の楽器の欄**（`form_problems`・`lead_sections` は設計と照らす・2026-10-03）→ 所見の欄だけ差分で直させる。
    形は `InstrumentationStep(form, design_sections)` で渡す（省けば歌もの）。

    呼ぶ場所は `plan_instruments` の直後（`song_band.run_instrumentation`・カバーの `song_arrange_cover`）で、
    `InstrumentationStep().run(plan, song)` → (編成, 残った所見＝いつも空)。保存から戻した編成は直させない（親の `run`）。
    ★検査は書き出し（`midi_out`）と同じ `kit_finding` を使う——2本にすると、検査は通ったのに鳴らないができる。
    ★直らなければ編成の工程で曲づくりを止める（`song_fix.StepUnfixed`・2026-09-30）。以前は残ったまま進み、
      書き出しがドラムを鳴らさなかった（2026-09-29 まで音源の既定のキット＝standard で代わりに鳴らした）。
      書き出しの「読めないキットは鳴らさない」は残してある——保存した楽譜を別の音源で鳴らすと、キットが読めないことがある。
    ★検査は音源のキットの一覧（`available_menu` で見せているもの）で見る——音源に無い番号も読めないとする（`drum_program`）。
    """
    key = "instrumentation"
    label = "編成"
    agent = _instrument_fixer
    return_hint = ("**直すのは所見の出た欄（drum_kit・harmony・音源に無い楽器の音色番号・主旋律の楽器 lead／lead_sections・"
                   "アカペラで外す楽器）だけ**です。担当・歌い方は変えません。")

    def __init__(self, form: str = "song", design_sections=None, *, cover: bool = False):
        """`form` は曲の形（`song_schema.Form`・2026-10-03）、`design_sections` は設計の区間（`lead_sections` を設計と照らす）。
        どちらも省けば、歌もので区間を照らさない（検査だけの試験）。`cover` はカバーの編成（主旋律の楽器を書けない・記A・2026-10-03）。"""
        self.form = form or "song"
        self.design = [s for s in (design_sections or []) if getattr(s, "kind", None)] or None
        self.cover = bool(cover)

    def design_items(self, plan: Instrumentation):
        return [(f"lead_sections の『{s}』", s, None) for s in (getattr(plan, "lead_sections", None) or [])]

    def rename_sections(self, plan: Instrumentation, mapping: dict[str, str]) -> Instrumentation:
        return plan.model_copy(update={"lead_sections": [mapping.get(s, s) for s in plan.lead_sections]})

    def form_check(self, plan: Instrumentation) -> list[str]:
        """**形と主旋律の楽器の欄だけ**の所見（`lead_sections` を設計と照らす＋`form_problems`）。形を変えたときに残す編成へ当てる
        （`song_form_checks.form_findings`）。工程の検査（`findings`）も同じ部品を通る。"""
        from song_checks import against_design              # noqa: E402
        out = against_design(self.design_items(plan), self.design) if self.design else []
        return out + form_problems(plan, self.form, self.design, cover=self.cover)

    def check(self, plan: Instrumentation) -> list[str]:
        from timbre_map import kit_finding, kit_names         # noqa: E402
        kit = strip_ascii_ws(plan.drum_kit)
        # ★**空のドラムの種類は所見**（2026-10-02・洗い出し）。以前は所見なしにドラムありで進み（`song_band` が standard として読む）、
        #   書き出しは音源の既定のキットで鳴らしていた。欄は必須（2026-10-04 から省略も型が差し戻す）、書くなら一覧の名前か none
        f = kit_finding(kit) if kit else (f"ドラムの種類（drum_kit）が空——【ドラムキット】の一覧の名前（{' / '.join(kit_names())}）"
                                          "か none を1つ書く")
        out = [f] if f else []
        # ★**ハモリの方針は3つのどれか**（2026-09-29）。以前は一覧に無い値でもそのまま渡り、none 以外は「重ねる」扱いになった
        # ★**小文字の3つだけ**（2026-10-02・洗い出し）。以前は大文字小文字を無視し、空も none と読んでいた（前後の空白だけ取り除く）
        h = strip_ascii_ws(plan.harmony)
        if why := edge_ws_problem(plan.harmony):           # 前後に全角の空白・改行などを付けない（2026-10-03・§0-6）
            out.append(f"ハモリの方針{why}")
        elif h not in HARMONY_CHOICES:
            out.append(f"ハモリの方針『{plan.harmony}』は一覧に無い——none / accent / full のどれかを小文字で書く"
                       + (f"（『{h.lower()}』）" if h.lower() in HARMONY_CHOICES else ""))
        # ★**音色が音源にあるか**（2026-09-29）。以前は見ておらず、音源に無い bank・番号は FluidSynth が別の音色で鳴らしていた
        #   （書いた音色で鳴らない）。音源の一覧が読めないとき（.sf3 など）は見ない
        from sf2_presets import read_presets
        banks = read_presets(soundfont() or "")
        if banks:
            for no, ins in enumerate(plan.instruments, 1):
                if (banks.get(ins.bank) or {}).get(ins.gm - 1) is None:
                    out.append(f"楽器{no}（{ins.part}）の音色『{ins.bank}-{ins.gm}』は音源に無い——一覧にある番号から選ぶ"
                               "（追加音色は一覧の `bank-番号`、それ以外は bank 0）")
        # ★**曲の形ごとの決まりと主旋律の楽器の欄**（2026-10-03・`form_problems`）。区間名の照合は親の決まり（`design_items`）
        return out + form_problems(plan, self.form, self.design, cover=self.cover)

    def show(self, plan: Instrumentation) -> str:
        _mel, _extra, kits = available_menu()
        rows = "\n".join(f"  {no}. {ins.bank}-{ins.gm}（{ins.part}）" for no, ins in enumerate(plan.instruments, 1)) or "  （なし）"
        lead = plan.lead
        secs = list(plan.lead_sections)
        quiet = {s.kind for s in self.design or [] if not getattr(s, "has_vocal", True)}
        # 主旋律の楽器の行は、形が歌もの以外か lead／lead_sections を書いたときだけ（歌もので書いていない編成の見せ方は以前のまま）
        lead_rows = ((f"主旋律の楽器（lead）: {lead if lead is not None else 'なし'}"
                      + (f" ／ 弾く区間（lead_sections）: {' / '.join(secs)}" if secs else "") + "\n"
                      + (("【設計の区間名】" + " / ".join(s.kind + ("（主旋律なし）" if s.kind in quiet else "")
                                                       for s in self.design) + "\n") if self.design else ""))
                     if (self.form != "song" or lead is not None or secs) else "")
        return (f"【いまの編成】ドラム: {plan.drum_kit} ／ ハモリ: {plan.harmony}\n"
                f"楽器（番号. bank-音色番号）:\n{rows}\n"
                + lead_rows
                + f"【音色】\n{_mel}\n{_extra}\n"
                f"【ドラムキット】{kits} / none（ドラムを使わない）\n【ハモリ】none / accent / full")

    def patch_problem(self, plan: Instrumentation, patch: InstrumentationPatch) -> str:
        """差分の楽器の番号が【いまの編成】の楽器の数を超えていないか・主旋律の楽器と外す楽器の書き方
        （差分の検証 `song_fix.guard_patch` が頼む間だけ読む）。"""
        return "／".join(x for x in (fix_no_problem(getattr(patch, "instruments", None) or [], len(plan.instruments)),
                                     lead_patch_problem(plan, patch)) if x)

    def apply(self, plan: Instrumentation, patch: InstrumentationPatch) -> tuple[Instrumentation, str]:
        upd, done = {}, []
        k = strip_ascii_ws(getattr(patch, "drum_kit", ""))
        if k:
            upd["drum_kit"] = k
            done.append(f"ドラムの種類を {plan.drum_kit} → {k}")
        h = strip_ascii_ws(getattr(patch, "harmony", ""))
        if h:
            upd["harmony"] = h
            done.append(f"ハモリの方針を {plan.harmony} → {h}")
        # ★楽器の数を超える番号は、差分の検証（`patch_problem`・`song_fix.guard_patch`）が差し戻すので来ない（2026-10-02。以前はここで黙って捨てた）。
        #   型を通らずに来たら（保存から戻した差分など）当てずに、当てなかったことを直しの記録に書く
        allf = list(getattr(patch, "instruments", None) or [])
        fixes, seen = [], set()
        for f in allf:                                      # 範囲外と同じ番号の2つ目は当てない（2026-10-03・同じ番号は1回だけ）
            if 1 <= f.no <= len(plan.instruments) and f.no not in seen:
                seen.add(f.no)
                fixes.append(f)
        if len(fixes) < len(allf):
            done.append("当てなかった: " + fix_no_problem(allf, len(plan.instruments)))
        ins = [i.model_copy() for i in plan.instruments]
        if fixes:
            for f in fixes:
                bank = f.bank if f.bank is not None else ins[f.no - 1].bank   # 空＝いまのバンクのまま（2026-10-04）
                done.append(f"楽器{f.no}の音色を {ins[f.no - 1].bank}-{ins[f.no - 1].gm} → {bank}-{f.gm}")
                ins[f.no - 1] = ins[f.no - 1].model_copy(update={"gm": f.gm, "bank": bank})
            upd["instruments"] = ins
        # ★**主旋律の楽器・弾く区間・外す楽器**（2026-10-03）。番号はどれも【いまの編成】の番号で読む。当てられない書き方
        #   （範囲外・主旋律の楽器を drop_lead なしで外す ほか）は差分の検証（`lead_patch_problem`）が差し戻すので来ない。
        #   型を通らずに来たら、主旋律の楽器と外す楽器の直しは当てずに、当てなかったことを記録に書く
        if why := lead_patch_problem(plan, patch):
            done.append("当てなかった（主旋律の楽器・外す楽器）: " + why)
        else:
            lead = plan.lead
            if patch.drop_lead:
                lead = None
                done.append(f"主旋律の楽器を外した（楽器{plan.lead} → なし）")
            elif patch.lead is not None:
                lead = patch.lead
                done.append(f"主旋律の楽器を {plan.lead if plan.lead is not None else 'なし'} → 楽器{lead}")
            if patch.clear_lead_sections:
                upd["lead_sections"] = []
                done.append("主旋律を弾く区間を空（主旋律のある区間の全部）に")
            elif patch.lead_sections:
                upd["lead_sections"] = [s.strip() for s in patch.lead_sections]
                done.append(f"主旋律を弾く区間を {'／'.join(upd['lead_sections'])} に")
            rm = sorted(set(patch.remove_instruments or []))
            if rm:
                keep = [no for no in range(1, len(ins) + 1) if no not in rm]
                upd["instruments"] = [ins[no - 1] for no in keep]
                done.append(f"楽器{'・'.join(map(str, rm))}を外した（残り{len(keep)}本）")
                # 残る楽器の番号は詰まる。主旋律の楽器は同じ楽器を指したまま番号だけ付け直す（外す楽器なら drop_lead が要る＝上の検証）
                if lead is not None and lead != keep.index(lead) + 1:
                    done.append(f"主旋律の楽器の番号を付け直した（同じ楽器: {lead} → {keep.index(lead) + 1}）")
                    lead = keep.index(lead) + 1
            if lead != plan.lead:
                upd["lead"] = lead
        if not upd:
            return plan, ("／".join(done) or "直しなし") + (f"／直さなかった理由: {patch.trouble}"
                                                          if getattr(patch, "trouble", "") else "")
        return plan.model_copy(update=upd), "／".join(done)



def instrument_name(ins: Instrument) -> str:
    """音色名を**音源から**引く（読めなければ GM の標準名）。"""
    from sf2_presets import read_presets
    banks = read_presets(soundfont() or "")
    got = (banks.get(ins.bank) or {}).get(ins.gm - 1)
    return got or gm_name(ins.gm)


def describe(plan: Instrumentation) -> str:
    """編曲工程へ渡す説明（**選ばれた楽器だけ**を並べる）。

    ★**その楽器にできる奏法も一緒に見せる**（2026-09-23）。編曲工程は名前で奏法を選ぶが、
      **何が選べるかは楽器で決まる**ので、渡さないと外れを書く——実機で
      「ベースに `fall`」「和音のリードに `fall`」と書かれ、**3本が弾かれた**。
      **受けられるものを先に見せる**のが、渡した側の仕事。
    ★**ドラムはそのキットで鳴らせる打楽器だけを見せる**（2026-09-29・`timbre_map.kit_drum_menu`）。キットによって
      入っている打楽器が違う（オーケストラ用キットはハイハット・シンバルの番号がティンパニ）のに、どのキットでも同じ一覧を見せていた。
    """
    from articulation import family_of                      # noqa: E402（循環回避）
    from timbre_map import kit_drum_menu                    # noqa: E402
    lines = []
    for i, ins in enumerate(plan.instruments, 1):
        _fam, _allow = family_of(max(0, int(ins.gm) - 1))    # GM は1始まり
        _note = ("／奏法: " + " ".join(sorted(_allow))) if _allow else "／奏法: 無し（音の中で高さが動かない）"
        lines.append(f"  {i}. **{instrument_name(ins)}**（{ins.bank}-{ins.gm}）— {ins.part}{_note}"
                     + (f"\n     （この楽器にさせたいこと）{ins.reason}" if ins.reason else ""))
    # ★**主旋律の楽器を伴奏・彩りへ知らせる**（2026-10-03・F10②・F19）。音量は伴奏の `lead_gain`。書いていない編成は以前のまま
    lead = plan.lead
    secs = list(plan.lead_sections)
    lead_line = ""
    if lead is not None and 1 <= lead <= len(plan.instruments):
        lead_line = (f"\n  主旋律: 楽器{lead}（{instrument_name(plan.instruments[lead - 1])}）が旋律の工程の音を"
                     f"そのまま弾く（区間: {' / '.join(secs) if secs else '主旋律のある区間の全部'}・音は機械が置く）。"
                     "**主旋律を parts に書き写さない**。この楽器の音量は伴奏の `lead_gain` に書く")
    return ("【この曲の編成】" + plan.reason + "\n" + "\n".join(lines)
            + lead_line
            + f"\n  ドラム: {plan.drum_kit}"
            + ("" if strip_ascii_ws(plan.drum_kit) == "none" else
               "\n    このキットで鳴らせる打楽器（drums の inst はこの名前だけ・かっこ内はその音の中身）: "
               + kit_drum_menu(plan.drum_kit))
            + f"\n  ハモリ: {plan.harmony}"
            + (f"（{plan.harmony_reason}）" if plan.harmony_reason else ""))
