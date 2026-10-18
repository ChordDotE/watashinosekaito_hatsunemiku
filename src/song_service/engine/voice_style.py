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

"""**声とスタイルプリセットを vpr に設定する**（ミクの歌い方を選べるようにする）。

## 経緯

以前「スタイルプリセットはプログラムから設定できない」と答えたが**誤りだった**。
人が V6 で設定して保存した vpr と、こちらが生成した vpr を**差分で比べたら**、
指定に必要なのは次の4か所だけと分かった（2026-08-22）:

    tracks[].parts[].aiVoice.compID    どの声で歌うか（ORIGINAL / SOFT …）
    tracks[].parts[].styleName         スタイルの表示名（"MIKU Dark" 等）
    tracks[].parts[].stylePresetID     スタイルの UUID
    voices[]                           プロジェクトが使う声の一覧（**登録が要る**）

> **教訓**: 「できない」と答える前に、**実データを差分で見る**。
> V6 のインストール先に定義ファイルは無かったが、**成果物の中に答えがあった**。

## 2026-08-29 の訂正: **4か所では足りなかった（音が変わっていなかった）**

上の「4か所だけ」は**書き出された vpr を見て言えること**であって、
**V6 がそれを読んで再適用するか**は確かめていなかった。実際に確かめたら適用されていない。

喋りの vpr で声2種 × スタイル7種を並べて人がミックスダウンした結果、
**スタイル違いの7版がサンプル単位で完全に同一**だった（差が小さいのではなく同じ音）。
変わったのは声（ORIGINAL / SOFT）だけ。

V6 で手動でスタイルを当て直して上書き保存した vpr を読むと、実体が分かった:

    tracks[].parts[].midiEffects   SingingSkill / VoiceColor / RobotVoice / Breath / Take
    tracks[].parts[].audioEffects  EQ・コンプ（スタイルが持つ音声加工）

**そして有効・無効を決めているのは各効果の `isBypassed`**。
`styleName` と `stylePresetID` だけ書くと、V6 は効果の器を作るが
**`isBypassed: true`（無効）のまま置く**——これが「名前は Dark なのに音は素のまま」の正体。

| | VoiceColor | S5Expression | S5Air | S5Character |
|---|---|---|---|---|
| No Effect | `isBypassed: true` | 50 | 0 | 0 |
| MIKU Dark（手で適用）| **`isBypassed: false`** | **40** | **2** | **-20** |
| MIKU Whisper（名前と ID だけ）| `isBypassed: true` | 50 | 0 | 0 |

S5 系の値は `.vsstyle` の `midiEffect.VoiceColor.parameter` と完全一致する。
そこで下の `STYLE_MIDI` に14種ぶんを持ち、`apply_choice` が
**効果そのものを書き込む**ようにした（経緯 → `docs/plan/voice_tone_control_study.md` §6-6）。

## ID はどう手に入れるか

V6 のインストール先にスタイル定義が見つからなかったので、
**人が1度設定して保存した vpr から採取**する（vpr は zip。`Project/sequence.json` の `voices[]` と
`tracks[].parts[]` の `styleName` / `stylePresetID` を読む。採取用の関数 `collect_from_vpr` は呼ばれていなかったので 2026-09-28 に消した）。
UUID は固定なので、一度集めれば以降ずっと使える。

## 前提

| # | 前提 | 崩れると |
|---|---|---|
| 1 | `stylePresetID` は環境に依らず固定 | 別 PC で無効な ID になる（V6 が既定へ戻すと思われる）|
| 2 | 使う声は `voices[]` に登録が要る | 声が切り替わらない |
| 3 | 未知のスタイル名は**設定しない**（黙って既定になるより明示的に失敗させる）| — |
"""
from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path

from song_checks import edge_ws_problem, strip_ascii_ws, trim_edge_ws   # 半角の名前の前後の空白（仕様書 §0-6・2026-10-03）

# ---- 実測で分かっている ID（人が V6 で設定した vpr から採取・2026-08-22）----
VOICES: dict[str, str] = {
    "HATSUNE_MIKU_V6_ORIGINAL": "BHKCEKSYNBXG3HB2",
    "HATSUNE_MIKU_V6_SOFT": "BNFDGP9ZRFSH5N9F",
}

# **全14種**（人が V6 で1つずつ設定した vpr から採取・2026-08-22）
STYLES: dict[str, str] = {
    "No Effect": "0c29827a-4289-495d-94d2-e23602d346c6",
    "MIKU Default": "6b38bc04-4034-4bda-b60e-388d507dbac3",
    "MIKU Boyish": "0b7372fb-6d1a-4f4b-946f-531a590c9890",
    "MIKU Boyish+": "138bdbb0-ff4f-4316-8c53-fa9872b6d4d1",
    "MIKU Breath": "a145d382-8fb5-4d8f-8313-49f21cb74cb3",
    "MIKU Breath+": "5b0ed81f-a3a5-448d-b8ee-921ff5730479",
    "MIKU Childish Hard": "a8f6d8bb-2bba-43af-9c07-65e1d54039f7",
    "MIKU Childish Soft": "6d3ce47f-168e-414e-a1f9-a5f1d7ed9abd",
    "MIKU Cute": "a63f4f67-bc1c-4ba7-8825-f43c4b658a62",
    "MIKU Cute+": "a29266b7-a2b2-4b32-b9ab-4bc8481b37eb",
    "MIKU Dark": "c0b199f9-feeb-4635-a226-5628fbafe9a4",
    "MIKU Robot": "c3bb4703-9474-4f1e-8501-c3076bfd6594",
    "MIKU Robot Hard": "81c25a5f-99aa-4259-979e-5007b4e07291",
    "MIKU Robot Soft": "50c26494-d458-4884-bdee-4b67ba09a04d",
    "MIKU Whisper": "0c07d084-4083-413a-96f9-be16e8de5da5",
}

# ---------------------------------------------------------------------------
# スタイルが実際に何をするか
# ---------------------------------------------------------------------------
# ★**一次情報**。V6 同梱の定義ファイルから読んだ（2026-08-22）。
#   出どころ: `C:\Program Files\Common Files\VOCALOID6\StylePreset\Extra\<UUID>.vsstyle`（JSON）
#   Web の製品ページには個別スタイルの説明が無かったので、**これが最良の出どころ**。
#
# 聞こえ方を分けているのは **3つの軸**（音声加工〔EQ・コンプ〕は
# **13種が共通で、MIKU Childish Hard だけ別**。2026-08-29 に全 .vsstyle を突き合わせて確認。
# 以前ここには「14種すべて同一」と書いていたが誤りだった）:
#
#   air        息の量        -48（息を抜く＝硬い）  〜 +100（息だけ＝囁き）
#   character  声の幼さ・太さ  -20（低め・落ち着く）  〜  +53（かなり幼い・軽い）
#   expression 抑揚の大きさ    19（ほぼ平坦）        〜   61（大きく揺れる）
#
# **この数値は LLM に渡さない**——`S5Air=-48` を見せても選びようがないため。
# 渡すのは下の STYLE_HINT（数値を「どう聞こえるか／どこに置くか」へ翻訳したもの）。
# ここに残すのは**根拠**として。ヒントの文言が疑わしくなったら、この表と突き合わせる。
STYLE_PARAMS: dict[str, dict] = {
    # name:               type            color      air  chara  expr
    "MIKU Default":       dict(type="Lead",    color="Clean",   air=0,   chara=0,   expr=50),
    "MIKU Boyish":        dict(type="Lead",    color="Warm",    air=0,   chara=-5,  expr=55),
    "MIKU Boyish+":       dict(type="Lead",    color="Bright",  air=0,   chara=-9,  expr=55),
    "MIKU Breath":        dict(type="Lead",    color="Whisper", air=20,  chara=3,   expr=35),
    "MIKU Breath+":       dict(type="Lead",    color="Whisper", air=38,  chara=0,   expr=40),
    "MIKU Childish Hard": dict(type="Lead",    color="Dreamy",  air=0,   chara=53,  expr=53),
    "MIKU Childish Soft": dict(type="Lead",    color="Soft",    air=0,   chara=36,  expr=26),
    "MIKU Cute":          dict(type="Lead",    color="Cute",    air=8,   chara=9,   expr=43),
    "MIKU Cute+":         dict(type="Lead",    color="Cute",    air=8,   chara=19,  expr=43),
    "MIKU Dark":          dict(type="Lead",    color="Dark",    air=2,   chara=-20, expr=40),
    "MIKU Robot":         dict(type="Robotic", color="Clean",   air=-48, chara=12,  expr=55),
    "MIKU Robot Hard":    dict(type="Robotic", color="Hard",    air=-48, chara=12,  expr=61),
    "MIKU Robot Soft":    dict(type="Robotic", color="Soft",    air=-28, chara=11,  expr=42),
    "MIKU Whisper":       dict(type="Backing", color="Whisper", air=100, chara=5,   expr=19),
}

# **LLM に渡すのはこれ**。上の数値を「どう聞こえるか／どんな曲・どの区間に置くか」へ翻訳した。
# 書式は `どう聞こえるか｜使いどころ` で揃える（選ぶ側が横に比べられるように）。
STYLE_HINT: dict[str, str] = {
    "MIKU Default":
        "癖のない素直な声。抑揚は中くらい｜曲想が決まらないとき・他と混ぜて基準にしたいとき",
    "MIKU Boyish":
        "少しだけ低く芯がある中性的な声。抑揚はやや大きめ｜ロック・応援歌・力強いサビ",
    "MIKU Boyish+":
        "Boyish をもう一段低く。より硬派｜バンドサウンド・攻めた曲",
    "MIKU Breath":
        "息を多めに含み、抑揚を抑えた声。語りかけるように聞こえる｜バラードのAメロ・落ちサビ",
    "MIKU Breath+":
        "Breath よりさらに息が多い。ほとんど呟き｜イントロの入り・曲のいちばん静かな所",
    "MIKU Childish Hard":
        "かなり幼く軽い声で、抑揚も大きい。元気で跳ねる｜アップテンポなポップス・童謡調・ネタ曲",
    "MIKU Childish Soft":
        "幼いが抑揚は小さく、平坦でやわらかい｜子守唄・ゆったりした可愛い曲",
    "MIKU Cute":
        "わずかに幼く、息も少し混ざる明るい声｜J-POP 全般の主旋律。**最も無難で使いやすい**",
    "MIKU Cute+":
        "Cute をもう一段幼く。甘さが増す｜アイドル系・キラキラした曲",
    "MIKU Dark":
        "低く落ち着いた声。息は少なく引き締まる｜切ない曲・重い曲・シリアスな歌詞",
    "MIKU Robot":
        "息をほぼ完全に抜いた無機質な声。抑揚は大きい｜テクノ・EDM・電子音メインの曲",
    "MIKU Robot Hard":
        "Robot をさらに抑揚強く。攻撃的で硬い｜激しいエレクトロ・ボカロックのサビ",
    "MIKU Robot Soft":
        "無機質だが角が丸く、抑揚も普通｜アンビエント寄りの電子音・淡々とした曲",
    "MIKU Whisper":
        "息だけで声が出ない（全音域で無声。実測）。**音程のある声部には使えない**｜"
        "息づかいの質感・効果音的な息の差し込み専用。音程のあるハモリは Default・Breath・Breath+",
    "No Effect":
        "加工なし（素の声）｜スタイルの影響を切り分けたいとき",
}

# ★**主旋律に使ってはいけないもの**（V6 の分類が Lead Vocal でないもの）。
LEAD_UNSUITABLE = {n for n, p in STYLE_PARAMS.items() if p["type"] == "Backing"}

#: ★**声の選択肢**（LLM が書く名前 → V6 の声の名前・`VOICES` のキー）。**書けるのはこの2つの名前だけ**
#:   （編成の `Instrumentation.voice` の説明が見せている名前・記法の仕様書 §15-3）。
#:   2026-10-02（判断6・本人「おすすめで良い」）まで、別名の表 `VOICE_ALIAS`（`オリジナル`・`標準`・`ソフト`・`やさしい`・
#:   V6 の名前 `HATSUNE_MIKU_V6_ORIGINAL` ほか）と大小を無視して読んでいた＝同じ声に書き方が何通りもあった。
#:   マスターの言葉（「やわらかい声で」）から声を選ぶのは編成の LLM の仕事で、機械は言葉を読み替えない。
VOICE_CHOICES: dict[str, str] = {
    "original": "HATSUNE_MIKU_V6_ORIGINAL",
    "soft": "HATSUNE_MIKU_V6_SOFT",
}

#: **所見の手がかりだけ**（以前の別名 → 正式な名前）。**読み替えには使わない**——知らない名前の所見に
#: 「『…』のように」と添えて書き直させるため（打楽器名の `timbre_map._DRUM_HINT` と同じ形）。
_VOICE_HINT = {
    "オリジナル": "original", "標準": "original", "default": "original",
    "hatsune_miku_v6_original": "original",
    "ソフト": "soft", "やわらかい": "soft", "やさしい": "soft", "hatsune_miku_v6_soft": "soft",
}
_STYLE_HINT_NAMES = {
    "dark": "MIKU Dark", "ダーク": "MIKU Dark", "暗い": "MIKU Dark", "重い": "MIKU Dark",
    "cute": "MIKU Cute", "キュート": "MIKU Cute", "かわいい": "MIKU Cute",
    "whisper": "MIKU Whisper", "ウィスパー": "MIKU Whisper", "ささやき": "MIKU Whisper",
    "囁き": "MIKU Whisper", "breath": "MIKU Breath", "息": "MIKU Breath",
    "boyish": "MIKU Boyish", "ボーイッシュ": "MIKU Boyish", "中性的": "MIKU Boyish",
    "childish": "MIKU Childish Soft / MIKU Childish Hard", "幼い": "MIKU Childish Soft / MIKU Childish Hard",
    "robot": "MIKU Robot", "ロボット": "MIKU Robot", "機械的": "MIKU Robot",
    "default": "MIKU Default", "標準": "MIKU Default", "素直": "MIKU Default",
    "none": "No Effect", "なし": "No Effect", "素": "No Effect",
}


def menu(*, for_lead: bool = True) -> str:
    """**LLM に見せる選択肢**（曲想からスタイルを選ばせるため）。

    数値ではなく「どう聞こえるか｜使いどころ」を渡す。
    `for_lead=True` のときはハモリ専用のものを除く（主旋律に置かれると痩せるため）。
    """
    lines = [
        "【ミクの歌い方（スタイルプリセット）】",
        "選ぶ軸は3つ: **息の多さ**（息が多い＝静か・近い / 少ない＝硬い・機械的）、",
        "**声の幼さ**（幼い＝明るく軽い / 低い＝落ち着き重い）、**抑揚の大きさ**。",
        "曲の激しさではなく**歌詞の距離感**で選ぶとよい（囁くのか、張り上げるのか）。",
        "",
    ]
    for k, v in STYLE_HINT.items():
        if for_lead and k in LEAD_UNSUITABLE:
            continue
        lines.append(f"  {k:<20} {v}")
    if for_lead:
        skipped = ", ".join(sorted(LEAD_UNSUITABLE))
        lines.append(f"\n  （{skipped} はハモリ専用のため主旋律の選択肢から外してある）")
    return "\n".join(lines)


# **曲づくりの途中で、編成が決めた歌い方**（`song_band.run_harmony_plan` が置き、`song_place.make_score` が楽譜の
# `voice_choice` へ写す）。vpr へ焼くのは**楽譜の `voice_choice` だけ**（`score_to_vpr.voice_choice_for`）。
# ★書き出しはここを読まない（2026-09-30）。同じプロセスの中で曲をまたいで残るので、書き出す曲（カバーなど）とは別の曲の決定でありうる。
CHOICE: dict[str, str] = {}


def set_choice(voice: str = "", style: str = "", harmony_style: str = "") -> None:
    CHOICE.clear()
    CHOICE.update({k: v for k, v in
                   (("voice", voice), ("style", style), ("harmony_style", harmony_style)) if v})


# ---------------------------------------------------------------------------
# スタイルを「効く形」で書き込むための材料（2026-08-29）
# ---------------------------------------------------------------------------
# ★一次情報。全 .vsstyle の `midiEffect` から採取した。
#   byp は .vsstyle の `Bypass`（1=バイパス / 0=有効）をそのまま持つ。
#   vpr 側の `isBypassed` は bool なので bool(byp) で写す。
# ★`take` は **Take.Lane630**（2026-08-29 追加）。
#   当初は「スタイルの属性ではなく V6 内部のテイク番号（操作の回数で変わる）」と判断し、
#   既存値を引き継ぐ実装にしていたが、**誤りだった**:
#     ・`.vsstyle` に `Take.Lane630` が載っている（＝スタイルの属性）
#     ・本人が V6 で14スタイルを当てて保存した vpr と **14/14 一致**
#     ・当てた順は 5,5,7,9,8,8,4,4,5,0,8,8,5,5 で、操作の回数（1,2,3…）ではない
#   「他の vpr ではばらつく（0 と 5）」ように見えたのは、**既存値を引き継ぐこの不具合自身の足跡**。
STYLE_MIDI: dict[str, dict] = {
    # name:               VoiceColor(byp, air, chara, expr)   Breath(byp, mode, exhale)  Robot(byp, mode)  Take
    "MIKU Default":       dict(vc=(1, 0, 0, 50),   br=(1, 1, 5), rv=(1, 1), take=0),
    "MIKU Boyish":        dict(vc=(0, 0, -5, 55),  br=(1, 0, 4), rv=(1, 2), take=5),
    "MIKU Boyish+":       dict(vc=(0, 0, -9, 55),  br=(1, 0, 4), rv=(1, 2), take=5),
    "MIKU Breath":        dict(vc=(0, 20, 3, 35),  br=(0, 0, 4), rv=(1, 0), take=7),
    "MIKU Breath+":       dict(vc=(0, 38, 0, 40),  br=(0, 0, 4), rv=(1, 1), take=9),
    "MIKU Childish Hard": dict(vc=(0, 0, 53, 53),  br=(0, 1, 6), rv=(1, 0), take=8),
    "MIKU Childish Soft": dict(vc=(0, 0, 36, 26),  br=(0, 1, 5), rv=(1, 0), take=8),
    "MIKU Cute":          dict(vc=(0, 8, 9, 43),   br=(0, 1, 4), rv=(1, 1), take=4),
    "MIKU Cute+":         dict(vc=(0, 8, 19, 43),  br=(0, 1, 4), rv=(1, 1), take=4),
    "MIKU Dark":          dict(vc=(0, 2, -20, 40), br=(0, 1, 5), rv=(1, 1), take=5),
    "MIKU Robot":         dict(vc=(0, -48, 12, 55), br=(0, 2, 3), rv=(0, 1), take=8),
    "MIKU Robot Hard":    dict(vc=(0, -48, 12, 61), br=(0, 2, 3), rv=(0, 0), take=8),
    "MIKU Robot Soft":    dict(vc=(0, -28, 11, 42), br=(0, 0, 4), rv=(0, 2), take=5),
    "MIKU Whisper":       dict(vc=(0, 100, 5, 19), br=(0, 1, 4), rv=(1, 0), take=5),
}

# **EQ・コンプ**。V6 が実際に書き出した vpr から採取した（値は 0..1 に正規化済み）。
# `.vsstyle` の生値（High F=122 等）から自力で正規化はできない——
# パラメータごとに尺度が違い（周波数は対数・ゲインは /360・Q は /60 …）、
# 1例からは規則を導けなかったため、**書き出された実物をそのまま持つ**。
# 13種で共通。MIKU Childish Hard だけ Gain が違う（下の _AUDIO_OVERRIDE で差分を当てる）。
_AUDIO_EFFECTS: list[dict] = [
    {"id": "C7E4ED13-FF1E-4fc2-a87c-65604789469D", "isBypassed": False, "isFolded": False,
     "parameters": [{"name": "HPF", "value": 0.0}, {"name": "High F", "value": 0.96875},
                    {"name": "High G", "value": 0.5416666865348816},
                    {"name": "High Mid F", "value": 0.7833333611488342},
                    {"name": "High Mid G", "value": 0.5555555820465088},
                    {"name": "High Mid Q", "value": 0.25},
                    {"name": "Low F", "value": 0.1764705926179886},
                    {"name": "Low G", "value": 0.3333333432674408},
                    {"name": "Low Mid F", "value": 0.4000000059604645},
                    {"name": "Low Mid G", "value": 0.4444444477558136},
                    {"name": "Low Mid Q", "value": 0.13333334028720856}]},
    {"id": "4D1F311D-DA90-459d-B581-003D003A1E5E", "isBypassed": False, "isFolded": False,
     "parameters": [{"name": "Attack", "value": 0.37168142199516296},
                    {"name": "Knee", "value": 0.5},
                    {"name": "Ratio", "value": 0.3499999940395355},
                    {"name": "Release", "value": 0.5181159377098083},
                    {"name": "Threshold", "value": 0.3333333432674408}]},
    {"id": "C7CE3316-6FFD-4140-8a7c-0AA8B8108549", "isBypassed": False, "isFolded": False,
     "parameters": [{"name": "Gain", "value": 0.7142857313156128}]},
]
# **13種と違うのはここだけ**（2026-08-29 に人が V6 で当てて保存した vpr から採取）。
# 全パラメータを比べたところ、`MIKU Childish Hard` は最終段の Gain が
# 0.714286 → 0.678571 に下がるだけで、EQ もコンプも共通と同一だった。
_AUDIO_OVERRIDE: dict[str, dict[str, dict[str, float]]] = {
    "MIKU Childish Hard": {
        "C7CE3316-6FFD-4140-8a7c-0AA8B8108549": {"Gain": 0.6785714030265808},
    },
}

_SINGING_SKILL_NAME = "75F04D2B-D8E4-44b8-939B-41CD101E08FD"


def _midi_effects(style_name: str, lane630: int | None = None) -> list[dict]:
    """V6 が書くのと同じ形の `midiEffects` を組む。

    素の声（No Effect）は**全効果をバイパス**した器だけを置く。

    `lane630` は通常 **None**（スタイルの値を使う）。
    数値を渡したときだけそれを優先する（既存 vpr の値を保ちたい場合の逃げ道）。
    """
    p = STYLE_MIDI.get(style_name)
    vc_byp, air, chara, expr = (True, 0, 0, 50) if p is None else (
        bool(p["vc"][0]), p["vc"][1], p["vc"][2], p["vc"][3])
    br_byp, br_mode, br_exh = (True, 1, 5) if p is None else (
        bool(p["br"][0]), p["br"][1], p["br"][2])
    rv_byp, rv_mode = (True, 1) if p is None else (bool(p["rv"][0]), p["rv"][1])
    take = lane630 if lane630 is not None else (0 if p is None else p.get("take", 0))
    return [
        {"id": "SingingSkill", "isBypassed": True, "isFolded": False,
         "parameters": [{"name": "Skill", "value": 5}, {"name": "Amount", "value": 5},
                        {"name": "Name", "value": _SINGING_SKILL_NAME}]},
        {"id": "VoiceColor", "isBypassed": vc_byp, "isFolded": False,
         "parameters": [{"name": "Exciter", "value": 0}, {"name": "Growl", "value": 0},
                        {"name": "Breathiness", "value": 0}, {"name": "Air", "value": 0},
                        {"name": "Mouth", "value": 0}, {"name": "Character", "value": 0},
                        {"name": "S5Expression", "value": expr},
                        {"name": "S5Air", "value": air},
                        {"name": "S5Character", "value": chara}]},
        {"id": "RobotVoice", "isBypassed": rv_byp, "isFolded": False,
         "parameters": [{"name": "Mode", "value": rv_mode}]},
        {"id": "Breath", "isBypassed": br_byp, "isFolded": False,
         "parameters": [{"name": "Mode", "value": br_mode}, {"name": "Type", "value": 0},
                        {"name": "Exhalation", "value": br_exh}]},
        # Take は**スタイルの属性**（`.vsstyle` の `Take.Lane630`）。上の表から書く。
        {"id": "Take", "isBypassed": False, "isFolded": False,
         "parameters": [{"name": "Lane630", "value": take}]},
    ]


def _write_style(part: dict, style: tuple[str, str]) -> str | None:
    """part にスタイルを**効く形で**書き込む。警告があれば返す。"""
    name, preset_id = style
    part["styleName"], part["stylePresetID"] = name, preset_id
    # ★既存の Lane630 を引き継がない（2026-08-29 是正）。スタイルの属性なので、
    #   引き継ぐと**別のスタイルのテイク番号が残る**——V6 の保存と食い違う唯一の項目だった。
    part["midiEffects"] = copy.deepcopy(_midi_effects(name))
    if name == "No Effect":
        part.pop("audioEffects", None)
        return None
    eff = copy.deepcopy(_AUDIO_EFFECTS)
    for e in eff:                                   # 共通と違う分だけ上書きする
        for pname, val in (_AUDIO_OVERRIDE.get(name, {}).get(e["id"]) or {}).items():
            for pr in e["parameters"]:
                if pr["name"] == pname:
                    pr["value"] = val
    part["audioEffects"] = eff
    return None


def apply_choice(vpr: Path, *, voice: str = "", style: str = "",
                 harmony_style: str = "", part_styles: list[str] | None = None,
                 verbose: bool = True) -> list[str]:
    """**主旋律とハモリに別々の歌い方**を当てる（2026-08-23）。

    歌トラックの**1本目を主旋律**、残りをハモリとみなす（`score_to_vpr` がその順で書く）。
    `MIKU Whisper` は V6 の分類こそ Backing Vocal だが、AIR=+100 で全音域が無声（息の音だけ）なので、
    音程のあるハモリ・声部には当てない（息づかいの質感・効果音的な息の差し込み専用。2026-10-05 の実測）。

    `part_styles` は**声部ごとの声**（2本目以降の歌トラックに順に当てる。空の欄は `harmony_style`）。
    楽譜から渡すのは `score_to_vpr.part_styles_for`（2026-09-27 に配線。それまで誰も渡していなかった）。

    あわせて `voices[]` を掃除する——テンプレート由来の**洛天依が残ったまま**で、
    パートの声だけミクに差し替えられていた（整合していない）。
    """
    # ★知らない名前は**当てずに据え置き、所見を返す**（最後の網）。名前は書く側の型で先に差し戻すのが本筋
    #   （`voice_problem` / `style_problem`・目録 表情-10）。所見には使える名前を添える。
    warns: list[str] = []
    v = resolve_voice(voice) if voice else None
    s_lead = resolve_style(style) if style else None
    s_har = resolve_style(harmony_style) if harmony_style else None
    if voice and not v:
        warns.append(f"{voice_problem(voice)}——声は据え置き")
    if style and not s_lead:
        warns.append(f"主旋律の{style_problem(style)}——据え置き")
    if harmony_style and not s_har:
        warns.append(f"ハモリの{style_problem(harmony_style)}——据え置き")
    # ★**声部ごとの声だけが決まっていても当てる**（2026-09-27・目録 声部-3）。以前は3つとも空なら
    #   ここで帰っていたので、声部の声（`part_styles`）だけの指定は黙って捨てられていた。
    if not (v or s_lead or s_har or any(part_styles or [])):
        return warns

    with zipfile.ZipFile(vpr) as z:
        blobs = {n: z.read(n) for n in z.namelist()}
    d = json.loads(blobs["Project/sequence.json"].decode("utf-8"))

    if v:                                                  # **voices[] を正しくする**
        d["voices"] = [{"compID": v[1], "name": v[0]}]
    n_lead = n_har = 0
    for tr in d.get("tracks", []):
        for p in tr.get("parts", []):
            if not p.get("aiVoice"):
                continue
            if v:
                p["aiVoice"]["compID"] = v[1]
            first = n_lead == 0
            # ★**声部ごとに歌い方を変えられる**（2026-09-23）。それまで
            #   **ハモリが何本あっても全部同じスタイル**だった（`s_har` 1つ）。
            #   掛け合いは「別の人が返している」ように聞こえてこそなので、
            #   声部ごとに指定できないと成立しない。指定が無ければ従来どおり `s_har`。
            pick = s_lead if first else (s_har or s_lead)
            if not first and part_styles:
                _i = n_har                                 # 2本目＝0番目の声部
                if _i < len(part_styles) and part_styles[_i]:
                    _s = resolve_style(part_styles[_i])
                    if _s:
                        pick = _s
                    else:
                        warns.append(f"{_i + 1}本目の声部の{style_problem(part_styles[_i])}"
                                     "——ハモリ全体の歌い方で鳴らす")
            if pick:
                w = _write_style(p, pick)      # 名前と ID だけでは適用されない（冒頭の訂正）
                if w and w not in warns:
                    warns.append(w)
            if first:
                n_lead += 1
            else:
                n_har += 1
    if not n_lead:
        warns.append("歌トラックが見つからない（歌い方を設定できなかった）")
        return warns
    # AI 歌唱トラック（type=2）でないと、声が V6 の既定に落ちる
    for tr in d.get("tracks", []):
        for p in (tr.get("parts") or []):
            if p.get("aiVoice") and tr.get("type") != 2:
                warns.append(f"トラック『{tr.get('name')}』が type={tr.get('type')}"
                             f"（AI 歌唱は 2）——V6 で声が既定に落ちる")

    blobs["Project/sequence.json"] = json.dumps(d, ensure_ascii=False).encode("utf-8")
    with zipfile.ZipFile(vpr, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in blobs.items():
            z.writestr(n, b)
    if verbose:
        print(f"  [歌い方] 声={v[0] if v else '（据え置き）'} / "
              f"主旋律={s_lead[0] if s_lead else '（据え置き）'}"
              + (f" / ハモリ{n_har}本={s_har[0]}" if n_har and s_har else "")
              + (f" / 声部ごと={[s or '既定' for s in part_styles]}" if any(part_styles or []) else ""))
    return warns


def _hint(name: str, table: dict[str, str], names) -> str:
    """所見に添える正式な名前（以前の別名・大小だけ違う書き方から）。見つからなければ空。読み替えには使わない。"""
    k = (name or "").strip()
    got = table.get(k.lower()) or table.get(k) or next((n for n in names if n.lower() == k.lower()), "")
    return f"——『{got}』のように" if got else ""


def voice_problem(name: str) -> str | None:
    """声の名前を検査する（**所見を返すだけ**・2026-09-27・目録 表情-10）。正式な名前なら None。

    ★それまで知らない名前は `apply_choice` で**当てずに据え置き**、所見は画面に出るだけで
      名前を書いた工程（編成）には届かなかった。**書く側の型の検証**（`field_validator`）でこれを使うと、
      LLM に「使える名前」つきで差し戻る（`instrumentation.Instrumentation.voice` が使っている）。
    ★**書けるのは `VOICE_CHOICES` の名前を1字も違わずに**（前後の半角の空白とタブだけは無視・仕様書 §0-6。全角の空白・改行などは所見）。**空も所見**——
      声には「空＝既定」の定義が無い（2026-10-02・判断6。以前は空を通し、何も当てずに V6 の既定の声で歌った）。
    """
    k = strip_ascii_ws(name)
    if k in VOICE_CHOICES:
        return None
    if why := edge_ws_problem(name):                       # 前後に全角の空白・改行などを付けない（2026-10-03）
        rest = voice_problem(trim_edge_ws(name))
        return f"声{why}" + (f"。そのうえで: {rest}" if rest else "")
    if not k:
        return f"声が空——使えるのは {' / '.join(VOICE_CHOICES)} のどれか1つ"
    return (f"声『{name}』は使えない{_hint(k, _VOICE_HINT, VOICE_CHOICES)}"
            f"（使えるのは {' / '.join(VOICE_CHOICES)} のどれか1つ。この綴りのまま書く）")


def style_problem(name: str, *, required: bool = False) -> str | None:
    """歌い方（スタイル）の名前を検査する（**所見を返すだけ**）。正式な名前なら None。`voice_problem` と同じ使い方。

    ★**書けるのは `STYLES` の名前を1字も違わずに**（`MIKU Dark`・`No Effect` …。前後の半角の空白とタブだけは無視・全角の空白などは所見）。
      2026-10-02（判断6）まで別名の表 `STYLE_ALIAS`（`dark`・`暗い`・`ささやき`・`幼い`→Childish Soft ほか）で読んでいた。
    `required=False`（既定）は**空を通す**——声部の `HarmonyPart.voice_style` は「空＝ハモリ全体の歌い方」と定義している。
    空の定義が無い欄（編成の `vocal_style`・`harmony_style`）は `required=True` で呼ぶ。
    """
    k = strip_ascii_ws(name)
    if k in STYLES:
        return None
    if why := edge_ws_problem(name):                       # 前後に全角の空白・改行などを付けない（2026-10-03）
        rest = style_problem(trim_edge_ws(name), required=required)
        return f"歌い方{why}" + (f"。そのうえで: {rest}" if rest else "")
    if not k:
        return (f"歌い方が空——使えるのは {' / '.join(STYLES)} のどれか1つ") if required else None
    return (f"歌い方『{name}』は使えない{_hint(k, _STYLE_HINT_NAMES, STYLES)}"
            f"（使えるのは {' / '.join(STYLES)} のどれか1つ。この綴りのまま書く）")


def resolve_voice(name: str) -> tuple[str, str] | None:
    """声の名前（`VOICE_CHOICES` のキー）→ (V6 の声の名前, compID)。正式な名前でなければ None（別名は読まない）。"""
    k = VOICE_CHOICES.get(strip_ascii_ws(name))
    return (k, VOICES[k]) if k else None


def resolve_style(name: str) -> tuple[str, str] | None:
    """歌い方の名前（`STYLES` のキー）→ (名前, UUID)。正式な名前でなければ None（別名は読まない）。"""
    k = strip_ascii_ws(name)
    return (k, STYLES[k]) if k in STYLES else None


# ※既にある vpr の声とスタイルだけを差し替える `apply_to_vpr`・AI 歌唱トラックの検査の写し `check_ai_tracks`・
#   ID の採取 `collect_from_vpr` は、どこからも呼ばれていなかったので 2026-09-28 に消した。当てるのは `apply_choice` の1本
#   （AI 歌唱トラック＝type=2 の検査もそちらにある。0 で書くと V6 は非 AI と解釈し、声が既定に落ちる）。
