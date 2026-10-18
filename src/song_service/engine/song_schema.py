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

"""本番形プロトタイプ: 楽曲スキーマ（調声・編曲ともパラメータ化）。

本実装では agent/tools/music/score_schema.py に対応。
- VocalTuning: per-note 調声意図（0..1）→ score_to_vpr が aiExp に写像（§3.2）
- ArrangementSpec/PartSpec: 編曲を「ステップ配列」で表現（固定パターン名でなく、
  LLM が任意のリズム/ボイシングを組めるパラメトリックモデル）
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

TICKS_PER_QUARTER = 480

#: **曲の形**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §2・F1）。歌もの・アカペラ（伴奏なし）・インスト（歌なし）。
#: 綴りはこの3つだけ。**空＝`song`**（読み手は `song_files.as_form`）。決めるのはコーディネーター（`draft_melody(form=…)`）で、
#: 工程の LLM の出力には形の欄を持たない。曲の記録 `Song.form` と楽譜 `SongScore.form` が持つ。
#: ※置き場はここ（楽譜が使うため・`song_model` は `song_schema` を読む側）。`song_model.Form` も同じもの。
#:   標準ライブラリだけの `song_files.FORMS` と同じ3つ（`tests/song_service/test_choice_enums.py` が突き合わせる）
Form = Literal["song", "acappella", "instrumental"]


# ============================ 調声 ============================
class VocalTuning(BaseModel):
    """per-note 調声意図（各 0..1・None は AI 既定0.5に委ねる）。aiExp へ写像。"""
    amplitude: float | None = Field(default=None, description="音量全体 → aiExp.amplitudeWhole")
    amplitude_attack: float | None = Field(default=None, description="→ aiExp.amplitudeStart")
    amplitude_release: float | None = Field(default=None, description="→ aiExp.amplitudeEnd")
    pitch_transition_in: float | None = Field(default=None, description="入りのピッチ遷移(すくい上げ) → aiExp.pitchTransitionStart")
    pitch_transition_out: float | None = Field(default=None, description="→ aiExp.pitchTransitionEnd")
    pitch_drift_in: float | None = Field(default=None, description="→ aiExp.pitchDriftStart")
    pitch_drift_out: float | None = Field(default=None, description="→ aiExp.pitchDriftEnd")
    #: ★**名前に反して ±0.9 半音も動く**（2026-09-23 実測: 0.0 で -0.94 / 1.0 で +0.90）。
    #:   「微調整」のつもりで振ると**音痴になる**。0.5 から離すときは半音の尺で考えること。
    pitch_fine: float | None = Field(default=None, description="→ aiExp.pitchFine（**±0.9半音動く**）")
    #: ★**書き先は `note.vibrato`**（2026-09-23）。`aiExp.vibrato*` は**効かない**ことを
    #:   V6 実機で測った（深さ 0.8 でも揺れ 0.003 半音＝何もしていないのと同じ）。
    #:   0〜1 をそのまま 127 倍して `depths` に書く（実測で線形・全開 ±0.52 半音）。
    vibrato_depth: float | None = Field(default=None, description="揺れの深さ → note.vibrato.depths")
    # ★**フォルマント（`formantWhole/Start/End`）は持たない**（2026-09-23）。
    #   vpr の定義にはあるので一度つないだが、**V6 実機で効かなかった**——
    #   0.0 と 1.0 で明るさ 2355Hz ↔ 2346Hz、音量も同じ（＝測定の底）。
    #   `aiExp.vibrato*` と同じ形の空振りなので、**つまみごと置かない**。
    #   効かないものを残すと、次に読む人が「使える」と思って組み込む。


class SongNote(BaseModel):
    lyric: str = Field(description="かな1〜2モーラ")
    key: int = Field(ge=0, le=127)
    start_tick: int = Field(ge=0)
    duration_tick: int = Field(gt=0)
    tuning: VocalTuning | None = None
    #: ★**歌の奏法**（2026-09-23）。しゃくり・落とし・ベンド。**伴奏と同じカタログ**を使う
    #:   （`articulation.py`）。書き出し先だけが違う——伴奏は MIDI の `pitchwheel`、
    #:   歌は vpr の `controllers.pitchBend`。
    #:   ※**次の音へ滑る系（`gliss`/`portamento`）は歌では使わない**——
    #:     V6 は放っておいても 100ms で次の音へ滑る（実測）ので、重ねると二重になり、
    #:     戻しの位置で音程が跳ねる。渡りは `tuning.pitch_transition_*` で伸び縮みさせる。
    artic: str = Field(default="", description="奏法の名前（空＝素で歌う）")
    artic_depth: float | None = Field(default=None, description="何半音動かすか（空＝標準）")
    #: ★**タイでまとめた続きの音の数**（2026-10-01・判-H）。旋律の ABC のタイ（`E2- | E2`）は1つの長い音として鳴らすが、
    #:   詞は ABC の歌詞行どおり書いた音ごとに印を書く（続きの音は `_`）。2番の詞（`song_resize`）がこの音に `1 + ties` 個の印を数える
    ties: int = Field(default=0, ge=0, description="タイでこの音にまとめた続きの音の数（詞の印は 1＋これ）")


class ChordSpan(BaseModel):
    chord: str = Field(description="C / Am / Cmaj7 / G7 等")
    start_tick: int = Field(ge=0)
    duration_tick: int = Field(gt=0)


class VoiceTrack(BaseModel):
    """メインに重ねる追加ボーカル声部（ハモリ/輪唱/オクターブ重ね/コーラス等）。
    vpr では別トラックになり、別シンガーも割り当て可能。notes はメインと同じ時間軸(start_tick)。"""
    title: str = Field(default="harmony", description="声部名（harmony/canon/double/chorus 等・表示用）")
    notes: list[SongNote] = Field(description="この声部の音符（メインと同じ tick 軸）")
    gain_db: float = Field(default=0.0, description="メイン(0dB)比のこの声部の相対音量")
    # ★**声部ごとの声**（2026-09-27・目録の声部-3）。声部の工程が狙いごとに選んだ歌い方を V6 まで運ぶ。
    #   以前は欄が無く、全部の声部がハモリ用の既定の歌い方になっていた（`score_to_vpr.part_styles_for` が読む）
    voice_style: str = Field(default="", description="この声部の歌い方（例 MIKU Breath）。空＝ハモリ用の既定")


# ============================ 編曲（パラメトリック）============================
class Tone(BaseModel):
    """鳴らす1音。**ふつうはコード構成音のインデックス**で指定する（コードに依存せず使い回せる）。

    ★**`key` を入れると、コードから独立した音程になる**（2026-09-17 追加）。
    対旋律（オブリガート）・リフ・ソロは**自分の旋律**を持つので、
    「いま鳴っているコードのどの音か」では書けない。

    > **なぜ足したか**: 伴奏が指定できるのは `root / chord / fifth / arp_up / arp_down` ——
    > **全部「コード構成音のどれを弾くか」**で、**独立した旋律を書く手段が無かった**。
    > そのため伴奏工程は reason に「**ピアノの対旋律**」と書いたのに、
    > データにする段でいちばん近い `arp_up`（分散和音）へ落ち、
    > **書いたつもりの対旋律がアルペジオになっていた**（2026-09-16『さあ立ち止まろう』）。
    > 同じ理由で、ソロが根音刻みになる件も直らなかった。
    """
    degree: int = Field(description="構成音index: 0=root,1=3rd,2=5th,3=7th（無ければ無視）")
    octave: int = Field(default=0, description="オクターブシフト")
    key: int | None = Field(default=None, description=(
        "**絶対音高**（MIDI ノート番号）。入っていれば `degree` より優先し、"
        "**コードに関係なく、この高さのまま鳴らす**（`register_octave` も足さない）。対旋律・リフ・ソロ用。"
        "※移調は**ここへ入れる前に済ませる**（`degree` と違い、あとから追従しない）"))


class StepEvent(BaseModel):
    """1小節パターン内の1イベント（拍位置・長さ・同時発音）。"""
    at_beat: float = Field(ge=0, description="小節内の拍位置（0=頭, 0.5=8分, 1=2拍目…）")
    dur_beats: float = Field(gt=0, description="長さ（拍）")
    tones: list[Tone] = Field(default_factory=list, description="同時発音（空=休符）")
    velocity: int = Field(default=64, ge=1, le=127)
    # ★**奏法はここに書かない**（2026-09-23 に `ArrangementSpec.articulations` へ移した）。
    #   当初は `articulation` / `articulation_bars` をここに置いたが、`steps` は
    #   **1小節ぶんのパターン**なので「フレーズの初めの1音だけ深く」が書けなかった。
    #   `articulation_bars`（どの小節に掛けるか）で粗く逃げていたが、**音単位には届かない**。


class DrumHit(BaseModel):
    inst: str = Field(description="打楽器名。GM の正式な名前（`bass_drum_1` / `acoustic_snare` / `closed_hi_hat` ほか・`timbre_map.GM_DRUM`）")
    at_beat: float = Field(ge=0)
    velocity: int = Field(default=100, ge=1, le=127)


class PartSpec(BaseModel):
    """伴奏1パート。pitched は steps、ドラムは drum_steps を使う。"""
    role: str = Field(description="bass/pad/strings/arpeggio/guitar/drums 等（表示用ラベル。ベースかどうかは `bass`）")
    # ★**ベースかどうかは伴奏の LLM が書いた `PartPlan.bass` をそのまま写す**（2026-10-02 本人決定・案b）。以前は組み立てが
    #   `role` の文字（bass で始まる）と `register_octave <= -2` から推し量っていた。既定値は置かない（保存済みの楽譜は
    #   `dev_tools/song_legacy/convert_record.py` が以前の判定で写す。黙って false で読むと、ベースの鳴り方が変わる）
    bass: bool = Field(description=(
        "ベースのパートか。true なら度数で書いた root を分数コードでは分母の音で弾き、和音の打鍵に7度・6度・テンションを積まない"
        "（`song_audio.pitched_events`）。ドラムのパートは false"))
    timbre: str = Field(default="", description="音色名（pitched用。drumsは無視）")
    gain: float = Field(default=0.5, ge=0, le=1.5, description=(
        "パートの音量。**強さ（velocity）とは別**に出す——MIDI ではチャンネルの CC11（エクスプレッション）"
        "（`midi_out`・2026-09-27）。強さに掛けるとアクセントが127に張り付いて消えていた"))
    register_octave: int = Field(default=0, description="パート全体のオクターブ（度数で書いた Tone にさらに加算。`key` で高さを書いた音には足さない）")
    bar_filter: str = Field(default="all", description="all / latter / first … どの小節に適用するか")
    steps: list[StepEvent] = Field(default_factory=list, description="pitched: 1小節パターン（コード毎に適用）")
    drum_steps: list[DrumHit] = Field(default_factory=list, description="drums: 基本1小節パターン")
    # ★**決まった形のフィル（`drum_fill` / `fill_every_bars`）は外した**（2026-09-27・目録の 打-2）。
    #   N小節ごとに、LLM が書いた小節を**タム16分＋クラッシュの固定フィルで丸ごと差し替えて**いた。
    #   小節は曲頭から数え、4拍前提なので 3/4 ではみ出した。LLM は `at_bars` で自分のフィルを書ける。
    #   ※前の版の楽譜にこの2つの鍵が残っていても読める（pydantic は知らない鍵を無視する）。鳴らしはしない。


class NoteArtic(BaseModel):
    """**音1つに掛ける奏法**（2026-09-23）。`steps` の外に置く。

    ★**パターンの中に書けない理由**: `steps` は「1小節ぶんのパターン」で、
      合う小節すべてに適用される。だから**「フレーズの初めの1音だけ深く落とす」が
      構造的に書けない**——同じ `StepEvent` が全小節で使い回されるため。
      本人「**初めの1音だけ深く落とすとか上げるとか、そういったこともあると思う**」。

      そこで**パターンはパターンのまま**にして、**鳴る音の側に重ねる**。
      位置（`tick`）で引くので、同じパターンから鳴る音でも1つずつ別に扱える。
    """
    part: int = Field(ge=0, description="`ArrangementSpec.parts` の何番目か（0始まり）")
    tick: int = Field(ge=0, description="その音が鳴り始める位置（曲頭からの tick）")
    name: str = Field(description="奏法の名前（scoop / fall / gliss …）")
    depth_st: float | None = Field(default=None, description=(
        "**何半音動かすか**。空なら楽器の標準（`articulation.py` が持つ）。\n"
        "**符号は奏法が決める**ので大きさだけ書く（フォールは下、ドゥイットは上）"))
    reason: str = Field(default="", description="なぜここでそうするか（標準から外すときは必ず）")


class ArrangementSpec(BaseModel):
    parts: list[PartSpec] = Field(default_factory=list)
    # ★**伴奏全体の音量 `accomp_gain` は外した**（2026-09-27・目録の 音量-4）。2026-09-06 に仮歌ミックスを
    #   廃止してから**どこにも効いておらず**、それでいて表示や道具の返事で「効いた」ように見えていた。
    #   歌と伴奏の釣り合いは vpr の伴奏トラックの初期音量（`score_to_vpr._accomp_gain_db`）が決める。
    #   ※前の版の楽譜にこの鍵が残っていても読める（知らない鍵は無視する）。
    #: ★**奏法は音ごと**（2026-09-23）。`parts` の中ではなく、ここに重ねる（上の注記）。
    articulations: list[NoteArtic] = Field(default_factory=list, description=(
        "音1つずつに掛ける奏法（空＝素で鳴らす。**前の版の曲は空のまま読める**）"))
    #: ★**人が MIDI の道具で手で書いた小節**（2026-09-26・`docs/plan/manual_edit_import.md`）。伴奏は
    #:   「楽器ごとの1小節のパターン × コード」なので、特定の小節の音は楽譜の言葉で書けない。楽器（`role`）×
    #:   小節の範囲ごとに手で書いた音をそのまま持ち、書き出すときは**パターンより優先**する。要素は dict:
    #:   `{"role", "start_bar", "end_bar", "notes": [[開始tick, 高さ, 長さ, 強さ], ...]}`（小節は0始まり・end は含まない）
    hand_bars: list[dict] = Field(default_factory=list, description="人が MIDI で手で書いた小節（パターンより優先）")
    #: ★**ドラムの種類**（2026-09-27）。編成の `drum_kit` を書き出しまで運ぶ。以前は楽譜に欄が無く、
    #:   `orchestra` と決めても ch10 の音色が切り替わらず、ティンパニのつもりの音が普通のタムで鳴っていた。
    drum_kit: str = Field(default="", description="ドラムの種類（編成の drum_kit。空＝音源の既定）")

    def is_empty(self) -> bool:
        """**伴奏が無い**（パートが0本）。アカペラ・楽譜から伴奏を作っていないカバー（2026-10-03・F1。以前は `arrangement` が None）。
        手で書いた小節・奏法はパートに掛かるもの（パートが無ければ鳴らない）なので見ない。"""
        return not self.parts


class LeadNote(BaseModel):
    """主旋律の楽器が弾く1音（旋律の工程の音を、詞なしでそのまま置いたもの・`song_place.place_lead`）。"""
    key: int = Field(ge=0, le=127)
    start_tick: int = Field(ge=0)
    duration_tick: int = Field(gt=0)


class LeadPart(BaseModel):
    """**主旋律を楽器で弾くパート**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §5-2・F2・F10②・F19・F20）。

    インストの主旋律と、歌ものの「歌と重ねて弾く」が同じ形。音は旋律の工程の音そのもの（同じ高さ・同じ長さ）で、
    楽器は編成の `lead`、音量は伴奏の `lead_gain`、弾く区間は編成の `lead_sections`（書いた範囲の音だけを置く）。
    ★**伴奏の楽譜（`arrangement`）の外に持つ**——伴奏は彩りを合わせるたびに組み直される（`song_band.run_colour` の `to_spec`）ので、
      そこに入れると組み直しで消える。書き出しは伴奏と同じ1か所（`midi_out.score_to_midi` が伴奏の MIDI の1トラックとして書く）。
    ★パターン（`PartSpec.steps`）では書けない（1小節の型の繰り返しなので旋律を持てない）。人の手直し（`hand_bars`）とは意味が違う
      （手直しは作り直しで引き継ぐ／捨てる対象・`manual_import.carry_hand_edits`）ので流用しない。
    ★**主旋律の楽器が無い曲（アカペラ・重ねない歌もの）も同じ欄を持ち、中身が空**（2026-10-03・F1 本人決定「データ形式は変えない」）。
      空＝楽器の番号 0・音色 ""・音量 0・区間も音も奏法も空（`LeadPart()`）。機械は既定の楽器を入れない。以前は `lead` が None
    """
    instrument: int = Field(default=0, ge=0, description="編成の楽器の番号（`Instrumentation.lead`・1始まり）。0＝主旋律の楽器が無い（空）")
    timbre: str = Field(default="", description="音色（伴奏のパートと同じ `gm<bank>-<番号>`）。空＝無い")
    gain: float = Field(default=0.0, ge=0, le=1.2, description="音量（伴奏の `lead_gain`・0.1〜1.2・伴奏のパートどうしの比）。0＝無い")
    sections: list[str] = Field(default_factory=list, description="弾く区間（編成の `lead_sections`。空＝主旋律のある区間の全部）")
    notes: list[LeadNote] = Field(default_factory=list)
    #: 奏法（音ごと）。`part` は奏法の表の P番号（伴奏のパートの次＝`len(arrangement.parts)`）の控えで、読み手は使わない——
    #: 伴奏のパートを外しても主旋律の奏法がずれないように、伴奏の `articulations` とは別に持つ
    articulations: list[NoteArtic] = Field(default_factory=list)

    @model_validator(mode="after")
    def _empty_or_whole(self):
        """楽器の番号 0（空）なら中身も全部空、番号があれば音色と音量（0.1〜1.2）がある——どちらかの形だけ。"""
        if self.instrument == 0:
            if self.timbre or self.gain or self.sections or self.notes or self.articulations:
                raise ValueError("主旋律の楽器の番号が 0（無い）なのに、音色・音量・区間・音・奏法のどれかが入っている")
        elif not self.timbre or not 0.1 <= self.gain <= 1.2:
            raise ValueError(f"主旋律の楽器（{self.instrument}番）に音色が無いか、音量 {self.gain} が 0.1〜1.2 の外")
        return self

    def is_empty(self) -> bool:
        """主旋律を楽器で弾かない（番号 0・中身が空）。"""
        return self.instrument == 0


class SongScore(BaseModel):
    title: str
    #: ★**曲の形**（2026-10-03・F1）。曲づくりの外（再生・状態・申し送り）は楽譜とファイル名から形を読む。
    #:   欄の無い古い楽譜は `song`（既定）として読む（変換は要らない）
    form: Form = Field(default="song", description="曲の形（song＝歌もの・acappella＝アカペラ・instrumental＝インスト）")
    tempo: float = Field(gt=0)
    time_signature: str = Field(default="4/4", pattern=r"^\d+/\d+$")
    lyrics_full: str = ""
    lyrics_display: str = Field(default="", description=(
        "**見せる歌詞**（1フレーズ＝1行・漢字/カタカナ/英字あり）。字幕・投稿・歌詞表示で使う。"
        "`lyrics_full` は歌うためのかな表記（を→お・長音は母音）なので読み物にならない。"
        "空＝この曲には表示用が無い（カバーなど、見せる歌詞を書かない曲）。読み手は `lyrics_full` へ落ちること"))
    vocal_notes: list[SongNote]
    extra_voices: list[VoiceTrack] = Field(default_factory=list,
                                           description="メインに重ねる追加声部（空=単声・後方互換）")
    chords: list[ChordSpan] = Field(default_factory=list)
    #: ★**どの形でも同じ欄を持ち、無いものは空**（2026-10-03・F1 本人決定「インストやアカペラでも、データ形式は変えないようにしたい。
    #:   空配列とかで」）。伴奏が無い（アカペラ・楽譜から伴奏を作っていないカバー）＝パートが0本の伴奏（`ArrangementSpec()`・`is_empty`）。
    #:   以前は None で、無いものの表し方が欄ごとに違った（歌の音符は `[]`・伴奏と主旋律の楽器は None）
    arrangement: ArrangementSpec = Field(default_factory=ArrangementSpec)
    #: ★**主旋律を弾く楽器**（2026-10-03・F2・F10②）。インストは必ずある・歌ものは編成が重ねると決めたときだけ・アカペラは無い
    #:   （無い＝中身が空の `LeadPart()`・`is_empty`。F1）
    lead: LeadPart = Field(default_factory=LeadPart)
    origin_abs_sec: float | None = Field(
        default=None,
        description="【時刻の唯一の基準】tick0 の絶対秒（元音源の時間軸での位置）。音声カバーで align が一度だけ確定し、"
                    "伴奏採譜・切り出し・ミックス・再ロードは全てこれを読む（再導出しない）。None=記憶生成等で絶対基準なし。")
    #: ★**声の種類**（編成が決めた `voice` / `style` / `harmony_style`・2026-09-26）。vpr を書くたびに焼き込む。
    #:   それまでは実行中のメモリ（`voice_style.CHOICE`）にしか無く、編成の工程が走らない編集（キーを変える・
    #:   声部を足すだけ）では**声の種類が抜けた vpr**（No Effect）が書き出されていた（『明日への足音』v2・v5）。
    voice_choice: dict[str, str] = Field(default_factory=dict, description=(
        "声の種類（voice=original/soft・style=MIKU Default 等・harmony_style）。"
        "空＝決めていない（編成の工程を通っていない曲＝編曲する前のカバーなど。インストは歌わないので空）"
        "＝既定の声で書き出す（`score_to_vpr.voice_choice_for`）。編曲したカバーは編成が決めた声の種類を写す"))
    #: ★**人が VOCALOID6 で手で直した、楽譜の言葉で書けないもの**（2026-09-26・`docs/plan/manual_edit_import.md`）。
    #:   手で描いたカーブ・発音記号の手直し・歌トラックの音量を、**区間つきで vpr から抜き出したまま**持ち、
    #:   vpr を書くたびに描き戻す（楽譜だけで出力が決まる）。要素は dict:
    #:   `{"kind": "curve", "track", "name", "start", "end", "events": [{"pos","value"}]}` /
    #:   `{"kind": "phoneme", "track", "start", "phoneme"}` / `{"kind": "volume", "track", "events"}`
    hand_edits: list[dict] = Field(default_factory=list, description="人が V6 で手で直した、楽譜に書けないもの")

    def duration_ticks(self) -> int:
        """曲の長さ（tick）。歌・コード・声部・**主旋律の楽器**の終わりのうち最後（2026-10-03・§5-5。インストは歌が0音）。
        伴奏のパートはコードの上（音程）か曲の小節の中（ドラム・`song_audio.song_bars`）にしか鳴らないので、コードの終わりで足りる。"""
        ends = [n.start_tick + n.duration_tick for n in self.vocal_notes]
        ends += [c.start_tick + c.duration_tick for c in self.chords]
        ends += [n.start_tick + n.duration_tick for v in self.extra_voices for n in v.notes]
        ends += [n.start_tick + n.duration_tick for n in self.lead.notes]
        return max(ends) if ends else 0
