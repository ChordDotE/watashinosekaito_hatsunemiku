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

"""**カバー曲を `Song` に読み込む**（カバーとオリジナルを同じ入れ物で扱うための橋渡し）。

## カバーの曲フォルダに何があるか（2026-08-22 実測・最新版で確認）

| ファイル | 中身 |
|---|---|
| `score_v{N}.json` | 旋律＋歌詞。**`chords` は空・`arrangement` は無い** |
| **`accompaniment_v{N}.mid`** | **伴奏を MIDI で再構成したもの**（YourMT3 で多楽器採譜）|
| `accompaniment_v{N}.wav` | 上を SoundFont で鳴らした音 |
| `_CURRENT_v{N}.txt` | いまの版の目印（作業台方式）|

> **最初この実装は「カバーは伴奏を音声のまま使う」という前提で書いたが、誤りだった。**
> 7月5日のバックアップ（MIDI 再構成が入る前）を見ていた。**最新の曲フォルダを見ること。**

## コードは LLM が書く（2026-10-02・組「カバーのコード」・本人了承の J1〜J8）

ここ（`add_sections`）は**区間と調**だけを起こす。カバーのコードは別の段（`song_cover_chords`）で LLM が書き、伴奏 MIDI・原曲の音声からの
機械の推定（`chord_estimate`）は確からしさつきの**材料**として見せるだけにした。以前（〜2026-10-02）はここで伴奏 MIDI から機械だけで
推定したコードを曲のコードにしていた——「LLM を通っていないので、やり直しても改善できない」（本人）・2回目からは自分たちが作った
伴奏から推定し直して原曲から離れていった・外れた推定（『きらきら星』の `D#m`）も黙っていた（設計資料 `docs/plan/song_cover_chords_llm.md`）。
オリジナル曲の尺の変更は工程の記録の区間・調・コードを読む（`song_record_frame`・2026-10-03）。記録が使えない古い曲だけ、
ここの区間（`add_sections`）と伴奏 MIDI から推定したコード（`add_machine_chords`）へ戻る（返事と記録に出す）。

## 前提

| # | 前提 | 崩れると |
|---|---|---|
| 1 | 区間（Aメロ/サビ）は**推定**——原曲の構成とは限らない | アレンジの区間指定が大まかになる |
| 2 | 調は**推定**（旋律の音から LLM が読む）——外れることがある | 伴奏・声部を書かせる C 基準の物差しがずれる（下） |
| 3 | 区間は**小節で切る**: 区間の頭＝その区間の最初のフレーズの最初の音がある小節、終わり＝次の区間の頭の前の小節（最後は曲の終わり） | 2つの区間の頭が同じ小節に来る切り方は型で差し戻す（`cover_structure_type`） |
| 4 | 歌い出しの前の小節（前奏）はどの区間にも入れない（尺の変更が区間を写すときと同じ数え方） | 前奏の小節には区間のコードも伴奏の組み直しも無い（記録に出す）。カバーの楽譜は tick 0＝区間の最初の音なので、ふつうは 0小節 |

## 調の推定と C 基準（2026-10-01・判-C）

カバーもオリジナルと同じく、LLM には高さを **C メジャー／A マイナー基準**で見せ・書かせ、機械がこの曲の調へ移す
（`song_arrange_cover`）。移す量は推定した調（`Song.key_of`）から `song_place.keys_shift` が出す。
★**推定した調は、曲の記録と道具の返事に必ず出す**（`key_fit` の合い具合つき・`Song.source["key"]`）。
見せるもの（コード表・歌の音域・声部の表）と書いたもの（`pitches`・リフ・声部）は同じ量だけ移るので、調が外れても
伴奏とコードの関係はずれない。損なうのは「C で考えられる」利点（C 基準のはずの表に調の外の和音が並ぶ）で、合い具合の数字で気づける。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from accomp_edit import NO_CHORD_MARK, estimate_bar_chords, load_accomp    # noqa: E402（オリジナル曲の尺の変更だけが使う）
from song_design import Section, bar_ticks, beats_per_bar, key_problem, key_shift   # noqa: E402
from song_llm import pick_model                             # noqa: E402
from song_model import Song, StepRecord, StepStatus         # noqa: E402
from song_checks import JsonInt  # noqa: E402（LLM が書く数と真偽の欄の型・2026-10-02 判断5）
from song_schema import SongScore                           # noqa: E402
from song_schema import TICKS_PER_QUARTER as Q              # noqa: E402
from song_table import spell_pitch                          # noqa: E402（LLM に見せる音名は国際式・表と同じ綴り）


# ---- 曲フォルダを読む -----------------------------------------------------------

def current_version(folder: Path) -> int:
    """`_CURRENT_v{N}.txt` から**いまの版**を取る（本番の作業台方式）。"""
    for p in folder.glob("_CURRENT_v*.txt"):
        m = re.search(r"_CURRENT_v(\d+)\.txt$", p.name)
        if m:
            return int(m.group(1))
    vs = [int(m.group(1)) for p in folder.glob("score_v*.json")
          if (m := re.search(r"score_v(\d+)\.json$", p.name))]
    return max(vs) if vs else 1


def find_parts(folder: Path, version: int | None = None) -> tuple[Path, Path | None, int]:
    """曲フォルダ → (score のパス, 伴奏MIDI のパス|None, 版番号)。

    **版を揃える**のが要点——score_v2 に accompaniment_v1 を合わせると、
    別の版の伴奏でコードを取ってしまう。

    ★**版を指定しなければ、作業台（固定名 `score.json`・`accompaniment.mid`）があればそれを読む**（2026-10-01・別件①）。
      作業台は「いまの曲」（`session.begin_session` が現在版を広げ、このセッションの手直しが載っている）。以前は現在版
      `score_v{N}` を先に読んでいたので、①セッションを閉じて手直し（移調など）が版に入ったあとにアレンジし直すと、台帳の手直しを
      もう一度当てて**移調が2回かかった**（+2 が +4・『きらきら星』の写しで再現）②同じセッションの中で台帳に残らない直し
      （尺の変更）の後に読むと、その直しが無い版から作り直していた。いまは「いまの曲」を読み、作り直しで消えるもの（声部）だけを
      当て直す（`song_edits.cover_reapply`）。版の番号は現在版（作業台は現在版を広げたもの）。
    """
    folder = Path(folder)
    if version is None and (folder / "score.json").exists():
        mid = folder / "accompaniment.mid"
        return folder / "score.json", (mid if mid.exists() else None), current_version(folder)
    v = version or current_version(folder)
    score = folder / f"score_v{v}.json"
    if not score.exists():
        score = folder / "score.json"                      # 作業中（固定名）
    mid = folder / f"accompaniment_v{v}.mid"
    if not mid.exists():
        mid = folder / "accompaniment.mid"
    return score, (mid if mid.exists() else None), v


# ---- 区間と調の推定（コードは別の段 `song_cover_chords` が LLM に書かせる）--------------------

def section_bars_range() -> tuple[int, int | None]:
    """設計の区間の小節数の範囲（`song_design.Section.bars` の型の `ge`・`le`）。カバーの区間も同じ型で持つので、
    構成推定の検査と型の説明はここから引く（手で書いた数を2か所に持たない・2026-10-03）。
    ★上限は `None`＝無し（2026-10-03・`song_viewpoints_review.md` §2 c2）。以前は 1〜32 で、原曲に 33 小節以上の区間があると
      構成推定が「chorus_a / chorus_b のように分けろ」と差し戻し、直らなければ止まった（読み手は上限を持たない）。"""
    lo, hi = 1, None
    for m in Section.model_fields["bars"].metadata:
        lo = getattr(m, "ge", None) if getattr(m, "ge", None) is not None else lo
        hi = getattr(m, "le", None) if getattr(m, "le", None) is not None else hi
    return int(lo), (int(hi) if hi is not None else None)


def _bars_range_text() -> str:
    lo, hi = section_bars_range()
    return f"{lo}〜{hi}小節" if hi is not None else f"{lo}小節以上"


class CoverSection(BaseModel):
    kind: str = Field(description=(
        "区間の**識別子**（verse / pre / chorus / chorus_a / chorus_b1 など。**自由**）。\n"
        "★**短い英数字だけにする。説明・かっこ書き・日本語を入れない**"
        "（`chorus_a (サビ前半)` のように書くと、下流の工程が名前で照合できず"
        "**ハモリや転調が黙って外れる**。2026-08-22 の実害）。説明は reason に書く"))
    from_phrase: JsonInt = Field(ge=1, description=(
        "この区間が始まるフレーズ番号（1始まり）。最初の区間は 1、後の区間ほど大きく（同じ番号・逆順は書けない）"))
    # ★2026-10-04（組N2）: 既定 3 をやめて必ず書かせる
    intensity: JsonInt = Field(ge=1, le=5, description="盛り上がり（**区間ごとに必ず書く**。1=静か 5=熱い。ふつうは 3）")

    # ★**区間名は設計の区間名と同じ文法**（2026-10-03・再レビュー 記P・仕様書 §15・§15-3）: 空は書けない・前後の空白は取り除いて持つ
    #   （区間名は「日本語を書く欄と、設計で決める名前」の側＝Python の `strip()`・§0-6）。以前は空・前後の空白つきの名前が型を通り、
    #   後の工程（伴奏・ハモリ）が空白つきの綴りを書かされた。取り除き方は設計の `Section._kind` と同じ `section_id`（1か所）
    @field_validator("kind")
    @classmethod
    def _kind_grammar(cls, v: str) -> str:
        from song_checks import section_id                  # noqa: E402
        k = section_id(v)
        if not k:
            raise ValueError("区間名 kind が空——区間には短い英数字の名前を付ける（verse・chorus など。空白だけも空と同じ）")
        return k


class CoverStructure(StepStatus):
    reason: str = Field(description="どう読んだか（曲を知っていればその知識も使ってよい）")
    key: str = Field(description=("推定した調（C / Am / F# / Bb など）。旋律の音使いから読む。"
                                  "長調は何も付けない・短調は `m` だけ（`F major`・`D minor` とは書かない・2026-10-02）。"
                                  "変化記号は `#` と `b` を1つだけ（記号の ♯ ♭ と、2つ重ねた Ebb・C## は書けない）。"
                                  "途中に空白を入れない（`A m` ではなく `Am`）"))
    sections: list[CoverSection] = Field(description=(
        "区間の切れ目（曲頭から順に・1つ以上）。最初の区間の `from_phrase` は 1、`from_phrase` は増える順で、"
        "区間名 `kind` は1つずつ。1区間は {0}（区間の頭から次の区間の頭の前まで。最後の区間は曲の終わりまで）"
        .format(_bars_range_text())))

    # ★**読めない調は型で差し戻す**（2026-10-01・判-C）。調から移調量を出して伴奏・声部の高さを移すので、読めない調のまま
    #   進むと移調量が出せない（以前は検証が無く、読めない綴りは後の `key_shift` で落ちた）
    @field_validator("key")
    @classmethod
    def _key_readable(cls, v: str) -> str:
        if why := key_problem(v):
            raise ValueError(why)
        return v


def cover_structure_type(n_phrases: int, phrase_bars: list[int] | None = None,
                         n_bars: int | None = None) -> type[CoverStructure]:
    """**フレーズの数を知っている構成推定の型**（2026-10-02・洗い出し H10。前例は `align_cover.corrected_type`）。
    区間が1つ以上・最初の `from_phrase` が 1・`from_phrase` が増える順（同じ番号・逆順は書けない）・最後の区間がフレーズの数以内・
    区間名が1つずつ、でなければ型で差し戻す（直らなければ `add_sections` が `StepUnfixed`「カバーの構成推定」で止める）。
    `phrase_bars`（フレーズごとの最初の音の小節・0始まり）を渡すと、**2つの区間の頭が同じ小節に来る切り方**も差し戻す
    （区間は小節で切る——同じ小節に頭が2つあると、前の区間が0小節になる。2026-10-02・組「カバーのコード」）。
    さらに `n_bars`（曲の小節の数）も渡すと、**区間の小節数が設計の区間の型の範囲（1 以上・`section_bars_range`。上限は 2026-10-03 に外した）に収まらない切り方**も
    差し戻す（2026-10-03・再レビュー）。以前は型を通ったあと `Section(bars=…)` の型の例外で落ち、道具は別の理由を返していた。

    ★以前は `add_chords` が黙って読み替えていた: 逆順・重なりは1小節に丸め、先頭が1でなければ前の音を先頭の区間に入れ、
    空なら全部 `verse`、フレーズの数を超える番号は落ちる（`IndexError`）。区間名が重なると区間の表（`bars_of`・`chords_of`）は後が勝った。"""
    from pydantic import model_validator

    class _CoverStructureForPhrases(CoverStructure):
        @model_validator(mode="after")
        def _sections_follow_phrases(self):
            secs = self.sections
            if not secs:
                raise ValueError(f"sections が空——区間を1つ以上書く（最初の区間の from_phrase は 1・フレーズは {n_phrases}個）")
            bad: list[str] = []
            if secs[0].from_phrase != 1:
                bad.append(f"最初の区間『{secs[0].kind}』の from_phrase が {secs[0].from_phrase}——最初は 1（曲頭のフレーズから）")
            for a, b in zip(secs, secs[1:]):
                if b.from_phrase <= a.from_phrase:
                    bad.append(f"区間『{b.kind}』の from_phrase {b.from_phrase} が前の区間『{a.kind}』（{a.from_phrase}）以下——増える順に書く")
            over = [s for s in secs if s.from_phrase > n_phrases]
            bad += [f"区間『{s.kind}』の from_phrase {s.from_phrase} がフレーズの数（{n_phrases}）を超える" for s in over]
            seen: dict[str, int] = {}
            for s in secs:
                seen[s.kind] = seen.get(s.kind, 0) + 1
            bad += [f"区間名『{k}』が{n}回ある（1つずつ・2回目は chorus_b のように別の名前に）" for k, n in seen.items() if n > 1]
            if phrase_bars and not bad:
                for a, b in zip(secs, secs[1:]):
                    ba, bb = phrase_bars[a.from_phrase - 1], phrase_bars[b.from_phrase - 1]
                    if bb <= ba:
                        bad.append(f"区間『{b.kind}』の頭（フレーズ{b.from_phrase}・{bb + 1}小節目）が前の区間『{a.kind}』の頭"
                                   f"（フレーズ{a.from_phrase}・{ba + 1}小節目）と同じ小節——区間は小節で切るので、"
                                   "頭が別の小節に来るフレーズで区切る")
            if phrase_bars and n_bars and not bad:
                lo, hi = section_bars_range()
                heads = [phrase_bars[s.from_phrase - 1] for s in secs]
                for s, h, e in zip(secs, heads, heads[1:] + [n_bars]):
                    if e - h < lo or (hi is not None and e - h > hi):
                        bad.append(f"区間『{s.kind}』が {e - h}小節（{h + 1}〜{e}小節目）——1区間は {_bars_range_text()}"
                                   + ("。長い区間はフレーズの切れ目で分ける（chorus_a / chorus_b のように）"
                                      if hi is not None and e - h > hi else ""))
            if bad:
                raise ValueError("sections: " + "／".join(bad[:8]))
            return self

    _CoverStructureForPhrases.__name__ = "CoverStructure"   # LLM に見える型の名前は変えない
    return _CoverStructureForPhrases


_structure = Agent(
    pick_model("COVER_STRUCT", "flash"), output_type=CoverStructure,
    system_prompt=(
        "あなたは編曲者です。**カバー曲の主旋律**を読んで、**調と区間の切れ目**を判断します。\n"
        "\n"
        "**区間の切れ目**は、休符の長さ・音域の変化・歌詞の内容から見ます。\n"
        "音域が上がって言葉が言い切りになる所がサビであることが多い。\n"
        "**知っている曲なら、その知識を使って構いません**（原曲の構成に寄せる）。\n"
        "\n"
        "★**区間は細かく分けてください（8小節を超えたら分割を考える）。**\n"
        "  曲の一部を切り出したカバー（サビだけ・1番だけ）では、**全体が1区間になりがち**です。\n"
        "  そうなると下流が「どこに何をするか」を選べなくなります——\n"
        "  実際、31秒のサビを1区間にしたら**曲全体にハモリが乗り続けて単調になった**（2026-08-22）。\n"
        "  同じサビでも `chorus_a` / `chorus_b`、`chorus_1回し目` / `chorus_2回し目` のように、\n"
        "  **歌詞の切れ目・旋律の反復・音域の山**で分けてください。\n"
        "\n"
        "★**区間は小節で切ります**: 区間の頭は、その区間の最初のフレーズが始まる小節です。"
        "頭が前の区間の頭と同じ小節に来るフレーズでは区切れません。\n"
        "コード進行は**別の工程（コード）が書く**ので、ここでは判断しません。\n"
        "調（キー）だけ、旋律の音使いから推定してください。"
    ),
)


def _split_by_gap(score: SongScore, gap_ticks: int) -> list[list[int]]:
    out, cur, prev = [], [], None
    for i, n in enumerate(score.vocal_notes):
        if prev is not None and n.start_tick - prev >= gap_ticks and cur:
            out.append(cur)
            cur = []
        cur.append(i)
        prev = n.start_tick + n.duration_tick
    if cur:
        out.append(cur)
    return out


def phrases_of(score: SongScore, gap_ticks: int = 240, *, min_phrases: int = 4) -> list[list[int]]:
    """休符で区切って**フレーズ**にする（音符の index の並び）。

    **カバーは実演奏由来なので休符が短い。** 0.5拍（240tick）で切ると
    31秒の曲が「1フレーズ」になり、区間を分けられなくなる（2026-08-22 の実害:
    区間が1つ＝曲全体がサビ扱いになり、**ハモリが100%の音に乗った**）。

    そこで**閾値を段階的に下げ**、それでも足りなければ**小節で機械的に切る**。
    """
    for g in (gap_ticks, 120, 60, 30):
        out = _split_by_gap(score, g)
        if len(out) >= min_phrases:
            return out
    # 休符では切れない（レガートな旋律）→ 2小節ごとに区切る
    bar = bar_ticks(score.time_signature or "4/4") * 2
    out, cur, at = [], [], bar
    for i, n in enumerate(score.vocal_notes):
        if n.start_tick >= at and cur:
            out.append(cur)
            cur = []
            at = (n.start_tick // bar + 1) * bar
        cur.append(i)
    if cur:
        out.append(cur)
    return out or [list(range(len(score.vocal_notes)))]


def load_cover(folder: Path, *, version: int | None = None, verbose: bool = True) -> Song:
    """**カバーの曲フォルダ**から Song を作る（score と伴奏 MIDI を版を揃えて読む）。"""
    folder = Path(folder)
    score_p, mid_p, v = find_parts(folder, version)
    score = SongScore.model_validate_json(score_p.read_text(encoding="utf-8"))

    song = Song(instruction=f"カバー『{score.title}』", origin="cover", version=1)
    song.score = score
    song.source = {"folder": str(folder), "score": score_p.name,
                   "accompaniment_mid": mid_p.name if mid_p else None, "cover_version": v}
    song.record(StepRecord(step="melody",
                           reason=f"原曲から採譜した旋律をそのまま使う（{len(score.vocal_notes)}音）"))
    song.record(StepRecord(step="lyrics", reason=f"原曲の歌詞（{len(score.lyrics_full)}字）"))

    if verbose:
        ph = phrases_of(score)
        dur = max(n.start_tick + n.duration_tick for n in score.vocal_notes) / Q * 60 / score.tempo
        print(f"  [カバー読込] 『{score.title}』BPM{score.tempo:.0f} v{v} "
              f"{len(score.vocal_notes)}音 {len(ph)}フレーズ {dur:.0f}秒")
        print(f"      score={score_p.name} / 伴奏MIDI={mid_p.name if mid_p else '**無し**'}")
    return song


#: 区間と調の LLM 呼び出しのラベル（`song_llm.run_sync`＝費用・依頼文の記録・型の差し戻しが直らなければこの名前で止める）
STRUCTURE_LABEL = "カバーの構成推定"


def cover_bars(score: SongScore) -> tuple[str, int, int]:
    """楽譜 → (拍子, 1小節の tick, 曲の小節の数＝最後の歌の音の終わりまで)。"""
    ts = score.time_signature or "4/4"
    bar = bar_ticks(ts)
    end = max(n.start_tick + n.duration_tick for n in score.vocal_notes)
    return ts, bar, max(1, -(-end // bar))


def add_sections(song: Song, *, verbose: bool = True) -> StepRecord:
    """**区間と調を LLM に推定させ、`Song` に載せる**（コードは載せない＝`song_cover_chords` の段）。

    旋律をフレーズに分けて見せ、区間の切れ目（どのフレーズから）と調を書かせる。呼び出しは共通の入口 `song_llm.run_sync`
    （ラベル「カバーの構成推定」）——費用（`song_usage`）・依頼文の記録に乗り、型の差し戻しが直らなければ `StepUnfixed` で止まる
    （2026-10-02 まで入口を通らず、費用にも依頼文の記録にも残らなかった）。
    """
    import song_llm
    from song_usage import record as record_usage
    score = song.score
    ts, bar, n_bars = cover_bars(score)
    ph = phrases_of(score)
    starts = [score.vocal_notes[idxs[0]].start_tick // bar for idxs in ph]   # フレーズの最初の音の小節（0始まり）
    listing = []
    for i, idxs in enumerate(ph, 1):
        ns = [score.vocal_notes[j] for j in idxs]
        at = ns[0].start_tick / bar + 1
        listing.append(f"  フレーズ{i}（{at:.1f}小節目〜 {len(ns)}音）「{''.join(n.lyric for n in ns)}」"
                       f" 音: {' '.join(spell_pitch(n.key) for n in ns[:12])}")
    # ★**区間の切れ目の形式は型で差し戻す**（2026-10-02・洗い出し H10・`cover_structure_type`）。直らなければ止める
    r = song_llm.run_sync(
        _structure, f"曲名『{score.title}』BPM{score.tempo:.0f} 拍子{ts} 全{n_bars}小節\n\n"
        f"【旋律】（フレーズは {len(ph)}個）\n" + "\n".join(listing) + "\n\n調と区間の切れ目を判断してください。",
        label="カバーの構成推定", output_type=cover_structure_type(len(ph), starts, n_bars))   # 区間の小節数の範囲も型で（2026-10-03）
    record_usage(_structure, r, STRUCTURE_LABEL)
    out = r.output

    # ★**区間は小節で切る**（2026-10-02・組「カバーのコード」・洗い出し H10 の残り）。区間の頭＝最初のフレーズの最初の音の小節、
    #   終わり＝次の区間の頭の前の小節（最後の区間は曲の終わりまで）。以前は区間の小節数を「最初の音から最後の音まで」で数えて
    #   曲頭から並べたので、前奏の小節を数えず・小節をまたぐフレーズを両方の区間で数え、合計が曲の小節数を超えた分を曲の終わりで切っていた
    from song_fix import AssemblyError
    heads = [starts[cs.from_phrase - 1] for cs in out.sections]
    if heads != sorted(set(heads)):                        # 型（`cover_structure_type`）が差し戻すので来ない。来たら不具合
        raise AssemblyError("カバーの構成推定に、型を通ったはずの同じ小節に頭が2つある区間の切れ目が来た: "
                            + " / ".join(f"{s.kind}={s.from_phrase}" for s in out.sections))
    ends = heads[1:] + [n_bars]
    # 音域の狙い（pitch_center）はカバーでは誰も決めない（原曲の旋律をそのまま使う）。設計の型の必須の欄なので、以前の型の既定 mid を
    #   機械が明示する（2026-10-04 組N2・読み手はオリジナルの旋律の工程だけ）
    sections = [Section(kind=cs.kind, has_vocal=True, intensity=cs.intensity, bars=e - h, pitch_center="mid")
                for cs, h, e in zip(out.sections, heads, ends)]
    place_sections(song, out.key, sections, heads[0])
    fit = key_fit(score.vocal_notes, [], out.key)          # 旋律だけの合い具合（コードの段が原曲の伴奏の推定も数えて書き直す）
    song.source = {**(song.source or {}), "key": fit}
    lead = (f"歌い出しの前の {heads[0]}小節（前奏）はどの区間にも入れていない（伴奏を組み直すとそこは鳴らない）"
            if heads[0] else "")
    rec = StepRecord(step="design", reason=f"区間と調を旋律から推定: {out.reason}", status=out.status, trouble=out.trouble,
                     warnings=[lead] if lead else [])
    song.record(rec)
    if verbose:
        print(f"  [区間] 調={out.key} / " + " → ".join(f"{s.kind}({s.bars})" for s in sections))
    return rec


def place_sections(song: Song, key: str, sections: list[Section], lead: int) -> None:
    """区間（小節数）と調を `Song` に載せる（新しく推定したときも、曲の記録から読んだときも同じここ）。

    `lead` は歌い出しの前の小節の数（どの区間にも入れない）。音符の区間（`sec_of_note`）は、音の始まる小節の区間——
    コードを小節で書かせるので、区間も小節で数える（以前はフレーズで数えていた）。
    """
    score = song.score
    ts, bar, _n = cover_bars(score)
    bars_of, at = {}, lead
    for s in sections:
        bars_of[s.kind] = (at + 1, at + s.bars)
        at += s.bars
    song.bars_of = bars_of
    song.sec_of_note = [next((k for k, (b0, b1) in bars_of.items() if b0 <= n.start_tick // bar + 1 <= b1), "")
                        for n in score.vocal_notes]
    if not all(song.sec_of_note):
        from song_fix import AssemblyError
        raise AssemblyError("カバーの区間に、どの区間にも入らない音がある（区間の小節が曲の音を覆っていない）")
    song.key_of = {s.kind: key for s in sections}
    song.design = {"key": key, "bpm": int(score.tempo), "time_signature": ts, "lead_bars": lead,
                   "sections": [s.model_dump() for s in sections]}


def cover_timeline(song: Song):
    """カバーの区間 → 時間軸（`SectionTimeline`。歌い出しの前の小節は曲頭の空き小節 `LEAD_IN`）。
    コードを置く組み立て（`song_place.place_chords`）と、区間ごとの音量の測定（`check_mix_quality`）が使う。
    ★曲の記録には残さない（`Song.timeline` に置かない）——記録に時間軸がある曲は、尺の検査（`_length_findings`）が
    前奏・後奏を区別して測る新しい曲の扱いになるため。"""
    from song_timeline import LEAD_IN, SectionSpan, SectionTimeline
    d = song.design or {}
    _ts, bar, _n = cover_bars(song.score)
    spans, at = [], 1
    if int(d.get("lead_bars") or 0) > 0:
        spans.append(SectionSpan(kind=LEAD_IN, bar_from=1, bar_to=int(d["lead_bars"]), has_vocal=False, intensity=1))
        at += int(d["lead_bars"])
    for s in d.get("sections") or []:
        spans.append(SectionSpan(kind=s["kind"], bar_from=at, bar_to=at + int(s["bars"]) - 1,
                                 has_vocal=bool(s["has_vocal"]), intensity=int(s["intensity"])))   # 既定で埋めない（2026-10-04 組N2）
        at += int(s["bars"])
    tl = SectionTimeline(bar_ticks=bar, spans=spans)
    tl.validate()
    return tl


def add_machine_chords(song: Song, *, verbose: bool = True) -> StepRecord:
    """**工程の記録が使えないオリジナル曲の尺の変更だけ**（`song_record_frame.from_record` が理由を返したとき・2026-10-03）:
    いまの伴奏 MIDI から機械でコードを推定し、区間（`add_sections` のあと）に写す。

    カバーはここを使わない（コードは LLM が書く・`song_cover_chords`）。オリジナル曲の尺の変更は工程の記録を使わず
    いまの楽譜と伴奏から起こす（`resize_song` の返事に出る）。取れない小節は `N.C.`（当て推量で埋めない・記録に出す）。
    """
    score = song.score
    ts, _bar, n_bars = cover_bars(score)
    src = song.source or {}
    mid_name = src.get("accompaniment_mid")
    chords: list[str] = []
    how = "伴奏MIDIが無い"
    if mid_name:
        try:
            tracks = load_accomp(Path(src["folder"]) / mid_name)
            chords = estimate_bar_chords(tracks, score.tempo, 0, n_bars, beats_per_bar=beats_per_bar(ts))
            how = f"伴奏MIDI（{mid_name}・{len(tracks)}トラック）から機械で推定"
        except Exception as e:
            how = f"伴奏MIDIの解析に失敗（{type(e).__name__}）"
    missing = not chords or all(c == NO_CHORD_MARK for c in chords)
    chords = chords or [NO_CHORD_MARK] * n_bars
    song.chords_of = {k: chords[b0 - 1:b1] for k, (b0, b1) in song.bars_of.items()}
    rec = StepRecord(step="chords", reason=how, warnings=[f"コードが取れていない（{how}）"] if missing else [])
    song.record(rec)
    if missing:
        from song_fix import record_change
        record_change(song, "chords", f"全{n_bars}小節のコードが取れず、和音なし（{NO_CHORD_MARK}）のままにした"
                                      "（当て推量で埋めていない）", how)
    if verbose:
        print(f"  [コード] {how}: {' '.join(chords[:16])}")
    return rec


# ---- 推定した調の合い具合（2026-10-01・判-C）----------------------------------------------

#: 長調の主音の綴り（相対長調で数える。移調量はこれで決まる＝`song_design.key_shift`）
_MAJOR_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_SCALE = (0, 2, 4, 5, 7, 9, 11)
#: 推定した調より、ほかの調のほうがこれだけ（割合の差）よく合えば警告に出す
FIT_MARGIN = 0.05


def key_fit(notes: list, chords: list[str], key: str) -> dict:
    """推定した調が、原曲の音（旋律＋伴奏 MIDI から取ったコード）にどれだけ合うか。**数えるだけで、調は変えない**。

    数えるのは「その調の音階（短調は相対長調の7音）に入る割合」——旋律は音の長さで重み、コードは小節ごとの構成音。
    移調量は相対長調で決まる（`key_shift`）ので、12通りの長調と比べ、いちばん合う調も出す。
    旋律だけでは決まらないことがある（『きらきら星』の写しは D E F# G A B の6音で、D と G のどちらにも 100%）——コードの
    構成音（G の音階に無い C# を A の和音が持つ）も一緒に数える。

    戻り値: `key`・`shift`（C 基準から移す半音）・`melody`・`chords`（割合・数えるものが無ければ None）・`best`（いちばん合う長調）・
    `ties`（同じくらい合う調）・`note`（記録と返事に出す一文）・`warning`（ほかの調のほうが明らかに合うときだけ）。
    """
    from song_audio import chord_notes                     # 和音の中身は鳴らす側と同じ読み方
    mel = [0.0] * 12
    for n in notes or []:
        mel[int(n.key) % 12] += max(1, int(n.duration_tick))
    cho = [0.0] * 12
    for c in chords or []:
        for k in chord_notes(c):
            cho[k % 12] += 1.0

    def share(w: list[float], tonic: int) -> float | None:
        tot = sum(w)
        return None if tot <= 0 else sum(w[(tonic + s) % 12] for s in _SCALE) / tot

    def score(t: int) -> float:
        got = [x for x in (share(mel, t), share(cho, t)) if x is not None]
        return sum(got) / len(got) if got else 0.0

    def pct(x: float | None) -> str:
        return "—" if x is None else f"{x:.0%}"

    sh = key_shift(key)
    m, c = share(mel, sh), share(cho, sh)
    best = max(range(12), key=lambda t: (round(score(t), 6), t == sh))
    ties = [t for t in range(12) if t != sh and abs(score(t) - score(sh)) < 1e-6]
    note = (f"調は {key}（原曲の旋律から推定・C 基準で書いた高さを {sh:+d}半音移して鳴らす）。"
            f"この調の音階に入る割合: 旋律の音 {pct(m)}・コードの音 {pct(c)}")
    if ties:
        note += "（同じくらい合う調: " + "・".join(_MAJOR_NAMES[t] for t in ties) + "）"
    warning = ""
    if best != sh and score(best) - score(sh) < FIT_MARGIN:   # わずかな差（『きらきら星』を G と読むと D との差は 2.5 ポイント）も見えるように
        note += (f"（いちばん合うのは {_MAJOR_NAMES[best]}: 旋律の音 {pct(share(mel, best))}・"
                 f"コードの音 {pct(share(cho, best))}）")
    elif best != sh:
        warning = (f"推定した調 {key} より {_MAJOR_NAMES[best]}（長調として。旋律の音 {pct(share(mel, best))}・"
                   f"コードの音 {pct(share(cho, best))}）のほうが原曲の音に合う——調が外れていると、伴奏・声部を書かせる"
                   "C 基準の物差しがずれる（鳴る高さどうしの関係はずれない）")
        note += f"。★{warning}"
    return {"key": key, "shift": sh, "melody": m, "chords": c, "best": _MAJOR_NAMES[best],
            "ties": [_MAJOR_NAMES[t] for t in ties], "note": note, "warning": warning}


def key_note(song: Song) -> str:
    """推定した調の一文（`add_sections`・コードの段が `Song.source["key"]` に残したもの）。無ければ空。道具の返事・記録に使う。"""
    return ((getattr(song, "source", None) or {}).get("key") or {}).get("note", "")


def chords_note(song: Song) -> str:
    """カバーのコードの一文（`song_cover_chords` が `Song.source["chords_note"]` に残したもの）。無ければ空。

    道具の返事に出す: LLM が書いたか（材料・機械の推定と違う小節の数・取れなかった小節の数・Web のコード譜）／
    曲の記録から読んだか。マスターが聴く前に、コードがどこから来たかを知れるように。"""
    return (getattr(song, "source", None) or {}).get("chords_note", "") or ""
