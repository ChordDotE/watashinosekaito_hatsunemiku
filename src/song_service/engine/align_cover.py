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

"""MIDI忠実コピーのカバー: 指示区間を抽出し、主旋律ch→ミク歌声、それ以外→元MIDI伴奏。

検証済みの参照MIDI（fetch_reference）と主旋律ch・区間指示を受け、
- 歌声: 主旋律chの区間の音符（音高・タイミングそのまま＝MIDI準拠）にミクの歌詞をLLM整列で乗せる
        （align_to_reference_jp で歌詞→音符を賢く割り当て：メリスマ/区切り/装飾音スキップを判断）
- 伴奏: 元MIDIの主旋律ch以外をその区間で切り出して accompaniment.mid にし、SoundFont（sf2_render）で鳴らす
両方が同じMIDIのタイミングで揃う。FluidSynth／SoundFont が無ければ伴奏は作らずに止まる（代わりの音で鳴らさない）。
"""
from __future__ import annotations

import json
import re
import statistics
import tempfile
from pathlib import Path

import mido
import numpy as np

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, BinaryContent

import song_llm
from align_reference import CELL_RULE, align_full_lyrics_jp, assign_or_stop, note_assign_findings, split_cell
from align_reference import LYRIC_LINES_RULE, lyric_lines_text   # 歌詞の行の切れ目を材料に見せる（2026-10-03）
from lyric_slots import kana_text_chars                     # 字を数えるときは行の切れ目（定義された区切り）を除く（2026-10-03）
from render_base import load_wav, write_wav
from song_tuning import apply_by_gap                        # 調声（カバーにも当てる・2026-09-01）
from midi_to_reference import extract_melody
from reference_fetch import SEC_TOL, _maou_lyrics_page, fetch_lyrics, web_material_note
# ★**LLM に見せる・書かせる高さは国際式の音名**（`C4`＝60・2026-10-01）。綴りは表と同じ `song_table.spell_pitch`、
#   読み手は `song_llm.pitch_name_to_midi`。以前は階名（`ド4`・`midi_to_reference.midi_to_solfa`）と MIDI 番号で見せていた。
#   参照の JSON に保存された `solfa` 欄（保存形式）は変えない——LLM へは MIDI 番号から綴り直して見せる
from song_table import spell_pitch


from song_lyrics import KANA_TEXT_RULE                      # noqa: E402（カバーの歌詞のかなの書き方・字の範囲は詞と同じ1つ）

_KANA_STEP = "カバーの歌詞のかな"                             # 型の差し戻しが直らなかったら、この名前で止める（`_kana_or_stop`）


def _wav_bytes(clip) -> bytes:
    """音の切り出しを wav のバイト列にする（LLM に聴かせる）。**一時ファイルの名前はそのつど別にする**（2026-10-03）。
    ★以前は `_miku_align_<区間の秒の hash>.wav` の決まった名前を書いて読んで消していたので、同じ区間を同時に扱う2つのプロセス
      （並行のセッション・テスト）が互いのファイルを消し、読む前に消えると例外→音声の経路を使わずに進んだ（テストが時々落ちた）。"""
    fd, name = tempfile.mkstemp(prefix="_miku_clip_", suffix=".wav")
    import os
    os.close(fd)
    tmp = Path(name)
    try:
        write_wav(tmp, clip)
        return tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)


def _kana_written(v: str) -> str:
    """**カバーの歌詞のかなは、かな以外を型で差し戻す**（2026-10-02 本人決定・案B）。字の範囲は詞の1フレーズと同じ
    （`song_lyrics.kana_text_problems`・仕様書 §2）。空白・改行・句読点・譜割りの印も書けない（どこも区切りとして読まない）。
    ★以前は読み手（`align_reference.align_full_lyrics_jp`・強制整列・SOFA・音声での割り当て）が空白・改行・`、。「」` を黙って落とし、
      それ以外の字（`！`・漢字・カタカナ）はそのまま音符に当てて VOCALOID の歌詞に入れていた。
    Web の歌詞・聴き取りの結果そのもの（外から来た材料）は変えない——差し戻すのは LLM が書いたかなだけ。"""
    from song_lyrics import kana_text_problems              # noqa: E402
    bad = kana_text_problems(v)
    if bad:
        raise ValueError("lyrics_kana: " + "／".join(bad))
    return v.strip()


def _kana_or_stop(agent, prompt):
    """歌詞のかなを書く LLM を呼ぶ。型の差し戻しが直らなければ `StepUnfixed`「カバーの歌詞のかな」で止める
    （`fetch_section_lyrics` が「歌詞はまだ取っていない」と返す。「カバーの歌詞の割り当て」と同じ止め方）。"""
    try:
        from song_usage import run_counted                  # noqa: E402（費用に入れる・2026-10-08）
        return run_counted(agent, _KANA_STEP, prompt).output
    except Exception as e:
        if song_llm.output_unfixed(e):
            raise song_llm.to_step_unfixed(e, _KANA_STEP, step=_KANA_STEP) from e
        raise


class _Lyrics(BaseModel):
    lyrics_kana: str = Field(description="この区間で歌う実際の歌詞（日本語版・全文・順番どおり）。" + KANA_TEXT_RULE)

    @field_validator("lyrics_kana")
    @classmethod
    def _kana(cls, v: str) -> str:
        return _kana_written(v)


_lyrics_agent: "Agent[None, _Lyrics]" = Agent(
    song_llm.pick_model("MELODY", "flash"), output_type=_Lyrics,
    system_prompt=(
        "あなたは歌詞に詳しい音楽家です。指定された曲・区間で実際に歌われる日本語歌詞をかなで全文返します。\n"
        "VOCALOID向けのかな: " + KANA_TEXT_RULE + "。小さい母音・ん・っは可。"
        "区間が繰り返し（例:コーラス2回）なら繰り返しぶんも全部書く。"
    ),
)


class _Corrected(BaseModel):
    reasoning: str = Field(description="音声を聴いた所見：音高/オクターブ誤りの指摘")
    # ★2026-10-01 まで MIDI 番号（`list[int]`）で書かせていた。いまは国際式の音名（見せる高さと同じ書き方）
    pitches: list[str] = Field(description="各音符の補正後の高さを時間順に（与えた音符数と同じ並び・1対1）。"
                                           "国際式の音名（`C4`＝中央のド・`E5`・`F#4`・`Bb3`。変化記号は `#` と `b`、"
                                           "オクターブの数字は必ず付ける）。"
                                           "明らかな音高/オクターブ誤りだけ直し、タイミングや音数は変えない"
                                           "（歌の高さ `C2`〜`C7` の中で書く）")

    @field_validator("pitches")
    @classmethod
    def _readable_pitches(cls, v: list[str]) -> list[str]:
        """**読めない音名・歌の高さの外は型で差し戻す**（2026-10-02・字コードの組）。以前は型の検証が無く、
        読めない音名と `C2`〜`C7`（36〜96）の外の高さを、補正の列から黙って除いていた（記録に残すだけ）。
        読み手は音名の記法の型（`song_llm.PITCH`）1つ。範囲は声部の検査と同じ"""
        bad = []
        for i, p in enumerate(v):
            if why := song_llm.PITCH.problems(p):
                bad.append(f"{i}番目: {why[0]}")
            elif not 36 <= song_llm.PITCH.read(p) <= 96:
                bad.append(f"{i}番目: 音名『{p}』は歌の高さ（`C2`〜`C7`）の外")
        if bad:
            raise ValueError("pitches: " + "／".join(bad[:8]))
        return [p.strip() for p in v]


def corrected_type(n: int) -> type[_Corrected]:
    """**頼んだ音の数を知っている音高補正の型**（2026-10-02 本人「おすすめで」・前例は `song_llm.added_voice_type`）。
    返した高さの数が頼んだ数 `n` と違えば型で差し戻す（pydantic_ai が LLM に書き直させ、直らなければ呼ぶ側が
    `StepUnfixed`「カバーの音高補正」で止める）。

    ★以前は数が違うと、機械が採譜の高さと返った高さを DTW で対応づけて各かなへ割り振っていた（返事には出していたが、
    LLM の出力を機械が書き換えていた＝規約 song_service_editing「機械は LLM の出力を黙って書き換えない」）。"""
    from pydantic import model_validator

    class _CorrectedForCount(_Corrected):
        @model_validator(mode="after")
        def _same_count(self):
            if len(self.pitches) != n:
                raise ValueError(f"pitches: 高さが {len(self.pitches)}個ある——頼んだ音は {n}個。"
                                 f"1音に1つ、同じ{n}個・同じ順で返す（音を足したり省いたりしない）")
            return self

    _CorrectedForCount.__name__ = "_Corrected"              # LLM に見える型の名前は変えない
    return _CorrectedForCount


_audio_correct_agent: "Agent[None, _Corrected]" = Agent(
    song_llm.pick_model("ALIGN", "flash"), output_type=_Corrected,
    system_prompt=(
        "あなたは絶対音感を持つ採譜者です。分離ボーカル音声と、各音符の機械採譜ピッチ（時刻つき）を渡します。\n"
        "**発声タイミングは実測済みなので変えません。** 各音符の『音高だけ』を音声に合わせて微修正してください。\n"
        "・明らかな音高/オクターブ誤りを直す ・大半は元のままで良い ・音符の数と順番は変えない（1対1で返す）。"
    ),
)


def _forced_align(vocal_wav: str, kana: str, t0: float, t1: float,
                  log: list | None = None) -> list | None:
    """区間ボーカル＋既知かな歌詞を WhisperX 強制アラインメント（venv_transcribe）。

    [{char, start秒, end秒, midi}, ...]（区間頭=0秒基準）。失敗時 None。
    """
    import os
    import subprocess
    here = Path(__file__).parent
    try:
        clip = load_wav(Path(vocal_wav))[int(t0 * SR): int(t1 * SR)]
        if len(clip) < SR:
            return None
        cw = here / f"_fa_clip_{abs(hash((t0, t1)))}.wav"
        tx = here / f"_fa_text_{abs(hash(kana))}.txt"
        write_wav(cw, clip)
        # かな以外は歌詞のかなの型が差し戻す（2026-10-02・案B。以前は空白・`、。「」` を落とした）。
        # 行の切れ目（改行＝定義された区切り・2026-10-03）は字ではないので除いて渡す
        tx.write_text("".join(kana_text_chars(kana)), encoding="utf-8")
        import sys
        py = sys.executable                                  # 統合 venv(py3.13) で実行（音声スタック同居）
        r = subprocess.run([str(py), str(here / "forced_align.py"), str(cw), str(tx)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=600, cwd=str(here.parent),   # service直下=相対パス基準
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        cw.unlink(missing_ok=True)
        tx.unlink(missing_ok=True)
        for line in reversed((r.stdout or "").strip().splitlines()):
            line = line.strip()
            if line.startswith("["):
                chars = json.loads(line)
                if log is not None:
                    log.append(f"[強制アラインメント] {len(chars)}文字を実測整列")
                return chars or None
    except Exception:
        pass
    return None


def _forced_align_en(vocal_wav: str, text: str, t0: float, t1: float) -> list | None:
    """区間ボーカル＋**英語原文**を英語wav2vec2で強制アラインメント → [{word,start,end}]（区間頭=0秒基準）。
    日本語FAは英語を前詰めするが、英語モデル＋正テキストなら英単語の実測時刻が正確に取れる。失敗時 None。"""
    import os
    import subprocess
    import sys
    here = Path(__file__).parent
    try:
        clip = load_wav(Path(vocal_wav))[int(t0 * SR): int(t1 * SR)]
        if len(clip) < SR // 2:
            return None
        cw = here / f"_fa_en_{abs(hash((t0, t1)))}.wav"
        tx = here / f"_fa_entx_{abs(hash(text))}.txt"
        write_wav(cw, clip)
        tx.write_text(text, encoding="utf-8")                  # 英語は空白＝語境界なのでそのまま
        r = subprocess.run([sys.executable, str(here / "forced_align.py"), str(cw), str(tx), "en"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=600, cwd=str(here.parent),   # service直下=相対パス基準
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        cw.unlink(missing_ok=True)
        tx.unlink(missing_ok=True)
        for line in reversed((r.stdout or "").strip().splitlines()):
            line = line.strip()
            if line.startswith("["):
                words = json.loads(line)
                return words or None
    except Exception:
        pass
    return None


def _english_phrases(text: str) -> list[str]:
    """歌詞原文から英語フレーズ（2語以上のASCII連なり）を抽出。en-FA用の正テキスト。"""
    import re
    out, seen = [], set()
    for p in re.findall(r"[A-Za-z][A-Za-z'’ ]*[A-Za-z]", text):
        p = p.strip()
        if len(p.split()) >= 2 and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out


def _transcribe_vad(vocal_wav: str, t0: float, t1: float, log: list | None = None) -> list[dict] | None:
    """分離ボーカル**全体**を faster-whisper(**vad_filter**)で1回文字起こし→キャッシュし、区間[t0,t1]を返す。
    全曲を一括で起こす方が末尾被覆が安定（クリップ起こしは端を取りこぼす）。1曲1回・全区間で再利用。
    vad_filter で伴奏区間の幻覚（クレジット文言の捏造）を防ぐ＝実際に歌われた語と時刻だけが残る。失敗時 None。"""
    import os
    import subprocess
    import sys
    here = Path(__file__).parent
    cache = Path(vocal_wav).with_suffix(".vad.json")       # 例: aud_0.vocals.vad.json（全曲ぶん・絶対秒）
    words = None
    if cache.exists():
        try:
            words = json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            words = None
    if words is None:
        try:
            r = subprocess.run([sys.executable, str(here / "transcribe_vad.py"), str(vocal_wav)],
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=900, cwd=str(here.parent), env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            for line in reversed((r.stdout or "").strip().splitlines()):
                if line.strip().startswith("["):
                    words = json.loads(line.strip())
                    cache.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
                    break
            if words is None and log is not None:          # 出力にJSON行なし＝transcribe_vad が異常終了
                log.append(f"[vad文字起こし:失敗] rc={r.returncode} JSON出力なし。stderr末尾: {(r.stderr or '')[-400:]}")
        except Exception as ex:
            if log is not None:
                log.append(f"[vad文字起こし:例外] {type(ex).__name__}: {ex}")
            words = None
    if not words:
        if log is not None:
            log.append("[vad文字起こし:なし] →区間特定/再整列はFA/階名のみ（vad補正なし）")
        return None
    sub = [w for w in words if t0 <= w["start"] <= t1]     # 区間でフィルタ（絶対秒・2番以降の混入を防ぐ）
    if log is not None:
        log.append(f"[vad文字起こし] 全{len(words)}語→区間{len(sub)}語（vad_filter・幻覚抑止）")
    return sub or None


def _correct_forced_pitches(vocal_wav: str, chars: list, abs_t0: float,
                            song_title: str, section_spec: str, log: list | None = None,
                            overrides: list | None = None) -> list:
    """FA の各かな音符の音高を Gemini が聴いて微修正（1対1）。タイミング(start/end)は触らない。

    実績のある音高補正を FA の上に乗せ直す（pyin f0 のオクターブ誤り等を耳で潰す）。

    ★**黙って補正なしで進まない**（2026-10-02・規約 song_service_editing「`except Exception` は止める・途中で見せる・
      差し戻しの合図を握らない」）。以前は全体を `except Exception` で包み、型の差し戻し（読めない音名・歌の高さの外）が
      直らなかったときも、通信の失敗も、補正なしの高さのまま黙って進んでいた（記録にも返事にも出なかった）。
      - 止める合図（`song_flow.NOT_SWALLOW`）は握らない
      - 型の差し戻し（読めない音名・歌の高さの外・頼んだ数と違う数＝`corrected_type`）が直らなかった
        （`song_llm.output_unfixed`）＝形式の所見が直らない → `StepUnfixed` で止める
        （整列の道具 `align_vocal` が「整列に失敗した」と所見つきで返す。整列の結果は使わない）
      - それ以外で補正できなかった（通信の失敗・音声が短い）→ 機械採譜の高さのまま進み、
        そのことを `log` と `overrides`（`align_vocal` が「整列で機械が決めたこと」として返事と記録に出す）に書く
    """
    from song_flow import NOT_SWALLOW                       # noqa: E402（止める・途中で見せる・差し戻しは握らない）

    def _uncorrected(why: str) -> None:
        if log is not None:
            log.append(f"[FA音高補正] ★補正なし（{why}）→ 機械採譜（pyin）の高さのまま")
        if overrides is not None:
            overrides.append(f"音高の補正（音声を聴いて高さを直す）をしなかった——{why}。高さは機械採譜（pyin）のまま")

    try:
        mids = [c.get("midi") for c in chars]                  # None は前後で補完
        last = None
        for i in range(len(mids)):
            mids[i] = last = (mids[i] if mids[i] is not None else last)
        last = None
        for i in range(len(mids) - 1, -1, -1):
            mids[i] = last = (mids[i] if mids[i] is not None else last)
        mids = [m if m else 67 for m in mids]
        t0, t1 = abs_t0 + chars[0]["start"], abs_t0 + chars[-1]["end"]
        clip = load_wav(Path(vocal_wav))[int(t0 * SR): int(t1 * SR)]
        if len(clip) < SR:
            for c, m in zip(chars, mids):
                c["midi"] = m
            _uncorrected(f"音声が {len(clip) / SR:.2f}秒と短い（1秒未満は聴かせない）")
            return chars
        data = _wav_bytes(clip)
        desc = " ".join(f"{i}:{c['char']}={spell_pitch(mids[i])}" for i, c in enumerate(chars))
        from song_usage import run_counted                  # noqa: E402（費用に入れる・2026-10-08）
        out = run_counted(_audio_correct_agent, "カバーの音高の聞き直し", [
            f"曲『{song_title}』の「{section_spec}」の分離ボーカルです。\n"
            f"各かなの機械採譜音高（発声タイミングは確定済み・index:かな=音名（国際式・`C4`＝中央のド）・"
            f"全{len(chars)}音）: {desc}\n"
            f"音声を聴いて、各かなの音高だけを微修正した音名の列を、同じ{len(chars)}個・同順で返してください。",
            BinaryContent(data=data, media_type="audio/wav")], output_type=corrected_type(len(chars))).output
        # 読めない音名・歌の高さの外（`_Corrected._readable_pitches`）と、頼んだ数と違う数（`corrected_type`）は型が差し戻すので、
        # ここへは読める音名が頼んだ数だけ来る。★2026-10-02 まで、数が違えば機械が DTW で各かなへ割り振っていた（LLM の出力の書き換え）
        gem = [song_llm.PITCH.read(p) for p in out.pitches]
        if len(gem) != len(chars):
            from song_fix import AssemblyError
            raise AssemblyError(f"カバーの音高補正に、型を通ったはずの数の違う高さが来た（{len(gem)}個・頼んだのは {len(chars)}個）")
        if log is not None:
            log.append(f"[FA音高補正] {len(chars)}かな → Gemini補正{len(gem)}（タイミング保持）")
        for c, p in zip(chars, gem):
            c["midi"] = p
        return chars
    except NOT_SWALLOW:                                     # 止める・途中で見せる・差し戻しは握らない
        raise
    except Exception as e:
        if song_llm.output_unfixed(e):                     # 型の差し戻しが直らなかった＝形式の所見が直らない → 止める
            raise song_llm.to_step_unfixed(e, "カバーの音高補正", step="カバーの音高補正") from e
        _uncorrected(f"{type(e).__name__}: {str(e)[:120]}")
        return chars


def _merge_yoon(chars: list) -> list:
    """拗音/小書きかな（ゃゅょぁぃぅぇぉ）を直前のかなへ結合＝「し」「ゃ」を「しゃ」1音に。
    FAは1文字ずつ整列するが VOCALOID の1ノート=1モーラなので結合が要る。start=主かな維持・end=小かなまで延長。
    ★「ゎ」はつながない（2026-10-02・組み立ての `lyric_slots._SMALL_KANA` と同じ扱い・仕様書 §2）。以前はここ・`sofa_align`・
      `lyric_display` だけ「ゎ」を小書きとして前の字につなぎ、組み立ては別の1モーラとして数えていた"""
    from lyric_slots import _SMALL_KANA                     # 小書きの一覧は組み立てと同じ1つ
    small = set(_SMALL_KANA)
    out: list = []
    for c in chars:
        if c.get("char") in small and out:
            out[-1]["char"] = out[-1].get("char", "") + c["char"]
            out[-1]["end"] = c.get("end", out[-1].get("end"))
        else:
            out.append(dict(c))
    return out


def _en_phon(s: str) -> str:
    """英単語→かなローマ字に寄せた音韻正規化（日本語にl/vが無い等）。DP照合・スパン特定用。"""
    import re
    s = re.sub(r"[^a-z]", "", s.lower())
    s = re.sub(r"^i(?=[^aeiou])", "ai", s)                     # 語頭の長母音I（I'll→airu, I'm→aimu）＝かな「あい」に対応
    s = re.sub(r"(.)\1", r"\1", s)                             # 重複子音縮約 ll→l
    return s.replace("l", "r").replace("v", "b").replace("y", "i")


def _english_runs(vad_words: list | None, clip_t0: float, end_sec: float) -> list[list[dict]]:
    """ASR語からASCII語（英語らしき）連なり（語間<1.5s）の run 一覧を返す。検知は音声由来＝
    日本語歌唱の外来語（シャイニングスター→"Sh ining star"）も混ざるので、採否は歌詞原文との照合で決めること。"""
    import re
    eng = [w for w in (vad_words or []) if clip_t0 <= w["start"] <= end_sec and re.search(r"[A-Za-z]", w.get("word", ""))]
    runs: list[list[dict]] = []
    cur: list[dict] = []
    for w in eng:
        if cur and w["start"] - cur[-1]["end"] >= 1.5:
            runs.append(cur)
            cur = []
        cur.append(w)
    if cur:
        runs.append(cur)
    return runs


def _match_eng_phrase(run: list[dict], eng_phrases: list[str]) -> tuple[str | None, float]:
    """ASR run のテキストに最も近い歌詞原文英語フレーズと一致率。<0.5 なら英語扱いしない（誤英語表記）。"""
    import difflib
    import re
    norm = lambda s: re.sub(r"[^a-z ]", "", s.lower()).strip()
    rtxt = norm(" ".join(w["word"] for w in run))
    best, ratio = None, 0.0
    for p in eng_phrases or []:
        r = difflib.SequenceMatcher(None, rtxt, norm(p)).ratio()
        if r > ratio:
            best, ratio = p, r
    return best, ratio


def _dp_group_kana(roms: list[str], wl: list[str]) -> list[int] | None:
    """かな（ローマ字化済み）列を語数分の**連続グループ**に分割し、各グループ↔語の類似度合計を最大化（DP）。
    返り値: 各かなの語index（単調・全語使用）。かな数<語数なら None。"""
    import difflib
    C, W = len(roms), len(wl)
    if C < W or W == 0:
        return None
    NEG = -1e9
    dp = [[NEG] * (W + 1) for _ in range(C + 1)]
    cut = [[0] * (W + 1) for _ in range(C + 1)]
    dp[0][0] = 0.0
    for j in range(1, C + 1):
        for w in range(1, W + 1):
            for k in range(w - 1, j):
                if dp[k][w - 1] <= NEG / 2:
                    continue
                g = "".join(roms[k:j])                         # 類似度×かな数（長いグループの一致を過小評価しない）
                sc = dp[k][w - 1] + difflib.SequenceMatcher(None, g, wl[w - 1]).ratio() * (j - k)
                if sc > dp[j][w]:
                    dp[j][w], cut[j][w] = sc, k
    if dp[C][W] <= NEG / 2:
        return None
    kw, j = [0] * C, C
    for w in range(W, 0, -1):
        k = cut[j][w]
        for t in range(k, j):
            kw[t] = w - 1
        j = k
    return kw


def _locate_kana_span(roms: list[str], phrase: str) -> tuple[float, tuple[int, int] | None]:
    """かな列（ローマ字化済み・歌詞順）の中から、英語フレーズに音韻が最も合う**連続スパン** [i,j) を特定。
    時間窓に頼らない＝FAが英語かなをどこに置いていても正しい集合を選べる（「しょん」取りこぼし防止）。"""
    import difflib
    T = _en_phon(phrase)
    best: tuple[float, tuple[int, int] | None] = (0.0, None)
    n = len(roms)
    for i in range(n):
        g = ""
        for j in range(i, min(n, i + 40)):
            g += roms[j]
            if len(g) < len(T) * 0.5:
                continue
            if len(g) > len(T) * 1.8:
                break
            r = difflib.SequenceMatcher(None, g, T).ratio()
            if r > best[0]:
                best = (r, (i, j + 1))
    return best


def _energy_end(vocals_wav: str, b1: float, cap: float = 3.0) -> float:
    """区間終端をエネルギーで補正: 声が続く限り延長し、0.35s無音が続いた所で止める（間奏/次コーラスは含めない）。
    ASR最終語end+固定padだと伸ばし終わりが切れ、最終フレーズが整列で圧縮される（かんじて+1.1sの実測原因）。"""
    y = load_wav(Path(vocals_wav)).astype(np.float32)
    hop = int(0.02 * SR)
    a0 = b1 - 2.0
    seg = y[int(a0 * SR):int((b1 + cap) * SR)]
    if len(seg) < hop * 4:
        return b1 + 0.8
    rms = np.array([np.sqrt(np.mean(seg[i:i + hop] ** 2)) for i in range(0, len(seg) - hop, hop)])
    thr = float(rms.max()) * 0.08
    sil = 0
    for k in range(int(2.0 / 0.02), len(rms)):
        if rms[k] > thr:
            sil = 0
        else:
            sil += 1
            if sil >= int(0.35 / 0.02):
                return a0 + (k - sil + 1) * 0.02 + 0.3
    return b1 + cap


def _extend_start_by_voice(vocals_wav: str, t_start: float, max_gap: float = 3.0, cap: float = 6.0) -> float:
    """区間始端の粗延長（_extend_end_by_voice の鏡像）: t_start以前にも歌声が続く限り（無音がmax_gap未満なら）
    始端を前へ遡る。ASRが冒頭のシャウト等を取りこぼすと歌詞照合の始端が遅れ、クリップ外の頭フレーズが
    区間先頭に押し込まれて min-gap 連打に潰れる（railgun 1番で実測: 実歌い出しが照合始端の約1.5s前
    →頭7音が75ms間隔に密集）。1番冒頭の手前は通常イントロ（>max_gapの無音/伴奏のみ）なので自然に停止する。"""
    y = load_wav(Path(vocals_wav)).astype(np.float32)
    hop = int(0.02 * SR)
    a0 = max(0.0, t_start - cap)
    seg = y[int(a0 * SR):int((t_start + 3.0) * SR)]
    if len(seg) < hop * 4:
        return t_start
    rms = np.array([np.sqrt(np.mean(seg[i:i + hop] ** 2)) for i in range(0, len(seg) - hop, hop)])
    thr = float(rms.max()) * 0.08
    first_voiced = min(int((t_start - a0) / 0.02), len(rms) - 1)
    gapf = int(max_gap / 0.02)
    k = first_voiced
    while k >= 0:
        if rms[k] > thr:
            first_voiced = k
        elif first_voiced - k > gapf:
            break
        k -= 1
    return min(t_start, a0 + first_voiced * 0.02)


def _extend_end_by_voice(vocals_wav: str, t_end: float, max_gap: float = 3.0, cap: float = 30.0) -> float:
    """区間終端の粗延長: t_end以降も歌声が続く限り（無音がmax_gap未満なら）終端を延ばす。
    ASR文字起こしが末尾語を取りこぼすと歌詞照合の終端が早くなり、末尾フレーズがクリップ外で詰め込まれる
    （フル新規実行で実際に発生: 294語ASRで67.1s判定・真の終端83s・70音詰め込み）。1番末尾の後は通常
    長い間奏（>max_gap）なので、有声追跡は間奏で自然に停止する。"""
    y = load_wav(Path(vocals_wav)).astype(np.float32)
    hop = int(0.02 * SR)
    a0 = max(0.0, t_end - 3.0)
    seg = y[int(a0 * SR):int((t_end + cap) * SR)]
    if len(seg) < hop * 4:
        return t_end
    rms = np.array([np.sqrt(np.mean(seg[i:i + hop] ** 2)) for i in range(0, len(seg) - hop, hop)])
    thr = float(rms.max()) * 0.08
    last_voiced = int((t_end - a0) / 0.02)
    gapf = int(max_gap / 0.02)
    k = last_voiced
    while k < len(rms):
        if rms[k] > thr:
            last_voiced = k
        elif k - last_voiced > gapf:
            break
        k += 1
    return max(t_end, a0 + last_voiced * 0.02)


def _pyin_midis(vocals_wav: str, clip_t0: float, starts: list[float], ends: list[float]) -> list:
    """各かな区間 [start,end)（clip_t0基準秒）の音高を pyin f0 中央値で実測（JA-FAを通らない英語かな用）。"""
    try:
        import librosa
        t0, t1 = min(starts) - 0.1, max(ends) + 0.3
        y = load_wav(Path(vocals_wav))[int((clip_t0 + t0) * SR):int((clip_t0 + t1) * SR)].astype(np.float32)
        f0, voiced, _ = librosa.pyin(y, fmin=130.0, fmax=1000.0, sr=SR, hop_length=256)
        times = librosa.times_like(f0, sr=SR, hop_length=256) + t0
        midi = librosa.hz_to_midi(f0)
        midi = np.where(voiced & ~np.isnan(midi), midi, np.nan)
        out = []
        for s, e in zip(starts, ends):
            seg = midi[(times >= s) & (times < max(e, s + 0.08))]
            seg = seg[~np.isnan(seg)]
            out.append(int(round(float(np.median(seg)))) if seg.size else None)
        return out
    except Exception:
        return [None] * len(starts)


def _segmented_forced_align(vocals_wav: str, lyrics_kana: str, vad_words: list | None,
                            clip_t0: float, end_sec: float, log: list | None = None,
                            gap_rms: float = 0.08, gap_sec: float = 0.35, max_seg: float = 6.0,
                            eng_phrases: list[str] | None = None) -> list | None:
    """vad語を**語境界で約5秒ごとに分割**し、歌詞かなを各セグメントへ割り当て、各セグメントを個別FA。
    連続歌唱でも短いクリップでFAするので**長区間のFA圧縮（音節の詰め込み）を回避**できる。
    返り値: chars [{char, start(clip_t0基準秒), end, midi}, ...]（FA失敗/材料不足時 None）。"""
    if not vad_words:
        return None
    import difflib
    lyr = kana_text_chars(lyrics_kana)                      # かな以外は型が差し戻す（2026-10-02・案B。以前は空白・`、。「」` を落とした）。行の切れ目（改行）は定義された区切りとして除く（2026-10-03）
    vw = [dict(w, rs=w["start"] - clip_t0, re=w["end"] - clip_t0)
          for w in vad_words if clip_t0 <= w["start"] <= end_sec]
    if len(vw) < 2 or len(lyr) < 2:
        return None
    # **無音ギャップ（フレーズ境界）で分割**: 歌声エネルギーが落ちる所で句を切る＝句を断たず（シャ分断防止）、
    # 各句は短いので長区間のFA圧縮（末尾詰め込み）も回避＝両取り。vad語~5秒の機械分割は廃止。
    yv = load_wav(Path(vocals_wav))[int(clip_t0 * SR):int(end_sec * SR)].astype(np.float32)
    hopv = int(0.02 * SR)
    rmsv = np.array([np.sqrt(np.mean(yv[i:i + hopv] ** 2)) for i in range(0, max(1, len(yv) - hopv), hopv)])
    voiced = rmsv > (float(rmsv.max()) * gap_rms if rmsv.size else 1.0)
    segs, i, mlen, gapf = [], 0, len(voiced), int(gap_sec / 0.02)   # gap_sec以上の無音で句を切る
    while i < mlen:
        if voiced[i]:
            j, last, sil = i, i, 0
            while j < mlen:
                if voiced[j]:
                    last, sil = j, 0
                else:
                    sil += 1
                    if sil > gapf:
                        break
                j += 1
            segs.append((i * 0.02, last * 0.02))
            i = j
        else:
            i += 1
    segs = [s for s in segs if s[1] - s[0] > 0.15] or [(vw[0]["rs"], vw[-1]["re"])]
    out = []                                                    # 安全弁: 無音の無いmax_seg超の長区間だけ ~5s で機械分割（圧縮回避）
    for a, b in segs:
        if b - a > max_seg:
            n = int((b - a) // 5) + 1
            for k in range(n):
                out.append((a + k * (b - a) / n, a + (k + 1) * (b - a) / n))
        else:
            out.append((a, b))
    segs = out
    rchar = None
    try:
        import pykakasi
        kks = pykakasi.kakasi()
        to_hira = lambda s: "".join(x["hira"] for x in kks.convert(s))
        _small = set("ゃゅょぁぃぅぇぉっー")                # 「ゎ」は小書きとして扱わない（`_merge_yoon` と同じ・2026-10-02）
        rchar = lambda c: "" if c in _small else "".join(x.get("hepburn", "") for x in kks.convert(c)).lower()
    except Exception:
        to_hira = lambda s: s
    # **割り当て用に単発FA(pass1)を先に実行**＝句をまたぐ対応（とじこめて/シャイニング）をFAに任せて各音節を正しい区間へ。
    # vad時刻はシャを早く誤るが、単発FAは"て"を伸ばし"し"を実位置(ギャップ後)へ置けるので割り当てが正確。timingは後段の区間別FAで精緻化。
    ct = [None] * len(lyr)
    pass1 = _forced_align(vocals_wav, lyrics_kana, clip_t0, end_sec)
    if pass1:
        p1 = [c for c in pass1 if c.get("char", "").strip()]
        for a, b, sz in difflib.SequenceMatcher(None, lyr, [c["char"] for c in p1], autojunk=False).get_matching_blocks():
            for k in range(sz):
                ct[a + k] = p1[b + k]["start"]                 # pass1は clip_t0基準秒（segs と同基準）
    if sum(t is not None for t in ct) < 2:                     # pass1不可→vad語時刻でフォールバック
        vc = []
        for w in vw:
            h = to_hira(w["word"])
            for i2, c in enumerate(h):
                vc.append((c, w["rs"] + (w["re"] - w["rs"]) * i2 / max(1, len(h))))
        for a, b, sz in difflib.SequenceMatcher(None, lyr, [c for c, _ in vc], autojunk=False).get_matching_blocks():
            for k in range(sz):
                ct[a + k] = vc[b + k][1]
    kn = [(i, t) for i, t in enumerate(ct) if t is not None]
    if len(kn) < 2:
        return None
    xi = [i for i, _ in kn]
    xt = [t for _, t in kn]
    for i in range(len(ct)):
        if ct[i] is None:
            ct[i] = float(np.interp(i, xi, xt))
    # **英語区間の分離**: 無音に加えて英語区間でも分割する。英語かなはJA-FA（前詰めする）に通さず、
    # en-FA（正原文＋英語モデル）の語時刻へ直接配置。かな集合は歌詞テキストの音韻照合＋pass1時刻で特定（時間窓に頼らない）。
    eng_regions: list[dict] = []
    if eng_phrases and rchar is not None:
        from collections import defaultdict
        roms_all = [rchar(c) for c in lyr]
        for run in _english_runs(vad_words, clip_t0, end_sec):
            best, ratio = _match_eng_phrase(run, eng_phrases)
            t0r, t1r = run[0]["start"] - clip_t0, run[-1]["end"] - clip_t0
            if best is None or ratio < 0.5:                     # 歌詞に無い"英語"＝日本語歌唱の誤英語表記（ASRの聞き取り由来）
                if log is not None:
                    log.append(f"[英語FA分割] 検知 {clip_t0 + t0r:.2f}〜{clip_t0 + t1r:.2f}s"
                               f"「{' '.join(w['word'] for w in run)}」一致率{ratio:.2f}<0.5 → 日本語歌唱の誤英語表記＝対象外")
                continue
            ws0 = clip_t0 + t0r - 0.4
            words = _forced_align_en(vocals_wav, best, ws0, min(end_sec, clip_t0 + t1r + 0.6))
            if not words or len(words) < 2:
                if log is not None:
                    log.append(f"[英語FA分割] 「{best}」en-FA失敗 → 通常FAのまま")
                continue
            words = [{**w, "start": w["start"] + ws0 - clip_t0, "end": w["end"] + ws0 - clip_t0} for w in words]
            sr_, span = _locate_kana_span(roms_all, best)       # 歌詞かな上の英語スパン（しょん等の尻尾も含む）
            if span is None or sr_ < 0.4:
                if log is not None:
                    log.append(f"[英語FA分割] 「{best}」かなスパン特定失敗({sr_:.2f}) → 通常FAのまま")
                continue
            i0, j0 = span
            rs_, re_ = words[0]["start"], words[-1]["end"]      # 音韻照合の取りこぼし（あ↔I'll等）を pass1時刻で拡張補完
            while i0 > 0 and rs_ - 0.1 <= ct[i0 - 1] <= re_ + 0.1:
                i0 -= 1
            while j0 < len(lyr) and rs_ - 0.1 <= ct[j0] <= re_ + 0.6:
                j0 += 1
            kw = _dp_group_kana(roms_all[i0:j0], [_en_phon(w["word"]) for w in words])
            if kw is None:
                continue
            starts = [0.0] * (j0 - i0)
            byw = defaultdict(list)
            for t, wi in enumerate(kw):
                byw[wi].append(t)
            for wi, ks in byw.items():                          # 各英単語の実測時間帯に、対応かなを均等配置
                w = words[wi]
                for m, t in enumerate(sorted(ks)):
                    starts[t] = w["start"] + (w["end"] - w["start"]) * (m + 0.5) / len(ks)
            eng_regions.append({"i0": i0, "j0": j0, "words": words, "starts": starts,
                                "phrase": best, "kw": kw, "ratio": ratio, "span_r": sr_})
        for reg in eng_regions:                                 # セグメントを英語区間（実測語スパン）で切り出し＝JAセグメントから除外
            rs_, re_ = reg["words"][0]["start"] - 0.05, reg["words"][-1]["end"] + 0.05
            nsg = []
            for a, b in segs:
                if b <= rs_ or a >= re_:
                    nsg.append((a, b))
                    continue
                if rs_ - a > 0.15:
                    nsg.append((a, rs_))
                if b - re_ > 0.15:
                    nsg.append((re_, b))
            segs = nsg
    eng_idx = {i for reg in eng_regions for i in range(reg["i0"], reg["j0"])}
    near = lambda t: min(range(len(segs)), key=lambda k: 0 if segs[k][0] <= t <= segs[k][1]
                         else min(abs(t - segs[k][0]), abs(t - segs[k][1])))
    seg_chars: dict[int, list[int]] = {k: [] for k in range(len(segs))}
    for i, t in enumerate(ct):
        if i in eng_idx:                                       # 英語かなはJAセグメントに入れない（en-FAで配置済み）
            continue
        seg_chars[near(t)].append(i)
    allc = []                                                  # 各セグメントを個別FA→clip_t0基準へオフセット連結
    for k, (a, b) in enumerate(segs):
        idx = sorted(seg_chars[k])
        if not idx:
            continue
        kana = "".join(lyr[i] for i in idx)
        clip_start = max(0.0, clip_t0 + a - 0.15)
        ch = _forced_align(vocals_wav, kana, clip_start, min(end_sec, clip_t0 + b + 0.4))
        if ch:
            off = clip_start - clip_t0                         # FAは clip_start=0秒基準 → clip_t0基準へ
            for c in ch:
                c["start"] += off
                c["end"] = c.get("end", c["start"]) + off
            allc += ch
    for reg in eng_regions:                                    # 英語かな: en-FA語時刻へ配置（音高は区間のpyin実測）
        i0, j0, starts = reg["i0"], reg["j0"], reg["starts"]
        ends = starts[1:] + [reg["words"][-1]["end"]]
        mids = _pyin_midis(vocals_wav, clip_t0, starts, ends)
        for t in range(j0 - i0):
            allc.append({"char": lyr[i0 + t], "start": starts[t], "end": ends[t], "midi": mids[t]})
        if log is not None:
            wt = " ".join(f"{w['word']}@{clip_t0 + w['start']:.2f}" for w in reg["words"])
            log.append(f"[英語FA分割] 「{reg['phrase']}」(ASR一致{reg['ratio']:.2f}/かなスパン{reg['span_r']:.2f}) "
                       f"かな{j0 - i0}音をJA-FAから分離し en-FA語時刻へ: {wt}")
            byw2: dict[int, list[int]] = {}
            for t, wi in enumerate(reg["kw"]):
                byw2.setdefault(wi, []).append(t)
            for wi in sorted(byw2):
                log.append(f"    {reg['words'][wi]['word']}: " + " ".join(
                    f"{lyr[i0 + t]}@{clip_t0 + starts[t]:.2f}" for t in byw2[wi]))
    if not allc:
        return None
    allc.sort(key=lambda c: c["start"])
    allc = _merge_yoon(allc)                                    # 拗音/小書きを直前へ結合＝「しゃ」を1音に
    if log is not None:
        log.append(f"[セグメント分割FA] {len(segs)}セグメント・{len(allc)}音節（**無音ギャップ=フレーズ境界で分割**＝句を断たず末尾圧縮も回避・拗音結合済み）")
        for k, (a, b) in enumerate(segs):                       # トレース: 分割位置と各セグメントへのかな割当
            idx = sorted(seg_chars[k])
            if idx:
                kana = "".join(lyr[i] for i in idx)
                log.append(f"  seg{k + 1}: {clip_t0 + a:.2f}〜{clip_t0 + b:.2f}s({b - a:.1f}s) "
                           f"かな{len(idx)}字「{kana[:4]}…{kana[-3:]}」→個別FA")
    return allc


def _build_score_energy(chars: list, tempo: float, song_title: str, section_spec: str,
                        vocals_wav: str, clip_t0: float, log: list | None = None) -> SongScore:
    """セグメント分割FAの chars → SongScore。音符長は**実唱エネルギーが切れる所で終える（休符を跨がない）**、
    **最終音は実唱エネルギー終端まで延長**（伸ばし）。タイミングはFA実測そのまま（再整列/スナップ無し）。"""
    mids = [c.get("midi") for c in chars]
    last = None                                                # 無声(None)は前後の音高で補完
    for i in range(len(mids)):
        if mids[i] is None:
            mids[i] = last
        else:
            last = mids[i]
    last = None
    for i in range(len(mids) - 1, -1, -1):
        if mids[i] is None:
            mids[i] = last
        else:
            last = mids[i]
    mids = [m if m else 67 for m in mids]
    shift = _octave_shift(mids)                                # ミク音域へ寄せる
    t0, bps, g = chars[0]["start"], tempo / 60.0, Q // 8
    y = load_wav(Path(vocals_wav))
    hop = int(0.02 * SR)
    rms = np.array([np.sqrt(np.mean(y[i:i + hop] ** 2)) for i in range(0, max(1, len(y) - hop), hop)])
    thr = (float(rms.max()) if rms.size else 0.0) * 0.12

    def energy_end_abs(abs_t: float) -> float:                 # abs_t 以降、0.16s連続でthr未満＝発音の切れ目
        st = int(abs_t / 0.02)
        sil = 0
        for k in range(st + 5, len(rms)):
            if rms[k] < thr:
                sil += 1
                if sil >= 8:
                    return (k - sil + 1) * 0.02
            else:
                sil = 0
        return len(rms) * 0.02

    a2t = lambda abs_sec: int(round((abs_sec - clip_t0 - t0) * bps * Q))
    starts, prev = [], -1
    for c in chars:
        stt = max(prev + 1, int(round((c["start"] - t0) * bps * Q)))
        prev = stt
        starts.append(stt)
    notes = []
    for i, c in enumerate(chars):
        ee = a2t(energy_end_abs(clip_t0 + c["start"]))         # 発音の切れ目tick
        nx = starts[i + 1] if i + 1 < len(chars) else ee       # 最終音は切れ目まで＝伸ばし
        nend = min(nx, ee) if ee > starts[i] else nx           # 次音 or 発音終わりの早い方（休符跨がない）
        notes.append(SongNote(lyric=c["char"], key=mids[i] + shift, start_tick=starts[i],
                              duration_tick=max(g, nend - starts[i])))
    if log is not None:
        span = notes[-1].start_tick + notes[-1].duration_tick
        log.append(f"[音符長] エネルギー基準（休符跨がない）＋最終音延長 / {len(notes)}音 末尾{span / Q * 60 / tempo:.1f}s")
    apply_by_gap(notes, tempo=tempo)                                        # ★調声（音高もタイミングも変えない）
    return SongScore(title=f"{song_title}_{section_spec}", tempo=tempo, time_signature="4/4",
                     lyrics_full="".join(c["char"] for c in chars), vocal_notes=notes, chords=[])


#: 音名での区間特定のあと、これ以上の歌の途切れ（拍）を間奏とみなして区間を分ける（`_largest_vocal_block`・記録と返事に出る）
INTERLUDE_GAP_BEATS = 4.0


def _largest_vocal_block(sl: list, gap_beats: float = INTERLUDE_GAP_BEATS) -> list:
    """区間内の大ギャップ（間奏・無歌＝gap_beats拍以上）で分割し、最長の連続ボーカル塊を返す。

    ブレス（〜2.6拍程度）は割らず、間奏（7拍〜・14秒等）で切る。sl=[[pitch,dur,start],...]。
    """
    if len(sl) < 3:
        return sl
    runs, cur = [], [sl[0]]
    for i in range(1, len(sl)):
        gap = sl[i][2] - (sl[i - 1][2] + sl[i - 1][1])      # 前音の終わり→次音の頭
        if gap >= gap_beats:
            runs.append(cur)
            cur = [sl[i]]
        else:
            cur.append(sl[i])
    runs.append(cur)
    return max(runs, key=lambda r: (r[-1][2] + r[-1][1]) - r[0][2])  # 総尺最長の塊


from build_lipsync import write_lipsync
from song_pipeline import OUT_ROOT, render_placeholder_vocal
from song_schema import TICKS_PER_QUARTER as Q
from song_schema import SongNote, SongScore
from score_to_vpr import score_to_vpr
from reference_fetch import identify_section
from reference_to_wav import SR


class _SecKana(BaseModel):
    reason: str = Field(description="全歌詞のどの構造（Aメロ/サビ/1番等）を区間として選び、なぜそう判断したか（実唱認識テキストとの照合根拠も）。情報が不足した点があれば明記")
    lyrics_kana: str = Field(description="指定区間の歌詞を VOCALOID 用かなにしたもの（全文・順番どおり）。" + KANA_TEXT_RULE)

    @field_validator("lyrics_kana")
    @classmethod
    def _kana(cls, v: str) -> str:
        return _kana_written(v)


_section_kana_agent: "Agent[None, _SecKana]" = Agent(
    song_llm.pick_model("MELODY", "flash"), output_type=_SecKana,
    system_prompt=(
        "与えられた『全歌詞』から、指定区間の歌詞を**実際に歌われる順序**で取り出し、VOCALOID 用かなにします。\n"
        "手順:\n"
        "① 全歌詞の構成（イントロ/Aメロ/Bメロ/サビ/1番/2番/大サビ等）を把握する。\n"
        "② 指定区間を特定。指定は次のいずれもあり得る:\n"
        "   ・構造ラベル: 『1番』＝歌い出し〜1番サビまで（**2番・最後のサビ繰り返し・落ちサビは含めない**）、『サビ』＝サビ1回ぶん、『Aメロ』等。\n"
        "     **1番はサビの最終行で機械的に切らない**: サビ直後に間奏を挟まず続くフレーズ（締めの行・Cメロ等）があれば1番に含める"
        "（実唱認識ヒントがあればそれが尽きる所まで＝間奏の手前までが1番）。\n"
        "   ・歌詞フレーズ範囲: 『「word1」から「word2」まで』等 → word1 を含む箇所から word2 を含む箇所までを取り出す。\n"
        "   ・『全体』＝全歌詞（歌唱順）。\n"
        "③ **原曲で実際に歌われる順序に並べ直す**（ネット歌詞はサビを冒頭に置く等、順序が前後する事があるので歌唱順へ直す）。\n"
        "④ 音符数の目安を与えるので、**概ねその音符数〜1.4倍のモーラ数に収める**（大幅超過は区間取り違え・繰り返し混入）。\n"
        "VOCALOID 用かなの書き方: " + KANA_TEXT_RULE + "。小さい母音・ん・っは可。\n"
        "**英語・外来語**は、取得した全歌詞の**綴り/語に忠実に**ひらがなで書く（聞き間違い由来の別語に直さない＝原歌詞の語を使う）。"
        "モーラを過剰に割らず**音符数の目安に収める**（dreamを「びりいぶ」のように無関係に増やさない）。\n"
        "**歌詞は与えられたものを使い、創作・補完しない。**"
    ),
)


class _HeardLyrics(BaseModel):
    lyrics_kana: str = Field(description="音声から聴き取った実際の歌詞（全文）。" + KANA_TEXT_RULE)

    @field_validator("lyrics_kana")
    @classmethod
    def _kana(cls, v: str) -> str:
        return _kana_written(v)


_audio_lyrics_agent: "Agent[None, _HeardLyrics]" = Agent(
    song_llm.pick_model("ALIGN", "flash"), output_type=_HeardLyrics,
    system_prompt=(
        "分離済みボーカル音声を聴き、実際に歌われている歌詞をかなで全文書き取ります（音に忠実に）。\n"
        "VOCALOID 用かなの書き方: " + KANA_TEXT_RULE + "。英語は聞こえたままひらがなで。創作しない。"
    ),
)


class _AudioAlign(BaseModel):
    reasoning: str = Field(description="音声を聴いた所見：どの音符にどの音節が乗るか・無音/伸ばしの判断")
    mora_per_note: list[str] = Field(
        description="各音符に乗る歌詞かな（音符数ぴったり・時間順）。1音符に1〜3かな。"
                    "歌っていない音符（間奏/前後の伸ばし）は空文字。与えた正しい歌詞を順に使い切る。" + CELL_RULE)


_audio_align_agent: "Agent[None, _AudioAlign]" = Agent(
    song_llm.pick_model("ALIGN", "flash"), output_type=_AudioAlign,
    system_prompt=(
        "あなたは歌唱の発音タイミングを聴き分ける専門家です。分離ボーカル音声・音符列（各音の鳴る秒数つき）・"
        "正しい歌詞（かな）を渡します。\n"
        "**音声を実際に聴いて**、各音符の時刻にその瞬間歌われている歌詞のかなを割り当ててください（mora_per_note）。\n"
        "・与えた正しい歌詞を順番どおり過不足なく使う（創作しない）。\n"
        "・日本語が音符より多い箇所は1音符に2〜3かな、伸ばし（メリスマ）は1かな、歌っていない音符は空文字。\n"
        "・" + CELL_RULE + "\n"
        "・" + LYRIC_LINES_RULE + "\n"
        "・出力 mora_per_note の長さは音符数ぴったり。聴いた発声タイミングに合わせるのが最優先。"
    ),
)


def align_full_lyrics_audio(vocal_wav: str, sl: list, ref_pitch: list, onsets: list,
                            sung: list, lyrics_kana: str, *, tempo: float, song_title: str,
                            section_spec: str, log: list | None = None,
                            overrides: list | None = None,
                            findings: list | None = None) -> "SongScore | None":
    """分離ボーカルを聴いて、正しい歌詞を各音符（実オンセット）へ割り当てる音声グラウンディング整列。

    タイミングは実オンセット（onsets）のまま、割り当て（mora_per_note）だけ音声で決める。使えなければ None（記号整列へ）。
    `findings` を渡すと、割り当てたかなが歌詞と一致しない（落ちた・重なった）ことを書く（記号整列と同じ照らし方
    `align_reference.assign_findings`・2026-10-03 再レビュー。`align_vocal` の返事の「割り当て:」に出る）。

    ★**黙って次の経路へ行かない**（2026-10-02・洗い出し H9 と中）:
      - 割り当ての数は音符の数を知っている型（`align_reference.assign_count_type`）が差し戻す。直らなければ
        `StepUnfixed`「カバーの歌詞の割り当て」で止める（以前は余りを捨て・足りない所を空で埋め、所見にも出さなかった）
      - 止める・途中で見せる・差し戻しの合図（`song_flow.NOT_SWALLOW`）は握らない
      - それ以外で使えなかった（通信の失敗・音声が短い・置ける音が2つ未満）→ 記号整列へ進み、そのことを `log` と
        `overrides`（`align_vocal` が返事と記録に出す）に書く
    """
    from song_flow import NOT_SWALLOW                       # noqa: E402（止める・途中で見せる・差し戻しは握らない）

    def _skip(why: str) -> None:
        if log is not None:
            log.append(f"[align(audio)] ★使わない（{why}）→ 記号整列へ")
        if overrides is not None:
            overrides.append(f"音声を聴いて歌詞を割り当てる整列を使わなかった——{why}。記号整列（音の並びだけで割り当てる）にした")

    try:
        bps = tempo / 60.0
        t0, t1 = sl[0][2] / bps, (sl[-1][2] + sl[-1][1]) / bps
        clip = load_wav(Path(vocal_wav))[int(t0 * SR): int(t1 * SR)]
        if len(clip) < SR:
            _skip(f"音声が {len(clip) / SR:.2f}秒と短い（1秒未満は聴かせない）")
            return None
        data = _wav_bytes(clip)
        kana = kana_text_chars(lyrics_kana)                 # かな以外は型が差し戻す（2026-10-02・案B。以前は空白・`、。「」` を落とした）。行の切れ目（改行）は定義された区切りとして除いて数え、行ごとに見せる（2026-10-03）
        notes_desc = " ".join(
            f"{i}:{round(onsets[i] * 60 / tempo, 2)}s/{spell_pitch(ref_pitch[i])}/{sung[i]}拍"
            for i in range(len(ref_pitch)))
        out = assign_or_stop(_audio_align_agent, [
            f"曲『{song_title}』の「{section_spec}」の分離ボーカルです（{round(t1 - t0, 1)}秒）。\n"
            f"正しい歌詞（かな・全{len(kana)}字・順番厳守。元の歌詞の行ごと）:\n{lyric_lines_text(lyrics_kana)}\n"
            f"音符列（index:鳴る秒数/音名（国際式・`C4`＝中央のド）/長さ拍・全{len(ref_pitch)}音）: {notes_desc}\n"
            f"音声を聴いて、各音符に乗るかなを mora_per_note（長さ{len(ref_pitch)}）で返してください。",
            BinaryContent(data=data, media_type="audio/wav")], _AudioAlign, len(ref_pitch), "カバーの歌詞の割り当て")
        if len(out.mora_per_note) != len(ref_pitch):
            from song_fix import AssemblyError
            raise AssemblyError(f"カバーの音声での歌詞の割り当てに、型を通ったはずの数の違う割り当てが来た"
                                f"（{len(out.mora_per_note)}個・音符は {len(ref_pitch)}個）")
        assign = list(out.mora_per_note)
        if log is not None:
            log.append(f"[align(audio)] 思考: {out.reasoning}")
            log.append(f"[align(audio)] mora_per_note({len(out.mora_per_note)}): {out.mora_per_note}")
        g = Q // 8
        notes = []
        for i, cell in enumerate(assign):
            ms = split_cell(cell)                           # 鳴らし分けるものごと（長音は等分しない・`lyric_slots.parts_of`）
            if not ms:
                continue
            on = onsets[i]
            nxt = onsets[i + 1] if i + 1 < len(onsets) else on + min(sung[i], 2.0)
            span = max((nxt - on) / len(ms), 0.05)
            dur = min(span, 1.5)                            # 長い間は伸ばしすぎず息継ぎを無音に
            for j, c in enumerate(ms):
                notes.append(SongNote(lyric=c, key=ref_pitch[i],
                                      start_tick=int(round((on + j * span) * Q)),
                                      duration_tick=max(g, int(round(dur * Q)))))
        if len(notes) < 2:
            _skip(f"歌詞を置けた音が {len(notes)}つ（2つ未満）")
            return None
        # 割り当てたかなが歌詞と一致しない（落ち・重なり）ことは所見にして返す（記号整列と同じ部品・2026-10-03）。この経路を使うときだけ書く
        note_assign_findings(assign, kana, log=log, findings=findings, tag="[align(audio)]")
        off = notes[0].start_tick
        for nn in notes:
            nn.start_tick -= off
        apply_by_gap(notes, tempo=tempo)                                    # ★調声（音高もタイミングも変えない）
        return SongScore(title=f"{song_title}_{section_spec}", tempo=tempo, time_signature="4/4",
                         lyrics_full="".join(n.lyric for n in notes), vocal_notes=notes, chords=[])
    except NOT_SWALLOW:                                     # 止める・途中で見せる・差し戻しは握らない（型の差し戻しが直らない＝StepUnfixed も）
        raise
    except Exception as e:
        _skip(f"{type(e).__name__}: {str(e)[:120]}")
        return None


def _transcribe_lyrics_from_audio(vocal_wav: str, sl: list, tempo: float,
                                  song_title: str, section_spec: str, log: list | None = None) -> str:
    """区間の分離ボーカルを聴いて歌詞を書き取る（ネット取得できない時のフォールバック）。書き取れなければ空。

    ★**書き取れなかった理由を `log` に残す**（2026-10-02・洗い出しの低）。以前は `except Exception` で止める合図ごと握り、
      理由を残さず空を返していた（次の取得元＝知識生成へ黙って移った）。止める・途中で見せる・差し戻しの合図は握らない。"""
    from song_flow import NOT_SWALLOW                       # noqa: E402

    def _none(why: str) -> str:
        if log is not None:
            log.append(f"[歌詞=実音源から聴取] ★書き取れなかった（{why}）→ 次の取得元へ")
        return ""

    try:
        bps = tempo / 60.0
        t0, t1 = sl[0][2] / bps, (sl[-1][2] + sl[-1][1]) / bps
        clip = load_wav(Path(vocal_wav))[int(t0 * SR): int(t1 * SR)]
        if len(clip) < SR:
            return _none(f"音声が {len(clip) / SR:.2f}秒と短い")
        data = _wav_bytes(clip)
        out = _kana_or_stop(_audio_lyrics_agent, [          # かな以外は型が差し戻す・直らなければ止める（2026-10-02・案B）
            f"曲『{song_title}』の「{section_spec}」の分離ボーカルです。歌詞をかなで全文書き取って。",
            BinaryContent(data=data, media_type="audio/wav")])
        return out.lyrics_kana.strip() or _none("聴き取りの LLM が空を返した")
    except NOT_SWALLOW:
        raise
    except Exception as e:
        return _none(f"{type(e).__name__}: {str(e)[:120]}")


#: 歌詞の取得元: **どこからも取れなかった**（ネット・聴き取り・知識のどれも空。`lyrics_for_section` の2つ目・2026-10-03）
LYRICS_NONE = "none"


def _cover_lyrics(song_title: str, section_spec: str, n_notes: int, *,
                  vocals_wav: str | None = None, sl: list | None = None,
                  tempo: float = 120.0, hint_urls: list | None = None,
                  force_knowledge: bool = False,
                  log: list | None = None) -> tuple[str, str, list[str]]:
    """カバー歌詞を取得。優先順: ①ネット取得 → ②実音源から聴取 → ③知識生成。
    (かな, ソース, 英語フレーズ原文リスト) を返す。英語原文は en-FA（英語区間の実測整列）用に保持。
    force_knowledge=True で①②をスキップし③のみ（web検索が同名異曲に支配されて脱出できない時の明示スイッチ。
    きらきら星で実測: 同名ポップ曲の歌詞しか返らず検問指摘→再試行のループになった・2026-07-25）。"""
    lr = {"ok": False} if force_knowledge else \
        fetch_lyrics(song_title, hint_urls=hint_urls)      # ① ネット（最優先・既知ソース→検索）
    if lr.get("ok"):
        sung = ""                                          # この区間で“実際に歌われている内容”を vad文字起こしで提示し抽出を正確化
        if vocals_wav and sl and Path(vocals_wav).exists():
            t0 = float(sl[0][2]) * 60.0 / tempo
            t1 = (float(sl[-1][2]) + float(sl[-1][1])) * 60.0 / tempo
            t1 = _extend_end_by_voice(vocals_wav, t1)          # 実唱ヒントは**歌声が続く所まで**（サビ後の締めフレーズを取りこぼさない）
            vw = _transcribe_vad(vocals_wav, t0, t1, log=log)   # キャッシュ済み（区間特定で取得済み）
            if vw:
                sung = "".join(w["word"] for w in vw)
        hint = f"この区間で実際に歌われている内容（音声認識・参考。これと一致する範囲を全歌詞から選ぶ）: {sung}\n" if sung else ""
        # ★**Web の全歌詞は、中身を言ってから命令ではないと断り、出所と同じレコード（JSON）にして渡す**（2026-10-03・再レビュー・本人決定）。
        #   以前は全歌詞をそのまま依頼文に連結し、断り書きも無かった（カバーのコードの Web の要約には付けていた）。言い方は1か所
        #   （`reference_fetch.web_material_note`）。JSON にすると本文の改行は `\n` になり、本文の側から見出しや行を作れない
        rec = json.dumps({"出典": lr.get("source_url", ""), "全歌詞": lr["lyrics_full"]}, ensure_ascii=False)
        out = _kana_or_stop(_section_kana_agent,           # かな以外は型が差し戻す・直らなければ止める（2026-10-02・案B）。全歌詞（Web）は材料のまま渡す
            f"曲名: {song_title}\n区間: {section_spec}\n対象区間の音符数: 約{n_notes}個（モーラ数の目安）\n"
            f"{hint}\n" + web_material_note("全歌詞", "Web から取った歌詞そのもの（`全歌詞` の欄・本文の改行は `\\n`）と、その出典")
            + f"\n【全歌詞】\n{rec}")
        kana = out.lyrics_kana
        if kana.strip():
            if log is not None:
                log.append(f"[歌詞=ネット取得] {lr['source_url']}{'（区間=実唱認識で照合）' if sung else ''}")
                log.append(f"[歌詞抽出 理由] {out.reason}")        # LLMの判断理由＝渡した情報が十分か検証用
            # 取得元は出典の URL つきで返す（道具の返事「歌詞取得(…)」に出る・記録が残らない経路なので返事が代わり・2026-10-03）
            return kana, f"web: {lr['source_url']}", _english_phrases(lr["lyrics_full"])   # 英語原文＝en-FA用（かな化で失わない）
        if log is not None:                                # ★黙って次の取得元へ移らない（2026-10-02・洗い出しの低）
            log.append(f"[歌詞=ネット取得] ★{lr['source_url']} の全歌詞から区間の歌詞を抜き出せなかった"
                       f"（抜き出しの LLM が空を返した・理由: {out.reason[:120]}）→ 次の取得元へ")
    if vocals_wav and sl and not force_knowledge:          # ② 実音源から聴取
        kana = _transcribe_lyrics_from_audio(vocals_wav, sl, tempo, song_title, section_spec, log=log)
        if kana.strip():
            if log is not None:
                log.append("[歌詞=実音源から聴取]")
            return kana, "audio", []
    kana = _kana_or_stop(_lyrics_agent,                    # ③ 知識生成（最終手段）。かな以外は型が差し戻す（2026-10-02・案B）
        f"曲『{song_title}』の区間「{section_spec}」で歌う歌詞を全文（かな）。"
        f"この区間の音符は約{n_notes}個。").lyrics_kana
    if log is not None:
        log.append("[歌詞=知識生成(フォールバック)]" + ("" if kana.strip() else " ★空を返した（歌詞が取れていない）"))
    # ★**空なら「取れなかった」と返す**（2026-10-03・再レビュー）。以前は空でも取得元 `knowledge` の成功の形で返し、道具は
    #   「歌詞取得(knowledge): 0字。次は align_vocal」と返していた。どこからも取れなかったことを取得元 `LYRICS_NONE` で区別する
    return kana, ("knowledge" if kana.strip() else LYRICS_NONE), []


def write_accompaniment(song_dir: Path, accomp: np.ndarray) -> int:
    """伴奏 np を accompaniment.wav に書き出す（工程5: build_accompaniment 用）。伴奏長を返す。"""
    song_dir.mkdir(parents=True, exist_ok=True)
    accomp = accomp if len(accomp) else np.zeros(SR)
    write_wav(song_dir / "accompaniment.wav", accomp)
    return len(accomp)


def save_cut_source(section: dict, song_dir: Path) -> None:
    """【検証用・本番は不要】確定した区間[start_sec,end_sec]の**元音源**を source/ に保存し、切り出し範囲を耳で検証可能に:
    `cut_vocals.wav`(分離ボーカル) と `cut_original.wav`(分離前の原曲)。失敗してもログのみ（本処理に影響なし）。"""
    st = section
    try:
        import librosa
        import soundfile as sf
        s, e = float(st.get("start_sec", 0.0)), float(st.get("end_sec", 0.0))
        if e <= s:
            return
        src = Path(song_dir) / "source"
        src.mkdir(parents=True, exist_ok=True)
        vw = st.get("vocals_wav")
        if vw and Path(vw).exists():                       # 分離ボーカルの当該区間
            y, _ = librosa.load(vw, sr=SR, mono=True, offset=s, duration=e - s)
            sf.write(str(src / "cut_vocals.wav"), y, SR)
        orig = None                                        # 分離前の原曲(aud_0.mp3/wav 等。ローカル注入はwavの場合あり)
        if vw:
            for pat in ("aud_*.mp3", "aud_*.wav", "aud_*.m4a", "aud_*.ogg"):
                cands = [p for p in Path(vw).parent.glob(pat)
                         if not re.search(r"\.(vocals|accomp|drums|bass|other)\.wav$", p.name)]
                if cands:
                    orig = cands[0]
                    break
        if orig and orig.exists():
            y, _ = librosa.load(str(orig), sr=SR, mono=True, offset=s, duration=e - s)
            sf.write(str(src / "cut_original.wav"), y, SR)
        if st.get("log") is not None:
            st["log"].append(f"[検証用] 切り出し元音源を保存: source/cut_vocals.wav・cut_original.wav（{s:.1f}-{e:.1f}s）")
    except Exception as ex:
        if st.get("log") is not None:
            st["log"].append(f"[検証用:切り出し保存:例外] {type(ex).__name__}: {ex}")


def render_vocal_and_mix(score: SongScore, song_dir: Path, *,
                         log: list | None = None) -> tuple[int, int]:
    """カバー仕上げ（工程6: render_and_mix 用）: score.json / vpr / 仮歌 を書き、
    伴奏が歌の長さに届いているかを確かめる。(歌声長, 伴奏長) を返す（伴奏長は**保存されている伴奏そのもの**の長さ）。

    伴奏書き出し（write_accompaniment）と分離したので、伴奏だけ／ミックスだけの再実行が可能。

    ★**`accompaniment.wav` を書き換えない**（2026-09-27・目録の 音量-5）。以前はここで伴奏を
      仮歌の長さ（+0.5秒）で切るか埋め、**モノラル・ピーク0.85にして上書き保存**していた。
      伴奏の道具・`replace_region`・命綱のたびに通るので、オリジナル曲では**後奏と残響が切られ**、
      音源（SoundFont）のステレオも失われていた。いまの出力（vpr）は長さの違いをそのまま扱える——
      伴奏の実長まで再生範囲を広げる（`score_to_vpr._extend_audio_region`）。音量の見積もりは短いほうに
      揃えて測る（`_sung_median_db`）。長さを揃えた音が要る出力を足すときは、**メモリの上で**揃えること。
    ★**伴奏全体の音量 `accomp_gain` は受け取らない**（同日・目録の 音量-4）。2026-09-06 に仮歌ミックスを
      廃止してから効いていなかった。釣り合いは vpr の伴奏トラックの初期音量（`score_to_vpr._accomp_gain_db`）。
    """
    song_dir.mkdir(parents=True, exist_ok=True)
    (song_dir / "score.json").write_text(score.model_dump_json(indent=2), encoding="utf-8")
    score_to_vpr(score, song_dir / "vocal.vpr")           # 声のIDは既定（MIKU_V6_COMP_ID）
    render_placeholder_vocal(score, song_dir / "placeholder.wav")
    write_lipsync(score, song_dir)                        # 口の動き（score＋仮歌が揃った時点で確定）
    voc = load_wav(song_dir / "placeholder.wav")
    n = len(voc)                                          # 基準長＝仮歌(スコア)尺
    acc_p = song_dir / "accompaniment.wav"
    has_acc = acc_p.exists()
    raw_acc = len(load_wav(acc_p)) if has_acc else 0       # 保存されている伴奏の長さ（不足＝末尾無音の検知用）
    if log is not None:                                   # 【整合性ログ】伴奏が仮歌より短い＝末尾が無音パディング（不変条件違反を★で追える）
        pad = max(0, n - raw_acc) / SR                    # 0.5s前後は仮歌末尾バッファ(render_placeholder_vocalの+0.5s)で無害
        log.append(f"[整合:仮歌↔伴奏] 仮歌{n / SR:.2f}s 伴奏切出{raw_acc / SR:.2f}s 無音パディング{pad:.2f}s "
                   f"→ {'OK' if pad < 0.6 or not has_acc else '★伴奏が歌唱区間に届かず無音(要調査)'}")
    # ★仮歌ミックス（draft_mix）は 2026-09-06 に廃止。**存在理由が消えた**ため
    #   ——`export_vocal` が同じセッションの中で実歌声を90秒で出すので、
    #   サイン波の仮歌で仕上がりを判断する場面が無くなった（判断材料としても不正確）。
    #   音量バランスは vpr に焼き込む経路が正（`score_to_vpr._accomp_gain_db`）。
    #   **整合性ログは残す**——「伴奏が歌唱区間に届いていない」を検知するのは mix とは別の役目だから。
    return n, raw_acc


def _tempo_map(path: Path) -> list[tuple[int, int]]:
    """[(tick, tempo_us_per_beat)] を時刻順で返す（先頭に既定120を補完）。"""
    evs: list[tuple[int, int]] = []
    for tr in mido.MidiFile(str(path)).tracks:
        t = 0
        for m in tr:
            t += m.time
            if m.type == "set_tempo":
                evs.append((t, m.tempo))
    evs.sort()
    if not evs or evs[0][0] > 0:
        evs = [(0, 500_000)] + evs
    return evs


def _tick_to_sec(tick: int, tpb: int, tmap: list[tuple[int, int]]) -> float:
    """フルテンポマップで tick→秒変換（伴奏レンダ＝mido実秒と一致させるため）。"""
    sec, last_t, last_tempo = 0.0, 0, tmap[0][1]
    for tt, tempo in tmap:
        if tt >= tick:
            break
        sec += (tt - last_t) / tpb * (last_tempo / 1e6)
        last_t, last_tempo = tt, tempo
    sec += (tick - last_t) / tpb * (last_tempo / 1e6)
    return sec


def _tempo_at(tick: int, tmap: list[tuple[int, int]]) -> float:
    """指定 tick で有効なテンポ（BPM）。区間内が定テンポなら歌声vprのテンポに使える。"""
    cur = tmap[0][1]
    for tt, tempo in tmap:
        if tt <= tick:
            cur = tempo
        else:
            break
    return 60_000_000 / cur


def _octave_shift(pitches: list[int]) -> int:
    """旋律全体をミク音域(≈C4-E5)へ寄せるオクターブシフト。"""
    if not pitches:                                        # 空の旋律（主旋律chに音符なし等の壊れたMIDI）で median が
        return 0                                           # 例外死しない（T2B新地実測: pian-room候補で StatisticsError）
    med, sh = statistics.median(pitches), 0
    while med + sh < 62:
        sh += 12
    while med + sh > 74:
        sh -= 12
    return sh


# ───────────────────────────────────────────────────────────────────────────
# 工程関数（プロンプトWF用）: 各工程は section dict を介して中間状態を受け渡す。
#   draft → 区間旋律 (section_from_midi / section_from_audio)
#   歌詞   → lyrics_for_section
#   整列   → align_section
#   伴奏   → accompaniment_for_section（→ write_accompaniment で書出）
#   仕上げ → render_vocal_and_mix
# section["kind"] ∈ {"midi","audio"}（記憶生成は section を使わず raw_score を直接整列）。
# ───────────────────────────────────────────────────────────────────────────


#: 区間指定 `section_spec` の**秒の範囲の書き方**（2026-10-02・仕様書 §15-2）: 半角の数字（小数可）・単位は「秒」か「s」
#: （少なくとも片方に付ける）・区切りは「から」。例 `60秒から88秒`・`57.6sから87.7s`
_SECS_SPEC = re.compile(r"([0-9]+(?:\.[0-9]+)?)(秒|s)?\s*から\s*([0-9]+(?:\.[0-9]+)?)(秒|s)?")
#: 崩した秒の範囲（全角の数字・区切り `〜` `~` `ー` `−` `-`）を**見つけて差し戻すためだけ**の広い形。これで読むことはしない
_NUM_LOOSE = r"[0-9０-９]+(?:[.．][0-9０-９]+)?"
_SEP_LOOSE = r"(?:から|〜|～|~|ー|−|-)"
_SECS_SPEC_LOOSE = re.compile(rf"{_NUM_LOOSE}\s*(?:秒|s|ｓ)\s*{_SEP_LOOSE}\s*{_NUM_LOOSE}"
                              rf"|{_NUM_LOOSE}\s*{_SEP_LOOSE}\s*{_NUM_LOOSE}\s*(?:秒|s|ｓ)")
#: 秒の範囲から音を拾うときの許し（先頭音の取りこぼしを避ける。区間特定（語+秒）の型と同じ値）
_SEC_TOL = SEC_TOL


def _section_ok(s: int, e: int, n: int, how: str) -> None:
    """区間の index が使える範囲か（型・秒の範囲の検査が保証している）。崩れていたら不具合として止める（曲全体にしない）。"""
    if not (0 <= s < e <= n) or e - s < 2:
        from song_fix import AssemblyError
        raise AssemblyError(f"カバーの区間特定（{how}）に、検査を通ったはずの使えない範囲が来た: index[{s},{e}]（参照の音は {n}個）")


def section_from_midi(midi_path: str, melody_channel: int, section_spec: str,
                      song_title: str) -> dict:
    """工程2(MIDI): 参照MIDIの主旋律chから区間を切り出し、整列・伴奏に必要な情報を section に詰める。"""
    midi_path = str(midi_path)
    raw, tpb, _ = extract_melody(Path(midi_path), melody_channel)
    raw = [(a, b, m) for a, b, m in raw if (b - a) / tpb >= 0.1]
    breaks = [0] + [i for i in range(1, len(raw)) if (raw[i][0] - raw[i - 1][1]) / tpb >= 0.75]

    sec = identify_section([m for _, _, m in raw], breaks, section_spec, song_title)   # LLM へは国際式の音名で見せる
    # ★**範囲の外・2音未満の index は型が差し戻す**（2026-10-02・洗い出し H8。`reference_fetch.section_type`）。以前はここで範囲に
    #   丸め、2音未満なら黙って曲全体を区間にしていた
    s, e = sec.start_index, sec.end_index
    _section_ok(s, e, len(raw), "音名")
    sl = raw[s:e]
    shift = _octave_shift([m for _, _, m in sl])
    tmap = _tempo_map(Path(midi_path))
    base = sl[0][0]
    tempo = _tempo_at(base, tmap)

    # 各音の歌う長さ（IOI）と開始拍（区間頭=0）
    onsets = [(st - base) / tpb for st, _, _ in sl]
    sung = [round(onsets[i + 1] - onsets[i], 2) for i in range(len(sl) - 1)] + \
           [round((sl[-1][1] - base) / tpb - onsets[-1], 2)]
    ref_pitch = [m + shift for _, _, m in sl]
    sl_breaks = [0] + [i - s for i in breaks if s < i < e]
    n_notes = len(sl)
    log = [f"=== MIDIカバー追跡: 『{song_title}』/ {section_spec} 区間[{s},{e}] {n_notes}音 ==="]
    return {"kind": "midi", "song_title": song_title, "section_spec": section_spec,
            "section_index": [s, e], "n_notes": n_notes, "tempo": tempo,
            "onsets": onsets, "sung": sung, "ref_pitch": ref_pitch, "sl_breaks": sl_breaks,
            "midi_path": midi_path, "melody_channel": melody_channel,
            "tpb": tpb, "base_tick": base, "end_tick": sl[-1][1], "tmap": tmap, "log": log}


def section_from_audio(reference_json: str, accomp_wav: str | None, section_spec: str,
                       song_title: str) -> dict:
    """工程2(音声): 採譜JSONから区間を切り出し（間奏除去・音声で音高補正・クランプ）、section に詰める。"""
    log = [f"=== 音声カバー追跡ログ: 『{song_title}』/ {section_spec} ===",
           f"[参照] {reference_json}  伴奏={accomp_wav}"]
    d = json.loads(Path(reference_json).read_text(encoding="utf-8"))
    raw = d["notes"]                                        # [[pitch, dur拍, start拍], ...]
    tempo = float(d.get("tempo") or 120.0)
    # 区間特定の LLM へは保存された階名（`solfa` 欄）でなく、音符の MIDI 番号から国際式で綴り直して見せる（2026-10-01）
    ref_keys = [int(e[0]) for e in raw]
    breaks = d.get("phrase_breaks", [0])
    log.append(f"[1.採譜] 全{len(raw)}音 tempo{tempo}")

    vocals_wav = d.get("vocals_wav")
    s = e = None
    used_vad = False
    # 区間特定(明示秒): section_spec に「60秒から88秒」「57.6sから87.7s」のような明示的な秒範囲が
    # あれば LLM 解釈を通さず直接使う（決定論）。同じ指定に毎回違う区間が返る「区間ガチャ」で
    # coordinator が draft_melody を連打し request_limit を使い切った事故への対処（2026-08-09・decisions.md）。
    # 「155拍から235拍」のような拍指定を誤って秒扱いしないよう、秒単位（秒/s）の明示がある時だけ発動する
    # ★**書き方は `60秒から88秒`（半角の数字・区切りは「から」）だけを読む**（2026-10-02・洗い出しの低・仕様書 §15-2）。以前は `\d`
    #   （全角の数字も）と区切り6通り（〜 ~ ー − -）を読んでいた。崩した書き方（`６０秒〜８８秒`）は読み替えずに差し戻す
    #   （道具 `draft_melody` が「区間切り出しに失敗」として文を返す）。逆向き・音が2つ未満の範囲も同じく差し戻す
    #   （以前は黙って LLM の区間特定へ移り、そこでも2音未満なら曲全体を区間にしていた）
    secs = [float(n[2]) * 60.0 / tempo for n in raw]        # 各音符の開始秒
    # ★**「拍」を含む文でも秒の範囲は読む**（2026-10-03・再レビュー）。以前は文に「拍」が1字でもあると秒の範囲を探さず
    #   （`1番の60秒から88秒（4拍子）` の秒も捨て）LLM の区間特定へ回していた。拍の指定 `155拍から235拍` は数字の後に単位
    #   「秒」「s」が無いので、秒の範囲の形（単位が少なくとも片方に付く）に当たらない——拍を秒として読むことはない（仕様書 §15-2）。
    #   単位の無い `155から235` が先にあっても、単位の付いた秒の範囲を探す（最初に当たった1つだけを見ていた）
    m = next((x for x in _SECS_SPEC.finditer(section_spec) if x.group(2) or x.group(4)), None)
    if m:
        a, b = float(m.group(1)), float(m.group(3))
        if not b > a:
            raise ValueError(f"区間指定「{section_spec}」の秒の範囲が逆向きか幅が無い（{a:g}秒から{b:g}秒）"
                             "——`60秒から88秒` のように前が小さい範囲で書く")
        cand = [i for i, t in enumerate(secs) if a - _SEC_TOL <= t <= b + _SEC_TOL]   # 先頭音の取りこぼし許容は語+秒経路と同じ
        if len(cand) < 2:
            raise ValueError(f"区間指定「{section_spec}」の {a:g}〜{b:g}秒に参照の音が {len(cand)}個しか無い"
                             f"（参照の音は {min(secs):.1f}〜{max(secs):.1f}秒にある）——2音以上が入る秒の範囲で書く")
        s, e = cand[0], cand[-1] + 1
        used_vad = True                                    # 明示指定を尊重し largest_vocal_block の再カットもしない
        log.append(f"[2.区間特定(明示秒)] 「{section_spec}」→ {a:.1f}〜{b:.1f}s = index[{s},{e}]（LLM解釈スキップ・決定論）")
    elif _SECS_SPEC_LOOSE.search(section_spec):
        raise ValueError(f"区間指定「{section_spec}」の秒の範囲が読めない——`60秒から88秒` の形"
                         "（半角の数字・単位は「秒」か「s」・区切りは「から」）で書く")
    if s is None and vocals_wav and Path(vocals_wav).exists():   # 区間特定(語+秒): 文字起こしで“実際に歌っている”範囲を秒で特定→index化
        vad_full = _transcribe_vad(vocals_wav, 0.0, 1e9, log=log)   # 全曲（キャッシュ・align_section と共用）
        if vad_full:
            from reference_fetch import identify_section_secs
            # 逆向き・音が2つ未満の範囲は型が差し戻す（2026-10-02・`reference_fetch.section_secs_type`）
            ss = identify_section_secs(vad_full, section_spec, song_title, note_secs=secs)
            cand = [i for i, t in enumerate(secs) if ss.start_sec - _SEC_TOL <= t <= ss.end_sec + _SEC_TOL]   # 先頭音を取りこぼさない
            s, e = cand[0], cand[-1] + 1
            used_vad = True
            log.append(f"[2.区間特定(語+秒)] 「{section_spec}」→ {ss.start_sec:.1f}〜{ss.end_sec:.1f}s = index[{s},{e}]  根拠: {ss.reason}")
    if s is None:                                          # 音名フォールバック（読み上げ/伴奏を歌と誤判別しうる）
        sec = identify_section(ref_keys, breaks, section_spec, song_title)
        s, e = sec.start_index, sec.end_index              # 範囲の外・2音未満は型が差し戻す（2026-10-02・洗い出し H8）
        log.append(f"[2.区間特定(音名)] 「{section_spec}」→ index[{s},{e}]  根拠: {sec.reason}")
    _section_ok(s, e, len(raw), "音声")
    sl = raw[s:e]
    log.append(f"[2.区間] index[{s},{e}] {len(sl)}音")
    section_changes: list[str] = []
    if not used_vad:                                       # vad区間特定済みなら間奏は既に除外済み＝largest_vocal_blockで再カットしない
        sl2 = _largest_vocal_block(sl)                     # 音名フォールバック時のみ: 間奏(大ギャップ)を除き連続塊に絞る
        if len(sl2) < len(sl):
            # ★**機械が LLM の決めた区間を縮めたことを、記録と返事に出す**（2026-10-03・再レビュー・目録「カバー-区間の間奏除去」）。
            #   以前は作業記録（log）にだけ書き、区間の index も LLM が決めたままで返していた（返事の「区間[s,e]」と実際の音がずれ、
            #   フレーズの切れ目 `sl_breaks` も縮める前の頭から数えていた）。縮めるのは続ける——音名での区間特定（`identify_section`）は
            #   歌の途切れ（秒）を知らず、型で差し戻す材料が無い（直すなら区間特定の型に間の長さを見せる＝`reference_fetch` 側の変更）
            k0 = next(i for i, x in enumerate(sl) if x is sl2[0])
            s2, e2 = s + k0, s + k0 + len(sl2)
            bps = tempo / 60.0
            section_changes.append(
                f"区間特定（LLM・音名）の index[{s},{e}]（{len(sl)}音・{sl[0][2] / bps:.1f}〜{(sl[-1][2] + sl[-1][1]) / bps:.1f}秒）を、"
                f"{INTERLUDE_GAP_BEATS:g}拍以上の歌の途切れ（間奏とみなす）で分け、いちばん長い歌の塊 index[{s2},{e2}]（{len(sl2)}音・"
                f"{sl2[0][2] / bps:.1f}〜{(sl2[-1][2] + sl2[-1][1]) / bps:.1f}秒）だけにした（機械が区間を縮めた）")
            log.append(f"[2b.間奏除去] ★{section_changes[-1]}")
            sl, s, e = sl2, s2, e2
    log.append(f"[2.区間 pyin音名] {' '.join(spell_pitch(int(p)) for p, _, _ in sl)}")

    vocals_wav = d.get("vocals_wav")
    # 音高は pyin採譜のまま（新規=純プログラム）。**区間レベルのGemini音高補正は廃止**:
    # 主経路(FA)は char毎 pyin を使うため補正結果が反映されず無駄call＋設計（補正は編集UCのみ）にも反するため。
    # 音高補正が要るときは align_section の pitch_correct=True（_correct_forced_pitches・理由出力つき）で実施。
    log.append("[3.音高] pyin採譜のまま（LLM音高補正は編集UC=pitch_correct で実施）")
    sl = [[int(p), min(float(dur), 4.0), float(st)] for p, dur, st in sl]  # 幻の長大音符をクランプ（伴奏尺の暴走防止）
    pitches = [int(p) for p, _, _ in sl]
    shift = _octave_shift(pitches)
    base_beat = sl[0][2]
    onsets = [st - base_beat for _, _, st in sl]           # 開始拍（区間頭=0）
    sung = [round(onsets[i + 1] - onsets[i], 2) for i in range(len(sl) - 1)] + [round(sl[-1][1], 2)]
    ref_pitch = [p + shift for p in pitches]
    sl_breaks = [0] + [i - s for i in breaks if s < i < e]
    abs_t0 = base_beat * 60 / tempo                        # 区間頭の絶対時刻（伴奏スライス基準）
    end_sec = (sl[-1][2] + sl[-1][1]) * 60 / tempo
    hint = _maou_lyrics_page(d.get("audio_url", ""))       # 音源と同サイトの歌詞ページを導出（検索不要）
    return {"kind": "audio", "song_title": song_title, "section_spec": section_spec,
            "section_index": [s, e], "n_notes": len(sl), "tempo": tempo,
            "sl": sl, "onsets": onsets, "sung": sung, "ref_pitch": ref_pitch, "sl_breaks": sl_breaks,
            "vocals_wav": vocals_wav, "accomp_wav": accomp_wav,
            "abs_t0": abs_t0, "start_sec": abs_t0, "end_sec": end_sec, "hint": hint, "log": log,
            # 機械が区間を変えたこと（`source/section.json` に残り、`align_section` が `overrides` の頭に入れて `align_vocal` の返事に出す）
            "section_changes": section_changes}


def lyrics_for_section(section: dict, instruction: str = "", force_knowledge: bool = False) -> tuple[str, str]:
    """工程3: 区間の歌詞（かな全文）を取得。(かな, 取得元) を返す。音声参照は聴取/ヒントURLも使う。
    force_knowledge=True でネット/聴取をスキップしLLM知識のみ（同名異曲にweb検索が支配された時の脱出口）。"""
    st = section
    if st["kind"] == "audio":
        kana, src, eng = _cover_lyrics(instruction or st["song_title"], st["section_spec"], st["n_notes"],
                                       vocals_wav=st.get("vocals_wav"), sl=st.get("sl"), tempo=st["tempo"],
                                       hint_urls=[st["hint"]] if st.get("hint") else None,
                                       force_knowledge=force_knowledge, log=st["log"])
    else:                                                  # MIDIは音源が無いので聴取フォールバックは無し
        kana, src, eng = _cover_lyrics(instruction or st["song_title"], st["section_spec"], st["n_notes"],
                                       force_knowledge=force_knowledge, log=st["log"])
    st["eng_phrases"] = eng                                # 英語原文＝align_section の en-FA 再配置で使用
    if eng:
        st["log"].append(f"[歌詞英語原文] {eng}")
    st["log"].append(f"[歌詞({src})] {kana}")
    return kana, src


def _fix_octave_outliers(chars: list, vocals_wav: str, clip_t0: float,
                         log: list | None = None) -> list:
    """pyinのオクターブ誤り（息混じり区間でサブハーモニックにロック）を補正。
    ①文脈（前後2音の中央値から6半音以上外れ・±12で4半音以内に収まる）で疑い音を挙げ、
    ②**音響証拠で確定**: 測定f0位置と2倍音（1オクターブ上）位置のスペクトルエネルギーを比較し、
    2倍位置が明確に強い時だけ+12する。真の低音（メロディの谷）は基音位置が強いので触らない。"""
    import librosa
    mids = [c.get("midi") for c in chars]
    suspects = []
    for i, c in enumerate(chars):
        m = mids[i]
        if m is None:
            continue
        neigh = [mids[j] for j in range(max(0, i - 2), min(len(mids), i + 3)) if j != i and mids[j] is not None]
        if len(neigh) < 2:
            continue
        med = sorted(neigh)[len(neigh) // 2]
        if m - med <= -6 and abs(m + 12 - med) <= 4:
            suspects.append(i)
    if not suspects:
        return chars
    y = load_wav(Path(vocals_wav)).astype(np.float32)
    nfft, hop = 4096, 512
    S = np.abs(librosa.stft(y, n_fft=nfft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=nfft)
    fixed, kept = [], []
    for i in suspects:
        c = chars[i]
        f_lo = float(librosa.midi_to_hz(c["midi"]))
        a = int((clip_t0 + c["start"]) * SR / hop)
        bnd = int((clip_t0 + max(c.get("end", c["start"] + 0.15), c["start"] + 0.1)) * SR / hop)
        seg = S[:, max(0, a):max(a + 1, bnd)]
        def band(fc_):                                         # 中心±4%帯のエネルギー
            sel = (freqs >= fc_ * 0.96) & (freqs <= fc_ * 1.04)
            return float(seg[sel].mean()) if sel.any() else 0.0
        e1, e2 = band(f_lo), band(f_lo * 2)
        if e2 > 2.5 * e1:                                      # 基音が無く2倍が強い＝サブハーモニック誤り
            fixed.append(f"{c['char']}{c['midi']}→{c['midi'] + 12}")
            c["midi"] += 12
        else:
            kept.append(f"{c['char']}{c['midi']}")
    if log is not None and (fixed or kept):
        log.append(f"[音程オクターブ補正] 疑い{len(suspects)}音→補正{len(fixed)}音: " + " ".join(fixed)
                   + (f" ／真の低音として維持: {' '.join(kept)}" if kept else ""))
    return chars






def _min_gap_spread(chars: list, g: float = 0.07, log: list | None = None) -> list:
    """発音可能な最小間隔gを保証する（SOFA等が音符転移を数msに潰す箇所への対症）。
    後ろから走査し、次の音との間隔<gなら自分を早める＝詰まりクラスタだけを先頭側の空きへ押し出す。
    un-cram（オンセット再配置）と違い**詰まっていない音は一切動かさない**。"""
    T = [c["start"] for c in chars]
    n = len(T)
    moved = 0
    for i in range(n - 2, -1, -1):
        if T[i + 1] - T[i] < g:
            T[i] = T[i + 1] - g
            moved += 1
    for i in range(1, n):                                      # 押し出しすぎの前方衝突は最小間隔で前へ倒す
        if T[i] < T[i - 1] + 0.02:
            T[i] = T[i - 1] + 0.02
    for c, t in zip(chars, T):
        c["start"] = t
    if log is not None and moved:
        warn = "★異常に多い＝区間切れ/整列崩壊の疑い（終端・入力を確認）" if moved > max(10, int(n * 0.15)) else ""
        log.append(f"[最小間隔] {moved}音を前へ押し広げ（間隔≥{g * 1000:.0f}ms=発音可能を保証・他は不動）{warn}")
    return chars




def _cram_runs(T: list[float], gap: float = 0.085) -> list[tuple[int, int]]:
    """連続音節間隔 < gap が続く run（詰め込み）の [i,j] 範囲を返す。"""
    runs, i, n = [], 0, len(T)
    while i < n - 1:
        if T[i + 1] - T[i] < gap:
            j = i
            while j + 1 < n and T[j + 1] - T[j] < gap:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _integrate_alignment(chars: list, vad_words: list | None, clip_t0: float, end_sec: float,
                         vocals_wav: str, log: list | None = None,
                         uncram_before: float | None = None) -> list:
    """FA土台の chars（start=clip_t0基準秒）を「vad粗アンカー＋自動un-cram」で自動改善（統合アライナを本番移植）。
    平均ではなく各信号を得意なことだけに使う: vad=大ズレ(>1s)のフレーズ位置補正、onset=詰め込みを実アタックへ展開。
    chars を in-place 更新して返す。初回ビルドから詰め込み/大ズレを自動低減（位置の最終調整は後段のLLM自動修正パス）。"""
    import difflib

    import librosa
    T = [c["start"] for c in chars]
    n = len(T)
    if n < 4:
        return chars
    cram_before = sum(j - i + 1 for i, j in _cram_runs(T))
    seg = load_wav(Path(vocals_wav))[int(clip_t0 * SR):int(end_sec * SR)].astype(np.float32)
    onsets = sorted(librosa.onset.onset_detect(y=seg, sr=SR, units="time", backtrack=True))
    moved = 0
    if vad_words:                                          # ① vad粗アンカー: FAが大きく外したフレーズ位置だけ vad へ寄せる
        import pykakasi
        kks = pykakasi.kakasi()
        to_hira = lambda s: "".join(x["hira"] for x in kks.convert(s))
        vc = []
        for w in vad_words:
            if not (clip_t0 <= w["start"] <= end_sec):
                continue
            h = to_hira(w["word"])
            for k, c in enumerate(h):
                vc.append((c, (w["start"] + (w["end"] - w["start"]) * k / max(1, len(h))) - clip_t0))
        sm = difflib.SequenceMatcher(None, [c["char"] for c in chars], [c for c, _ in vc], autojunk=False)
        anchors = []
        for a, b, sz in sm.get_matching_blocks():
            for k in range(sz):
                anchors.append((T[a + k], vc[b + k][1]))
        mono = []
        for ft, vt in sorted(set(anchors)):
            if not mono or (ft > mono[-1][0] + 1e-3 and vt > mono[-1][1] + 1e-3):
                mono.append((ft, vt))
        if len(mono) >= 4:
            af = np.array([x[0] for x in mono])
            av = np.array([x[1] for x in mono])
            for i in range(n):
                vad_implied = float(np.interp(T[i], af, av))
                if abs(vad_implied - T[i]) > 1.0:          # 大ズレのフレーズ位置だけ
                    T[i] = vad_implied
                    moved += 1
            for i in range(1, n):
                if T[i] <= T[i - 1]:
                    T[i] = T[i - 1] + 0.04
    for (i, j) in _cram_runs(T):                           # ② 自動un-cram: 詰め込み run を実オンセットのフレーズへ展開
        if uncram_before is not None and T[i] > uncram_before:   # 選択適用: 終端フレーズ等は触らない（un-cramが悪化させる領域）
            continue
        C = j - i + 1
        start_t = T[i] - 0.06
        phrase = []
        for o in onsets:
            if o < start_t:
                continue
            if phrase and o - phrase[-1] > 0.45:
                break
            phrase.append(o)
            if len(phrase) >= max(C, int(C * 1.4)):
                break
        if len(phrase) >= C:
            sel = [phrase[round(k * (len(phrase) - 1) / (C - 1))] for k in range(C)] if C > 1 else [phrase[len(phrase) // 2]]
        else:
            sel = list(np.linspace(start_t, start_t + max(0.3 * (C - 1), 0.1), C))
        prev = T[i - 1] if i > 0 else start_t - 0.1
        for k in range(C):
            T[i + k] = t = max(sel[k], prev + 0.06)
            prev = t
    for k in range(1, n):
        if T[k] <= T[k - 1] + 0.085:
            T[k] = T[k - 1] + 0.1
    for c, t in zip(chars, T):
        c["start"] = t
    chars.sort(key=lambda c: c["start"])
    cram_after = sum(j - i + 1 for i, j in _cram_runs([c["start"] for c in chars]))
    if log is not None:
        log.append(f"[統合整列] 自動改善: 詰め込み {cram_before}→{cram_after}音 / vad粗位置補正 {moved}音"
                   "（位置の最終調整は後段のLLM自動修正パスに委ねる）")
    return chars


def _realign_english_by_enfa(chars: list, vad_words: list | None, vocals_wav: str,
                             clip_t0: float, end_sec: float, eng_phrases: list[str],
                             log: list | None = None, tail_to_next: bool = True,
                             snap_head_onset: bool = False) -> list:
    """英語区間を**英語FA（正原文＋英語wav2vec2）**で実測整列し、かなを語対応で配置する（本命の英語精度改善）。
    ① ASRでASCII語の連なり=英語区間を検知 → ② 歌詞原文の英語フレーズと照合（不一致=日本語歌唱の誤英語表記
    〔例: シャイニングスター→"Sh ining star"〕は触らない）→ ③ 一致区間を en-FA で単語時刻を実測
    → ④ かな集合は**歌詞テキストの音韻照合＋現時刻で特定**（時間窓に頼らない＝しょん等の尻尾も含む）
    → ⑤ DP分割で語対応し各語の時間帯に均等配置。tail_to_next=True で最終語の終端を次のかな直前まで延長
    （en-FAは伸ばし音を早めに切る癖があるため）。JA-FAの結果には触れない＝2分割の良さを保ったまま英語だけ直せる。"""
    import difflib
    import re
    if not eng_phrases or not chars:
        return chars
    eng = [w for w in (vad_words or []) if clip_t0 <= w["start"] <= end_sec and re.search(r"[A-Za-z]", w.get("word", ""))]
    if not eng:
        return chars
    runs, cur = [], [eng[0]]                                   # 連続する英語語を run にまとめる（語間 <1.5s）
    for w in eng[1:]:
        if w["start"] - cur[-1]["end"] < 1.5:
            cur.append(w)
        else:
            runs.append(cur)
            cur = [w]
    runs.append(cur)
    try:
        import pykakasi
        kks = pykakasi.kakasi()
        rom = lambda s: "".join(x.get("hepburn", "") for x in kks.convert(s)).lower()   # noqa: E731
    except Exception:
        return chars
    norm = lambda s: re.sub(r"[^a-z ]", "", s.lower()).strip()   # noqa: E731
    for run in runs:
        rtxt = norm(" ".join(w["word"] for w in run))
        best, ratio = None, 0.0                                # ② ASR区間テキスト↔歌詞原文の英語フレーズを照合
        for p in eng_phrases:
            r = difflib.SequenceMatcher(None, rtxt, norm(p)).ratio()
            if r > ratio:
                best, ratio = p, r
        if best is None or ratio < 0.5:                        # 歌詞に無い"英語"＝日本語歌唱の誤表記→触らない
            if log is not None:
                log.append(f"[英語FA] 検知区間 {run[0]['start']:.2f}〜{run[-1]['end']:.2f}s"
                           f"「{' '.join(w['word'] for w in run)}」一致率{ratio:.2f}<0.5 → 対象外（日本語歌唱の誤英語表記）")
            continue
        ws0, we0 = run[0]["start"] - 0.4, run[-1]["end"] + 0.6   # ③ 英語FA（正原文）で単語時刻を実測
        words = _forced_align_en(vocals_wav, best, ws0, we0)
        if not words or len(words) < 2:
            if log is not None:
                log.append(f"[英語FA] 「{best}」の整列失敗→この区間はFA/un-cramのまま")
            continue
        words = [{**w, "start": w["start"] + ws0 - clip_t0, "end": w["end"] + ws0 - clip_t0} for w in words]
        if snap_head_onset:                                    # 区間先頭語の頭を**最強オンセット**へ（直前ブレスをFAが
            try:                                               # 語頭と誤認して早取りする問題〔I'll早すぎ〕への対症）
                import librosa
                w0 = words[0]
                yseg = load_wav(Path(vocals_wav))[int(clip_t0 * SR):int(end_sec * SR)].astype(np.float32)
                env = librosa.onset.onset_strength(y=yseg, sr=SR)
                onf = librosa.onset.onset_detect(y=yseg, sr=SR, units="frames", backtrack=False)
                tt = librosa.frames_to_time(onf, sr=SR)
                cand = [(float(env[f]), float(t)) for f, t in zip(onf, tt)
                        if w0["start"] - 0.05 <= t <= min(w0["start"] + 0.55, w0["end"] - 0.05)]
                if cand:
                    _, t = max(cand)
                    if t - w0["start"] > 0.08:
                        if log is not None:
                            log.append(f"[英語FA] 先頭語頭を強オンセットへ {clip_t0 + w0['start']:.2f}→{clip_t0 + t:.2f}s（ブレス回避）")
                        w0["start"] = t
            except Exception:
                pass
        rs, re_ = words[0]["start"], words[-1]["end"]
        roms_all = [re.sub(r"[^a-z]", "", rom(c["char"])) for c in chars]
        sr_, span = _locate_kana_span(roms_all, best)          # ④ かな集合は歌詞テキストで特定（時間窓に頼らない）
        if span is None or sr_ < 0.4:
            if log is not None:
                log.append(f"[英語FA] 「{best}」かなスパン特定失敗({sr_:.2f}) → 触らない")
            continue
        i0, j0 = span                                          # 先頭側は**テキスト照合のみ**（時刻拡張は直前の日本語かな
        while j0 < len(chars) and rs - 0.1 <= chars[j0]["start"] <= re_ + 0.6:   # 〔せかいえ の え等〕を巻き込むため廃止。
            j0 += 1                                            # あ↔I'll は _en_phon の語頭I→ai 規則で照合側が拾う）
        idx = list(range(i0, j0))
        C = len(idx)
        if C < 2:
            continue
        # ⑤ かな→語対応: DP分割（かな列を語数分の連続グループに分け、各グループのローマ字↔語の類似度合計を最大化）
        kw = _dp_group_kana([rom(chars[i2]["char"]) for i2 in idx], [_en_phon(w["word"]) for w in words])
        if kw is None:
            continue
        from collections import defaultdict
        byw = defaultdict(list)
        for j in range(C):
            byw[kw[j]].append(j)
        spans = [(w["start"], w["end"]) for w in words]
        if tail_to_next and j0 < len(chars):                   # 最終語の終端を次のかな直前まで延長（伸ばし音の早切り補正）
            nxt = chars[j0]["start"]
            if nxt - 0.15 > spans[-1][1]:
                spans[-1] = (spans[-1][0], min(nxt - 0.15, spans[-1][1] + 1.0))
        before = [chars[i2]["start"] for i2 in idx]            # トレース: 上書き前の時刻を保存
        for wi, ks in byw.items():                             # 各英単語の実測時間帯に、対応かなを均等配置
            ws_, we_ = spans[wi]
            for m, j in enumerate(sorted(ks)):
                chars[idx[j]]["start"] = ws_ + (we_ - ws_) * (m + 0.5) / len(ks)
        if log is not None:
            wt = " ".join(f"{w['word']}@{clip_t0 + w['start']:.2f}" for w in words)
            log.append(f"[英語FA] 「{best}」を英語モデルで実測整列（一致率{ratio:.2f}）→ {C}音を語対応配置"
                       f"（尻尾{'延長' if spans[-1][1] > words[-1]['end'] + 1e-6 else 'そのまま'}）: {wt}")
            for wi in sorted(byw):                             # トレース: 語ごとの かな・上書き前→後
                ks = sorted(byw[wi])
                log.append(f"    {words[wi]['word']}: " + " ".join(
                    f"{chars[idx[j]]['char']}{clip_t0 + before[j]:.2f}→{clip_t0 + chars[idx[j]]['start']:.2f}" for j in ks))
    for k in range(1, len(chars)):                             # 単調化は**歌詞順を保ったまま**クランプ（sortすると
        if chars[k]["start"] <= chars[k - 1]["start"]:         # 直前の日本語かなと順序が入れ替わり歌詞が崩れる）
            chars[k]["start"] = chars[k - 1]["start"] + 0.04
    return chars


def _realign_english_by_asr(chars: list, vad_words: list | None, clip_t0: float, end_sec: float,
                            log: list | None = None) -> list:
    """英語区間（ASRが英語=ASCII語として認識した箇所）の**詰まった**音節を、英語ASR語の時間へ比例配分で再配置する。
    日本語FA(wav2vec2 ja)は英語を前詰めして詰まらせるので、ASR(多言語)の単語時刻に広げる。**詰まっているrunだけ**触る
    （シャイニングスター等の"日本語歌唱をASRが英語表記した"箇所は既に正常配置なので触らない）。chars を in-place 更新。
    ※語単位の意味対応（あいる↔I'll）は未実装（ローマ字difflibは短語/聞き間違いで脆く保留）。ここは詰まり救済のみ。"""
    import re
    eng = [w for w in (vad_words or []) if clip_t0 <= w["start"] <= end_sec and re.search(r"[A-Za-z]", w.get("word", ""))]
    if len(eng) < 2 or not chars:
        return chars
    runs, cur = [], [eng[0]]                                   # 連続する英語語を run にまとめる（語間 <1.5s）
    for w in eng[1:]:
        if w["start"] - cur[-1]["end"] < 1.5:
            cur.append(w)
        else:
            runs.append(cur)
            cur = [w]
    runs.append(cur)
    moved = 0
    for run in runs:
        rs, re_ = run[0]["start"] - clip_t0, run[-1]["end"] - clip_t0   # clip_t0基準のASR英語区間
        sel = [i for i, c in enumerate(chars) if rs - 0.3 <= c["start"] <= re_ + 0.8]   # FAがこの帯に置いた音節（英語のかな）
        if len(sel) < 2:
            continue
        idx = list(range(sel[0], sel[-1] + 1))
        C = len(idx)
        gaps = [chars[idx[m]]["start"] - chars[idx[m - 1]]["start"] for m in range(1, C)]   # **詰まっているrunだけ再配置**
        if gaps and sorted(gaps)[len(gaps) // 2] > 0.13:       # 既に十分広がっている＝FA正常（例: シャイニングスター）→触らない
            continue
        durs = [w["end"] - w["start"] for w in run]
        total = sum(durs) or 1.0
        bounds, acc = [0], 0.0                                 # 音節を英語各語へ「語の長さに比例」配分
        for dr in durs[:-1]:
            acc += dr
            bounds.append(round(C * acc / total))
        bounds.append(C)
        k = 0
        for wi, w in enumerate(run):
            cnt = bounds[wi + 1] - bounds[wi]
            for j in range(cnt):                               # 語内は均等（語の発声時間に等間隔）
                chars[idx[k]]["start"] = (w["start"] - clip_t0) + (w["end"] - w["start"]) * (j + 0.5) / max(1, cnt)
                k += 1
        moved += C
    chars.sort(key=lambda c: c["start"])
    for k in range(1, len(chars)):                             # 単調性
        if chars[k]["start"] <= chars[k - 1]["start"]:
            chars[k]["start"] = chars[k - 1]["start"] + 0.04
    if log is not None and moved:
        log.append(f"[英語ASR再配置] 英語{len(runs)}区間・{moved}音をASR単語時刻へ配置（日本語FAの前詰めを回避）")
    return chars


def _verify_alignment_abs(score: SongScore, vocals_wav: str, origin_abs: float, tempo: float,
                          section_start_abs: float | None = None, log: list | None = None,
                          tol: float = 0.2) -> tuple[float | None, bool]:
    """【絶対検算ガード】スコア音符の**絶対時刻**が、フル実歌声の実オンセットに合うかを"元音源の時間軸"で検算。
    自己導出フレームでなく**絶対基準**で測る＝今回の取り違え（自己整合フレームでの誤検算）を機械的に防ぐ。3条件で判定:
      ① origin が区間検出開始(section_start_abs)から大きく乖離=整列破綻（大ズレ検出）
      ② ±1s 走査で実オンセットに最も合う lag（|lag|>tol=origin微ズレ。≈0なら整合）
      ③ 最良一致率が低い=整列精度不足
    戻り: (微補正 lag, ok)。ok=False は★（origin/整列が要確認）。"""
    import librosa
    try:
        y = load_wav(Path(vocals_wav)).astype(np.float32)      # フル・絶対時間軸
        on = np.array(librosa.onset.onset_detect(y=y, sr=SR, units="time", backtrack=True))
        ns = sorted(score.vocal_notes, key=lambda n: n.start_tick)
        if len(on) < 8 or len(ns) < 8:
            return None, True
        note_abs = np.array([origin_abs + n.start_tick / Q * 60.0 / tempo for n in ns])
        grid = np.arange(-1.0, 1.0001, 0.01)                   # ±1s を 10ms 刻みで走査
        hits = [float(np.mean(np.abs((note_abs + lag)[:, None] - on[None, :]).min(axis=1) < 0.05)) for lag in grid]
        best = float(grid[int(np.argmax(hits))])
        hit_best = max(hits)
        hit0 = float(np.mean(np.abs(note_abs[:, None] - on[None, :]).min(axis=1) < 0.05))
        far = section_start_abs is not None and abs(origin_abs - section_start_abs) > 2.5  # ①大ズレ
        low = hit_best < 0.55                                                              # ③精度不足
        ok = (abs(best) <= tol) and (not far) and (not low)
        if log is not None:
            warn = []
            if far:
                warn.append(f"origin が区間開始{section_start_abs:.1f}sから{origin_abs - section_start_abs:+.1f}s乖離=整列破綻疑い")
            if low:
                warn.append(f"最良一致率{hit_best:.0%}が低い=整列精度不足")
            if abs(best) > tol and not far:
                warn.append(f"音符を{best:+.2f}sずらすと最も合う=origin微ズレ")
            log.append(f"[絶対検算] origin={origin_abs:.3f}s 区間開始{('%.1f' % section_start_abs) if section_start_abs is not None else '?'}s "
                       f"lag0一致{hit0:.0%}/最良lag{best:+.2f}s({hit_best:.0%}) → "
                       + ("OK（絶対基準で整合）" if ok else "★要確認: " + " / ".join(warn)))
        return best, ok
    except Exception as ex:
        if log is not None:
            log.append(f"[絶対検算] 実行失敗: {type(ex).__name__}（検算スキップ）")
        return None, True


def verify_head_lyrics(vad_full: list, clip_t0: float, lyrics_kana: str,
                       window_sec: float = 4.0) -> tuple[bool, str]:
    """【割り当て検証】区間の頭で「実際に歌われている言葉（ASR）」と「割り当てた歌詞の先頭」が一致するか。

    整列は "旋律に歌詞を順番に貼る" だけなので、区間が別のフレーズを指していても機械的には成立してしまう
    （実測事故: サビの歌詞をBメロの音に貼り、vprは「シャイニングスター」・音は「悩むのはもうやめた」）。
    ASR は既に取得済みなので、**貼った結果と実音を突き合わせる**ことで、原因が何であれこの型の事故を検出できる。
    戻り: (一致したか, 説明文)。
    """
    try:
        import difflib
        import re as _re

        import pykakasi
        kks = pykakasi.kakasi()

        def to_romaji(s: str) -> str:
            """かなも英字も **ローマ字** に正規化して比較可能にする。
            ASR は英語詞を "Shining star" とラテン文字で返す一方、歌詞は「しゃいにんぐすたあ」とかな。
            どちらか片方に寄せないと、正しい区間でも別物と誤判定する（実測・2026-07-26）。"""
            out = "".join(x.get("hepburn", "") or x.get("orig", "") for x in kks.convert(s))
            return _re.sub(r"[^a-z]", "", out.lower())

        heard_raw = "".join(w["word"] for w in (vad_full or [])
                            if clip_t0 - 0.3 <= w["start"] < clip_t0 + window_sec)
        heard = to_romaji(heard_raw)
        assigned = to_romaji(lyrics_kana)
        if len(heard) < 6 or len(assigned) < 6:
            return True, ""                                    # 材料不足では判定しない（誤検知を出さない）
        n = min(len(heard), len(assigned), 40)                 # 頭の同じ長さぶんで比較
        ratio = difflib.SequenceMatcher(None, heard[:n], assigned[:n], autojunk=False).ratio()
        if ratio >= 0.40:
            return True, f"区間頭の一致 {ratio:.0%}（実音「{heard_raw[:14]}」/ 歌詞「{lyrics_kana[:14]}」）"
        return False, (f"★区間頭が不一致（一致 {ratio:.0%}）: 実際に歌っているのは「{heard_raw[:16]}」なのに、"
                       f"割り当てた歌詞は「{lyrics_kana[:16]}」。**区間が別のフレーズを指している**疑い")
    except Exception:
        return True, ""                                        # 検証自体の失敗で本処理を止めない


def _section_bounds_by_lyrics(vad_full: list, lyrics_kana: str, log: list | None = None) -> tuple[float, float] | None:
    """【区間境界を決定論で確定】既知の歌詞(かな)を、フル文字起こし(かな化)に **difflib 照合**し、その歌詞が歌われている
    [start_sec, end_sec] を返す。LLMの盲目推定（崩れたASRから当てずっぽう＝取り違えの原因）でなく"正解歌詞との一致"で
    境界を決める＝崩れたASRにも部分一致で頑健・**再現的（temperature非依存）**。一致が弱ければ None（LLM/従来へ委ねる）。"""
    import difflib

    import pykakasi
    kks = pykakasi.kakasi()
    to_hira = lambda s: "".join(x["hira"] for x in kks.convert(s))
    ac, at = [], []                                            # ASR をかな1文字列に展開（各かなに時刻）
    for w in vad_full:
        h = to_hira(w["word"])
        for k, ch in enumerate(h):
            if ch.strip():
                ac.append(ch)
                at.append(w["start"] + (w["end"] - w["start"]) * k / max(1, len(h)))
    _kana_h = lambda c: chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c   # カタカナ→ひらがな（1文字変換）
    lk = [_kana_h(c) for c in lyrics_kana if ("ぁ" <= c <= "ん") or ("ァ" <= c <= "ヶ") or c == "ー"]  # 既にかな＝pykakasiで再変換しない（膨張バグ回避）
    if len(ac) < 5 or len(lk) < 5:
        return None
    sm = difflib.SequenceMatcher(None, lk, ac, autojunk=False)
    mt = sorted(at[b.b + k] for b in sm.get_matching_blocks() if b.size > 0 for k in range(b.size))
    if len(mt) < 3:
        return None
    clusters, cur = [], [mt[0]]                                # 一致時刻を 8s ギャップでクラスタ化＝曲全体に散った誤一致を分離
    for t in mt[1:]:
        if t - cur[-1] <= 8.0:
            cur.append(t)
        else:
            clusters.append(cur)
            cur = [t]
    clusters.append(cur)
    main = max(clusters, key=len)                              # 最も一致が密なクラスタ＝本体の区間（共通かなの散発一致に引っ張られない）
    start_sec, end_sec = main[0], main[-1]
    if log is not None:
        weak = "★一致率が低く区間がズレている可能性" if sm.ratio() < 0.30 else ""
        log.append(f"[区間=歌詞照合] 既知歌詞{len(lk)}字をASR{len(ac)}字へ決定論照合（一致率{sm.ratio():.0%}・最密クラスタ{len(main)}点）→ "
                   f"{start_sec:.1f}〜{end_sec:.1f}s（LLM非依存・temp無関係）{weak}")
    return (start_sec, end_sec) if end_sec > start_sec + 1.0 else None


def align_section(section: dict, lyrics_kana: str, *, pitch_correct: bool = False) -> SongScore:
    """工程4: section の旋律＋歌詞を整列し SongScore を返す。
    音声参照=WhisperX**セグメント分割**強制アラインメント（語境界~5秒分割でFA→長区間の圧縮回避、
    音符長=エネルギー基準で休符跨がず＋最終音延長）→（失敗時）音声グラウンディング→記号整列の順。MIDI=記号整列。
    pitch_correct: 音声参照の音高をLLMで微修正（修正UCのみTrue。初回はFalse=純プログラム）。
    """
    st = section
    log = st["log"]
    score = None
    # ★**歌詞のかなは型を通ったもの**（2026-10-02・案B）。かな以外が来たら不具合として止める（読み手は落とさない・規約「組み立ては補正しない」）
    from song_lyrics import kana_text_problems              # noqa: E402
    if bad := kana_text_problems(lyrics_kana):
        from song_fix import AssemblyError
        raise AssemblyError("整列に、歌詞のかなの型を通ったはずの歌詞が来なかった: " + "／".join(bad))
    # ★**実測が LLM の決めた区間・基準を上書きしたら、そのことを返す**（2026-09-27・目録の道具-10・(d) 残す）。
    #   上書きは続ける（歌詞の照合が当て推量に勝つ）が、`align_vocal` の返事に載せてコーディネーターへ見せる。
    # 区間を用意したとき機械が区間を変えたこと（`section_from_audio` の間奏除去・2026-10-03）も同じ返事に出す（整列し直すたびに出る）
    st["overrides"] = list(st.get("section_changes") or [])
    st["assign_findings"] = []                             # 割り当ての過不足（目録の道具-11）
    _llm_bounds = (st.get("start_sec"), st.get("end_sec"))
    if st["kind"] == "audio":
        vocals_wav = st.get("vocals_wav")
        if vocals_wav and Path(vocals_wav).exists():       # 最優先: WhisperX セグメント分割強制アラインメント
            clip_t0 = max(0.0, st["abs_t0"] - 1.0)         # 暫定（区間検出LLM由来＝取り違えうる）。下で歌詞照合により確定
            end_sec = st["end_sec"]
            vad_full = _transcribe_vad(vocals_wav, 0.0, 1e9, log=log)   # フル文字起こし（section_from_audio とキャッシュ共用）
            refined = _section_bounds_by_lyrics(vad_full, lyrics_kana, log=log) if vad_full else None
            if refined:                                     # 物理下限ガード: Nモーラの歌唱には最低 N×0.15s 必要。
                from sofa_align import kana_to_moras        # 繰り返し歌詞×低品質ASRで最密クラスタが1コーラスに収縮し、
                need = len(kana_to_moras(lyrics_kana)) * 0.15   # 全歌詞を短窓へ押し込む事故の遮断（T12: 12345で461モーラ→47s窓＝9音/秒）
                if refined[1] - refined[0] < need:
                    log.append(f"[区間=歌詞照合] ★棄却: 照合区間{refined[1] - refined[0]:.1f}sは歌詞"
                               f"{len(kana_to_moras(lyrics_kana))}モーラの物理下限{need:.1f}sより短い"
                               f"（繰り返し歌詞でクラスタ収縮の疑い）→ 区間特定(LLM)の値を使用")
                    refined = None
            if refined:                                    # 【区間確定】既知歌詞をASRに照合した実start/endで上書き＝LLM盲目推定の取り違えを排除
                # 前方延長は「ASRが歌い出しを取りこぼした」補償。連続歌唱の曲では**前のフレーズに食い込む**ため
                # 上限を小さく（6→2s）し、さらに **LLMが特定した区間開始より前へは絶対に延ばさない**
                # （実測事故: サビ53.9s開始なのに46.2→40.2sまで遡り、Bメロの音にサビの歌詞を貼った・2026-07-26）
                _llm_start = float(st.get("start_sec") or 0.0)
                _s0 = _extend_start_by_voice(vocals_wav, refined[0], cap=2.0)
                if _llm_start > 0 and _s0 < _llm_start - 0.5:
                    log.append(f"[区間始端] ★前方延長を抑制: {_s0:.1f}s は区間特定(LLM)の開始 {_llm_start:.1f}s より前"
                               f"（前のフレーズに食い込む恐れ）→ {max(refined[0], _llm_start):.1f}s を採用")
                    _s0 = max(refined[0], _llm_start)
                if refined[0] - _s0 > 0.3:
                    log.append(f"[区間始端] 有声追跡で {refined[0]:.1f}→{_s0:.1f}s に前方延長（ASRの冒頭取りこぼし補償）")
                clip_t0 = max(0.0, _s0 - 0.5)              # 歌い出し直前から（先頭音を取りこぼさない小マージン）
                _e0 = _extend_end_by_voice(vocals_wav, refined[1])   # ASR末尾取りこぼし対策: 歌声が続く限り粗延長（間奏で停止）
                end_sec = _energy_end(vocals_wav, _e0)     # →エネルギー基準の精密終端（伸ばし終わりを切らない）
                if _e0 - refined[1] > 1.0:
                    log.append(f"[区間終端] 有声追跡で {refined[1]:.1f}→{_e0:.1f}s に延長（ASRの末尾取りこぼし補償）→精密終端{end_sec:.1f}s")
                st["abs_t0"] = _s0                         # 絶対検算の基準も（前方延長込みの）実始端に同期
                if _llm_bounds[0] is not None:
                    st["overrides"].append(
                        f"区間特定（LLM）の {float(_llm_bounds[0] or 0):.1f}〜{float(_llm_bounds[1] or 0):.1f}秒を、"
                        f"歌詞の照合で {_s0:.1f}〜{end_sec:.1f}秒に置き換えた（実際に歌っている位置）")
            ok_head, head_msg = verify_head_lyrics(vad_full, clip_t0, lyrics_kana)  # 貼る前に「実音 vs 歌詞」を照合
            if head_msg:
                log.append(f"[区間頭の検証] {head_msg}")
            st["head_mismatch"] = "" if ok_head else head_msg   # align_vocal が戻り値でLLMに伝える（検問も拾う）
            vad = ([w for w in vad_full if clip_t0 <= w["start"] <= end_sec] if vad_full   # 区間内に絞る
                   else _transcribe_vad(vocals_wav, clip_t0, end_sec, log=log))
            # 【標準=x31レシピ】歌用アライナSOFAで一括整列（分割不要・伸ばし音に強い）→ 音程pyin実測＋オクターブ補正
            # → min-gap（発音可能間隔の保証）→ 英語のみen-FAで上書き。UST正解比 RMS366→90ms/p90 657→123ms（study.md）。
            chars = None
            try:
                from sofa_align import sofa_align
                chars = sofa_align(vocals_wav, lyrics_kana, clip_t0, end_sec, log=log, vowel_start=True)
            except Exception as ex:
                log.append(f"[SOFA] 例外 {type(ex).__name__}: {ex} → 従来FA系へフォールバック")
            if chars:
                mids = _pyin_midis(vocals_wav, clip_t0, [c["start"] for c in chars], [c["end"] for c in chars])
                for c, md in zip(chars, mids):
                    c["midi"] = md
                chars = _fix_octave_outliers(chars, vocals_wav, clip_t0, log=log)   # サブハーモニック誤りを音響証拠で補正
                chars = _min_gap_spread(chars, g=0.075, log=log)
                chars = _realign_english_by_enfa(chars, vad, vocals_wav, clip_t0, end_sec,
                                                 st.get("eng_phrases") or [], log=log,
                                                 tail_to_next=True, snap_head_onset=True)
            else:                                          # 【フォールバック】従来の発話FA系（SOFA不成立時）
                chars = (_segmented_forced_align(vocals_wav, lyrics_kana, vad, clip_t0, end_sec, log=log,
                                                 eng_phrases=st.get("eng_phrases") or [])
                         if vad else None)
                if chars is None:                          # vad無し等→従来の一括FAにフォールバック
                    chars = _forced_align(vocals_wav, lyrics_kana, clip_t0, end_sec, log=log)
                if chars:
                    # 統合整列: vad粗アンカー＋自動un-cram（発話FAは英語/歌唱で詰めるため。SOFA経路では不要=有害）
                    chars = _integrate_alignment(chars, vad, clip_t0, end_sec, vocals_wav, log=log)
                    # 英語区間の保険: セグメント分割FAが英語を分離配置済みなら何もしない
                    if not any("en-FA語時刻へ" in ln for ln in log):
                        chars = _realign_english_by_asr(chars, vad, clip_t0, end_sec, log=log)   # 詰まり均し
                        chars = _realign_english_by_enfa(chars, vad, vocals_wav, clip_t0, end_sec,
                                                         st.get("eng_phrases") or [], log=log)   # en-FA再配置
            if chars:
                if pitch_correct:                          # LLM音高補正（OFFで純プログラム＝pyin f0のみ）
                    chars = _correct_forced_pitches(vocals_wav, chars, clip_t0,
                                                    st["song_title"], st["section_spec"], log=log,
                                                    overrides=st["overrides"])   # 補正しなかったら返事に出す（2026-10-02）
                # 音符長=実唱エネルギー基準（休符跨がない）＋最終音延長。再整列/onsetスナップは廃止（FA実測のまま）
                score = _build_score_energy(chars, st["tempo"], st["song_title"], st["section_spec"],
                                            vocals_wav, clip_t0, log=log)
                origin = clip_t0 + chars[0]["start"]                # スコア tick0 の絶対秒（暫定）
                lag, ok = _verify_alignment_abs(score, vocals_wav, origin, st["tempo"],   # 絶対基準で検算（大ズレ/微ズレ/精度を判定）
                                                section_start_abs=st.get("abs_t0"), log=log)
                if lag is not None and 0.08 < abs(lag) <= 0.9:     # 微ズレ→origin を実オンセットへ吸着補正し再検算（系統的な遅れを自動是正）
                    origin += lag
                    log.append(f"[絶対検算] origin を {lag:+.2f}s 補正（自己フレームでなく絶対基準へ吸着）→ {origin:.3f}s")
                    st["overrides"].append(f"歌の基準（曲頭）を実際の歌い出しへ {lag:+.2f}秒 ずらした（{origin:.2f}秒）")
                    lag, ok = _verify_alignment_abs(score, vocals_wav, origin, st["tempo"],
                                                    section_start_abs=st.get("abs_t0"), log=log)
                if not ok:                                          # 大ズレ等は★で明示（黙って進めない＝取り違え防止）
                    log.append("[絶対検算] ★整列が絶対基準と不整合のまま。LLM/人手で位置を要修正（このまま確定しない）。")
                st["start_sec"] = origin                            # 【唯一の基準】伴奏/切出/mix/再ロードは全部これを読む
                score.origin_abs_sec = origin                       # scoreに焼き込み＝永続化（再導出しない）
                end_tick = max(n.start_tick + n.duration_tick for n in score.vocal_notes)
                st["end_sec"] = origin + end_tick / Q * 60.0 / st["tempo"]
                # 【隙間の明示化】区間頭→歌い出しの僅差を score の第1音tickから算出し st に保持（下流は再検出せずこれを引き継ぐ）
                st["vocal_lead_sec"] = min(n.start_tick for n in score.vocal_notes) / Q * 60.0 / st["tempo"]
                st["vocal_onset_sec"] = st["start_sec"] + st["vocal_lead_sec"]   # 歌い出しの絶対秒（明示）
                log.append(f"[隙間] 区間頭{st['start_sec']:.3f}s → 歌い出し{st['vocal_onset_sec']:.3f}s（前奏{st['vocal_lead_sec']:.3f}s）"
                           f"＝scoreの第1音tickから算出・st['vocal_lead_sec']に保持（再検出しない）")
                # 【整合性ログ】区間長==スコア長 でなければ伴奏が末尾に届かない（不変条件・違反は★で追える）
                _score_span = end_tick / Q * 60.0 / st["tempo"]
                _sec_span = st["end_sec"] - st["start_sec"]
                log.append(f"[整合:区間↔スコア] スコア長{_score_span:.2f}s 区間長{_sec_span:.2f}s 歌い出し{st['start_sec']:.2f}s "
                           f"終端{st['end_sec']:.2f}s → {'OK' if abs(_score_span - _sec_span) < 0.1 else '★不一致(伴奏が末尾に届かない・要調査)'}")
                _last_on = origin + max(n.start_tick for n in score.vocal_notes) / Q * 60.0 / st["tempo"]
                if end_sec - _last_on > 5.0:                   # 歌唱が続くのに歌詞が尽きている＝抽出漏れの検知
                    log.append(f"[整合:歌詞↔歌唱] ★歌唱は{end_sec:.1f}sまで続くが最終音の開始は{_last_on:.1f}s"
                               f"＝**歌詞の後半欠落の疑い**（『1番』の範囲をサビで切りすぎ? fetch_section_lyricsを広めに再実行推奨）")
                _how = "SOFA歌用アライナ(x31標準)" if any(ln.startswith("[SOFA] 整列") for ln in log) else "WhisperXセグメント分割FA(フォールバック)"
                log.append(f"[5.割り当て] {_how}"
                           f"{'＋Gemini音高補正' if pitch_correct else '（音高=pyin実測・純プログラム）'} {len(chars)}文字")
        if score is None and vocals_wav and Path(vocals_wav).exists():  # 次善: 音声グラウンディング
            score = align_full_lyrics_audio(vocals_wav, st["sl"], st["ref_pitch"], st["onsets"], st["sung"],
                                            lyrics_kana, tempo=st["tempo"], song_title=st["song_title"],
                                            section_spec=st["section_spec"], log=log,
                                            overrides=st["overrides"],   # 使わなかったら返事に出す（2026-10-02）
                                            findings=st["assign_findings"])   # 割り当ての落ち・重なり（2026-10-03）
            if score is not None:
                log.append("[5.割り当て] 音声グラウンディング採用（FA失敗）")
    if score is None:                                      # フォールバック / MIDI: 記号ベース整列
        score = align_full_lyrics_jp(st["ref_pitch"], st["sung"], st["sl_breaks"], lyrics_kana,
                                     song_title=st["song_title"], starts=st["onsets"], tempo=st["tempo"], log=log,
                                     findings=st["assign_findings"])
    score = score.model_copy(update={"title": f"{st['song_title']}_{st['section_spec']}"})
    # ★**調声はここで当てる**（2026-09-02）。カバーの整列は経路が枝分かれしていて
    #   （音声グラウンディング／強制アラインメント／記号ベース／MIDI）、
    #   **枝ごとに足していたら MIDI 経路を2回続けて取りこぼした**——
    #   `align_cover` の3箇所 → `align_reference` の3箇所と直したのに、実機ではまだ 0/97音だった。
    #   **`align_section` が全部の出口**なので、ここ1箇所に置けば漏れない。
    #   音高もタイミングも変えないので、原曲への忠実さは損なわない。
    apply_by_gap(score.vocal_notes, tempo=score.tempo)
    log.append("[5.最終ノート(かな:音名@拍)] " +
               " ".join(f"{n.lyric}:{spell_pitch(n.key)}@{round(n.start_tick / Q, 2)}"
                        for n in score.vocal_notes))
    return score


#: 区間の頭より前の値を**履歴ごと**持ち込む CC（RPN/NRPN の選択とデータ入力）。
#: 最後の値だけにすると順序が崩れ、ピッチベンドの幅などが別の番号へ入る。
_CC_REPLAY_ALL = {6, 38, 96, 97, 98, 99, 100, 101}


def write_cover_accompaniment_midi(midi_path: Path, melody_channel: int, start_sec: float, end_sec: float,
                                   tempo_bpm: float, out_mid: Path) -> tuple[float, int]:
    """元MIDIの**主旋律ch以外**を [start_sec, end_sec) で切り出し、伴奏 MIDI（accompaniment.mid の形）を書く。

    - 時間の原点＝区間の頭（楽譜の tick 0）。テンポは `tempo_bpm`（楽譜のテンポ）1つで、各イベントの
      **実秒を保ったまま** tick へ直す（元のテンポマップどおりに鳴り、伴奏編集の道具＝`accomp_edit` が
      楽譜と同じ拍の座標で読める）。tpb=480・チャンネルごとに1トラック（`load_accomp` が1トラック=1楽器で読むため）。
    - 区間の頭より前のプログラムチェンジ・CC・ピッチベンドは tick 0 に持ち込む（音色・音量・ペダルをそのまま）。
      区間の頭で鳴り続けている音（ドラム以外）は tick 0 から鳴らし直し、区間の終わりで切る。
    - ドラム（ch9）・CC・プログラムチェンジ・システムエクスクルーシブは区間内もそのまま。

    戻り値は (伴奏の実際の終わり（区間頭からの秒）, 区間内のノート数)。終わりは
    min(区間の終わり, 元MIDIの伴奏の最後の音の終わり＋0.4秒)——以前の簡易合成と同じ長さの決め方。
    """
    mf = mido.MidiFile(str(midi_path))
    names: dict[int, str] = {}                             # ch → 元のトラック名
    for tr in mf.tracks:
        nm = next((m.name for m in tr if m.type == "track_name" and m.name.strip()), None)
        for m in tr:
            if nm and hasattr(m, "channel") and m.channel not in names:
                names[m.channel] = nm.strip()
    pre: dict[int, list[tuple[float, object, mido.Message]]] = {}   # ch → 区間前の状態（時刻, 重複キー, msg）
    held: dict[tuple[int, int], int] = {}                  # 区間の頭の時点で鳴っている音 → vel
    body: list[tuple[float, mido.Message]] = []            # 区間内（区間頭からの秒, msg）
    sysex_pre: list[mido.Message] = []
    sysex_in: list[tuple[float, mido.Message]] = []
    last_off, t = 0.0, 0.0
    lo, hi = start_sec - 1e-6, end_sec - 1e-6              # 秒の足し算の丸めで境界の音が区間の外へ落ちないように
    for msg in mf:                                         # 反復は秒（テンポマップ込み）
        t += msg.time
        if msg.type == "sysex":
            if t < lo:
                sysex_pre.append(msg)
            elif t < hi:
                sysex_in.append((max(0.0, t - start_sec), msg))
            continue
        ch = getattr(msg, "channel", None)
        if ch is None or ch == melody_channel:             # 主旋律はミクが歌う＝伴奏から抜く
            continue
        is_off = msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)
        if is_off:
            last_off = max(last_off, t)
        if t < lo:
            if msg.type == "note_on" and not is_off:
                held[(ch, msg.note)] = msg.velocity
            elif is_off:
                held.pop((ch, msg.note), None)
            elif msg.type == "control_change":
                evs = pre.setdefault(ch, [])
                key = ("cc", msg.control, len(evs)) if msg.control in _CC_REPLAY_ALL else ("cc", msg.control)
                evs.append((t, key, msg))
            elif msg.type in ("program_change", "pitchwheel"):
                pre.setdefault(ch, []).append((t, (msg.type,), msg))
        elif t < hi:
            body.append((max(0.0, t - start_sec), msg))
    stop = max(0.0, min(end_sec, last_off + 0.4) - start_sec)
    span = end_sec - start_sec

    spb = 60.0 / tempo_bpm
    def tick(sec: float) -> int:
        return int(round(sec / spb * Q))

    tracks: dict[int, list[tuple[int, int, mido.Message]]] = {}   # ch → (tick, 並び順, msg)
    for ch, evs in pre.items():                            # 区間前の状態: 同じキーは最後の値だけ（RPN は全部）を元の順で
        last: dict[object, tuple[float, mido.Message]] = {}
        for tt, key, m in evs:
            last[key] = (tt, m)
        for tt, m in sorted(last.values(), key=lambda x: x[0]):
            tracks.setdefault(ch, []).append((0, 0, m.copy(time=0)))
    on: set[tuple[int, int]] = set()
    for (ch, note), vel in held.items():
        if ch != 9:                                        # 伸ばしている音は頭から鳴らし直す（打楽器は叩き直さない）
            tracks.setdefault(ch, []).append((0, 1, mido.Message("note_on", channel=ch, note=note, velocity=vel)))
            on.add((ch, note))
    n_notes = len(on)
    for sec, m in body:
        k = tick(sec)
        if m.type == "note_on" and m.velocity > 0:
            on.add((m.channel, m.note))
            n_notes += 1
            tracks.setdefault(m.channel, []).append((k, 1, m.copy(time=0)))
        elif m.type in ("note_on", "note_off"):
            if (m.channel, m.note) in on:
                on.discard((m.channel, m.note))
                tracks.setdefault(m.channel, []).append((k, 1, mido.Message("note_off", channel=m.channel, note=m.note)))
        else:
            tracks.setdefault(m.channel, []).append((k, 1, m.copy(time=0)))
    for ch, note in sorted(on):                            # 区間の終わりで切る
        tracks[ch].append((tick(span), 2, mido.Message("note_off", channel=ch, note=note)))

    out = mido.MidiFile(ticks_per_beat=Q)
    cond = mido.MidiTrack()
    cond.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    evs = [(0, m) for m in sysex_pre] + [(tick(s), m) for s, m in sysex_in]
    prev = 0
    for k, m in evs:
        cond.append(m.copy(time=k - prev))
        prev = k
    cond.append(mido.MetaMessage("end_of_track", time=max(0, tick(stop) - prev)))   # 鳴らす長さ＝伴奏の終わり
    out.tracks.append(cond)
    for ch in sorted(tracks):
        mt = mido.MidiTrack()
        mt.append(mido.MetaMessage("track_name", name=names.get(ch) or ("Drums" if ch == 9 else f"ch{ch + 1}"), time=0))
        prev = 0
        for k, _, m in sorted(tracks[ch], key=lambda x: (x[0], x[1])):
            mt.append(m.copy(time=k - prev))
            prev = k
        out.tracks.append(mt)
    out_mid.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(out_mid))
    return stop, n_notes


def accompaniment_for_section(section: dict, *, cover_sec: float = 0.0,
                              mid_out: Path | None = None) -> np.ndarray:
    """工程5: section の区間に対応する伴奏 np を返す。MIDI=元MIDI伴奏を SoundFont でレンダ、音声=分離実楽器のスライス。
    伴奏が無い（accomp_wav なし／実質無音アカペラ）の場合は空配列＝伴奏なしとして扱う。
    cover_sec>0: 伴奏が**少なくとも cover_sec 秒（=スコア全長）をカバー**するよう終端を延長（末尾の伸ばしに伴奏が欠けないように）。
    mid_out（MIDI のみ）: 切り出した伴奏 MIDI の置き場（曲フォルダの accompaniment.mid）。None なら一時ファイル。

    ★MIDI の伴奏は **SoundFont（`sf2_render.render_midi`）で鳴らす**（2026-09-30・本人「カバーも soundfont で」）。
      以前は numpy の簡易合成（正弦波＋倍音・ドラムはノイズ）だった。FluidSynth／SoundFont が無ければ例外で止まる
      （オリジナル曲の `_render_accomp` と同じく、代わりの音では鳴らさない）。
    """
    st = section
    if st["kind"] == "midi":
        import sf2_render
        start_sec = _tick_to_sec(st["base_tick"], st["tpb"], st["tmap"])
        end_sec = _tick_to_sec(st["end_tick"], st["tpb"], st["tmap"])
        if cover_sec:
            end_sec = max(end_sec, start_sec + cover_sec)
        with tempfile.TemporaryDirectory() as td:
            mid = Path(td) / "accompaniment.mid"
            stop, n_notes = write_cover_accompaniment_midi(Path(st["midi_path"]), st["melody_channel"],
                                                           start_sec, end_sec, st["tempo"], mid)
            n = max(0, int((start_sec + stop) * SR) - int(start_sec * SR))
            acc = np.zeros(0)
            if n_notes and n:
                wav = Path(td) / "accompaniment.wav"
                sf2_render.render_midi(mid, wav, gain=0.8)   # 鳴らせなければここで例外（MIDI も曲フォルダへ置かない）
                acc = load_wav(wav)
            if mid_out is not None:                        # 鳴らせてから置く（wav と mid の食い違いを残さない）
                Path(mid_out).parent.mkdir(parents=True, exist_ok=True)
                Path(mid_out).write_bytes(mid.read_bytes())
        if not len(acc):
            st["log"].append("[6.伴奏] 区間に主旋律以外の音が無い→伴奏なし")
            return acc
        acc = acc[:n] if len(acc) >= n else np.concatenate([acc, np.zeros(n - len(acc))])
        st["log"].append(f"[6.伴奏] 元MIDIの主旋律ch{st['melody_channel']}以外 {n_notes}音を SoundFont で"
                         f" {n / SR:.2f}s 鳴らした（{start_sec:.2f}s〜）")
        return acc
    accomp_wav = st.get("accomp_wav")                      # 音声参照: demucs インスト（SR=44100）
    if not accomp_wav or not Path(accomp_wav).exists():
        st["log"].append("[6.伴奏] 伴奏音源なし→伴奏なし（アカペラ）")
        return np.zeros(0)
    acc_full = load_wav(Path(accomp_wav))
    end_sec = max(st["end_sec"], st["start_sec"] + cover_sec) if cover_sec else st["end_sec"]  # スコア全長をカバー
    acc = acc_full[int(st["start_sec"] * SR): int(end_sec * SR)]
    if len(acc) and float(np.sqrt(np.mean(acc.astype(np.float64) ** 2))) < 1e-4:  # 実質無音→伴奏なし
        st["log"].append("[6.伴奏] 区間がほぼ無音→伴奏なし（アカペラ）")
        return np.zeros(0)
    try:                                                   # 【隙間の明示化】区間頭→初楽器音（伴奏も同じ start_sec から切る＝歌声と同基準）
        import librosa
        o = librosa.onset.onset_detect(y=acc[:int(3 * SR)].astype(np.float32), sr=SR, units="time", backtrack=True)
        st["accomp_lead_sec"] = float(o[0]) if len(o) else 0.0
        st["log"].append(f"[隙間] 区間頭{st['start_sec']:.3f}s → 初楽器音 +{st['accomp_lead_sec']:.3f}s（伴奏は歌声と同じ start_sec 基準で切出）")
    except Exception:
        pass
    return acc
