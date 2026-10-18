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

"""本番形プロトタイプ: score_to_vpr（LibreSVIP 骨格 ＋ aiExp 調声焼き込み ＋ 伴奏同梱）。

ステップ1: LibreSVIP で vpr 骨格（is_ai_singer=True / 日本語・伴奏があれば InstrumentalTrack 同梱）
ステップ2: sequence.json を後処理し、SongNote.tuning(VocalTuning) を per-note aiExp に写像
（§3.2。VOCALOID:AI は aiExp を尊重することを実機確認済み）。
伴奏トラックには音量バランスの初期値も焼き込む（V6 のミックスダウン一発で完成ミックスに
なるように。単位は 1/10dB＝V6 実機でフェーダー表示を確認済み・2026-08-10）。
"""
from __future__ import annotations

import json
import math
import zipfile
import wave
from pathlib import Path

from libresvip.extension.manager import plugin_manager
from libresvip.model.base import (InstrumentalTrack, Note, Project, SingingTrack,
                                  SongTempo, TimeSignature)

from song_schema import SongScore, VocalTuning

SEQ = "Project/sequence.json"

AIEXP_DEFAULT = {
    "pitchFine": 0.5, "pitchDriftStart": 0.5, "pitchDriftEnd": 0.5,
    "pitchScalingCenter": 0.5, "pitchScalingOrigin": 0.5,
    "pitchTransitionStart": 0.5, "pitchTransitionEnd": 0.5,
    "amplitudeWhole": 0.5, "amplitudeStart": 0.5, "amplitudeEnd": 0.5,
    "vibratoLeadingDepth": 0.5, "vibratoFollowingDepth": 0.5,
}

# VocalTuning フィールド → aiExp フィールド（vibrato_depth は2つに展開）
_MAP = {
    "amplitude": ["amplitudeWhole"],
    "amplitude_attack": ["amplitudeStart"],
    "amplitude_release": ["amplitudeEnd"],
    "pitch_transition_in": ["pitchTransitionStart"],
    "pitch_transition_out": ["pitchTransitionEnd"],
    "pitch_drift_in": ["pitchDriftStart"],
    "pitch_drift_out": ["pitchDriftEnd"],
    "pitch_fine": ["pitchFine"],
    # ★**`vibrato_depth` はここでは写さない**（2026-09-23）。V6 実機で測ったところ
    #   `aiExp.vibratoLeading/FollowingDepth` は**効かなかった**（深さ 0.8 でも
    #   揺れ 0.003 半音＝何もしていないのと同じ）。書き先は `_write_vibrato`。
    # ★**フォルマントは写さない**（2026-09-23）。vpr の定義にはあるが、
    #   V6 実機で**効かなかった**（0.0 と 1.0 で明るさ 2355Hz ↔ 2346Hz）。
}


def _tuning_to_aiexp(tuning: VocalTuning | None) -> dict:
    ae = dict(AIEXP_DEFAULT)
    if tuning is None:
        return ae
    for field, targets in _MAP.items():
        val = getattr(tuning, field)
        if val is not None:
            for t in targets:
                ae[t] = float(val)
    return ae


#: ビブラートの速さ（`rates` に書く値）。実測で **64 ＝ 4.9Hz**、深さを変えても速さは変わらない。
#: 人の歌のビブラートは 5〜6Hz なので既定はここ。
VIBRATO_RATE = 64

#: 揺れを**音のどこから始めるか**・揺れを掛ける**最短の長さ（秒）**。**基準は `song_tuning` の1か所**。
#: ★2026-09-27 まで、ここ（書き出し）が「揺れる部分が 0.5秒未満なら書かない」と**黙って落としていた**
#:   （目録 表情-7）。楽譜には揺れがあるのに鳴らず、歌い方の工程も知らなかった。いまは**揺れを決める側**
#:   （`song_tuning.apply_tuning`・`vocal_expression.apply_plan`）が秒で見て、ここは楽譜のとおり写す。
#:   短い音に揺れが書かれていたら、写したうえで所見に出す（`song_tuning.vibrato_findings`）。
from song_tuning import VIBRATO_FROM, VIBRATO_MIN_SEC  # noqa: E402,F401（MIN_SEC は旧い読み手のために残す）


def _write_vibrato(jn: dict, tuning: VocalTuning | None, sec_per_tick: float = 0.0) -> None:
    """**音符側の `vibrato`** に揺れを書く（2026-09-23）。**楽譜のとおり写す**（`sec_per_tick` は使わない・旧い呼び方の受け皿）。

    ★★**`aiExp.vibratoLeadingDepth/FollowingDepth` は効かない。** V6 実機で測った——
      深さ 0.8 を書いても、`isAiVibratoEnabled` を true にしても、
      **揺れは 0.003 半音**（＝測定の底）だった。つまり `song_tuning` の
      「1拍以上伸ばす音は揺らす（0.8）」は **2026-08 からずっと空振りしていた**。
      **vpr には 0.8 と書かれていた**ので、ファイルを開いても気づけなかった
      ——**入っていることと、効いていることは別**。

    効くのはこちら（`note.vibrato` の `depths` / `rates` の点列）。実測:

    | 書く値 | 0 | 16 | 32 | 64 | 96 | 127 |
    |---|---|---|---|---|---|---|
    | 揺れ（半音・片振幅）| 0.006 | 0.066 | 0.132 | 0.264 | 0.394 | 0.523 |

    **1あたり 0.00412 半音できれいに線形**。全開でも ±0.52 半音で、
    人の歌（±0.3〜0.5）とほぼ同じ幅＝**0〜1 の目盛りをそのまま 127 倍してよい**。
    実測と経緯は [accomp_articulation.md §10](../../../../docs/plan/accomp_articulation.md)。
    """
    depth = getattr(tuning, "vibrato_depth", None) if tuning else None
    if not depth or float(depth) <= 0:                     # None・0・負は「揺らさない」
        return
    dur = int(jn.get("duration", 0))
    span = int(dur * (1.0 - VIBRATO_FROM))
    if span <= 0:                                          # 短すぎる音（点が重なる＝書けない）
        return
    v = max(1, min(127, round(float(depth) * 127)))        # 範囲外は端に丸める
    jn["vibrato"] = {
        "type": 1, "duration": span,
        "depths": [{"pos": 0, "value": v}, {"pos": span, "value": v}],
        "rates": [{"pos": 0, "value": VIBRATO_RATE}, {"pos": span, "value": VIBRATO_RATE}],
    }


#: 歌でベンドを書くときの幅（半音）。**`pitchBendSens`（PBS）で指定する**。
#: 伴奏の RPN と違い、vpr は**パート単位・曲の途中でも変えられる**（実測で確認）。
VOCAL_BEND_RANGE = 12

#: 歌の音域は狭いので、**`portamento` と `gliss` は使わない**（下の注記）。
_VOCAL_SKIP = {"gliss", "portamento"}


def _where(tick: int, bar_ticks: int) -> str:
    """所見に書く位置。**表と同じ書き方**（`12:1.50`・拍は 0 始まり）。小節の長さが分からなければ tick。"""
    if bar_ticks <= 0:
        return f"{tick}tick"
    bar, rest = divmod(int(tick), bar_ticks)
    return f"{bar + 1}:{rest / 480:.2f}"


def _write_controllers(part: dict, src_notes, warns: list[str], bar_ticks: int = 0) -> None:
    """**歌の奏法**を vpr の `controllers` へ書く（2026-09-23）。

    ★**伴奏と同じカタログを使う**（`articulation.py`）。同じ「しゃくり」を2か所に
      定義すると必ずずれるので、**違うのは書き出し先だけ**にする。

    ★**次の音へ滑る系は書かない。** V6 は放っておいても **100ms で次の音へ滑る**
      （2026-09-23 実測）。そこへ PIT を重ねると**二重に滑り、戻しの位置で音程が跳ねる**
      （実測で +2.2 → +0.7 のつまずきが出た）。**渡りは `pitch_transition_*` の仕事**。

    ★**戻しは次の音の直前**。`note_off` と同時に 0 へ戻すと**リリースの余韻が跳ね上がる**
      （伴奏の MIDI で踏んだのと同じ）。

    描かなかった奏法は `warns` へ**位置つきで**積む。★2026-09-27 まで、積んだ箱（`_artic_warns`）を
      **誰も読んでいなかった**（目録 表情-8）。いまは `score_to_vpr` が印字し、`findings` で呼び側へ返す。
    """
    import articulation as artic                           # noqa: E402（循環回避）

    pit: list[dict] = []
    used = False
    for i, sn in enumerate(src_notes):
        name = (getattr(sn, "artic", "") or "").strip()
        if not name:
            continue
        at = _where(sn.start_tick, bar_ticks)
        if name in _VOCAL_SKIP:
            warns.append(f"{at}『{sn.lyric}』の『{name}』は描かなかった"
                         f"（歌では使わない。V6 が自前で次の音へ滑るので二重になる）")
            continue
        # ★**声を楽器で代用しない**（2026-09-23）。人の声はカタログの楽器ではないので、
        #   縛りは `vocal_artic.VOCAL_NAMES` に任せ、ここでは**形だけ**引く。
        # ★深さは歌の崩しの型が必ず書かせ、組み立て（`vocal_artic.apply_plan`）が楽譜に写す（2026-10-04）。空は不具合として止める
        #   （以前は空をカタログの基準の形で描いた。古い楽譜の空は変換 `convert_record.py` が当時の鳴り方の深さを書き込む）
        depth = getattr(sn, "artic_depth", None)
        if depth is None and artic.has_depth(name):
            from song_fix import AssemblyError              # noqa: E402
            raise AssemblyError(f"{at}『{sn.lyric}』の歌の崩し『{name}』に深さが無い——歌の崩しの型が必ず書かせているはずの形でない")
        pts = artic.shape_of(name, depth)
        if pts is None:
            warns.append(f"{at}『{sn.lyric}』の知らない奏法『{name}』は描かなかった")
            continue
        used = True
        for tick, st in artic.expand(pts, sn.duration_tick):
            pit.append({"pos": sn.start_tick + tick,
                        "value": int(max(-8192, min(8191, round(st / VOCAL_BEND_RANGE * 8192))))})
        end = sn.start_tick + sn.duration_tick
        if artic.ends_bent(pts):
            nxt = src_notes[i + 1].start_tick if i + 1 < len(src_notes) else None
            back = end + 240 if nxt is None else min(end + 240, nxt - 1)
            pit.append({"pos": max(end, back), "value": 0})
        else:
            pit.append({"pos": end, "value": 0})
    if not used:
        return
    merged: dict[int, int] = {}
    for e in sorted(pit, key=lambda x: x["pos"]):
        merged[e["pos"]] = e["value"]                      # 同じ位置は後勝ち
    part["controllers"] = [
        {"name": "pitchBendSens", "events": [{"pos": 0, "value": VOCAL_BEND_RANGE}]},
        {"name": "pitchBend", "events": [{"pos": p, "value": v} for p, v in sorted(merged.items())]},
    ]


def _singing_track(title: str, notes) -> SingingTrack:
    return SingingTrack(title=title, note_list=[
        Note(start_pos=n.start_tick, length=n.duration_tick, key_number=n.key, lyric=n.lyric)
        for n in notes])


#: 伴奏フェーダーの下限。**−12 では足りない**（2026-09-20）——仮歌を実歌声の大きさに
#: 合わせたので、見積もりは −14dB 前後に落ち着く。旧 −12 のままだと**全曲がここで頭打ち**になる。
#: vpr が受け付ける範囲は −60〜+6dB（`v6_export._FADER_MIN/MAX`）なので、その内側に取る。
_GAIN_MIN, _GAIN_MAX = -40.0, 6.0


def _sung_median_db(voc, acc, sr: int) -> tuple[float, float] | None:
    """歌と伴奏の「**歌っている区間の**音量」(dB) を返す。

    ★**`v6_export._measure_lead` と同じ測り方にそろえてある**（2026-09-20）。
      以前はファイル全体の RMS で見積もり、検証は 0.25秒窓・歌区間の中央値で測っていた。
      **測り方が違うものを比べていた**ので、前奏・間奏・後奏の長さや休符の多さで
      見積もりがずれた（9曲でばらつき 1.82dB → そろえて 1.37dB）。
    """
    import numpy as np
    n = min(len(voc), len(acc))
    if n < sr:
        return None
    w = int(sr * 0.25)
    fr = [(i, i + w) for i in range(0, n - w, w)]
    if not fr:
        return None
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0   # noqa: E731
    v = np.array([rms(voc[a:b]) for a, b in fr])
    a = np.array([rms(acc[a:b]) for a, b in fr])
    sing = v > (v.max() * 0.08)                            # 歌っている窓だけ（間奏を入れると不当に低く出る）
    if sing.sum() < 4:
        return None
    db = lambda x: 20 * math.log10(x) if x > 1e-12 else -99.0                  # noqa: E731
    return db(float(np.median(v[sing]))), db(float(np.median(a[sing])))


def _accomp_gain_db(song_dir: Path) -> float:
    """伴奏トラックの初期フェーダー値（dB）。狙いは「歌声が伴奏より lead_db 前に出る」。

    仮歌（placeholder）と伴奏の音量比から算出する。**仮歌の振幅は実歌声に合わせてある**
    （`song_pipeline.PLACEHOLDER_AMP`）ので、ここは素直に比を取るだけでよい。

    ★**測って分かった残差は `mix.json` の `accomp_trim_db` に覚えてある**（2026-09-20）。
      仮歌はあくまでサイン波なので、曲ごとに 1dB 強のばらつきが残る。
      一度実歌声で測れば**その曲では以後ほぼ一定**（『加速させて』6回で幅 0.7dB）なのに、
      版を作るたびに vpr を作り直して捨てていたため、**毎回 V6 を2回起こしていた**。
      書き込むのは `v6_export._fix_balance`。
    """
    lead_db, trim_db = 4.0, 0.0
    try:                                                   # 過去に指定した音量バランス（mix.json）を踏襲
        mj = song_dir / "mix.json"
        if mj.exists():
            d = json.loads(mj.read_text(encoding="utf-8"))
            lead_db = float(d.get("vocal_lead_db", 4.0))
            trim_db = float(d.get("accomp_trim_db", 0.0))
    except Exception:
        pass
    try:
        import soundfile as sf
        ph = song_dir / "placeholder.wav"
        ac = song_dir / "accompaniment.wav"
        if ph.exists() and ac.exists():
            v, sr = sf.read(ph, dtype="float64")
            a, _ = sf.read(ac, dtype="float64")
            if v.ndim > 1:
                v = v.mean(axis=1)
            if a.ndim > 1:
                a = a.mean(axis=1)
            m = _sung_median_db(v, a, sr)
            if m is not None:
                return max(_GAIN_MIN, min(_GAIN_MAX, m[0] - m[1] - lead_db + trim_db))
    except Exception:
        pass
    return -lead_db + trim_db                              # 計測不能時: 伴奏を lead_db ぶん下げる素朴な初期値


def _fix_global_tempo(data: dict, tempo: float) -> None:
    """vpr の **global tempo** を曲の実テンポに合わせる（2026-09-01 統合）。

    LibreSVIP が書き出す vpr はテンポを2か所に持ち、**`global` はライブラリ既定の
    120BPM が有効のまま**になる。V6 はオーディオの `region` をこの global tempo で
    解釈するため、BPM88 の曲では伴奏 wav が**途中で切れて配置される**
    （実機で 21小節で切れた。歌は 24小節まであった）。

    歌のノートは tick なので影響を受けず、**伴奏だけが短くなる**形で出る。
    `isEnabled` は触らない——無効化したときの V6 の解釈が未確認なので、値を揃えるほうが安全。
    """
    g = ((data.get("masterTrack") or {}).get("tempo") or {}).get("global")
    if isinstance(g, dict):
        g["value"] = int(round(tempo * 100))


def _extend_audio_region(data: dict, accomp: Path) -> None:
    """伴奏の再生範囲を **wav の実長**まで広げる（2026-09-01 統合）。

    vpr のオーディオパートは `region` で再生範囲を持ち、LibreSVIP はこれを
    **プロジェクトの長さ（＝歌トラックの終わり）** で書く。伴奏のほうが長いと
    そこで打ち切られ、**アウトロが V6 でだけ鳴らない**（実測 103.6秒の伴奏に対し
    region が 99.4秒＝4.2秒が消えていた）。wav 自体は正しいので気づきにくい。

    **単位はミリ秒**。移植計画には「wav秒 × 960（120BPM基準）」と書いてあったが、
    実際に書き出した vpr 6本で `region.end / wav秒 = 1.000`（ms）だったので実物に合わせる。
    **短いほうへは動かさない**——広げると無音を足すことになる。
    """
    if not accomp.exists():
        return
    with wave.open(str(accomp)) as w:
        want = w.getnframes() / w.getframerate() * 1000.0
    for tr in data.get("tracks", []):
        for part in (tr.get("parts") or []):
            r = part.get("region")
            if isinstance(r, dict) and 0 < float(r.get("end") or 0) < want - 50:  # 50ms は誤差
                r["end"] = want


#: 初音ミク V6 の compID。**vpr に入っていないと V6 が声を解決できない**——
#: 起動すると「**ボイスバンクが使用できません**」（`xReplaceUnavailableVoiceDlg`）が出たまま止まり、
#: 自動操作からは「メニュー項目が押せない」としか見えないので、**読み込みが遅いのと区別がつかない**
#: （2026-09-20 に実際に踏んだ。原因が分からないまま 271秒×2回を捨てた）。
#:
#: ★**既定値にしてあるのは、渡し忘れが起きるから**。同日の点検で、本番の
#: `rearrange_song` / `remake_from_step`（2箇所）/ `resize_song` の**4箇所が渡していなかった**。
#: いまは直後に `render_final` が走って上書きするので表に出ないが、**順番が変われば壊れる**。
#: 「呼ぶ側が忘れないようにする」ではなく、**忘れても正しくなる**ようにする。
#: ★★**空を渡す道も塞いである**——`None` / `""` は `ValueError` にする。
#: 「声を入れない vpr」に使い道は無く、**作れてしまうと黙って壊れた物が出来る**だけだから。
#: 既定値だけでは `voice_comp_id=None` と明示した呼び出しが素通りしてしまい、塞ぎ切れていなかった。
#: ※値は `voice_style.VOICES["HATSUNE_MIKU_V6_ORIGINAL"]` と同じ。**ずれたら別人の声になる**ので
#: `tests/song_service/test_song_pipeline.py` が一致を見張っている（import で結ばないのは、
#: 歌い方の適用を「失敗しても曲を落とさない」遅延 import にしてあるため）。
MIKU_V6_COMP_ID = "BHKCEKSYNBXG3HB2"


def _voices(score: SongScore) -> list[tuple[str, list]]:
    """歌トラックの並び（**主旋律が先、声部が後**）。vpr のトラックもこの順に書く。

    ※**V6 に歌トラックをいくつまで置いて書き出せるかは確かめていない**（2026-10-03）。保存済みの曲で V6 の書き出しまで
      済んでいるのは3本（主旋律＋声部2）まで（`dev_tools/scratch/s5065b362_c_tracks.py`）。アカペラは4〜5本になりうる——実機で確かめる。
    """
    return [(score.title, score.vocal_notes)] + [(v.title, v.notes) for v in score.extra_voices]


def _bar_ticks(score: SongScore) -> int:
    """1小節の tick（拍子から引く。決め打ちしない）。**計算は `song_design.bar_ticks` の1か所**（2026-10-02）。

    ★以前はここで `480×4÷分母×分子` を自前で計算していた（写し）。曲づくりの拍子（`TimeSignature`＝4/4・3/4・6/8・2/4）では
      同じ値になるが、写しは片方だけ直る（読めない拍子は `bar_ticks` が `ValueError`・こちらは黙って計算していた）。
    """
    from song_design import bar_ticks                      # noqa: E402（song_design は重いので呼ぶ時に・循環も避ける）
    return bar_ticks(score.time_signature)


def vocal_artic_findings(score: SongScore) -> list[str]:
    """**vpr へ描かない歌の奏法**の所見（2026-09-27・目録 表情-8）。vpr は書かない。

    `gliss` / `portamento`（V6 が自前で滑るので二重になる）と、カタログに無い奏法は描かない
    ——これは V6 へ訳すときの**組み立て**の決まりなので残す。ただし**描かなかったことは黙らない**。
    歌の崩しの工程（`vocal_artic.VOCAL_NAMES`）は滑る系を最初から選ばせないので、ここに出るのは
    人の手の直し・古い曲・工程の外から入った奏法。呼び側は `song_fix.record_change(song, "vocal_artic", …)` で残す。
    """
    bar = _bar_ticks(score)
    out: list[str] = []
    for title, notes in _voices(score):
        w: list[str] = []
        _write_controllers({}, notes, w, bar)
        out += [f"『{title}』{x}" for x in w]
    return out


def vibrato_findings_of(score: SongScore) -> list[str]:
    """**揺れが周りきらない短い音に揺れが書かれている**所（2026-09-27・目録 表情-7）。vpr は書かない・楽譜も直さない。

    揺れを決める側（`song_tuning.apply_tuning(tempo=)`・`vocal_expression.apply_plan(tempo=)`）がテンポを
    知っていれば出ない。出るのはテンポを渡さずに規則を当てた経路・人の手の直し・古い曲。
    """
    from song_tuning import vibrato_findings                # noqa: E402（循環回避）
    bar = _bar_ticks(score)
    out: list[str] = []
    for title, notes in _voices(score):
        out += [f"『{title}』{x}" for x in vibrato_findings(notes, score.tempo, where=lambda t: _where(t, bar))]
    return out


def voice_gain_value(gain_db: float) -> int:
    """声部の音量（dB）→ vpr のトラック音量の値（**1/10 dB 単位**・伴奏トラックと同じ）。

    V6 が受け付ける範囲（`v6_export._FADER_MIN/MAX`＝−60〜+6dB）の外は書かない——端に収める。
    """
    from v6_export import _FADER_MAX, _FADER_MIN            # noqa: E402（ここでだけ使う）
    return int(max(_FADER_MIN, min(_FADER_MAX, round(float(gain_db) * 10))))


def score_to_vpr(score: SongScore, out_path: Path, *, is_ai_singer: bool = True,
                 voice_comp_id: str = MIKU_V6_COMP_ID, embed_accomp: bool | None = None,
                 findings: list[str] | None = None) -> Path:
    """楽譜を vpr へ訳す（**組み立て**。楽譜に書かれたものを変えない）。

    `findings` に list を渡すと、**訳せずに描かなかったもの**（歌の奏法など）の所見を足して返す。
    呼び側はそれを `song_fix.record_change` で曲の記録へ残す（後の工程に見える）。

    ★**伴奏を同梱するかは楽譜の形で決める**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §4-4・§5-3）。
      `embed_accomp` を省くと形から（アカペラ＝同梱しない・それ以外＝同じフォルダの `accompaniment.wav` を同梱）。
      アカペラで同梱を頼むと `ValueError`——前の形の伴奏 wav がフォルダに残っていても、フォルダにあるかでは決めない。
      **インストの楽譜は vpr にしない**（歌が0音の空の歌トラックができる）——来たら呼び手の不具合として `ValueError`。
    """
    if not voice_comp_id:                                  # ★「声を入れない vpr」は作らせない（上の注記）
        raise ValueError(
            "voice_comp_id が空。声のIDの無い vpr は VOCALOID6 が開けず、"
            "「ボイスバンクが使用できません」で止まる。既定のまま呼ぶか、別の声のIDを渡す")
    form = getattr(score, "form", "song") or "song"
    if form == "instrumental":
        raise ValueError("インストの楽譜は vpr にしない（歌が無い。完成品は伴奏から作る・song_pipeline.compose）")
    if embed_accomp is None:
        embed_accomp = form != "acappella"
    elif embed_accomp and form == "acappella":
        raise ValueError("アカペラの vpr に伴奏は同梱しない（embed_accomp=True は渡せない）")
    numerator, denominator = (int(x) for x in score.time_signature.split("/"))
    bar_ticks = _bar_ticks(score)
    # メイン＋追加声部（extra_voices）を各 SingingTrack に。1声なら従来どおり1トラック。
    voices = _voices(score)
    project = Project(
        song_tempo_list=[SongTempo(position=0, bpm=score.tempo)],
        time_signature_list=[TimeSignature(bar_index=0, numerator=numerator, denominator=denominator)],
        track_list=[_singing_track(t, ns) for t, ns in voices],
    )
    # 伴奏があれば同梱（wav は vpr(zip) 内に実体で入る＝V6 のミックスダウン一発で完成ミックス。
    # 人手工程を「書き出し→mix依頼」の2段から「書き出し」1段に減らす・2026-08-10 V6実機検証済み）
    accomp = out_path.parent / "accompaniment.wav"
    has_accomp = embed_accomp and accomp.exists()
    if has_accomp:
        project.track_list.append(InstrumentalTrack(title="accompaniment", audio_file_path=str(accomp)))
    # ステップ1
    plugin_manager.plugins["svs"]["vpr"].dump(
        out_path, project, {"is_ai_singer": is_ai_singer, "default_lang_id": 0})

    # ステップ2: aiExp 注入（全トラックをループ＝メイン＋各追加声部）
    with zipfile.ZipFile(out_path) as z:
        data = json.loads(z.read(SEQ).decode("utf-8"))
        others = {n: z.read(n) for n in z.namelist() if n != SEQ}

    _artic_warns: list[str] = []                           # 歌の奏法で掛けられなかったもの
    for ti, (tr, (title, src_notes)) in enumerate(zip(data.get("tracks", []), voices)):
        for part in tr.get("parts", []):
            if part.get("aiVoice"):                        # voice_comp_id は必ず入っている（上で検査済み）
                part["aiVoice"]["compID"] = voice_comp_id
                # 初音ミクV6は日英2言語(langID 0/1)。LibreSVIP既定の[0]のみだと langID 不一致で「未インストール」扱いになる
                part["aiVoice"]["langIDs"] = [{"langID": 0}, {"langID": 1}]
            for jn, sn in zip(part.get("notes", []), src_notes):
                jn["aiExp"] = _tuning_to_aiexp(sn.tuning)
                _write_vibrato(jn, sn.tuning, 60.0 / score.tempo / 480)
            w: list[str] = []
            _write_controllers(part, src_notes, w, bar_ticks)
            _artic_warns += [f"『{title}』{x}" for x in w]
        # ★**声部の音量を vpr へ書く**（2026-09-27・目録 声部-2）。それまで `gain_db` は仮歌と MIDI にしか
        #   効かず、**V6 の書き出しでは声部が主旋律と同じ大きさ**で鳴っていた（ハモリの工程が「-4〜-8 が自然、
        #   0 にすると歌が濁る」と決めても届かなかった）。**主旋律（1本目）は触らない**——いまの大きさのまま。
        #   単位は伴奏トラックと同じ 1/10 dB（2026-08-10 に V6 のフェーダー表示で確認した形）。
        if ti >= 1 and tr.get("type") == 2:
            gain = float(getattr(score.extra_voices[ti - 1], "gain_db", 0.0) or 0.0)
            tr["volume"] = {**(tr.get("volume") or {}), "isFolded": True,
                            "events": [{"pos": 0, "value": voice_gain_value(gain)}]}
    # ★**描かなかった奏法を黙らない**（2026-09-27・目録 表情-8）。箱に積むだけで誰も読んでいなかった。
    for w in _artic_warns:
        print(f"  [歌の奏法] ★{w}")
    if findings is not None:
        findings.extend(_artic_warns)
    # ★**短い音の揺れは写したうえで所見に出す**（2026-09-27・目録 表情-7）。決める側がテンポを知っていれば出ない。
    _vib = vibrato_findings_of(score)
    for w in _vib:
        print(f"  [揺れ] ★{w}")
    if findings is not None:
        findings.extend(_vib)
    if has_accomp:                                         # 伴奏トラックに音量バランスの初期値（1/10dB 単位）
        db = _accomp_gain_db(out_path.parent)
        for tr in data.get("tracks", []):
            if tr.get("type") == 1:                        # type=1 がオーディオトラック
                tr["volume"] = {"isFolded": True, "height": 40.0,
                                "events": [{"pos": 0, "value": int(round(db * 10))}]}
    # LibreSVIP は version 5.0.0（VOCALOID5世代）で書く→VOCALOID6 が V6 ボイスバンクを認識しない。6.5.1 に上げる
    data["version"] = {"major": 6, "minor": 5, "revision": 1}

    _fix_global_tempo(data, float(score.tempo))
    if has_accomp:
        _extend_audio_region(data, accomp)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, raw in others.items():
            z.writestr(name, raw)
        z.writestr(SEQ, json.dumps(data, ensure_ascii=False))

    # ステップ3: **歌い方（声とスタイル）を焼き込む**（2026-09-01 統合）。
    # 声2種×スタイル14種の ID も適用関数も前から揃っていたのに、**パイプラインが
    # 一度も呼んでいなかった**——書き出された曲は全部 `No Effect`（加工なし）で、
    # `voices[]` にはテンプレート由来の洛天依が残っていた。
    # 決定は編成工程が決め、楽譜（`score.voice_choice`）に持つ（2026-09-26〜）。楽譜に無ければ既定の声のまま。
    _apply_voice_style(out_path, score)
    _apply_hand_edits(out_path, score)
    return out_path


def _apply_hand_edits(vpr: Path, score: SongScore) -> None:
    """**人が V6 で手で直した、楽譜に書けないもの**（`score.hand_edits`）を vpr へ描き戻す（2026-09-26）。

    手で描いたカーブは、その区間の生成したカーブを消してから人の点を入れる。発音記号は、その位置で始まる音符に
    書いて保護する（V6 が歌詞から付け直さないように）。**失敗しても曲は落とさない**。
    """
    edits = getattr(score, "hand_edits", None) or []
    if not edits:
        return
    try:
        with zipfile.ZipFile(vpr) as z:
            blobs = {n: z.read(n) for n in z.namelist()}
        d = json.loads(blobs[SEQ].decode("utf-8"))
        tracks = {tr.get("name"): tr for tr in d.get("tracks", []) if tr.get("type") == 2}
        for e in edits:
            tr = tracks.get(e.get("track"))
            if tr is None or not tr.get("parts"):
                print(f"  [手の直し] 『{e.get('track')}』という歌のトラックが無いので描き戻せなかった")
                continue
            part = tr["parts"][0]
            base = int(part.get("pos", 0))
            if e["kind"] == "curve":
                ctrls = part.setdefault("controllers", [])
                c = next((x for x in ctrls if x.get("name") == e["name"]), None)
                if c is None:
                    c = {"name": e["name"], "events": []}
                    ctrls.append(c)
                s, t = int(e["start"]) - base, int(e["end"]) - base
                kept = [x for x in c.get("events", []) if not (s <= int(x["pos"]) <= t)]
                c["events"] = sorted(kept + [{"pos": int(x["pos"]) - base, "value": x["value"]} for x in e["events"]],
                                     key=lambda x: x["pos"])
            elif e["kind"] == "phoneme":
                for n in part.get("notes") or []:
                    if base + int(n["pos"]) == int(e["start"]):
                        n["phoneme"], n["isProtected"] = e["phoneme"], True
            elif e["kind"] == "volume":
                tr["volume"] = {**(tr.get("volume") or {}), "events": e["events"]}
        blobs[SEQ] = json.dumps(d, ensure_ascii=False).encode("utf-8")
        with zipfile.ZipFile(vpr, "w", zipfile.ZIP_DEFLATED) as z:
            for n, b in blobs.items():
                z.writestr(n, b)
    except Exception as ex:
        print(f"  [手の直し] 描き戻せなかった（そのまま続行）: {type(ex).__name__}: {ex}")


def voice_choice_for(score: SongScore) -> dict[str, str]:
    """この楽譜で使う声の種類。**楽譜（`score.voice_choice`）だけを見る**。無ければ空＝既定の声（何も焼かない）。

    ★2026-09-26 まで実行中のメモリ（`voice_style.CHOICE`）しか見ておらず、編成の工程が走らない編集
      （キーを変える・声部を足すだけ）では空になり、**声の種類が抜けた vpr**（No Effect）を書き出していた
      （『明日への足音』v2・v5）。いまは楽譜が持つ。
    ★**実行中のメモリ・版ごとの記録へは落ちない**（2026-09-30）。以前は楽譜に無いと `CHOICE`（同じプロセスで最後に
      編成が決めた声）→ 曲フォルダの `record*.json` の編成の決定、の順に引いていた。記録から引くのは楽譜が声を持たない
      古い曲のためで、リリース前なので外した。`CHOICE` は**別の曲の決定でありうる**——新しい曲を作ったあと同じプロセスで
      カバー（編成の工程を通らず、楽譜に声を持たない）を書き出すと、前の曲の声がカバーに焼かれた。楽譜に無い声を
      当て推量で埋めず、既定の声のまま書き出す（新しい曲は `song_place.make_score` が編成の決定を楽譜に書く）。
    """
    return dict(getattr(score, "voice_choice", None) or {})


def part_styles_for(score: SongScore | None) -> list[str]:
    """**声部ごとの声**（`VoiceTrack.voice_style`）を声部の順に並べる（2026-09-27・目録 声部-3）。

    ★それまで `voice_style.apply_choice(part_styles=)` は**どこからも渡されておらず**、
      ハモリの工程が区間ごとに決めた声（掛け合いは声を変えると「別の人が返している」ように聞こえる）が
      全部 `harmony_style` 1つになっていた。空の欄は「既定（`harmony_style`）のまま」。
    """
    if score is None:
        return []
    return [str(getattr(v, "voice_style", "") or "") for v in (score.extra_voices or [])]


def _apply_voice_style(vpr: Path, score: SongScore) -> None:
    """声の種類を vpr へ焼く（楽譜が持つもの・`voice_choice_for`）。**失敗しても曲は落とさない**。"""
    try:
        import voice_style
        choice = voice_choice_for(score)
        styles = part_styles_for(score)
        if choice or any(styles):
            for w in voice_style.apply_choice(vpr, **choice, part_styles=styles or None):
                print(f"  [歌い方] ★{w}")
    except Exception as e:
        print(f"  [歌い方] 焼き込めなかった（そのまま続行）: {type(e).__name__}: {e}")
