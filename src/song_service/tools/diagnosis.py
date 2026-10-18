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

"""診断・評価・工程検問（節目レビュー）。検査官プロンプトは prompts/reviewer.md（規約: プロンプトは.md分離）。"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from align_reference import align_score_lyrics
from align_reference import align_to_reference_jp as _align_jp
from align_reference import check_timing, fix_timing, load_reference
from align_cover import (accompaniment_for_section, align_section, lyrics_for_section,
                         render_vocal_and_mix, save_cut_source, section_from_audio, section_from_midi,
                         write_accompaniment)
from machine_eval import similarity
from song_llm import generate_added_voice, pick_model
from song_checks import JsonFloat, JsonInt  # LLM が書く数と真偽の欄の型・2026-10-02 判断5
from song_table import spell_pitch                     # LLM に見せる高さは国際式の音名（表と同じ綴り・2026-10-01）
from song_fix import StepUnfixed                       # 形式の所見が直らずに工程で止まった合図
from song_flow import NOT_SWALLOW                      # 「失敗しても落とさない」except が握ってはいけない合図
from reference_to_wav import SR
from song_pipeline import compose
from song_schema import SongScore
from reference_fetch import fetch_audio_reference as _fetch_audio_ref
from reference_fetch import fetch_reference as _fetch_ref

import song_files                                      # 曲フォルダのファイルの名前の決まりと形の呼び名（2026-10-03）
from session import (SONGS, MusicSession, with_scope, _append_log, _trace, _traced, _locked, _save_section,
                     _save_ref_index, _save_accomp_bounds, _save_section_bounds, _write_fetch_log,
                     _clarify_msg, _build_log_digest, _slug, _song_status, _current_version,
                     begin_session, end_session, _snapshot_version_if_changed, _read_index, _update_index,
                     _retire_scoreless_session)
from coordinator import coordinator


_REVIEWER: Agent[None, str] = Agent(
    pick_model("REVIEW", "flash"),
    system_prompt=(Path(__file__).parents[1] / "prompts" / "reviewer.md").read_text(encoding="utf-8"))

#: 検査官の1行目（`prompts/reviewer.md`「出力は必ず1行目を『所見: 健全』『所見: 破綻あり』『所見: 要確認』のいずれかで始め」）
_VERDICT = re.compile(r"所見: (健全|破綻あり|要確認)")


def review_verdict(text: str) -> str:
    """検査官の返事の判定（`健全`・`破綻あり`・`要確認`）。1行目が決まりの書き方でなければ空（前後の空白だけ無視）。"""
    first = (text or "").strip().split("\n", 1)[0].strip()
    m = _VERDICT.match(first)
    return m.group(1) if m else ""


@_REVIEWER.output_validator
def _verdict_first(output: str) -> str:
    """★**1行目が決まりの書き方でなければ書き直させる**（2026-10-02・洗い出し 2-2）。以前は `startswith("所見: 破綻あり")` で読み、
    全角のコロン・前置きのある返事は「破綻」を数え損ねて、3回目で止める安全装置が効かなかった。"""
    from pydantic_ai import ModelRetry
    if not review_verdict(output):
        raise ModelRetry("1行目を『所見: 健全』『所見: 破綻あり』『所見: 要確認』のどれかで始める"
                         "（コロンは半角の `:` と空白1つ。前置きを書かない）。根拠はその後に続ける。")
    return output


@coordinator.tool
@_traced
@with_scope
def review_progress(ctx: RunContext[MusicSession]) -> str:
    """【工程検問】カバーで、重い工程に入る前と完成の前に呼ぶ総点検（①align_vocalの前=材料検問 ②build_accompanimentの前=整列検問
    ③render_finalの前=最終検問・カバーでは必須）。新しく作る曲（オリジナル）とその作り直しでは呼ばなくてよい（曲づくりの中で最終の点検役が
    曲全体を見て、NG なら作り直してから返る）。機械計測の事実＋取得歌詞を自動収集し、**別視点のLLM検査官**が健全性を判定して
    方向転換を指示する。個々のツールが「一応成功」でも組み合わせが破綻しているケース（参照×区間×歌詞×歌×伴奏の不整合）に
    手戻りが小さいうちに気づくための目。所見が「破綻あり」なら推奨アクション（やり直し/安全停止/要確認）に従うこと。"""
    # ★依頼文には「会話の流れ」（過去のやり取り・ミクの内心）も同梱されている。**経緯は点検に要るので渡す**（2026-09-27 本人判断）。
    #   2026-09-26 に、崩れたミクの内心（「再送だから何もしない」）を読んだ検査官が、作れている曲を「無視している」と判定した。
    #   崩れそのものは起きないように直した（main_agent の出力例・ループ途中結果の見せ方）。検査官の指示には
    #   「作れているかは機械計測の事実で判断する」と役割を書いた（prompts/reviewer.md）。
    facts = [f"依頼: {ctx.deps.instruction}",
             f"経路: {'カバー(faithful)' if ctx.deps.faithful else 'オリジナル/未確定'} "
             f"参照JSON={'あり' if ctx.deps.reference_json else 'なし'} "
             f"参照MIDI={'あり' if ctx.deps.reference_midi else 'なし'}"]
    # ★**曲の形を材料の頭に置く**（2026-10-03・§7-2）。無いと検査官が「歌がまだ無い（インスト）」「伴奏が無い（アカペラ）」を破綻と読む
    from tools.editing import form_of
    form = form_of(ctx)
    if form != "song":                                     # 歌ものはいままでどおり（材料を変えない）
        facts.insert(0, "形=" + {"acappella": "アカペラ（伴奏なしが正しい。声だけで和声とリズムを作る）",
                                "instrumental": "インスト（歌なしが正しい。主旋律は楽器が弾く）"}[form])
    st = ctx.deps.section
    if st:                                                 # 整列前でも見える材料の事実（区間×歌詞の整合は材料検問の主戦力）
        sec = ""
        if st.get("start_sec") is not None and st.get("end_sec") is not None:
            sec = f" 区間{st['start_sec']:.1f}〜{st['end_sec']:.1f}秒"
        facts.append(f"材料: kind={st.get('kind')} 区間の旋律音数={st.get('n_notes')}{sec} tempo={st.get('tempo')}")
    s = ctx.deps.score
    if s is not None and s.vocal_notes:
        span = s.duration_ticks() / 480 * 60.0 / s.tempo
        durs = sorted(n.duration_tick for n in s.vocal_notes)
        med = durs[len(durs) // 2] / 480 * 60.0 / s.tempo
        keys = [n.key for n in s.vocal_notes]
        facts.append(f"歌: {len(s.vocal_notes)}音 / {span:.1f}秒（{len(s.vocal_notes) / max(span, 0.1):.1f}音/秒・"
                     f"音符長中央値{med * 1000:.0f}ms・最短{durs[0] / 480 * 60.0 / s.tempo * 1000:.0f}ms）"
                     f"tempo={s.tempo:.0f} 音域{spell_pitch(min(keys))}-{spell_pitch(max(keys))}")   # 国際式（以前は MIDI 番号）
        # ★**最短音符長の数字だけでは判断できない**（2026-09-20）。
        #   『加速させて』(BPM180) は「最短83ms」を見た LLM が「人間の歌唱常識レンジ内」と書いて
        #   通したが、実際は**16分の4個つづきが10回あり、全部が1拍で終わって8分へ戻る**形で、
        #   本人に「めっちゃ早口のところがある」と言われた。**長さではなく続き方**が効く。
        #   判定はしない——ゾーンにするかは作り手の意図なので、**事実だけ添えて LLM に渡す**。
        _ns = sorted(s.vocal_notes, key=lambda n: n.start_tick)
        _st = [(_ns[i].start_tick - _ns[i - 1].start_tick) / 120.0 for i in range(1, len(_ns))]
        _runs, _cur = [], 0
        for _c in [abs(x - 1.0) < 0.01 for x in _st] + [False]:
            if _c:
                _cur += 1
            elif _cur:
                _runs.append(_cur + 1)
                _cur = 0
        # ★**「無い」も材料として出す**（2026-09-20）。連なりがゼロのときに行ごと省いたら、
        #   「早口を直して」と言われた LLM の手元に**何の材料も残らず**、
        #   手近な短い音符（8分＝167ms）を犯人にして全部4分へ広げ、**正常な曲を間延びさせた**。
        #   ★**出ていない事実は「無かった」ではなく「見ていない」と読まれる。**
        _alone = sum(1 for k in _runs if (k - 1) / 4.0 <= 1.0)
        _longest = (max(_runs) - 1) / 4.0 if _runs else 0.0
        facts.append(f"16分の連なり: {len(_runs)}回（最長 {_longest:.2f}拍）／"
                     f"うち1拍以内で終わるもの {_alone}回"
                     f"　※ゾーン（2小節以上つづける）か、行末で走り切るか、"
                     f"区間の設計でそう決めた場合のみ可。"
                     f"**0回なら16分は早口の原因ではない**")
    else:
        facts.append("歌: なし（インストなので正しい）" if form == "instrumental" else "歌: まだ無い")
    if ctx.deps.lyrics:
        moras = len([c for c in ctx.deps.lyrics if c.strip() and c not in "ゃゅょぁぃぅぇぉっー、。"])
        facts.append(f"取得歌詞({len(ctx.deps.lyrics)}字・約{moras}モーラ): {ctx.deps.lyrics[:300]}")
    song_dir = SONGS / ctx.deps.song_id
    if form == "acappella":                                # 前の形の伴奏が残っていても測らない（形で決める・§4-4）
        facts.append("伴奏: なし（アカペラなので正しい）")
    elif (song_dir / song_files.ACCOMP_WAV).exists():
        try:
            import soundfile as _sf
            facts.append(f"伴奏: {_sf.info(str(song_dir / song_files.ACCOMP_WAV)).duration:.1f}秒")
        except Exception:
            pass
    if ctx.deps.reference_json:
        try:
            facts.append(f"参照データ: {len(load_reference(ctx.deps.reference_json)['notes'])}音")
        except Exception:
            facts.append("参照データ: 読込失敗")
    try:                                                   # 各ツールがログにだけ吐いた★警告と直近評価を回収（表面化）
        t = (song_dir / song_files.BUILD_LOG).read_text(encoding="utf-8", errors="replace")
        warns = [ln.strip()[:150] for ln in t.splitlines() if "★" in ln][-8:]
        if warns:
            facts.append("機械警告(★): " + " ／ ".join(warns))
        evs = [ln.strip()[:130] for ln in t.splitlines() if re.search(r"evaluate_(song|timing)\.out", ln)][-2:]
        if evs:
            facts.append("直近評価: " + " ／ ".join(evs))
    except Exception:
        pass
    try:
        from song_usage import run_counted                  # noqa: E402（費用に入れる・2026-10-08）
        verdict = run_counted(_REVIEWER, "検査官", "\n".join(facts)).output
    except Exception as e:
        return f"検査官を呼べなかった（{e}）。機械計測の事実のみ:\n" + "\n".join(facts)
    if review_verdict(verdict) == "破綻あり":
        ctx.deps.review_breaks += 1
        if ctx.deps.review_breaks >= 3:                    # 命綱: 自動修復の無限ループ防止（§5.4b）
            verdict += ("\n（検査官補足: 破綻の指摘はこれで3回目。これ以上の自動修復はせず、"
                        "途中成果を保存して『マスターに確認してほしい』の最終報告で終了すること）")
    return verdict

@coordinator.tool
@_traced
@with_scope
def evaluate_song(ctx: RunContext[MusicSession]) -> str:
    """現在の曲を自己評価し、timing 等の問題点を返す（問題なければ『問題なし』）。"""
    if ctx.deps.score is None:
        return "曲が未生成。先に compose/align を。"
    if getattr(ctx.deps.score, "form", "song") == "instrumental":   # 歌の検査（曲頭・格子・重なり・調）は歌に当てるもの（2026-10-03・§7-2）
        return ("**判定: 対象外**——インストなので歌が無い（主旋律は楽器が弾く）。この検査（歌のタイミング・歌と調のずれ）は当てない。"
                "伴奏と区間の地図の長さは check_consistency、音量の起伏は check_mix_quality で見る。")
    head = ""
    broken = ""
    if ctx.deps.reference_json:
        ref = load_reference(ctx.deps.reference_json)["notes"]
        if not ref:                                        # railgun事故: 参照が空(0音)でも「問題なし」と言っていた
            broken = "参照データが空（音符0件）＝音高の比較ができない。参照の取得/変換が壊れている。"
        else:
            sim = similarity([n.key for n in ctx.deps.score.vocal_notes], [p for p, _ in ref])
            head = f"参照との音高一致度 {sim['similarity_100']}/100。"
            if sim["similarity_100"] < 40:
                broken = head + "**参照とほぼ一致しない＝参照か整列が壊れている可能性が高い**。"
    if broken:
        return (broken + "「問題なし」ではない。このまま render_final で完成扱いにせず、"
                "何が壊れたか点検し、直せなければマスターに確認する最終報告で終了して。")
    # ★**判定の語を「OK / 注意 / NG」にそろえる**（2026-09-20 本人指示）。
    #   「問題なし」という言い方は**曲全体のお墨付き**として読まれる。この検査が見ているのは
    #   構造の壊れ（曲頭・グリッド・重なり・調のずれ）だけで、**良い曲かどうかは見ていない**。
    if ctx.deps.faithful:    # カバー=実演奏タイミング（vad再整列）。拍グリッド評価は対象外＝fix_song_timingで量子化させない
        return (head + "**判定: OK**（構造の壊れは無い）。"
                "※カバーは原曲の実演奏タイミングを再現＝拍グリッド/曲頭無音は評価対象外。"
                "fix_song_timing は使わない＝再整列が壊れる。タイミングは evaluate_timing で参考確認")
    # **音程の事故もここで見る**（2026-09-07 追加）。timing しか見ていなかったので、
    # 歌だけが移調されて伴奏と三全音ずれた曲を「問題なし」と言って通した。
    from song_pipeline import _check_in_key
    issues = check_timing(ctx.deps.score) + _check_in_key(ctx.deps.score)
    # ★NG でも**意図して格子から外した音**（スウィング・タメ）は直さない（2026-09-27・目録の道具-2）。
    #   以前の案内は「NG なら fix」だけで、スタイルアレンジで置いたスウィングを fix_song_timing が戻していた。
    return head + ("**判定: OK**（構造の壊れは無い）" if not issues
                   else "**判定: NG** — " + " / ".join(issues)
                   + "\n※意図して格子から外した音（スウィング・タメ）なら直さない。意図していない所だけ、"
                     "範囲を絞った fix_song_timing(start, end)（`小節:拍`）か replace_region で直す")

@coordinator.tool
@_traced
@with_scope
def evaluate_timing(ctx: RunContext[MusicSession], song_id: str | None = None) -> str:
    """**作成物のタイミングを実音で客観評価**（自己評価用）: 歌声↔伴奏の頭ずれ・進行ぶれ、
    （参照ボーカルstemがあれば）歌声↔実ボーカルの整列を相互相関/chroma-DTWで測る。
    evaluate_song が拍グリッド/音高なのに対し、こちらは**書き出した実際の音の時間整合**を見る（音楽はタイミング一致が要）。"""
    from timing_eval import analyze_timing
    from timing_eval import verdict as _tv
    sid = song_id or ctx.deps.song_id
    song_dir = SONGS / sid
    if not song_dir.exists():
        return f"見つからない: {song_dir}"
    from tools.editing import form_of
    v = _tv(analyze_timing(song_dir, form=form_of(ctx) if sid == ctx.deps.song_id else _form_in(song_dir)))
    _trace(ctx.deps.song_id, "evaluate_timing", result=v)
    return f"タイミング評価: {v}"

@coordinator.tool
@_traced
@with_scope
def find_cramped_regions(ctx: RunContext[MusicSession], max_gap_beats: JsonFloat = 0.12,
                         min_run: JsonInt = 3) -> str:
    """確定スコアを走査し**音節が詰まった区間**（連続ノート間隔 < max_gap_beats が min_run音以上連続）を列挙。
    歌詞が崩れて聞こえる箇所の**自動検知**＝どこを inspect_region/replace_region で直すかを特定する。
    人間に区間を言われなくても、これで詰め込み箇所と拍範囲・歌詞を把握できる。"""
    s = ctx.deps.score
    if s is None:
        return "曲が未生成。先に align_vocal か load_existing_song を。"
    from tools.editing import form_refusal
    if why := form_refusal(ctx, "find_cramped_regions", "調べなかった"):  # インストは歌が無い（§7-2）
        return why
    Qd = 480
    ns = sorted(s.vocal_notes, key=lambda n: n.start_tick)
    runs, i = [], 0
    while i < len(ns) - 1:
        if (ns[i + 1].start_tick - ns[i].start_tick) / Qd < max_gap_beats:
            j = i
            while j + 1 < len(ns) and (ns[j + 1].start_tick - ns[j].start_tick) / Qd < max_gap_beats:
                j += 1
            if j - i + 1 >= min_run:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    if not runs:
        return f"詰め込み区間なし（間隔<{max_gap_beats}拍が{min_run}音連続する箇所は無い）。"
    # 位置は inspect_song の表と同じ `小節:拍`（区間は終わりを含まない・2026-10-01・ず-18。以前は楽譜の頭からの拍）
    from song_edits import at_text, score_bar_ticks
    bar = score_bar_ticks(s)
    lines = [f"詰め込み区間 {len(runs)}個（間隔<{max_gap_beats}拍が{min_run}音以上連続＝歌詞が崩れて聞こえる箇所・"
             f"位置は `小節:拍`・終わりは含まない）:"]
    for a, b in runs:
        sb, eb = at_text(ns[a].start_tick, bar), at_text(ns[b].start_tick + ns[b].duration_tick, bar)
        lines.append(f"  {sb}〜{eb} {b - a + 1}音 「{''.join(ns[k].lyric for k in range(a, b + 1))}」"
                     f" → inspect_region(\"{sb}\", \"{eb}\") で実オンセットを見て replace_region で置き直す")
    return "\n".join(lines)

@coordinator.tool
@_traced
@with_scope
def find_misplaced_regions(ctx: RunContext[MusicSession], min_gap_sec: JsonFloat = 0.5) -> str:
    """確定スコアを走査し**音節が"無音区間"に置かれた箇所**（歌声が鳴っていない無音ギャップ中に音符の頭がある）を列挙。
    詰め込み(find_cramped)とは別の不具合＝**フレーズの絶対位置ズレ**を自動検知する。ASRはフレーズ頭を無音ギャップの
    手前に誤検出しがちで、これが「サビの歌い出しが早すぎる/前後が乱れる」の主因。音声カバーのみ（vocals_wavが要る）。"""
    import numpy as _np

    from render_base import load_wav
    s = ctx.deps.score
    st = ctx.deps.section
    if s is None:
        return "曲が未生成。先に align_vocal か load_existing_song を。"
    from tools.editing import form_refusal
    if why := form_refusal(ctx, "find_misplaced_regions", "調べなかった"):
        return why
    vw = (st or {}).get("vocals_wav")
    origin = s.origin_abs_sec if s.origin_abs_sec is not None else (st or {}).get("start_sec")  # 【基準一本化】scoreの確定originを最優先（st["start_sec"]は古い検出値で0.7sズレ得る）
    if not vw or origin is None or not Path(vw).exists():
        return "音声参照(vocals_wav/区間)が無く無音ギャップ判定不可（音声カバーのみ対応）。"
    Qd, tempo = 480, s.tempo
    end_sec = (st or {}).get("end_sec") or (origin + s.duration_ticks() * 60.0 / tempo / Qd)
    y = load_wav(Path(vw))[int(origin * SR):int(end_sec * SR)].astype(_np.float32)
    hop = int(0.02 * SR)                                   # 無音ギャップ（origin基準秒＝score相対秒）を検出
    rms = _np.array([_np.sqrt(_np.mean(y[i:i + hop] ** 2)) for i in range(0, max(1, len(y) - hop), hop)])
    thr = float(rms.max()) * 0.08
    gaps, i, m = [], 0, len(rms)
    while i < m:
        if rms[i] < thr:
            j = i
            while j < m and rms[j] < thr:
                j += 1
            if (j - i) * 0.02 > min_gap_sec:               # min_gap_sec 以上の無音だけ＝フレーズ境界
                gaps.append((i * 0.02, j * 0.02))
            i = j
        else:
            i += 1
    if not gaps:
        return "無音ギャップ無し＝位置ズレ判定対象なし。"
    ns = sorted(s.vocal_notes, key=lambda n: n.start_tick)
    nsec = lambda k: ns[k].start_tick / Qd * 60.0 / tempo

    def _gap_bad(a, b):                                    # 位置ズレの起きている無音ギャップを2条件で検出
        if any(a + 0.12 < nsec(k) < b - 0.03 for k in range(len(ns))):   # ①ギャップ"内"に音節の頭
            return True
        for k in range(len(ns) - 1):                       # ②"跨ぎ": 手前(ギャップ前1.8s内)に頭があり次音がギャップ後＝フレーズ分断（頭が早すぎる）
            if a - 1.8 < nsec(k) <= a and nsec(k + 1) >= b - 0.05 and nsec(k + 1) - nsec(k) > 1.0:
                return True
        return False
    bad_gaps = [(a, b) for a, b in gaps if _gap_bad(a, b)]
    if not bad_gaps:
        return f"無音ギャップ{len(gaps)}個あるが、ギャップ内に置かれた音節は無い＝位置ズレなし。"
    from song_edits import at_text, score_bar_ticks
    bar = score_bar_ticks(s)
    lines = [f"位置ズレ区間 {len(bad_gaps)}個（**無音ギャップ中に音節の頭がある＝歌っていない時刻の音符**。フレーズ全体をギャップ直後へ・"
             "位置は `小節:拍`・終わりは含まない）:"]
    for (a, b) in bad_gaps:                                # ギャップ前後±2sを広く取り、散らばったフレーズの頭〜末尾を確実に含める（狭いと頭を取りこぼす）
        idx = [k for k in range(len(ns)) if a - 2.0 <= nsec(k) <= b + 2.0]
        if not idx:
            continue
        a0, b0 = idx[0], idx[-1]
        sb, eb = at_text(ns[a0].start_tick, bar), at_text(ns[b0].start_tick + ns[b0].duration_tick, bar)
        ga, gb = at_text(a * tempo / 60.0 * Qd, bar), at_text(b * tempo / 60.0 * Qd, bar)
        lines.append(f"  {sb}〜{eb} 「{''.join(ns[k].lyric for k in range(a0, b0 + 1))}」"
                     f"（無音ギャップ {ga}〜{gb}＝{a:.1f}-{b:.1f}s に音節あり）→ inspect_region(\"{sb}\", \"{eb}\") で無音ギャップを見て、"
                     "**このフレーズ全体をギャップ直後の実オンセットから**置き直す replace_region")
    return "\n".join(lines)

def _form_in(song_dir: Path) -> str:
    """開いていない曲の形（作業台の楽譜、無ければいちばん新しい版の楽譜の `form`。読めなければ song）。"""
    import json as _json
    cands = [song_dir / song_files.SCORE] + sorted(
        (p for p in song_dir.glob("score_v*.json") if song_files.split_version(p.name)),
        key=lambda p: -song_files.version_of(p.name))
    for p in cands:
        if p.exists():
            try:
                return song_files.as_form(_json.loads(p.read_text(encoding="utf-8")).get("form"))
            except Exception:
                return "song"
    return "song"


def _unmeasured(why: str) -> str:
    """**測れなかった**ことを、「問題なし」と読ませずに返す（2026-09-20）。

    ★実際に起きたこと: `check_mix_quality` が「区間が取れなかったので測れない
      （伴奏MIDIが無い曲かもしれない）。」と返し、**コーディネーターは所見なしとして素通りした**。
      しかも①**伴奏MIDIは在った**（理由の推測が外れていた）②失敗経路の文面が
      「曲は壊れていないので**このまま進んでよい**」で、**測れなかったことを
      積極的に『問題なし』へ変換していた**。

    **出ていない事実は「無かった」ではなく「見ていない」。** 進んでよいかを決めるのは
    この関数ではなく、材料を見たうえでのコーディネーター。
    """
    return ("★**測れなかった＝「問題なし」ではない。見ていないだけ。**\n"
            f"　理由: {why}\n"
            "　この軸（区間ごとの音量の起伏）は**未確認のまま**。次のどちらかをすること:\n"
            "　・`inspect_song()` の小節表で**強弱**（`X`=強く `x`=ふつう `.`=弱く）を見て、"
            "区間ごとに差が付いているかを自分で確かめる\n"
            "　・それもできなければ、**報告に「音量の起伏は確認できていない」と書く**。"
            "確認したふりをしない。")


# ★表の音名・コード名の綴りと和音での役割は `engine/song_table.py` にだけ置く（2026-09-30・記-16）。
#   ここに 2026-09-23 に移す前の写し（`_PITCH`・`_DEGREE`・`_pitch_name`・`_chord_root`）が呼ばれないまま残っていたので消した
#   ——写しがあると、綴りを直すときに片方だけ直る。


def _key_regions_for(ctx: RunContext[MusicSession]) -> list[tuple[int, str]] | None:
    """いまの楽譜の調の区間（表の綴りに使う）。**決められなければ None**（表はこれまでどおり ♯ で綴る）。

    楽譜は調を持たないので、曲の記録（組み立てが残した区間の調 `key_of` と時間軸）から引く（`song_table.key_regions`）。
    ★記録の区間の調は、あとから移調した曲（`adjust_song` の移調＝台帳の `semitones`）ではいまの調と食い違うので、
      移調の頼みがある曲では使わない（綴りを間違えるより、これまでどおりの ♯ のほうがまし）。カバーなど設計の無い曲も同じ。
    """
    try:
        from session import load_record
        from song_table import key_regions
        song = load_record(SONGS / ctx.deps.song_id, version=ctx.deps.record_version)
        if song is None or any("semitones" in (getattr(r, "value", None) or {})
                               for r in (getattr(song, "requests", None) or [])):
            return None
        return key_regions(song) or None
    except Exception:                                      # 記録が読めなくても表は出す（綴りが ♯ になるだけ）
        return None


def pitch_speller(ctx: RunContext[MusicSession]):
    """道具の返事で鳴る高さを綴る関数 `(MIDI 番号, tick) → 音名` を返す。曲の表（`inspect_song`）と同じく区間の調の向きで綴る
    （F の曲の B♭ は `Bb4`。調が決められない曲は ♯）。記法の仕様書 §0-1（2026-10-01）——以前は区間の観察・書き直しの返事・
    伴奏の観察だけが調を見ずに ♯ で綴り、同じ音が表では `Bb4`、道具の返事では `A#4` と出ていた。"""
    from song_table import _key_at
    keys = _key_regions_for(ctx)
    return lambda k, tick: spell_pitch(int(k), _key_at(keys, int(tick)))


@coordinator.tool
@_traced
@with_scope
def inspect_song(ctx: RunContext[MusicSession], start_bar: JsonInt = 1, end_bar: JsonInt = 0) -> str:
    """**曲を1枚で見る**（歌の全音符に伴奏の文脈を添えた表＋小節ごとの編成）。

    ★**「なんか違和感がある」と言われて、どの検査にも当てはまらないときに呼ぶ。**
      他の診断ツールは**あらかじめ決めた観点**しか見ない（詰め込み・ハモリの音域・16分の連なり…）。
      マスターの違和感がその一覧に無いことは普通にあるので、**そのときは自分の目で見る**。

    返すもの（良し悪しは**何も判断していない**。名指すのはあなた）:
      ・歌の1音ごと … `歌詞|音高|小節:拍|長さ(拍)|コード|和音での役割|d`
        - **音高・コード** は曲の調の綴り（国際式の音名。F の曲の B♭ は `Bb4`・`Bb`）。調が分からない曲（カバー・移調した曲）は ♯ で綴る
        - **小節:拍** は小節内の位置（`3:2.00` = 3小節目の拍2.00＝3拍目の頭。拍は0から）。**拍頭かどうかが分かる**
        - **和音での役割** root/3rd/5th/7th/9th… と **「外」**（コードに無い音）
        - **d** はドラムの打点と重なっている印
        - 歌が無い区間は `--- 歌なし N拍（M〜K小節）---` で挟む。息継ぎ・**前奏・間奏・後奏**が
          ここに出る＝フレーズの切れ目が見え、**小節で下の表と突き合わせられる**
        - **追加声部（ハモリ）は同じ行の右端**に `音高(主旋律との差)`＝**どこでどう重なるか**。
          主旋律と位置がずれた音（輪唱など）は `[声部名のみ]` の行で出す
      ・小節ごと … `小節|コード|最低音|楽器ごとの打点(音数)`。
        打点は**16分の枠**（4/4 で16個・3/4 と 6/8 で12個）に、伴奏の書き方と同じ記号
        （`X`=強く `x`=ふつう `.`=弱く・`-`=打点なし）＝**伴奏が拍のどこでどの強さで鳴るか**。
        **歌が無い小節・歌が休んでいる拍もここに出る**ので、間奏や息継ぎの裏で
        伴奏が何をしているかが見える（`x-x-x-x-x-x-x-x-`＝8分の刻み／
        `x---x---x---x---`＝4分／`x---------------`＝頭だけ／`----------------`＝出ていない）

    見方の例（ここに無い見方をしてよい。**それがこのツールの目的**）:
      ・言葉と旋律 … 1音に2モーラ乗っていないか／助詞が長く伸びていないか／
        単語の途中で休みが入っていないか／高い音に「い」「う」が来ていないか
      ・言葉と伴奏 … 内容語が拍頭に来ているか／全部の音が `d` と重なって張り付いていないか／
        「外」の音が強拍に長く居座っていないか
      ・ハモリ … 差が3度（`-3`/`-4`）で揃っているか／**どこで重なり、どこで抜けるか**（`-` の並び）／
        主旋律が高い所でハモリも上がりきっていないか／コードから「外」の音に重ねていないか
      ・構成 … 厚みの推移が歌の盛り上がりと合っているか／**全部の楽器が同じ枠で鳴っていないか**
        （打点が縦に揃う＝全員同じ刻み＝単調に聞こえる元）／
        **歌が休んでいる拍で伴奏が妙な所を叩いていないか**

    ※長い曲は `start_bar` / `end_bar` で小節を絞れる（既定は全部）。
      **毎ターン呼ぶものではない**——1曲で1万トークン前後ある。指摘を受けたときと、仕上げの確認に。
    ※インスト（歌が無い曲）は、歌の行の代わりに**主旋律の行**（主旋律を弾く楽器の音）と小節ごとの表を出す。
    """
    # ★**表を組むのは `engine/song_table.py`**（2026-09-23 に移した）。
    #   表そのものは道具ではなく、楽譜から組み立てるだけのもの。声部を書く工程
    #   （`harmony.write_voices`）も同じ表を使うので、**engine 側に1つだけ置く**。
    #   ここに残すと `engine` → `tools` の逆流が要る（層の向きは 道具 → 本体）。
    from song_table import song_table
    # ★インストの主旋律は伴奏 MIDI の1トラック（§5-2）。そのトラックの名前は書き出す側（`midi_out`）の決まりを読む（無ければ空＝行は出さずにそう書く）
    import midi_out

    body, b0, b1, shown = song_table(ctx.deps.score, SONGS / ctx.deps.song_id,
                                     start_bar=start_bar, end_bar=end_bar, keys=_key_regions_for(ctx),
                                     lead_track=getattr(midi_out, "LEAD_TRACK", ""))
    _trace(ctx.deps.song_id, "inspect_song", bars=f"{b0}-{b1}", notes=shown)
    return body


@coordinator.tool
@_traced
@with_scope
def inspect_region(ctx: RunContext[MusicSession], start: str, end: str) -> str:
    """確定スコアの区間 [start, end)（どちらも `小節:拍`＝inspect_song の表と同じ書き方。終わりは含まない）を
    **修正用に観察**する。現行ノートに加え、原曲ボーカルの
    **実オンセット（`小節:拍`）・推定音高・vad語**を返す＝replace_region で正しいノートを書き直す材料。
    特に詰め込み区間（音節が0コンマ数秒に固まった所）を、実オンセットの位置へ置き直すのに使う。"""
    s = ctx.deps.score
    if s is None:
        return "曲が未生成。先に align_vocal か load_existing_song を。"
    from tools.editing import form_refusal                 # インストは歌0音（断る表は1か所・2026-10-03・再レビュー）
    if why := form_refusal(ctx, "inspect_region", head="調べなかった"):
        return why
    from song_edits import at_text, score_bar_ticks, span_text, span_ticks
    from song_table import _hold_shown
    tempo = s.tempo
    Qd = 480
    bar = score_bar_ticks(s)
    try:
        s_tick, e_tick = span_ticks(start, end, bar)
    except ValueError as e:
        return f"見られなかった: {e}"
    # ★位置は `小節:拍`（表と同じ・2026-10-01・ず-18）。ここから先の計算は楽譜の頭からの拍で、見せるときに `pos` で書く
    start_beat, end_beat = s_tick / Qd, e_tick / Qd

    def pos(beat: float) -> str:
        return at_text(beat * Qd, bar)

    sp = pitch_speller(ctx)                                # 鳴る高さを区間の調の向きで（曲の表と同じ綴り）
    reg = [n for n in s.vocal_notes if start_beat <= n.start_tick / Qd < end_beat]
    # ★高さは国際式の音名（`C4`＝60・曲の表と同じ綴り。replace_region の `note` と同じ書き方）。2026-10-01 まで MIDI 番号。
    #   伸ばす音は楽譜の歌詞 `-` ではなく書く印 `_` で見せる（replace_region の `lyric` と同じ）
    cur = " ".join(f"{_hold_shown(n.lyric)}:{sp(n.key, n.start_tick)}@{at_text(n.start_tick, bar)}×{n.duration_tick / Qd:.2f}"
                   for n in reg)
    lines = [f"区間 {span_text(s_tick, e_tick, bar)}（終わりは含まない）tempo={tempo} 拍子={s.time_signature} "
             f"現行{len(reg)}音（かな:音名@小節:拍×長さ拍）:",
             f"  {cur or '（ノートなし）'}",
             f"  区間の歌詞(現行ノート・崩れている可能性): 「{''.join(n.lyric for n in reg)}」"]
    meta = []                                                  # 【曲情報】原題/会社名/指示＝曲構造(Aメロ/サビ等)推測の手掛かり（フレーズ境界の判断に効く）
    if ctx.deps.instruction:
        meta.append(f"指示「{ctx.deps.instruction}」")
    try:
        import json as _mj
        _rj = SONGS / ctx.deps.song_id / "source" / "reference.json"
        if _rj.exists():
            _r = _mj.loads(_rj.read_text(encoding="utf-8"))
            if _r.get("url"):
                meta.append(f"原曲ファイル名「{Path(_r['url']).name}」(日本語化前の原題/配布元の手掛かり)")
    except Exception:
        pass
    if meta:
        lines.append("  [曲情報] " + " / ".join(meta) + " ← 曲名/原題から曲構造を推測し、Aメロ/サビ等のフレーズ境界の判断に使う")
    import difflib as _dl                                   # lyrics.txt 由来の"正しい歌詞"を区間へ割り当て併記＝現行ノートの取り違え/重複を正す基準
    from lyric_slots import kana_text_chars                 # かなだけ（空白・句読点は歌詞のかなの型が差し戻す・2026-10-02・案B）。
    full = kana_text_chars(ctx.deps.lyrics or "")          # 行の切れ目（改行）は定義された区切りとして数えない（2026-10-03）
    ns_all = sorted(s.vocal_notes, key=lambda n: n.start_tick)
    ridx = [i for i, n in enumerate(ns_all) if start_beat <= n.start_tick / Qd < end_beat]
    if full and ridx:
        nl = [n.lyric for n in ns_all]
        n2f: dict[int, int] = {}                            # 現行ノートidx → 正しい歌詞idx（共通部分で対応付け）
        for a, b, sz in _dl.SequenceMatcher(None, nl, full, autojunk=False).get_matching_blocks():
            for k in range(sz):
                n2f[a + k] = b + k
        before = [n2f[i] for i in range(ridx[0]) if i in n2f]            # 区間直前の正しい歌詞位置
        after = [n2f[i] for i in range(ridx[-1] + 1, len(ns_all)) if i in n2f]  # 区間直後
        fs = (max(before) + 1) if before else 0
        fe = min(after) if after else len(full)
        if 0 <= fs <= fe <= len(full):
            lines.append(f"  ★区間の正しい歌詞(lyrics.txt由来): 「{''.join(full[fs:fe])}」"
                         " ← **この音節列を順に**置く（推定で増減・重複させない。現行ノートの歌詞が違っていてもこちらが正）")
    st = ctx.deps.section
    vw = (st or {}).get("vocals_wav") if st else None
    _sc = ctx.deps.score
    origin = (_sc.origin_abs_sec if _sc is not None and _sc.origin_abs_sec is not None  # 【基準一本化】scoreの確定origin最優先（st["start_sec"]は古い検出値で0.7sズレ得る）
              else ((st or {}).get("start_sec") if st else None))
    if vw and origin is not None and Path(vw).exists():
        import numpy as np
        import librosa
        from render_base import load_wav
        pad = 1.5                                              # 詰め込みは前方へ圧縮されるので前後に余白を取り、広い窓で信頼できるオンセットを得る（狭い窓だと過剰検出で詰め直しが直らない）
        ws = max(0.0, origin + start_beat * 60.0 / tempo - pad)
        we = origin + end_beat * 60.0 / tempo + pad
        y = load_wav(Path(vw))
        seg = y[int(ws * SR):int(we * SR)].astype(np.float32)
        if len(seg) > SR // 2:
            on = librosa.onset.onset_detect(y=seg, sr=SR, units="time", backtrack=True)
            f0, _vf, _ = librosa.pyin(seg, fmin=65, fmax=1000, sr=SR, hop_length=256)
            hop = 256 / SR

            def pitch_at(t):                                   # オンセット直後0.15sのpyin中央値→MIDI
                i0 = int(t / hop)
                win = f0[i0:i0 + int(0.15 / hop)]
                win = win[~np.isnan(win)]
                return int(round(69 + 12 * np.log2(float(np.median(win)) / 440))) if len(win) else 0

            crepe_at = None                                    # 別採譜(torchcrepe/CREPE)の音高も参考併記＝LLMが2手法で照合できる
            try:
                import torch
                import torchcrepe
                y16 = librosa.resample(seg, orig_sr=SR, target_sr=16000)
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                cp, per = torchcrepe.predict(torch.tensor(y16, dtype=torch.float32)[None], 16000, 160,
                                             fmin=65, fmax=1000, model="full", return_periodicity=True,
                                             batch_size=512, device=dev)
                cp, per, chop = cp[0].cpu().numpy(), per[0].cpu().numpy(), 160 / 16000

                def crepe_at(t):                               # オンセット直後0.15sのcrepe中央値（有声のみ）→MIDI
                    i0 = int(t / chop)
                    w = cp[i0:i0 + int(0.15 / chop)]
                    w = w[per[i0:i0 + int(0.15 / chop)] > 0.5]
                    return int(round(69 + 12 * np.log2(float(np.median(w)) / 440))) if len(w) else 0
            except Exception:
                crepe_at = None

            def _nm(m: int, beat: float) -> str:               # 推定音高 → 国際式の音名（0＝声が取れなかった）
                return sp(m, max(0.0, beat) * Qd) if m > 0 else "?"

            if crepe_at:
                onb = [(round((ws + float(o) - origin) * tempo / 60.0, 2), pitch_at(float(o)), crepe_at(float(o))) for o in on]
                lines.append(f"  実オンセット{len(onb)}個[小節:拍, pyin音高, crepe音高(別採譜)・音名・?=取れず]（区間±{pad}s）: " +
                             " ".join(f"({pos(b)},{_nm(m1, b)},{_nm(m2, b)})" for b, m1, m2 in onb))
                lines.append("  ※pyinとcrepeが一致する音高は信頼度高。食い違う所は旋律の流れで判断（どちらか/中間を選ぶ）。")
            else:
                onb = [(round((ws + float(o) - origin) * tempo / 60.0, 2), pitch_at(float(o))) for o in on]
                lines.append(f"  実オンセット{len(onb)}個[小節:拍,推定音高(pyin)・音名・?=取れず]（区間±{pad}s）: " +
                             " ".join(f"({pos(b)},{_nm(m, b)})" for b, m in onb))
            lines.append("  ★詰め込み音節を、対応する実オンセットの位置へ1つずつ置き直す。**書き換えるのは詰め込まれた音節（＋それに押されてズレた直後の数音）だけ**。"
                         "音節間隔が**正常(~0.3拍以上)に戻ったらそこで止める**。replace_region の範囲は**最小限**にし、"
                         "**遠くの正常な区間まで巻き込んで書き直さない**こと（前回 区間の終わりを広げ過ぎ、正常部分まで再オーサリングして壊した。"
                         "範囲は『詰め込み run ＋ 直後の押し出された分』に限定）。")
            hop2 = int(0.02 * SR)                              # 無音ギャップ(フレーズ境界)＝歌声エネルギーが落ちる区間
            grms = np.array([np.sqrt(np.mean(seg[k:k + hop2] ** 2)) for k in range(0, max(1, len(seg) - hop2), hop2)])
            gthr = float(grms.max()) * 0.08
            gaps, k, mm = [], 0, len(grms)
            while k < mm:
                if grms[k] < gthr:
                    k0 = k
                    while k < mm and grms[k] < gthr:
                        k += 1
                    if (k - k0) * 0.02 > 0.35:                  # 0.35s以上の無音だけ＝フレーズ境界
                        gb0 = round((ws + k0 * 0.02 - origin) * tempo / 60.0, 2)
                        gb1 = round((ws + k * 0.02 - origin) * tempo / 60.0, 2)
                        gaps.append((gb0, gb1))
                else:
                    k += 1
            if gaps:
                lines.append("  ★[無音ギャップ(フレーズ境界)]: " + " ".join(f"[{pos(a)}〜{pos(b)}]" for a, b in gaps) +
                             " ← **この無音区間には音節を置かない**。次フレーズ（例: サビの頭）は**ギャップ直後の実オンセットから**始める"
                             "（ASRの時刻はギャップ前にズレがちなので、ギャップを信じてフレーズごと後ろへ寄せる）。")
                ob = [x[0] for x in onb]                        # オンセット拍をギャップで前後セグメントに分け、各々の個数を出す
                seg_ranges, prev = [], start_beat
                for gs, ge in sorted(gaps):
                    seg_ranges.append((prev, gs))
                    prev = ge
                seg_ranges.append((prev, end_beat))
                segc = [(s0, s1, sum(1 for b in ob if s0 - 0.1 <= b < s1 + 0.1)) for s0, s1 in seg_ranges]
                lines.append("  ★[ギャップ区切り＝各セグメントで歌うオンセット数]: " + " ／ ".join(f"[{pos(s0)}〜{pos(s1)}]{c}音" for s0, s1, c in segc) +
                             " ← **歌詞を前から順に、各セグメントのオンセット数ぶんだけ割り振る**（例: 前セグメント5音=とじこめて、後セグメント9音=しゃいにんぐすたあ）。"
                             "ギャップを跨いで詰め込まない／手前セグメントの末尾音を後ろへ巻き込まない。")
        cache = Path(vw).with_suffix(".vad.json")
        if cache.exists():
            try:
                import json as _j
                words = _j.loads(cache.read_text(encoding="utf-8"))
                vr = [(round((w["start"] - origin) * tempo / 60.0, 2), w["word"]) for w in words
                      if ws <= w["start"] <= we]
                if vr:
                    lines.append("  vad語[小節:拍]: " + " ".join(f"{w}@{pos(b)}" for b, w in vr))
            except Exception:
                pass
        amid = SONGS / ctx.deps.song_id / "_midi_accomp" / "accompaniment_mt3.mid"
        if amid.exists():                                  # 採譜済み伴奏MIDIをビート/和声アンカーとして渡す（歌を伴奏に合わせやすく）
            try:
                import pretty_midi
                apm = pretty_midi.PrettyMIDI(str(amid))
                bass, drums = [], []
                for ins in apm.instruments:                # 伴奏MIDIは accompaniment.wav(=歌と同じorigin基準)から採譜→拍直変換
                    nm = "" if ins.is_drum else pretty_midi.program_to_instrument_name(ins.program)
                    for nt in ins.notes:
                        b = round(nt.start * tempo / 60.0, 2)
                        if not (start_beat <= b < end_beat):
                            continue
                        (drums if ins.is_drum else (bass if "Bass" in nm else [])).append(
                            b if ins.is_drum else (b, nt.pitch))
                if bass:
                    lines.append("  [伴奏MIDI]ベース[小節:拍,音名]: "
                                 + " ".join(f"({pos(b)},{sp(p, b * Qd)})" for b, p in sorted(set(bass))[:24]))
                if drums:
                    lines.append("  [伴奏MIDI]ドラムの打点[小節:拍]: "
                                 + " ".join(pos(b) for b in sorted(set(round(x, 1) for x in drums))[:32]))
                if bass or drums:
                    lines.append("  ※伴奏のベース/ドラムは拍・和声の基準。歌の音節（特にサビ頭）を伴奏の強拍・コード根音に合わせて置くと自然に揃う。")
            except Exception:
                pass
    else:
        lines.append("  （原曲ボーカル参照なし→現行ノートのみ。拍/音高は手掛かりから判断）")
    lines.append("→ replace_region(start,end,[{lyric,note,at,beats},...]) で区間を上書き"
                 "（start・end・at は `小節:拍`＝この返事・inspect_song の表と同じ書き方、beats は長さ（拍）、"
                 "note は国際式の音名 `C4`＝中央のド・`F#4`・`Bb3`、lyric は1モーラ・伸ばすなら `_`）")
    return "\n".join(lines)


@coordinator.tool
@_traced
@with_scope
def check_mix_quality(ctx: RunContext[MusicSession]) -> str:
    """**書き出した音そのものを測って**、抑揚が狙いどおり付いているか見る（仕上げの関門）。

    `evaluate_song` はタイミング（曲頭・グリッド・重なり）しか見ない。
    **音として良くなったか**——区間ごとの音量差が付いているか——は別で、
    **鳴らしてからでないと分からない**。実際、楽譜の上では 24dB の起伏があるのに
    音は 3dB しか差が無かった、ということが起きている。

    **編集のあとに呼ぶ**（新規作成では仕上げで自動的に通る）。
    機械は所見を並べるだけで、良し悪しは判断しない——**意図した「薄さ」**（落ちサビ・
    無音の間）を機械で落とさないため、判断はここのコーディネーターがする。
    """
    sid = ctx.deps.song_id
    song_dir = SONGS / sid
    if not sid or not song_dir.exists():
        return f"曲が未ロード。先に load_existing_song を。（song_id={sid!r}）"
    from tools.editing import form_of
    if form_of(ctx) == "acappella":                       # 伴奏の音が無い（前の形の伴奏が残っていても測らない・2026-10-03・§7-2）
        _trace(sid, "check_mix_quality.out", kind="not-applicable", reason="acappella")
        return ("**対象外**——アカペラなので伴奏の音が無く、この検査（伴奏の区間ごとの音量の起伏）は当てない。"
                "声部どうしの釣り合いは inspect_song の追加声部の音量で見る。")
    wav = song_dir / "accompaniment.wav"
    if not wav.exists():
        wav = next(iter(sorted(song_dir.glob("accompaniment_v*.wav"), reverse=True)), None)
    if wav is None or not wav.exists():
        _trace(sid, "check_mix_quality.out", kind="unmeasured", reason="伴奏wavなし")
        return _unmeasured("伴奏の音がまだ無い（先に render_and_mix か render_final を）")
    try:
        from measure_mix import inspect_levels, section_levels
        from session import load_record
        rec = load_record(song_dir, version=ctx.deps.record_version)
        if rec is not None and rec.theme is not None:       # オリジナル曲: 工程の記録の時間軸
            tl = getattr(rec, "timeline", None)
            tempo = rec.score.tempo if rec.score is not None else (ctx.deps.score.tempo if ctx.deps.score else 120.0)
        else:
            # ★カバー: **区間と調だけ**（コードの LLM は呼ばない・2026-10-02 組「カバーのコード」）。曲の記録にあれば読み、無ければ
            #   区間と調の LLM（カバーの構成推定）で起こして記録に残す。以前はコードの推定まで丸ごと呼んだうえ、カバーは時間軸を
            #   持たないので、いつも「区間を起こせなかった」になっていた（オリジナル曲も同じ経路で同じだった）
            from song_cover_chords import dirty, prepare_cover
            from song_cover_load import cover_timeline, load_cover
            song = load_cover(song_dir, verbose=False)
            prepare_cover(song, rec, chords=False, verbose=False)
            if dirty(song):
                from tools.editing import _cover_save
                _cover_save(ctx, song_dir, song, rec)
            tl, tempo = cover_timeline(song), song.score.tempo
        if tl is None:
            _trace(sid, "check_mix_quality.out", kind="unmeasured", reason="timeline なし")
            return _unmeasured("区間（Aメロ・サビ等）を起こせなかった")
        levels = section_levels(wav, tl, tempo)
        findings = inspect_levels(levels, tl)
    except StepUnfixed as e:                               # カバーの区間と調（LLM「カバーの構成推定」）の形式が直らずに止まった
        _trace(sid, "check_mix_quality.out", kind="stopped", step=e.step, findings=len(e.findings))
        return e.message(edit=True, how=(
            "**音量は測っていない**（「問題なし」ではない）。この曲（カバー）の区間を旋律から起こす所で止まった。"
            "**check_mix_quality()** をもう一度呼ぶと起こし直す。同じ所で止まるなら、報告に「音量の起伏は確認できていない」と書く。"))
    except NOT_SWALLOW:                                    # 止める・組み立ての不具合の合図は「測れなかった」に変えない（2026-10-03・再レビュー）
        raise
    except Exception as e:
        _trace(sid, "check_mix_quality.out", kind="unmeasured", reason=f"{type(e).__name__}")
        return _unmeasured(f"{type(e).__name__}: {e}")
    order = " / ".join(f"{k}:{v:.1f}dB" for k, v in levels.items())
    vals = [v for v in levels.values() if v > -90]
    span = (max(vals) - min(vals)) if len(vals) >= 2 else 0.0
    _trace(sid, "check_mix_quality.out", span_db=round(span, 1), findings=len(findings))
    if not findings:
        return (f"実測: 抑揚は付いている（区間の差 {span:.1f}dB）。{order}\n"
                f"目安は 1〜2dB=わずか / 3〜6dB=はっきり / 7dB以上=劇的。")
    return (f"実測: 区間の差 {span:.1f}dB。{order}\n"
            f"**所見**（良し悪しは判断していない・あなたが決める）:\n"
            + "\n".join(f"  ・{w}（{ev}）" for w, ev in findings)
            + "\n意図した薄さ（落ちサビ・間）ならこのまま進む。"
            "直すなら replace_accompaniment_region で（その区間の伴奏そのものを変える）。")


def _accompaniment_wav(song_dir: Path) -> Path | None:
    """いまの伴奏の音。**作業台があればそれだけ**、無ければ現在版（`check_mix_quality` と同じ選び方）。
    ★2026-09-27: 以前は作業台と現在版の**長いほう**を採っていた。旋律から作り直した直後は
    作業台（新しい伴奏）と前の版（古い伴奏）が並んでいるので、古い伴奏の長さで「尺が違う」と誤報し、
    コーディネーターが無い食い違いを追って作り直しを重ね、30分の上限で落ちた（『風に呼ばれたら』）。"""
    p = song_dir / "accompaniment.wav"
    if p.exists():
        return p
    cur = _current_version(song_dir)
    p = song_dir / song_files.versioned(song_files.ACCOMP_WAV, cur) if cur else None
    return p if p is not None and p.exists() else None


def _length_findings(song_dir: Path, s: SongScore, form: str = "song") -> list[str]:
    """歌と伴奏の尺を、**区間の地図（timeline）と照らして**測る。
    前奏・後奏は歌が無いのが正しいので、歌の終わりは「最後の歌の区間の終わり」と、
    伴奏の終わりは「最後の区間の終わり」と比べる（歌の終わり＝伴奏の終わり、とは見ない）。
    ★**インストは歌が無いので、伴奏と区間の地図の長さ比べだけ**（2026-10-03・§7-2「観点を捨てない」）。
      アカペラは伴奏が無いので、伴奏の長さは測らない（前の形の伴奏が残っていても混ぜない）。"""
    from session import load_record
    from song_schema import TICKS_PER_QUARTER as Q
    if not s.vocal_notes and form != "instrumental":
        return []
    sec = 60.0 / s.tempo / Q                                # 1 tick の秒数
    voc = max(n.start_tick + n.duration_tick for n in s.vocal_notes) * sec if s.vocal_notes else 0.0
    acc = 0.0
    wav = _accompaniment_wav(song_dir) if form != "acappella" else None
    if wav is not None:
        import wave
        with wave.open(str(wav)) as w:
            acc = w.getnframes() / w.getframerate()
    rec = load_record(song_dir)
    tl = getattr(rec, "timeline", None) if rec is not None else None
    if tl is None or not tl.spans:                          # 地図（時間軸）を持たない曲（カバー）は、歌と伴奏をそのまま比べる
        if acc and s.vocal_notes and abs(voc - acc) > max(3.0, voc * 0.05):
            return [f"歌と伴奏の尺が違う（歌 {voc:.1f}秒 / 伴奏 {acc:.1f}秒・区間の地図が無いので前奏・後奏を区別していない）"]
        return []
    out: list[str] = []
    sung = [x for x in tl.spans if x.has_vocal] if s.vocal_notes else []   # インスト（歌0音）は歌の区間と比べない
    if sung:
        sung_end = max(x.bar_to for x in sung) * tl.bar_ticks * sec
        if abs(voc - sung_end) > max(3.0, sung_end * 0.05):
            where = "はみ出している（後ろの区間に歌が乗る）" if voc > sung_end else "より先に終わっている"
            out.append(f"歌が歌の区間から{where}（歌の終わり {voc:.1f}秒 / 最後の歌の区間の終わり {sung_end:.1f}秒）"
                       f"——旋律か設計をやり直した後かもしれない")
    end = max(x.bar_to for x in tl.spans) * tl.bar_ticks * sec
    # 伴奏の音は最後の音の響き（残響）の分だけ区間より長い——後ろは余裕を持たせる
    if acc and (acc < end - 3.0 or acc > end + max(8.0, end * 0.05)):
        out.append(f"伴奏の長さが区間の地図と違う（伴奏 {acc:.1f}秒 / 区間の終わり {end:.1f}秒）"
                   f"——尺を変えたのに伴奏が追いついていないかもしれない"
                   f"（伴奏だけ先に終わるのは `ending=vocal` の狙いのこともある）")
    return out


@coordinator.tool
@_traced
@with_scope
def check_consistency(ctx: RunContext[MusicSession]) -> str:
    """**曲の中の食い違いを測る**（編集のあとの全体検査）。

    工程をやり直すと、**直した所の下流が置いていかれる**ことがある——
    旋律を変えたのに詞のモーラ数が合っていない、尺を変えたのに伴奏が元の長さのまま、など。
    ここは**事実を測って並べるだけ**で、良し悪しは判断しない。
    **どの工程をやり直すかは、所見を見てあなたが決める**（意図した形かもしれないため）。

    測るのは4つ:
      ・**歌詞と音符**（モーラ数が音符数と合っているか）
      ・**歌と伴奏の尺**（歌が歌の区間に、伴奏が区間の地図に収まっているか。前奏・後奏に歌が無いのは正しい）
      ・**ハモリと主旋律**（音数・音域が離れすぎていないか）
      ・**タイミング**（曲頭・グリッド・重なり。`evaluate_song` と同じ見方）

    ★**編集のあとに1回呼ぶ**。音量の起伏は `check_mix_quality` が別に見る。
    ※インストは歌が無いので、伴奏と区間の地図の長さだけを測る。
    """
    sid = ctx.deps.song_id
    song_dir = SONGS / sid
    s = ctx.deps.score
    if s is None:
        return "曲が未ロード。先に load_existing_song を。"
    if getattr(s, "form", "song") == "instrumental":       # 歌の観点は当てない。伴奏と区間の地図の長さ比べは通す（観点を捨てない・2026-10-03）
        found = _length_findings(song_dir, s, "instrumental")
        _trace(sid, "check_consistency.out", findings=len(found), form="instrumental")
        head = "整合の実測: インスト（歌は無い——歌詞と音符・ハモリ・歌のタイミングは測らない）"
        if not found:
            return head + "\n**判定: OK**（伴奏と区間の地図の長さに食い違いなし）"
        return (head + f"\n**判定: 注意**（{len(found)}件）——良し悪しは判断していない。**意図した形ならこのまま進む**:\n"
                + "\n".join(f"  ・{x}" for x in found))
    findings: list[str] = []

    # ① 歌詞と音符（**ひらがな1文字＝1モーラではない**——拗音は前にくっつく）
    from lyric_slots import moras
    n_notes = len(s.vocal_notes)
    n_moras = len(moras(s.lyrics_full or ""))
    if n_notes and abs(n_moras - n_notes) > max(2, n_notes * 0.02):
        findings.append(f"歌詞と音符の数が合っていない（音符 {n_notes} / モーラ {n_moras}）"
                        f"——旋律か詞のどちらかをやり直した後かもしれない")

    # ② 歌と伴奏の尺（アカペラは伴奏を測らない）
    findings += _length_findings(song_dir, s, getattr(s, "form", "song"))

    # ③ ハモリと主旋律
    for v in (s.extra_voices or []):
        if not v.notes:
            continue
        r = len(v.notes) / max(1, n_notes)
        if r > 1.2:
            findings.append(f"ハモリ『{v.title}』が主旋律より多い（{len(v.notes)}音 / 主 {n_notes}音）")
        lo, hi = min(n.key for n in v.notes), max(n.key for n in v.notes)
        mlo, mhi = min(n.key for n in s.vocal_notes), max(n.key for n in s.vocal_notes)
        if lo > mhi + 12 or hi < mlo - 12:
            findings.append(f"ハモリ『{v.title}』の音域が主旋律から1オクターブ以上離れている"
                            f"（ハモリ {spell_pitch(lo)}〜{spell_pitch(hi)} / 主 {spell_pitch(mlo)}〜{spell_pitch(mhi)}）")

    # ④ タイミング（カバーは実演奏の再現なのでグリッドを見ない）
    if not ctx.deps.faithful:
        from align_reference import check_timing
        findings += [f"タイミング: {x}" for x in check_timing(s)[:3]]

    _trace(sid, "check_consistency.out", findings=len(findings))
    # ★追加声部は**名前まで出す**（2026-09-20）。`adjust_voice` は名前で1つだけ直す／消すので、
    #   数だけ出していると「消したい声部の呼び名」が分からず、足すしか無くなる。
    head = (f"整合の実測: 音符 {n_notes} / モーラ {n_moras}"
            + (" / 追加声部 "
               + " ".join(f"『{v.title}』({len(v.notes)}音)" for v in s.extra_voices)
               if s.extra_voices else ""))
    # ★**判定の語を「OK / 注意 / NG」にそろえる**（2026-09-20 本人指示）。
    #   以前は「食い違いなし。このまま進んでよい」と返しており、**全体のお墨付き**として読まれた。
    #   射程と死角は `@with_scope` が必ず添えるので、ここは**判定だけ**を言う。
    if not findings:
        return head + "\n**判定: OK**（食い違いなし）"
    return (head + f"\n**判定: 注意**（{len(findings)}件）"
            "——良し悪しは判断していない。**意図した形ならこのまま進む**:\n"
            + "\n".join(f"  ・{x}" for x in findings)
            + "\n直すなら: 詞のずれ→remake_from_step('詞', 頼み)／"
              "尺のずれ→resize_song か build_accompaniment／ハモリ→adjust_voice か adjust_song(add_voice=)。")
