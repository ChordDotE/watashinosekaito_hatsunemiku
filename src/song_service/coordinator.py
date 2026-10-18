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

"""音楽コーディネーター: Agent定義と実行入口 run()。
ツールは tools/（creation/editing/diagnosis/accomp/humming/recall）が import 時にデコレータ登録する（静的import）。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic_ai import Agent

from align_cover import render_vocal_and_mix
import song_files
from song_llm import pick_model
from session import (SONGS, MusicSession, _append_log,
                     _save_ref_index, _save_accomp_bounds, _save_section_bounds, _write_fetch_log,
                     _clarify_msg, _build_log_digest, _slug, _song_status, _current_version,
                     begin_session, end_session, _snapshot_version_if_changed, _read_index, _update_index,
                     _retire_scoreless_session, settle_folder_title)

PROMPTS = Path(__file__).parent / "prompts"
COORD_PROMPT = (PROMPTS / "coordinator.md").read_text(encoding="utf-8")   # 規約: プロンプトは.md分離
# ★**工程の一覧はコードから差し込む**（2026-09-26）。指示書に手で並べていた一覧は工程を足すたびに古くなり
#   （編成／伴奏／ハモリで止まっていた）、コーディネーターは歌い方や声の種類を変える道を知らなかった。
from song_remake import describe_roles                     # noqa: E402
COORD_PROMPT = COORD_PROMPT + "\n\n" + describe_roles()

coordinator: Agent[MusicSession, str] = Agent(
    pick_model("COORD", "flash"), deps_type=MusicSession, system_prompt=COORD_PROMPT)

#: 進行役（コーディネーター）自身の呼び出しの、費用の記録のラベル
COORD_LABEL = "進行役"

last_song_id: str = ""                                   # 直近 run() が最後に扱った曲フォルダ（申し送り生成が読む）


def run(instruction: str, *, audio_path: str = "", request_limit: int = 25) -> str:
    """request_limit: LLM 呼び出し回数のハード上限（無限ループ防止。超えたら優雅な打ち切り）。
    既定25＝標準WF＋工程検問3回＋品質パスの余裕（検問導入で20→25・2026-07-11）。"""
    from pydantic_ai.usage import UsageLimits

    index = _read_index()                                # 起動時に全曲状態を読み、エージェントに渡す（修正時の参照）
    prompt = instruction if not index else f"{instruction}\n\n【既存曲一覧（修正/再ミックス時の参照）】\n{index}"
    sess = MusicSession(instruction=instruction, song_id=datetime.now().strftime("%Y%m%d_%H%M%S"))
    if audio_path:                                       # マスターが口ずさんだ音（鼻歌・口笛・肉声）
        sess.input_audio = audio_path
        prompt += (f"\n\n【マスターから音が届いている】{Path(audio_path).name}\n"
                   "**read_humming() で読み取ってから**、曲づくり／作り直しに入ること。")
    initial_id = sess.song_id                            # 発番フォルダ（編集/一覧なら後で畳む）
    _append_log(sess.song_id, f"\n{'=' * 70}\n[{datetime.now():%Y-%m-%d %H:%M}] セッション開始 指示: {instruction}")
    # ★**工程ごとの依頼文と出力を、曲のフォルダの log_prompts_v<版>/ に残す**（2026-09-26・`song_llm.PROMPT_LOG`。名前は `song_files.prompts_dir`）。
    #   曲名でフォルダ名が途中で変わるので、書くたびに `sess.song_id` から引く
    import song_llm
    from session import _working_version

    def _prompt_dir():
        d = SONGS / sess.song_id
        if not d.exists():
            return None
        return d / song_files.prompts_dir(_working_version(d) or (_current_version(d) + 1))
    song_llm.PROMPT_LOG = _prompt_dir
    try:
        _r = coordinator.run_sync(prompt, deps=sess, usage_limits=UsageLimits(request_limit=request_limit))
        result_output = _r.output
        # ★**費用を実測で残す**（2026-09-27）。工程の呼び出しは log_prompts_v<版>/ の各ファイルに1行ずつ、
        #   コーディネーター自身の分はここ（道具を呼ぶたびに文脈ごと送り直すので、工程より大きいことがある）
        # ★**進行役の分も費用の記録（`song_usage.USAGE`）に積む**（2026-10-08）。以前は使用量の行を書くだけで、
        #   単価を掛けた費用はどこにも無かった（曲の費用が工程の分だけになり、少なく出ていた）。
        #   曲の記録（各工程の `usd`）には割り当てない——進行役は曲づくりの周の外で、作り直し・編集・一覧も
        #   同じ進行役が受け持つので、どの工程の費用とも言えない。費用は下の [費用] の行に残す
        from song_usage import USAGE as _steps, record as _rec_usage
        n_steps = len(_steps)
        _rec_usage(coordinator, _r, COORD_LABEL)
        _co = _steps[n_steps:]
        _append_log(sess.song_id,
                    f"[使用量] コーディネーター（{getattr(coordinator.model, 'model_name', '')}）: "
                    f"{song_llm.usage_line(_r).removeprefix('使用量: ')}\n"
                    f"[使用量] 工程 {n_steps}回: 入力 {sum(u['in'] for u in _steps[:n_steps]):,}"
                    f" / 出力 {sum(u['out'] for u in _steps[:n_steps]):,}\n"
                    f"[費用] 工程 ${sum(u['usd'] for u in _steps[:n_steps]):.3f}"
                    f" ＋ 進行役 ${sum(u['usd'] for u in _co):.3f}"
                    f" ＝ ${sum(u['usd'] for u in _steps):.3f}（単価は song_usage.PRICES）")
    except Exception as e:                               # §5.4b-③ 優雅な打ち切り: 上限到達でも例外死せず、
        from pydantic_ai.exceptions import UsageLimitExceeded   # 途中成果を版として確定し人間の言葉で報告する
        if not isinstance(e, UsageLimitExceeded):
            # ★**どこで落ちたかを残す**（2026-09-22）。型と文言だけを書いていたため、
            #   `ValueError: list.remove(x): x not in list` のような**どこにでもある文言**から
            #   場所が引けなかった——**曲を1本作るのに十数分と課金がかかる**ので、
            #   場所を知るためだけにもう一度回す羽目になる。**落ちたときこそ一番情報が要る。**
            import traceback
            _append_log(sess.song_id,
                        f"=== 異常終了 ===\n{type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
        result_output = (f"ここまでやったけど、作業回数の上限（{request_limit}回）に届いたから一旦やめておくね。"
                         f"途中までの結果はぜんぶ保存してあるよ。『続きを仕上げて』と言ってもらえれば、"
                         f"保存したところから再開できるからね。")
        _append_log(sess.song_id, f"[上限] request_limit={request_limit} 到達→優雅な打ち切り（成果はディスクに永続済み）")
    finally:
        song_llm.PROMPT_LOG = None                        # 次の実行で別の曲へ書かないように
    # 工程の入出力・カバー内部トレースは _trace で統一ログ（`song_files.BUILD_LOG`）に時系列追記済み。ここでは要約を追記。
    summary = f"=== 最終報告 ===\n{result_output}"
    try:                                                # 曲名の作り直しで付け替えられなかった曲フォルダの名前を、版を確定する前に最後にもう一度（2026-10-03）
        settle_folder_title(sess)
    except Exception:
        pass
    song_dir = SONGS / sess.song_id            # 作業対象フォルダ（render_final/load/mix で sess.song_id 更新済み）
    if sess.score is not None and song_dir.exists():    # G3 命綱: メモリ上のscoreが未永続のままセッションが終わる事故の防止
        try:                                            # （align_vocal/fix_song_timing はメモリのみ＝
            sp = song_dir / song_files.SCORE            #   LLMがrender系を忘れて終了すると編集が消える。差分がある時だけ書く
            # ★★**読んだだけのセッションでは発火させない**（2026-09-20）★★
            #   以前は「作業台の score.json と一致するか」だけを見ていたため、
            #   **編集ツールを1つも呼んでいないセッションでも作り直し**が走った。
            #   実害: 『ショートさせ』を調べただけで vpr・伴奏・仮歌・口の動きが作り直され、
            #   **作ったときの歌い方の選択（メモリにしか無い値）が失われた**。
            #   → **読み込み時の姿と変わっていなければ、何もしない。**
            #     新規作成は `score_at_load` が None＝従来どおり必ず書き出す。
            _now = sess.score.model_dump_json(indent=2)
            _edited = sess.score_at_load is None or _now != sess.score_at_load
            if _edited and (not sp.exists() or sp.read_text(encoding="utf-8") != _now):
                render_vocal_and_mix(sess.score, song_dir)
                _append_log(sess.song_id, "[命綱] セッション終了時にメモリ上のscoreが未永続だったため書き出した（score/vpr/仮歌）")
        except Exception:
            pass
    # ★**編集のあとの最終の点検の命綱**（2026-10-04・案C）。大きい編集をしたのに書き出さずに終わった（点検待ちの束が残っている）なら、
    #   版を確定する前に1回呼び、結果を最終報告の後ろに足す（ミクへの申し送りに載る）。点検役が呼べなくても版の確定は止めない
    if sess.pending_edits and song_dir.exists():
        try:
            from types import SimpleNamespace
            from tools.editing import review_edits
            _gate = review_edits(SimpleNamespace(deps=sess), "実行の終わり（書き出さずに終わった）")
        except Exception as e:                           # noqa: BLE001
            _gate = f"\n【最終の点検役（編集のあと）】呼べなかった（{type(e).__name__}）。"
        if _gate:
            result_output += "\n" + _gate.strip()
            summary += _gate
    if song_dir.exists() and (song_dir / song_files.SCORE).exists():  # セッション終了: 最終状態を版として確定（開始時の状態は前セッションが確定済み＝残る）
        base = getattr(sess, "edit_base_version", None) or _current_version(song_dir)
        ver = end_session(song_dir, instruction)        # 作業結果を版として確定し、作業台（固定名）を片付ける
        # ★**前の版との差を、この回の頼みと照らして報告に必ず載せる**（2026-10-11・本人承認・`song_version_check`）。
        #   頼んでいないのに変わった所・頼んだのに変わっていない所が、報告に載らないまま何度も起きた（『てのひらのシグナル』v15 など）
        try:
            from song_version_check import version_check_text
            _check = version_check_text(song_dir, base, ver, instruction)
        except Exception as e:                           # noqa: BLE001（比べられなくても版の確定と報告は止めない。黙らない）
            _check = f"\n【前の版から変わった所】比べられなかった（機械の側の失敗・{type(e).__name__}）"
        if _check:
            result_output += "\n" + _check.strip()
            summary += _check
        # ★**編集のあとの最終の点検役の判断も、報告に必ず載せる**（2026-10-14）。道具の返事にだけ出て、最終報告から落ちることがあり、
        #   点検役の指摘（Eb6 など）がミクにもマスターにも届かなかった（『てのひらのシグナル』v26）
        if getattr(sess, "edit_gate_log", None):
            _g = "\n【最終の点検役（編集のあと）の判断】" + " ".join(sess.edit_gate_log[-1].split())[:700]
            if _g.strip() not in result_output:
                result_output += _g
                summary += _g
        summary += (f"\n[版] このセッション＝v{ver}（実体は {song_files.versioned(song_files.SCORE, ver)}"
                    f" / {song_files.versioned(song_files.VPR, ver)} など。"
                    f"固定名は作業台なので確定時に消える。過去版 *_v1〜v{ver - 1} は保持）")
    _append_log(sess.song_id, summary)                  # 統一ログ(song_files.BUILD_LOG)へ要約を追記
    _update_index(sess.song_id)                         # 全曲インデックス(songs.md)を更新＝曲名・状態・版を記録
    _retire_scoreless_session(initial_id)               # 編集/一覧/停止セッションの裸フォルダを畳む（ログは_sessions.logへ）
    if sess.song_id != initial_id:
        _retire_scoreless_session(sess.song_id)
    global last_song_id                                 # 申し送り生成（handoff）が統一ログを読むために公開
    last_song_id = sess.song_id
    try:                                                # コンソール出力は最後（失敗しても永続ログは確保済み）
        print(f"[{sess.song_id}] {result_output}")
    except Exception:
        print(result_output.encode("ascii", "replace").decode())
    return result_output


# ツール登録（デコレータ副作用のための静的import。coordinator 定義後に読み込む必要がある）
from tools import accomp, creation, diagnosis, editing, humming, manual, recall, style  # noqa: E402,F401
