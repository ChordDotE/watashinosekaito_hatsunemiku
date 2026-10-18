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

"""新規作成の工程ツール（標準WF: fetch_reference→draft_melody→fetch_section_lyrics→align_vocal→
build_accompaniment→render_and_mix→render_final）。"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from align_reference import align_to_reference_jp as _align_jp
from align_reference import check_timing, fix_timing, load_reference
from align_cover import (accompaniment_for_section, align_section, lyrics_for_section,
                         render_vocal_and_mix, save_cut_source, section_from_audio, section_from_midi,
                         write_accompaniment)
from machine_eval import similarity
from song_llm import generate_added_voice, pick_model
from song_checks import JsonBool  # LLM が書く数と真偽の欄の型・2026-10-02 判断5
from song_checks import strip_ascii_ws  # 半角の名前の前後の空白（§0-6・2026-10-03）
from song_table import spell_pitch                     # LLM に見せる高さは国際式の音名（表と同じ綴り・2026-10-01）
from reference_to_wav import SR
from song_pipeline import compose
import song_files                                      # 曲フォルダの名前の決まり・曲の形の綴り（2026-10-03）
import v6_export                                       # V6 を自動操作して歌声を書き出す
from song_schema import SongScore
from reference_fetch import fetch_audio_reference as _fetch_audio_ref
from reference_fetch import fetch_reference as _fetch_ref

from session import (SONGS, MusicSession, _append_log, _trace, _traced, _locked, _save_section,
                     _save_ref_index, _save_accomp_bounds, _save_section_bounds, _write_fetch_log,
                     _clarify_msg, _build_log_digest, _slug, _song_status, _current_version,
                     begin_session, end_session, _snapshot_version_if_changed, _read_index, _update_index,
                     _retire_scoreless_session, _working_version, NEXT_EXPORT, ensure_working_version,
                     _repoint_paths, settle_folder_title)
from coordinator import coordinator


def _audio_cover_available() -> bool:
    """**音声からのカバーが作れる環境か**（2026-09-02 に切り分けを細かくした）。

    カバーには**2つの経路**があり、要るものが違う:

    | 経路 | 中身 | GPU |
    |---|---|---|
    | **MIDI が見つかる** | 記号の対応づけ＋伴奏は元 MIDI を SoundFont で鳴らす | **要らない** |
    | 音声しか無い | 音源分離（demucs）→採譜（YourMT3）→強制アラインメント（SOFA）| **要る** |

    ★最初は「カバーなら全部 GPU 必須」で弾いていたが、**それは粗すぎた**——
    移植前の本番が「作曲機能ごと GPU 必須」で弾いていたのと同じ形を、一段内側で作っていた。
    実測でも MIDI 経路は**1分**で終わり、分離も採譜も走っていない（『きらきら星』2026-09-02）。

    この関数が見るのは**音声経路だけ**。MIDI 経路は環境を問わず通す。
    """
    svc = Path(__file__).resolve().parents[1]
    if not (svc / "sofa_venv").exists() or not (svc / "venv_mt3").exists():
        return False                                       # オリジナルだけ導入した環境
    # ★**`import torch` をここでしない**（2026-09-02）。GPU の有無を見るためだけに
    #   torch（7.4GB の依存）を**プロセスへ読み込んでしまう**——GPU 無しの環境で
    #   カバーを断るだけの判定が、重いものを引き込むのは本末転倒。
    #   `nvidia-smi` の有無で見る（コントロールパネルの `has_nvidia_gpu` と同じ考え方）。
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return False
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and "GPU" in (r.stdout or "")
    except Exception:
        return False


#: 工程名 → 工程の記録の名前（`song_flow._rec` に渡す名前＝`STOP_AFTER` が照らす名前）。
#: ★**工程を全部載せる**（2026-10-02・洗い出し H15）。以前は 歌い方・歌の崩し・声部・彩り・奏法 が無く、`show_after` を黙って
#:   捨てて最後まで作った（止める指定が消え、費用は戻らない）。`tests/song_service/test_tool_leniency.py` が `STEP_ORDER` との漏れを落とす。
#:   伴奏だけ記録の名前が `arrange`（`song_context._STEP_KEY` の `arrangement` とは違う）。
SHOW_AFTER_STEP = {"テーマ": "theme", "設計": "design", "コード": "chords", "グルーヴ": "groove",
                   "旋律": "melody", "詞の設計": "lyric_plan", "サビの詞": "lyric_chorus", "詞": "lyrics",
                   "曲名": "naming",                       # 2026-10-03 の追加工程
                   "歌い方": "expression", "歌の崩し": "vocal_artic",
                   "編成": "instrumentation", "伴奏": "arrange", "ハモリ": "harmony",
                   "声部": "voices", "彩り": "colour", "奏法": "articulation"}


def _resolve_show_after(name: str, form: str = "song") -> str:
    """`show_after` の工程名 → 工程の記録の名前（空なら空）。工程名は `remake_from_step` と同じ読み手（`song_remake.resolve_step`＝
    工程名の綴りそのまま・2026-10-02 に別名をやめた）。
    **分からない工程名は `ValueError`**（選べる工程名を添える。2026-10-02 まで黙って空＝最後まで作る にしていた）。
    ★**曲の形 `form` に無い工程も `ValueError`**（2026-10-03・インストの『詞』など。止まらずに最後まで作ってしまうので作り始めない）。"""
    if not (name or "").strip():
        return ""
    from song_remake import absent_step_problem, resolve_step
    label = resolve_step(name)
    if label not in SHOW_AFTER_STEP:
        raise ValueError(f"show_after の『{name}』がどの工程か分からない。選べる工程名: {' / '.join(SHOW_AFTER_STEP)}")
    if (why := absent_step_problem(label, form)):
        raise ValueError(f"show_after の{why.split('（作り直さなかった）')[0]}")
    return SHOW_AFTER_STEP[label]


def _next_step_after(steps_json: Path) -> str:
    """途中保存から、**まだ作っていない最初の工程**の名前を返す（無ければ空）。その曲の形で走る工程だけを見る（2026-10-03）。"""
    try:
        from song_context import _STEP_KEY
        from song_gate import FORM_STEPS, STEP_ORDER
        from song_model import form_of_record
        d = json.loads(steps_json.read_text(encoding="utf-8-sig"))
        saved = d.get("steps") or {}
        runs = FORM_STEPS[form_of_record(d)]
        for name in STEP_ORDER:
            if name in runs and saved.get(_STEP_KEY.get(name, "")) is None:
                return name
    except Exception:
        pass
    return ""


def _name_folder(ctx: RunContext[MusicSession], title: str) -> None:
    """曲フォルダに曲名を付ける（`YYYYMMDD_HHMMSS` → `YYYYMMDD_HHMMSS_曲名`）。

    ★**曲名が決まったらすぐ付ける**（2026-09-26・本人指摘）。それまでは最後の `render_final` で
      付けていたので、**途中で止めた曲・失敗した曲は数字だけのフォルダ**のまま残り、
      人がフォルダを開いてもどの曲か分からなかった。最後の曲名は詞のあとの工程『曲名』で決まる（2026-10-03。それより前の
      工程で止まった・落ちた曲はテーマの仮の曲名）。`draft_melody` が曲を返した時点（＝最後の曲名）・途中で止めた時点・
      失敗した時点で付ける。**付けるのは裸のフォルダだけ**。あとで曲名だけを作り直して曲名が変わったときの付け替えは
      `session.retitle_song_folder`（2026-10-03。`remake_from_step` が呼ぶ）。
    ★**付けられなかったら黙らずログに残す**。Windows は中のファイルを誰かが開いていると
      フォルダの名前を変えられない（2026-09-25 の実機確認で、ログを読み続けていたため実際に失敗した）。
    ★**曲名の作り直しで付け替えられなかった名前**（`deps.folder_title`）があれば、ここで先にもう一度試す（2026-10-03）。
    命名は**新規作成（裸のタイムスタンプ）のみ**。編集（命名済みフォルダ）は名前を維持
    ＝再slug差（の↔_等）でフォルダ名を二重化/rename失敗するのを防ぐ。
    """
    settle_folder_title(ctx.deps, SONGS)
    slug = _slug(title or "")
    old_id = ctx.deps.song_id or ""
    if not slug or not re.fullmatch(r"\d{8}_\d{6}", old_id):
        return
    old_dir, new_dir = SONGS / old_id, SONGS / f"{old_id}_{slug}"
    if not old_dir.exists() or new_dir.exists():
        return
    try:
        old_dir.rename(new_dir)
    except OSError as e:                                   # 致命ではない＝元の名前のまま続ける
        _trace(old_id, "name_folder.warn", reason=f"{type(e).__name__}: {e}"[:160],
               hint="中のファイルを開いているものがあると名前を変えられない。次の書き出しでもう一度試す")
        return
    ctx.deps.song_id = new_dir.name
    _repoint_paths(ctx.deps, old_dir, new_dir)             # 同じ会話で持っている参照・区間の音のパスも新しいフォルダへ（2026-10-03）
    _trace(new_dir.name, "name_folder.out", old=old_id, new=new_dir.name)


def _mark_failed(ctx: RunContext[MusicSession], err: Exception) -> None:
    """落ちた曲の曲フォルダに止めた印（`song_files.PROGRESS`＝`_progress.json`）を書く（**失敗しても何もしない**）。

    書く中身は `_progress_message` と同じ形（`title` / `stage` / `step`）に、
    **失敗の理由（`failed`）と、次に作る工程（`next`）**を足したもの。
    `load_existing_song` はこれを見て「失敗で止まった曲」と案内し、
    `remake_from_step` は `title` から途中保存を引いて続きを作る。
    """
    try:
        import song_flow
        wip = song_flow.LAST_AUTOSAVE
        if wip is None:
            return                                         # 1工程も済んでいない＝続きの材料が無い
        steps_json = wip.with_suffix(".steps.json")
        if not steps_json.exists():
            return
        title = wip.stem
        nxt = _next_step_after(steps_json)
        _name_folder(ctx, title)                           # 失敗した曲も、フォルダを見てどの曲か分かるように
        song_dir = SONGS / ctx.deps.song_id
        song_dir.mkdir(parents=True, exist_ok=True)
        (song_dir / song_files.PROGRESS).write_text(json.dumps({
            "title": title, "stage": "失敗で止まった", "step": "",
            "failed": f"{type(err).__name__}: {str(err)[:200]}", "next": nxt,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        _trace(ctx.deps.song_id, "draft_melody.out", kind="failed_resumable",
               title=title, next=nxt, reason=type(err).__name__)
        # ★**曲の一覧にも載せる**。`coordinator.run` は正常に終わったときしか一覧を更新しない
        #   ので、落ちた曲は一覧に無く、**ミクから「どの曲?」と指せなかった**。
        from session import _update_index
        _update_index(ctx.deps.song_id)
        # ★**失敗の通知から曲を辿れるようにする**。`last_song_id` も正常終了のときしか
        #   入らないので、`song_runner` は落ちた曲がどれか知らなかった。
        import coordinator as _coord_mod
        _coord_mod.last_song_id = ctx.deps.song_id
    except Exception:
        pass


def _stopped_message(ctx: RunContext[MusicSession], err) -> str:
    """**形式の所見が直らずに止まった曲**の目印を曲フォルダに書き、コーディネーターへ出す文を返す（2026-09-30）。

    止めた印（`song_files.PROGRESS`）は途中で止めた曲・失敗した曲と同じ形（`title` / `stage` / `step` / `next`）に、
    **止まった工程と直らなかった所見（`stopped`）**を足したもの。`load_existing_song` はこれを見て「止まった曲」と案内し、
    `remake_from_step` は `title` から途中保存を引いて止まった工程から続きを作る。`song_runner` は完了通知に止まったことを添える。
    ★失敗（`failed`）とは分ける——壊れたのではなく、書いたものの形式が直らなかったので止めた（続きはマスターの頼みで作る）。
    """
    import song_flow
    title = ""
    # ★**途中保存が書けたかを見て文を分ける**（2026-10-03・再レビュー）。`song_flow._autosave` は書けなければ `LAST_AUTOSAVE` を空にする。
    #   以前は保存の成否に関係なく「ここまでの工程は保存してある」と返し、次の会話の `remake_from_step` が続きを作れなかった
    wip = song_flow.LAST_AUTOSAVE
    saved = err.step == "テーマ" or wip is not None       # テーマで止まった曲は元から保存が無い（文は StepUnfixed.message が書く）
    try:
        title = wip.stem if wip is not None else ((err.song.title() if err.song is not None else "") or "")
        _name_folder(ctx, title)                           # 止まった曲も、フォルダを見てどの曲か分かるように
        song_dir = SONGS / ctx.deps.song_id
        song_dir.mkdir(parents=True, exist_ok=True)
        # ★テーマは作り直しの起点にならない（`remake_from_step` に無い工程＝別の曲になる）。`next` を書くと次の会話で
        #   `remake_from_step(step='テーマ')` を案内してしまうので書かない。続きは draft_melody でもう一度（2026-09-30）
        # ★止めた理由（`why`）を印に残す（2026-10-04 組P2b）。形式の所見が直らなかった（`StepUnfixed`）のか、最終の点検役の NG が
        #   残った（`FinalUnpassed`）のかで、続きの案内（`load_existing_song`）と完了通知（`song_runner`）の言い方が変わる
        why = getattr(err, "why", "") or "書いたものの形式が直らなかった"
        mark = {"title": title, "stage": f"{err.step or '最後'}の工程で止まった（{why}）", "step": err.step,
                "stopped": {"step": err.step, "findings": list(err.findings)[:20], "why": why}}
        if err.step != "テーマ" and saved:
            mark["next"] = err.step
        if not saved:                                      # 続きの材料が無い（`load_existing_song` が draft_melody へ案内する）
            mark["unsaved"] = True
        (song_dir / song_files.PROGRESS).write_text(json.dumps(mark, ensure_ascii=False, indent=2), encoding="utf-8")
        _trace(ctx.deps.song_id, "draft_melody.out", kind="stopped", title=title, step=err.step,
               findings=len(err.findings))
        from session import _update_index                  # 曲の一覧に載せる（ミクから「どの曲?」と指せるように）
        _update_index(ctx.deps.song_id)
        import coordinator as _coord_mod
        _coord_mod.last_song_id = ctx.deps.song_id         # 完了通知（`song_runner`）がこの曲の目印を読む
    except Exception as e:                                 # 目印が書けなくても、止まったことは返す
        _trace(ctx.deps.song_id, "draft_melody.warn", stopped=f"{type(e).__name__}: {e}"[:120])
    # 保存できなかったときの文は `StepUnfixed.message` が `song_flow.AUTOSAVE_FAILED` を見て書く（新しい曲の言い方のまま・2026-10-03）
    return (f"オリジナル曲『{title or '（曲名未定）'}』は**途中で止まった**。\n" + err.message())


def _cover_stopped(ctx: RunContext[MusicSession], tool: str, err, *, how: str) -> str:
    """カバーの工程（区間特定・歌詞の取得・歌詞の割り当て・音高補正）で**形式の所見が直らずに止まった**ときの返事（2026-10-02）。

    以前は道具の `except Exception` が止める合図（`StepUnfixed`）まで受け止め、MIDI 経路は「参照MIDIは解析できなかった」と返して
    参照を外していた（参照は壊れていない）。カバーは工程の記録を持たないので、続きは同じ道具をもう一度呼ぶ（`how` に書く）。
    """
    _trace(ctx.deps.song_id, f"{tool}.out", kind="stopped", step=err.step, findings=len(err.findings))
    return err.message(edit=True, how=how)


def _progress_message(ctx: RunContext[MusicSession], song, step: str) -> str:
    """**ここまで作った、を人へ出す形にして返す。** 音になる段なら音を作る。"""
    from song_preview import show_progress

    try:
        _name_folder(ctx, song.title() or "")              # 途中で止めた曲も、フォルダを見てどの曲か分かるように
    except Exception:
        pass
    song_dir = SONGS / ctx.deps.song_id
    song_dir.mkdir(parents=True, exist_ok=True)
    got = show_progress(song, song_dir)
    # ★**続きを作れるように、曲名を残す**——`remake_from_step` は `_wip/<曲名>.json` を
    #   曲名で引くが、途中の曲には楽譜（score.json）がまだ無く、そこから曲名を取れない。
    title = ""
    try:
        title = song.title() or ""
        (song_dir / song_files.PROGRESS).write_text(
            json.dumps({"title": title, "stage": got.get("stage", ""), "step": step},
                       ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    _trace(ctx.deps.song_id, "draft_melody.out", kind="progress", step=step,
           stage=got.get("stage", ""), shown=got.get("kind"))
    # 止めた工程の名前（`show_after` に書いた工程）。見せる材料の段（`stage`）は歌い方・声部などを持たないので、名前は工程から引く
    shown = {v: k for k, v in SHOW_AFTER_STEP.items()}.get(step) or got.get("stage", step)
    head = (f"**{shown}まで作ったところで、いったん手を止めた。**"
            f"（曲名は『{title}』・ここまでの工程は保存済み）\n")
    if got.get("kind") == "audio":
        body = (f"聴いてもらう音: `{got['path']}`（{got.get('seconds')}秒）\n"
                # 本番と違う所（コード・グルーヴの段は曲頭の空き小節なし）・鳴らせなかったもの（2026-10-01・判-E）
                + (f"音について: {got['note']}——**聴いてもらうときに、これも伝えること。**\n" if got.get("note") else "")
                + f"**マスターに「ここまで作ったから聴いてみて」と伝えて、再生を勧めること。**")
    elif got.get("kind") == "text":
        body = (f"見てもらう内容:\n{got.get('text', '')}\n"
                f"**マスターに読んでもらい、方向が合っているか聞くこと。**")
    else:
        body = "まだ見せられるものが無い。"
    tail = ("\n続けるときは **remake_from_step** を使う——マスターの言葉と工程の記録を読んで、"
            "**どの工程から作り直すか**を決める（そのまま先へ進めるなら、まだ作っていない工程を指定する）。")
    return head + body + tail


@coordinator.tool
@_locked
def draft_melody(ctx: RunContext[MusicSession], section_spec: str = "全体", song_title: str = "",
                 show_after: str = "", kind: str = "", form: str = "") -> str:
    """工程2: 区間の主旋律を用意する。fetch_reference 済みなら参照（MIDI=コピー/音声=採譜）から区間を切り出し
    deps.section に格納。**参照が無ければ記憶から旋律＋歌詞を生成**（この場合 fetch_section_lyrics は不要）。

    section_spec=コピー範囲（例: サビ / 1番 / 全体 / 「Aから Bまで」）。song_title=曲名（省略時は指示文）。

    `show_after`（任意・オリジナル曲のみ）: **ここまで作ったら、いったんマスターに見せる**。
    指定できるのは工程の名前（テーマ / 設計 / コード / グルーヴ / 旋律 / 詞の設計 / サビの詞 / 詞 / 曲名 / 歌い方 / 歌の崩し /
    編成 / 伴奏 / ハモリ / 声部 / 彩り / 奏法）。分からない名前なら作り始めずに返す。
    ★**全部作り切らずに途中で相談したいとき**に使う——手本が心もとないとき（鼻歌など）、
    下流が大きい工程（旋律を直すと詞も伴奏もやり直し）、マスターが「途中で見せて」と言ったとき。
    **そこまでの成果は保存される**ので、あとから続きを作れる（`remake_from_step`）。
    指定しなければ従来どおり最後まで作る。
    ★**工程が書いたものの形式（記法・個数・長さ・名前）が直らなければ、その工程で止まる**（2026-09-30）。
    返事にどの工程で何が直らなかったかが出るので、最終報告でマスターに伝える。続きは `remake_from_step`（その工程から）。

    `kind`（**必ず書く**）: **original**＝新しく作る曲（口ずさみを手本にする曲も含む）／
    **cover**＝既存の曲を再現する曲。**マスターの言葉から、あなたが判断する**。
    ※ミクの受け取り方の文に「カバーではなく」とあっても、それは説明であってカバーの依頼ではない。
    ※cover なのにお手本（fetch_reference の結果）が無いときは、想像で作らずに止まる。

    `form`（任意・2026-10-03）: **曲の形**。**song**＝歌もの（ミクの歌と伴奏・省いたらこれ）／**acappella**＝アカペラ
    （伴奏なし・声だけ）／**instrumental**＝インスト（歌なし・演奏だけ）。この綴りのまま書く（読めない綴りなら作り始めずに返す）。
    **マスターの言葉から、あなたが判断する**（「アカペラで」「歌なしで」「インストで」「BGM にして」）。迷えば作り始めずに聞く。
    ※カバー（kind='cover'・お手本あり）は song だけ（それ以外は作り始めずに返す）。
    """
    title = song_title or ctx.deps.instruction
    ctx.deps.lyrics = None
    _trace(ctx.deps.song_id, "draft_melody.in", section_spec=section_spec, song_title=title,
           kind=kind or "(未指定)", form=form or "(未指定=song)",
           ref_midi=bool(ctx.deps.reference_midi), ref_audio=bool(ctx.deps.reference_audio_accomp),
           ref_json=bool(ctx.deps.reference_json))
    # ★**曲の形はコーディネーターが決める**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §2・F1①）。読めない綴りと、
    #   カバーで song 以外は**作り始めずに返す**（事実の確認だけ。カバーは原曲の歌と伴奏の両方を前提に組まれている）
    try:
        the_form = song_files.as_form(form)
    except ValueError as e:
        _trace(ctx.deps.song_id, "draft_melody.out", kind="refused", reason="form", given=str(form)[:40])
        return f"作り始めなかった: {e}。"
    if the_form != "song" and ((kind or "").strip() == "cover"
                               or ctx.deps.reference_midi or ctx.deps.reference_json):
        _trace(ctx.deps.song_id, "draft_melody.out", kind="refused", reason="cover-form", form=the_form)
        return (f"作り始めなかった: カバーは歌もの（form='song'）だけ——{song_files.FORM_LABEL[the_form]}にはできない"
                "（カバーは原曲の歌と伴奏の両方を前提に組まれている）。"
                f"{song_files.FORM_LABEL[the_form]}で作るなら、原曲をなぞらないオリジナル曲（kind='original'）として作る。"
                "原曲の伴奏だけ（歌抜き）を取り出す道はまだ無い——マスターがそれを望んでいるなら、そう伝えて聞く。")
    # ★**`kind` と `show_after` は、参照（原曲のお手本）で区間を切り出す前に読む**（2026-10-03・再レビュー）。以前は参照があると
    #   ここより前に返り、`kind` を読まず・`show_after` を黙って捨てていた（オリジナルのつもりでもカバーとして切り出した）
    kind = strip_ascii_ws(kind)                            # 前後は半角の空白とタブだけ（§0-6・2026-10-03）。書けるのは小文字の 2 語だけ（2026-10-02 まで大小を無視して読んでいた）
    if kind not in ("original", "cover"):
        _trace(ctx.deps.song_id, "draft_melody.out", kind="ask_kind", given=kind)
        return ("`kind` を決めてから呼び直して: **original**＝新しく作る曲（口ずさみを手本にする曲も含む）／"
                "**cover**＝既存の曲を再現する曲。マスターの言葉から判断する"
                "（ミクの受け取り方の文にある「カバーではなく」は説明であって、カバーの依頼ではない）。")
    if ctx.deps.reference_midi or ctx.deps.reference_json:
        if kind == "original":
            _trace(ctx.deps.song_id, "draft_melody.out", kind="refused", reason="original-with-reference")
            return ("作り始めなかった: kind='original'（新しく作る曲）だが、原曲のお手本（参照）が読み込まれている"
                    "——参照があると原曲から区間を切り出す（カバーになる）。カバーなら kind='cover' で呼び直す。"
                    "オリジナルとして作るなら、先に set_reference('') で参照を外してから呼ぶ。")
        if (show_after or "").strip():
            _trace(ctx.deps.song_id, "draft_melody.out", kind="refused", reason="show_after-cover")
            return ("作り始めなかった: show_after はオリジナル曲の工程（テーマ〜奏法）で止める指定——カバーは原曲から区間を"
                    "切り出すので、止める工程が無い。show_after を空にして呼び直す。")
    from song_fix import StepUnfixed
    from song_flow import NOT_SWALLOW
    if ctx.deps.reference_midi and ctx.deps.melody_channel is not None:
        try:
            st = ctx.deps.section = section_from_midi(ctx.deps.reference_midi, ctx.deps.melody_channel,
                                                      section_spec, title)
        except StepUnfixed as e:                           # 区間特定の LLM の形式が直らずに止まった＝参照は壊れていない（外さない）
            return _cover_stopped(ctx, "draft_melody", e, how=(
                "**区間はまだ用意していない**（参照の MIDI はそのまま使える）。続けるときは draft_melody(section_spec=…) で"
                "区間の言い方を変えて（『サビ』『1番』『Aメロから Bメロまで』など）もう一度呼ぶ。"))
        except NOT_SWALLOW:                                # 組み立ての不具合などの止める合図は握らない
            raise
        except Exception as e:                             # 壊れた/構成が変な参照MIDIはセッション死でなく方向転換へ
            _trace(ctx.deps.song_id, "draft_melody.out", kind="error", reason=f"{type(e).__name__}")
            ctx.deps.reference_midi = ctx.deps.reference_json = None
            return (f"この参照MIDIは解析できなかった（{type(e).__name__}: {e}）。壊れているか特殊な構成みたい。"
                    f"fetch_reference で別の候補（別の検索語・音声参照）を試すか、見つからなければマスターに確認して終了して。")
        _trace(ctx.deps.song_id, "draft_melody.out", kind="midi", midi=Path(ctx.deps.reference_midi).name,
               ch=ctx.deps.melody_channel, section=st['section_index'], n_notes=st['n_notes'],
               tempo=round(st['tempo'], 1))
        return (f"区間旋律(MIDIコピー)用意: 『{title}/{section_spec}』 区間{st['section_index']} {st['n_notes']}音 "
                f"tempo{round(st['tempo'], 1)}。次は fetch_section_lyrics(section_spec) で歌詞取得。")
    if ctx.deps.reference_json:                            # 音声参照 or 記譜のみ参照（set_reference）
        try:
            st = ctx.deps.section = section_from_audio(ctx.deps.reference_json, ctx.deps.reference_audio_accomp,
                                                       section_spec, title)
        except StepUnfixed as e:                           # 区間特定の LLM の形式が直らずに止まった
            return _cover_stopped(ctx, "draft_melody", e, how=(
                "**区間はまだ用意していない**（参照はそのまま使える）。続けるときは draft_melody(section_spec=…) で"
                "区間を言い直してもう一度呼ぶ（`60秒から88秒` のように秒で書くと、区間特定の LLM を通さずに切り出す）。"))
        except NOT_SWALLOW:
            raise
        except Exception as e:
            _trace(ctx.deps.song_id, "draft_melody.out", kind="error", reason=f"{type(e).__name__}")
            return (f"この参照の区間切り出しに失敗した（{type(e).__name__}: {e}）。"
                    f"fetch_reference で参照を取り直すか、区間指定（section_spec）を変えて試して。"
                    f"続かなければマスターに確認して終了して。")
        _save_section(ctx.deps.song_id, st)               # 中間生成物を保存（編集時に復元→align_vocal で再整列可能に）
        _trace(ctx.deps.song_id, "draft_melody.out", kind="audio", section=st['section_index'],
               start_sec=round(st['start_sec'], 1), end_sec=round(st['end_sec'], 1),
               n_notes=st['n_notes'], tempo=round(st['tempo'], 1), accomp=bool(ctx.deps.reference_audio_accomp))
        # 秒範囲を返すのが重要: これが無いと LLM は指定どおり切れたか判定できず、
        # 表現を変えて draft_melody を連打する（2026-08-09 の区間ガチャ事故・decisions.md）
        return (f"区間旋律(音声採譜)用意: 『{title}/{section_spec}』 区間{st['section_index']}"
                f"＝元曲の{st['start_sec']:.1f}〜{st['end_sec']:.1f}秒 {st['n_notes']}音 "
                f"tempo{round(st['tempo'], 1)} 伴奏={'あり' if ctx.deps.reference_audio_accomp else 'なし'}。"
                f"指定との差が2秒程度までならこのまま fetch_section_lyrics(section_spec) へ進む。"
                f"大きくズレている場合だけ『60秒から88秒』のように**秒で明示指定**して1回やり直す。"
                # 機械が区間を縮めた（間奏とみなした歌の途切れで分けた）ことを返事にも出す（2026-10-03・再レビュー・カバーの組）
                + ("\n★機械が区間を縮めた: " + " ／ ".join(st["section_changes"])
                   + "（このままでよければ fetch_section_lyrics へ。間奏をまたいで歌いたいなら『60秒から88秒』のように秒で指定し直す）"
                   if st.get("section_changes") else ""))
    # ★**カバーかどうかは、コーディネーターが決める**（2026-09-25・本人判断）。
    #   以前は**指示文全体に「カバー」の文字があるか**で決めていた。指示文にはマスターの言葉だけでなく
    #   **ミクの受け取り方**も入っていて、ミクが「**カバーではなく**オリジナル曲だし」と書いただけで
    #   **オリジナル曲が作れなくなった**（VM の配布検証で3回連続で断られ、ミクはマスターに言い直しを頼んだ）。
    #   しかも断る文言の「オリジナルなら指示に明記して」は、**明記しても判定が変わらない**ので逃げ道にならなかった。
    #   ★**残すのは事実の確認だけ**——「カバーだと言ったのに、お手本が無い」は機械で止める
    #   （参照ゼロで作ると**存在しない曲の幻覚カバー**になる。E2E T7 で実測）。
    # `kind` は区間を切り出す前に読んだ（上）
    if kind == "cover":
        _trace(ctx.deps.song_id, "draft_melody.out", kind="refused", reason="カバーだが参照なし")
        return ("カバーなのにお手本（MIDI/音源）が無い——想像で作ると実在の曲の再現にならないので、ここで止める。"
                "fetch_reference をまだ呼んでいなければ先に呼ぶ。呼んで見つからなかったなら、"
                "曲名・アーティスト・音源の URL かファイルがもらえないか、確認したいことを最終報告に書いて終了する。")
    # ★**オリジナル曲はここから作る**（2026-09-01 に入口を差し替え）。
    #   従来は `generate_song`＝**LLM 2回**（歌詞全文→旋律）で、区間の概念が無く、
    #   伴奏は3テンプレ固定、音源は numpy の合成音7種、調声も転調もハモリも無かった。
    #   新しい `song_compose` は **11工程＋関門**——テーマ→設計→コード→**グルーヴ**
    #   →旋律→**詞の設計→サビの詞→詞**→編成→伴奏→ハモリ（旋律とサビの詞の3案→選抜は 2026-09-30 に1案へ）。
    #   区間・強さ・終わり方を設計が決め、以降がそれに従う（グルーヴは 2026-09-07、
    #   詞の3段化は 2026-09-13 に追加）。★**カバーはここを通らない**（`align_cover` 側）。
    #   詳細は docs/plan/song_migration_decision.md。
    from song_compose import compose_song
    from song_fix import StepUnfixed
    from song_flow import ShowProgress
    try:                                                   # 止める工程が分からなければ、作り始める前に返す（費用をかけない）
        stop_after = _resolve_show_after(show_after, the_form)
    except ValueError as e:
        _trace(ctx.deps.song_id, "draft_melody.out", kind="refused", reason="show_after", given=show_after)
        return f"作り始めなかった: {e}。止めずに最後まで作るなら show_after を空にする。"
    try:
        song = compose_song(ctx.deps.instruction, verbose=False, stop_after=stop_after, form=the_form)
    except ShowProgress as e:                              # ★ここまでで一度出す（保存済み・ロックも抜ける）
        return _progress_message(ctx, e.song, e.step)
    except StepUnfixed as e:
        # ★**形式の所見が直らずに、その工程で止まった**（2026-09-30 本人方針）。失敗ではなく「止まった曲」として返す——
        #   どの工程で何が直らなかったかをミクからマスターへ伝え、頼まれたら止まった工程から続きを作る（`remake_from_step`）
        return _stopped_message(ctx, e)
    except Exception as e:
        # ★**途中で落ちても、続きから作れるように目印を残す**（2026-09-25）。
        #   それまでは、落ちた曲のフォルダには楽譜も `progress.json` も無く、
        #   ①`_retire_scoreless_session` が「曲にならなかった」と見なして `_archive` へ退避し、
        #   ②`load_existing_song` は「見つからない」を返し、**コーディネーターは作り直しを選んだ**
        #   ——済んだ工程（実測でテーマ〜詞の関門の半分近く）をもう一度払う形。
        #   **工程の成果は `_wip/<曲名>.steps.json` に全部残っていた**のに、辿り着く道が無かった。
        #   ★**例外は投げ直す**——失敗の記録と通知（`song_runner`）の動きは変えない。
        #     ここで握って文字列を返すと、コーディネーターが障害中に続きを試みて費用を重ねる。
        _mark_failed(ctx, e)
        raise
    ctx.deps.section = {"kind": "memory", "raw_score": song.score, "song": song}
    # ★**工程の記録を曲フォルダに置く**（2026-09-25・段階3）。`_wip/<曲名>` は作っている最中の
    #   途中保存で、曲名で引く1本なので同じ曲名の別の曲と取り合う。ここに置けば楽譜と同じ版で残る。
    try:
        _name_folder(ctx, song.title() or "")              # 曲名が決まった＝ここで付ける（2026-09-26）
        from session import save_record
        _d = SONGS / ctx.deps.song_id
        _d.mkdir(parents=True, exist_ok=True)
        save_record(_d, song)
    except Exception as e:                                 # 記録が書けなくても曲づくりは止めない
        _trace(ctx.deps.song_id, "draft_melody.warn", record=f"{type(e).__name__}: {e}"[:120])
    rs = song.score
    _trace(ctx.deps.song_id, "draft_melody.out", kind="memory", title=rs.title, form=song.form,
           notes=len(rs.vocal_notes), steps=len(song.history))
    # 止める工程を指定したのに最後まで来た＝その工程がこの曲では走らなかった（ハモリが無くて声部を作らない など）。黙らない
    unstopped = (f"★show_after の『{show_after.strip()}』の工程はこの曲では走らなかった（作るものが無かった）ので、"
                 "止まらずに最後まで作った。マスターにそう伝える。\n" if stop_after else "")
    from song_gate import FORM_STEPS                       # 済んだ工程はこの曲の形で走った工程（2026-10-03）
    done = "→".join(s for s in ("テーマ", "設計", "コード", "旋律", "詞", "編成", "伴奏") if s in FORM_STEPS[song.form])
    if song.form == "instrumental":
        head = (f"オリジナル曲（インスト・歌なし）: 『{rs.title}』（{done}まで済み）。"
                f"主旋律は楽器で弾く。fetch_section_lyrics は不要→次は align_vocal。\n")
    elif song.form == "acappella":
        head = (f"オリジナル曲（アカペラ・伴奏なし）: 『{rs.title}』{len(rs.vocal_notes)}音"
                f"（{done}・ハモリまで済み）。歌詞も生成済みなので fetch_section_lyrics は不要→次は align_vocal。\n")
    else:
        head = (f"オリジナル曲: 『{rs.title}』{len(rs.vocal_notes)}音"
                f"（{done}まで済み）。"
                f"歌詞・伴奏とも生成済みなので fetch_section_lyrics は不要→次は align_vocal。\n")
    return (f"{unstopped}{head}"
            f"{song.summary()}")                           # 点検役の判断・機械の直し（関門-2・2026-09-27 に実装）

@coordinator.tool
@_locked
def fetch_section_lyrics(ctx: RunContext[MusicSession], section_spec: str = "全体",
                         use_knowledge: JsonBool = False) -> str:
    """工程3: 区間の歌詞（かな全文）をネット→（音声参照なら）聴取→知識の順で取得し deps.lyrics に格納。
    先に draft_melody が必要。記憶生成（参照なし）の場合は歌詞生成済みなので呼ばず align_vocal へ。
    **use_knowledge=True**: ネット検索が**同名異曲の歌詞**を返し続ける時（検問がそう指摘した時）の脱出口＝
    LLMの記憶から取得する（定番曲・童謡など超有名曲向け。検索語を変えても同じ結果なら迷わずこちら）。"""
    st = ctx.deps.section
    _trace(ctx.deps.song_id, "fetch_section_lyrics.in", section_spec=section_spec,
           kind=(st or {}).get("kind"), use_knowledge=use_knowledge)
    if st is None:
        return "区間が未用意。先に draft_melody を呼ぶこと。"
    if st.get("kind") == "memory":
        return "記憶生成は歌詞も生成済み。fetch_section_lyrics は不要→align_vocal を呼ぶ。"
    from song_fix import StepUnfixed
    try:
        kana, src = lyrics_for_section(st, ctx.deps.instruction, force_knowledge=use_knowledge)
    except StepUnfixed as e:                               # 歌詞を取る LLM の形式が直らずに止まった（2026-10-02。以前は曲づくりごと落ちた）
        return _cover_stopped(ctx, "fetch_section_lyrics", e, how=(
            "**歌詞はまだ取っていない**。続けるときは fetch_section_lyrics をもう一度呼ぶ（同名異曲が疑わしければ use_knowledge=True）。"
            "同じ所で止まるなら最終報告でマスターに伝えて終える。"))
    # ★**どこからも取れなかったら、空の歌詞を保存せず・整列へ進めずに返す**（2026-10-03・再レビュー・カバーの組）。以前は
    #   「歌詞取得(knowledge): 0字。次は align_vocal」と成功の形で返していた。取れなかったことは取得元 `LYRICS_NONE` で分かる
    from align_cover import LYRICS_NONE
    if src == LYRICS_NONE or not kana.strip():
        _trace(ctx.deps.song_id, "fetch_section_lyrics.out", source=src, chars=0, kind="none")
        return ("歌詞が取れなかった（ネットの歌詞・音声の聴き取り・知識のどれも空）。align_vocal へは進まない。"
                + ("" if use_knowledge else
                   "同名異曲の疑いがなければ fetch_section_lyrics(use_knowledge=True) を1回試し、")
                + "それでも取れなければ、マスターに歌詞か歌詞ページの URL をもらえないか、最終報告で確認して終える。")
    ctx.deps.lyrics = kana
    try:                                                   # 歌詞も中間生成物として保存（編集時の再整列で再fetch不要）
        sd = SONGS / ctx.deps.song_id / "source"
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "lyrics.txt").write_text(kana, encoding="utf-8")
    except Exception:
        pass
    from lyric_slots import kana_text_chars                # かなだけ（空白・句読点は型が差し戻す・2026-10-02・案B）。
    n = len(kana_text_chars(kana))                         # 行の切れ目（改行）は定義された区切りとして数えない（2026-10-03）
    _trace(ctx.deps.song_id, "fetch_section_lyrics.out", source=src, chars=n)
    # Web から取った歌詞は出典の URL を見せる（取得元は `web: <URL>`）。マスターに出典を伝えられるように
    shown = f"Web・出典 {src[len('web:'):].strip()}" if src.startswith("web:") else src
    return f"歌詞取得({shown}): {n}字。次は align_vocal で旋律へ整列。"

@coordinator.tool
@_locked
def align_vocal(ctx: RunContext[MusicSession], revise_pitch: JsonBool = False) -> str:
    """工程4: 区間の旋律へ歌詞を整列して歌声スコアを作る（deps.score）。
    音声参照=WhisperX強制アラインメント／MIDI=記号整列／記憶生成=共通アライナ。
    先に draft_melody（カバーは fetch_section_lyrics も）が必要。
    revise_pitch: 音声カバーの音高をLLMで微修正（修正依頼時のみTrue。初回はFalse=純プログラム）。MIDI/記憶は無関係。"""
    st = ctx.deps.section
    _trace(ctx.deps.song_id, "align_vocal.in", kind=(st or {}).get("kind"), revise_pitch=revise_pitch,
           lyrics_chars=len(ctx.deps.lyrics or ""))
    if st is None:
        return "区間が未用意。先に draft_melody を。"
    if st.get("kind") == "memory":
        # ★**新しいオリジナル経路は再整列しない**（2026-09-01）。
        #   `song_compose` は**フレーズ単位で音数ちょうどに詞を当て**、そのうえで
        #   音符ごとの調声（ビブラート・しゃくり・語尾）まで付けている。
        #   `align_score_lyrics` は `vocal_notes` を丸ごと置き換えるので、通すと**当てはめも調声も消える**。
        #   ※`song` を持たない旧経路（歌詞全文だけを持つ）を `align_score_lyrics` へ回す分岐は 2026-09-30 に外した
        #     （`memory` の区間を作るのは `draft_melody` だけで、必ず `song` を持つ）
        ctx.deps.score = st["raw_score"]
        ctx.deps.faithful = False
        s = ctx.deps.score
        _trace(ctx.deps.song_id, "align_vocal.out", kind="memory", title=s.title, notes=len(s.vocal_notes))
        if getattr(s, "form", "song") == "instrumental":   # インストは歌が無い（2026-10-03）
            return (f"整列(記憶生成): 『{s.title}』はインスト（歌なし）なので、整列する歌は無い。"
                    "伴奏と主旋律は render_final が一括で書き出す→次は render_final（曲全体は曲づくりの中で最終の点検役が見たので、"
                    "最終検問 review_progress は要らない）。")
        # 高さは国際式の音名（`C4`＝60・曲の表と同じ綴り）。2026-10-01 まで階名（`ド4`）で返していた
        return (f"整列(記憶生成): 『{s.title}』{len(s.vocal_notes)}音 / 音名: "
                + " ".join(spell_pitch(n.key) for n in s.vocal_notes) +
                "。オリジナルは伴奏も render_final が自動生成→build_accompaniment不要、次は evaluate_song。")
    if not ctx.deps.lyrics:
        return "歌詞が未取得。先に fetch_section_lyrics を。"
    from song_fix import StepUnfixed
    from song_flow import NOT_SWALLOW
    try:
        ctx.deps.score = align_section(st, ctx.deps.lyrics, pitch_correct=revise_pitch)
    except StepUnfixed as e:                               # 歌詞の割り当て・音高補正の LLM の形式が直らずに止まった（2026-10-02）
        return _cover_stopped(ctx, "align_vocal", e, how=(
            "**歌はまだ作っていない**（区間と歌詞はそのまま）。続けるときは align_vocal をもう一度呼ぶ"
            + ("（音高補正で止まったなら revise_pitch=False で補正なしにもできる——その旨をマスターに伝える）"
               if revise_pitch else "") + "。同じ所で止まるなら最終報告でマスターに伝えて終える。"))
    except NOT_SWALLOW:
        raise
    except Exception as e:                                 # 整列の内部例外もセッション死でなく方向転換へ（error_handling規約）
        _trace(ctx.deps.song_id, "align_vocal.out", kind="error", reason=f"{type(e).__name__}")
        return (f"整列に失敗した（{type(e).__name__}: {e}）。参照や区間が壊れている可能性が高い。"
                f"fetch_reference で参照を取り直すか draft_melody で区間を取り直して。続かなければマスターに確認して終了して。")
    ctx.deps.faithful = True
    _save_section(ctx.deps.song_id, st)                    # 再整列で区間（start_sec等）が動くことがある→常に最新を永続化
    s = ctx.deps.score
    # ★**実測が区間を上書きしたこと・割り当ての過不足を埋めたことを返す**（2026-09-27・目録の道具-10・道具-11）
    _seen = [f"・{x}" for x in (st.get("overrides") or [])] + \
            [f"・割り当て: {x}" for x in (st.get("assign_findings") or [])]
    if _seen:
        _append_log(ctx.deps.song_id, "[整列] 機械が決めたこと:\n" + "\n".join(_seen))
    _machine = ("\n【整列で機械が決めたこと】\n" + "\n".join(_seen)) if _seen else ""
    _trace(ctx.deps.song_id, "align_vocal.out", kind=st['kind'], notes=len(s.vocal_notes),
           revise_pitch=revise_pitch)
    if st.get("head_mismatch"):                            # 実音と割り当て歌詞の食い違い＝区間の取り違え（黙って完成させない）
        _trace(ctx.deps.song_id, "align_vocal.warn", head="mismatch")
        return (f"整列はしたが**歌詞と実際の歌が食い違っている**: {st['head_mismatch']}\n"
                f"このまま先に進めると『別の場所の音に別の歌詞を貼った曲』になる。"
                f"draft_melody で区間を取り直す（section_spec を『サビ』『1番』など変えて指定し直す）か、"
                f"fetch_section_lyrics で歌詞をその区間のものに取り直してから align_vocal をやり直して。"
                f"直らなければマスターに確認する最終報告で終了して。" + _machine)
    span = s.duration_ticks() / 480 * 60.0 / s.tempo       # スコア実時間（秒）
    dens = len(s.vocal_notes) / span if span > 0 else 0.0
    durs = sorted(n.duration_tick for n in s.vocal_notes)
    med = (durs[len(durs) // 2] / 480 * 60.0 / s.tempo) if durs else 1.0
    if dens > 8.0 or med < 0.09:                           # 主判定=音符長中央値<90ms（破綻実測: 12345=72ms/railgun=53ms）。
        # 密度単独は8.0まで許容（テンポ250のカットタイム曲が正当に6.1音/秒＝旧閾値6.0は偽陽性だった・新地T2B実測）
        _trace(ctx.deps.song_id, "align_vocal.warn", density=f"{dens:.1f}/s", median_dur=f"{med * 1000:.0f}ms")
        return (f"整列はしたが**結果が異常**: {len(s.vocal_notes)}音/{span:.0f}秒（{dens:.1f}音/秒・音符長中央値{med * 1000:.0f}ms）。"
                f"歌として物理的に成立しない＝区間の切り出しか参照の中身（伴奏混在MIDI等）が崩れている可能性が高い。"
                f"このまま先の工程に進まず、作りかけを完成扱いにしないこと。"
                f"マスターに『整列が崩れたので確認してほしい』と報告する最終報告で終了して。" + _machine)
    return (f"整列(忠実カバー・{st['kind']}{'＋LLM音高補正' if revise_pitch else ''}): "
            f"『{s.title}』{len(s.vocal_notes)}ノート。次は build_accompaniment で伴奏を用意。" + _machine)

@coordinator.tool
@_locked
def build_accompaniment(ctx: RunContext[MusicSession]) -> str:
    """工程5: 区間の伴奏を用意して accompaniment.wav と accompaniment.mid に書き出す。
    MIDI参照=元MIDIの主旋律以外を区間で切り出して SoundFont でレンダ／**音声参照=YourMT3で多楽器MIDI化→フリー音源レンダ（外部公開対応＝原曲音源でなく自作）**。
    YourMT3は venv_mt3 を subprocess 呼び。失敗時は demucs 実楽器スライスにフォールバック。
    先に align_vocal（音声は強制アラインメントで伴奏尺が確定）が必要。記憶生成では不要（render_final が伴奏生成）。"""
    st = ctx.deps.section
    _trace(ctx.deps.song_id, "build_accompaniment.in", kind=(st or {}).get("kind"))
    if st is None or ctx.deps.score is None:
        return "区間/歌声が未用意。先に draft_melody→…→align_vocal を。"
    if st.get("kind") == "memory":
        return "記憶生成（オリジナル）は伴奏を render_final が自動生成する。build_accompaniment は不要→render_final へ。"
    from song_schema import TICKS_PER_QUARTER as _Q
    from render_base import load_wav
    s = ctx.deps.score
    song_dir = SONGS / ctx.deps.song_id
    cover = s.duration_ticks() * 60.0 / s.tempo / _Q if s is not None else 0.0   # スコア全長＝伴奏も最低この長さをカバー
    if st["kind"] == "audio":
        # 外部公開対応: 伴奏を **YourMT3 で多楽器MIDI化→フリーSoundFontでレンダ**（原曲音源でなく"自作"＝公開可）。
        # YourMT3は本番venvで動かないので venv_mt3 を subprocess 呼び。start=歌声origin に合わせて採譜＝歌と整合。
        # 【基準一本化】score.origin_abs_sec（align が確定した唯一の基準）を最優先。st["start_sec"] もこれに同期し、
        # 以降の save_cut_source / _save_section_bounds / accompaniment_for_section も同じ origin を使う（取り違え防止）。
        origin = s.origin_abs_sec if (s is not None and s.origin_abs_sec is not None) else st["start_sec"]
        st["start_sec"] = origin
        alen, kind, unpub = _audio_accomp(ctx, st, song_dir, cover)
        _save_section_bounds(ctx.deps.song_id, st)          # 区間境界を永続化（再スライス/再解析用）
        save_cut_source(st, song_dir)                       # 【検証用】切り出し範囲の元音源を保存
    else:                                                  # MIDI参照: 元MIDIの伴奏を SoundFont でレンダ
        # 伴奏 MIDI も曲フォルダに残す（2026-09-30）——音声カバーと同じく、伴奏の譜面編集・移調・尺の変更・
        # コード推定が accompaniment.mid を前提にしている。時間の原点は楽譜の tick 0（区間の頭）。
        alen = write_accompaniment(song_dir, accompaniment_for_section(
            st, cover_sec=cover, mid_out=song_dir / "accompaniment.mid"))
        _keep_original_accomp(song_dir)
        kind = "元MIDI伴奏(SoundFont)＋.mid"
        unpub = ""
    ctx.deps.accomp_bounds = (st.get("start_sec"), st.get("end_sec"))  # どの区間で切ったかを記録（stale検知用）
    _save_accomp_bounds(ctx.deps.song_id, ctx.deps.accomp_bounds)     # 永続化＝セッションをまたいでも検知可能に
    _trace(ctx.deps.song_id, "build_accompaniment.out", kind=kind, acc_sec=round(alen / SR, 1),
           acapella=(alen == 0), section_sec=f"{st.get('start_sec', 0):.1f}-{st.get('end_sec', 0):.1f}")
    if 0 < alen / SR < cover * 0.5:                        # 伴奏が歌の半分未満＝切り出し失敗（railgun: 歌16sに伴奏1s→無音mix）
        return (f"伴奏用意({kind})したが**異常**: {round(alen / SR, 1)}s しか書けていない（歌は{cover:.1f}s）"
                f"＝伴奏の切り出しが失敗している。このまま進めると伴奏がほぼ無音の曲になる。"
                f"参照や区間が壊れていないか点検し、直せなければマスターに確認する最終報告で終了して。")
    return (f"伴奏用意({kind}): {round(alen / SR, 1)}s を accompaniment.wav に書出"
            f"{'（区間が無音→伴奏なし＝アカペラ）' if alen == 0 else ''}。次は render_and_mix。"
            + (f"\n{unpub}" if unpub else ""))


#: **原曲から分離した音を伴奏に使ったときの知らせ**（2026-09-27・目録の道具-9・公開性の軸）。
#: 分離した原曲の音は、原曲の録音そのもの——**ネットに出せない**。以前は返事の括弧書き（「実楽器(分離…)」）にしか出ず、
#: 作り直し（`render_and_mix` の区間変更）では何も言わずにこちらへ切り替えていた。
UNPUBLISHABLE = ("★**この伴奏は原曲の録音から分離した音**（{why}）。**ネットに公開できない**"
                 "（原曲の録音そのものが入っている）。自分で聴くだけなら使える。公開したいなら、"
                 "YourMT3 で採り直せる環境で build_accompaniment をやり直す。最終報告でマスターに必ず伝える。")


def _keep_original_accomp(song_dir: Path) -> None:
    """原曲から作った伴奏 MIDI の写しを `source/` に置く（2026-10-02・組「カバーのコード」）。アレンジし直すと作業台の
    `accompaniment.mid` は新しい編曲で書き直されるので、カバーのコードの材料（原曲の伴奏からの推定・`song_cover_chords`）はこちらを読む。"""
    import shutil
    from song_cover_chords import ORIGINAL_ACCOMP
    mid = song_dir / "accompaniment.mid"
    if mid.exists():
        (song_dir / "source").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(mid, song_dir / "source" / ORIGINAL_ACCOMP)


def _audio_accomp(ctx, st: dict, song_dir: Path, cover: float) -> tuple[int, str, str]:
    """音声カバーの伴奏を作る。**まず YourMT3（採譜して自前の音源で鳴らす＝公開できる）**、だめなら分離した原曲音。

    戻り値は (長さ, 種類, 公開できないときの知らせ（できるなら空）)。`build_accompaniment` と
    `render_and_mix`（区間が変わったときの作り直し）の両方がここを通る——作り直しだけ分離音に落ちていた。
    """
    from render_base import load_wav
    start_sec = st["start_sec"]
    end_sec = max(st["end_sec"], start_sec + cover)
    try:
        from reconstruct_accompaniment import reconstruct_accompaniment_mt3
        r = reconstruct_accompaniment_mt3(Path(st["vocals_wav"]).parent, start_sec, end_sec, song_dir)
        alen = len(load_wav(song_dir / "accompaniment.wav"))
        _keep_original_accomp(song_dir)
        return alen, f"YourMT3多楽器MIDI(公開用)＋.mid【{r.get('info', '')}】", ""
    except Exception as ex:                                # 失敗時は demucs 実楽器スライスへ（知らせて記録する）
        alen = write_accompaniment(song_dir, accompaniment_for_section(st, cover_sec=cover))
        why = f"YourMT3 で採れなかった: {type(ex).__name__}"
        note = UNPUBLISHABLE.format(why=why)
        _trace(ctx.deps.song_id, "accompaniment.unpublishable", reason=why)
        _append_log(ctx.deps.song_id, f"[伴奏] ★公開できない伴奏（原曲から分離した音）を使った: {why}")
        return alen, f"実楽器(分離・YourMT3不可:{type(ex).__name__})", note

@coordinator.tool
@_locked
def render_and_mix(ctx: RunContext[MusicSession]) -> str:
    """工程6: 歌声スコア→score.json/vpr/仮歌を書き、伴奏を歌の長さに揃える。
    音量バランスは自動（歌声が伴奏より前に出る）。伴奏全体の音量を指定する引数は外した（効いていなかった・2026-09-27）。
    先に align_vocal（カバーは build_accompaniment も）が必要。完了後 evaluate_song→（必要なら fix）→ render_final。"""
    st = ctx.deps.section
    _trace(ctx.deps.song_id, "render_and_mix.in", kind=(st or {}).get("kind"))
    if ctx.deps.score is None:
        return "歌声が未生成。先に align_vocal を。"
    if st is not None and st.get("kind") == "memory":
        return "記憶生成（オリジナル）は render_final が一括書出する。render_and_mix は不要。"
    from tools.editing import form_refusal                 # インストに vpr を書こうとして落ちていた（2026-10-03・再レビュー）
    if why := form_refusal(ctx, "render_and_mix"):
        return why
    song_dir = SONGS / ctx.deps.song_id
    # stale検知: align_vocal 再実行で区間が変わったのに伴奏が旧境界のままなら、伴奏を最新区間で自動再生成
    # ★**作り直しも build_accompaniment と同じ道（YourMT3 → だめなら分離音＋知らせ）を通す**（2026-09-27・道具-9）。
    #   以前はここだけ**黙って分離した原曲音**で作り直していた（公開できない音に入れ替わっても誰も知らない）。
    regen = ""
    if st is not None and st.get("kind") == "audio":
        cur = (st.get("start_sec"), st.get("end_sec"))
        if ctx.deps.accomp_bounds is not None and cur != ctx.deps.accomp_bounds:
            from song_schema import TICKS_PER_QUARTER as _Q
            cover = ctx.deps.score.duration_ticks() * 60.0 / ctx.deps.score.tempo / _Q
            _alen, _kind, _unpub = _audio_accomp(ctx, st, song_dir, cover)
            ctx.deps.accomp_bounds = cur
            _save_accomp_bounds(ctx.deps.song_id, cur)
            _append_log(ctx.deps.song_id, f"[整合] 区間変更を検知→伴奏を最新区間{cur}で再生成（stale防止・{_kind}）")
            regen = (f"\n区間が変わったので伴奏を作り直した（{_kind}）。" + (f"\n{_unpub}" if _unpub else ""))
    vlen, alen = render_vocal_and_mix(ctx.deps.score, song_dir, log=(st.get("log") if st else None))
    if st is not None and st.get("log"):                  # カバー内部トレースも統一ログ(build_log.txt)へ
        _append_log(ctx.deps.song_id, "--- カバー内部トレース ---\n" + "\n".join(st["log"]))
    _trace(ctx.deps.song_id, "render_and_mix.out", vocal_sec=round(vlen / SR, 1),
           acc_sec=round(alen / SR, 1))
    return (f"下書き書出: output/{ctx.deps.song_id}/（score.json/vpr/仮歌）"
            f" 歌声{round(vlen / SR, 1)}s/伴奏{round(alen / SR, 1)}s。"
            f"次は evaluate_song で自己評価→問題なければ render_final。" + regen)

@coordinator.tool
@_locked
def fetch_reference(ctx: RunContext[MusicSession], song_name: str, hint_url: str = "") -> str:
    """原曲のお手本を取得する（カバーの前段）。まず MIDI を検索(Tavily/grounding)してDL・主旋律ch選定、
    無ければ音声(MP3)をDL→demucs分離→採譜にフォールバック。採用参照を deps と source/ に記録（修正時に復元可）。
    **どのDLファイルを“正”として採用したか・各候補の同定結果は source/fetch_log.txt に必ず記録**する。
    結果として MIDI 参照 か 音声参照 のどちらかが deps にセットされ、次に draft_melody を呼べる。
    **hint_url**: 指示や会話に**公式ページ/音源のURL**があれば渡す（検索結果より優先して使う＝同名異曲の決定打）。
    指示文に http で始まるURLが書かれていたら必ずここに入れること。"""
    # ★**URL でない hint_url は黙って捨てない**（2026-10-02・洗い出し 2-3）。以前は http で始まらなければ無かったことにして検索した
    if (hint_url or "").strip() and not re.match(r"https?://", hint_url.strip()):
        return (f"取りに行かなかった: hint_url『{hint_url}』は URL ではない（http:// か https:// で始まる URL を書く）。"
                "URL が無ければ hint_url を空にして呼び直す。")
    hint_url = (hint_url or "").strip()
    ctx.deps.reference_json = ctx.deps.reference_midi = None   # 再取得時に前回の参照状態を必ずリセット
    ctx.deps.melody_channel = ctx.deps.reference_audio_accomp = None  # （MIDI/音声を排他に保つ）
    src = SONGS / ctx.deps.song_id / "source"       # 参照素材は曲フォルダ内 source/ に隔離
    src.mkdir(parents=True, exist_ok=True)
    hints = [hint_url] if str(hint_url).startswith("http") else None
    _trace(ctx.deps.song_id, "fetch_reference.in", song_name=song_name, hint_url=hint_url or "なし")
    info = _fetch_ref(song_name, f"ref_{ctx.deps.song_id}.mid", prefer_vocal=True, out_dir=src,
                      hint_urls=hints)
    _trace(ctx.deps.song_id, "fetch_reference.midi", ok=info.get("ok"),
           candidates=len(info.get("candidates", [])), chosen=Path(info["midi_path"]).name if info.get("ok") else None,
           identity=info.get("identity"), need_clarify=info.get("need_clarification", False), reason=info.get("reason", ""))
    if info.get("need_clarification"):                      # C: MIDIが同名異曲で確定不能→聞き返し（音声へ流さない）
        _write_fetch_log(src, song_name, midi_info=info, audio_info=None, adopted=None)
        return _clarify_msg(song_name, "MIDI", info)
    if not info.get("ok"):                                   # MIDI が無ければ音声(MP3)参照へフォールバック
        # ★**音声経路は GPU が要る**（分離・採譜・強制アラインメント）。無い環境では
        #   ここで止めて、**「知っている範囲で新しく1曲作る」提案に切り替える**（2026-09-02）。
        #   ★勝手にオリジナルへ滑り込ませない——参照ゼロのまま作ると
        #   「**存在しない曲の幻覚カバー**」になる（E2E T7 で実測済み）。**必ず人に確認する。**
        if not _audio_cover_available():
            _trace(ctx.deps.song_id, "fetch_reference.out", kind="refused",
                   reason="MIDI 無し・音声経路は GPU が必要")
            _write_fetch_log(src, song_name, midi_info=info, audio_info=None, adopted=None)
            return (f"『{song_name}』の楽譜（MIDI）が見つからなかった。"
                    f"音源から採るには**音を分けて採譜する処理**が要るんだけど、"
                    f"この環境には NVIDIA GPU が入っていないからそこまではできないんだ。\n"
                    f"**代わりにできること**: 自分が知っているその曲の雰囲気をもとに、"
                    f"**新しく1曲作る**ことはできる。ただし**原曲の再現ではない**から、"
                    f"別の曲になる。\n"
                    f"→ マスターに『原曲そのままは無理だけど、"
                    f"雰囲気を汲んで新しく作るのはどう？』と**聞いてから**終了して。"
                    f"『それでいい』と言われたら、次の依頼で"
                    f"**オリジナル曲として**作る（そのときは指示にその旨を明記してもらう）。")
        ainfo = _fetch_audio_ref(song_name, out_dir=src, hint_urls=hints)
        _trace(ctx.deps.song_id, "fetch_reference.audio", ok=ainfo.get("ok"),
               url=ainfo.get("audio_url"), identity=ainfo.get("identity"),
               need_clarify=ainfo.get("need_clarification", False), reason=ainfo.get("reason", ""))
        if ainfo.get("need_clarification"):                # C: 音声候補が同名異曲で確定不能→聞き返し
            _write_fetch_log(src, song_name, midi_info=info, audio_info=ainfo, adopted=None)
            return _clarify_msg(song_name, "音声", ainfo)
        if not ainfo.get("ok"):
            _write_fetch_log(src, song_name, midi_info=info, audio_info=ainfo, adopted=None)
            return (f"参照取得 失敗（MIDI/音声とも）: MIDI={info.get('reason')} / 音声={ainfo.get('reason')}。"
                    f"『{song_name}』のお手本が見つからなかった。**カバー依頼なら想像で作ってはいけない**"
                    f"（実在曲の再現にならない）ので、記憶生成（draft_melody）へは進まず、ここで作業を止めて"
                    f"「曲名が正しいか・音源のURLやファイルをもらえないか」をマスターに確認する最終報告で終了すること。")
        ctx.deps.reference_json = ainfo["reference_json"]
        ctx.deps.reference_audio_accomp = ainfo.get("accompaniment_wav")  # 実楽器の伴奏（demucs分離）
        _save_ref_index(src, type="audio", url=ainfo.get("audio_url"),  # 採用参照＋URLを記録（修正時に復元）
                        solfa=Path(ainfo["reference_json"]).name,
                        accomp=Path(ainfo["accompaniment_wav"]).name if ainfo.get("accompaniment_wav") else None)
        _write_fetch_log(src, song_name, midi_info=info, audio_info=ainfo, adopted=("audio", ainfo.get("audio_url")))
        return (f"参照採用(音声・音源分離採譜): {ainfo['audio_url']} 同定={ainfo.get('identity')}"
                f"「{ainfo.get('identity_reason')}」 {ainfo['note_count']}音 tempo{ainfo['tempo']}"
                f" 伴奏インスト={'あり' if ainfo.get('accompaniment_wav') else 'なし'}。source/fetch_log.txt に記録。次は draft_melody。")
    ctx.deps.reference_json = info["reference_json"]
    ctx.deps.reference_midi = info.get("midi_path")          # 忠実コピー用にMIDI実体とchも保持
    ctx.deps.melody_channel = info.get("chosen_channel")
    adopted_file = Path(info["midi_path"]).name
    _save_ref_index(src, type="midi", url=info.get("midi_url"),  # 採用参照＋URLを記録（修正時に復元）
                    midi=adopted_file, melody_channel=info.get("chosen_channel"))
    _write_fetch_log(src, song_name, midi_info=info, audio_info=None,
                     adopted=("midi", adopted_file, info.get("midi_url")))
    cands = info.get("candidates", [])
    comp = "; ".join(f"{c.get('file')}={c['note_count']}音/ch{c['ch']}/score{c['score']}/同定{c.get('match')}" for c in cands)
    return (f"参照採用(MIDI): {adopted_file} ← {info['midi_url']} (ch{info['chosen_channel']}・{info['note_count']}音・"
            f"score{info['chosen_score']}・同定={info.get('identity')}「{info.get('identity_reason')}」)。"
            f"候補{len(cands)}件[{comp}]。source/fetch_log.txt に詳細記録。次は draft_melody。")

@coordinator.tool
@_locked
def render_final(ctx: RunContext[MusicSession]) -> str:
    """確定した曲を vpr/伴奏/仮歌ミックスに書き出す（実歌声はこの次の `export_vocal` で作る）。
    アカペラは伴奏を作らない。**インスト（歌なし）はここで完成品（`master_inst_v{N}.wav`）まで書く**（export_vocal は無い）。
    フォルダ名は YYYYMMDD_HHMMSS[_タイトル] で統一。"""
    if ctx.deps.score is None:
        return "まだ曲が無い。先に draft_melody→…→align_vocal で曲を作る。"
    _trace(ctx.deps.song_id, "render_final.in", faithful=ctx.deps.faithful, title=ctx.deps.score.title)
    # リネームは faithful/オリジナル共通（オリジナルだけ compose が新名フォルダを別に作り、
    # 旧フォルダが序盤ログごと残骸化していた）。オリジナルは draft_melody の時点で済んでいる（2026-09-26）
    _name_folder(ctx, ctx.deps.score.title)
    new_dir = SONGS / ctx.deps.song_id
    if ctx.deps.faithful:                              # v2 忠実コピー: 伴奏(MIDIコピー)を再生成せずフォルダを整える
        ver = ensure_working_version(new_dir)          # ★書き出した版の印を置く（次の export_vocal が版を決められるように・出-3）
        _trace(ctx.deps.song_id, "render_final.out", mode="faithful", folder=ctx.deps.song_id, version=ver)
        return (f"書き出し完了(忠実コピー): {ctx.deps.song_id}/（vpr/伴奏）。{NEXT_EXPORT}")
    # ★**書き出した版の印を置く**（2026-09-30・出-3）。新しい曲には印が無く、直後の export_vocal が「版が分からない」と断っていた。
    #   ★**書き出しの前に置く**（2026-10-03）——インストは書き出し（`compose`）が完成品 `master_inst_v{N}.wav` まで書くので、
    #   その版 N を `compose` の時点で引けるように（作業中の印 `_WORKING_v{N}`）
    new_dir.mkdir(parents=True, exist_ok=True)
    ver = ensure_working_version(new_dir)
    form = getattr(ctx.deps.score, "form", "song")
    # ★**インストは完成品をここで書くので、その前に点検待ちの束を見る**（2026-10-04・案C）。歌もの・アカペラは export_vocal の頭で見る。
    #   NG でも書き出す（所見を返すだけ・マスターが聴いて決める）
    from tools.editing import review_edits
    gate = review_edits(ctx, "render_final") if form == "instrumental" else ""
    meta = compose(ctx.deps.song_id, ctx.deps.score)       # 声のIDは compose の既定（MIKU_V6_COMP_ID）。形の分岐は compose の中
    _trace(ctx.deps.song_id, "render_final.out", mode="original", form=form, status=meta['status'], version=ver)
    if form == "instrumental":
        # インストは V6 を使わない（F8・§5-3）。完成品は書き出しが伴奏に仕上げを当てて書く
        done = song_files.master("instrumental", ver)
        if not (new_dir / done).exists():
            return (f"書き出しはしたが、完成品 {done} ができていない（status={meta['status']}）。"
                    "インスト（歌なし）なので export_vocal は無い——不具合としてマスターに伝える。")
        return (f"書き出し完了（インスト・歌なし）: {meta['song_id']}/{done} が完成品。status={meta['status']} "
                f"validate={meta['validate_issues'] or 'なし'}。**曲は完成**——歌声の書き出し（export_vocal）は無い。"
                "このまま play_song で流せる（ミクは歌わずに流す）。" + gate)
    return (f"書き出し完了: {meta['song_id']}/（vpr/{'仮歌（伴奏なし・アカペラ）' if form == 'acappella' else '伴奏/仮歌'}）"
            f" status={meta['status']} "
            f"validate={meta['validate_issues'] or 'なし'}。{NEXT_EXPORT}")


def _score_form(ctx: RunContext[MusicSession], song_dir: Path, ver: int) -> str:
    """いまの曲の形（楽譜 `SongScore.form`）。開いている楽譜が無ければ、作業台の楽譜→その版の楽譜から読む。
    欄の無い古い楽譜・楽譜が無い曲は song（楽譜は機械が書くので、読めない綴りは不具合として `ValueError`）。"""
    s = ctx.deps.score
    if s is not None:
        return getattr(s, "form", "song") or "song"
    for name in (song_files.SCORE, song_files.versioned(song_files.SCORE, ver)):
        p = song_dir / name
        if p.exists():
            return song_files.as_form(json.loads(p.read_text(encoding="utf-8")).get("form"))
    return "song"


@coordinator.tool
@_locked
def export_vocal(ctx: RunContext[MusicSession]) -> str:
    """**ミクの歌声を書き出して曲を完成させる**（VOCALOID6 を自動操作する）。

    `render_final` の次に呼ぶ。これが通れば完成品（歌もの `master_song_v{N}.wav`・アカペラ `master_acappella_v{N}.wav`）ができ、
    マスターは**聴くだけ**でよくなる（V6 を開いて操作する手間が要らない）。
    **インスト（歌なし）には歌声の書き出しは無い**（render_final で完成している）——呼んでもそう返すだけで V6 は起こさない。

    VOCALOID6 が無い環境では**書き出さずにその旨を返す**。曲そのものは仮歌ミックスまで
    出来上がっているので、マスターに書き出しを頼めばこれまでどおり完成できる。
    """
    settle_folder_title(ctx.deps, SONGS)                   # 曲名の作り直しで付け替えられなかった曲フォルダの名前を、V6 を起こす前にもう一度（2026-10-03）
    song_dir = SONGS / ctx.deps.song_id
    if not song_dir.exists():
        # ※ここは記録に残さない（`_trace` は曲フォルダを作ってしまう）。断った理由は返事に書いてある
        return "まだ曲フォルダが無い。先に render_final まで通す。"
    # ★**作業中と確定済みの両方を見る**（2026-09-06 修正）。確定済み（`_CURRENT_v{N}`）は、前のセッションで
    #   確定した版を開き直して書き出すときのため。作業中の印は `render_final` が置く（2026-09-30・出-3）——
    #   それまで新しい曲には印が無く、render_final の直後でもここで「版が分からない」と断っていた
    #   （2026-09-06 のコメントは「render_final を通ると版は確定し作業中の印は消える」と書いていたが、
    #   確定はセッションの終わりの `end_session` で、render_final ではない）。
    ver = _working_version(song_dir) or _current_version(song_dir)
    if not ver:
        # ★**断ったことを記録に残す**（2026-09-30・出-3）。以前は記録に何も残らず、build_log からは遠回りの理由が追えなかった
        _trace(ctx.deps.song_id, "export_vocal.refused", reason="版の印（_WORKING_v/_CURRENT_v）が無い",
               fixed=",".join(p.name for p in sorted(song_dir.glob("vocal*.vpr"))) or "なし")
        return "版が分からない。先に render_final を通す。"

    form = _score_form(ctx, song_dir, ver)
    if form == "instrumental":
        # ★**インストは V6 を起こさない**（2026-10-03・§5-3・F8）。曲は render_final で完成している
        _trace(ctx.deps.song_id, "export_vocal.refused", reason="インスト（歌なし）", version=ver)
        done = song_files.master("instrumental", ver)
        return (f"インスト（歌なし）なので歌声の書き出しは無い。曲は render_final で完成している（{done}）"
                + ("" if (song_dir / done).exists() else "——ただしその完成品がまだ無いので、先に render_final を通す")
                + "。")
    # ★**書き出しの入口で、点検待ちの束を見る**（2026-10-04・案C）。大きい編集があれば最終の点検役を1回呼ぶ（V6 を起こす前——
    #   書き出しは数分かかる）。**NG でも書き出す**（所見を返すだけ・直す案は提案。マスターが聴いて決める）
    from tools.editing import review_edits
    gate = review_edits(ctx, "export_vocal")
    _trace(ctx.deps.song_id, "export_vocal.in", version=ver, form=form)
    ok, detail = v6_export.export_song_vocal(song_dir, ver)
    _trace(ctx.deps.song_id, "export_vocal.out", ok=ok, detail=detail)
    if ok:
        return (f"歌声を書き出した: {detail}。**曲は完成**（{song_files.master(form, ver)} が完成品）。"
                f"このまま play_song で聴かせられる。" + gate)
    # 失敗しても曲は残っている。**人手の道が今までどおり残っている**ことを必ず伝える
    return (f"自動での書き出しはできなかった（{detail}）。"
            f"vocal_v{ver}.vpr を VOCALOID6 で開いてミックスダウンし、"
            f"**既定の名前のまま**曲フォルダに保存してもらえば完成する。" + gate)
