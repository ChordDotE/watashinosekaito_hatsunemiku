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

"""音楽の作成・編集ツール（song_service への単一入口）。

自然文の指示をそのまま音楽コーディネーター（song_service・別プロセス/別venv）へ渡す。
作成・編集・mix・問い合わせの区別はコーディネーター側の LLM が判断する（ミク側の分岐なし）。

- **全部 Fire&Forget**: subprocess を起動して即返す（作成15〜20分・mixも数十秒〜のため待たない）。
  完了はミクが次の会話で songs.md / _last_run.json のスナップショットから気づく
- **同期戻り値に現状スナップショット**を同梱: 曲一覧・直近実行の状態。一覧/状態の質問には
  ミクがこの戻り値だけで即答できる
- **lockは直列・キューなし**: 実行中の新依頼は Failure(retryable)（「いま作業中」）

権限レベル: 2 (external寄り・ローカル実行)
副作用: song_service のサブプロセスを起動し、曲フォルダ（miku/data/songs/）を生成・更新する
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from agent.config import (get_song_service_dir, get_songs_dir, is_song_service_available,
                          load_settings)
from agent.schemas import Failure, Success
from song_service import song_files

_settings = load_settings()


class WorkOnMusicInput(BaseModel):
    instruction: str = Field(description="これから自分（ミク）が取り組む曲づくりの内容を自然文で"
                                         "（曲名・やってほしいこと・『さっきの曲』等の文脈は具体名に展開して書く）")
    # ★**マスターが口ずさんだ音**（2026-09-23）。鼻歌・口笛・肉声を旋律の手本として使う。
    #   **ミク（LLM）が書く値ではない**——Orchestrator はツール名だけを選ぶので、
    #   `runner._handle_work_on_music` が `attached_audio` から詰める（`voice_enroll` と同じ作法）。
    audio_path: str = Field(default="", description="口ずさんだ音のローカルパス（attached_audio の要素）")


class WorkOnMusicValue(BaseModel):
    now_working: bool = Field(description="いま自分が曲づくりに取りかかったか（＝作業に入った状態）。待っている間に終わったら False")
    snapshot: str = Field(description="今ある曲と、自分の直近の作業の状態（この内容だけで一覧・進捗の質問に即答できる）")
    note: str = Field(description="マスターに伝えるべきこと（かかる時間・お願いしたい人手作業など）")
    report: str = Field(default="", description="待っている間に終わったときの、曲づくりの報告の全文（確かめた歌詞・楽譜・作り方など）")


WorkOnMusicResult = Success[WorkOnMusicValue] | Failure


def _snapshot(svc: Path) -> str:
    """songs.md＋_last_run.json＋lock状態から現状スナップショットを組み立てる（読み取りのみ・即時）。"""
    parts: list[str] = []
    songs = get_songs_dir() / "songs.md"
    if songs.exists():
        rows = [ln for ln in songs.read_text(encoding="utf-8").splitlines() if ln.startswith("| 2")]
        parts.append("【曲一覧（song_id｜曲名｜状態｜いまの版｜最後に触った日時）】\n" + "\n".join(rows[-12:])
                     if rows else "曲はまだ無い")
        # ★**最近触った曲の版ごとの「いつ・何を直した」**（2026-10-11・本人了承）。「朝の版を直して」を版の番号に直して頼めるように
        recent = sorted(rows, key=lambda ln: ln.rstrip(" |").rsplit("|", 1)[-1].strip())[-3:]
        for ln in reversed(recent):
            sid = ln.split("|")[1].strip()
            text = song_files.versions_text(get_songs_dir() / sid, last=5)
            if text:
                parts.append(f"【『{ln.split('|')[2].strip()}』の最近の版】\n{text}")
    last = get_songs_dir() / "_last_run.json"
    if last.exists():
        try:
            d = json.loads(last.read_text(encoding="utf-8"))
            parts.append(f"【自分の直近の作業】{d.get('status')}（{d.get('time')}） 取り組んだ内容={str(d.get('instruction'))[:60]}"
                         + (f" 結果={str(d.get('summary'))[:160]}" if d.get("summary") else "")
                         + (f" うまくいかなかった理由={str(d.get('reason'))[:160]}" if d.get("reason") else ""))
        except Exception:
            pass
    if (get_songs_dir() / ".runner.lock").exists():
        parts.append("【いま作業中】自分はいま別の曲に取りかかっている（終わるまで次の曲には移れない）")
    return "\n".join(parts) or "（まだ作った曲は無い）"


def _keep_audio(src: str) -> str:
    """★**添付音声を、消えない場所へ写してから渡す**（2026-09-23）。

    ブラウザ添付は `agent_main` が**一時ファイル**に書き、`process_agent_request` の
    `finally` で**リクエストが終わった瞬間に消す**。ところがこのツールは Fire&Forget で、
    song_service が実際にその音を開くのは**数秒〜数分あと**——**そのままでは必ず消えている**。
    （`voice_enroll` が同じ形で動くのは、あちらが**リクエスト中に同期で読む**から。）

    写し先は `data/songs/_input/`。曲フォルダはまだ無いので、ここに置いて
    `read_humming` が曲フォルダへ引き取る。
    """
    import shutil

    p = Path(src)
    if not p.exists():
        return ""
    dest_dir = get_songs_dir() / "_input"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{datetime.now():%Y%m%d_%H%M%S}{p.suffix or '.bin'}"
    try:
        shutil.copy2(p, dest)
    except Exception:
        return ""
    return str(dest)


#: 終わるのを待つ上限（秒）。確認だけの依頼は 76〜114 秒だった（2026-10-09 実測・コーディネーターの読み書きがほとんど）
WAIT_SECONDS = 180
#: 待っている間に終わったときに返す報告の上限（ツール結果の 4000 字の契約に収める）
REPORT_SHOWN = 3500


def _read_last() -> dict:
    try:
        return json.loads((get_songs_dir() / "_last_run.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_finished(d: dict, launched_at: datetime) -> bool:
    """この起動の結果が `_last_run.json` に書かれたか（それより前の実行の記録は見ない）。"""
    if d.get("status") not in ("done", "failed", "busy"):
        return False
    try:
        return datetime.strptime(str(d.get("time")), "%Y-%m-%d %H:%M:%S") >= launched_at.replace(microsecond=0)
    except ValueError:
        return False


def _wait_for_result(launched_at: datetime, wait_mark: Path) -> dict | None:
    """上限まで終わるのを待つ。終われば `_last_run` の中身、終わらなければ None（裏で続く）。

    待つのをやめるときは、印を消した**あとにもう一度**読む——song_runner は印を見てから `_last_run` を書くので、
    その間に印を消すと「知らせは省かれたのに、ここでも結果を返さない」が起きうる。読み直せばその回も拾える。
    """
    import time
    deadline = time.time() + WAIT_SECONDS
    try:
        while time.time() < deadline:
            d = _read_last()
            if _is_finished(d, launched_at):
                return d
            time.sleep(1.0)
    finally:
        wait_mark.unlink(missing_ok=True)
    time.sleep(1.0)
    d = _read_last()
    return d if _is_finished(d, launched_at) and d.get("caller_waiting") else None


def _finished_result(d: dict, svc: Path, log_path: Path) -> WorkOnMusicResult:
    """待っている間に終わった依頼の結果（報告の全文つき）。"""
    if d.get("status") == "busy":
        return Failure(retryable=True,
                       reason="自分はいま別の曲に取りかかっている最中。今の曲を作り終えてからにする。"
                              "いまは曲のデータを開けないので、覚えている内容で答えるなら、実際の曲とは違うかもしれないと断ってから話す。\n"
                              + _snapshot(svc))
    if d.get("status") == "failed":
        return Failure(reason=f"曲づくりが失敗して終わった（{str(d.get('reason'))[:300]}）。マスターにそのまま伝える。")
    report = str(d.get("report") or d.get("summary") or "")
    if len(report) > REPORT_SHOWN:
        report = report[:REPORT_SHOWN] + f"…（以下略・全{len(report)}字）"
    return Success(value=WorkOnMusicValue(
        now_working=False,
        snapshot=_snapshot(svc),
        report=report,
        note=("曲づくりは、待っている間に終わった。report がその報告の全文で、これが事実。"
              "マスターには report に書いてあることを元に話す（書いていないことを足さない）。"
              "曲を作ったり直したりしていれば、聴ける形にできたかも report に書いてある。"),
    ), side_effects=[f"曲づくりの作業をして終えた（作業ログ: {log_path.name}）"],
        summary="曲づくりが終わり、報告を受け取った")


def run_work_on_music(input_data: WorkOnMusicInput) -> WorkOnMusicResult:
    settings = load_settings()
    if not is_song_service_available(settings):
        return Failure(reason="いまは曲づくりができない状態になっている（設定でオフ、または準備が未完了）")
    svc = get_song_service_dir()
    py = svc / "venv" / "Scripts" / "python.exe"

    lock = get_songs_dir() / ".runner.lock"
    if lock.exists():                                      # 直列・キューなし（stale判定は runner 側が行う）
        return Failure(retryable=True,
                       # ★2026-10-09: 開けなかったのに、会話で決めた歌詞を「実際に歌った歌詞」として読み上げた
                       reason="自分はいま別の曲に取りかかっている最中。今の曲を作り終えてからにする。"
                              "いまは曲のデータを開けないので、覚えている内容で答えるなら、実際の曲とは違うかもしれないと断ってから話す。\n"
                              + _snapshot(svc))

    audio = _keep_audio(input_data.audio_path) if input_data.audio_path else ""

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    # 実行ログは logs/request_<日時>.log に残す（DEVNULL に捨てると採譜失敗などの原因が追えなくなる）
    logs = svc / "logs"
    logs.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    log_path = logs / f"request_{stamp}.log"
    # ★**指示はファイルで渡す**（2026-10-09）。指示には会話の全文が入るので、会話が長い夜は
    #   Windows のコマンドラインの上限（32,767字）を超えて起動の段階で落ちた（WinError 206・約3.8万字）。
    #   実行ログの隣に残るので、何を渡したかも後から読める。
    instr_path = logs / f"request_{stamp}.txt"
    instr_path.write_text(input_data.instruction, encoding="utf-8")
    log_fp = log_path.open("w", encoding="utf-8", errors="replace")
    wait_mark = get_songs_dir() / ".caller_waiting"
    launched_at = datetime.now()
    wait_mark.write_text(launched_at.isoformat(), encoding="utf-8")
    try:
        subprocess.Popen(                                  # 結果は _last_run.json / songs.md に残る
            [str(py), str(svc / "song_runner.py"), "request", "--instruction-file", str(instr_path)]
            + (["--audio", audio] if audio else []),
            cwd=str(svc), creationflags=creationflags,
            stdout=log_fp, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except OSError as e:                                   # 起動できなかった＝曲づくりは始まっていない
        log_fp.close()
        wait_mark.unlink(missing_ok=True)
        return Failure(reason=f"曲づくりを始められなかった（起動に失敗: {type(e).__name__}）。"
                              "作業には入っていないので、マスターにそのまま伝える。")
    # ★**数分で終わる依頼（確認・比較・作り方の質問など）は、終わるまで待って報告を全文で返す**（2026-10-09 本人判断）。
    #   それまでは全部 Fire&Forget で、結果は完了の知らせ（約200字の要約）でしか戻らず、確かめた歌詞や楽譜が
    #   ミクの手元に来なかった（記憶の分断）。呼び方は1つのまま、終わるのが早ければ同期で返す。
    #   上限を過ぎたら従来どおり裏で続け、終わったら完了の知らせで伝える。
    done = _wait_for_result(launched_at, wait_mark)
    if done is not None:
        return _finished_result(done, svc, log_path)
    return Success(value=WorkOnMusicValue(
        now_working=True,
        snapshot=_snapshot(svc),
        # ★2026-09-05: 歌声の書き出しを自分でやるようになった（`export_vocal`）。
        #   **「できたらすぐ歌う」とは書かない**——できた瞬間に歌い出しかねないため（本人指摘）。
        #   知らせる → 頼まれたら歌う、の順にしてある。
        # ★2026-10-03: 歌の入らない曲（インスト）も作れるようになったので、形に依らない言い方にした
        #   （インストには歌声の書き出しが無い。完成品は演奏を仕上げたもので、流して一緒に聴く）。
        # ★2026-10-10 本人: 3分待っても終わらなかったときは**ここでは話さない**。ツールを呼ぶ前の一言はもう伝わっていて、
        #   3分たつとマスターはブラウザを離れていることが多く、そこへ話しかけられても「何だろう」になる。
        #   それまでここで伝えさせていたこと（何で作り始めたか・後からの注文の扱い・言い切らない）は、
        #   呼ぶ前の一言（available_tools.md）と、あとで聞かれたときの答え方に回した。
        note=(f"いま自分は曲づくりに取りかかった。{WAIT_SECONDS // 60}分待っても終わらなかったので、裏で続けている"
              "（新しい曲は15〜20分くらい）。**この返事では何も話さない（response は null）**——呼ぶ前の一言はもう伝えてあり、"
              "マスターはいま別のことをしているかもしれないので、改めて話しかけない。できたら知らせる。"
              "あとでマスターに聞かれたときのために: "
              # ★2026-10-09: 直しの依頼で、決まる前に「新しく作り直す形になる」と言い切った（工程を決めたのは3分後）
              "今ある曲を直すときは、どこから直すか（手直しで済むか、途中から作り直すか）を曲づくりの中で自分が決めるので、言い切らない。"
              # ★2026-10-09: 確認を頼んだ直後に、結果が届く前から中身を言い切った（12:29・22:48）
              "結果が届く前に中身を推測で言わない。"
              # ★2026-10-09: 始めた後の注文が曲に入らないのに「仕上げておく」と引き受けた（本人と文案を確認済み）
              "ここから後で足された注文は、いま作っている曲には入らず、終わってから作り直すことになる。"
              "聴ける形にするところまで自分でやる。聴きたいと言われたら、その場で聴かせられる"
              "（歌の入る曲は歌って、歌の入らない曲は流して一緒に聴く。VOCALOID が入っていない環境だと、"
              "歌の入る曲は仮歌までで、そのときはマスターに書き出しをお願いすることになる）。"),
    ), side_effects=[f"曲づくりを開始（作業ログ: {log_path.name}）"],
        summary="自分の曲づくりを始めた（できるまで少しかかる）")
