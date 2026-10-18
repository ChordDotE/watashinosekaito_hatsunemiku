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

"""song_service の CLI 入口（本番はミク側ツール music_request がこのプロセスを起動する）。

使い方（song_service ディレクトリを cwd に実行。output/ 相対参照の前提）:
    venv/Scripts/python.exe song_runner.py request "<自然文の指示>"
    venv/Scripts/python.exe song_runner.py request --instruction-file <指示を書いたファイル>   # 本番の入口
    venv/Scripts/python.exe song_runner.py selfcheck

責務（§5.3 Phase0 / §5.4b の命綱④）:
- request: 二重起動ロック（.runner.lock・pid生存確認で stale 自動掃除）→ _last_run.json(running)
  → 時間 watchdog（settings song_service.max_runtime_minutes・既定60分。未知エラー/スタックの絶対の天井。
     上限の時点で完成品〔master_<形>_v{N}.wav〕までできていれば failed にせず done で閉じる）
  → coordinator.run(指示) → _last_run.json(done/failed＋要約)
- selfcheck: 設置先の受け入れ確認（存在チェックのみ・数秒・ネット不要。setup スクリプトの最終工程）
"""
from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

SVC = Path(__file__).resolve().parent
os.chdir(SVC)                                            # output/ 相対参照と engine サブプロセスの前提
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# tools_bin の ffmpeg を PATH 先頭に注入（whisperx.load_audio が "ffmpeg" を subprocess 起動するため。
# setup_song_service.ps1 が隔離配置し、システムへのインストールは要求しない。engine の
# サブプロセス群もこのプロセスから env を継承するので、ここ1か所で全工程に効く）
_FFMPEG_BIN = SVC / "tools_bin" / "ffmpeg" / "bin"
if _FFMPEG_BIN.exists():
    os.environ["PATH"] = str(_FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")

import song_files                                        # noqa: E402  名前の決まりの1か所（標準ライブラリだけ・SVC 直下）

# 曲の置き場は `miku/data/songs`（2026-09-01・ソースツリーに成果物を置かない）
SONGS = SVC.parents[1] / "data" / "songs"
LOCK = SONGS / ".runner.lock"
LAST = SONGS / "_last_run.json"
#: 呼び手（ミクの work_on_music）が結果を待っている印。ある間に終われば、完了の知らせを出さない（2026-10-09）
WAIT_MARK = SONGS / ".caller_waiting"


def _all_settings() -> dict:
    """設定の全体。読む入口は `engine/song_settings.load()` の1か所（`MIKU_SETTINGS`・検証用の重ね書き
    `MIKU_SETTINGS_OVERLAY` もここで効く・2026-10-01）。"""
    if str(SVC / "engine") not in sys.path:
        sys.path.insert(0, str(SVC / "engine"))
    import song_settings                                   # noqa: E402
    return song_settings.load()


def _settings() -> dict:
    try:
        # ★utf-8-sig（2026-09-25・`song_settings` が読む）。下の except が**黙って空にする**ので、BOM が付くと
        #   作曲機能の設定（実行時間の上限など）が**何も言わずに既定値へ戻っていた**
        return _all_settings().get("song_service", {})
    except Exception:
        return {}


def _write_last(status: str, **kw) -> None:
    LAST.parent.mkdir(parents=True, exist_ok=True)
    LAST.write_text(json.dumps({"status": status, "time": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                                "pid": os.getpid(), **kw}, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 裏作業の記録（KV `background_tasks`）────────────────────────
# _last_run.json が song_service 固有の詳細なのに対し、こちらは**全裏作業の共通形式**。
# ミクが毎ターン system_state で現状を把握するための記録で、
# 完了通知（_notify_done）とは独立に**必ず**書く —— 通知が抑制されても事実は残す
# （2026-08-11: 夜間の自発話しかけ抑制に巻き込まれて完了が消え、「まだ作業中」と誤答した）。
#
# 本体の miku/src/utils/task_record.py と同じ形式だが、**別 venv で import できない**ため
# 標準ライブラリだけで同等の処理を持つ。読んで直して書き戻すので、割り込み更新は
# KV の If-Match（楽観ロック）で弾く —— 排他はプロセスをまたげないので、調停は KV に委ねる。

TASK_ID = f"song_service_{datetime.now():%Y%m%d_%H%M%S}"
TASK_STARTED_AT = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
_KV_KEY = "background_tasks"
_ETAG_ABSENT = '"0"'
_MAX_TASKS = 10


def _kv_base_url() -> str:
    try:
        kv = _all_settings().get("services", {}).get("kv", {})
        return f"http://127.0.0.1:{int(kv.get('port', 8010))}"
    except Exception:
        return "http://127.0.0.1:8010"


def _write_task(status: str, *, title: str = "", summary: str = "",
                reason: str = "", detail: dict | None = None, notify: bool = True) -> None:
    """裏作業の記録を KV へ書く。失敗しても曲づくりは止めない（記録が無いだけ）。"""
    import urllib.error
    import urllib.request

    base = _kv_base_url()
    record = {
        "task_id": TASK_ID, "source": "song_service", "label": "曲づくり",
        "status": status, "started_at": TASK_STARTED_AT,
        "finished_at": None if status == "running" else f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "title": title, "summary": summary, "reason": reason,
        "notify": notify,                                # 曲はマスターに頼まれて作るもの＝完了を伝える（呼び手が結果を受け取った回は伝え済み）
        "detail": detail or {},
    }
    for _ in range(3):                                   # 409（割り込み更新）なら読み直してやり直す
        tasks, etag = [], _ETAG_ABSENT
        try:
            with urllib.request.urlopen(f"{base}/kv?key={_KV_KEY}", timeout=10) as r:
                etag = r.headers.get("ETag", _ETAG_ABSENT)
                loaded = json.loads(r.read().decode("utf-8")).get("tasks", [])
                tasks = loaded if isinstance(loaded, list) else []
        except urllib.error.HTTPError as e:
            if e.code != 404:                            # 404 = まだ1件も無い（正常）
                print(f"[task] 記録を読めなかった: {e}")
                return
        except Exception as e:
            print(f"[task] 記録を読めなかった（本体停止中など）: {type(e).__name__}: {e}")
            return
        merged = [t for t in tasks if t.get("task_id") != TASK_ID] + [record]
        body = json.dumps({"key": _KV_KEY,
                           "value": json.dumps({"tasks": merged[-_MAX_TASKS:]}, ensure_ascii=False)},
                          ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/kv", data=body, method="POST",
            headers={"Content-Type": "application/json", "If-Match": etag})
        try:
            with urllib.request.urlopen(req, timeout=10):
                return
        except urllib.error.HTTPError as e:
            if e.code == 409:
                continue                                 # 割り込まれた → 読み直し
            print(f"[task] 記録を書けなかった: {e}")
            return
        except Exception as e:
            print(f"[task] 記録を書けなかった（本体停止中など）: {type(e).__name__}: {e}")
            return
    print("[task] 記録が競合し続けた（今回の記録は残らない）")


def _notify_done(summary: str) -> None:
    """Gateway に完了を通知し、ミク起点の自発話しかけ（mode=3）を起こす。

    本体が起動していない/失敗しても**無視**する（成果は _last_run.json・songs.md に残るので、
    次の会話でミクが music_request のスナップショットから気づける＝二段構え・§5.4c-F1）。
    """
    import urllib.request
    try:
        port = 5001
        try:
            port = int(_all_settings()["services"]["gateway"]["port"])
        except Exception:
            pass
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/webhook/task_done",
            data=json.dumps({"source": "song_service", "summary": summary[:400]}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            print(f"[notify] gateway 応答: {r.status}")
    except Exception as e:
        print(f"[notify] 通知できなかった（本体停止中など。次の会話で気づく）: {type(e).__name__}: {e}")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True                                      # 触れないが存在はする
    except OSError:
        return False


def _acquire_lock() -> bool:
    """ロック獲得。実行中の別プロセスがあれば False（stale＝プロセス消滅は自動掃除）。"""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            info = json.loads(LOCK.read_text(encoding="utf-8"))
            if _pid_alive(int(info.get("pid", -1))):
                return False
        except Exception:
            pass                                         # 壊れたロックは stale 扱い
        LOCK.unlink(missing_ok=True)
    LOCK.write_text(json.dumps({"pid": os.getpid(), "start": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}),
                    encoding="utf-8")
    return True


def _resumable_note(started: float) -> str:
    """落ちた曲が**続きから作れる**なら、ミクへの通知に添える一文（作れなければ空）。

    ★**これが無いと、ミクは「もう一度作る?」と聞いて全部作り直しに行く**（2026-09-25）。
      済んだ工程は `_wip` に残っていて、`draft_melody` が曲フォルダに目印（`_progress.json`）を
      書いている。ミクがそれを知る経路は、この通知だけ。
    ★**この回に書かれた目印だけ**を見る（開始時刻より新しいもの）。`last_song_id` は
      前の回の曲を指したまま残ることがあり、それを拾うと**別の曲を続きから作る**と伝えてしまう。
    """
    try:
        import coordinator
        sid = coordinator.last_song_id
        pg_path = SONGS / sid / song_files.PROGRESS if sid else None
        if not pg_path or not pg_path.exists() or pg_path.stat().st_mtime < started:
            return ""
        pg = json.loads(pg_path.read_text(encoding="utf-8"))
        if not pg.get("failed"):
            return ""
        nxt = pg.get("next") or ""
        return (f"。ただし**『{pg.get('title', '')}』の済んだ工程は保存されていて、続きから作れる**"
                + (f"（{nxt}から）" if nxt else "")
                + "——マスターが続きを望んだら、新しく作り直さずに『続きから作って』と頼むこと"
                  "（済んだぶんの費用を二重に払わずに済む）")
    except Exception:
        return ""


def _stopped_note(started: float) -> str:
    """この回の曲が**形式の所見が直らずに工程で止まった**なら、完了通知の頭に添える一文（止まっていなければ空・2026-09-30）。

    ★**止まった曲は「完成」ではない**。コーディネーターの最終報告だけだと、申し送りの要約（`handoff`）で
      止まったことが落ちうる。`draft_melody` が曲フォルダに書く目印（`_progress.json` の `stopped`）から、
      どの工程で止まったか・続きから作れることを**機械が確実に**添える。見るのは `_resumable_note` と同じく、この回に書かれた目印だけ。
    """
    try:
        import coordinator
        sid = coordinator.last_song_id
        pg_path = SONGS / sid / song_files.PROGRESS if sid else None
        if not pg_path or not pg_path.exists() or pg_path.stat().st_mtime < started:
            return ""
        pg = json.loads(pg_path.read_text(encoding="utf-8"))
        stp = pg.get("stopped")
        if not isinstance(stp, dict):
            return ""
        step = stp.get("step") or pg.get("next") or ""
        why = stp.get("why") or "書いたものの形式が直らなかった"   # 最終の点検役の NG が残ったときは、その旨（2026-10-04 組P2b）
        # ★**何が直らなかったかも添える**（2026-10-04 組F3・4‑3）。以前は「◯の工程で止まった（理由）」だけで、所見をマスターへ
        #   伝えるかは作曲担当の最終報告次第だった。先頭2件・1件120字（`tools/editing` と同じ切り方。切ったら切ったと書く）
        fs = [str(f) for f in (stp.get("findings") or [])]
        left = ("直らなかった所見: " + " / ".join(_cut(f, STOPPED_FINDING_SHOWN) for f in fs[:2])
                + (f" ほか{len(fs) - 2}件" if len(fs) > 2 else "") + "。") if fs else ""
        if not step:                                       # 最終の点検役が原因の工程を選べなかった（続きは NG を読んで工程を選ぶ）
            return (f"【曲はまだできていない】『{pg.get('title', '')}』は最後の点検で止まった（{why}）。{left}"
                    "ここまでは保存してあり、マスターが頼めば残った NG の原因の工程から続きを作れる。")
        return (f"【曲はまだできていない】『{pg.get('title', '')}』は{step}の工程で止まった"
                f"（{why}）。{left}ここまでは保存してあり、マスターが頼めば{step}から続きを作れる。")
    except Exception:
        return ""


#: 完了通知の頭に添える、直らなかった所見の1件の上限（字）と、要約・タスクの要約の上限（字）。どれも切ったら切ったと書く（2026-10-04 組F3・4‑3）
STOPPED_FINDING_SHOWN = 120
SUMMARY_SHOWN = 500
TASK_SUMMARY_SHOWN = 200


def _cut(text: str, limit: int) -> str:
    """`song_fix.cut_note`（切ったら「…（以下略・全N字）」）。engine を読めない場面でも同じ形で切る。"""
    try:
        from song_fix import cut_note
        return cut_note(text, limit)
    except Exception:                                    # noqa: BLE001
        t = text or ""
        return t if len(t) <= limit else t[:limit] + f"…（以下略・全{len(t)}字）"


#: **実行時間の上限の既定（分）**。設定（`song_service.max_runtime_minutes`）が無いときに使う。
#: ★2026-09-30 に 30 → 60（出-2）。『孤独のまま』は作曲に約26分（差し戻し2回）・仕上げに約3分かかり、歌声の書き出しは
#:   済んでいたのに30分の上限で failed になった。曲の長さを形式の検査にはしない（設計の判断）ので、上限のほうを延ばした
DEFAULT_MAX_MINUTES = 60


def _finished_master(started: float) -> song_files.Master | None:
    """この回に**完成品までできた曲**があれば、その完成品（`song_files.Master`）。無ければ None。

    見るのはこの回の曲（`coordinator.last_song_id`）の完成品 `master_<形>_v{N}.wav` で、開始時刻より新しいものだけ
    （完成の判定は `song_files` の1か所・`session._song_status` の「完成」と同じ基準。2026-10-03）。
    歌もの・アカペラは歌声の書き出しを仕上げたとき、インストは `render_final` が演奏を仕上げたときにできる。
    """
    try:
        coordinator = sys.modules.get("coordinator")
        sid = getattr(coordinator, "last_song_id", "") or ""
        d = SONGS / sid if sid else None
        if d is None or not d.is_dir():
            return None
        fresh = [m for m in song_files.masters(d) if m.path.stat().st_mtime >= started]
        return max(fresh, key=lambda m: m.path.stat().st_mtime) if fresh else None
    except Exception:
        return None


#: 時間切れの要約の「どこまで済んで完成したか」（形ごと。インストでは歌声の話をしない）
_DONE_HOW = {"song": "歌声の書き出しまで済んで完成している",
             "acappella": "声だけの曲（アカペラ）として歌声の書き出しまで済んで完成している",
             "instrumental": "演奏だけの曲（インスト）として演奏の仕上げまで済んで完成している"}


def _on_timeout(instruction: str, minutes: float, started: float) -> int:
    """実行時間の上限に達したときの後始末。戻り値はプロセスの終了コード。

    ★**完成品までできていれば failed にしない**（2026-09-30・出-2）。『孤独のまま』は 13:34:53 に歌声を
      書き出し終えていたのに、そのあとの最終報告の途中で30分の上限に達して failed 扱いになった。曲は完成しているので
      done で閉じ、完成を知らせる（最終報告は書けていないので、上限で打ち切ったことを添える）。
      インストは歌声の書き出しが無いので、完成品（`render_final` が書く）で同じに判定する（2026-10-03）。
      作業台（固定名）は片付けずに残る——次にこの曲を開くと、同じ版の続きとして確定される（`session.begin_session`）
    """
    done = _finished_master(started)
    sid = getattr(sys.modules.get("coordinator"), "last_song_id", "") or ""
    if done and sid:
        title = sid.split("_", 2)[-1] if sid.count("_") >= 2 else sid
        summary = (f"『{title}』は{_DONE_HOW[done.form]}（{done.path.name}）。ただし実行時間の上限（{minutes:.0f}分）に"
                   f"達したので、最後の確認と報告の途中で作業を打ち切った")
        _write_last("done", instruction=instruction, summary=summary, timeout=True)
        _write_task("done", title=title, summary=_cut(summary, TASK_SUMMARY_SHOWN), detail={"song_id": sid, "timeout": True})
        _notify_done(summary)
        return 0
    reason = f"watchdog: {minutes:.0f}分の上限を超過（スタックの疑い）。成果物は output/ に残っている"
    _write_last("failed", instruction=instruction, reason=reason)
    _write_task("failed", reason=reason)
    return 2


def cmd_request(instruction: str, *, audio_path: str = "") -> int:
    import time
    started = time.time()
    if not _acquire_lock():
        print("busy: 別の作業が実行中（.runner.lock）。終わってから依頼して。")
        _write_last("busy", instruction=instruction)
        return 3
    minutes = float(_settings().get("max_runtime_minutes", DEFAULT_MAX_MINUTES))

    def _watchdog() -> None:                             # §5.4b-④ 絶対の天井（未知エラー・スタックにも効く）
        try:
            code = _on_timeout(instruction, minutes, started)
        except Exception:                                # 後始末で落ちても、必ずプロセスは終える
            code = 2
        LOCK.unlink(missing_ok=True)
        os._exit(code)

    timer = threading.Timer(minutes * 60, _watchdog)
    timer.daemon = True
    timer.start()
    _write_last("running", instruction=instruction)
    _write_task("running")
    try:
        import _bootstrap  # noqa: F401  sys.path 初期化（song_service/ と engine/）
        import coordinator
        out = coordinator.run(instruction, audio_path=audio_path)   # 上限到達は run() 内で優雅な打ち切り済み
        # ★**形式の所見が直らずに工程で止まった曲**なら、要約と通知の頭にそう書く（2026-09-30・`_stopped_note`）
        stopped = _stopped_note(started)
        summary = _cut(" ".join((stopped + " " + str(out) if stopped else str(out)).split()), SUMMARY_SHOWN)
        # ★**呼び手（ミクの work_on_music）が待っていれば、結果は呼び手が全文で受け取る**（2026-10-09）。
        #   確認だけの依頼は数分で終わる。そのときは完了の知らせ（自発話しかけ）を出さず、伝え済みとして記録する
        #   （二重に伝えない）。判定は `_last_run` を書く直前——呼び手は待つのをやめた直後に `_last_run` を読み直すので、
        #   ここで「待っている」と見て知らせを省いた回は、必ず呼び手の側で結果が返る。
        caller_waiting = WAIT_MARK.exists()
        _write_last("done", instruction=instruction, summary=summary, report=str(out),
                    caller_waiting=caller_waiting, **({"stopped": True} if stopped else {}))
        song_id = getattr(coordinator, "last_song_id", "") or ""
        # song_id は「<日付>_<時刻>_<曲名>」。曲名だけを title に出す（system_state は1行なので短く）
        title = song_id.split("_", 2)[-1] if song_id.count("_") >= 2 else song_id
        _write_task("done", title=title, summary=_cut(summary, TASK_SUMMARY_SHOWN), detail={"song_id": song_id},
                    notify=not caller_waiting)
        if caller_waiting:
            return 0
        from handoff import compose_handoff              # 作業ログを読んで「ミクへの申し送り」に要約（失敗時は報告文）
        handoff = compose_handoff(coordinator.last_song_id, out, instruction)
        _notify_done(f"{stopped} {handoff}" if stopped else handoff)
        # ※通知先(Gateway)が「マスターは不在の可能性が高い→必要ならメールで知らせる」まで文面で促す
        return 0
    except Exception as e:                               # 未知例外: 理由を残して失敗（会話側は list/スナップショットで気づく）
        reason = f"{type(e).__name__}: {e}"
        caller_waiting = WAIT_MARK.exists()                # 呼び手が待っていれば、失敗も呼び手が受け取る（上の done と同じ）
        _write_last("failed", instruction=instruction, reason=reason, caller_waiting=caller_waiting)
        if caller_waiting:                                 # 呼び手がその場で伝えるので、伝え済みとして閉じる（失敗は常に見せる欄に載せない）
            _write_task("done", reason=reason, notify=False)
            return 1
        _write_task("failed", reason=reason)
        _notify_done(f"依頼「{instruction[:60]}」の作業が失敗して終わった（{reason[:150]}）"
                     + _resumable_note(started))
        return 1
    finally:
        timer.cancel()
        LOCK.unlink(missing_ok=True)


def cmd_selfcheck() -> int:
    """設置先の受け入れ確認（存在チェックのみ）。"""
    import shutil
    # ok が None の項目は「入っていないが、それでいい」＝合否に数えない（下のカバー用一式）
    checks: list[tuple[str, bool | None, str]] = []

    def add(name: str, ok: bool | None, note: str = "") -> None:
        checks.append((name, ok, note))

    def _venv_ok(name: str) -> bool:
        # python.exe だけでは不十分: uv venv は pip install の前に exe を作るため、
        # インストール失敗した空 venv を見逃す。代表パッケージ torch の実体まで見る
        return ((SVC / name / "Scripts" / "python.exe").exists()
                and (SVC / name / "Lib" / "site-packages" / "torch").exists())

    # 本体 venv の目印は **librosa**（torch ではない）——オリジナル専用の構成に torch は入らないため。
    # librosa はどちらの構成にも入り、`setup_song_service.ps1` の $mainMarker と揃えてある
    add("venv(py3.13)", (SVC / "venv" / "Scripts" / "python.exe").exists()
        and (SVC / "venv" / "Lib" / "site-packages" / "librosa").exists(), "曲づくり本体")

    # ★**あとから足した依存が行き渡っているか**（2026-09-23 追加）。
    #   ここが無いと、**依存が1つも入っていないのに PASS と出る**——クリーン VM で実測した
    #   （`setup_song_service.ps1 -OriginalOnly` を再実行しても、既存 venv には入らなかった。
    #   `setup_song_service.ps1` 側も直したが、**受け入れ確認でも見る**——導入の経路は
    #   スクリプトだけではない〔手で入れた人・古い配布物から更新した人〕）。
    #   ★**代表1つでなく、その機能に要るものを並べる**。1つ入っていれば OK にすると、
    #   まさに今回の `librosa` と同じ穴になる。
    site = SVC / "venv" / "Lib" / "site-packages"
    hum_missing = [m for m in ("faster_whisper", "ctranslate2", "onnxruntime", "av", "tokenizers")
                   if not (site / m).exists()]
    add("文字起こし(鼻歌・口笛・肉声)", not hum_missing,
        "CPU だけで動く" if not hum_missing
        else f"★{('/'.join(hum_missing))} が無い → setup_song_service.ps1 を再実行")

    # ★カバー用は**ひとまとめに扱う**。`setup_song_service.ps1 -OriginalOnly`（GPU の無い人向け）は
    #   これらを入れないので、1つずつ必須にすると**必ず FAIL**になる。
    #   すると setup が exit 1 を返し、`supervisor.install_song_service` が導入失敗と見なして
    #   `song_service.enabled` を true にせず、**オリジナル曲すら使えないまま終わる**（2026-09-02 検出）。
    #   そこで「**6つとも無ければオリジナル専用の構成**」と判断して減点しない。
    #   一部だけ欠けている場合は**フル導入が途中で失敗した**とみなして、従来どおり ✗ にする。
    cover_ckpts = SVC / ".mt3_checkpoints"
    cover = [
        ("sofa_venv(py3.10)", _venv_ok("sofa_venv"), "SOFA用"),
        ("venv_mt3(py3.10)", _venv_ok("venv_mt3"), "YourMT3用"),
        ("SOFAモデル", (SVC / "sofa_model" / "step.100000.ckpt").exists()
         and (SVC / "sofa_model" / "japanese-extension-sofa.txt").exists(), "1.2GB ckpt＋辞書"),
        ("SOFAリポジトリ", (SVC / "sofa_repo" / "infer.py").exists(), "qiuqiao/SOFA clone"),
        ("YourMT3チェックポイント",
         cover_ckpts.exists() and any(cover_ckpts.rglob("*.ckpt")), "536MB"),
    ]
    original_only = not any(ok for _n, ok, _t in cover)
    if original_only:
        checks.append(("カバー用一式", None,
                       "未導入（音源しか無い曲のカバーには NVIDIA GPU が要ります。"
                       "オリジナル曲と、楽譜〔MIDI〕が見つかる曲のカバーは GPU 無しで作れます）"))
    else:
        checks.extend(cover)
    sys.path.insert(0, str(SVC / "engine"))
    try:
        import sf2_render
        add("FluidSynth+SoundFont", sf2_render.available(),
            "tools_bin/ 既定 or env MIKU_FLUIDSYNTH/MIKU_SOUNDFONT")
    except Exception as e:
        add("FluidSynth+SoundFont", False, f"判定失敗: {e}")
    add("ffmpeg", shutil.which("ffmpeg") is not None, "demucs/音声処理に必要")
    s = _settings()
    import song_settings                                   # noqa: E402（engine/ は上で sys.path に入れてある）
    sp = song_settings.base_path()
    add("settings.json", sp.exists(), str(sp))
    if song_settings.overlay_path() is not None:           # 検証用の重ね書きで動いているなら、そう出す
        _op = song_settings.overlay_path()
        add("設定の重ね書き", _op.exists(), str(_op))
    add("song_serviceキー", bool(s), "enabled/max_runtime_minutes")
    try:
        SONGS.mkdir(parents=True, exist_ok=True)
        probe = SONGS / ".selfcheck"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        add("output/ 書込", True)
    except Exception as e:
        add("output/ 書込", False, str(e))

    ok_all = True
    for name, ok, note in checks:
        if ok is None:                                     # 「入っていないが、それでいい」＝減点しない
            print(f"  － {name:24} {note}")
            continue
        ok_all &= ok
        print(f"  {'✓' if ok else '✗'} {name:24} {note}")
    print("selfcheck:", "PASS" if ok_all else "FAIL（✗の項目は setup_song_service.ps1 の再実行または手動配置を）")
    return 0 if ok_all else 1


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "request":
        # `--audio <path>`（任意）＝マスターが口ずさんだ音。**付けなければ従来どおり**。
        # 引数の解析はここだけ（`argparse` を入れると selfcheck 側の呼び出しまで作法が変わる）。
        # `--instruction-file <path>`＝指示をファイルで受ける（本番の work_on_music はこちら。
        # 会話の全文が入る指示はコマンドラインの上限 32,767字を超えうる・2026-10-09）。手で打つときは従来どおり文字列でよい。
        if sys.argv[2] == "--instruction-file" and len(sys.argv) >= 4:
            instruction = Path(sys.argv[3]).read_text(encoding="utf-8")
            argv = sys.argv[4:]
        else:
            instruction, argv = sys.argv[2], sys.argv[3:]
        audio = ""
        if "--audio" in argv:
            i = argv.index("--audio")
            audio = argv[i + 1] if i + 1 < len(argv) else ""
        return cmd_request(instruction, audio_path=audio)
    if len(sys.argv) >= 2 and sys.argv[1] == "selfcheck":
        return cmd_selfcheck()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
