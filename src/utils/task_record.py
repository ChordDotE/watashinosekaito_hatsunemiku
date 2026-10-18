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

"""裏作業（バックグラウンドタスク）の記録 — KV キー `background_tasks` の読み書き。

## なぜ要るのか

裏作業は**失敗しても会話が普通に進む**。だから記録が無いと、壊れていること自体に
誰も気づけない（2026-08-11 に顔照合で同じ型の事故を起こした）。さらに曲づくりでは
「完了したのにミクが『まだ作業中』と答える」誤りが実際に起きた —— 完了通知が
夜間の自発話しかけ抑制に巻き込まれて、事実ごと消えていたため。

そこで **記録と通知を分ける**:

  裏作業 ─┬─→ ここで記録を書く（必須・無条件）           ← 事実の保存
          └─→ /webhook/task_done を叩く（任意・失敗してよい）← 「今すぐ気づいて」の合図

記録を書くのは**裏作業自身の責務**とする（受け口に書かせると、通知経路の不具合が
そのまま記録の欠落になる）。読むのは `agent_main._fetch_system_state` で、毎ターン
system_state へ注入される。ツールを呼ばなくてもミクが現状を把握できる。

## データの持ち方

KV キー1つ（`background_tasks`）に正式 JSON。`mood_state` と同じ持ち方に揃えている:

    {"tasks": [ {task_id, source, label, status, started_at, finished_at,
                 title, summary, reason, notify, detail}, ... ]}

共通項目は固定し、**タスク固有の情報は `detail` に逃がす**（読む側は共通項目しか見ない）。
本格的な DB ならタスクごとにレコードを分けるのが定石だが、ここは
「データをまとめたい」という運用上の要望を優先して1キーに置く（2026-08-11 ユーザー判断）。
そのぶん read-modify-write になるため、**KV の楽観ロック（ETag / If-Match）で
lost update を防ぐ**。

## 別 venv からの書き込み

`song_service` は別プロセス・別 venv でこのモジュールを import できないため、
同等の処理を自前で持つ。**排他はクライアント側では実現できない**
（プロセスをまたぐため）ので、調停は必ず KV 側の If-Match に委ねること。
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import requests

KV_KEY = "background_tasks"
ETAG_ABSENT = '"0"'

MAX_TASKS = 10           # 保持する件数（溢れたら done の古い順に捨てる）
STALE_RUNNING_HOURS = 6  # これを超えて running のままなら「中断した可能性」を添える

_RETRY = 3               # 409（割り込み更新）が返ったときの読み直し回数

# 時間では消さない。**伝えたら消す**（ミクが伝えた記録は会話ログへ移るので、箱に残す必要がない）。
# 当初は「完了から24時間で表示を落とす」にしていたが、時間は「伝えたかどうか」と無関係で、
# 「まだ伝えていないのに消える」「伝えたのに残る」の両方が起きるため取りやめた（2026-08-11）。
# 取り下げの経路: LLM 出力の acknowledged_task_id → runner → remove()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _get(base_url: str) -> tuple[list[dict[str, Any]], str]:
    """現在の記録と ETag を返す。未作成・壊れていれば空リスト。"""
    try:
        resp = requests.get(f"{base_url}/kv", params={"key": KV_KEY}, timeout=10)
        if resp.status_code == 404:
            return [], ETAG_ABSENT
        resp.raise_for_status()
        etag = resp.headers.get("ETag", ETAG_ABSENT)
        tasks = json.loads(resp.text).get("tasks", [])
        return (tasks if isinstance(tasks, list) else []), etag
    except Exception as e:
        print(f"[WARN] 裏作業の記録を読めませんでした: {e}")
        return [], ETAG_ABSENT


def _trim(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """件数上限に収める。**捨てるのは `done` の古い順から**。

    単純に古い順で捨てると、まだ誰も気づいていない `failed` が、もう伝え終えた `done` に
    押し出されて消えうる。「黙って失敗する裏作業をゼロにする」のが目的なので、
    失敗と実行中は残す（時系列の並びは保つ）。
    """
    if len(tasks) <= MAX_TASKS:
        return tasks
    done_idx = [i for i, t in enumerate(tasks) if t.get("status") == "done"]
    drop = set(done_idx[: len(tasks) - MAX_TASKS])
    trimmed = [t for i, t in enumerate(tasks) if i not in drop]
    return trimmed[-MAX_TASKS:]      # failed だらけの異常時の保険（古い順に落とす）


def _mutate(base_url: str, task_id: str, change) -> bool:
    """読んで直して書き戻す（409 なら読み直して再試行）。

    排他はクライアント側では実現できない（本体と song_service の別プロセスが書く）ため、
    KV の If-Match に委ねている → docs/components/kv.md
    """
    for _ in range(_RETRY):
        tasks, etag = _get(base_url)
        try:
            resp = requests.post(
                f"{base_url}/kv",
                json={"key": KV_KEY,
                      "value": json.dumps({"tasks": change(tasks)}, ensure_ascii=False)},
                headers={"Content-Type": "application/json", "If-Match": etag},
                timeout=10,
            )
            if resp.status_code == 409:
                continue                      # 割り込まれた → 読み直してやり直す
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"[WARN] 裏作業の記録を書けませんでした (task_id={task_id}): {e}")
            return False
    print(f"[WARN] 裏作業の記録が競合し続けました (task_id={task_id})")
    return False


def update(base_url: str, task_id: str, *, source: str, label: str, status: str,
           started_at: str = "", finished_at: str | None = None,
           title: str = "", summary: str = "", reason: str = "",
           notify: bool = False, detail: dict[str, Any] | None = None) -> bool:
    """`task_id` の記録を書く（同じ task_id があれば差し替え・無ければ追加）。"""
    record = {
        "task_id": task_id, "source": source, "label": label, "status": status,
        "started_at": started_at or _now(),
        # 終了した作業には必ず時刻を入れる（呼び出し側に持ち回らせない）
        "finished_at": finished_at if finished_at is not None
                       else (None if status == "running" else _now()),
        "title": title, "summary": summary, "reason": reason, "notify": notify,
        "detail": detail or {},
    }

    def _change(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # 開始時に書いた started_at を完了時に引き継ぐ（呼び出し側が持ち回らなくて済む）
        previous = next((t for t in tasks if t.get("task_id") == task_id), None)
        if previous and not started_at:
            record["started_at"] = previous.get("started_at") or record["started_at"]
        return _trim([t for t in tasks if t.get("task_id") != task_id] + [record])

    return _mutate(base_url, task_id, _change)


def remove(base_url: str, task_id: str) -> bool:
    """伝え終えた裏作業を箱から出す（無ければ何もしない）。

    ミクが完了・失敗をマスターに伝えると、その事実は会話ログに残り、やがて長期記憶へ移る。
    箱は「まだ伝えていないこと」の置き場なので、伝え終えたものは取り下げる。
    作業の詳細（曲の一覧・作業ログなど）は各サービス側の SoT に残るので、ここから消しても失われない。
    """
    return _mutate(base_url, task_id,
                   lambda tasks: [t for t in tasks if t.get("task_id") != task_id])


def _hours_since(stamp: str | None) -> float | None:
    if not stamp:
        return None
    try:
        return (datetime.now() - datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600
    except (ValueError, TypeError):
        return None


def visible_tasks(base_url: str) -> list[dict[str, Any]]:
    """いまミクに見せるべき記録だけを返す。

    - `failed` は常に出す（黙って失敗している裏作業をゼロにするのが主目的）
    - `running` は常に出す
    - `done` は `notify` が真のものだけ。**伝えるまで出し続ける**
      （`notify` が偽＝記憶の整理のような「ミクの中で完結する作業」の成功は報告しない）

    伝え終えたものは `remove()` で箱から出るので、ここで時間による足切りはしない。
    """
    out = []
    for t in _get(base_url)[0]:
        status = t.get("status")
        if status in ("failed", "running"):
            out.append(t)
        elif status == "done" and t.get("notify"):
            out.append(t)
    return out


def format_for_system_state(base_url: str) -> list[str]:
    """裏作業の行を全部（実行中＋伝えていない完了・失敗）。互換のため残す。

    ★2026-09-27 からミクへは**2つの欄に分けて**渡す（本人判断）:
      - `<system_state>` … いまの状態（`format_running`）。マスターに報告する項目ではない
      - `<pending_reports>` … マスターに伝えて片付ける結果（`format_pending`）
    それまでは顔・声の登録状態と、伝えるべき完了報告が同じ「システム状態」に並び、
    **何を伝えるべきかが区別できなかった**（8月の完了報告が1か月以上伝えられずに残っていた）。
    """
    return format_running(base_url) + format_pending(base_url)


def format_running(base_url: str) -> list[str]:
    """`<system_state>` に載せる、実行中の裏作業の行。"""
    return _format(base_url, ("running",))


def format_pending(base_url: str) -> list[str]:
    """`<pending_reports>` に載せる、まだマスターに伝えていない完了・失敗の行。

    各行の末尾に **`[id: ...]`** を添える。これが無いと、伝えたことを LLM が
    `acknowledged_task_id` で返せない（実測では代わりに時刻や、存在しない id を返した）。
    完了時刻は日付つきで書く（時刻だけだと、何日も前の報告が今日のことに見える）。
    """
    return _format(base_url, ("done", "failed"))


def _format(base_url: str, statuses: tuple[str, ...]) -> list[str]:
    lines = []
    for t in visible_tasks(base_url):
        if t.get("status") not in statuses:
            continue
        label = t.get("label") or t.get("source") or "裏作業"
        tid = t.get("task_id", "")
        status = t.get("status")
        if status == "running":
            at = str(t.get("started_at") or "")[11:16]
            note = ""
            elapsed = _hours_since(t.get("started_at"))
            if elapsed is not None and elapsed > STALE_RUNNING_HOURS:
                note = "（開始から時間が経ちすぎている。途中で中断した可能性がある）"
            lines.append(f"- 自分の裏作業（{label}）: 実行中（{at} 開始）{note} [id: {tid}]")
        elif status == "done":
            at = str(t.get("finished_at") or "")[:16]
            title = t.get("title") or ""
            lines.append(f"- 自分の裏作業（{label}）: 完了（{at}）{title} [id: {tid}]")
            if t.get("summary"):
                lines.append(f"  → {t['summary']}")
        elif status == "failed":
            at = str(t.get("finished_at") or "")[:16]
            lines.append(f"- 自分の裏作業（{label}）: **失敗**（{at}）{t.get('reason', '')}"
                         f" [id: {tid}]")
    return lines
