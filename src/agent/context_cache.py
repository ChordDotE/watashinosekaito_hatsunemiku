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

"""Gemini の明示キャッシュ（`cachedContents`）でプロンプトの固定部を使い回す。

## なぜ要るか（2026-08-29 実測）

暗黙キャッシュ（自動・無料・**保証なし**）は、プレフィックスが一致している間しか効かない。
本番ログを数えたところ、効かない場面が2つあった:

| | キャッシュ率 | 原因 |
|---|---|---|
| 自由行動 | **0%** | 5 時間おきに起動するため、次までに失効する |
| 通常会話の loop2 以降 | **0%** | `tool_result` が入りプレフィックスが崩れる |

自由行動だけで月 3,600 円ぶん、毎回フルプライスで送っていた。
明示キャッシュは**保証があり**、`tool_result` が増えても崩れない（枠が別のため）。

## 何をキャッシュするか

`user_prompt` のうち、更新頻度が低い前半だけ:

    instructions（人格・ルール）            17,566字   起動時に固定
    記憶スキーマガイド                        3,328字   ほぼ不変
    記憶データ                              17,131字   日次の記憶整理で更新
    直近の過去会話                            5,489字   日次
    ミクのメモ内容（miku_note）              40,343字   1日1〜3回更新
    ────────────────────── ここまでキャッシュ（約45,000 tok・全体の90%）
    システムの現在状態 / this_conversation / 時刻 / 現在心境 / 入力  ← 毎回送る

`this_conversation` を**入れない**のは、毎ターン必ず追記されるため。含めると
毎ターン作り直しになり、書き込み料のほうが高くつく。

## 費用（Gemini 3.5 Flash）

| | $/1M tokens |
|---|---|
| 入力 | 1.50 |
| **キャッシュ読み** | **0.15**（1/10）|
| 保管 | 1.00 / 時間 |

保管は**時間課金**なので持ちっぱなしにしない。使う直前に作り、使い終わり（無通信
`_IDLE_SEC`）で消す。加えて TTL を短く切ってあるので、**プロセスが落ちて削除の信号が
届かなくても、Google 側で自動的に失効する**（保管料は最大 TTL ぶん）。

## 壊れないこと

キャッシュは**あれば使う**もので、無くても会話は成立する。作成・参照・削除のいずれが
失敗しても例外を投げず、呼び出し元は従来どおり全文を送る経路へ落ちる。
`_ENABLED = False` にすれば完全に従来動作に戻る。
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_logger = logging.getLogger(__name__)

# 明示キャッシュを使うか。False で従来動作（暗黙キャッシュ任せ）に戻る
_ENABLED = True

#: キャッシュの TTL（秒）。**削除の信号が届かなかったときの保険**。
#: 無通信での破棄（_IDLE_SEC）より長くしておくが、長すぎると
#: プロセスが落ちたときに払い続けることになるので短く切る。
_TTL_SEC = 5 * 60

#: 最後に使われてからこれだけ経ったら破棄する。会話が途切れたあと保管料を垂れ流さない。
#:
#: **短くしても体験は悪くならない**（2026-08-30 ユーザー判断）。作り直しが起きるのは
#: 「ループの頭」だけで、そこは元々 LLM を待つ場所。作成は**無料**で 2 秒しかかからず、
#: 利用者は1人・会話は間欠的なので、企業システムのように作り直しが積み重ならない。
#: 10分 → 3分で保管料は 1/3（月 $5.3 → $1.6）。
_IDLE_SEC = 3 * 60

#: これ未満のトークン数では作らない（モデルの下限を下回ると作成が失敗するため）。
#: Gemini 3.5 Flash の下限は 4,096。余裕を見て倍にしてある
_MIN_TOKENS = 8192

#: 1トークンあたりの文字数の目安（日本語・実測 1.39 字/tok）。作成前の足切りに使う
_CHARS_PER_TOKEN = 1.39

#: context を「キャッシュに載せる前半」と「毎回送る後半」に割る目印。
#: `agent_main._build_context` が埋め、`agent.main._build_user_prompt` が分割時に**必ず取り除く**
#: （LLM には渡さない）。見出しを境界にすると system_state や first_meeting が空のとき
#: 消えてしまうため、明示的なマーカーを置いている。
#: ここに置くのは、`agent_main` と `agent.main` の両方から参照でき、かつ
#: 依存の向き（agent_main → runner → agent.main）を壊さないため。
CACHEABLE_TAIL_MARK = "\n<<<CACHE_BOUNDARY>>>\n"


@dataclass
class _Entry:
    """いま持っているキャッシュ1件。"""

    name: str                      # Google 側のリソース名（google_cached_content に渡す値）
    fingerprint: str               # 元になった本文のハッシュ。変わったら作り直す
    tokens: int                    # キャッシュされたトークン数（保管料の計算に使う）
    created_at: float
    last_used_at: float
    expires_at: float
    model: str

    def alive(self, now: float) -> bool:
        return now < self.expires_at


_lock = threading.Lock()
#: 作成を直列化する。**状態用の `_lock` とは別**——作成には数秒かかるので、
#: その間 status() などが固まらないようにする。`_create_lock` を持ったまま
#: `_lock` を取ることはあるが、逆向きは無い（デッドロックしない）
_create_lock = threading.Lock()
_entry: _Entry | None = None
_client = None                     # google.genai.Client（遅延生成）

#: 作成・破棄の履歴（メモリ）。プロセスが落ちれば消える
history: list[dict] = []

#: 作成・破棄を時刻つきで残すファイル。**費用レポートはこれを読む**。
#: 会話ログに載せるだけでは、**破棄が会話の外で起きる**ため作成と対にならない
HISTORY_FILE = Path(__file__).resolve().parents[2] / "data" / "logs" / "context_cache.jsonl"


def _get_client():
    """google.genai クライアントを遅延生成する。失敗したら None（＝キャッシュ無効）。"""
    global _client
    if _client is not None:
        return _client
    try:
        from google import genai
        _client = genai.Client()
    except Exception as exc:                      # SDK 未導入・認証不備など
        _logger.warning("[context_cache] クライアントを作れないため無効化: %r", exc)
        _client = False
    return _client or None


def _cache_key(system_instruction: str, cached_text: str) -> str:
    """照合に使う本文。instructions と前半の両方（どちらかが変われば別のキャッシュ）。"""
    return system_instruction + "\n---\n" + cached_text


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _record(event: str, **kw) -> None:
    """作成・破棄をログと履歴に残す（あとから保管料を計算できるように）。

    保管料は「トークン数 × 保管した時間」なので、**作成と破棄が時刻つきで残っていないと
    後から計算できない**。記録先は3つで、役割が違う:

    | 先 | 用途 |
    |---|---|
    | `history`（メモリ）| 実行中の状態確認。プロセスが落ちれば消える |
    | 会話ログ（`agent_log`）| そのターンで何が起きたかを追う |
    | **`HISTORY_FILE`（JSONL）**| **費用の積算**。`scripts/llm_cost_report.py` が読む |

    **破棄は会話の外で起きる**（無通信3分の掃除・プロセス停止時）ので、会話ログだけでは
    作成と破棄が対にならず、保管時間を仮定値で埋めるしかなかった（2026-08-31 是正）。
    """
    rec = {"event": event, "at": time.time(), **kw}
    history.append(rec)
    if len(history) > 500:
        del history[:250]
    _logger.info("[context_cache] %s %s", event, kw)
    print(f"[context_cache] {event} " + " ".join(f"{k}={v}" for k, v in kw.items()))
    _append_history_file(rec)
    try:
        from agent.agent_logger import log as agent_log
        agent_log("context_cache", f"明示キャッシュ: {event}", meta=rec)
    except Exception:
        pass          # 会話の外（起動直後など）では記録先が無い。JSONL 側に残るので実害はない


def _append_history_file(rec: dict) -> None:
    """費用集計用に1行追記する。**失敗しても会話は止めない**。

    1日あたり数行しか増えないので、会話ログのような肥大は起きない。
    """
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:
        _logger.warning("[context_cache] 履歴の追記に失敗: %r", exc)


def is_enabled() -> bool:
    return _ENABLED


def get_name(fingerprint_source: str) -> str | None:
    """いま使えるキャッシュ名を返す。無い・古い・別内容なら None。

    呼び出し元は None のとき従来どおり全文を送る。
    """
    if not _ENABLED:
        return None
    now = time.time()
    with _lock:
        e = _entry
        if e is None:
            return None
        if not e.alive(now):
            _record("expired", name=e.name, age_sec=int(now - e.created_at))
            _clear_locked()
            return None
        if e.fingerprint != _fingerprint(fingerprint_source):
            # 記憶や miku_note が変わった → 作り直す（ここでは消すだけ。次回作られる）
            _record("stale", name=e.name)
            _drop_locked()
            return None
        e.last_used_at = now
        return e.name


def get_if_matches(system_instruction: str, cached_text: str) -> str | None:
    """同じ内容のキャッシュが生きていれば名前を返す。**作らないし壊さない**。

    段階1（相手が話している最中の方針づくり）から使う。`get_name` と違い、
    内容が食い違っても**捨てない**——段階1は1ターンに最大3回走るので、ここで
    作り直しを起こすと費用が跳ねるうえ、**記憶の取得が一時的に失敗しただけで
    固定部が変わる**（VDB のタイムアウトで実際に起きた・2026-08-30）。
    使えないときは黙って諦め、呼び出し元は従来どおり全文を送る。
    """
    if not _ENABLED:
        return None
    now = time.time()
    with _lock:
        e = _entry
        if e is None or not e.alive(now):
            return None
        if e.fingerprint != _fingerprint(_cache_key(system_instruction, cached_text)):
            return None
        e.last_used_at = now      # 使っている間は idle 破棄させない
        return e.name


def ensure(system_instruction: str, cached_text: str, model: str) -> str | None:
    """キャッシュを用意して名前を返す。作れなければ None（＝従来経路）。

    `cached_text` は user_prompt のうち固定部だけ。ここに毎ターン変わるものを
    含めてはいけない（含めると毎回作り直しになり、書き込み料のほうが高くつく）。

    **作成は直列化する**。ブラウザのリロードで接続が2回起きると先回り作成が並行し、
    2つ作られて片方が解放されずに保管料を払い続けていた（2026-08-30 実機で検出）。
    """
    if not _ENABLED:
        return None

    # ★**指紋には system_instruction も入れる**（2026-10-09）。キャッシュは instructions ごと Google 側に置かれ、
    #   使うリクエストからは instructions を送れない。前半（記憶など）だけで照合していたため、
    #   ブラウザ接続時の事前作成（状況の節が空の instructions）や別モードで作ったキャッシュが、
    #   別のモードのターンでもそのまま使われ、状況の節（「自発話しかけ」など）が抜けたまま答えていた。
    key = _cache_key(system_instruction, cached_text)
    name = get_name(key)
    if name:
        return name

    approx_tokens = int(len(cached_text) / _CHARS_PER_TOKEN)
    if approx_tokens < _MIN_TOKENS:
        return None                                # 下限割れ。作っても失敗する

    client = _get_client()
    if client is None:
        return None

    with _create_lock:
        # 待っている間に別のスレッドが作り終えているかもしれない
        name = get_name(key)
        if name:
            return name
        return _create_locked(client, system_instruction, cached_text, model)


def _create_locked(client, system_instruction: str, cached_text: str, model: str) -> str | None:
    """実際に作る。`_create_lock` 保持前提。"""
    try:
        from google.genai import types
        cache = client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(
                system_instruction=system_instruction,
                contents=[cached_text],
                ttl=f"{_TTL_SEC}s",
                display_name="miku_context",
            ),
        )
    except Exception as exc:
        # **握り潰すと原因が追えない**。従来経路で会話は続くが、ログには必ず残す
        _logger.warning("[context_cache] 作成に失敗（従来経路で続行）: %r", exc)
        print(f"[context_cache] 作成に失敗（従来経路で続行）: {exc!r}")
        return None

    now = time.time()
    tokens = 0
    try:
        tokens = int(cache.usage_metadata.total_token_count or 0)
    except Exception:
        pass

    global _entry
    with _lock:
        _entry = _Entry(
            name=cache.name, fingerprint=_fingerprint(_cache_key(system_instruction, cached_text)), tokens=tokens,
            created_at=now, last_used_at=now, expires_at=now + _TTL_SEC, model=model,
        )
    _record("created", name=cache.name, tokens=tokens, ttl_sec=_TTL_SEC)
    return cache.name


def prewarm_async() -> None:
    """ブラウザが繋がった時点で、裏でキャッシュを作っておく（2026-08-30）。

    **これが無いと、その会話の最初の発話だけ 2 秒余計に待たされる**
    （キャッシュはループの頭で作られるので、初回は作成を待つことになる）。
    接続から最初の発話までは、タイピングでも音声入力でも数秒あるため、
    その間に作り終えられる。

    接続処理をブロックしないよう**別スレッド**で走らせ、失敗しても黙って諦める
    （会話が始まればループ側の `ensure()` が作り直す）。
    """
    if not _ENABLED:
        return

    def _work() -> None:
        try:
            # context の材料はここでは持っていないので、agent_main 側の組み立てを借りる。
            # 循環 import を避けるため、呼ぶ瞬間に読み込む
            from agent_main import build_prewarm_material
            material = build_prewarm_material()
            if not material:
                print("[context_cache] 事前作成: 材料が揃わないため見送った")
                return
            instructions, cache_head, model = material
            name = ensure(instructions, cache_head, model)
            if not name:
                print("[context_cache] 事前作成: 作成できなかった（会話開始時に作り直す）")
        except Exception as exc:
            # **握り潰すと原因が追えない**。会話は止めないが、ログには必ず残す
            _logger.warning("[context_cache] 事前作成に失敗: %r", exc)
            print(f"[context_cache] 事前作成に失敗（会話には影響しない）: {exc!r}")

    threading.Thread(target=_work, daemon=True, name="Context-Cache-Prewarm").start()


def touch() -> None:
    """使われたことを記録し、TTL を延ばす（会話が続いている間は保持する）。"""
    if not _ENABLED:
        return
    now = time.time()
    with _lock:
        e = _entry
        if e is None or not e.alive(now):
            return
        # 残りが半分を切ったときだけ延長する（毎ターン API を叩かない）
        if e.expires_at - now > _TTL_SEC / 2:
            e.last_used_at = now
            return
        target = e
    client = _get_client()
    if client is None:
        return
    try:
        from google.genai import types
        client.caches.update(
            name=target.name,
            config=types.UpdateCachedContentConfig(ttl=f"{_TTL_SEC}s"),
        )
    except Exception as exc:
        _logger.warning("[context_cache] TTL 延長に失敗: %r", exc)
        return
    with _lock:
        if _entry is not None and _entry.name == target.name:
            _entry.expires_at = now + _TTL_SEC
            _entry.last_used_at = now


def sweep_if_idle() -> None:
    """最後に使われてから `_IDLE_SEC` 経っていたら破棄する。

    会話が途切れたあと保管料を払い続けないための掃除。定期スレッドから呼ぶ。
    """
    if not _ENABLED:
        return
    now = time.time()
    with _lock:
        e = _entry
        if e is None:
            return
        if now - e.last_used_at < _IDLE_SEC:
            return
        _record("idle_drop", name=e.name, idle_sec=int(now - e.last_used_at))
        _drop_locked()


def invalidate(reason: str = "") -> None:
    """内容が変わったので捨てる（miku_note 更新など）。次に使うときに作り直される。"""
    if not _ENABLED:
        return
    with _lock:
        if _entry is None:
            return
        _record("invalidated", name=_entry.name, reason=reason)
        _drop_locked()


def _drop_locked() -> None:
    """Google 側からも消す（失敗しても手元の参照は捨てる）。_lock 保持前提。"""
    e = _entry
    if e is None:
        return
    client = _get_client()
    if client is not None:
        try:
            client.caches.delete(name=e.name)
        except Exception as exc:
            _logger.warning("[context_cache] 削除に失敗（TTL で自然に消える）: %r", exc)
    _clear_locked()


def _clear_locked() -> None:
    globals()["_entry"] = None


def status() -> dict:
    """いまの状態（費用レポート・デバッグ用）。"""
    now = time.time()
    with _lock:
        e = _entry
        if e is None:
            return {"enabled": _ENABLED, "active": False}
        return {
            "enabled": _ENABLED, "active": True, "name": e.name, "tokens": e.tokens,
            "age_sec": int(now - e.created_at), "idle_sec": int(now - e.last_used_at),
            "expires_in_sec": int(e.expires_at - now),
            # 保管料の見込み（$1.00 / 1M tokens / 時間）
            "storage_cost_so_far": e.tokens * (now - e.created_at) / 3600 / 1e6,
        }
