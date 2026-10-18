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

"""WF 起動単位のログ記録ユーティリティ。

各 WF（ミクメイン応答 / 会話ナレッジ登録 / 会話文記憶更新）の入口で `open_log(...)` を
コンテキストマネージャとして開き、その配下のあらゆる関数（subagent / tool / runner）から
`log("phase", "label", **fields)` で同じファイルへ書き込める。

【設計の要点】
  - contextvars で「現在の WF 起動」ログハンドルを保持するため、関数シグネチャを変えなくて良い
  - asyncio タスクごとに独立した値を持つので、自由行動と通常会話が並列でも干渉しない
  - 各イベントは 1 つの JSON オブジェクト（indent=2）+ 空行区切りで追記
  - プロンプト・LLM 出力・ツール結果は文字数制限せずに全文書く
"""
from __future__ import annotations

import contextvars
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# WF 名定数（ファイル名のプレフィックスに使う）
WF_MAIN_CHAT = "ミクメイン応答"
WF_KNOWLEDGE_REGISTER = "会話ナレッジ登録"
WF_MEMORY_UPDATE = "会話文記憶更新"

# プロジェクトの logs ディレクトリ（settings.json を見ずに直書き：パスは規約として固定）
_LOGS_DIR = Path(__file__).resolve().parents[2] / "data" / "logs"


class AgentLogger:
    """WF 起動 1 回分のログハンドル。"""

    def __init__(self, file_path: Path, wf_name: str, mode: int | None) -> None:
        self.file_path = file_path
        self.wf_name = wf_name
        self.mode = mode
        self._fh = file_path.open("w", encoding="utf-8")
        self._start_time = time.time()
        self._first_entry = True

    def write(
        self,
        phase: str,
        label: str,
        *,
        input: Any | None = None,
        output: Any | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """1 イベントを 1 つの JSON オブジェクトとして追記する。

        Args:
            phase: 機械可読のフェーズ名（kv_loaded / main_chat_request 等）
            label: 人間可読のラベル（処理対象を実態に合わせて記述）
            input:  この処理への「入力」（dict / str / 任意）
            output: この処理からの「出力」（dict / str / 任意）
            meta:   ループ番号・ツール名・経過秒数等の周辺情報

        いずれも None なら出力に含めない（純粋なメタイベントは meta のみで OK）。
        """
        entry: dict[str, Any] = {
            "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "phase": phase,
            "label": label,
        }
        if input is not None:
            entry["input"] = input
        if output is not None:
            entry["output"] = output
        if meta is not None:
            entry["meta"] = meta
        text = json.dumps(_expand_json_strings(entry), ensure_ascii=False, indent=2, default=_json_default)
        if not self._first_entry:
            self._fh.write("\n\n")
        self._fh.write(text)
        self._fh.flush()
        self._first_entry = False

    def close(self) -> None:
        try:
            self._fh.flush()
        finally:
            self._fh.close()

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self._start_time


def _try_parse_as_json(s: str) -> Any:
    """文字列を JSON としてパースする。複数オブジェクトが連結されていれば list で返す。

    json.loads は複数オブジェクトを受け付けないため raw_decode で順次抽出する。
    dict / list 以外にパースされる場合（数値・bool 等）や失敗時は None を返す。
    """
    stripped = s.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    decoder = json.JSONDecoder()
    results: list = []
    idx = 0
    while idx < len(stripped):
        while idx < len(stripped) and stripped[idx] in " \t\n\r":
            idx += 1
        if idx >= len(stripped):
            break
        if stripped[idx] not in "{[":
            return None  # オブジェクト間に非 JSON テキストがある
        try:
            obj, end = decoder.raw_decode(stripped, idx)
        except json.JSONDecodeError:
            return None
        results.append(obj)
        idx = end
    if not results:
        return None
    parsed = results[0] if len(results) == 1 else results
    return parsed if isinstance(parsed, (dict, list)) else None


def _expand_json_strings(obj: Any) -> Any:
    """dict/list/str を再帰的に走査し、JSON 文字列をオブジェクトに展開する。

    展開することで json.dumps の indent=2 が自動でインデントを揃えてくれる。
    複数 JSON オブジェクトが連結された文字列は list に変換して展開する。
    """
    if isinstance(obj, str):
        parsed = _try_parse_as_json(obj)
        if parsed is not None:
            return _expand_json_strings(parsed)
        return obj
    if isinstance(obj, dict):
        return {k: _expand_json_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_json_strings(item) for item in obj]
    return obj


def _json_default(obj: Any) -> Any:
    """JSON 化できない値（pydantic BaseModel / dataclass / Path 等）に対応する。"""
    # pydantic BaseModel
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # dataclass
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    # Path
    if isinstance(obj, Path):
        return str(obj)
    # bytes は要約だけ
    if isinstance(obj, bytes):
        return f"<bytes len={len(obj)}>"
    return repr(obj)


# --- プロンプトマーカー ---

_MARKER_RE = re.compile(r"^<<<[\w.]+:(?:START|END)>>>\n?", re.MULTILINE)


def mark_prompt(name: str, text: str) -> str:
    """プロンプトテキストにセクションマーカーを付与する（ログ記録用）。

    ログ上で <<<name:START>>> / <<<name:END>>> を grep してセクションを特定できる。
    LLM に渡す前に必ず strip_markers() で除去すること。
    """
    return f"<<<{name}:START>>>\n{text}\n<<<{name}:END>>>"


def strip_markers(text: str) -> str:
    """mark_prompt で付与したマーカーを除去して元のテキストに戻す（LLM 送出前に使う）。"""
    return _MARKER_RE.sub("", text)


# --- contextvar とアクセス API ---

_current_log: contextvars.ContextVar[Optional[AgentLogger]] = contextvars.ContextVar(
    "miku_agent_current_log", default=None
)


class _LogContext:
    """`with open_log(...) as logger:` のコンテキスト管理。"""

    def __init__(self, logger: AgentLogger) -> None:
        self._logger = logger
        self._token: contextvars.Token | None = None

    def __enter__(self) -> AgentLogger:
        self._token = _current_log.set(self._logger)
        return self._logger

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc is not None:
                # 例外時は最後に error イベントを残す
                try:
                    self._logger.write(
                        "error",
                        "WF 起動中に例外が発生",
                        meta={
                            "exception_type": exc_type.__name__ if exc_type else "",
                            "message": str(exc) if exc else "",
                        },
                    )
                except Exception:
                    pass
            try:
                self._logger.write(
                    "end",
                    "WF 起動終了",
                    meta={
                        "elapsed_sec": round(self._logger.elapsed_sec, 3),
                        "success": exc is None,
                    },
                )
            except Exception:
                pass
            self._logger.close()
        finally:
            if self._token is not None:
                _current_log.reset(self._token)
                self._token = None


def open_log(wf_name: str, mode: int | None = None) -> _LogContext:
    """WF 入口で呼ぶ。`with open_log(...)` でコンテキストを張る。

    ファイル名: `<wf_name>_YYYYMMDD_HHMMSS_mmm[_modeN].log`
    （末尾の mmm はミリ秒。同一秒内の連続起動でも衝突しないようにする）
    """
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # %f はマイクロ秒 6 桁。先頭 3 桁を残してミリ秒に丸める
    # （同一秒内の連続呼び出しでも衝突しないようにファイル名にミリ秒を含める）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    name = f"{wf_name}_{ts}"
    if mode is not None:
        name += f"_mode{mode}"
    name += ".log"
    file_path = _LOGS_DIR / name
    logger = AgentLogger(file_path, wf_name, mode)
    # 起動の最初のイベントを書いておく
    logger.write(
        "start",
        f"{wf_name} WF 起動",
        meta={"wf_name": wf_name, "mode": mode, "log_file": str(file_path)},
    )
    return _LogContext(logger)


def get_current_logger() -> AgentLogger | None:
    """現在のコンテキストに紐づく logger を返す。未 open なら None。"""
    return _current_log.get()


def is_logging() -> bool:
    """現在ログ記録中かを返す。重い処理（文字列構築等）の前にスキップしたい時用。"""
    return _current_log.get() is not None


def log(
    phase: str,
    label: str,
    *,
    input: Any | None = None,
    output: Any | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    """現在コンテキストの logger に 1 イベント書き込む。未 open なら no-op。

    Args:
        phase / label: 機械可読フェーズ名 + 人間可読ラベル
        input:  この処理への「入力」を表現する dict（または任意の値）
        output: この処理からの「出力」を表現する dict（または任意の値）
        meta:   ループ番号・ツール名・経過秒数等の周辺情報 dict
    """
    lg = _current_log.get()
    if lg is None:
        return
    try:
        lg.write(phase, label, input=input, output=output, meta=meta)
    except Exception as e:
        # ログ書き込み失敗は本処理を止めない
        print(f"[WARN] agent_logger.log 失敗 ({phase}): {e}")


# --- LLM の使用量 ---
#
# ★**使用量の取り出しと記録はここ1か所**（2026-10-08）。以前は主応答の1か所でしか記録しておらず、
#   段階1（plan_stage）・つなぎの相槌・記憶整理・ナレッジ登録・メモ更新などの費用がどこにも残っていなかった
#   （README の費用の目安が記録に出ている分だけで算出され、少なく出ていた）。
#   LLM を呼んだら `record_llm_usage(agent, result, kind=...)` を1行呼ぶ。
#
# ★**入力・出力が 0 で記録されていた件**（2026-09-22〜10-08）: pydantic_ai は使用量を genai-prices で取り出し、
#   取り出しに失敗すると例外を握りつぶして**入力・出力 0・details だけ**を返す。2026-09-22 の版上げで
#   genai-prices 0.1.7 が入り、pydantic_ai 1.107.6 が受け取れない項目を返すようになって毎回失敗していた
#   （requirements.txt で 0.0.66 に固定して直した）。同じことが起きたら気づけるように、
#   「details に入力のトークンがあるのに input_tokens が 0」を `usage_incomplete` として記録に出す。

#: 呼び出しの種類を書いていない記録の印
KIND_UNKNOWN = "unknown"

#: 会話の記録（`open_log`）の外で呼ばれた LLM の使用量を書くファイルの名前の頭。
#: 段階1・つなぎの相槌・曲の選曲のように、会話の記録が開く前／別のスレッドで呼ばれるものがここへ入る。
#: 1日1ファイル（名前の時刻は**その日の最初の記録**。`log_archive` の月ごとの束ねに乗る形）
WF_DETACHED_USAGE = "LLM使用量"

_detached_lock = threading.Lock()
_detached_path: dict[str, Path] = {}      # 日付 → その日のファイル


def usage_of(obj: Any) -> Any:
    """pydantic_ai の実行結果（`AgentRunResult` / `StreamedRunResult` / `AgentRun`）または
    `RunUsage` そのものから `RunUsage` を取り出す。

    1.107 から `usage` は**プロパティ**（括弧つきで呼ぶと非推奨の警告が出る）。古い版の
    メソッドの形でも取れるようにしておく。取れなければ None。
    """
    if obj is None:
        return None
    if hasattr(obj, "input_tokens"):
        return obj
    u = getattr(obj, "usage", None)
    if u is None:
        return None
    if not hasattr(u, "input_tokens") and callable(u):
        try:
            u = u()
        except Exception:
            return None
    return u if hasattr(u, "input_tokens") else None


def usage_fields(obj: Any) -> dict[str, Any]:
    """使用量から記録する数字を取り出す（**取り出しはここ1か所**）。

    返すキー:
      input_tokens       入力（キャッシュから読んだ分を**含む**）
      output_tokens      出力（**思考のトークンを含む**。Gemini は思考も出力の単価で課金される）
      cache_read_tokens  入力のうちキャッシュから読んだ分（割引の単価）
      thoughts_tokens    出力のうち思考の分
      requests           API を呼んだ回数
      details            pydantic_ai の内訳そのまま（モダリティ別など）
      usage_incomplete   True のとき、入力・出力が取れていない（上の 2026-09-22 の件。数字は信用しない）
    """
    u = usage_of(obj)
    if u is None:
        return {"usage_incomplete": True}
    details = dict(getattr(u, "details", None) or {})
    inp = int(getattr(u, "input_tokens", 0) or 0)
    out = int(getattr(u, "output_tokens", 0) or 0)
    cache = int(getattr(u, "cache_read_tokens", 0) or 0) or int(details.get("cached_content_tokens", 0) or 0)
    thoughts = int(details.get("thoughts_tokens", 0) or 0)
    prompt_in_details = sum(int(v or 0) for k, v in details.items() if k.endswith("_prompt_tokens"))
    f: dict[str, Any] = {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_tokens": cache,
        "thoughts_tokens": thoughts,
        "input_audio_tokens": int(getattr(u, "input_audio_tokens", 0) or 0),
        "requests": int(getattr(u, "requests", 0) or 0),
    }
    # 0 の内訳は書かない（以前からの書式。集計側は無いキーを 0 と読む）
    f = {k: v for k, v in f.items() if v or k in ("input_tokens", "output_tokens", "requests")}
    if details:
        f["details"] = details
    if (inp == 0 and (prompt_in_details or cache)) or (out == 0 and thoughts):
        f["usage_incomplete"] = True
    return f


def model_name_of(agent: Any) -> str:
    """Agent のモデル名（`gemini-3.5-flash` 等）。取れなければ空。"""
    try:
        m = getattr(agent, "model", None)
        return str(getattr(m, "model_name", "") or (m if isinstance(m, str) else "") or "")
    except Exception:
        return ""


def _write_detached(phase: str, label: str, meta: dict[str, Any]) -> None:
    """会話の記録の外で呼ばれた分を、その日の `LLM使用量_*.log` へ同じ書式で追記する。"""
    day = datetime.now().strftime("%Y%m%d")
    with _detached_lock:
        path = _detached_path.get(day)
        if path is None:
            _LOGS_DIR.mkdir(parents=True, exist_ok=True)
            found = sorted(_LOGS_DIR.glob(f"{WF_DETACHED_USAGE}_{day}_*.log"))
            path = found[0] if found else _LOGS_DIR / (
                f"{WF_DETACHED_USAGE}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.log")
            _detached_path.clear()
            _detached_path[day] = path
        entry = {
            "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "phase": phase,
            "label": label,
            "meta": meta,
        }
        text = json.dumps(entry, ensure_ascii=False, indent=2, default=_json_default)
        sep = "\n\n" if path.exists() and path.stat().st_size > 0 else ""
        with path.open("a", encoding="utf-8") as fh:
            fh.write(sep + text)


def log_llm_usage(model: str, usage: Any, *, label: str = "", kind: str = KIND_UNKNOWN) -> None:
    """LLM 呼び出し 1 回分のトークン使用量を記録する。**失敗しても本処理を止めない**。

    Args:
        model: モデル名（記録の `model`）
        usage: pydantic_ai の実行結果か `RunUsage`（`usage_fields` が取り出す）
        label: 人間可読のラベル
        kind:  呼び出しの種類（`main`・`plan_stage`・`filler` 等。集計で使う印）

    会話の記録（`open_log`）の中ならそのファイルへ、外なら `LLM使用量_<日付>.log` へ書く
    （どちらも phase は `llm_usage`・同じ書式）。会話の記録の中では `mode` も添える。
    """
    try:
        meta: dict[str, Any] = {"model": model, "kind": kind or KIND_UNKNOWN}
        lg = _current_log.get()
        if lg is not None and lg.mode is not None:
            meta["mode"] = lg.mode
        meta.update(usage_fields(usage))
        if meta.get("usage_incomplete"):
            print(f"[WARN] LLM 使用量が取れていない（{kind}・{model}）: 入力・出力が 0。"
                  "pydantic_ai と genai-prices の版の組み合わせを確かめる")
        text = label or f"LLM usage ({model})"
        if lg is not None:
            log("llm_usage", text, meta=meta)
        else:
            _write_detached("llm_usage", text, meta)
    except Exception as e:
        print(f"[WARN] LLM 使用量の記録に失敗 ({kind}): {e}")


def record_llm_usage(agent: Any, result: Any, *, kind: str, label: str = "") -> None:
    """`agent.run(...)` の結果の使用量を記録する（モデル名は agent から引く）。LLM を呼んだら1行これを呼ぶ。"""
    model = model_name_of(agent)
    log_llm_usage(model, result, label=label or f"{kind} usage", kind=kind)


def log_subagent_request(
    subagent: str,
    system_prompt: str,
    user_prompt: Any,
    *,
    meta: dict[str, Any] | None = None,
    **extra_inputs: Any,
) -> None:
    """Subagent の LLM 呼び出し前に入力プロンプト一式を記録する。

    Args:
        system_prompt / user_prompt: LLM への入力プロンプト本体
        extra_inputs: subagent への追加入力（attached_images / current_note 等）。
                      input ブロックにマージされる。
        meta: 入力ではない周辺情報（attempt 番号等）。meta ブロックにマージされる。
    """
    input_payload: dict[str, Any] = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    input_payload.update(extra_inputs)
    meta_payload: dict[str, Any] = {"subagent": subagent}
    if meta:
        meta_payload.update(meta)
    log(
        "subagent_request",
        f"{subagent} サブエージェントLLMへの入力",
        input=input_payload,
        meta=meta_payload,
    )


def log_subagent_response(
    subagent: str,
    output: Any,
    **extra_meta: Any,
) -> None:
    """Subagent の LLM 呼び出し直後に出力を記録する。"""
    log(
        "subagent_response",
        f"{subagent} サブエージェントLLMからの出力",
        output=output,
        meta={"subagent": subagent, **extra_meta},
    )
