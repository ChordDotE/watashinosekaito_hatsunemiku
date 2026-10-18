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

"""本番形プロトタイプ: 作曲 LLM 層（Gemini Flash・settings.json 準拠）。

song_plan_compose / melody_compose を pydantic_ai + Gemini Flash で実装。
LLM は音楽用語（拍・かな・コード名・編曲スタイル）で出力し、コードが tick / SongScore に変換。
作曲ガイドライン（§3.5 G-1..G-6）の機械的制約をプロンプトに反映。
"""
from __future__ import annotations

import json
import os
import re

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent

from song_schema import (
    ArrangementSpec, ChordSpan, DrumHit, PartSpec, SongNote, SongScore, StepEvent, Tone, VoiceTrack,
)

Q, BAR = 480, 1920

# ── settings.json からモデル・API キー ──
# 読む入口は `song_settings.load()` の1か所（`MIKU_SETTINGS`＝丸ごと差し替え・`MIKU_SETTINGS_OVERLAY`＝検証用の重ね書き・2026-10-01）
import song_settings as _song_settings                       # noqa: E402
from song_checks import JsonBool, JsonFloat  # noqa: E402（LLM が書く数と真偽の欄の型・2026-10-02 判断5）
from song_checks import half_width, strip_ascii_ws  # noqa: E402（半角の記法の前後の空白・仕様書 §0-6・2026-10-03）
_SETTINGS_PATH = _song_settings.base_path()
_SETTINGS = _song_settings.load()
os.environ.setdefault("GEMINI_API_KEY", _SETTINGS["api_keys"]["google"])
# モデル tier 使い分け: 大事な判断は pro、単純作業は flash（llm_models.md 方針）
# 曲づくり用のキー `song_service.pro_model` / `flash_model` / `lite_model` を読む。無い古い設定では `llm.*`（2026-10-03・`song_settings.models`）
_MODELS = _song_settings.models(_SETTINGS)
_PRO = _MODELS["pro"]      # pro を指定した工程用（2026-10-18 から指定する工程は無い。大事な判断も flash）
_FLASH = _MODELS["flash"]  # コーディネーター・点検役・検索（`reference_fetch._GOOGLE`）等
_LITE = _MODELS["lite"]    # 機械的タスク
_TIERS = {"pro": _PRO, "flash": _FLASH, "lite": _LITE}


def pick_model(role: str, default: str) -> str:
    """役割ごとにモデルを選ぶ。env `MIKU_<ROLE>_MODEL`(pro/flash/lite or ID) で上書き可。"""
    v = os.environ.get(f"MIKU_{role}_MODEL", default)
    return _TIERS.get(v, v)


# ---- やり直し（**判定と待ちはここ1箇所だけ**・2026-09-11 集約）---------------------
#
# ★同じ日に**2回**、`503 This model is currently experiencing high demand ...
#   Please try again later.` で**曲が丸ごと落ちた**。
#   `song_compose._call`（いまは `song_context._call`）にはやり直しがあったが、**選抜・関門・編成・伴奏・ハモリは
#   `_call` を通らず直接 `run_sync` している**ので、そこには効いていなかった。
#   判定が1箇所に無いと、こういう取りこぼしが起きる。

#: **混雑・一時的な不調**。待てば直るのでやり直す。**400/401 は直らない**のでやり直さない。
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

#: 何回目かの待ち（秒）。**1→4秒では混雑に足りなかった**（503 は数十秒続く）。
#: 最悪でも合計 65秒で、1曲 8〜10分に対しては小さい。
RETRY_WAITS = (1.0, 4.0, 15.0, 45.0)


def net_errors() -> tuple:
    """やり直してよい通信の例外（`httpx` が無い環境でも落ちないようにする）。"""
    try:
        import httpx
        return (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError,
                httpx.TimeoutException, ConnectionError)
    except Exception:                                       # noqa: BLE001
        return (ConnectionError,)


def retryable(e: Exception) -> bool:
    """**待てば直るものか**。`pydantic_ai` は HTTP の失敗を `ModelHTTPError` で投げる。"""
    if isinstance(e, net_errors()):
        return True
    code = getattr(e, "status_code", None)
    return isinstance(code, int) and code in RETRY_STATUS


# ---- 型の差し戻しが直らなかったとき（2026-09-30 本人方針・判-19）---------------------
#
# 出力の型（`Literal`＝選べる値・`field_validator`・`model_validator`）が受け付けないと、pydantic_ai が LLM に
# 検証エラーを返して書き直させる。既定の回数で直らなければ `UnexpectedModelBehavior` になる。
# それまでこの例外の行き先は工程でばらばらだった（曲づくりが例外で落ちる／「失敗しても曲を落とさない」工程で
# 飛ばして進む／記録にも残らない）。**形式の所見が差分で直らなかったとき（`song_fix.StepUnfixed`）と同じく、
# その工程で止める**。変換は LLM を呼ぶ共通の入口（`run_sync`）の1か所で行う。

def output_unfixed(e: BaseException) -> bool:
    """**出力の検証の再試行を使い切った例外か**（通信の失敗・出力が長さの上限で切れた・内容の検閲は含めない）。"""
    try:
        from pydantic_ai.exceptions import IncompleteToolCall, UnexpectedModelBehavior
    except Exception:                                       # noqa: BLE001
        return False
    if not isinstance(e, UnexpectedModelBehavior) or isinstance(e, IncompleteToolCall):
        return False
    m = (getattr(e, "message", "") or str(e)).lower()
    return "exceeded maximum output retries" in m or "exceeded max retries" in m


def validation_findings(e: BaseException) -> list[str]:
    """再試行を使い切った例外から、**最後に LLM へ返した検証エラー**を1件1文で取り出す（`StepUnfixed.findings`）。

    ★**日本語・1始まりで返す**（2026-10-04 組F3・4‑3）。この所見は止めた返事としてマスターへ渡るのに、以前は pydantic の英語・
      0始まりの位置（`parts.0.instrument: Input should be less than or equal to 15`）のままだった（作曲担当の画面では「パート1」）。
      位置は `type_error_text` が「parts の1番目の instrument」に、よく出る型のエラー（範囲・選択肢・必須・型・個数）は日本語にする。
      訳せないものは原文のまま「（型の差し戻しの原文）」と添える。"""
    def _one(d: dict) -> str:
        return type_error_text(d.get("loc"), d.get("msg", ""), d.get("input"), d.get("type", ""), d.get("ctx") or {})

    c: BaseException | None = e.__cause__ or e.__context__
    seen = 0
    while c is not None and seen < 6:
        errs = getattr(c, "errors", None)
        if callable(errs):                                  # pydantic の ValidationError
            try:
                return [_one(d) for d in errs()]
            except Exception:                               # noqa: BLE001
                pass
        tr = getattr(c, "tool_retry", None)                 # pydantic_ai の ToolRetryError（ModelRetry の差し戻し）
        if tr is not None:
            body = getattr(tr, "content", "")
            if isinstance(body, str):
                return [body]
            return [_one(d) for d in body]
        c, seen = c.__cause__ or c.__context__, seen + 1
    return [str(e.__cause__ or e)]


def type_error_where(loc) -> str:
    """検証エラーの位置 → 1始まりの日本語（`('parts', 0, 'instrument')` → `parts の1番目の instrument`）。"""
    s = ""
    for x in loc or ():
        if isinstance(x, int):
            s += f" の{x + 1}番目" if s else f"{x + 1}番目"
        else:
            s += (f"の {x}" if s.endswith("番目") else f" の {x}") if s else str(x)
    return s or "（出力全体）"


def type_error_text(loc, msg: str, value, typ: str = "", ctx: dict | None = None) -> str:
    """検証エラー1件 → 日本語の所見の文（位置は1始まり）。よく出る型だけ言い換え、ほかは原文に「（型の差し戻しの原文）」を添える。"""
    import re
    ctx = ctx or {}
    v = repr(value)
    shown = f"（書いた値: {v[:80] + '…' if len(v) > 80 else v}）"
    jp = {
        "missing": "必須の欄が無い（書く）",
        "extra_forbidden": "書けない欄（型に無い欄）",
        "less_than_equal": f"{ctx.get('le')} 以下であるべき",
        "less_than": f"{ctx.get('lt')} より小さいべき",
        "greater_than_equal": f"{ctx.get('ge')} 以上であるべき",
        "greater_than": f"{ctx.get('gt')} より大きいべき",
        "literal_error": f"選べる値のどれでもない——選べるのは {ctx.get('expected')}",
        "enum": f"選べる値のどれでもない——選べるのは {ctx.get('expected')}",
        "string_type": "文字列であるべき",
        "int_type": "整数であるべき", "int_parsing": "整数であるべき", "int_from_float": "整数であるべき（小数は書けない）",
        "float_type": "数であるべき", "float_parsing": "数であるべき",
        "bool_type": "true か false であるべき", "bool_parsing": "true か false であるべき",
        "list_type": "並び（リスト）であるべき", "dict_type": "欄の組（オブジェクト）であるべき",
        "model_type": "欄の組（オブジェクト）であるべき",
        "too_short": f"{ctx.get('min_length')}個以上であるべき（いま {ctx.get('actual_length')}個）",
        "too_long": f"{ctx.get('max_length')}個以下であるべき（いま {ctx.get('actual_length')}個）",
        "string_too_short": f"{ctx.get('min_length')}文字以上であるべき",
        "string_too_long": f"{ctx.get('max_length')}文字以下であるべき",
        "string_pattern_mismatch": f"書き方が決まりに合わない（決まり: {ctx.get('pattern')}）",
    }.get(typ or "")
    if jp is None and typ in ("value_error", "assertion_error"):
        # 工程の `field_validator`・`model_validator` が書いた文（日本語）なら、頭の英語（`Value error, `）だけ外してそのまま
        body = re.sub(r"^(Value error|Assertion failed),\s*", "", msg or "")
        if re.search(r"[぀-ヿ一-鿿]", body):
            jp = body
    if jp is None:
        jp = f"{msg}（型の差し戻しの原文）"
    return f"{type_error_where(loc)}: {jp}{shown}"


def _retries_of(e: BaseException) -> int:
    import re
    m = re.search(r"\((\d+)\)|count of (\d+)", getattr(e, "message", "") or str(e))
    return int(next(g for g in m.groups() if g)) if m else 0


def to_step_unfixed(e: BaseException, label: str, step: str = ""):
    """再試行を使い切った例外 → `song_fix.StepUnfixed`（止める工程・直らなかった所見）。
    いま作っている曲があれば、止める前に `<出力名>_fix` を `failed` で積む（`StepFixer.run` と同じ形）。"""
    from song_fix import StepUnfixed                        # noqa: E402（循環を避けてここで）
    sc = __import__("sys").modules.get("song_context")
    step = step or (sc.step_of_label(label) if sc is not None else label)
    findings = validation_findings(e)
    if sc is not None and label and sc.step_of_label(label) != step:   # 点検役（関門）など、止める工程と別の呼び出し
        findings = [f"{label}の出力: {f}" for f in findings]
    n = _retries_of(e)
    note = f"{label or step} の出力が型の検証を通らなかった（書き直しを{n}回頼んだ）"
    song = getattr(sc, "CURRENT_SONG", None) if sc is not None else None
    key = sc.step_output_key(step) if sc is not None else ""
    if song is not None and key:
        try:
            from song_model import StepRecord                  # noqa: E402
            song.record(StepRecord(step=f"{key}_fix", reason=note, warnings=list(findings), status="failed"))
        except Exception:                                   # noqa: BLE001（記録が書けなくても止まることは知らせる）
            pass
    return StepUnfixed(step, findings, song=song, rounds=n, last_note=note, via="type")


class KeepPrevious(BaseModel):
    """**前回のまま**（作り直しに巻き込まれた工程が、変える必要が無いと判断したとき）。2026-09-26。

    本人:「各工程で何も変えないスキップだけは選べるようにしてもよいかも。変更の経緯を一式渡せば、
    各工程が変更をするかしないかは正しく判断できる気がする」「出力もスキップのひとことだけにすれば、
    出力トークンも抑えられるかも」。**書き写させずに機械が前回の出力へ差し替える**（一字一句同じになる）。
    """
    keep_previous: JsonBool = Field(default=True, description="前回の出力をそのまま使う")
    reason: str = Field(description="なぜ変える必要が無いか（上流で何が変わり、なぜこの工程には響かないか）")


#: 「前回のまま」を選んだのに、前回の出力が今の形に合わず**機械が捨てて作り直させた**工程（2026-09-27・目録の道具-12）。
#: (工程名, LLM の理由, 合わなかった理由)。記録（`song_fix.record_change`）にも残す。曲ごとに呼ぶ側が空にしてよい
KEEP_REJECTED: list[tuple[str, str, str]] = []


def keep_rejected(sc, step: str, why: str, err: Exception) -> None:
    """「前回のまま」を機械が捨てたことを残す。いま作っている曲（`song_context.CURRENT_SONG`）があれば記録にも。"""
    detail = f"{type(err).__name__}: {str(err).splitlines()[0][:120] if str(err) else ''}"
    KEEP_REJECTED.append((step, why, detail))
    try:
        from song_fix import record_change
        key = getattr(sc, "_STEP_KEY", {}).get(step, step) if sc is not None else step
        record_change(getattr(sc, "CURRENT_SONG", None), key,
                      f"LLM は「前回のまま」を選んだ（理由: {why[:120]}）が、前回の出力が今の形に合わず"
                      f"使えなかったので、機械が同じ依頼でもう一度作らせた",
                      f"前回の出力の読み込みに失敗（{detail}）")
    except Exception:                                       # noqa: BLE001（記録が書けなくても曲づくりは止めない）
        pass


def keep_unfit(sc, step: str, why: str, findings: list[str]) -> None:
    """「前回のまま」を、**新しい形の形式の検査**（形を変えたとき・F11①）が通さなかったので取り消したことを残す。
    `KEEP_REJECTED` にも載せる（作り直しの返事の「前回のままを選んだが作り直させた工程」に出る）。"""
    detail = "新しい形の形式の所見: " + " / ".join(f[:100] for f in findings[:5])
    KEEP_REJECTED.append((step, why, detail))
    try:
        from song_fix import record_change
        key = getattr(sc, "_STEP_KEY", {}).get(step, step) if sc is not None else step
        record_change(getattr(sc, "CURRENT_SONG", None), key,
                      f"LLM は「前回のまま」を選んだ（理由: {why[:120]}）が、前回の出力は新しい形の形式の検査を通らなかったので、"
                      "前回のままを取り消して所見を添えてもう一度作らせた", detail)
    except Exception:                                       # noqa: BLE001（記録が書けなくても曲づくりは止めない）
        pass


class _Kept:
    """`KeepPrevious` が返ったときの結果。出力だけ前回のものに差し替え、ほか（費用など）は元の結果を見せる。"""

    def __init__(self, r, output):
        self._r, self.output = r, output

    def __getattr__(self, name):
        return getattr(self._r, name)


#: ★**工程ごとの依頼文と出力を残す先**（2026-09-26・本人指示）。呼ぶと書き先のフォルダを返す関数（無ければ残さない）。
#:   `coordinator.run` が「いま扱っている曲のフォルダ／log_prompts_v<作業中の版>」を返す関数を入れる
#:   （曲名でフォルダの名前が途中で変わるので、パスではなく関数で持つ）。
#:   それまで作曲側は**依頼文そのものをどこにも残していなかった**——『風の軌跡』で「編成の意図が伴奏に届いたか」を
#:   調べるのに、記録から依頼文を組み立て直す必要があった（ミク本体は依頼文を丸ごと残している）。
PROMPT_LOG = None
_PROMPT_SEQ = [0]


def usage_line(r) -> str:
    """1回（または1セッション）の使用量を1行で（2026-09-27・モデルを替えたときの費用を実測で比べるため）。
    思考のトークンは `details` に入る（Gemini は thoughts_tokens）。取れなければその旨を書く。"""
    try:
        from song_usage import usage_of                     # noqa: E402（循環を避けてここで。usage はプロパティ・1.107〜）
        u = usage_of(r)
        det = " ".join(f"{k}={v:,}" for k, v in sorted((u.details or {}).items()) if v)
        return (f"使用量: 入力 {u.input_tokens:,}（うちキャッシュ {u.cache_read_tokens:,}）"
                f" / 出力 {u.output_tokens:,} / 呼び出し {u.requests}回" + (f" / {det}" if det else ""))
    except Exception as e:                                  # noqa: BLE001（再生の結果は usage を持たない）
        return f"使用量: 取れなかった（{type(e).__name__}）"


def _log_call(label: str, agent, prompt, r) -> None:
    """1回の呼び出しの依頼文と出力を `PROMPT_LOG()` のフォルダへ1ファイルで書く。**失敗しても曲づくりは止めない**。"""
    if PROMPT_LOG is None:
        return
    try:
        import re
        from datetime import datetime
        d = PROMPT_LOG()
        if d is None:
            return
        d.mkdir(parents=True, exist_ok=True)
        _PROMPT_SEQ[0] += 1
        out = getattr(r, "output", None)
        body = out.model_dump() if hasattr(out, "model_dump") else out
        name = re.sub(r'[\\/:*?"<>|\s]+', "_", label or "呼び出し")[:40]
        text = (f"# {label or '（ラベルなし）'}  {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                f"モデル: {getattr(getattr(agent, 'model', None), 'model_name', '') or ''}  "
                f"出力の形: {getattr(getattr(agent, 'output_type', None), '__name__', '')}\n"
                f"{usage_line(r)}\n\n"
                f"## 依頼文\n{prompt if isinstance(prompt, str) else repr(prompt)}\n\n"
                f"## 出力\n{json.dumps(body, ensure_ascii=False, indent=1, default=str)}\n")
        (d / f"{datetime.now():%H%M%S}_{_PROMPT_SEQ[0]:03d}_{name}.md").write_text(text, encoding="utf-8")
    except Exception as e:                                  # noqa: BLE001
        print(f"  [依頼文の記録] 書けなかった（続行）: {type(e).__name__}: {e}")


#: ★**どの工程にも渡す書き方の決まり**（2026-10-03・洗い出しの判断4 本人決定・仕様書 §0-6）。前後の空白を読み手が黙って
#:   取り除くだけで、LLM には「付けない」と言っていなかった。指示書ごとに同じ文を散らさず、ラベルの付いた呼び出しが必ず通る
#:   `run_sync` の1か所で依頼文の末尾に足す（【曲の形】の見出しと同じ所）。どの欄が半角の記法かは仕様書 §0-6 の表
EDGE_RULE = ("\n\n※どの欄も、書く値の前後に余計な文字（空白・改行）を付けません。"
             "記法と名前の欄（ABC・コード・調・音名・打点・位置 `小節:拍`・at_bars・区切りの記号・打楽器名・ドラムキット・奏法・"
             "声と歌い方の名前など）は半角の字だけで書く欄で、前後に全角の空白・改行などが付いていると所見として返します"
             "（前後の半角の空白は読み飛ばします）。")


def run_sync(agent, prompt, *, label: str = "", stop_step: str = "", **kw):
    """`agent.run_sync` を**混雑・通信の失敗だけやり直して**呼ぶ。

    ★**モデルが変な出力を返したときはやり直さない**——同じものが返るので費用の無駄。
    ★**出力の型の差し戻しが直らなかったら、その工程で止める**（2026-09-30・`to_step_unfixed`）。
      止める工程はラベルから引く（`旋律(直し1)`→旋律）。点検役（ラベル `関門`）のように工程名でないラベルは、
      呼び側が `stop_step` で止める工程を渡す。ラベルの無い呼び出しは例外をそのまま投げる。

    ★**再生中は、保存済みの工程をここで返す**（2026-09-24）。`song_replay.install` は
      `compose_v3` に取り込まれた名前しか差し替えないので、**自分のモジュールから
      直接ここを呼ぶ工程は素通りしていた**（歌い方・歌の崩し・声部・彩り・奏法）。
      `label` を渡す呼び出しは必ずここを通るので、拾うならここが漏れない。
    """
    import sys
    import time
    if label:
        from song_replay import hit as _replay_hit          # noqa: E402（循環を避けてここで）
        got = _replay_hit(label)
        if got is not None:
            return got
        # ★**頼まれごとを、その工程の依頼文へ足す**（2026-09-25・段階2）。
        #   ラベルの付いた呼び出しは必ずここを通るので、工程ごとに書き足すより漏れない
        #   （再生の継ぎ目を置いたのと同じ理由）。曲づくりの外（`song_context` を読んでいない
        #   ／台帳が空）では何もしない。
        # ★**曲の形も同じ所で足す**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §2）。【曲の形】の見出しを
        #   ラベルの付いた呼び出し全部（工程・差分の直し・点検役）へ。歌もので形を変えていなければ空（依頼文は今までと同じ）
        # ★**書き方の決まり（前後に余計な文字を付けない・`EDGE_RULE`）も同じ所で足す**（2026-10-03・判断4）
        _sc = sys.modules.get("song_context")
        if _sc is not None and isinstance(prompt, str):
            # 差分の直し（`<工程>(直し{n})`）には「指摘への対応」の頼みを足さない（差分の型に `response` が無い・2026-10-04 組P2a）
            prompt = (prompt + _sc.form_heading() + _sc.asks_for(_sc.step_of_label(label), patch="(直し" in label)
                      + EDGE_RULE)
    # ★**「前回のまま」を選べる**（2026-09-26）。作り直しに巻き込まれた下流の工程だけ（`KEEP_ALLOWED`）。
    #   頼まれた当の工程と、関門に差し戻された工程には出さない（変えるために呼んでいるので）。
    keep_type = prev = None
    if label:
        _sc = sys.modules.get("song_context")
        _step = _sc.step_of_label(label) if _sc is not None else ""
        if _sc is not None and _step in getattr(_sc, "KEEP_ALLOWED", ()) and _step in _sc.PREV_OUTPUTS:
            keep_type = getattr(agent, "output_type", None)
            prev = _sc.PREV_OUTPUTS[_step]
            if isinstance(keep_type, type) and issubclass(keep_type, BaseModel):
                kw = {**kw, "output_type": [keep_type, KeepPrevious]}
            else:
                keep_type = None
            # 選べるのは**その工程の最初の呼び出しだけ**。2回目以降は関門のやり直し＝変えるために呼んでいる
            _sc.KEEP_ALLOWED.discard(_step)
    last: Exception | None = None
    for i in range(len(RETRY_WAITS) + 1):
        try:
            r = agent.run_sync(prompt, **kw)
            if keep_type is not None and isinstance(getattr(r, "output", None), KeepPrevious):
                try:
                    out = keep_type.model_validate(prev)
                except Exception as e:                      # 前回の出力が今の型に合わない＝差し替えられない
                    kw = {k: v for k, v in kw.items() if k != "output_type"}
                    keep_type = None
                    print(f"  [前回のまま] {label}: 前回の出力が今の形に合わないので作り直す")
                    # ★**捨てたことを記録に残す**（2026-09-27・目録の道具-12）。LLM は「前回のまま」を選んだのに、
                    #   機械がそれを捨てて作り直させた——記録に無いと、作り直しの報告で「変える必要が無い」と
                    #   判断した工程が黙って作り直されたように見える。
                    keep_rejected(_sc, _step, r.output.reason, e)
                    # ★**捨てた1回目の費用も数える**（2026-09-30）。呼び側が数えるのは返した結果（2回目）だけなので、
                    #   ここで数えないと1回目の費用が `USAGE` から漏れ、記録の合計が実際の費用より少なくなる
                    from song_usage import record as _rec_usage   # noqa: E402（循環を避けてここで）
                    _rec_usage(agent, r, label)
                    _log_call(label, agent, prompt, r)      # 「前回のまま」と答えたこと自体も残す
                    r2 = agent.run_sync(prompt, **kw)
                    _log_call(label, agent, prompt, r2)
                    return r2
                # ★**形を変えたときは、前回のままにした出力にも新しい形の形式の検査を当てる**（2026-10-03・F11①）。
                #   所見が出たら前回のままを取り消し、所見を添えてもう一度作らせる（取り消したことは記録に残す）
                bad = _sc.keep_findings(_step, out) if hasattr(_sc, "keep_findings") else []
                if bad:
                    kw = {k: v for k, v in kw.items() if k != "output_type"}
                    keep_type = None
                    print(f"  [前回のまま] {label}: 前回の出力が新しい形の形式の検査を通らないので作り直す（{bad[0][:80]}）")
                    keep_unfit(_sc, _step, r.output.reason, bad)
                    from song_usage import record as _rec_usage   # noqa: E402（捨てた1回目の費用も数える）
                    _rec_usage(agent, r, label)
                    _log_call(label, agent, prompt, r)
                    prompt = (prompt + "\n\n★**前回のままにはできない**——前回の出力は、新しい形の形式の検査を通らなかった:\n"
                              + "\n".join(f"・{b}" for b in bad[:12])
                              + "\nこの所見が直るように、この工程の出力を書く（変える必要の無い所は前回のまま残してよい）。")
                    r2 = agent.run_sync(prompt, **kw)
                    _log_call(label, agent, prompt, r2)
                    return r2
                _sc.KEPT.append((_step, r.output.reason))
                print(f"  [前回のまま] {label}: {r.output.reason[:80]}")
                _log_call(label, agent, prompt, r)          # 「前回のまま」と答えたこと自体を残す
                return _Kept(r, out)
            _log_call(label, agent, prompt, r)
            return r
        except Exception as e:                              # noqa: BLE001
            if label and output_unfixed(e):                 # 型の差し戻しが直らなかった＝その工程で止める（通信のやり直しはしない）
                raise to_step_unfixed(e, label, stop_step) from e
            if not retryable(e):
                raise
            last = e
            if i >= len(RETRY_WAITS):
                break
            w = RETRY_WAITS[i]
            code = getattr(e, "status_code", None)
            print(f"  [通信] {label or '呼び出し'} が {type(e).__name__}"
                  f"{f'（HTTP {code}）' if code else ''} で落ちた。{w:.0f}秒後にやり直す"
                  f"（{i + 2}/{len(RETRY_WAITS) + 1}回目）")
            time.sleep(w)
    raise last if last else RuntimeError(f"{label or '呼び出し'} が理由不明で失敗")


# 階名 ↔ MIDI。**カバーでも曲づくりでも使う**（旧オリジナル経路を消したとき
# 巻き添えで落としてしまい、`NameError` で発覚した・2026-09-02）。
_SOLFA = {"ド": 0, "レ": 2, "ミ": 4, "ファ": 5, "ソ": 7, "ラ": 9, "シ": 11}


def solfa_to_midi(solfa: str, octave: int) -> int:
    """階名 → MIDI 番号。**♯/♭ は前後どちらに書かれていても受ける**。

    ★以前は**後置（`ソ♯`）しか解釈せず**、LLM が書いた**前置（`♯ソ`）は
    `_SOLFA.get(s, 0)` の既定値で黙って「ド」になっていた**（2026-09-07 実測で発覚）。
    知らない階名がすべてドに化ける作りで、**間違った音が音符数を保ったまま通る**——
    エラーも警告も出ないので、聴くまで気づけない。**知らない綴りは落とす。**
    """
    s = solfa.strip()
    acc = 0
    for mark, d in (("♯", 1), ("#", 1), ("♭", -1), ("b", -1)):
        if s.endswith(mark) and len(s) > len(mark):
            acc, s = d, s[: -len(mark)]
            break
        if s.startswith(mark) and len(s) > len(mark):
            acc, s = d, s[len(mark):]
            break
    if s not in _SOLFA:
        raise ValueError(f"階名として読めない: {solfa!r}（ド/レ/ミ/ファ/ソ/ラ/シ に ♯♭ を付けた形だけ）")
    return (octave + 1) * 12 + _SOLFA[s] + acc


# ---- 国際式の音名（scientific pitch notation）——旋律の外で LLM が書く高さの記法（2026-10-01・判-A・判-B）----------
#
# ★**旋律（ABC）以外で LLM が書く高さは、全部この書き方1つ**（本人「1,2ともに標準記法にして」）: 伴奏・彩りの `pitches`・`riff`、
#   声部の `note`、ハモリ追加（`generate_added_voice`）の `note`。`C4`＝中央のド（MIDI 60）・`E5`・`F#4`・`Bb3`。
#   変化記号は ASCII の `#` と `b`（コード記号 `F#m`・`Bb` と同じ）。**書き方は1通り**——階名（`ミ5`）・♯♭ の記号・オクターブ無し・
#   小文字の音名（ABC では `e` が1オクターブ上を指すので取り違えやすい）は読まず、記法の所見にして書き直させる。
#   以前は伴奏・彩りが固定ドの階名＋オクターブ（`ミ5`・独自の組み合わせ）、声部が階名と英語の両方を受けていた（1つの欄に2通り）。
# ★**異名は標準どおり**: オクターブの数字は**文字（幹音）に付く**ので `B#3`＝`C4`＝60・`Cb4`＝`B3`＝59（以前の階名の読み手
#   `シ#3`＝60 と同じ高さ）。変化記号は1つだけ（`##`・`bb` は読まない）。
# ★読み手（`pitch_name_to_midi`）と記法の型（`PITCH`）はここ1つ。検査（伴奏・彩りの `written`・声部の型）と組み立て
#   （`song_arrange._pitch_of`・`harmony.voice_to_track`）が同じものを使う。書き手（表の綴り）は `song_table.spell_pitch`。
#   ★オクターブの数字は半角だけ（2026-10-02・洗い出し。Python の `\d` は全角の `４` も当たり、`C４` を `C4` と読んでいた）
#   ★**オクターブは `-1` か数字1つ**（2026-10-02・組「洗い出し-コードとABC」・仕様書 §10 の文法）。以前は先頭の 0（`C04`）・
#     `-0`（`C-0`＝`C0`）も同じ高さに読んでいた（同じ高さの書き方が何通りにもなる）。範囲の外（`C10`）は書き方の所見になる
_PITCH_RE = re.compile(r"([A-G])([#b]?)(-1|[0-9])")
_PITCH_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
#: 所見の文に添える書き方（指示書・型の説明と同じ例）
PITCH_HINT = "国際式の音名（`C4`＝中央のド・`E5`・`F#4`・`Bb3`。変化記号は `#` と `b`、オクターブの数字は必ず付ける）"


def pitch_name_to_midi(text: str) -> int | None:
    """**国際式の音名**（`C4`・`F#4`・`Bb3`・`C-1`）→ MIDI 番号。読めなければ None（鳴らせる高さの外でも数は返す）。
    前後は半角の空白とタブだけ取り除く（前後の全角の空白・改行などは読めない＝`PITCH` の所見・§0-6・2026-10-03）。"""
    m = _PITCH_RE.fullmatch(strip_ascii_ws(text))
    if not m:
        return None
    acc = {"#": 1, "b": -1}.get(m.group(2), 0)
    return (int(m.group(3)) + 1) * 12 + _PITCH_PC[m.group(1)] + acc


def _pitch_why(text: str) -> str:
    """読めない音名の、読めない理由（所見の文に添える）。"""
    s = strip_ascii_ws(text)
    if not s:
        return "空"
    if any(x in s for x in ("ド", "レ", "ミ", "ファ", "ソ", "ラ", "シ")):
        return "階名（ドレミ）は使わない（ド=C・レ=D・ミ=E・ファ=F・ソ=G・ラ=A・シ=B）"
    if any(x in s for x in "♯♭"):
        return "変化記号は記号の ♯ ♭ ではなく ASCII の `#` と `b`"
    if s[:1] in "abcdefg":
        return "音名の文字は大文字（`e5` ではなく `E5`）"
    if not any(c.isdigit() for c in s):                     # `isdigit` は全角の数字も数える（全角は下で「半角で」と返す）
        return "オクターブの数字が無い"
    if any(c.isdigit() and not "0" <= c <= "9" for c in s):
        return "数字は半角（`C４` ではなく `C4`）"
    if s[:1] in "#b" and s[1:2] in tuple(_PITCH_PC):
        return "変化記号は音名の後ろ（`#F4` ではなく `F#4`）"
    if re.fullmatch(r"[A-G](##|bb|#b|b#)-?\d+", s):
        return "変化記号は1つだけ"
    if re.fullmatch(r"[A-G][#b]?-?0[0-9]+", s) or re.fullmatch(r"[A-G][#b]?-0", s):
        return "オクターブの数字の頭に 0 を付けない・`-0` は書かない（`C04` ではなく `C4`・`C-0` ではなく `C0`）"
    if re.fullmatch(r"[A-G][#b]?-?[0-9]{2,}", s) or re.fullmatch(r"[A-G][#b]?-[2-9]", s):
        return "オクターブは -1〜9——鳴らせる高さ（MIDI 0〜127＝`C-1`〜`G9`）の外"
    return "書き方が違う"


def _pitch_problems(text: str) -> list[str]:
    k = pitch_name_to_midi(text)
    if k is None:
        return [f"音名『{text}』が読めない——{_pitch_why(text)}。{PITCH_HINT}で書く"]
    if not 0 <= k <= 127:
        return [f"音名『{text}』は鳴らせる高さ（MIDI 0〜127＝`C-1`〜`G9`）の外"]
    return []


from song_checks import Notation                            # noqa: E402（記法の型・2026-09-29）

#: **国際式の音名＝型1つ**（`song_checks.Notation`）。`read` は MIDI 番号（`pitch_name_to_midi`）
#: 前後の空白は半角の空白とタブだけ取り除き、ほかの空白（全角・NBSP・改行など）は所見（`half_width`・§0-6・2026-10-03）
PITCH = Notation("音名（国際式・`C4`＝中央のド）", half_width(_pitch_problems), pitch_name_to_midi)


# ★`midi_to_solfa`・`score_to_solfa`（MIDI 番号 → 固定ドの階名）は 2026-10-01 に外した。LLM に見せる高さは国際式の音名
#   （`song_table.spell_pitch`）にそろえ、本番から呼ばれなくなった（道具の返事の「階名: …」が最後の呼び手だった）。


# ── 追加声部の生成（モードB: 既存メインに重ねる声部。ハモリ/輪唱/重ね/コーラスを指示文から自律判別）──
#
# ★**声部の工程（`harmony.voices_for`）と同じ物差し・同じ部品**（2026-10-02・仕様書 §0-2・§10・§13・§14・ず-12。
#   本人「2通りに書けるものは良くない」「標準に寄せて1つにそろえる」）:
#   - 高さは **C メジャー／A マイナー基準**で見せて書かせ、機械が区間の調へ移す（移調量は `song_place.keys_shift`、
#     tick ごとの量は `harmony.section_shift_at`、見せる綴りは `song_table.c_basis_*`）。以前はハモリ追加だけ鳴る高さだった
#   - 歌える範囲は声部と同じ `C2`〜`C7`（鳴る高さで型が差し戻す）・推奨 `D3`〜`D6`（2026-10-05 に A#3 から広げた）（依頼文に C 基準の範囲で出す・
#     所見は `check_voices`）。以前はハモリ追加だけ `C2`〜`B6`
#   - 言葉 `lyric` は声部の言葉と同じ記法（`harmony.VOICE_WORD`・1音1モーラ・伸ばしは `_`）。以前は「うー」も書けた
#   - 組み立ては声部と同じ `harmony.voice_to_track`（位置・長さ・高さの読み手、伸ばしの訳し、表情の規則も同じ）。
#     以前はここに読み手の写しがあり、表情（しゃくり・揺れ・語尾の抜き）が付かず V6 の既定のままだった

class _AddedNote(BaseModel):
    # ★**声部の言葉と同じ記法**（`harmony.VOICE_WORD`・仕様書 §13）。欄の名前は以前のまま `lyric`
    lyric: str = Field(description=(
        "歌う**1モーラ**（**ひらがなだけ**。「きゃ」のような拗音は2文字で1モーラ）。前の音を伸ばすなら `_`"
        "（【メイン旋律】の伸ばしと同じ印。「ー」や `-` ではなく）。ハモリ/重ねはメインと同じ語、コーラスは「う」「あ」のあとに `_` で伸ばす。"
        "2モーラ以上（「うー」「かな」）・カタカナ・漢字・英字・「を」・`う_` のような印つき・`~` `*`・途中の空白・小書きだけ（`ゃ`）は"
        "書き直しを頼まれます（音を分けて1モーラずつ書く。`_` はそれだけで1つの音）"))
    # ★国際式の音名（2026-10-01・判-A/B）。**C 基準**（2026-10-02・ず-12①。以前は鳴る高さ）
    note: str = Field(description=(
        "高さ。国際式の音名（`C4`＝中央のド・`E5`・`F#4`・`Bb3`。変化記号は `#` と `b`、オクターブの数字は必ず付ける）。"
        "**【メイン旋律】の音名と同じ物差し**——**C メジャー／A マイナー基準（移調前）**で、書いた音は機械が区間の調へ移して鳴らす"
        "（依頼文の頭に移す量）。歌える範囲は移したあとの鳴る高さで `C2`〜`C7`"))
    # ★**位置は表と同じ `小節:拍`**（2026-10-01・仕様書 §0-4・§12・ず-12②）。読み手は声部と同じ `harmony.at_to_tick`。
    #   拍が1小節の長さ以上かは拍子を知っている型（`added_voice_type`）が見る
    at: str = Field(description=(
        "鳴らし始める位置。**【メイン旋律】と同じ書き方** `小節:拍`（`12:1.50`＝12小節目・拍1.50。小節は楽譜の頭から1・"
        "拍は 0 からの小数2桁・半角の数字）。ハモリ/重ねはメインと同じ位置、輪唱はずらす"))
    beats: JsonFloat = Field(gt=0, description="長さ（拍・4分音符＝1.0。【メイン旋律】の長さと同じ単位）")

    @field_validator("lyric")
    @classmethod
    def _readable_lyric(cls, v: str) -> str:
        from harmony import VOICE_WORD                      # noqa: E402（声部の言葉の記法は1つ・循環を避けてここで）
        if bad := VOICE_WORD.problems(v):
            raise ValueError(bad[0])
        return v

    @field_validator("at")
    @classmethod
    def _readable_at(cls, v: str) -> str:
        from song_checks import POSITION                    # noqa: E402（位置の記法の型は1つ）
        if bad := POSITION.problems(v):
            raise ValueError(bad[0])
        return v

    @field_validator("note")
    @classmethod
    def _readable_note(cls, v: str) -> str:
        # 読めるかだけ（声部の `VoiceNote` と同じ）。歌える範囲は移したあとの鳴る高さで見るので、移調量を知っている型
        # （`added_voice_type`）が見る
        if bad := PITCH.problems(v):
            raise ValueError(bad[0])
        return v

    @field_validator("beats")
    @classmethod
    def _sounding_beats(cls, v: float) -> float:
        from harmony import beats_problem                   # noqa: E402（長さの決まりは声部と同じ1つ）
        if why := beats_problem(v):
            raise ValueError(why)
        return v


class _AddedVoiceOut(BaseModel):
    reasoning: str = Field(description="指示の解釈（ハモリ/輪唱/オクターブ重ね/コーラスのどれか・対象区間・音程方針）")
    # ★空の名前は型で差し戻す（2026-10-02・洗い出し。以前は空なら黙って `harmony` にしていた）
    title: str = Field(description="声部名 harmony / canon / double / chorus 等（空にしない）")
    # ★範囲は声部（`harmony.HarmonyPart.gain_db`）と同じ（2026-10-02・洗い出し・仕様書 §15）。以前は範囲が無く、
    #   書き出し（`score_to_vpr`）が −60〜+6dB に黙って丸めていた（返事は書いた値を言う）。
    #   2026-10-03 に −18〜0 から書き出しの限界 −60〜+6 へ広げた（`harmony.VOICE_GAIN_DB`・§2 c10）
    gain_db: JsonFloat = Field(ge=-60, le=6, description="メイン(0dB)比の音量（−60〜+6）。ハモリ/コーラスは控えめ(-3〜-6)が自然")
    notes: list[_AddedNote] = Field(description="追加声部の音符（鳴る順に・位置は `小節:拍`・高さは C 基準）")

    @field_validator("title")
    @classmethod
    def _named(cls, v: str) -> str:
        if not (v or "").strip():
            raise ValueError("声部名（title）が空——harmony / canon / double / chorus などを書く")
        return v


def added_voice_type(bar_beats: float, shift_at, n_bars: int | None = None) -> type[_AddedVoiceOut]:
    """**拍子と移調量を知っているハモリ追加の型**（2026-10-01・ず-12②／2026-10-02・ず-12①③）。
    拍が1小節の長さ以上の位置（声部の `position_notation` と同じ所見）と、区間の調へ移した鳴る高さが歌える範囲
    （`harmony.VOICE_LO`〜`VOICE_HI`＝`C2`〜`C7`）の外の音を型で差し戻す。`shift_at` は tick → 移調量（`harmony.section_shift_at`）。
    ★**1本の声部の中で音が重なる・曲の外に置く音も差し戻す**（2026-10-03・記K・仕様書 §14）。工程の声部（`harmony.voice_findings`）と
      同じ決まり——1本の声部は同時に1音しか歌えない（前の音が終わる前に次の音が始まると重なり）、音は曲の中に少しでも掛かる
      （始まりが曲の終わり＝`n_bars` 小節目の終わりより前）。以前は定義が無く、重なった音も曲の外の音もそのまま組み立てへ渡った。
      `n_bars`（曲の小節数・依頼文の頭に出す数と同じ）を渡さなければ曲の外は見ない。"""
    from pydantic import model_validator
    from harmony import VOICE_HI, VOICE_LO, at_to_tick, beats_to_ticks   # noqa: E402
    from song_checks import position_notation               # noqa: E402
    from song_table import spell_pitch                      # noqa: E402

    pos = position_notation(bar_beats)
    bt = int(round(bar_beats * Q))

    class _AddedVoiceForMeter(_AddedVoiceOut):
        @model_validator(mode="after")
        def _beats_in_bar(self):
            bad: list[str] = []
            spans: list[tuple[int, int, int]] = []           # (開始, 終わり, 番号)
            for i, n in enumerate(self.notes, 1):
                why = pos.problems(n.at)
                bad += [f"{i}番: {w}" for w in why]
                if why:
                    continue
                t = at_to_tick(n.at, bt)
                spans.append((t, t + beats_to_ticks(n.beats), i))
                if n_bars and t >= n_bars * bt:
                    bad.append(f"{i}番: 位置 {n.at} は曲の外（曲は {n_bars}小節＝`{n_bars + 1}:0.00` より前に置く）"
                               "——曲の中の位置に直すか、この音をやめる")
                sh = int(shift_at(t))
                k = PITCH.read(n.note) + sh
                if not VOICE_LO <= k <= VOICE_HI:
                    bad.append(f"{i}番: 音名『{n.note}』はこの位置で {sh:+d}半音移して鳴る高さ {spell_pitch(k)} になり、"
                               f"歌える範囲（鳴る高さで `{spell_pitch(VOICE_LO)}`〜`{spell_pitch(VOICE_HI)}`）の外"
                               "——オクターブを変えるか別の音にする（書く高さは C 基準のまま）")
            spans.sort()
            for (_a0, a1, ia), (b0, _b1, ib) in zip(spans, spans[1:]):
                if b0 < a1:
                    bad.append(f"{ia}番と{ib}番: 音が重なっている（{ia}番が終わる前に{ib}番が始まる）。1本の声部は"
                               "**同時に1音しか歌えない**——前の音の長さか、次の音の位置を直す")
            if bad:
                raise ValueError("／".join(bad[:8]))
            return self

    _AddedVoiceForMeter.__name__ = "_AddedVoiceOut"         # LLM に見える型の名前は変えない
    return _AddedVoiceForMeter


_voice_agent: "Agent[None, _AddedVoiceOut]" = Agent(
    pick_model("VOICE", "flash"), output_type=_AddedVoiceOut,
    system_prompt=(
        "あなたは編曲者です。与えられた【メイン旋律】に重ねる**追加ボーカル声部を1本**作ります。\n"
        "人間の指示から声部の種類を**自分で判断**してください:\n"
        "・ハモリ: 各音にメインと**協和**する音（主に3度/6度の上下・その瞬間のコード構成音を優先）。拍はメインに合わせる。\n"
        "・輪唱(カノン/ラウンド): メインと**同じ旋律・歌詞**を、指定の拍だけ**遅らせて**始める。\n"
        "・オクターブ/ユニゾン重ね: メインと同じ（必要なら±オクターブ）。\n"
        "・バックコーラス: 控えめに。別歌詞（う・あ を `_` で伸ばす）や掛け合いも可。\n"
        "指示で区間が限定（例: サビ / 1番）されていれば**その範囲だけ**作る。\n"
        "高さは国際式の音名（`C4`＝中央のド・`E5`・`F#4`・`Bb3`）で、**【メイン旋律】の音名・【コード】と同じ物差し**——"
        "**C メジャー／A マイナー基準（移調前）**です（新しく作る曲の旋律・コード・声部と同じ。依頼文の頭に、機械が区間の調へ移す量が"
        "書いてあります）。メインの3度上なら、【メイン旋律】の音名から3度上の音をその物差しのまま書けば合います。\n"
        "音域: 依頼文の**声の推奨音域**（この物差しに直した範囲）の中に置く。歌える範囲は移したあとの鳴る高さで `C2`〜`C7`"
        "（外は書き直しを頼まれます）。\n"
        "位置 `at` は【メイン旋律】と同じ `小節:拍`（小節は楽譜の頭から1・拍は 0 からの小数2桁＝`12:1.50`。"
        "拍は1小節の長さより小さく、次の小節の頭は小節を進めて `13:0.00`）。長さ `beats` は拍数（4分音符＝1.0・16分＝0.25）。\n"
        "1本の声部は**同時に1音しか歌えない**ので、前の音が終わってから次の音を置く（重ねない）。音は曲の中に置く"
        "（依頼文の『曲の長さ』の小節数より後ろに置かない）。\n"
        "言葉 `lyric` は**ひらがなだけで、1つの音に1モーラ**（「きゃ」のような拗音は2文字で1モーラ）。2モーラ以上の言葉は"
        "音を分けて1モーラずつ書く（「うー」は「う」の音と `_` の音）。前の音を伸ばすなら `_`（「ー」や `-` ではなく。"
        "メインの伸ばしも `_` で見せています）。「を」は「お」、「へ」は「え」と書く。\n"
        "`_` はそれだけで1つの音に書く（`か_` ではなく `か` の音と `_` の音）。`~` `*`・言葉の途中の空白・小書きだけの言葉（`ゃ`）は書けない。\n"
        "音量 `gain_db` はメイン（0dB）比で −18〜0（声部名 `title` は空にしない）。\n"
        "位置・長さ・高さ・言葉は書いたとおりに鳴ります。表情（しゃくり・揺れ・語尾の抜き）だけは、主旋律と同じ規則で機械が付けます。\n"
        "並行5度/オクターブの連続や濁りは避ける。"
    ),
)


#: ハモリ追加（編集の道具 `adjust_song(add_voice=)`）の LLM 呼び出しのラベル。工程の記録・再生の表には載らない
#: （保存する工程の出力ではない・`tests/song_service/test_step_registry.py` の `not_replayed`）
ADDED_VOICE_LABEL = "ハモリ追加"


def generate_added_voice(score: SongScore, instruction: str, *, shift_of, bars_of: dict,
                         keys: list | None = None) -> tuple[VoiceTrack, list[str]]:
    """既存スコアのメイン旋律に重ねる追加声部を、人間の指示からLLMが生成して (VoiceTrack, 所見) で返す（モードB）。

    `shift_of`（区間 → 移調量・`song_place.keys_shift`）と `bars_of`（区間 → (開始小節, 終了小節)）は**必須**——声部の工程
    （`voices_for`）と同じく、主旋律とコードを C 基準で見せ、書かれた音を区間の調へ移す（2026-10-02・ず-12①）。
    分からない曲では呼ばない（呼び手が「調が分からないので足せない」と返す。黙って 0 にしない）。
    `keys`（`[(開始tick, 調名)]`・`song_table.key_regions`）は依頼文の頭に調の名前を添えるためだけ（無ければ移調量だけ）。

    ★**位置は `小節:拍`・伸ばしは `_`**（2026-10-01・仕様書 §12・§16・ず-12②）。主旋律とコードは曲の表と同じ書き手
      （位置＝`artic_plan.tick_key`・伸ばし＝`song_table._hold_shown`・C 基準の綴り＝`song_table.c_basis_*`）で見せる。
    ★組み立ては声部と同じ `harmony.voice_to_track`、音域の所見も声部と同じ `check_voices`（移したあとの鳴る高さで見る）。
    ★呼び出しは共通の入口 `run_sync`（ラベル `ADDED_VOICE_LABEL`）。型の差し戻しが直らなければ `song_fix.StepUnfixed` を投げる
      （呼び手の道具が「足さなかった」と返す）。
    """
    from artic_plan import tick_key                        # noqa: E402（位置の書き方は表と同じ1つ）
    from harmony import (VoiceNote, VoicePart, check_voices,  # noqa: E402（声部と同じ部品）
                         section_shift_at, sing_range_text, voice_to_track)
    from song_design import bar_ticks as _bar_ticks         # noqa: E402
    from song_table import _hold_shown, c_basis_chord, c_basis_header, c_basis_pitch   # noqa: E402

    if shift_of is None or not bars_of:
        raise ValueError("ハモリ追加に区間の移調量（shift_of）か区間の位置（bars_of）が無い——C 基準で見せられない")
    bt = _bar_ticks(score.time_signature or "4/4")
    shift_at = section_shift_at(shift_of, bars_of, bt)
    n_bars = max([int(b1) for _b0, b1 in bars_of.values()]
                 + [(n.start_tick + n.duration_tick - 1) // bt + 1 for n in score.vocal_notes] or [1])
    main = "\n".join(f"  {i}|{tick_key(n.start_tick, bt)}|{_hold_shown(n.lyric)}|"
                     f"{c_basis_pitch(n.key, n.start_tick, shift_at)}|{n.duration_tick / Q:.2f}拍"
                     for i, n in enumerate(score.vocal_notes, 1))
    chords = ("\n".join(f"  {tick_key(c.start_tick, bt)}|{c_basis_chord(c.chord, c.start_tick, shift_at)}"
                        for c in score.chords)
              or "  （コード情報なし＝メロディと調から判断）")
    prompt = (f"指示: {instruction}\nテンポ(BPM): {score.tempo}／拍子: {score.time_signature or '4/4'}"
              f"（1小節は 4分音符 {bt / Q:g}拍＝拍 `0.00`〜`{bt / Q - 0.25:.2f}`）／曲の長さ: {n_bars}小節"
              f"（音は `{n_bars + 1}:0.00` より前に置く）\n"
              f"{c_basis_header(shift_at, bt, n_bars, keys)}\n"
              f"{sing_range_text(shift_at, bt, bars_of)}\n"
              f"【メイン旋律】番号|小節:拍|かな|音名|長さ:\n{main}\n"
              f"【コード】小節:拍|コード:\n{chords}\n"
              f"この指示に沿った追加声部を1本作ってください。")
    # ★**共通の入口（`run_sync`）をラベル付きで通す**（2026-10-03・再レビュー）。以前は `_voice_agent.run_sync` を直に呼んでいて、
    #   費用（`song_usage`）・依頼文の記録・書き方の決まり（`EDGE_RULE`）に乗らず、型の差し戻しが直らないと
    #   `UnexpectedModelBehavior` のまま作曲担当ごと落ちた。いまは直らなければ `StepUnfixed`（工程名は `ADDED_VOICE_LABEL`）で止まる
    from song_usage import record as _rec_usage             # noqa: E402
    r = run_sync(_voice_agent, prompt, label=ADDED_VOICE_LABEL, output_type=added_voice_type(bt / Q, shift_at, n_bars))
    _rec_usage(_voice_agent, r, ADDED_VOICE_LABEL)
    out = r.output
    part = VoicePart(reason=out.reasoning, title=out.title.strip(),
                     # 追加の声部は狙いが1つ（この1本）なので 1（2026-10-04 組N2・`aim` は必須。以前は省いて 0）
                     notes=[VoiceNote(at=a.at, beats=a.beats, note=a.note, word=a.lyric, aim=1) for a in out.notes])
    vt, warns = voice_to_track(part, bt, gain_db=float(out.gain_db), shift_at=shift_at,
                               tempo=float(getattr(score, "tempo", 0) or 0) or None)
    # 曲の形は楽譜から（アカペラでは「曲の9割以上に乗っている」を所見にしない・F7。2026-10-03）
    return vt, warns + check_voices(score.vocal_notes, [vt], getattr(score, "form", "song") or "song")


# ---- 旧オリジナル経路は 2026-09-02 に削除 -----------------------------------
# `generate_song`（LLM 2回で曲を作る）と、その専用の型・エージェント・伴奏3テンプレを落とした。
# 区間の概念が無く、伴奏は band / pop / music_box の3つから返すだけで、
# 音源は numpy の合成音7種、調声も転調もハモリも無かった。
# いまは `song_compose` の8工程＋関門が作る（2026-09-01 に入口を差し替え、実機で確認済み）。
# ここに残るのは**カバーでも使う**もの——階名の変換・`generate_added_voice`（ハモリ追加）。
