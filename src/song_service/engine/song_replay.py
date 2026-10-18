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

"""**保存した工程の出力を差し込んで、LLM 抜きでパイプラインを回す**（2026-08-23）。

## 何のためか

1. **API キーもネットも要らない環境で通しの検証ができる**
   （GPU 無しの仮想マシンで、オリジナル曲が最後まで作れるかを確かめる）
2. **変換だけを直したときの回帰試験**——同じ LLM 出力から同じ音が出るかを見る。
   実際「`bar_filter` の判定を1本化したら音が変わらないか」を MIDI のハッシュで
   照合したが、あれを**毎回できる形**にする
3. **費用ゼロで何度でも回せる**（1曲 $0.8 → **$0**）

## 仕組み

工程の出力は `.steps.json` に**オブジェクトのまま**保存されている
（theme / design / chords / melody / lyrics / instrumentation / arrangement / harmony）。
LLM の呼び出しは `song_context._call` の1箇所に集約されているので、
そこを「保存済みを返す」関数に差し替えれば、**残り全部は本物のコードが動く**。

関門（`step_gate`）も LLM を使うので、こちらは「通す」に倒す
（再生は**判断の再現ではなく変換の検証**が目的）。

## 使い方

    import replay
    replay.install(Path("out/v3/生身の声.steps.json"))
    song = compose_song("（依頼文は保存済みのものが使われる）")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import song_design as compose_v2
import song_context as ctx                                 # 差し替える呼び出し口（2026-09-30 に song_compose から分けた）
import song_gate as step_gate

# 工程のラベル（`_call` の第3引数）→ `.steps.json` の steps のキー
LABEL_TO_STEP = {
    "テーマ": "theme", "設計": "design", "コード": "chords",
    # ★**グルーヴを登録し忘れていた**（2026-09-07 に工程を足したときの漏れ・09-08 修正）。
    #   保存はされていたのに引けず、再生が `RuntimeError` で落ちた。
    #   **工程を足したら、この表にも足す**——ここが漏れると再生も「続きから」も使えなくなる。
    "グルーヴ": "groove",
    # ※旋律とサビの詞の3案（`旋律1`〜`3`・`サビの詞1`〜`3`）は 2026-09-30 に1案にしたので外した
    "旋律": "melody", "旋律(空)": "melody",
    # ★**詞の工程を分解したときの追加**（2026-09-13）。上の警告どおり、ここへも足す
    #   ——足さないと、新しい曲の再生・編集・「続きから」が `RuntimeError` で落ちる。
    "詞の設計": "lyric_plan",
    "サビの詞": "lyric_chorus",
    "詞": "lyrics",
    "曲名": "naming",                                      # ★2026-10-03 の追加工程（詞のあと・どの形でも）
    # ★**2026-09-17〜23 に足した5工程が、まるごと抜けていた**（09-24 に検出・修正）。
    #   これらは `_call` を通らず**自分のモジュールから直接 `song_llm.run_sync` を呼ぶ**ので、
    #   差し替えの網の外にあった——**再生モードでも本物を呼んで課金されていた**
    #   （「LLM を呼ばない」と表示したまま）。`run_sync` 側で label を見て拾うようにした。
    "歌い方": "expression", "歌の崩し": "vocal_artic",
    "編成": "instrumentation", "伴奏": "arrangement", "ハモリ": "harmony",
    "彩り": "colour", "声部": "voices", "奏法": "articulation",
}

# 出力の型（保存された dict をこれに戻す）
_TYPES: dict[str, Any] = {}


def _types() -> dict[str, Any]:
    if _TYPES:
        return _TYPES
    from artic_plan import ArticPlan
    from harmony import HarmonyPlan, VoicePart
    from instrumentation import Instrumentation
    from song_arrange import ArrangePlan, ColourPlan
    from song_lyrics import ChorusLyrics, LyricPlan, MelodyLyrics
    from song_melody import BareMelody
    from song_naming import Naming
    from vocal_artic import VocalArticPlan
    from vocal_expression import ExpressionPlan
    _TYPES.update({
        "theme": compose_v2.Theme, "design": compose_v2.Design, "chords": compose_v2.Chords,
        "groove": compose_v2.Groove,                       # ★2026-09-07 の追加工程・09-08 に登録漏れを修正
        "melody": BareMelody, "lyrics": MelodyLyrics,
        "lyric_plan": LyricPlan,                           # ★2026-09-13 の追加工程（詞の分解）
        "lyric_chorus": ChorusLyrics,
        "naming": Naming,                                  # ★2026-10-03 の追加工程（曲名）
        "expression": ExpressionPlan,                     # ★2026-09-23 の追加工程（歌い方）
        "vocal_artic": VocalArticPlan,                     # ★同（歌の崩し）
        "instrumentation": Instrumentation, "arrangement": ArrangePlan, "harmony": HarmonyPlan,
        "colour": ColourPlan,                              # ★2026-09-17 の追加工程（彩り）
        "voices": VoicePart,                               # ★2026-09-23 の追加工程（声部）
        "articulation": ArticPlan,                         # ★同（伴奏の奏法）
    })
    return _TYPES


class Replay:
    """保存済みの工程出力を返す差し替え。"""

    def __init__(self, steps_path: Path, *, strict: bool = True):
        self.src = Path(steps_path)
        d = json.loads(self.src.read_text(encoding="utf-8"))
        # ★**走らなかった工程の空の出力（`skipped`）は保存に無いものとして扱う**（2026-10-03・F1）。形を変えて作り直すとき、
        #   前の形で走らなかった工程（インストの詞など）は新しい形で一から書く。いまの形で走らない工程は呼ばれない
        from song_model import made
        self.raw: dict[str, Any] = {k: v for k, v in (d.get("steps") or {}).items() if made(v)}
        self.instruction: str = d.get("instruction", "")
        # 曲の形（2026-10-03）。欄の無い古い記録は song（`song_model.form_of_record`＝知らない綴りは止める）
        from song_model import form_of_record
        self.form: str = form_of_record(d)
        self.history: list[dict] = d.get("history") or []
        self.used: list[str] = []
        self.missing: list[str] = []
        # strict=True … 保存に無い工程は**エラー**（再生＝変換の検証。LLM を1回も呼ばない）
        # strict=False … 保存に無い工程は**本物を呼ぶ**（＝**続きから作る**）
        self.strict = strict

    def call(self, agent, prompt: str, label: str):
        key = LABEL_TO_STEP.get(label)
        obj = self.raw.get(key) if key else None
        if obj is None:
            self.missing.append(label)
            if self.strict:
                raise RuntimeError(
                    f"再生できない工程『{label}』——保存に {key!r} が無い。"
                    f"（保存されているのは {sorted(self.raw)}）")
            print(f"  [続きから] {label} は保存に無いので**本物を呼ぶ**")
            self.last_was_live = True
            return _orig["_call"](agent, prompt, label)
        self.used.append(label)
        self.last_was_live = False
        # ★前回の「機械がした変更」「差分で直させた結果」を引き継ぐ（2026-09-27 監査）。戻した出力は直したあとの値で、
        #   処理をもう一度走らせても記録は出ない。引き継がないと、後の工程に理由が届かない
        try:
            import song_context as _sc
            from song_fix import carry_records
            carry_records(_sc.CURRENT_SONG, key, self.history)
        except Exception as e:                             # 記録の引き継ぎで再生を止めない
            print(f"  [再生] 前回の記録を引き継げなかった: {type(e).__name__}: {e}")
        model = _types().get(key)
        return _mark(model.model_validate(obj) if model else obj)

    # ---- 関門 ----
    # 再生では「判断の再現」ではなく**変換の検証**が目的なので、通すだけでよい。
    # ただし**続きから作る**ときは話が別で、**新しく作った工程は本物に見てもらう**
    # （でないと再開したぶんだけ品質の担保が抜ける）。直前の工程が本物だったかで分ける。
    last_was_live: bool = False

    # ★**本物の `judge` と引数を揃える**——ここは差し替え（スタンドイン）なので、
    #   向こうに引数が増えたらここも増やさないと `TypeError` で落ちる。
    #   実際 2026-09-16 に `material`（関門へ読ませる本文）を足したとき、ここを忘れて
    #   **再生が全部落ちた**。`**_kw` で余りを受け、本物へはそのまま渡す。
    #: ★**差し戻しの打ち止めで「所見つきで進む」にする点検役**（2026-09-30・出-1）。(点検役の名前, 判断)。
    #:   `song_compose._compose_song_rounds` が打ち止めの周で入れる。その点検役の最初の判断だけに使う
    stop_gate: tuple[str, Any] | None = None

    def judge(self, step, reason, status, trouble, findings, *,
              attempt=0, context="", verbose=True, **_kw):
        if not self.strict and self.last_was_live:
            return _orig["judge"](step, reason, status, trouble, findings,
                                  attempt=attempt, context=context, verbose=verbose, **_kw)
        # ★**保存から戻した工程の点検役は、保存した判断をそのまま返す**（2026-09-30・出-1）。以前は一律に
        #   「（再生）通す」を返していたので、前に点検役が書いた理由と「気になったところ」（`notes`・後の工程へ届く）が
        #   記録から消えた——差し戻しで工程を残す周・編集の作り直し（`song_remake.rebuild`）・再生のどれでも。
        #   LLM は呼ばない。判断はその点検役の最初の呼び出しにだけ使う（2回目以降はやり直しの結果なので戻さない）
        used = self.__dict__.setdefault("_gates_used", set())
        if step not in used:
            used.add(step)
            if self.stop_gate and step == self.stop_gate[0]:
                if verbose:
                    print(f"  [関門] {step}: **所見つきで進む**（差し戻しの打ち止め）")
                return self.stop_gate[1]
            from song_flow import usage_record_name       # noqa: E402（循環を避けてここで）
            name = usage_record_name(f"関門({step})")       # 点検役の記録の名前（melody_gate など）
            h = next((h for h in reversed(self.history) if h.get("step") == name), None) if name else None
            if h and h.get("request") in ("accept", "stop"):
                note = "前の判断を保存から戻した（点検役は呼んでいない）"
                if verbose:
                    print(f"  [再生] 関門（{step}）→ {h['request']}（{note}）")
                return step_gate.Verdict(reason=f"【保存から戻した判断】{h.get('reason', '')}", action=h["request"],
                                         notes=h.get("notes", "") or "", guidance=h.get("guidance", "") or "", redo_from="",
                                         machine=note)
        if verbose:
            print(f"  [再生] 関門（{step}）→ 通す")
        return step_gate.Verdict(reason="（再生）検証目的のため通す", action="accept", guidance="", notes="", redo_from="")

    # ---- 専用関数の工程（編成・伴奏・ハモリ）----
    def wrap_fn(self, name: str, label: str, orig):
        """保存にあれば差し込み、無ければ**本物の関数を引数ごと呼ぶ**。

        `_call` へ流すと agent が None になって落ちるので、ここは関数ごと分ける。
        """
        def _f(*a, **k):
            key = LABEL_TO_STEP.get(label)
            if self.raw.get(key) is not None:
                return self.call(None, "", label)
            self.missing.append(label)
            if self.strict:
                raise RuntimeError(f"再生できない工程『{label}』——保存に {key!r} が無い")
            print(f"  [続きから] {label} は保存に無いので**本物を呼ぶ**")
            self.last_was_live = True
            return orig(*a, **k)
        return _f


_state: Replay | None = None
_orig: dict[str, Any] = {}


def _mark(out: Any) -> Any:
    """保存から戻した出力に**印 `_replayed` を付ける**（工程の直し `fix_*` が、それを直させないため）。

    ★**`wrap_fn` 経由（編成・伴奏・ハモリ）で戻した出力には印が付いていなかった**（2026-09-28）。
      印は工程の関数（`arrange` の `_mark_replayed` など）が `run_sync` の戻り値の型を見て付けていたが、
      `wrap_fn` は工程の関数ごと差し替えるので、そこを通らない。印の無い伴奏を `fix_arrangement` が
      直しにかかり、ラベル「伴奏(直し1)」は再生の表に無いので**再生中に本物の LLM を呼んでいた**
      （費用がかかり、返事が毎回違うので**再生のたびに伴奏が変わった**。『傷だらけの音』『これが証だ』）。
    """
    try:
        out.__dict__["_replayed"] = True
    except Exception:                                      # noqa: BLE001（印を付けられない型は、そのまま返す）
        pass
    return out


class _Replayed:
    """保存から戻した出力。**`run_sync` の戻り値と同じ形**（`.output` を持つ）にする。

    `usage()` は**わざと持たない**——復元に費用はかからないので、
    `song_usage.record` が何も積まないのが正しい（あちらは例外を握りつぶす）。
    """
    __slots__ = ("output",)

    def __init__(self, output: Any) -> None:
        self.output = output


def hit(label: str) -> _Replayed | None:
    """`song_llm.run_sync` から呼ぶ。**その工程が保存にあれば戻した出力を返す**。

    ★**なぜ `run_sync` に継ぎ目を置くか**（2026-09-24）。
      `install` が差し替えるのは呼び出し口（`song_context`）の名前だけなので、
      **自分のモジュールから直接 LLM を呼ぶ工程は素通りしていた**
      ——歌い方・歌の崩し・声部・彩り・奏法の5つが、
      「LLM を呼ばない」はずの再生モードで**本物を呼んで課金されていた**。
      `label` を渡す呼び出しは全部ここを通るので、ここで拾えば取りこぼさない。

    `label` が表に無いときは、続きから作る（strict=False）なら**何もしない**（素通り）、
    再生（strict=True）なら**落とす**（2026-09-28）。表への登録漏れは
    `tests/song_service/test_step_registry.py` が落として教える。
    """
    st = _state
    if st is None or not label:
        return None
    key = LABEL_TO_STEP.get(label)
    if key is None:
        # ★**再生（strict）では、表に無いラベルも本物を呼ばない**（2026-09-28）。以前は素通りさせていたので、
        #   直しの呼び出し（「伴奏(直し1)」など・ラベルが動的で表に載らない）が**再生中に本物の LLM を呼び**、
        #   費用がかかったうえ、返事が毎回違って再生のたびに伴奏が変わった。黙って課金するより、落として知らせる
        if st.strict:
            st.missing.append(label)
            raise RuntimeError(f"再生中に、保存に無い LLM の呼び出し『{label}』が起きた（再生は LLM を呼ばない）")
        return None
    obj = st.raw.get(key)
    if obj is None:
        st.missing.append(label)
        if st.strict:
            raise RuntimeError(
                f"再生できない工程『{label}』——保存に {key!r} が無い。"
                f"（保存されているのは {sorted(st.raw)}）")
        print(f"  [続きから] {label} は保存に無いので**本物を呼ぶ**")
        st.last_was_live = True
        return None
    st.used.append(label)
    st.last_was_live = False
    model = _types().get(key)
    return _Replayed(_mark(model.model_validate(obj) if model else obj))

# ※2026-09-24 に置いた `replayed()`（伴奏を戻したら彩りを足さない判定）は 09-25 に外した。
#   彩りを作り直しても新しい彩りが鳴らない原因だった。伴奏は骨組みだけを保存するようにしたので、
#   二重を防ぐ判定そのものが要らなくなった。
#   （伴奏に彩りが混ざった 2026-09-25 までの記録を読み替える `normalize_record` も、2026-09-30 に外した——古い記録は読まない）


def resume(steps_path: Path) -> Replay:
    """**続きから作る**。保存済みの工程は $0 で復元し、無い工程だけ本物を呼ぶ。

    通信エラーなどで途中で落ちたときの立て直し用（工程ごとの保存は
    `song_flow._autosave` が書いている）。**やり直しの費用は落ちた所から先だけ**。

        import replay
        replay.resume(Path("out/v3/夜風と歩幅.steps.json"))
        song = compose_song(replay.instruction())
    """
    return install(steps_path, strict=False)


def install(steps_path: Path, *, strict: bool = True) -> Replay:
    """差し替える。**LLM は一度も呼ばれなくなる。**

    ★`_call` を差し替えるだけでは足りない。編成・伴奏・ハモリは `_call` を通らず
    **直接 `run_sync` している**ので、そちらも差し替える（2026-08-23 に発覚）。
    差し替えるのは**呼び出し口（`song_context`）の名前**。工程のモジュールは `ctx._call(...)`・`ctx.judge(...)` の形で
    呼ぶので、ここを差し替えれば全工程に効く（2026-09-30 に `song_compose` から分けるまでは `song_compose` の名前を差し替えていた。
    `from … import` で取り込んだ名前は、定義元のモジュールを差し替えても効かない）。
    """
    global _state
    _state = Replay(steps_path, strict=strict)
    names = ("_call", "judge",
             "plan_instruments", "arrange", "plan_harmony")
    if not _orig:
        for n in names:
            _orig[n] = getattr(ctx, n)
    ctx._call = _state.call
    ctx.judge = _state.judge
    for fn, label in (("plan_instruments", "編成"), ("arrange", "伴奏"),
                      ("plan_harmony", "ハモリ")):
        ctx.__dict__[fn] = _state.wrap_fn(fn, label, _orig[fn])
    if strict:
        print(f"■ 再生モード: {_state.src.name}"
              f"（保存されている工程 {len(_state.raw)}件・**LLM を呼ばない**）")
    else:
        print(f"■ 続きから: {_state.src.name}"
              f"（保存済み {sorted(_state.raw)} は $0 で復元し、残りだけ作る）")
    return _state


def uninstall() -> None:
    for n, f in _orig.items():
        setattr(ctx, n, f)


def instruction() -> str:
    return _state.instruction if _state else ""


def form() -> str:
    """戻している記録の曲の形（再生・続きから作る途中でなければ song）。`song_compose.compose_song` が `form` を省かれたときに読む。"""
    return _state.form if _state else "song"
