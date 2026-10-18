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

"""**曲づくりの流れの仕組み**——工程の記録（`_rec`）・途中保存・途中で一度出す・前の工程への差し戻し・工程のあいだの受け渡し。

曲づくりが途中で抜ける合図は3つ: ここまでで一度出す（`ShowProgress`）・前の工程への差し戻し（`RedoFrom`）・
形式の所見が直らずに止まる（`song_fix.StepUnfixed`・2026-09-30）。どれも途中保存が残り、`remake_from_step` で続きを作れる。

2026-09-30 に `song_compose` から分けた。工程のモジュール（`song_outline`・`song_melody_step`・`song_lyrics`・`song_band` など）は
ここの `_rec`（記録と途中保存）と `_maybe_redo_from`（差し戻しの合図）を使い、流れの本体（`song_compose`）が
合図（`RedoFrom`・`ShowProgress`）を受けて作り直す・一度出す。

依存の向き: `song_context`（共有の状態・呼び出し口）→ **ここ** → 工程のモジュール → `song_compose`（工程を順に呼ぶ本体と入口）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from song_context import _STEP_KEY, step_of_label, step_output_key
from song_gate import FORM_STEPS, GATE_RECORD, GATE_STEP, STEP_ORDER, affected, is_ancestor, redo_targets
from song_model import STEP_LABEL, Song, StepRecord, made, skipped_output

# ---- 費用の計測 --------------------------------------------------------------
# 単価と記録の実体は `song_usage` が持つ（**二重管理にしない**）。ここは同じ実体を指すだけ。
from song_usage import USAGE                               # noqa: E402


# **工程が終わるたびに書き出す先**（None なら書かない）。
# 保存は最後に1回だけだったので、**途中で落ちると数分ぶんの LLM 出力が消えていた**。
# 書くのはローカルの JSON なので費用はゼロ。ここに残っていれば `replay.resume` で続きから作れる。
# 途中保存の置き場は**曲の置き場と同じ**（`song_pipeline.OUT_ROOT`）。
# テスト環境では `out/v3` に貯めていたが、本番はソースツリーに成果物を置かない。
AUTOSAVE_DIR: Path | None = Path(__file__).parents[3] / "data" / "songs" / "_wip"

#: **いま作っている曲が最後に途中保存された場所**（1曲ごとに `compose_song` の頭で消す）。
#: ★**途中で落ちた曲を、続きから作れるようにするため**（2026-09-25）。
#:   `compose_song` が例外で抜けると `Song` は呼び出し側に返らないので、曲名が分からず
#:   `draft_melody` は目印（`progress.json`）を書けなかった。曲名はここから拾う。
LAST_AUTOSAVE: Path | None = None
#: いまの曲の途中保存が**書こうとして失敗した**か（2026-10-03・再レビュー）。まだ一度も書いていないのとは別（`song_fix.StepUnfixed.message` が読む）
AUTOSAVE_FAILED: bool = False

#: **ここまで作ったら一度出す**工程（空なら最後まで作る）。`_compose_once` が張り、必ず戻す。
#: ★**工程の共通の出口（`_rec`）で見る**ので、工程ごとに書き足す必要がない——
#: 「一覧に足し忘れて効かない」が、このリポジトリで繰り返し起きている型なので避けた。
STOP_AFTER: str = ""


class ShowProgress(Exception):
    """**ここまでで一度出す**合図（2026-09-23）。

    本人:「**ミク自身がここで一旦止めようと決めたんだったら、ツールは一回抜けて
    メインのエージェントに戻して、また次、人間の指示を仰ぐ**」

    ★**「待つ」状態は作らない。** 待ち状態を持つと、返事が来るまで `.runner.lock` を
    握り続けて**他の曲づくりが全部塞がる**（返事が翌日なら丸一日）。
    **保存して抜ける**なら、ロックは自然に外れ、知らせる仕組み（`_notify_done`）もそのまま効く。

    再開は**編集と同じ経路**（本人:「編集の経路があるから、それと似たような感じになると思う」）
    ——`_wip/<曲名>.json` に工程が残っているので、`remake_from_step` が
    「どこから作り直すか」を決めて `replay.resume` で $0 復元する。
    """

    def __init__(self, step: str, song: "Song"):
        super().__init__(f"{step} まで作ったところで一度出す")
        self.step, self.song = step, song


class RedoFrom(Exception):
    """**前の工程まで戻ってやり直す**合図（2026-09-01）。

    関門が「この工程をやり直しても直らない。原因は前段にある」と判断したときに投げる。
    受けるのは `compose_song`——**その工程以降の出力を捨てて `replay.resume` で作り直す**。
    今日入れた「続きから作る」の逆向きで、**同じ機構をそのまま使う**。
    """

    def __init__(self, step: str, song: Song, guidance: str = "", who: str = ""):
        super().__init__(f"{step} から作り直す: {guidance[:120]}")
        self.step, self.song, self.guidance = step, song, guidance
        #: 差し戻した点検役の名前（空なら受け手が記録の最後の点検役から引く）。最終の点検役（2026-10-04 組P2b）は名前を付けて投げる——
        #: 差し戻しの回数を1段目と別に数えるため
        self.who = who


# 関門のラベル → 差し戻しの起点にできる工程名（**`song_gate.GATE_STEP` が正**・2026-09-27 に1か所へ）
_GATE_STEP = GATE_STEP


from song_fix import STOPS                                  # noqa: E402

#: **工程の中の「失敗しても曲全体は落とさない」`except Exception` が握ってはいけない例外**（2026-09-30）。
#: 止める（形式の所見が直らない・組み立ての不具合）・途中で一度見せる（`ShowProgress`＝`_rec` が投げる）・差し戻し（`RedoFrom`）。
#: 以前は `STOPS` だけを先に通していて、`show_after` にハモリ・声部・彩り・歌い方などを指定しても途中で抜けられなかった
#: （`_rec` の `ShowProgress` を `except Exception` が握っていた）。そういう所では `except NOT_SWALLOW: raise` を先に書く
NOT_SWALLOW = (*STOPS, ShowProgress, RedoFrom)


def _redo_options(gate_label: str) -> list[str]:
    """その関門で**実際に戻れる工程**（打ち止めなら無し）。関門の依頼文にはこれだけを見せる（2026-09-27・目録の関門-2）。
    ★**この曲の形で走る工程だけ**（2026-10-03・`song_gate.FORM_STEPS`）——走らない工程へは戻れない。"""
    if NO_MORE_REDO:
        return []
    import song_context as _ctx                            # いまの形（モジュールの属性で読む）
    runs = FORM_STEPS.get(_ctx.FORM, FORM_STEPS["song"])
    return [s for s in redo_targets(gate_label) if s in runs]


# **もう差し戻せない**（回数の上限 `MAX_REDO_SONG` に達した。費用の上限は 2026-09-30 に外した・出-1）。立てると `_maybe_redo_from` は投げず、
# **そのまま最後まで作る**。他の関門（設計・詞・伴奏）が「直せなくても曲は作り、記録に残す」
# 形で揃っているのに、**ここだけ途中の曲を返していた**（2026-09-02 是正）。
# 途中の曲を返すと、後段（`song_pipeline.compose`）が**伴奏の無い曲**を書き出そうとして
# 別の壊れ方をする。上限は「これ以上やり直さない」であって「作るのをやめる」ではない。
# ※これは**内容**（点検役）の差し戻しの話。**形式**の所見が差分で直らなかったときは、打ち止めかどうかに関係なく
#   その工程で曲づくりを止める（`song_fix.StepUnfixed`・2026-09-30 本人方針）。止めた曲は途中保存が残り、
#   マスターが頼めば止まった工程から続きを作る（`remake_from_step`）——途中の曲を書き出す所へは進まない。
NO_MORE_REDO = False


def _maybe_redo_from(v, gate_label: str, song: Song, rejected: dict[str, Any] | None = None) -> None:
    """関門が前工程を指していたら `RedoFrom` を投げる。指していなければ何もしない。

    `rejected` … 点検役が退けた稿で、まだ曲へ載せていないもの（`Song` の属性名 → 出力）。投げる前に曲へ載せる
    （2026-10-04 組F1・1‑1）。旋律・詞の点検役は曲へ載せる前のループの中で呼ばれるので、載せずに投げると差し戻しの周回
    （`song_compose._compose_song_rounds`）の保存に旋律・詞が無く、「前回の出力」（`PREV_OUTPUTS`）に入らなかった——
    旋律・詞は前の稿を見ずに一から書き直し、点検役も変わった所を比べられなかった。"""
    if NO_MORE_REDO:
        return                                             # 差し戻しは打ち止め。このまま先へ
    if v.action != "redo" or not getattr(v, "redo_from", ""):
        return
    here = _GATE_STEP.get(gate_label, gate_label)
    back = v.redo_from.strip()
    if back not in STEP_ORDER:
        return                                             # 知らない工程名は無視（鳴らさない側に倒す）
    if here in STEP_ORDER and STEP_ORDER.index(back) >= STEP_ORDER.index(here):
        return                                             # 前でないなら自工程のやり直しに任せる
    # ★**祖先でない工程へは戻さない**（2026-09-25）。列の上で前でも、依存の木で
    #   `here` の祖先でなければ、そこを作り直しても `here` は同じものが出てくる
    #   （伴奏の関門が「詞から」と言っても、伴奏は詞を読まない）＝差し戻しが空回りする。
    if here in STEP_ORDER and not is_ancestor(back, here):
        return                                             # 自工程のやり直しに任せる
    # ★関門には実際に戻れる工程だけを見せる（`_redo_options`）ので、上の3つの「黙って無視」は通らないはず。
    #   通ったら記録に残す（点検役の判断を黙って捨てない・2026-09-27）——ここへ来るのは通ったあとだけ
    for key, out in (rejected or {}).items():
        setattr(song, key, out)
    raise RedoFrom(back, song, v.guidance or v.reason)


def autosave_name(title: str) -> str:
    """途中保存・差し戻しの保存・作り直しの作業場のファイル名に使う曲名（`song_files.slug`＝曲フォルダ名と同じ1か所）。空なら `無題`。

    ★2026-10-03（再レビュー）: 以前は曲名をそのまま使っていたので、`?`・`:` で保存できず（途中保存は黙って失敗）・
      `/` でフォルダが1段でき・作り直しの作業場は書けずに落ちた。
    """
    from song_files import slug
    return slug(title) or "無題"


def _autosave(song: Song) -> bool:
    """途中経過を `<曲名>.steps.json` に書く。**失敗しても曲づくりは止めない**。書けたら True。

    ★**曲名が途中で変わったら古いほうを消す**。以前は `Song.title()` が 楽譜 → 詞 → テーマ の順に見たので、
      詞ができた時点で名前が入れ替わり、**古い名前のファイルが迷子として残った**（実際『夜風の現在地』という
      中途半端な記録が1つ残った）。前に書いた場所を覚えておき、名前が変わったら古いほうを消す。
      ★2026-10-03（曲名の工程）から、**曲名の工程の前は仮の曲名（テーマ）の名前で書き、曲名が決まったら新しい名前で書いて
      仮の名前のほうを消す**（`Song.title()` が 楽譜 → 曲名の工程 → 仮の曲名）。差し戻しの次の周は、前の周に書いた場所を
      `song_compose._start_song` が引き継ぐ（前の周の曲名の途中保存を残さない）。
    ★名前は曲名の slug（`autosave_name`・2026-10-03）。
    ★**書けなかったら `LAST_AUTOSAVE` を空にする**（2026-10-03・再レビュー）。以前は失敗を黙って握り、前に書いた場所
      （または空）を指したままにしていた——落ちた・止まった曲の目印（`tools/creation._mark_failed`）が「済んだ工程は保存してある」と
      伝える元はここなので、いまの曲の最後の工程が残っていないのに「保存してある」と伝えうる。
    """
    global LAST_AUTOSAVE, AUTOSAVE_FAILED
    if AUTOSAVE_DIR is None:
        return False
    try:
        path = AUTOSAVE_DIR / f"{autosave_name(song.title())}.json"
        prev = getattr(song, "_autosave_path", None)
        song.save(path)
        if prev is not None and prev != path:
            for old in (prev, prev.with_suffix(".steps.json")):
                old.unlink(missing_ok=True)
        song._autosave_path = path                         # type: ignore[attr-defined]
        LAST_AUTOSAVE = path
        AUTOSAVE_FAILED = False
        return True
    except Exception as e:                                 # noqa: BLE001（保存に失敗しても曲づくりは止めない）
        LAST_AUTOSAVE = None
        AUTOSAVE_FAILED = True
        print(f"  [途中保存] ★書けなかった（{type(e).__name__}: {str(e)[:120]}）——この曲の続きは保存から作れない")
        return False


# ---- 記録の工程ごとの費用（2026-09-30・出-4）----------------------------------------------
# **工程の記録の `usd` は、その工程で使った費用の合計**。呼び出しは `song_usage.USAGE` に1回ずつ積まれ（ラベルつき）、
# ここでラベルから記録の名前を引いて足し込む:
#   ・工程の呼び出し・差分の直し（`旋律(直し1)`）・自工程のやり直し → その工程の記録（同じ名前が複数あれば**最後の記録**）
#   ・点検役（`関門(旋律)`・`関門(旋律・聞き直し)`）→ その点検役の記録（`melody_gate` など）
#   ・保存から戻した工程（再生）は呼び出しが無いので 0。差し戻しで捨てた周回の費用は、`song_compose` が
#     差し戻しの記録（`redo`）に載せる——**1曲の記録の合計＝実際に払った費用**になる
# 割り当ては `_rec` のたびと、1周の終わり（`final=True`）。割り当て先の記録がまだ無い呼び出しは次に回し、
# 1周の終わりでも行き先が無ければ `usage` の記録にまとめて載せる（黙って落とさない）。
#: 記録へ割り当てていない呼び出しの始まり（`USAGE` の添字）。None なら割り当てない（曲づくりの外）。
USAGE_FROM: int | None = None
_PENDING_USAGE: list[dict] = []

#: 工程の出力の名前（`Song` の属性）と、記録の名前が違うもの。伴奏は `arrange`。
#: ※声部は 2026-09-30（出-13）から `voices` の名前で記録する（`song_band.run_voices`）。以前は `harmony` の記録に寄せていた
_RECORD_OF = {"arrangement": "arrange"}


def begin_usage() -> None:
    """この周の費用の割り当てを始める（`song_compose._compose_once` の頭）。"""
    global USAGE_FROM
    USAGE_FROM = len(USAGE)
    _PENDING_USAGE.clear()


def end_usage() -> None:
    """曲づくりの外では割り当てない（`_compose_once` の終わり）。"""
    global USAGE_FROM
    USAGE_FROM = None
    _PENDING_USAGE.clear()


def usage_record_name(label: str) -> str:
    """呼び出しのラベル → 費用を載せる記録の名前（分からなければ空）。"""
    label = label or ""
    if label.startswith("関門("):                          # 点検役: 関門(旋律)・関門(旋律・聞き直し)
        body = label[len("関門("):].rstrip(")").split("・")[0]
        if body in GATE_RECORD:                            # 工程名でない関門（仕上げ → finish_gate・2026-10-03）
            return GATE_RECORD[body]
        key = step_output_key(GATE_STEP.get(body, body))
        return f"{_RECORD_OF.get(key, key)}_gate" if key else ""
    key = step_output_key(step_of_label(label))            # 旋律・旋律(空)・旋律(直し1)・サビの詞 …
    return _RECORD_OF.get(key, key) if key else ""


def settle_usage(song: Song | None, *, final: bool = False) -> bool:
    """まだ割り当てていない呼び出しの費用を、曲の記録へ足し込む。何か足したら True。

    `final=True`（1周の終わり）なら、行き先の無い呼び出しを `usage` の記録にまとめる。
    """
    global USAGE_FROM
    if USAGE_FROM is None or song is None:
        return False
    _PENDING_USAGE.extend(USAGE[USAGE_FROM:])
    USAGE_FROM = len(USAGE)
    changed, left = False, []
    for u in _PENDING_USAGE:
        name = usage_record_name(u.get("step", ""))
        rec = next((r for r in reversed(song.history) if r.step == name), None) if name else None
        if rec is None:
            left.append(u)
            continue
        rec.usd += float(u.get("usd", 0.0) or 0.0)
        changed = True
    _PENDING_USAGE[:] = left
    if final and left:
        labels = sorted({u.get("step") or "（ラベルなし）" for u in left})
        song.record(StepRecord(step="usage", reason="工程の記録に割り当てられなかった呼び出しの費用: " + "、".join(labels),
                               usd=sum(float(u.get("usd", 0.0) or 0.0) for u in left)))
        _PENDING_USAGE.clear()
        changed = True
    return changed


def _rec(song: Song, step: str, out: Any, *, request: str = "",
         warnings: list[str] | None = None, verbose: bool = False) -> StepRecord:
    """工程の出力から **StepRecord を作って `Song.history` に積む**（D8/D9）。

    `status` / `trouble` は工程自身の申告（`StepStatus` を継承した出力型が持つ）。
    **妥協・失敗はここで初めて「見える」**——今までは reason だけ残して先へ進んでいた。
    """
    rec = StepRecord(
        step=step,
        reason=(getattr(out, "reason", "") or "").strip(),
        # 工程の申告は型（`song_model.Status`）が3つの綴りだけを受ける——ここで読み替えない（2026-10-02。以前は strip・lower で読んだ）
        status=getattr(out, "status", "ok"),
        trouble=(getattr(out, "trouble", "") or "").strip(),
        warnings=list(warnings or []),
        request=request,
    )
    song.record(rec)
    # ★**費用は呼び出しのラベルで記録へ割り当てる**（2026-09-30・出-4）。以前はここで `usd=USAGE[-1]["usd"]`
    #   ＝直前の呼び出し1回ぶんを入れていたので、記録の合計（$1.02）が実際（$3.34）と合わなかった
    settle_usage(song)
    _autosave(song)                                        # 途中で落ちても、ここまでは残す
    if verbose and not rec.is_clean():
        print(f"      ▲{STEP_LABEL.get(step, step)}: {rec.status}"
              + (f" — {rec.trouble[:100]}" if rec.trouble else ""))
        for w in rec.warnings:
            print(f"      ★{w}")
    if STOP_AFTER and step == STOP_AFTER:                  # ★ここまでで一度出す（保存は上の _autosave 済み）
        raise ShowProgress(step, song)
    return rec


def drop_keys(step: str) -> set[str]:
    """`step` から作り直すとき、**保存から捨てる工程のキー**（依存の木を辿る・2026-09-25）。

    それまでは `STEP_ORDER` の**列の後ろ全部**を捨てていた——歌い方を直すと編曲が丸ごと、
    詞を直すと詞を読まない編成・伴奏まで作り直した。**作り直しと関門の差し戻しの2か所が
    同じ計算を別々に持っていた**ので、ここ1か所にまとめた。
    """
    return {_STEP_KEY[s] for s in affected(step) if s in _STEP_KEY}


def prev_outputs_of(steps: dict, step: str) -> dict:
    """`step` から作り直すとき、**作り直す工程それぞれの前回の出力**（工程名 → 出力）。`PREV_OUTPUTS` へ積む。
    走らなかった工程の空の出力（`skipped`）は前回の出力ではない（`song_model.made`）。"""
    return {s: steps[_STEP_KEY[s]] for s in affected(step)
            if s in _STEP_KEY and made(steps.get(_STEP_KEY[s]))}


def remade_steps(kept: dict, form: str) -> set[str]:
    """`kept`（次の周に保存から戻す工程のキー → 出力）のとき、**次の周で作り直す工程**（その形で走る工程のうち、保存に無いもの）。"""
    return {s for s in STEP_ORDER if s in FORM_STEPS[form] and s in _STEP_KEY and _STEP_KEY[s] not in kept}


def prev_outputs_for(known: dict, kept: dict, form: str) -> dict:
    """**次の周で作り直す工程それぞれの前回の出力**（工程名 → 出力・2026-10-04 組F1・1‑4）。`known` は周をまたいで
    最後に作った出力（キー → 出力）。差し戻しで捨てる工程だけでなく、前の周で捨ててまだ作り直していない工程にも前の稿を渡す。"""
    return {s: known[_STEP_KEY[s]] for s in remade_steps(kept, form) if made(known.get(_STEP_KEY[s]))}


def fill_skipped(song: Song) -> None:
    """**その曲の形で走らない工程の出力を、中身が空・状態 skipped で入れる**（2026-10-03・F1 本人決定「インストやアカペラでも、
    データ形式は変えないようにしたい。空配列とかで」）。曲を作り始めるとき（`song_compose._start_song`）に呼ぶ。

    以前は走らない工程の出力が None のままで、記録の `steps` の形が曲の形で変わっていた（インストの詞・アカペラの伴奏が null）。
    いまはどの形でも全工程が同じ型の出力を持ち、走らなかった工程は `skipped_output`（理由も空——機械は中身を作らない）。
    出力の型は再生の表（`song_replay._types`）の1か所から引く。"""
    from song_replay import _types                          # noqa: E402（循環を避けてここで）
    for step in FORM_STEPS["song"]:
        if step in FORM_STEPS[song.form]:
            continue
        key = step_output_key(step)
        setattr(song, key, skipped_output(_types()[key]))


@dataclass
class ComposeState:
    """**工程のあいだで受け渡す値**（2026-09-30・`song_compose._compose_once_body` を工程ごとの関数に分けたときに作った）。

    **工程の出力は曲（`song`）が持つ**——後の工程は前の工程の出力を `song.theme`・`song.melody`・`song.lyrics` などで読む
    （規約「工程はそれぞれ自分の持ち物だけを持つ」）。ここに置くのは、工程の出力ではないが後の工程が使う**途中の値**だけ
    （組み立てた歌の音符・時間軸・編成から引いた楽器の番号・楽譜など）。欄を足すときは、どの工程が書き・どの工程が読むかを横に書く。
    """
    song: Song
    instruction: str = ""
    verbose: bool = True
    #: 工程ごとの理由・所見の作業用の控え（工程名 → 文）。1周の中でだけ使う——伴奏の点検役へ渡す `arrange`、
    #: 声部の記録に添える `harmony` など。曲には残さない（曲の記録は `song.history`。旧 steps.json の `_meta` は 2026-09-30 に外した）
    log: dict = field(default_factory=dict)
    # ---- 骨組み（`song_outline`）が書く ----
    ch_map: dict = field(default_factory=dict)             # 区間 → 小節ごとのコード（コードの工程）
    # ---- 旋律（`song_melody_step`）が書く ----
    sung: list = field(default_factory=list)               # 歌のある区間（設計の区間のうち `has_vocal`）
    top_c: int = 0                                         # 旋律が書く空間（Cメジャー・移調前）での最高音
    # ---- 詞（`song_lyrics`）が書く ----
    by_index: dict = field(default_factory=dict)           # フレーズ番号 → 歌う詞（譜割りつき）
    # ---- 歌の組み立てが書く（時間軸・調・コードは `song_place.build_timeline`、音符は `place_vocal`／`place_lead`）----
    ts: str = "4/4"                                        # 拍子
    bar: int = 0                                           # 1小節の tick
    timeline: Any = None                                   # 区間の位置（`SectionTimeline`・区間の位置の正）
    shift_of: Any = None                                   # 区間 → 移調量（転調を含む）
    notes: list = field(default_factory=list)              # 歌の音符（`SongNote`）
    spans: list = field(default_factory=list)              # コード（`ChordSpan`）
    bars_of: dict = field(default_factory=dict)            # 区間 → (開始小節, 終了小節)（時間軸から導出）
    sec_of_note: list = field(default_factory=list)        # 音符 i → 区間名（時間軸から導出）
    key_of: dict = field(default_factory=dict)             # 区間 → キー
    #: 主旋律を詞なしで置いた音（`(区間名, LeadNote)` の並び）。`song_place.place_vocal`／`place_lead` が書き、
    #: `song_place.make_score` が編成の `lead`・`lead_sections` で主旋律の楽器のパートにする（2026-10-03）
    lead_notes: list = field(default_factory=list)
    # ---- 本体が書く（歌の組み立てのあと）----
    brief_text: str = ""                                   # 申し送り（歌い方より後の工程が読む）
    # ---- 演奏（`song_band`）が書く ----
    title: str = ""                                        # 曲名（工程『曲名』の値・無ければ仮の曲名・`Song.step_title`・2026-10-03）
    genre_name: str = ""                                   # 曲の種類の呼び名
    inst_gms: list = field(default_factory=list)           # 編成の楽器 (bank, 番号)
    inst_desc: str = ""                                    # 編成の説明（伴奏・彩りへ渡す）
    allow_drums: bool = True
    groove_note: str = ""                                  # グルーヴの土台（伴奏・奏法へ渡す）
    vocal_range: dict = field(default_factory=dict)        # 区間 → 歌の最低音・最高音（移調前）
    arrange_rec: Any = None                                # 伴奏の記録（関門が読む）
    intensity_of: dict = field(default_factory=dict)       # 区間 → 強さ
    awarn: list = field(default_factory=list)              # 伴奏の組み立ての所見
    spec: Any = None                                       # 伴奏の組み立て（`ArrangementSpec`）
    # ---- 楽譜（`song_place.make_score`）が書く ----
    score: Any = None
