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

"""**任意の工程から作り直す**（人の指示で・2026-09-15 新設）。

## なぜ要るか

本人:「**全ての工程を後から編集できるようにして。要は、人間と同じように任意の工程から
修正しきるように**」

それまで後から直せたのは **編成・伴奏・ハモリ**（`rearrange_song`）と**尺**（`resize_song`）、
あとはピンポイント（移調・音量・区間の音符）だけ。テーマ・設計・コード・グルーヴ・旋律・
詞は**入口が無かった**（`song_edit.run_edit` は「未実装スキップ」のうえ、
**どこからも呼ばれていなかった**）。

## 仕組み——**作成中の差し戻しと同じ機械**を人が起動する

曲づくりの途中で関門が「詞から作り直して」と判断したときに走る流れ（`compose_song` の
`RedoFrom`）が、既にこの形になっている:

  ①その工程以降の保存を捨てる（`STEP_ORDER` / `_STEP_KEY`）
  ②残った前段を **$0 で復元**する（`song_replay.resume`）
  ③**やり直す理由**を申し送り（`CARRY`）に載せて、その工程から先を作り直す
  ④後工程は**必ず順に走る**ので置いていかれない

違うのは**起動するのが関門か人間か**だけ。ここはそこへ「人の指示」を流し込む。

★**`rearrange_song` と同じ方針**——「これまでの作成経緯」と「マスターの指示」を両方渡す。
★**元の版は消さない**（新しい版として保存する）。**作り方の記録も無傷**——作業場を
  `_wip/_remaking/` に隔離するので、途中で落ちても元から作り直しをやり直せる
  （隔離しないと、途中経過の自動保存が**同じ曲名なので元の記録を上書きする**）。
★編成・伴奏・ハモリだけで足りる依頼は、**安い `rearrange_song` のほうを使う**
（こちらは前段の復元と後工程の作り直しを伴うため）。

★**曲の形を変える**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §4）のも同じ機械（`rebuild_form`・道具は `change_form`）。
  違うのは起点で、工程名ではなく**形の差と新しい形の形式の検査から機械が決める**（`song_gate.form_change`）。
  その曲の形に無い工程（インストの『詞』など）は `rebuild` が作り直さずに断る（`absent_step_problem`）。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import song_compose as _sc                                 # 1周ぶん作る（`_compose_once`）
import song_context as _ctx                                # 曲づくりの共有の状態（頼まれごと・前回の出力・拍子）
import song_flow as _flow                                  # 捨てる工程・前回の出力・途中保存の置き場
import song_replay as _replay
from song_gate import FORM_STEPS, STEP_ORDER
from song_model import Song, made

#: ★**工程名は `STEP_ORDER` の表記そのままだけを受け付ける**（2026-10-02・判断6・本人「おすすめで良い」）。
#:   それまでは別名の表 `_ALIAS`（構成→設計・リズム→グルーヴ・しゃくり→歌の崩し・リフ→彩り・合いの手→声部・英語のキー名 ほか）で
#:   読み替えていた＝同じ工程に書き方が何通りもあった。マスターの言葉（「リズムを変えて」）をどの工程に当てるかは
#:   コーディネーターが【工程の一覧】（`STEP_ROLES`）を見て決める。違う名前は `step_problem` が選べる工程名を添えて返す。
#: テーマは `STEP_ORDER` に無いが、断る理由を「知らない工程」でなく正確に伝えるため名前としては受ける（`rebuild` が弾く）。
STEP_NAMES: tuple[str, ...] = ("テーマ", *STEP_ORDER)

#: ★**工程ごとに「何を決めるか」と「どんな頼みならここか」**（2026-09-26・本人指示）。
#:   コーディネーターが自分の工程を知らず、「声が高い。落ち着いた感じで歌っていい」を**移調だけ**で応えた
#:   （声の種類は編成・歌い回しは歌い方が決めるのに、指示書に書いていなかった。作り直しの工程の一覧も
#:   編成／伴奏／ハモリで止まっていた）。**ここが正**で、`describe_roles()` がコーディネーターの指示へ差し込む。
#:   工程を足したらここにも足す——`tests/song_service/test_step_roles.py` が漏れを落とす。
STEP_ROLES: dict[str, tuple[str, str]] = {
    "設計": ("構成（区間の並びと小節数）・テンポ・キー・歌う高さの目安・歌い方の型・使う工夫・終わり方",
             "『構成を変えて』『テンポを変えて』『もっと短く』『終わり方を変えて』"),
    "コード": ("区間ごとのコード進行と転調", "『コード進行を明るく』『サビで転調して』"),
    "グルーヴ": ("ドラムとベースの土台・区間ごとの行数と1行の音数・息の区切り",
               "『リズムを跳ねる感じに』『ノリを変えて』"),
    "旋律": ("歌のメロディ（音の高さ・長さ・リズム・伸ばし）", "『メロディを変えて』『サビのメロディをもっと高揚させて』"),
    "詞の設計":("決め台詞（hook）・その置き場・誰が誰に歌うか・区間ごとに言うこと",
               "『決め台詞を変えて』『歌詞のテーマを変えて』"),
    "サビの詞": ("サビの歌詞（詞の設計の決め台詞を推しメロに乗せて、先に書く）", "『サビの歌詞を書き直して』"),
    "詞": ("サビ以外の歌詞（確定したサビへ向かう形で）", "『Aメロの歌詞を直して』"),
    # ★曲名（2026-10-03）: 詞のあとに最後の曲名を決める（テーマは仮の曲名）。どの工程も曲名を読まないので、ここから作り直しても
    #   ほかは作り直さない（楽譜の曲名は組み立てが写し直す）。曲名を変える専用の道具は作らない
    #   ★作曲担当（コーディネーター）に見せる文には仮の曲名の仕組みを書かない（2026-10-03・本人了承。内部の事情）
    "曲名": ("曲の名前（詞の中身を読んで決める）",
             "『曲名を変えて』『もっと短いタイトルに』『サビの言葉をタイトルに』"),
    "歌い方": ("フレーズごとの強さ（ささやく・押し殺して・ふつう・しっかり・張り上げて）と"
               "歌い回し（やさしく・伸ばしきる・切って）と入りの表情",
             "『落ち着いた感じで歌って』『サビはもっと張って』『やさしく歌って』"),
    "歌の崩し": ("1音ずつのしゃくり・落とし・ベンド", "『もっとこぶしを効かせて』『しゃくりを減らして』"),
    "編成": ("使う楽器とドラムの有無・ハモリを重ねるか・**歌う声（original＝芯のある標準／soft＝やわらかく息の多い声）と"
             "声の種類（MIKU Default・MIKU Dark・MIKU Breath・MIKU Whisper・MIKU Cute など）**",
             "『和楽器風に』『バンドっぽく』『落ち着いた声で』『声をやわらかく』"),
    "伴奏": ("楽器ごとの弾くパターン（伴奏の骨組み）", "『伴奏を静かに』『サビをもっと厚く』"),
    "ハモリ": ("どこにどんな濃さでハモリを重ねるか", "『サビだけハモって』『ハモリを控えめに』"),
    "声部": ("合いの手・追っかけ（重ねる声の実際の音）", "『合いの手を入れて』"),
    "彩り": ("対旋律・リフ（歌のすき間に入る楽器の飾り）", "『ギターのリフを足して』『対旋律を変えて』"),
    "奏法": ("楽器の音ごとのしゃくり・フォール・ベンド", "『ギターを泣かせて』『サックスらしく吹いて』"),
}


def describe_roles() -> str:
    """コーディネーターの指示へ差し込む「工程の一覧」（何を決めるか・どんな頼みならここか・一緒にやり直す工程）。"""
    lines = ["【工程の一覧——自分がどんな工程で曲を作っているか】（作り直しは remake_from_step(工程名, 頼み)）"]
    for s in STEP_ORDER:
        what, asks = STEP_ROLES.get(s, ("", ""))
        after = redone_after(s)
        lines.append(f"・**{s}**: {what}。例 {asks}"
                     + (f"\n    （作り直すと {'・'.join(after)} も一緒にやり直す）" if after else ""))
    lines.append(_form_note())
    return "\n".join(lines)


def _form_note() -> str:
    """曲の形ごとに無い工程（`FORM_STEPS` から引く・2026-10-03）。"""
    from song_files import FORM_LABEL, FORMS
    rows = [f"{FORM_LABEL[f]}（{f}）＝{'・'.join(_absent(f))}" for f in FORMS if _absent(f)]
    return ("※**曲の形で無い工程**: " + "／".join(rows)
            + "。その曲に無い工程は remake_from_step で選べない（歌や伴奏を足す・抜くなら change_form）。")


def _absent(form: str) -> list[str]:
    """その形の曲に無い工程（作る順）。"""
    return [s for s in STEP_ORDER if s not in FORM_STEPS[form]]


def absent_step_problem(step: str, form: str) -> str | None:
    """**その形の曲に無い工程**を頼まれたときの断りの文（選べる工程の一覧つき）。ある工程なら None。

    `rebuild`（`remake_from_step`）・`show_after` が作り始めずに返す（§3）。形そのものを変えるのは `change_form`（`rebuild_form`）。
    """
    if step not in _absent(form):
        return None
    from song_files import FORM_LABEL
    how = {"instrumental": "歌を乗せるなら change_form('song', 頼み)（声だけにするなら 'acappella'）",
           "acappella": "伴奏を付けるなら change_form('song', 頼み)（歌を抜いて演奏だけにするなら 'instrumental'）",
           }.get(form, "形を変えるなら change_form(形, 頼み)")
    ok = [s for s in STEP_ORDER if s in FORM_STEPS[form]]
    return (f"この曲は{FORM_LABEL[form]}なので『{step}』の工程は無い（作り直さなかった）。{how}。"
            f"この曲で選べる工程: {' / '.join(ok)}")


#: テーマからやり直すと**別の曲**になるので、起点にしない（`rebuild` が弾く）。
#: ★`STEP_ORDER` には**元から入っていない**（関門も テーマ へは差し戻せない）。
#: ここに挙げるのは、**断る理由を「知らない工程」でなく正確に伝えるため**。
_NOT_A_START = {"テーマ"}


def resolve_step(name: str) -> str:
    """工程名（`STEP_NAMES` の表記そのまま・前後の空白だけは無視）を返す。違う書き方なら空文字（**読み替えない**）。"""
    k = (name or "").strip()
    return k if k in STEP_NAMES else ""


def step_problem(name: str, form: str = "song") -> str | None:
    """工程名の所見（選べる工程名を添える）。正式な名前で、曲の形 `form` にある工程なら None。`remake_from_step` が返す文。"""
    if (k := resolve_step(name)):
        return absent_step_problem(k, form)
    return (f"『{name}』は工程の名前ではない。step には次の名前を**この綴りのまま**書く: {' / '.join(STEP_ORDER)}"
            "（マスターの言葉がどの工程に当たるかは【工程の一覧】で決める。英語のキー名・言い換えは読まない）")


def rebuild(song: Song, step: str, request: str, *, songs_dir: Path,
            verbose: bool = True, op: str = "set", scope: str = "",
            supersedes: list[str] | tuple[str, ...] = ()) -> Song:
    """**`step` から先を、`request`（人の指示）を読んで作り直す。**

    `song` はそのまま（**元の版は消さない**）。戻り値が作り直した `Song`。

    Args:
        song: 作り直す元の曲（`Song.load` で読んだもの）
        step: どの工程から（`resolve_step` を通した `STEP_ORDER` の表記）
        request: マスターの指示（そのまま申し送りへ載る）
        songs_dir: `data/songs`（工程の保存を書く場所）

    Raises:
        ValueError: 工程名が不正、テーマを起点に指定したとき、またはその曲の形に無い工程（§3）のとき
    """
    # ★**テーマを先に弾く**——`STEP_ORDER` に元から入っていないので、後ろで見ると
    #   「知らない工程」という**理由の違う断り方**になる。断る理由は正確に伝える。
    if step in _NOT_A_START:
        raise ValueError(f"{step}からは作り直さない（別の曲になる）。新しく作るほうが早い")
    if step not in STEP_ORDER:
        raise ValueError(f"知らない工程: {step!r}（{' / '.join(STEP_ORDER)} から選ぶ）")
    # ★**その曲の形に無い工程は作り直さない**（2026-10-03・§3）。インストの『詞』・アカペラの『伴奏』など。形を変えるのは rebuild_form
    if (why := absent_step_problem(step, song.form)):
        raise ValueError(why)

    # ---- ①この工程から先の保存を捨てる ----
    data: dict[str, Any] = song.to_json()
    drop = _flow.drop_keys(step)                           # 依存の木を辿る（列の後ろ全部ではない）
    # ★**作り直す工程それぞれに、前回の出力を渡す**（2026-09-26・本人指示）。保存からは捨てる（＝LLM を呼ぶ）が、
    #   前回の稿を土台として見せ、**変える必要がある所だけ変えてもらう**（`song_context.asks_for` が依頼文へ足す）。
    #   見せていなかったころは「サビの歌詞を」で A・B メロまで一から書き直していた。
    _prev_out = _flow.prev_outputs_of(data.get("steps") or {}, step)
    kept = {k: v for k, v in (data.get("steps") or {}).items()
            if k not in drop and made(v)}                  # 走らなかった工程の空の出力は作り始めに入れ直す（F1）
    data["steps"] = kept
    if verbose:
        print(f"  [作り直し] **{step}から**／残す工程: {sorted(kept)}\n"
              f"      指示: {request[:150]}")

    # ---- ②指示を**台帳**に書いて、その工程と下流を作り直す ----
    #   ★以前は `CARRY`（宛先なしの申し送り）に入れていたので、**全工程に配られ、作り直しのあと消えた**
    #     ——次の作り直しでは誰も覚えていなかった。台帳は**曲の記録に残り**、頼まれた工程と
    #     その下流にだけ届く（`song_requests.deliver`・2026-09-25 段階2）。
    #   ★置き換える古い頼みは**コーディネーターが `supersedes` で指す**（機械は勝手に消さない）。
    from song_requests import add as _add_request
    _key = _ctx._STEP_KEY.get(step, "")
    _prev = song.reason_of({"arrangement": "arrange"}.get(_key, _key)) if _key else ""
    _add_request(song.requests, request, step, op=op, scope=scope, origin="マスター",
                 version=song.version, prev=_prev, supersedes=supersedes)
    # ★**巻き込まれた下流は「前回のまま」を選べる**（2026-09-26・本人案）。頼まれた当の工程は選べない（変えるために呼ぶ）
    # ★**前の版で最終の点検役の NG が残って止まっていたら、その NG を作り直す工程へ申し送る**（2026-10-04 組F1・4‑1）。以前は申し送りが
    #   空になり、NG の中身はコーディネーターが頼みに書き写した分しか工程へ届かなかった
    ng = final_ng_note(song, step)
    return _run_rebuild(song, data, prev_out=_prev_out, keep_allowed=set(_prev_out) - {step},
                        songs_dir=songs_dir, verbose=verbose, form=song.form, carry=[ng] if ng else [])


def final_ng_note(song: Song, step: str) -> str:
    """**前の版で残った最終の点検役の NG**の申し送り（`carry_note` の1行・無ければ空）。曲の記録の最後の最終の点検役の判断が
    通していない（＝最終の点検役の NG で止まった曲）ときだけ。宛先は続きを作る工程 `step`（そこと、それを読む工程へ届く）。
    最終の点検役の申し送りとして扱うので、差し戻しの周回（`song_compose._compose_song_rounds`）が直し終えるまで残す。"""
    from song_final_gate import FINAL, FINAL_RECORD        # noqa: E402（循環を避けてここで）
    from song_gate_brief import carry_note                 # noqa: E402
    recs = [r for r in (song.history or []) if r.step == FINAL_RECORD]
    if not recs or recs[-1].request == "accept":
        return ""
    r = recs[-1]
    body = (f"（前の版で、最終の点検役が差し戻しを使い切っても残った NG。その続きを作る）NG: {' '.join((r.reason or '').split())}"
            + (f"／どう直すか: {' '.join(r.guidance.split())}" if (r.guidance or "").strip() else ""))
    return carry_note(step, FINAL, body)


def _run_rebuild(song: Song, data: dict[str, Any], *, prev_out: dict, keep_allowed: set[str],
                 songs_dir: Path, verbose: bool, form: str, form_from: str = "",
                 keep_check=None, form_request: str = "", carry: list[str] | tuple[str, ...] = ()) -> Song:
    """**作り直しの機械**（`rebuild`・`rebuild_form` の共通）。`data`（残す工程だけを `steps` に入れた記録）を作業場に書き、
    保存から $0 で戻しつつ、無い工程だけ本物を呼んで1周作る。台帳（`song.requests`）は呼び手が書いておく。

    `form` はこの周の曲の形（`song_context.FORM`）、`form_from` は形を変えるときの前の形、
    `keep_check` は「前回のまま」にした出力に当てる検査（`song_context.KEEP_CHECK`・形を変えるときだけ）、
    `form_request` は形を変える頼み（【曲の形】の見出しに載る・`song_context.FORM_REQUEST`）。
    `carry` は作り直すあいだだけ載せる申し送り（前の版で残った最終の点検役の NG・`final_ng_note`）。終わったら外す。
    """
    # ---- 前段を $0 で復元する土台を書く ----
    #   ★**作業場は `_wip/_remaking/` に隔離する**。`_wip` 直下に置くと、途中経過の
    #     自動保存（`_autosave`）が**同じ曲名なので元の記録を上書きする**——
    #     作り直しが途中で落ちると、**元の記録が消えたまま残り、やり直すこともできない**。
    #     （2026-09-15 の実地検証で実際に『この衝動』の記録を潰した。）
    work = songs_dir / "_wip" / "_remaking"
    work.mkdir(parents=True, exist_ok=True)
    src = work / f"{_flow.autosave_name(song.title())}.steps.json"   # 曲名の slug（2026-10-03・`?`・`:`・`/` で書けなかった）
    src.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")

    _pending = [r for r in _ctx.PENDING_REQUESTS               # 同じ回に読み取った鼻歌などは残す
                if all(r.id != q.id for q in song.requests)]
    _ctx.PENDING_REQUESTS[:] = list(song.requests) + _pending
    _ctx.PREV_OUTPUTS.clear()
    _ctx.PREV_OUTPUTS.update(prev_out)
    _ctx.KEEP_ALLOWED.clear()
    _ctx.KEEP_ALLOWED.update(keep_allowed)
    _ctx.KEPT.clear()
    _ctx.FORM, _ctx.FORM_FROM, _ctx.KEEP_CHECK = form, form_from, keep_check   # 曲の形（2026-10-03）
    _ctx.FORM_REQUEST = form_request
    _replay.resume(src)
    _saved_dir = _flow.AUTOSAVE_DIR
    _flow.AUTOSAVE_DIR = work                              # 途中経過も隔離先へ
    from song_gate import reset_progress
    reset_progress()                                       # 無進展の記録も差し戻しの打ち止めも1曲ごと（`compose_song` と同じ）
    _flow.NO_MORE_REDO = False
    try:
        _ctx.CARRY.extend(carry)                           # 外すのは下の finally（`CARRY.clear()`）
        # `instruction` は元の依頼文（`_state.instruction`）。保存に無ければ元の曲から。
        # ★**差し戻しの周回で作る**（2026-10-03・再レビュー）。以前は `_compose_once` を直に呼んでいたので、作り直しの途中で
        #   点検役が前の工程への差し戻し（`RedoFrom`）を選ぶと、例外のまま道具まで抜けて費用ごと捨てられた。
        #   周回は新しい曲と同じ（`song_compose._compose_song_rounds`）: 差し戻し先から先を捨てて作り直し（作り直しの範囲の外へ
        #   戻るなら範囲が広がる）、回数の上限（`MAX_REDO_SONG`）で打ち止め。差し戻しの保存も作業場（`_remaking/`）に書く
        new = _sc._compose_song_rounds(_replay.instruction() or song.instruction, verbose=verbose, stop_after="")
        # 「前回のまま」の記録は差し戻しの周回が書く（2026-10-04 組P2b に `_compose_song_rounds` へ移した——最終の点検役の差し戻しでも選べる）
    finally:
        _flow.AUTOSAVE_DIR = _saved_dir
        _replay.uninstall()
        _replay._state = None
        _ctx.CARRY.clear()
        _ctx.PENDING_REQUESTS.clear()                      # 台帳は新しい曲（new.requests）に載っている
        _ctx.ACTIVE_REQUESTS = []
        _ctx.PREV_OUTPUTS.clear()
        _ctx.FINAL_PREV.clear()                            # 最終の点検役が差し戻したときの稿も（2026-10-04 組P2b）
        _ctx.FINAL_REDONE.clear()                          # その差し戻しの範囲も（2026-10-04 組F1）
        _ctx.CURRENT_SONG = None
        _ctx.KEEP_ALLOWED.clear()
        _ctx.KEPT.clear()
        _ctx.METER = "4/4"                                 # 拍子も1曲ごと（2026-09-27）
        _ctx.FORM, _ctx.FORM_FROM, _ctx.KEEP_CHECK = "song", "", None   # 曲の形も1曲ごと
        _ctx.FORM_REQUEST = ""
        _flow.NO_MORE_REDO = False                         # 打ち止めを次の曲へ持ち越さない
    # ★**成功したときだけ片づける**。落ちたときは `_remaking/` に途中まで残し、
    #   何が起きたか追えるようにする（元の記録は無傷のまま）。
    shutil.rmtree(work, ignore_errors=True)

    new.parent = song.version                              # どの版から派生したか
    new.origin = "edit"
    return new


#: 形を変える頼みの台帳の宛先（2026-10-03・§4-5）。工程ではないので、台帳から工程へは届かない（`song_requests.reaches`）——
#: 記録と「どう作ったか」（`recall_making`）のための行。今回の作り直しへは【曲の形】の見出しで届ける
FORM_TARGET = "形"


def rebuild_form(song: Song, form: str, request: str, *, songs_dir: Path, verbose: bool = True) -> Song:
    """**曲の形を `form` へ変えて作り直す**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §4・F4②・F9①・F11①）。

    `rebuild` と同じ機械（保存から前段を $0 で戻す・作業場を `_wip/_remaking/` に隔離・新しい版・元の版は消えない）。
    **どの工程を作り直すかは機械が決める**（`song_gate.form_change`）——足す工程・外す工程・残す工程に新しい形の形式の検査
    （`song_form_checks.form_findings`）を当てて所見の出た工程と、それらを読む工程を作り直す。残す工程（`keep`）は保存から戻す。
    作り直す工程のうち所見の無いものは「前回のまま」を選べるが、選んだ出力にも同じ検査を当て、所見が出たら取り消して作り直させる（F11①）。

    `song` はそのまま（元の版は消さない）。戻り値が新しい形の `Song`（`form` は新しい形・`origin=edit`・`parent=元の版`）。
    記録に `StepRecord(step="form", request=新しい形, reason=…, warnings=届け先の無い頼み)` を積み、台帳に宛先「形」の行を足す。

    Args:
        song: 元の曲（工程の記録を持つ・`Song.load_record` で読んだもの）
        form: 新しい形（`song_files.as_form` で読む・空なら song）
        request: マスターの頼み（台帳に残り、この作り直しの全工程へ【曲の形】の見出しと一緒に届く）
        songs_dir: `data/songs`

    Raises:
        ValueError: 読めない形・いまと同じ形・工程の記録が無い曲（カバー・古い曲）・途中で止まった曲（先に続きを作る）
    """
    from song_files import FORM_LABEL, as_form
    from song_gate import form_change
    from song_model import StepRecord
    from song_requests import add as _add_request
    new_form, old_form = as_form(form), song.form
    if new_form == old_form:
        raise ValueError(f"この曲はもう{FORM_LABEL[new_form]}なので、形は変えなかった")
    steps: dict[str, Any] = dict(song.to_json().get("steps") or {})
    if not made(steps.get("theme")):
        raise ValueError("この曲には工程の記録が無い（カバー・記録の無い古い曲）ので、形を変えて作り直せない")
    # 途中で止まった曲（楽譜がまだ無い）。声部・彩りなど「作るものが無かった」工程は出力が None でも完成しているので、楽譜で見る
    if song.score is None:
        raise ValueError("この曲は途中で止まっている（楽譜がまだ無い）。先に remake_from_step で続きを作ってから形を変える")

    # ---- ①足す・外す・作り直す・残す工程を機械が決める（新しい形の形式の検査を残す工程に当てる）----
    from song_form_checks import form_findings              # 担当B: 残す工程の出力に新しい形の形式の検査を当てた所見
    fc = form_change(old_form, new_form, check=lambda s: form_findings(song, s, new_form))
    key = _ctx.step_output_key
    data: dict[str, Any] = song.to_json()
    data["form"] = new_form
    # 走らなかった工程の空の出力（`skipped`・F1）は残しも前回の出力にもしない（新しい形の作り始めに入れ直す）
    data["steps"] = {key(s): steps[key(s)] for s in fc.keep if made(steps.get(key(s)))}
    prev_out = {s: steps[key(s)] for s in fc.remake if made(steps.get(key(s)))}
    # 所見の出た工程は前回のままにできない（同じ所見でまた止まるだけ）。足す工程には前回の出力が無い
    keep_allowed = set(prev_out) - set(fc.findings)
    if verbose:
        print(f"  [形の変更] {FORM_LABEL[old_form]}→{FORM_LABEL[new_form]}／足す: {list(fc.add)}／外す: {list(fc.remove)}"
              f"／作り直す: {list(fc.remake)}／残す: {list(fc.keep)}\n      頼み: {request[:150]}")

    # ---- ②台帳に宛先「形」の行を残す（外した工程に宛てた頼みは消さず、届け先が無いことを記録に出す）----
    words = (request or "").strip() or f"{FORM_LABEL[new_form]}にする"
    _add_request(song.requests, words, FORM_TARGET, origin="マスター", version=song.version,
                 prev=f"{FORM_LABEL[old_form]}（{old_form}）")
    stranded = [f"[{r.id}] {r.target}" + (f"（{r.scope}）" if r.scope else "") + f"「{r.words[:60]}」"
                for r in song.requests if r.status in ("active", "reapply") and r.target in fc.remove]

    def _keep_check(step: str, out) -> list[str]:
        """前回のままにした出力を、いま作っている曲の写しに入れて新しい形の検査を当てる（上流は新しい形で作り直し済み）。"""
        import copy
        cur = _ctx.CURRENT_SONG
        if cur is None:
            return []
        trial = copy.copy(cur)
        setattr(trial, key(step), out)
        return form_findings(trial, step, new_form)

    from song_usage import USAGE
    spent_from = len(USAGE)
    # ★頼みは今回の作り直しの全工程へ【曲の形】の見出しで届ける（台帳の「形」の行は工程へは届かない）
    new = _run_rebuild(song, data, prev_out=prev_out, keep_allowed=keep_allowed, songs_dir=songs_dir,
                       verbose=verbose, form=new_form, form_from=old_form, keep_check=_keep_check,
                       form_request=(request or "").strip())
    spent = sum(float(u.get("usd", 0.0) or 0.0) for u in USAGE[spent_from:])

    # ---- ③記録に残す（§4-5）----
    kept_llm = [r.step for r in new.history if r.request == "keep"]
    why = (f"{FORM_LABEL[old_form]}→{FORM_LABEL[new_form]}。"
           f"足した工程: {'・'.join(fc.add) or 'なし'}／外した工程: {'・'.join(fc.remove) or 'なし'}／"
           f"作り直した工程: {'・'.join(fc.remake) or 'なし'}／前の版のまま残した工程: {'・'.join(fc.keep) or 'なし'}"
           + (f"／作り直しで「前回のまま」を選んだ工程: {'・'.join(kept_llm)}" if kept_llm else "")
           + "".join(f"／新しい形の形式の所見で作り直した『{s}』: {' / '.join(f[:80] for f in fs[:3])}"
                     for s, fs in fc.findings.items())
           + f"。この変更で使った費用 ${spent:.3f}（内訳は各工程の記録）")
    new.record(StepRecord(step="form", request=new_form, reason=why,
                          warnings=[f"届け先の無い頼み（外した工程宛て・台帳には残す。形を戻せばまた届く）: {s}" for s in stranded]))
    return new


def redone_after(step: str, form: str = "song") -> list[str]:
    """`step` を作り直すと**一緒にやり直す工程**（作る順・自分は含まない）。依存の木を辿り、曲の形 `form` で走る工程だけ。"""
    from song_gate import affected
    hit = affected(step) - {step}
    return [s for s in STEP_ORDER if s in hit and s in FORM_STEPS[form]]


def describe_steps(form: str = "song") -> str:
    """コーディネーターへ見せる「どの工程から直せるか」の一覧（曲の形 `form` で走る工程だけ）。

    ★**依存の木を辿った範囲を見せる**（2026-09-25）。それまでは列の後ろ全部を見せていたので、
      コーディネーターは「歌い方を直すと編曲も全部やり直し」と読み、費用を見て作り直しを避けていた。
    """
    lines = []
    for s in STEP_ORDER:                                   # テーマは STEP_ORDER に無い
        if s not in FORM_STEPS[form]:
            continue
        after = redone_after(s, form)
        lines.append(f"  {s}: ここから作り直すと "
                     f"**{' → '.join(after) or '（ほかは作り直さない）'}** も一緒にやり直す")
    return "\n".join(lines)
