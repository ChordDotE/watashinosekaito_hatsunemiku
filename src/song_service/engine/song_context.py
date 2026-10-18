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

"""**工程のあいだで共有するもの**——LLM の呼び出し口・前の工程の渡し方・申し送り・いま作っている曲の状態。

2026-09-30 に `song_compose`（約3,900行）から分けた（本人「並列で作業できるように、工程ごとのファイルに分けて」）。

## 呼び出し口（再生の差し替え口）

曲づくりの工程は、LLM をここの名前で呼ぶ——`_call`（テーマ・設計・コード・グルーヴ・旋律・詞）、`judge`（点検役）、
`plan_instruments`（編成）・`arrange`（伴奏）・`plan_harmony`（ハモリ）。**再生（`song_replay.install`）はこのモジュールの名前を
差し替える**（保存済みの出力を返す）。工程のモジュールが `from song_context import _call` のように取り込むと差し替えが効かないので、
**工程は必ず `ctx._call(...)`・`ctx.judge(...)` の形（モジュールの属性）で呼ぶ**。

## いま作っている曲の状態

`CURRENT_SONG`・`ACTIVE_REQUESTS`・`METER`・`FORM`（曲の形）などはここのモジュール変数が正。1曲ごとに `compose_song`（`song_compose`）と
作り直し（`song_remake.rebuild`）の `finally` が戻す。読むときも**モジュールの属性で読む**（`from song_context import METER` は
取り込んだ時点の値に固まる）。`song_llm.run_sync` もここ（`sys.modules["song_context"]`）から頼みごと・前回の出力を引く。
"""
from __future__ import annotations

import json
from contextlib import contextmanager

from song_gate import DEPS, judge                         # noqa: F401（judge は点検役の呼び出し口・再生が差し替える）
from song_usage import record as record_usage             # 費用は1箇所で数える
from song_model import made                               # 工程の出力があるか（走らなかった工程の空の出力は「無い」・F1）
from harmony import plan_harmony                         # noqa: F401（ハモリの呼び出し口・再生が差し替える）
from instrumentation import plan_instruments             # noqa: F401（編成の呼び出し口・再生が差し替える）
from song_arrange import arrange                         # noqa: F401（伴奏の呼び出し口・再生が差し替える）


# ★**やり直しの判定と待ちは `song_llm` が1箇所で持つ**（2026-09-11 集約）。
#   ここにだけ持っていたせいで、**選抜・関門・編成・伴奏・ハモリ**（`_call` を通らず
#   直接 `run_sync` する経路）に効かず、**503 で曲が丸ごと落ちた**。
from song_llm import RETRY_WAITS as _RETRY_WAITS   # noqa: E402,F401（表示に使う）
from song_llm import run_sync as _run_sync         # noqa: E402


def _call(agent, prompt: str, label: str):
    """agent を呼びつつ**トークンと費用を記録**し、**通信エラーだけやり直す**。

    ★費用の判定と記録は `song_usage.record` に任せる（2026-09-01）。
      ここには独自の `_tier_of` があり、**`str(agent.model)` を見ていた**——
      これが返すのは `'GoogleModel()'` でモデル名が入っておらず、
      **pro を呼んでいても常に flash 価格で計上していた**（過去の報告値が33%過少）。
      `song_usage` は集計を1箇所に寄せるために作ったのに、ここだけ乗り換えていなかった。

    ★通信エラーで工程が死ぬと、**そこまでの数分ぶんが消える**（実測: 詞の工程で
      `httpx.ReadError` に当たり、9分25秒ぶんを失った）。ここで数回粘る。
      それでも落ちたときのために、工程ごとの保存（`_autosave`）と再開（`replay.resume`）がある。
    """
    r = _run_sync(agent, prompt, label=label)
    record_usage(agent, r, label)
    return r.output


# **後の工程へ渡す申し送り**（差し戻しの理由・直せなかったことなど）。`brief` に載せて後段へ届ける。
# ※旋律を3案→選抜で作っていたときは、選抜の「他の案の惜しい点」もここに載せていた（2026-09-30 に1案にしたので無くなった）
CARRY: list[str] = []

#: **次の曲づくりへ持ち込む頼まれごと**（2026-09-25・段階2）。作り直し（`song_remake.rebuild`）と
#: 鼻歌（`tools/humming.py`）が積み、`song_compose._compose_once_body` が曲の台帳（`song.requests`）へ移す。
#: ★マスターの頼みは `CARRY` に入れない——`CARRY` は宛先なしで全工程へ配られ、作り直しのあと消えていた。
PENDING_REQUESTS: list = []
#: **いま作っている曲の台帳**。`song_llm.run_sync` がラベルから工程を引き、届く頼みを依頼文へ足す。
ACTIVE_REQUESTS: list = []
#: ★**作り直す工程の、前回の出力**（2026-09-26・本人指示）。工程名 → 作り直す前の版のその工程の出力（dict）。
#:   それまでは作り直す工程の保存を**捨てたうえで、前回の出力を LLM に見せていなかった**（渡していたのは理由の短い文だけ）。
#:   そのため「サビの歌詞を書き直して」で、下流の詞の工程が A・B メロまで一から書き直した（255字中212字が変わった）。
#:   人の作詞家・編曲者は前の稿に手を入れる。**前回の出力を土台として渡し、変える必要がある所だけ変えてもらう**。
#:   積むのは作り直し（`song_remake.rebuild`）と関門の差し戻し（`song_compose._compose_song_rounds`）。1曲ごとに空にする。
PREV_OUTPUTS: dict = {}
#: ★**最終の点検役が差し戻したときの稿**（2026-10-04・組P2b）。工程名 → その時点の出力（差し戻した工程と、それを読む工程）。
#:   最終の点検役の2周目が「前の稿から変わった所」を比べる相手（`song_gate_brief.final_rejudge_text`）。`PREV_OUTPUTS` は
#:   1段目の点検役の差し戻しでも入れ替わるので、別に持つ。積むのは差し戻しの周回（`song_compose._compose_song_rounds`）。1曲ごとに空にする
FINAL_PREV: dict = {}
#: ★**最終の点検役が差し戻して作り直させた工程**（差し戻した工程と、それを読む工程・2026-10-04 組F1）。`FINAL_PREV` は差し戻したときの
#:   **全工程の写し**になった（2周目に、1段目の差し戻しで変わった工程も比べるため）ので、どれが差し戻しの範囲かはここで持つ。1曲ごとに空にする
FINAL_REDONE: set = set()
#: 前回の出力を依頼文に載せる上限（字）。実測の最大は旋律の約1.7万字（2026-09-26）
PREV_OUTPUT_LIMIT = 40000
#: ★**「前回のまま」を選べる工程**（2026-09-26）。作り直しに**巻き込まれた下流**だけ（頼まれた当の工程は入れない）。
#:   `song_llm.run_sync` が最初の呼び出しで出力の形に `KeepPrevious` を足し、選ばれたら前回の出力へ差し替える。
KEEP_ALLOWED: set = set()
#: 「前回のまま」を選んだ工程と理由（作り直しの報告と記録に使う）
KEPT: list = []
#: ★**いま作っている曲**（2026-09-26）。`asks_for` が**前段の出力**を引くために使う
#:   （`song_compose._compose_once_body` とカバーの `arrange_song` が入れる）。
CURRENT_SONG = None

#: ★**いま作っている曲の拍子**（2026-09-27）。設計が決まったところで `song_outline.run_groove` が入れ、1曲ごとに戻す。
#:   旋律の枠（1小節の16分の数）・行の拍数・グルーヴへ渡す秒数は、ここから `grid_per_bar()` / `beats_per_bar()` で引く。
#:   それまで `GRID_PER_BAR = 16`（4/4 の決め打ち）で、3/4 の曲『風に呼ばれたら』は旋律が1行8拍・伴奏が1行6拍になり、
#:   歌が区間ごとに遅れてサビが後奏に乗った（歌の遅れ: scene2 で6小節・scene3 で12小節）。
METER = "4/4"

#: ★**いま作っている曲の形**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §2）。song／acappella／instrumental。
#:   入れるのは入口だけ——`song_compose.compose_song(form=…)`（コーディネーターの `draft_melody(form=…)`）と作り直し
#:   （`song_remake.rebuild`＝曲の記録の形・`rebuild_form`＝新しい形）。1曲ごとに `song` へ戻す。走る工程は `song_gate.FORM_STEPS[FORM]`。
#:   工程の出力の型には形の欄を持たせない（LLM が変えられない）。工程へは `form_heading()` が【曲の形】の見出しで知らせる
#:   （`song_llm.run_sync` がラベルの付いた呼び出し全部に足す）。読むときは**モジュールの属性で**（`ctx.FORM`）
FORM = "song"
#: **形を変えて作り直している途中なら、前の形**（`song_remake.rebuild_form` が入れる・それ以外は空）
FORM_FROM = ""
#: 形を変える**マスターの頼み**（`rebuild_form` が入れる）。台帳の宛先「形」の行は工程へは届かないので、【曲の形】の見出しに載せて
#: この作り直しの全工程へ届ける
FORM_REQUEST = ""
#: ★**「前回のまま」にした出力に当てる形式の検査**（2026-10-03・F11①）。(工程名, 前回の出力) → 所見の文の並び（空＝通った）。
#:   形を変えたときだけ `song_remake.rebuild_form` が入れる（新しい形の形式の検査＝`song_form_checks.form_findings`）。
#:   `song_llm.run_sync` が `KeepPrevious` の差し替えの後ろで呼び、所見が出たら前回のままを取り消して作り直させる
KEEP_CHECK = None


def form_heading() -> str:
    """**【曲の形】の見出し**（依頼文の末尾に足す。歌もので形を変えていなければ空——歌ものの依頼文は今までと同じ）。

    無い工程は `song_gate.FORM_STEPS` から引く（手書きの一覧を持たない）。
    """
    if FORM == "song" and not FORM_FROM:
        return ""
    from song_files import FORM_LABEL                       # noqa: E402（標準ライブラリだけの名前の決まり）
    from song_gate import FORM_STEPS                        # noqa: E402
    what = {
        "song": "ミクの歌と、楽器の伴奏で作る曲。",
        "acappella": ("伴奏の楽器もドラムも無く、ミクの声だけで作る曲。主旋律と、ハモリ・声部（重ねる声）で和声とリズムを作る"
                      "（グルーヴのドラム・ベースの行は鳴らさず、ノリの参考として読む）。"),
        "instrumental": ("歌の入らない、演奏だけの曲。旋律の工程が書く主旋律は、編成が選ぶ楽器（`lead`）で弾く"
                         "（歌詞・歌声は無い。息継ぎ・モーラの言葉は「フレーズの切れ目・音の数」と読む）。"),
    }[FORM]
    absent = [s for s in FORM_STEPS["song"] if s not in FORM_STEPS[FORM]]
    lines = [f"【曲の形】この曲は**{FORM_LABEL[FORM]}**（`{FORM}`）——{what}"]
    if absent:
        lines.append(f"  この曲には {'・'.join(absent)} の工程は無い（書いても使われない）。")
    if FORM_FROM:
        lines.append(f"  ★**形を変えて作り直している**: 前の版は{FORM_LABEL.get(FORM_FROM, FORM_FROM)}（`{FORM_FROM}`）。"
                     "前回の出力は前の形で書かれているので、新しい形に合わせて見直す（前の形のまま残してよい所は残す）。")
        if FORM_REQUEST:
            lines.append(f"  形を変えるときのマスターの頼み: 「{FORM_REQUEST}」")
    return "\n\n" + "\n".join(lines)


def keep_findings(step: str, out) -> list[str]:
    """「前回のまま」にした `step` の出力 `out` に、形を変えたときの形式の検査を当てた所見（検査が無ければ空）。

    いま作っている曲（`CURRENT_SONG`）の写しにその出力を入れて検査する（上流はもう新しい形で作り直してある）。
    """
    if KEEP_CHECK is None:
        return []
    return list(KEEP_CHECK(step, out) or [])

#: 曲の種類（`theme.genre`）→ 依頼文・点検役に見せる呼び名
_GENRE_NAME = {"pop": "ミドルのポップス", "children": "童謡・子供向け",
               "ballad": "ゆったりバラード", "anthem": "応援歌・行進曲（4分中心・揃える）"}

# 工程名（`STEP_ORDER`）→ 保存の中でのキー（`Song.to_json()["steps"]`）
_STEP_KEY = {"設計": "design", "コード": "chords", "グルーヴ": "groove", "旋律": "melody",
             "詞の設計": "lyric_plan", "サビの詞": "lyric_chorus", "詞": "lyrics", "曲名": "naming",
             "歌い方": "expression", "歌の崩し": "vocal_artic",
             "編成": "instrumentation", "伴奏": "arrangement", "ハモリ": "harmony",
             "声部": "voices", "彩り": "colour", "奏法": "articulation"}


def step_of_label(label: str) -> str:
    """LLM 呼び出しのラベル → 工程名（`旋律2`→旋律・`旋律(再)`→旋律・`サビの詞3`→サビの詞）。"""
    return (label or "").split("(")[0].rstrip("0123456789")


#: 前段の出力を依頼文に載せる上限（字・1工程あたり）。実測の最大は旋律の約1万字（2026-09-26『風の軌跡』）。
#: 超えたら切ったことを依頼文とログに書く（黙って切らない）
UPSTREAM_OUTPUT_LIMIT = 60000


def step_output_key(step: str) -> str:
    """工程名 → 曲（`Song`）の上でその工程の出力を持つ属性名（＝記録の `steps` のキー）。"""
    return "theme" if step == "テーマ" else _STEP_KEY.get(step, "")


def _gate_notes(song, key: str) -> str:
    """その工程の関門が最後に書いた `notes`（無ければ空）。関門の記録は `<出力の名前>_gate`（伴奏だけ `arrange_gate`）。"""
    name = {"arrangement": "arrange_gate"}.get(key, f"{key}_gate")
    for r in reversed(getattr(song, "history", []) or []):
        if r.step == name:
            return getattr(r, "notes", "") or ""
    return ""


#: ★**旋律は ABC だけを渡す**（2026-09-30・出-15 本人方針）。旋律は保存の形（`BareMelody`）に同じ中身を2通り持つ——
#:   `sketch` の【書かれたABC】（LLM が書いたもの・直しのあとの版）と、そこから起こした音符の JSON（`phrases`）。
#:   丸ごと渡すと1回の依頼あたり約2万字を約10工程へ毎回貼っていた（『孤独のまま』）。同じ音は ABC にあるので JSON は渡さない。
#:   ABC の無い古い記録（枠記法の頃の `sketch`）は、旋律がそこにしか無いので JSON のまま渡す。
#:   音の高さは ABC も JSON も **C 基準・移調前**（実音が要る工程——歌い方・歌の崩し・声部——は本文の表で実音を読む）。
#:   ABC と一緒に区間ごとの移調量を添える（`_transpose_note`）。
UPSTREAM_OMIT: dict[str, tuple[str, ...]] = {"melody": ("phrases",)}
_ABC_MARK = "【書かれたABC】"


def _transpose_note(song) -> str:
    """旋律の ABC を実音へ直す移調量（区間ごと。転調は書いた区間から後ろ全部）。設計が無ければ空。"""
    design = getattr(song, "design", None)
    if design is None or not getattr(design, "sections", None):
        return ""
    from song_design import key_shift                      # noqa: E402（循環を避けてここで）
    import song_checks as ck                                # noqa: E402
    names = [s.kind for s in design.sections]
    mod_at = {(ck.resolve_section(m.at_section, names) or m.at_section): m.key
              for m in design.modulations}
    cur, runs = design.key, []                              # [(キー, [区間…])]
    for s in design.sections:
        cur = mod_at.get(s.kind, cur)
        if runs and runs[-1][0] == cur:
            runs[-1][1].append(s.kind)
        else:
            runs.append((cur, [s.kind]))
    try:
        body = "／".join(f"{'・'.join(ks) if len(runs) > 1 else '全区間'}: {k}（{key_shift(k):+d}半音）" for k, ks in runs)
    except ValueError:                                      # 読めない調（設計の検査が所見にする）。添え書きは出さない
        return ""
    return f"※旋律の ABC は C 基準・移調前です。実際に鳴る高さ＝書かれた高さ＋移調量（{body}）"


def _upstream_data(key: str, out) -> tuple[dict, str]:
    """前段の出力を依頼文に載せる形 (データ, 添え書き)。`UPSTREAM_OMIT` の欄は、同じ中身が別の欄にあるときだけ省く。"""
    data = out.model_dump() if hasattr(out, "model_dump") else out
    omit = UPSTREAM_OMIT.get(key, ())
    if not omit or not isinstance(data, dict):
        return data, ""
    if key == "melody" and _ABC_MARK not in str(data.get("sketch") or ""):
        return data, ""                                     # ABC の無い古い記録: 旋律は JSON にしか無い
    data = {k: v for k, v in data.items() if k not in omit}
    note = f"※音符の JSON（{'・'.join(omit)}）は省いた——同じ旋律が sketch の{_ABC_MARK}にある（直しのあとの版）。"
    t = _transpose_note(CURRENT_SONG) if key == "melody" else ""
    return data, note + ("\n" + t if t else "")


def upstream_intents(step: str) -> str:
    """`step` が読んでいる前段（`song_gate.DEPS`）が**決めたことを全部**——出力をそのまま（理由・楽器ごとの役と理由・
    区間・音符まで）。無ければ空。`song_llm.run_sync` がラベルの付いた呼び出しすべてに足す。

    ★**仕組みで全部渡す**（2026-09-26・本人指示「オブジェクトの中に仕組みとして組み込んで。都度付け足すだけだと
      絶対に漏れが出る」）。それまでは工程ごとに手書きの要約（`brief()`・`instrumentation.describe()` など）と
      前段の `reason` だけを渡していて、**要約から漏れた決定は届かなかった**。実害:『風の軌跡』で編成が
      ピアノに「ポツポツとした単音の旋律を紡ぐ」役を与えたのに、`describe()` が楽器ごとの理由を載せず、
      伴奏はピアノに和音の刻みしか書かなかった（ピアノは全区間で伴奏全体より 6〜10dB 小さく、主役にならなかった）。
      同じ「前段の決定が届かない」指摘は何度も受けていた（08-20 意図・09-11 歌の音域・09-26 前回の出力／意図）。
      **依存の木（`DEPS`）に書いた前段は、記録にある出力が丸ごと届く**——工程を足しても木に書けば自動で渡る。
      手書きの要約はそのまま残す（読みやすい要点）。**細部と理由の正はこちら**。
      `tests/song_service/test_upstream_all.py` が、全工程・全前段・全項目が届くことを確かめる。
    """
    song = CURRENT_SONG
    if song is None or step not in DEPS:
        return ""
    blocks = []
    for dep in DEPS[step]:
        key = step_output_key(dep)
        out = getattr(song, key, None) if key else None
        if not made(out):                                  # まだ無い・この形で走らなかった（空の出力・F1）は渡さない
            continue
        data, omit_note = _upstream_data(key, out)          # 旋律は ABC だけ（音符の JSON は省く・出-15）
        body = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
        if len(body) > UPSTREAM_OUTPUT_LIMIT:
            print(f"  [前段の出力] {step} へ渡す {dep} が {len(body):,}字あり、{UPSTREAM_OUTPUT_LIMIT:,}字で切った")
            body = body[:UPSTREAM_OUTPUT_LIMIT] + f"…（以下略・全{len(body):,}字）"
        blocks.append(f"■{dep}（{key}）\n{body}" + (f"\n{omit_note}" if omit_note else ""))
        gate = _gate_notes(song, key)
        if gate:
            blocks.append(f"■{dep} の点検役の所見（壊れてはいないが弱いと書いたこと。この工程で補えるなら補う）\n{gate}")
        from song_fix import machine_changes                # noqa: E402（循環を避けてここで）
        changed = machine_changes(song, key)
        if changed:                                        # ★機械がどうしても直したこと（2026-09-27・本人「理由と最新の楽譜を LLM に」）
            blocks.append(f"■{dep} に機械がした変更（上の出力は、直した後の値です）\n"
                          + "\n".join(f"・{c}" for c in changed))
    if not blocks:
        return ""
    return ("【前の工程が決めたこと（この工程が読む工程の出力・全部）】\n" + "\n".join(blocks)
            + "\n※上の依頼文は要点。**細部と理由（なぜそうしたか・楽器ごとの役と理由など）はここが正**。"
              "前の工程の決定と意図に沿って作り、沿えない所は理由を reason に書く。")


def asks_for(step: str, *, patch: bool = False) -> str:
    """`step` へ届く**前段の意図**・**頼まれごと**・**前回の出力**の文面（無ければ空）。依頼文の末尾に足す。

    `patch` … 差分の直しの呼び出し（ラベル `<工程>(直し{n})`）。差分の型は「指摘への対応」（`response`）を持たないので、
    その頼み（`response_ask`）を足さない。
    """
    if not step:
        return ""
    parts = []
    t0 = upstream_intents(step)
    if t0:
        parts.append(t0)
    asked = ""
    if ACTIVE_REQUESTS:
        from song_requests import deliver                 # noqa: E402（循環を避けてここで）
        asked = deliver(ACTIVE_REQUESTS, step)
        if asked:
            parts.append(asked)
    prev = PREV_OUTPUTS.get(step)
    if prev is not None:
        # ★**前回の出力も前段と同じ省き方で見せる**（2026-10-01・仕様書 §17 ず-7）。旋律の作り直し・差し戻しでは、前回の旋律が
        #   音符の JSON（`BareNote.solfa`・`octave`＝内部の階名）のまま届いていた。前段の渡し方（`_upstream_data`）と同じく
        #   ABC（【書かれたABC】）だけを見せる——決まっている方針「旋律は ABC だけ渡す」。「前回のまま」の差し替え
        #   （`song_llm.run_sync`）は `PREV_OUTPUTS` そのものを読むので、見せ方を変えても影響しない
        prev, prev_note = _upstream_data(step_output_key(step), prev)
        body = json.dumps(prev, ensure_ascii=False, separators=(",", ":"), default=str)
        cut = len(body) > PREV_OUTPUT_LIMIT
        parts.append(
            "【この工程の前回の出力（作り直す前の版）】\n" + body[:PREV_OUTPUT_LIMIT] + ("…（以下略）" if cut else "")
            + (f"\n{prev_note}" if prev_note else "")
            + "\n※**これを土台にする。** 頼まれたこと・差し戻しの理由・上流で変わった所に合わせて"
              "**変える必要がある所だけ変え、ほかは前回のまま残す**（言葉も音も、変える理由の無い所は同じにする）。"
              "変えた所とその理由は reason に書く。"
            + ("\n※**この工程は作り直しに巻き込まれただけ**。上流の変化とここまでの経緯を読んで、"
               "**この工程は何も変える必要が無い**と判断したら、出力を書かずに **KeepPrevious**（理由だけ）を返してよい"
               "——前回の出力がそのまま使われる。上流の変化がこの工程に響くなら、変える所だけ変えた出力を返す。"
               if step in KEEP_ALLOWED else ""))
    # ★**差し戻しの指摘は宛先に1回だけ・巻き込まれた工程には別の節で**（2026-10-04 組F1・1‑5）。以前は申し送り（`brief`）で
    #   全工程へ配り、宛先では response の頼みと合わせて2回入っていた。差分の直し（`patch`）は形式を直す呼び出しなので足さない
    if not patch:
        direct, upper = carry_for(step)
        if (t := upstream_text(step, upper)):
            parts.append(t)
        if (direct or prev is not None) and (t := response_ask(bool(asked), step=step)):
            parts.append(t)
    return "".join(f"\n\n{p}" for p in parts)


@contextmanager
def redo_base(step: str, prev):
    """**自工程のやり直しの間だけ、退けた稿を「前回の出力」に置く**（2026-10-04 組F4・1‑2）。`asks_for` が
    【この工程の前回の出力】（「これを土台に…変える必要がある所だけ」）と `response` の頼みを足す——差し戻しの周と同じ口。
    抜けるとき元へ戻す（差し戻しの周の前回の出力＝前の周の稿は、最終の点検役の比べる相手・上流で変わった所に使う）。
    伴奏（新しい曲 `song_band.run_arrange_gate`・カバー `song_arrange_cover._arrange_gate`）が使う。"""
    had, old = step in PREV_OUTPUTS, PREV_OUTPUTS.get(step)
    PREV_OUTPUTS[step] = prev
    KEEP_ALLOWED.discard(step)                             # 変えるために呼ぶ（「前回のまま」は選ばせない）
    try:
        yield
    finally:
        if had:
            PREV_OUTPUTS[step] = old
        else:
            PREV_OUTPUTS.pop(step, None)


def _indent_lines(body: str, pad: str = "    ") -> str:
    """指示の本文を、行ごとに字下げして返す（空行は落とす・行の中の空白はそのまま）。"""
    return "\n".join(pad + ln.strip() for ln in (body or "").splitlines() if ln.strip())


def response_ask(asked: bool = False, *, step: str = "") -> str:
    """**作り直す工程へ、指摘ごとの対応（`response`）を頼む文**（2026-10-04・組P2a）。この工程への差し戻し（`CARRY` の差し戻しの行で
    宛先がこの工程）と、作り直しの頼み（台帳から届いた頼み・`asked`）と、上流への差し戻し（巻き込まれた工程）がこの周の指摘。
    どれも無ければ空（指摘の無い呼び出しでは `response` は空のまま）。
    ★**宛先への指摘の本文はここに1回だけ書く**（2026-10-04 組F1・1‑5）。巻き込まれた工程へは上の節（`upstream_text`）を指すだけ。
    `asks_for` が、前回の出力のある工程（作り直す工程）と、差し戻しの宛先の工程に足す。"""
    from song_gate_brief import redo_ask               # noqa: E402（循環を避けてここで）
    direct, upper = carry_for(step)
    # ★指示の改行（前提・目的・【必須】…の見出しごとの行）を保ち、2行目以降を字下げする（2026-10-04。以前は空白をまとめて1行に詰めていた）
    points = [f"・{who or '点検役'}の点検役が「{target}から作り直す」と差し戻した:\n" + _indent_lines(body)
              for target, who, body in direct]
    if upper:
        points.append("・上の【上流への差し戻し】（上流の工程への指摘。この工程に響く所があれば、その対応を書く）")
    if asked:
        points.append("・上の【頼まれごと】（マスターの作り直しの頼み）")
    if not points:
        return ""
    return redo_ask("\n".join(points)
                    + "\n（この工程に向けた指摘でなければ、その指摘には触れなくてよい。触れる指摘が1つも無ければ `response` は空のまま）")


def _chosen_approach(theme):
    """テーマが選んだ方向の案（`Theme.approaches[chosen-1]`）。欄の無い型・番号が案の外なら None。"""
    aps = list(getattr(theme, "approaches", None) or [])
    k = getattr(theme, "chosen", 0) or 0
    return aps[k - 1] if isinstance(k, int) and 1 <= k <= len(aps) else None


def brief(theme, design=None, chords=None, melody=None) -> str:
    """**これまでの判断の申し送り**。次の工程が「なぜそうなっているか」を知るために渡す。

    人が曲を作るときは全体の設計を持ったまま各パートを作る。工程を分けた以上、
    **前段の意図を渡さないと各段が独立に判断してしまう**——旋律が「息継ぎの暇を与えない
    切迫感」を狙っても、詞と伴奏はそれを知らずに作られていた（2026-08-20 に指摘を受けて発覚）。
    噛み合うかどうかが運任せになる。
    """
    # **様式を先頭に置く**。イントロの長さ・構成・編成・リズムはすべてこれに従うので、
    # 各工程がばらばらに判断するとチグハグな曲になる（2026-08-22・本人指摘）
    # 様式・人物・声・歌い方・工夫・終わり方はどれも必須の欄（2026-10-04・組N2）。以前は `getattr(…, 既定)` で読み、書き忘れは
    #   型の既定のまま申し送りに載った。空の文字・null・空の並びは「無し」を選んだ印なので、行ごと出さない（以前と同じ）
    out = [f"■**様式**: {theme.era_style}"
           + (f"\n  （様式が要求すること）{theme.era_reason}" if theme.era_reason else ""),
           # ★**選んだ方向**（2026-10-05）。テーマは方向の案を比べてから芯を決める（`Theme.approaches`）。選んだ案と理由を全工程へ
           #   （詞はこの方向で書く——暗示なら言葉で言い切らない、など）。比べたほかの案は前段の出力の全部（`upstream_intents`）に載る
           *([f"■**曲の方向**: {_ap.direction}——{_ap.gist}\n  （ほかの案と比べてこれを選んだ理由）{theme.chosen_reason}"]
             if (_ap := _chosen_approach(theme)) is not None else []),
           f"■**この曲で言いたいこと**: {getattr(theme, 'message', theme.subject)}",
           f"■**曲を通しての心の動き**: {getattr(theme, 'arc', '（指定なし）')}",
           f"■題材: {theme.subject}／情景: {theme.scene}／感情: {theme.emotion}",
           f"  （その判断の理由）{theme.reason}"]
    # **誰が誰に、どの瞬間に言っているか**（詞も旋律もここから決まる）
    if theme.singer:                                       # 言いたいことのすぐ後ろ（曲の方向の行があっても位置は同じ）
        out.insert(next(i for i, s in enumerate(out) if s.startswith("■**この曲で言いたいこと**")) + 1, f"■**歌っている人**: {theme.singer}\n"
                      f"  **誰に**: {theme.listener}\n"
                      f"  **いつ・どこで**: {theme.moment}")
    # ★**曲の形**（2026-10-03・§2）を先頭に。歌ものは書かない（歌ものの申し送りは今までと同じ）
    if FORM != "song":
        from song_files import FORM_LABEL                   # noqa: E402
        out.insert(0, f"■**曲の形**: {FORM_LABEL[FORM]}（`{FORM}`）"
                      + ("——伴奏の楽器は無く、声だけで和声とリズムを作る" if FORM == "acappella"
                         else "——歌は無く、主旋律は楽器で弾く"))
    # ★**声を構想から全工程へ渡す**（2026-09-09 新設）。
    #   それまで声は終盤の「編成」工程が独断で決めており、**構想も設計も声に触れていなかった**。
    #   `Dark` も `Cute` も持っているのに毎回 SOFT × Breath になり、
    #   本人に「テンポに対して声が高かった。ダークとか使えなかったのか。
    #   **コンセプトの時点で検討した形跡はあった?**」と言われた（＝無かった）。
    _tone = {"落ち着いた": "**暗め・大人びた声**（`MIKU Dark` が近い）",
             "息づかい": "**息を多く含む声**（`MIKU Breath` / `MIKU Breath+`）",
             "かわいい": "**明るく高い声**（`MIKU Cute` / `MIKU Childish Soft`）",
             "低め": "**少年寄りの落ち着いた声**（`MIKU Boyish`）",
             "標準": "**素直な声**（`MIKU Default`）"}
    # ※声のトーン・歌い方の型・終わり方は型（`Literal`）で選択肢が閉じている（2026-09-30）ので、表を直接引く。
    #   以前の `.get(x, '')` は選択肢に無い綴りを黙って空の説明にしていた。表に無い値が来たら不具合（KeyError）
    if theme.voice_tone:
        out.append(f"■**声のトーン**: {theme.voice_tone}——{_tone[theme.voice_tone]}"
                   f"\n  **歌い方（スタイル）はこれに合わせて選んでください。**")
    if theme.voice_height:
        out.append(f"■**歌う高さ**: {theme.voice_height}"
                   # 高さは国際式の音名（2026-10-01・仕様書 §0-1。以前は MIDI 番号の 58〜86）
                   f"（ミクが気持ちよく鳴るのは A#3〜D6。それ以上も出るが痩せる）")
    if design is not None:
        out.append(f"■**全体方針**: {design.concept}")
        out.append(f"  （構成の理由）{design.reason}")
        # ★**歌い方の型と使う工夫を、全工程へ渡す**（2026-09-09 新設）。
        #   人は同じ脳で全工程をやるので「引き継ぎ」が要らないが、こちらは工程を分けた以上、
        #   **渡さなかったものは存在しないのと同じ**になる。
        _sty = {"続ける": "**切れ目なくつなぐ**（レガート・語り）。休みは少なく、1音は長め",
                "区切る": "**短く区切って畳み掛ける**。休みは多く、伸ばしは短め",
                "詰める": "**言葉数を多く入れる**。1音は短く、密度は高め",
                "保つ": "**一音一音をたっぷり伸ばす**。密度は低く、伸ばしは長め"}
        if design.singing_style:
            out.append(f"■**歌い方の型**: {design.singing_style}"
                       f"——{_sty[design.singing_style]}"
                       f"\n  **旋律・グルーヴ・詞のすべてがこの型に従ってください。**")
        if design.devices:
            out.append(f"■**この曲で使う工夫**: {' / '.join(design.devices)}"
                       f"\n  **ここに書かれていない工夫は、使わなくてよい**"
                       f"（全部盛りにすると、どれも効かなくなる）。"
                       f"どこで効かせるかは全体方針を見ること。")
        # **終わり方**は設計で決まる。編曲がここを知らないと「最後が弱い」曲になる
        if design.ending:
            how = {"fade": "**余韻を残して減らしていく**（アウトロで薄くしていく）",
                   "hit": "**最後に全部鳴らして切る**（ジャーンで終わる）",
                   "vocal": "**歌だけ残して消える**（伴奏が先に止む）",
                   "loop": "終わった感じを作らずそのまま切る"}[design.ending]
            out.append(f"■**終わり方**: {how}"
                       + (f"\n  （理由）{design.ending_reason}"
                          if design.ending_reason else ""))
    if chords is not None:
        out.append(f"■コードの意図: {chords.reason}")
    if melody is not None:
        out.append(f"■**旋律の意図**: {melody.reason}")
    # 差し戻しの理由などの申し送り。★**差し戻しの行（`carry_note`）はここに載せない**（2026-10-04 組F1・1‑5）——宛先の工程へは
    #   `asks_for` が1回だけ、巻き込まれた工程へは「上流への差し戻し」として別の節で渡す。以前はここで全工程へ配り、宛先では2回入っていた
    for note in carry_plain():
        out.append(f"■**前の工程からの申し送り**: {note}")
    return "\n".join(out)


def carry_plain() -> list[str]:
    """申し送り（`CARRY`）のうち、**差し戻しの行でないもの**（打ち止めの所見・人が直している範囲など。宛先が無く全工程へ配る）。"""
    from song_gate_brief import parse_carry                # noqa: E402（循環を避けてここで）
    return [n for n in CARRY if not parse_carry(n)]


def carry_for(step: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """申し送り（`CARRY`）の差し戻しの行を、`step` から見て (宛先の行, 巻き込みの行) に分ける（2026-10-04 組F1・1‑5）。
    宛先＝差し戻された工程そのもの。巻き込み＝差し戻された工程を読む工程（依存の木の子孫・`song_gate.is_ancestor`）。
    どちらでもない工程には届けない。行は (差し戻された工程, 差し戻した点検役, 指摘)。"""
    from song_gate import is_ancestor                      # noqa: E402
    from song_gate_brief import parse_carry                # noqa: E402
    direct, upper = [], []
    for note in CARRY:
        got = parse_carry(note)
        if not got or not step:
            continue
        if got[0] == step:
            direct.append(got)
        elif is_ancestor(got[0], step):
            upper.append(got)
    return direct, upper


#: 巻き込まれた工程へ渡す、上流への指摘の要旨の上限（字・1件）。全文は宛先の工程へ届く。切ったら書く
UPPER_GIST = 600


def upstream_text(step: str, upper: list[tuple[str, str, str]]) -> str:
    """**巻き込まれた工程へ渡す節**（2026-10-04 組F1・1‑5／§2‑2）: 上流への差し戻しの要旨と、上流で変わった所
    （作り直した前段の、前の稿といまの稿の違い＝`song_draft_diff`）。`upper` が空なら空。"""
    if not upper:
        return ""
    from song_draft_diff import diff_text                  # noqa: E402
    from song_gate import _ALL_STEPS, is_ancestor          # noqa: E402
    rows = []
    for target, who, body in upper:
        t = _indent_lines(body)                            # 改行を保つ（2026-10-04）
        gist = t[:UPPER_GIST] + (f"…（以下略・全{len(t)}字。全文は{target}の工程へ渡した）" if len(t) > UPPER_GIST else "")
        rows.append(f"・{who or '点検役'}の点検役 → {target}: {gist}")
    out = ("【上流への差し戻し（この工程は、上流の作り直しに巻き込まれて作り直す）】\n" + "\n".join(rows)
           + "\n※この指摘は上流の工程に向けたもの。**この工程に響く所だけ**合わせる。")
    song = CURRENT_SONG
    diffs = []
    for s in [x for x in _ALL_STEPS if x in PREV_OUTPUTS]:
        if s == step or not is_ancestor(s, step) or song is None:
            continue
        key = step_output_key(s)
        if key and (t := diff_text(s, key, PREV_OUTPUTS[s], getattr(song, key, None))):
            diffs.append(t)
    if diffs:
        out += ("\n\n【上流で変わった所（作り直した前段の、前の稿といまの稿の違い。機械が並べた事実）】\n" + "\n".join(diffs))
    return out
