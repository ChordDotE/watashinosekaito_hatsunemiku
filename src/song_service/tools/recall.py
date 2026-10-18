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

"""**どう作ったかを思い出す**（読むだけ・何も変えない）。2026-09-15 新設。

## なぜ要るか

本人:「**後から読む手段はあるはずだけど、ログは見れるよね。ツールを使って何も編集せずに
ログだけ読む、みたいなのってできなかったっけ**」

工程ごとの判断（構想・設計・コード・土台・旋律・**詞の設計**・サビの詞・編成・編曲）は
`data/songs/_wip/<曲名>.steps.json` に**全部残っている**。だが**それを読む手段が無かった**——
`diagnosis.py` の道具は音符・尺・音量を**測る**もので、`work_on_music` の戻り値は曲一覧と
実行状態だけ。ミクが「こんな風に作った」と話せていたのは、**作った直後の申し送り（200字）が
会話の記憶に残っていたあいだ**だけで、日が変わると引けなかった。

★**読むだけ**。曲の実体（楽譜・音・版）には一切触れない。
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic_ai import RunContext

from coordinator import coordinator
from session import SONGS, MusicSession, with_scope

#: 工程の保存先（`song_flow.AUTOSAVE_DIR` と同じ場所を指す）
WIP = SONGS / "_wip"

#: 表示する工程の順（作る順）と、その見出し
_STEP_LABEL = [
    ("theme", "① どんな曲にしたかったか"),
    ("design", "② 構成・テンポ・キー・声の高さ"),
    ("chords", "③ コード進行"),
    ("groove", "④ 土台（ドラム・ベース・行の形）"),
    ("melody", "⑤ 旋律"),
    ("lyric_plan", "⑥ 詞の設計（決め台詞・置き場）"),
    ("lyric_chorus", "⑦ サビの詞"),
    ("lyrics", "⑧ 詞"),
    ("instrumentation", "⑨ 編成（楽器と歌い方）"),
    ("arrangement", "⑩ 編曲"),
    ("harmony", "⑪ ハモリ"),
]


def _steps_path(song_id: str) -> Path | None:
    """曲フォルダ名（`YYYYMMDD_HHMMSS_タイトル`）から工程の記録を引く。

    ★**まず曲フォルダの記録**（2026-09-25・段階3 から版ごとに `record_v{N}.json`。作業中は `record.json`）。
    それより前の曲は `_wip` に**曲名**で置かれているので、フォルダ名の末尾を使う。曲名が変わっている場合に
    備えて、`score.json` の `title` でも探す。
    """
    from session import _current_version
    d = SONGS / song_id
    if d.exists():
        for p in (d / "record.json", d / f"record_v{_current_version(d)}.json"):
            if p.exists():
                return p
    for title in (song_id.split("_", 2)[-1], _title_of(song_id)):
        if not title:
            continue
        p = WIP / f"{title}.steps.json"
        if p.exists():
            return p
    return None


def _title_of(song_id: str) -> str:
    p = SONGS / song_id / "score.json"
    if not p.exists():
        return ""
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("title") or ""
    except Exception:
        return ""


def _trim(s: str, n: int = 420) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[:n] + "…"


@coordinator.tool
@with_scope
def recall_making(ctx: RunContext[MusicSession], song_id: str = "") -> str:
    """**その曲をどう作ったか**を思い出す（**読むだけ**・曲は何も変わらない）。

    工程ごとに「何を狙って、なぜそうしたか」と、**決め台詞・曲名の由来・hook の置き場・
    転調・やり直しになった工程**を返す。アカペラ・インストなら**曲の形**と、形を変えた曲
    （「最初はインストで作り、あとで歌を足した」など）は**形を変えた履歴**も返す。

    **呼ぶ場面は2つ**:
    1. **作り方を聞かれたとき**——「なんでこの曲名にしたの?」「ここで盛り上がるのはどうして?」
    2. ★★**違和感を指摘されたとき（調べる前に、まずこれ）**★★
       ——`inspect_song` だけ見ると「変な所」は必ず見つかるが、**それが狙いだったかは
       ここを読まないと分からない**。読まずに直すと**意図して作った所を壊す**。
       ※実例（2026-09-20）: サビ頭でドラムがほぼ消えているのを「腰砕け」と断定したが、
       **サビ頭のブレイクはロックの定番の演出**でもあった。
       ★**2つ目の場面を消さないこと**（経緯は decisions.md 2026-09-20）。

    `song_id` を省くと、いま開いている曲（無ければ最後に作った曲）。
    """
    sid = song_id or ctx.deps.song_id or ""
    if not sid:                                            # 最後に作った曲（フォルダ名が日時順）
        dirs = sorted(p.name for p in SONGS.iterdir()
                      if p.is_dir() and p.name[:8].isdigit())
        sid = dirs[-1] if dirs else ""
    if not sid:
        return "どの曲か分からない。曲名か『さっきの曲』のように教えてもらう。"
    path = _steps_path(sid)
    if path is None:
        return (f"『{sid}』の作業の記録が残っていない（工程の記録は曲フォルダの `record_v<版>.json`、"
                "途中で止めた曲は `_wip/<曲名>.steps.json`）。"
                "**記録を消したあとの曲では思い出せない**——正直にそう伝える。")

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return f"記録が読めなかった（{type(e).__name__}）。壊れている可能性がある: {path.name}"
    steps = doc.get("steps") or {}
    out: list[str] = [f"■ 『{doc.get('title') or sid}』をどう作ったか"]
    if inst := doc.get("instruction"):
        out.append(f"  最初の依頼: {_trim(inst, 200)}")
    # ★**曲の形と、形を変えた履歴**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §4-5）。
    #   「最初はインストで作り、あとで歌を足した」と話せるように。形の変更は記録の `form` の行（作り直しが積む）と台帳の「形」の行
    import song_files
    try:
        form = song_files.as_form(doc.get("form"))
    except ValueError:
        form = "song"
    changes = [h for h in (doc.get("history") or []) if h.get("step") == "form"]
    if form != "song" or changes:
        out.append(f"  曲の形: {song_files.FORM_LABEL[form]}"
                   + {"acappella": "（伴奏の無い、声だけの曲）", "instrumental": "（歌の入らない、演奏だけの曲）"}.get(form, ""))
    for h in changes:
        out.append(f"  形を変えた: {_trim(h.get('reason'), 240)}"
                   + (f"（{h.get('time')}）" if h.get("time") else ""))

    # ---- 設計が数字で決めたもの（聞かれやすいので先に出す）----
    d, th = steps.get("design") or {}, steps.get("theme") or {}
    facts = [f"BPM {d.get('bpm')}", f"キー {d.get('key')}",
             (f"声の高さ {th.get('voice_height')}（最高音 {d.get('top_note')}）" if form != "instrumental"
              else f"主旋律の最高音 {d.get('top_note')}"),       # インストは声の欄を使わない（主旋律の音域は声の枠のまま・F2）
             f"歌い方の型「{d.get('singing_style')}」" if form != "instrumental" else "",
             f"使った工夫 {'・'.join(d.get('devices') or []) or '（なし）'}",
             f"終わり方 {d.get('ending')}"]
    out.append("  決めたこと: " + " ／ ".join(str(x) for x in facts if x))
    if mods := (d.get("modulations") or []):
        out.append("  転調: " + " ／ ".join(
            f"{m.get('at_section')} から {m.get('key')}（{_trim(m.get('reason'), 80)}）" for m in mods))

    # ---- 曲名の由来（詞のあとの工程『曲名』が決める・どの形でも・2026-10-03。テーマは仮の曲名）----
    from song_model import made                            # 走らなかった工程（空の出力・skipped・F1）は「無い」
    nm = steps.get("naming") or {}
    work = th.get("working_title") or ""
    if made(nm) and isinstance(nm, dict) and nm.get("title"):
        out.append(f"\n  ★曲名『{nm.get('title')}』は詞のあとの工程で決めた"
                   + ("（テーマの仮の曲名のまま）" if nm.get("title") == work
                      else (f"（テーマの仮の曲名は『{work}』）" if work else "")))
        if nm.get("reason"):
            out.append(f"   そう決めた理由: {_trim(nm.get('reason'))}")
    elif work:
        out.append(f"\n  ★仮の曲名『{work}』はテーマの工程で決めた（狙いは下の①。最後の曲名の工程はまだ）")

    # ---- 詞の設計（決め台詞の由来と置き場。ここがいちばん聞かれる）----
    if made(lp := steps.get("lyric_plan")) and isinstance(lp, dict):
        out.append(f"\n  ★決め台詞「{lp.get('hook')}」"
                   f"（{lp.get('hook_section')} の {lp.get('hook_line')}行目に置いた）")
        out.append(f"   そう決めた理由: {_trim(lp.get('reason'))}")
        if cands := (lp.get("hook_candidates") or []):
            out.append(f"   候補にしていた言葉: {' / '.join(cands)}")
        out.append(f"   誰が誰に: {_trim(lp.get('viewpoint'), 120)}")
        for s in (lp.get("sections") or []):
            out.append(f"   {s.get('kind')}: {s.get('says')}")
    elif form == "instrumental":
        # ★**詞の工程が無い曲**（インスト・2026-10-03）。決め台詞は無い（曲名は上で曲名の工程から出した）
        out.append("\n  歌詞なし（インストなので詞の工程が無い）")

    # ---- 工程ごとの狙い ----
    out.append("")
    for key, label in _STEP_LABEL:
        st = steps.get(key)
        if not isinstance(st, dict) or not made(st):       # 走らなかった工程は狙いを持たない（F1）
            continue
        if key == "lyric_plan":                            # 上で詳しく出した
            continue
        if r := st.get("reason"):
            out.append(f"  {label}: {_trim(r)}")
        if t := st.get("trouble"):
            out.append(f"     うまくいかなかったこと: {_trim(t, 200)}")

    # ---- やり直し（ここが「苦労した話」になる）----
    redone = [h for h in (doc.get("history") or [])
              if h.get("status") != "ok" or h.get("request") == "redo"]
    if redone:
        out.append("\n  作り直した工程:")
        for h in redone:
            out.append(f"   {h.get('step')}: {_trim(h.get('reason'), 200)}")

    # ★**マスターに頼まれたこと**（台帳・2026-09-25）。「なんでここに合いの手があるの?」に
    #   「前にマスターが『サビに足して』と言ったから」まで答えられるように。
    # ★**作風の手本にしたもの**（下調べ・2026-09-26）。借りた元を聞かれたら隠さず答えられるように
    styles = [r for r in (doc.get("requests") or []) if r.get("origin") == "下調べ"]
    if styles:
        out.append("\n  作風の手本にしたもの（ネットで下調べ。借りたのは作風だけ）:")
        for r in styles:
            out.append(f"   {_trim((r.get('words') or '').split(chr(10))[0], 80)}")
    asks = [r for r in (doc.get("requests") or []) if r.get("origin") not in ("鼻歌", "下調べ")]
    if asks:
        out.append("\n  マスターに頼まれたこと:")
        for r in asks:
            gone = {"superseded": "（あとで取り消し）",
                    "reapply": "（作り直しで当てはまらなくなり、当て直し待ち）"}.get(r.get("status"), "")
            where = r.get("target", "") + (f"（{r.get('scope')}）" if r.get("scope") else "")
            out.append(f"   {r.get('time', '')} {where}: 「{_trim(r.get('words'), 120)}」{gone}")

    out.append("\n★これは**読んだだけ**——曲は何も変わっていない。"
               "マスターには**自分の言葉で**話すこと（工程名やキー名をそのまま言わない）。")
    return "\n".join(out)
