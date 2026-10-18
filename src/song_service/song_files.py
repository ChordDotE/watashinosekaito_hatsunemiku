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

"""**曲フォルダのファイルの名前の決まり**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §6・F12〜F18）。

曲フォルダに置くファイルの名前の作り方と読み方を、ここ1か所に集める。session・song_runner・handoff・play_song・
Gateway・手直しの指紋（`manual_diff`）・書き出し（`song_pipeline`・`v6_export`）がここを読む。
★**標準ライブラリだけで書く**（song_service の外＝ミク本体・Gateway からも読めるように。pydantic も読まない）。

## 名前の決まり（1行）

`master_` で始まるのが**完成品**（聴くもの）、`log_` が**ログ**（読まなくてよい記録）、`_` が**機械の印**、
接頭辞の無いものと `source/` が**途中の物**（F15②）。版番号はどれも**名前の最後・拡張子の前に `_v{N}`**（F16①）。
ログは曲フォルダの中に `log_` を付けて置く（`log/` フォルダは作らない・F18①）。

| 種類 | 名前 | 何か |
|---|---|---|
| 完成品 | `master_song_v{N}.wav`・`master_acappella_v{N}.wav`・`master_inst_v{N}.wav` | 聴くもの（形が名前に入る） |
| 途中の物 | `score`・`record`・`mix`・`vocal`（vpr）・`accompaniment`（wav/mid）・`placeholder`・`lipsync` の `_v{N}`、`preview.wav`、`source/` | 作業台（固定名）は作業中だけ |
| ログ | `log_build.txt`・`log_v6export_v{N}.txt`・`log_prompts_v{N}/` | 作業記録・V6 の自動操作のログ・依頼文の写し |
| 機械の印 | `_meta.json`・`_progress.json`・`_manifest_v{N}.json`・`_CURRENT_v{N}.txt`・`_WORKING_v{N}.txt`・`_versions.jsonl` | 書き出しの状態・止めた印・指紋・版の印・版の記録（いつ・何を直した） |
| V6 の生の書き出し（一時） | `vocal_v{N}.wav` | V6 が vpr と同じ名前で出す。仕上げのあと完成品へ付け替える（F14①） |

形（`song`／`acappella`／`instrumental`）の綴りの決まりもここに置く（`song_model.Form` と同じ3つ・テストが突き合わせる）。
古い名前から新しい名前への一度だけの付け替えの表は `legacy_rename`（F13①・F17②。付け替えは本人の承認のあと手で行う）。
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

# ---- 形 ----------------------------------------------------------------------

#: 曲の形（`song_model.Form` と同じ3つ・この順）。**空＝`song`**
FORMS = ("song", "acappella", "instrumental")
DEFAULT_FORM = "song"
#: 人に見せる呼び名
FORM_LABEL = {"song": "歌もの", "acappella": "アカペラ", "instrumental": "インスト"}
#: 完成品の名前に入る綴り（`master_<これ>_v{N}.wav`・F12①）
FORM_TAG = {"song": "song", "acappella": "acappella", "instrumental": "inst"}
_TAG_FORM = {v: k for k, v in FORM_TAG.items()}
#: 形の前後で取り除く字（半角の空白とタブ・`song_checks.ASCII_WS` と同じ）
_ASCII_WS = " \t"


def form_problem(text: str | None) -> str:
    """形の綴りが読めないときの所見（読めれば空）。綴りは `FORMS` の3つだけ・前後の半角の空白とタブだけ無視・大文字小文字は区別・空＝song。

    ★前後の全角の空白・改行などは読まない（2026-10-03・仕様書 §0-6 の判断4 本人決定。半角の記法の部品は `song_checks.strip_ascii_ws`——
      ここは標準ライブラリだけで書くので同じ字（半角の空白とタブ）を `_ASCII_WS` に写す）。"""
    v = (text or "").strip(_ASCII_WS)
    if not v or v in FORMS:
        return ""
    return (f"形『{text}』は読めない——形は {' / '.join(FORMS)} のどれかをこの綴りのまま書く"
            "（空なら song。大文字・別名・日本語は読まない）")


def as_form(text: str | None) -> str:
    """形の綴りを読む（前後の半角の空白とタブを取り除き、空なら `song`）。読めない綴りは `ValueError`（所見の文つき）。"""
    if (why := form_problem(text)):
        raise ValueError(why)
    return (text or "").strip(_ASCII_WS) or DEFAULT_FORM


# ---- 曲名からファイル・フォルダの名前を作る -------------------------------------------

#: フォルダ名から落とすノイズ語（コードが付ける区間「_全体」や「修正版」など。版は版管理が担う）
SLUG_NOISE = {"全体", "修正版", "新規", "カバー", "フル"}


def slug(title: str) -> str:
    """**曲名 → ファイル・フォルダの名前に使える綴り**（曲フォルダ名・途中保存・差し戻しの保存・作り直しの作業場の1か所）。

    日本語はそのまま残し、ファイル名に使えない字（`\\ / : * ? " < > |`）と空白を `_` にする。ノイズ語（`SLUG_NOISE`）を落とし、40字で切る。
    空になりうるので、呼び手が既定の名前を添える。
    ★2026-10-03（再レビュー）: 曲名をそのままファイル名にしていた所が3か所あり（途中保存・差し戻しの保存・作り直しの作業場）、
      `?`・`:` で保存できない・`/` でフォルダが1段できた。`session._slug`（2026-07-10・フォルダ名）をここへ移して同じ1つを通す。
    """
    t = re.sub(r'[\\/:*?"<>|\s]+', "_", title or "").strip("_")
    parts = [p for p in t.split("_") if p and p not in SLUG_NOISE]
    return "_".join(parts)[:40]


# ---- 接頭辞 --------------------------------------------------------------------

MASTER_PREFIX = "master_"                                 # 完成品
LOG_PREFIX = "log_"                                       # ログ
MARK_PREFIX = "_"                                         # 機械の印

# ---- 途中の物（作業台＝固定名。版の写しは `versioned(名前, N)`）---------------------

SCORE = "score.json"                                      # 楽譜
VPR = "vocal.vpr"                                         # V6 のデータ（人が V6 で開いて直す）
GUIDE = "placeholder.wav"                                 # サイン波の仮歌
ACCOMP_WAV = "accompaniment.wav"                          # 伴奏の音
ACCOMP_MID = "accompaniment.mid"                          # 伴奏の楽譜（人が MIDI の道具で直す）
MIX = "mix.json"                                          # 音量の設定
LIPSYNC = "lipsync.json"                                  # 口の動き
RECORD = "record.json"                                    # 工程の記録と台帳
PREVIEW = "preview.wav"                                   # 途中で止めたときに聴かせる音（版を持たない）
SOURCE_DIR = "source"                                     # 外の材料（カバーのお手本・鼻歌。中の名前は外の道具が付ける）

#: 作業台に広げて版を切るファイル（`session._VERSIONED` と同じ並び・テストが突き合わせる）
WORKBENCH = (SCORE, VPR, GUIDE, ACCOMP_WAV, ACCOMP_MID, MIX, LIPSYNC, RECORD)

#: **形ごとの持ち物**（§4-4）。その形の版が持つ作業台のファイル。書き出しの最初に、作業台からこれに無いファイルを消す
#: （前の形のファイルが新しい版に写らないように）。アカペラは伴奏 wav/mid を、インストは vpr・仮歌・口の動きを持たない
FORM_FILES: dict[str, tuple[str, ...]] = {
    "song": WORKBENCH,
    "acappella": tuple(n for n in WORKBENCH if n not in (ACCOMP_WAV, ACCOMP_MID)),
    "instrumental": tuple(n for n in WORKBENCH if n not in (VPR, GUIDE, LIPSYNC)),
}

# ---- ログ ----------------------------------------------------------------------

BUILD_LOG = "log_build.txt"                               # 全工程の入出力・検問の所見（以前は build_log.txt）


def v6export_log(version: int) -> str:
    """V6 の自動操作のログ（以前は `vocal_v{N}.export.log`）。"""
    return f"{LOG_PREFIX}v6export_v{_n(version)}.txt"


def prompts_dir(version: int) -> str:
    """工程の LLM への依頼文と出力の写しのフォルダ（以前は `prompts_v{N}/`）。"""
    return f"{LOG_PREFIX}prompts_v{_n(version)}"


# ---- 機械の印 ------------------------------------------------------------------

META = "_meta.json"                                       # 書き出しの状態（以前は meta.json）
PROGRESS = "_progress.json"                               # 途中で止めた・失敗した目印（以前は progress.json）


def manifest(version: int) -> str:
    """版のファイルの指紋（以前は `manifest_v{N}.json`）。"""
    return f"{MARK_PREFIX}manifest_v{_n(version)}.json"


def current_mark(version: int) -> str:
    """現在版の印。"""
    return f"_CURRENT_v{_n(version)}.txt"


def working_mark(version: int) -> str:
    """作業中の版の印。"""
    return f"_WORKING_v{_n(version)}.txt"


# ---- V6 の生の書き出し（一時）---------------------------------------------------

def v6_raw(version: int) -> str:
    """V6 が `vocal_v{N}.vpr` から書き出す生の歌声（vpr と同じ名前で出るので変えられない）。仕上げのあと完成品へ付け替える。"""
    return f"vocal_v{_n(version)}.wav"


# ---- 版番号（F16①: 名前の最後・拡張子の前に `_v{N}`）------------------------------

_VERSIONED_RE = re.compile(r"^(?P<base>.+?)_v(?P<n>[1-9]\d*)(?P<ext>\.[A-Za-z0-9]+)?$")


def _n(version: int) -> int:
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(f"版の番号は1以上の整数（{version!r}）")
    return version


def versioned(name: str, version: int) -> str:
    """固定名 → その版の名前（`score.json` → `score_v3.json`・拡張子の無いフォルダ `log_prompts` → `log_prompts_v3`）。"""
    stem, dot, ext = name.rpartition(".")
    if not dot:
        return f"{name}_v{_n(version)}"
    return f"{stem}_v{_n(version)}.{ext}"


def split_version(name: str) -> tuple[str, int] | None:
    """版つきの名前 → (固定名, 版)。`score_v3.json` → (`score.json`, 3)。版の形でなければ None。

    ★`_v{N}` の後ろに拡張子が1つだけの形しか読まない（以前の `vocal_v3.export.log`・`vocal_v3_raw.wav` は版の形ではない）。
    """
    m = _VERSIONED_RE.match(name)
    if not m:
        return None
    return m["base"] + (m["ext"] or ""), int(m["n"])


def version_of(name: str) -> int | None:
    """名前の版番号（版の形でなければ None）。"""
    got = split_version(name)
    return got[1] if got else None


# ---- 完成品 --------------------------------------------------------------------

_MASTER_RE = re.compile(r"^master_(?P<tag>" + "|".join(map(re.escape, FORM_TAG.values()))
                        + r")_v(?P<n>[1-9]\d*)\.wav$")


def master(form: str, version: int) -> str:
    """完成品の名前（`master_song_v3.wav`・`master_acappella_v3.wav`・`master_inst_v3.wav`）。"""
    return f"{MASTER_PREFIX}{FORM_TAG[as_form(form)]}_v{_n(version)}.wav"


def parse_master(name: str) -> tuple[str, int] | None:
    """完成品の名前 → (形, 版)。**完成品でなければ None**（`vocal_v3.wav`・`final_v3.wav`・`accompaniment_v3.wav` は完成品ではない）。"""
    m = _MASTER_RE.fullmatch(name)                 # ★fullmatch: `$` だけだと末尾が改行の名前が通る（Gateway の配信の白名簿）
    if not m:
        return None
    return _TAG_FORM[m["tag"]], int(m["n"])


def is_master(name: str) -> bool:
    return parse_master(name) is not None


class Master(NamedTuple):
    """曲フォルダにある完成品1つ。"""
    version: int
    form: str
    path: Path


def masters(song_dir: Path) -> list[Master]:
    """曲フォルダの完成品（版の順）。無ければ空。"""
    out = []
    if song_dir.is_dir():
        for p in song_dir.iterdir():
            if p.is_file() and (got := parse_master(p.name)):
                out.append(Master(got[1], got[0], p))
    return sorted(out, key=lambda m: (m.version, FORMS.index(m.form)))


def master_of(song_dir: Path, version: int) -> Master | None:
    """その版の完成品（無ければ None）。1つの版の形は1つなので、見つかった最初の1つ。

    ★1つの版に形の違う完成品を作らないのは書き出しの側（2026-10-03・再レビュー）: 作業中の版に別の形の完成品が
      あれば、書き出し（`song_pipeline.compose`）がその版を確定して次の版を開く（`session.open_version_for_form`）。"""
    return next((m for m in masters(song_dir) if m.version == version), None)


# ---- 版の記録（いつ作った・何を直した）----------------------------------------------

#: 版ごとの「いつ作ったか・何を直したか」（1行1版・版を確定するたびに追記）
VERSIONS = "_versions.jsonl"


def _stamp(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return ""


def _requests(song_dir: Path, version: int) -> list[dict]:
    try:
        return json.loads((song_dir / versioned(RECORD, version)).read_text(encoding="utf-8")).get("requests") or []
    except Exception:
        return []


def _request_text(r: dict) -> str:
    """頼みごと1件を人の言葉で短く（音符の列など機械の書式は、何をしたかの頭だけ）。"""
    words = re.sub(r"\s+", " ", str(r.get("words") or "")).strip()
    if "@" in words and ": " in words:
        words = words.split(": ", 1)[0]
    scope = re.sub(r"(\d+):[\d.]+", r"\1小節", str(r.get("scope") or ""))
    return (f"{scope}の{words}" if scope and scope not in words else words)[:50]


def version_changes(song_dir: Path, version: int, base: int) -> list[str]:
    """その版で増えた頼みごと（前の版の記録に無いもの）の、最後の2件（頼まれた順）。"""
    old = {r.get("id") for r in _requests(song_dir, base)} if base else set()
    new = [r for r in _requests(song_dir, version) if r.get("id") not in old]
    return [t for t in (_request_text(r) for r in new[-2:]) if t]


def record_version(song_dir: Path, version: int, base: int, asked: str = "") -> None:
    """版を確定したときに1行残す（`session.end_session`）。`asked` はその回に頼まれた言葉。"""
    entry = {"v": version, "made": datetime.now().strftime("%Y-%m-%d %H:%M"), "base": base or None,
             "changes": version_changes(song_dir, version, base), "asked": (asked or "").strip()[:80]}
    with open(song_dir / VERSIONS, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def version_log(song_dir: Path) -> list[dict]:
    """版ごとの記録（古い順）。

    ★**記録の無い版（2026-10-11 より前）はファイルから組み立てる**: 時刻は版を確定したときに書く指紋（`_manifest_v{N}.json`）の
      更新時刻（無ければ楽譜）、直した内容は頼みごとの差分。完成品があれば `mastered`（完成品の時刻）と `form` を足す。
    """
    logged: dict[int, dict] = {}
    try:
        for ln in (song_dir / VERSIONS).read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(ln)
                logged[int(e["v"])] = e
            except Exception:
                continue
    except OSError:
        pass
    found = {v for p in song_dir.glob("score_v*.json") if (v := version_of(p.name))} if song_dir.is_dir() else set()
    out: list[dict] = []
    prev = 0
    for v in sorted(found | set(logged)):
        e = dict(logged.get(v) or {})
        if not e:
            src = song_dir / manifest(v)
            e = {"v": v, "made": _stamp(src if src.exists() else song_dir / versioned(SCORE, v)),
                 "base": prev or None, "changes": version_changes(song_dir, v, prev), "asked": ""}
        if (m := master_of(song_dir, v)):
            e["mastered"], e["form"] = _stamp(m.path), FORM_LABEL[m.form]
        out.append(e)
        prev = v
    return out


def version_note(e: dict) -> str:
    """版1つを1行で（例: `10/10 23:24 13小節〜15小節の音符を置き直した`）。"""
    made = str(e.get("made") or "")[5:].replace("-", "/")
    if not e.get("base"):
        what = "最初の版"
    else:
        what = "／".join(e.get("changes") or []) or (f"頼まれたこと「{e['asked']}」" if e.get("asked") else "")
    return " ".join(x for x in (made, what, "" if e.get("mastered") else "（完成品なし）") if x)


def versions_text(song_dir: Path, last: int = 5) -> str:
    """新しいほうから `last` 版ぶんを1版1行で（`第14版 10/10 23:24 …`）。"""
    return "\n".join(f"第{e['v']}版 {version_note(e)}" for e in version_log(song_dir)[-last:])


# ---- 種類----------------------------------------------------------------------

def kind_of(name: str) -> str:
    """名前から種類を読む: `master`（完成品）・`log`（ログ）・`mark`（機械の印）・`work`（途中の物）。"""
    if is_master(name):
        return "master"
    if name.startswith(LOG_PREFIX):
        return "log"
    if name.startswith(MARK_PREFIX):
        return "mark"
    return "work"


# ---- 古い名前からの付け替え（一度だけ・F13①・F17②）-----------------------------------

#: (古い名前の形, 新しい名前の作り方)。**古い曲は全部歌もの**なので完成品は `master_song_…`。
#: 付け替えるのは生きている曲と保管庫（`_archive/`・`.backup/`・`_backup_20260815_before_vpr_rebuild/`）の両方（F17②）。
#: ここに無い名前（楽譜・伴奏・vpr・`final.wav`・`draft_mix*.wav`・`vocal_v{N}_raw.wav` ほか）は付け替えない
_LEGACY: tuple[tuple[re.Pattern[str], object], ...] = (
    (re.compile(r"^vocal_v([1-9]\d*)\.wav$"), lambda m: master("song", int(m[1]))),
    (re.compile(r"^build_log\.txt$"), lambda m: BUILD_LOG),
    (re.compile(r"^vocal_v([1-9]\d*)\.export\.log$"), lambda m: v6export_log(int(m[1]))),
    (re.compile(r"^prompts_v([1-9]\d*)$"), lambda m: prompts_dir(int(m[1]))),
    (re.compile(r"^meta\.json$"), lambda m: META),
    (re.compile(r"^progress\.json$"), lambda m: PROGRESS),
    (re.compile(r"^manifest_v([1-9]\d*)\.json$"), lambda m: manifest(int(m[1]))),
)


def legacy_rename(name: str) -> str | None:
    """古い名前 → 新しい名前（付け替えないものは None）。曲フォルダの直下のファイル・フォルダの名前1つを渡す。"""
    for pat, make in _LEGACY:
        if (m := pat.match(name)):
            return make(m)                                # type: ignore[operator]
    return None
