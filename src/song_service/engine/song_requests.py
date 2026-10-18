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

"""**頼まれごとの台帳**（2026-09-25・`docs/plan/song_revision_design.md` §5・段階2）。

## なぜ要るか

人の編曲者は、作り直しのときに**ファイルではなく「頼まれたことのメモ」を持ち越す**。
旋律を書き直しても、「サビに合いの手」「ギターは泣かせて」を見ながら新しい旋律に合わせて作り直す。

それまでは、マスターの作り直しの指示も鼻歌の手本も `song_compose.CARRY`（**宛先の無い1本の
申し送り**）に入れて全工程へ配り、**作り直しが終わると消していた**。記録にも残らない。
なので頼んだことはその1回の作り直しにしか効かず、**次の作り直しでは誰も覚えていなかった**。

## 考え方

- **置き換え／足す／消すは、工程の設定ではなく頼まれたことの意味で決まる**。
  意味の判断（どの工程に・何を・どの範囲で）は**コーディネーター**、記録と伝達は**機械**
  （2026-09-25 のカバー判定と同じ分け方）
- **頼まれた工程と、その下流にだけ届ける**（依存の木 `song_gate.DEPS` を辿る）。
  「ギターを泣かせて」は伴奏にも奏法にも効くが、歌い方には届かない
- **古い頼みを勝手に消さない**。置き換えるときはコーディネーターが `supersedes` で指す
  （範囲が空の頼みどうしを機械が自動で置き換えると、「サビを明るく」のあとの「A メロを静かに」が
  前の頼みを消してしまう）
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime

#: 頼みの種類。**意味で決める**（コーディネーター）
#: ★`keep`＝**続いている頼み**（2026-10-14・本人了承）。一度通った頼み（「最高音はサビまで取っておく」「シ↓グナル」）が、
#:   後の版の手直しで黙って崩れていた（『てのひらのシグナル』v19・v20〜26）。台帳に残るのが「音符を置き直した」の羅列だけで、
#:   意図が残らなかったため。`tools/editing.keep_request` が書き、曲を開くたび・編集のあとの点検役に【続いている頼み】として見せる
OPS = {"set": "変える", "add": "足す", "remove": "消す", "keep": "守る"}


def standing(requests: list["Request"]) -> list["Request"]:
    """**続いている頼み**（`op == "keep"` の生きている行）。"""
    return [r for r in requests if r.status == "active" and r.op == "keep"]


@dataclass
class Request:
    """台帳の1行。"""
    id: str
    words: str                       # マスターの言葉（コーディネーターへ届いた依頼文）／鼻歌なら読み取った手本
    target: str                      # どの工程にかかるか（`STEP_ORDER` の表記。例: 伴奏・声部）
    op: str = "set"                  # set / add / remove
    scope: str = ""                  # 範囲（空＝全体。例: サビ・`third_up`・`12:0〜14:0`）
    origin: str = "マスター"          # マスター／鼻歌／下調べ（作風）／直した記録（マスターの頼みを受けて）＝編集の道具が書いた「したこと」
    status: str = "active"           # active / superseded / reapply（当て直し待ち）
    time: str = ""
    version: int = 0                 # どの版で頼まれたか
    superseded_by: str = ""
    prev: str = ""                   # 頼まれた時点の、その工程の前回の意図（差分で直すための文脈）
    #: **機械が作り直しのあとに当て直すための値**（段階3）。速さ `{"tempo"}`・移調 `{"semitones"}`・
    #: 足した声部 `{"track"}`・声部の直し `{"transpose","gain_db"}`・音符 `{"start_tick","end_tick","notes"}`
    value: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Request":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def add(requests: list[Request], words: str, target: str, *, op: str = "set", scope: str = "",
        origin: str = "マスター", version: int = 0, prev: str = "",
        supersedes: list[str] | tuple[str, ...] = (), value: dict | None = None) -> Request:
    """台帳に1行足す。`supersedes` に挙げた行は**置き換えられた**にする（消さない＝経緯は残る）。"""
    op = op if op in OPS else "set"
    value = dict(value or {})
    # ★**移調は「その時点からの差」なので、置き換えた移調の分を足し込む**（2026-09-26）。
    #   『明日への足音』で「+2」のあとに「声を低く」と頼まれ、コーディネーターはいまの状態から -5 動かして
    #   前の +2 を置き換えた（意図は原曲から -3）。台帳に -5 だけが残り、次に作り直すと原曲から -5 が当たる
    #   ——いまより2半音低い曲になる。置き換えるときは、消える差ぶんを新しい行へ移す。
    if "semitones" in value:
        folded = sum(int(q.value.get("semitones", 0)) for q in requests
                     if q.id in supersedes and q.status in ("active", "reapply")
                     and q.target == target and "semitones" in q.value)
        if folded:
            value["semitones"] = int(value["semitones"]) + folded
            words = f"{words.strip()}（置き換えた移調 {folded:+d} を足して、原曲から {value['semitones']:+d}）"
    if value.get("transpose") is not None and value.get("title"):   # 声部1つの移調も同じ（同じ名前の声部どうし）
        folded = sum(int(q.value.get("transpose") or 0) for q in requests
                     if q.id in supersedes and q.status in ("active", "reapply")
                     and q.target == target and q.value.get("title") == value["title"])
        if folded:
            value["transpose"] = int(value["transpose"]) + folded
            words = f"{words.strip()}（置き換えた移調 {folded:+d} を足して、合わせて {value['transpose']:+d}）"
    r = Request(id=f"r{uuid.uuid4().hex[:6]}", words=words.strip(), target=target, op=op,
                scope=scope.strip(), origin=origin, version=version, prev=prev.strip()[:400],
                time=datetime.now().strftime("%Y-%m-%d %H:%M"), value=value)
    for q in requests:
        if q.id in supersedes and q.status in ("active", "reapply"):
            q.status, q.superseded_by = "superseded", r.id
    requests.append(r)
    return r


def active(requests: list[Request]) -> list[Request]:
    return [r for r in requests if r.status == "active"]


def reaches(r: Request, step: str) -> bool:
    """その頼みが `step` に届くか——**頼まれた工程と、その下流**。"""
    from song_gate import DEPS, affected
    if r.target != "テーマ" and r.target not in DEPS:
        return False                                     # 工程でない先（組み立ての設定など）は工程へは届けない
    return step in affected(r.target)


def deliver(requests: list[Request], step: str) -> str:
    """`step` へ渡す文面（届く頼みが無ければ空）。

    ★**作り直しの文脈を2種類に分けて書く**（§5.3）——頼まれた当の工程なら**頼まれた所だけ変える**、
      上流への頼みで作り直しているなら**前回の意図を保って合わせ直す**。人の編曲者の振る舞い。
    """
    rs = [r for r in active(requests) if reaches(r, step)]
    if not rs:
        return ""
    lines = ["【頼まれていること（台帳・作り直しても守る）】"]
    for r in rs:
        where = r.target + (f"（{r.scope}）" if r.scope else "")
        if r.origin == "鼻歌":
            lines.append(f"・[{r.id}] マスターが口ずさんだ手本（{where}へ）:\n{r.words}")
            continue
        if r.origin == "下調べ":                           # 作風の下調べ（tools/style.py）。複数行の1枚
            lines.append(f"・[{r.id}] 作風の手本（{where}へ）:\n{r.words}")
            continue
        lines.append(f"・[{r.id}] {where}を{OPS.get(r.op, '変える')}: 「{r.words}」"
                     f"（{r.origin}・{r.time}）")
        if r.target == step and r.prev:
            lines.append(f"    頼まれる前のこの工程の意図: {r.prev}")
    lines.append("※**この工程が頼まれた当の工程なら、頼まれた所だけ変え、ほかは前回の意図を保つ。**"
                 "上流への頼みのために作り直しているなら、**前回の意図を保ったまま新しい上流に合わせ直す。**")
    return "\n".join(lines)


def listing(requests: list[Request]) -> str:
    """コーディネーターに見せる一覧（生きている頼みと当て直し待ち・置き換えるときに id で指す）。"""
    rs = [r for r in requests if r.status in ("active", "reapply")]
    if not rs:
        return ""
    return "\n".join(f"  [{r.id}] {r.target}" + (f"（{r.scope}）" if r.scope else "")
                     + f"を{OPS.get(r.op, '変える')}: 「{r.words[:80]}」（{r.origin}・{r.time}）"
                     + ("  ★当て直し待ち" if r.status == "reapply" else "")
                     for r in rs)
