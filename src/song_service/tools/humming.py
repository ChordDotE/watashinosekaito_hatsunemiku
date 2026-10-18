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

"""**マスターが口ずさんだ音を読み取る**ツール（2026-09-23 新設）。

チャットに添付された鼻歌・口笛・肉声を音符に起こし、**1枚のテキスト**にして頼まれごとの台帳へ載せる。
載せ先は `song_requests`（宛先は**設計**・origin は「鼻歌」）——**人の頼みを工程へ届ける既存の仕組み**で、
`asks_for()` が読んで**設計から下の工程の依頼文に入る**。だから工程もデータの形も増やさない。

## 新規作成でも作り直しでも、同じ1枚を使う

  新規作成 … `compose_song` が走るとき台帳から設計以下の工程へ届く
  作り直し … `remake_from_step('旋律', request)` の `request` に同じ1枚を添える

## なぜ構造体で渡さないか

音符の一覧にすると**全部が同じ確からしさに見える**。実際は確かな音とそうでない音があり、
**その差が下流の判断材料**になる。テキストなら「ここは怪しい」「ここはしゃくり」を一緒に書ける。
本人:「**参照結果を全部テキストみたいな感じで渡せばいいのかなとも思った**」

権限レベル: 1（ローカルのファイルを読むだけ・外部送信なし）
副作用: 曲の台帳（`song_requests`）に1行足す。ほかは `ctx.deps.humming_sheet` に載せるだけ
"""
from __future__ import annotations

from pathlib import Path

from pydantic_ai import RunContext

from session import MusicSession, _trace, _traced
from coordinator import coordinator


@coordinator.tool
@_traced
def read_humming(ctx: RunContext[MusicSession], audio_path: str = "") -> str:
    """マスターが口ずさんだ音（鼻歌・口笛・肉声）を読み取って、旋律の手本として使えるようにする。

    `audio_path` は省略してよい（**届いている音を自動で使う**）。
    読み取った結果は**このあとの旋律工程へ自動で渡る**ので、呼んだあとは普通に曲を作ればよい。
    作り直すときは `remake_from_step('旋律', ...)` を呼べば、同じ手本が届く。

    ★**読み取りは完璧ではない**。口ずさみ自体がずれていることもあるので、
      **不自然になるところは変えてよい**——そう書いたテキストが渡る。
    """
    src = audio_path or ctx.deps.input_audio or ""
    if not src:
        return ("音が届いていない。マスターに『鼻歌を録って送って』とお願いするか、"
                "音なしで進めるなら普通に曲づくりを始めて。")
    p = Path(src)
    if not p.exists():
        _trace(ctx.deps.song_id, "read_humming.out", kind="error", reason="not_found")
        return f"その音が見つからない: {src}。マスターに送り直してもらって。"
    try:
        from humming import to_sheet, transcribe
        res = transcribe(p)
    except Exception as e:                                 # 壊れた音声でセッションを殺さない（error_handling 規約）
        _trace(ctx.deps.song_id, "read_humming.out", kind="error", reason=f"{type(e).__name__}")
        return (f"この音は読み取れなかった（{type(e).__name__}: {e}）。"
                f"形式が特殊か、壊れているかもしれない。音なしで進めるか、録り直してもらって。")
    if not res["notes"]:
        _trace(ctx.deps.song_id, "read_humming.out", kind="empty")
        return ("音は読めたけど、歌になっている部分が見つからなかった（無音か、声として拾えなかった）。"
                "もう少し大きめに、続けて口ずさんでもらうと採れる。")

    sheet = to_sheet(res, lyrics=_lyrics_if_voice(p, res))
    ctx.deps.humming_sheet = sheet
    _keep_with_song(ctx.deps.song_id, p, sheet)
    import song_context
    from song_requests import add as _add_request
    # ★**台帳に「鼻歌」として載せる**（2026-09-25・段階2）。以前は `CARRY`（宛先なしの申し送り）に
    #   入れていたので、作り直しのあとには消えていた。台帳なら曲の記録に残り、旋律を作り直しても手本が届く。
    #   宛先は**設計**——`CARRY` のときも設計から下の全工程に届いていた（テーマには届かない）。
    #   旋律だけにすると、設計が鼻歌の速さや長さを知らなくなる。
    if all(getattr(r, "words", "") != sheet for r in song_context.PENDING_REQUESTS):   # 二度呼ばれても重ねない
        _add_request(song_context.PENDING_REQUESTS, sheet, "設計", op="add", origin="鼻歌")
    v = res["voice"]
    _trace(ctx.deps.song_id, "read_humming.out", kind=v["kind"], notes=len(res["notes"]),
           sec=res["duration"], h2h1=v["h2h1"])
    return (f"{v['kind']}として読み取った（{res['duration']}秒・{len(res['notes'])}音）。"
            f"手本は旋律工程へ渡してあるので、このまま曲づくり／作り直しを進めてよい。\n\n{sheet}")


def _keep_with_song(song_id: str, src: Path, sheet: str) -> None:
    """**渡された音そのものと、読み取った手本を、曲のフォルダへ残す**（本人指示 2026-09-23）。

    本人:「**ファイルをそのまま保存するのがベストだけど、そうじゃないにしても、採譜などした
    データぐらいは全部ログとして残さないといけない**」——あとで編集するときに参照できるように。
    ★**潰す前の形で残す**（音符列に潰してから保存すると「そこは装飾だった」と言い直せない）。
    失敗しても本筋を止めない（手本は申し送りに載っていて、曲づくりは進められる）。
    """
    import shutil

    from session import SONGS

    try:
        d = SONGS / (song_id or "_unknown") / "source"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d / f"humming{src.suffix or '.bin'}")
        (d / "humming_sheet.md").write_text(sheet, encoding="utf-8")
    except Exception:
        pass


def _lyrics_if_voice(path: Path, res: dict) -> str:
    """**肉声のときだけ**歌詞を採る（鼻歌・口笛には言葉が無いので回さない）。

    ★**ひらがなで出させる**。読みの辞書は要らない——用途は「その語が音符を何個要求するか」を
      数えることで、**声を別の言葉へ置き換えることではない**。かなで返れば数えるだけで済む
      （実測でかな 42文字＝正解 42音ちょうど）。かな版は**そのまま歌う用の歌詞**にもなる。
    ★採れなくても失敗にしない（言葉なしの手本として使えば足りる）。
    ★**倍音の強さで当たりを付けてから回す**。文字起こしは 25秒の音に 25秒かかるので、
      鼻歌・口笛と分かっているなら回さない（回しても言葉は無く、幻覚だけが返る）。
    """
    if res["voice"]["kind"] != "肉声かもしれない":
        return ""
    try:
        from transcribe_vad import transcribe_vad
        words = transcribe_vad(str(path), language="ja")
    except Exception:
        return ""
    return " ".join(w["word"] for w in words).strip()
