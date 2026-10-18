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

"""自分の作った曲を歌う・流すツール（完成した曲をブラウザで再生する）。

`work_on_music`（曲づくり・15〜20分の Fire&Forget）と違い、**これは同期ツール**。
やることは「台帳を読む → 曲を決める → ブラウザへ予約を送る」だけで数ミリ秒で終わり、
**戻る時点で結果は確定している**（見つかった／送れた）。

ただし**実際に音が鳴るのはツールが返った後**。ミクの返事（「聴いてね」）の音声が流れ終わってから
ブラウザが鳴らし始める（songPlayer.js が `voice:end` で開始）。ツール内で鳴らすと返事と重なる。
そのため戻り値は「歌った」ではなく「**これから歌う**」と表す。

鳴らせるのは **完成品 `master_<形>_v{N}.wav` がある版だけ**（2026-10-03・名前の決まりは `song_service/song_files.py` の
1か所。本体からも同じファイルを読む）。歌もの・アカペラは `export_vocal` が VOCALOID6 の書き出しを仕上げて、
インストは `render_final` が演奏を仕上げて書く（V6 が無い環境ではマスターが手で書き出す）。仮歌（サイン波）は聴かせるものではないので対象にしない。

**曲の形で、何をするかが変わる**（戻り値の `mode`）:

| 形 | mode | ミク |
|---|---|---|
| 歌もの・アカペラ | `sing` | 自分の歌声で歌う（口も動く） |
| インスト（歌の入らない演奏だけの曲） | `play` | **歌わずに流す**（口は動かさない・「聴いてね」・流している間は歌と同じく話しかけない。F8①） |
| まだ歌になっていない作りかけ | `preview` | そこまでの音を聴いてもらう（歌わない・口も動かさない） |

★**まだ歌になっていない曲は、作りかけの音を聴かせる**（2026-09-23）。
曲づくりを途中で止めたとき（`draft_melody(show_after=...)`）、曲フォルダに `preview.wav` が置かれる。
鳴らせる版が1つも無くてもこれがあれば流す。歌える版があるときはそちらが優先（完成品のほうが聴きたいはず）。

**楽譜を直したあとの曲は、直す前の版で鳴らす**: 口の動きは楽譜を直すたびに作り直されるが、
完成品は書き出した（仕上げた）ときのまま。いまの版の完成品が無ければ、その版では鳴らさない
（口は新しい楽譜・歌は古い音源でズレる・handoff_2026-08-15_90cd9524.md §5-3）。

Orchestrator は**ツール名だけを選び引数は渡さない**ため、どの曲かはハンドラ側で解決する
（`runner._handle_play_song` が会話文脈を渡す）。

権限レベル: 1（ローカル読み取り＋ブラウザへの通知）
副作用: ブラウザで曲の再生が始まる（流れている間ミクのマイクはミュートされる）
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from agent.config import (get_song_service_dir, get_songs_dir, is_song_service_available,
                          load_settings)
from agent.host_bridge import get_song_playback_reserver
from agent.schemas import Failure, Success
# ★**完成の判定は song_service 側と同じファイルを読む**（2026-10-03・§7-1）。`song_files.py` は標準ライブラリだけで
#   書かれていて、本体の venv からも `miku/src` を起点に読める（`song_service` は名前空間パッケージ）。写しを作らない
from song_service import song_files

#: 曲の状態（songs.md の状態列）の言い方。**song_service の `session.DONE_LABEL`・`STALE_LABEL` と同じ言葉**
#: （本体は session を読めない〔pydantic の型と engine を引き込む〕ので写しを持つ。テストが両方を突き合わせる）
DONE_LABEL = {"song": "完成(実歌声)", "acappella": "完成(アカペラ)", "instrumental": "完成(インスト)"}
STALE_LABEL = {"song": "完成(編集あり・要再書き出し)", "acappella": "完成(編集あり・要再書き出し)",
               "instrumental": "完成(編集あり・要仕上げ直し)"}

_FORM_OF_LABEL = {v: k for k, v in song_files.FORM_LABEL.items()}


class PlaySongInput(BaseModel):
    hint: str = Field(default="", description="どの曲を聴かせてほしいかの手がかり（マスターの言葉＋直近の会話。"
                                              "曲名がはっきりしなくてよい。『さっきのバージョン』のような"
                                              "版の指定もそのまま入れてよい）")


class SongCandidate(BaseModel):
    """鳴らせる曲1つぶんの状態（選曲の判断材料）。"""
    song_id: str
    title: str
    latest_version: int = Field(default=0, description="台帳上の現在版（0=不明）")
    playable_versions: list[int] = Field(default_factory=list,
                                         description="完成品が実在する版＝実際に鳴らせる版")
    # ★**版ごとの曲の形**（2026-10-03）。選曲 LLM に「歌か演奏か」を見せる——「歌って」で演奏だけの版を
    #   選ばれると「歌う」と言って歌わないことになる。形は版ごとに違いうる（インストに歌を足した、など）
    version_forms: dict[int, str] = Field(default_factory=dict, description=(
        "鳴らせる版ごとの曲の形（歌もの＝歌う／アカペラ＝伴奏の無い声だけの曲を歌う／インスト＝歌の入らない演奏だけの曲を歌わずに流す）"))
    latest_form: str = Field(default="", description="いまの版の曲の形（歌もの／アカペラ／インスト）")
    # ★**版ごとの「いつ作った・何を直した」**（2026-10-11・本人了承）。版の番号だけでは「朝の版」「直す前の版」を当てられない
    version_notes: dict[int, str] = Field(default_factory=dict, description=(
        "版ごとの、作った日時と何を直したか（例: 10/10 23:24 13小節〜15小節の音符を置き直した）"))
    stale: bool = Field(default=False, description="いまの版の完成品がまだできていない（前の版なら鳴らせる）")
    # ★**まだ歌になっていない作りかけ**（2026-09-23）。`preview.wav` だけがある曲。
    #   選曲 LLM に**完成品と区別させる**ために出す——「あの曲歌って」で作りかけを選ばれると困るが、
    #   「さっきの続き聴かせて」では**これが正解**になる。判断できるように材料を渡す。
    in_progress: bool = Field(default=False, description=(
        "まだ歌になっていない作りかけ（途中まで作って手を止めた曲）。"
        "歌えないが、そこまでの音は聴いてもらえる"))


class PlaySongValue(BaseModel):
    # ★**「歌う」と「聴かせる」を言い分けられるようにした**（2026-09-23 本人指示）。
    #   以前は `now_singing: bool` だけで、**まだ歌になっていない音を流すときも「歌う」**に
    #   なってしまった。本人:「**今から歌うねって喋りながら、歌声がないやつを聞かせてくれるのは
    #   ちょっと違和感**」——ミクの中で別の出来事にならないと、言葉も自然にならない。
    #   ★2026-10-03: 歌の入らない曲（インスト）を流す `play` を足した（F8①）。
    mode: str = Field(description=(
        "これから何をするか。sing＝自分の歌声で歌う（口も動く）／"
        "play＝曲を流して、一緒に聴く（**歌わない・口も動かない**。ふつうは歌の入らない演奏だけの曲）／"
        "preview＝まだ歌になっていない、作りかけの音を聴いてもらう（歌わない・口も動かない）"))
    stage: str = Field(default="", description=(
        "preview のとき、どこまで出来ているか（例: 旋律まで / 伴奏まで）。"
        "**何が聴こえるかを自分の言葉で言うために使う**"))
    now_singing: bool = Field(description="これから歌う状態に入ったか（mode が play・preview のときは False）")
    title: str = Field(description="これから流す曲の名前")
    note: str = Field(description="マスターに伝えるべきこと（いつ始まるか・やめてほしい時どうするか）")


PlaySongResult = Success[PlaySongValue] | Failure

_NOTE = ("この返事をし終えたら曲が流れ始めて、自分は歌詞のとおりに口を動かして歌う。"
         "途中でやめてほしいときは、マイクのボタンを押すか、文字で話しかけてもらえれば歌をやめて聞く"
         "（歌っている間は自分の歌を拾わないようにマイクを切っているので、声で呼ぶならボタンを押してもらう）。")

#: ★**歌の入らない曲（インスト）を流すとき**（2026-10-03・F8①）。歌わない・口も動かさない・「聴いてね」。
#:   流している間は歌と同じく自分からは話しかけない（ブラウザ側も歌と同じ扱いでマイクを切る）
_NOTE_PLAY = ("この返事をし終えたら曲が流れ始める。**歌の入らない、演奏だけの曲（インスト）**なので、"
              "自分は歌わないし口も動かさない——「聴いてね」と、マスターと一緒に聴く。"
              "流れている間は、歌っているときと同じく自分からは話しかけない。"
              "途中でやめてほしいときは、マイクのボタンを押すか、文字で話しかけてもらえれば止めて聞く"
              "（流している間はマイクを切っているので、声で呼ぶならボタンを押してもらう）。")

#: ★**歌の入った曲なのに、口の動きのデータが無いとき**（2026-10-03）。Gateway が `sing` を `play` に落とす
#:   （口の形のファイル `lipsync_v{N}.json` が無いと口を動かせない）。「歌うね」と言って口が動かない食い違いを作らない
_NOTE_NO_LIP = ("この返事をし終えたら曲が流れ始める。自分の歌声が入った曲だけど、"
                "口の動きのデータが見当たらないので、今回は口を動かさずに流す——歌っているふりはせず、"
                "「聴いてね」と、マスターと一緒に聴く。"
                "流れている間は、歌っているときと同じく自分からは話しかけない。"
                "途中でやめてほしいときは、マイクのボタンを押すか、文字で話しかけてもらえれば止めて聞く"
                "（流している間はマイクを切っているので、声で呼ぶならボタンを押してもらう）。")


def _progress_stage(song_id: str) -> str:
    """途中で手を止めた曲の「どこまで作ったか」（止めた目印 `_progress.json`）。無ければ空。"""
    try:
        p = get_songs_dir() / song_id / song_files.PROGRESS
        return str(json.loads(p.read_text(encoding="utf-8")).get("stage", "")) if p.exists() else ""
    except Exception:
        return ""


def _score_form(song_dir: Path, version: int) -> str:
    """その版の楽譜（`score_v{N}.json` の `form`）の形。無い・読めないときは歌もの（古い曲は全部歌もの）。

    完成品がまだ無い版の形を知るときだけ使う（完成品があれば形は名前から読む）。
    song_service 側の `session.score_form` と同じ読み方（テストが突き合わせる）。
    """
    try:
        d = json.loads((song_dir / song_files.versioned(song_files.SCORE, version)).read_text(encoding="utf-8"))
        return song_files.as_form(d.get("form"))
    except Exception:
        return song_files.DEFAULT_FORM


def _is_stale(song_dir: Path, current: int, playable: list[int]) -> bool:
    """**いまの版の完成品がまだできていない**か（＝その版では鳴らせない）。

    2026-08-17 に判定を変えた。従来は「完成ミックスより楽譜が新しいか」で見ていたが、
    これは**書き出し済みかどうかを見ていなかった**。いまは「現在版 v{N} の完成品があるか」だけを見る。
    版番号が取れない古い曲（`_CURRENT_v*.txt` が無い）は、この判定を使わない。
    """
    if current <= 0:
        return False                                       # 版が分からない旧曲は従来どおり扱う
    return current not in playable


def _versions_with_audio(song_dir: Path) -> dict[int, str]:
    """**実際に鳴らせる版** → その形（`song`／`acappella`／`instrumental`）。

    完成品は `master_<形>_v{N}.wav`（名前の決まりは `song_files` の1か所。song_service 側の
    `session._song_status`・`song_runner`・`handoff`・Gateway の配信許可と同じ判定）。
    V6 の生の書き出し（`vocal_v{N}.wav`）や、昔の完成ミックスの写しは見ない——仕上げ前の音だったり、
    固定名からの写しで名前と中身が食い違ったりする（2026-08-17 に全部の版が同じ中身だった）。
    """
    return {m.version: m.form for m in song_files.masters(song_dir)}


def _current_version(song_dir: Path) -> int:
    """`_CURRENT_v{N}.txt` から現在版を読む（無ければ 0）。"""
    for p in song_dir.glob("_CURRENT_v*.txt"):
        m = re.fullmatch(r"_CURRENT_v(\d+)\.txt", p.name)
        if m:
            return int(m.group(1))
    return 0


def _singable(songs_dir: Path) -> list[SongCandidate]:
    """曲ごとの状態を返す（鳴らせる／仕上げ直しが要る／どの版が鳴らせるか・形）。

    実在するファイルで判定する（台帳の表記は当てにしない）。**stale な曲も落とさずに返す**
    ——落とすと「直したから書き出し直してほしい」と言えず、別の曲を歌ってしまう
    （2026-08-16 に実際に起きた。名指しされた曲が除外されて候補が1つになり、
    曲名を照合しないまま無関係な曲が鳴った）。

    ★**作りかけ（`preview.wav` だけ）も返す**（2026-09-23 追加）。ここで落としていたので、
    下の「作りかけを流す」分岐に**一度も到達していなかった**——`draft_melody(show_after=...)` で
    途中まで作っても、**聴かせる手段が無かった**。機能そのものが成立しない穴で、
    **単体テストでは出ない**（UI 側のテストは分岐に入った後だけを見ていた）。
    """
    out = songs_dir                                        # **曲の置き場**（サービスの実体とは別）
    index = out / "songs.md"
    cands: list[SongCandidate] = []
    if not index.exists():
        return cands
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| 2"):                     # 曲の行（song_id が日時始まり）だけ
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 2:
            continue
        song_id, title = cols[0], cols[1]
        song_dir = out / song_id
        current = _current_version(song_dir)
        forms = _versions_with_audio(song_dir)
        playable = sorted(forms)
        in_progress = not playable and (song_dir / song_files.PREVIEW).exists()
        if not playable and not in_progress:
            continue                                       # 鳴らせる音が1つも無い＝まだ聴かせられない
        latest = forms.get(current) or (_score_form(song_dir, current) if current else
                                        (forms[max(forms)] if forms else song_files.DEFAULT_FORM))
        cands.append(SongCandidate(
            song_id=song_id, title=title,
            latest_version=current,
            playable_versions=playable,
            version_forms={v: song_files.FORM_LABEL[f] for v, f in forms.items()},
            latest_form=song_files.FORM_LABEL[latest],
            version_notes={e["v"]: song_files.version_note(e) for e in song_files.version_log(song_dir)},
            stale=_is_stale(song_dir, current, playable),
            in_progress=in_progress,
        ))
    return cands


def _refresh_index_status(songs_dir: Path, cands: list[SongCandidate]) -> None:
    """台帳（songs.md）の状態列を実ファイルに合わせ直す。

    完成品が曲フォルダに置かれても（マスターが手で V6 の書き出しを置いたときなど）、**台帳は古いまま**になる。
    更新するのは `adopt_vocal_export` だが、「書き出したから歌って」と言われた流れでは
    そのツールを通らずに歌うため（2026-08-18 実機。歌自体は実ファイルを見るので正しく鳴り、
    ずれていたのは台帳の表示だけだった）。歌う前に必ずここを通るので、そのついでに直す。

    触るのは**状態列だけ**。曲名・版・更新日時は song_service が書いたものを尊重する。
    言い方は形ごと（`DONE_LABEL`・`STALE_LABEL`＝song_service の `session` と同じ言葉）。
    """
    index = songs_dir / "songs.md"
    if not index.exists() or not cands:
        return
    # ★**作りかけは触らない**（2026-09-23）。ここへ来る候補に作りかけが混ざるようになったので、
    #   除かないと**まだ歌になっていない曲の状態列に「完成(実歌声)」と書いてしまう**。
    #   状態は song_service が書いたものを尊重する（この関数は書き出しの遅れを直すためだけにある）。
    want = {}
    for c in cands:
        if c.in_progress:
            continue
        latest = _FORM_OF_LABEL.get(c.latest_form, song_files.DEFAULT_FORM)
        if c.stale:
            want[c.song_id] = STALE_LABEL[latest]
        else:
            shown = c.version_forms.get(c.latest_version) or c.version_forms.get(max(c.playable_versions))
            want[c.song_id] = DONE_LABEL[_FORM_OF_LABEL.get(shown, latest)]
    if not want:
        return
    try:
        lines = index.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    changed = False
    for i, line in enumerate(lines):
        if not line.startswith("| 2"):                     # 曲の行（song_id が日時始まり）だけ
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 5 or cols[0] not in want or cols[2] == want[cols[0]]:
            continue
        cols[2] = want[cols[0]]
        lines[i] = "| " + " | ".join(cols) + " |"
        changed = True
    if changed:
        try:
            index.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            pass                                           # 台帳が書けなくても歌えるので致命ではない


def _normalize(s: str) -> str:
    """照合用の正規化（記号・空白を落とす。曲名は区間名などが付くため緩く突き合わせる）。"""
    return re.sub(r"[\s　_・「」『』\"'（）()]+", "", s).lower()


def _pick(cands: list[SongCandidate], hint: str) -> tuple[list[SongCandidate], str]:
    """文字列マッチで曲を選ぶ（**選曲 LLM が使えないときのフォールバック**）。

    かつては「歌える曲が1つなら名前を照合せずに歌う」としていたが、**名指しされた曲が
    書き出し待ちで除外されると無関係な曲が鳴る**という事故を起こした（2026-08-16）。
    そのため、手がかりがあるのに一致しないときは**決めない**。
    """
    h = _normalize(hint)
    hit = [c for c in cands if _normalize(c.title) and _normalize(c.title) in h]
    if hit:
        return hit, "曲名が会話に出ていた"
    # 曲名が長い（区間名つき等）と埋もれるので、逆方向（曲名の頭から数文字が手がかりに含まれるか）も見る
    loose = [c for c in cands if len(_normalize(c.title)) >= 3 and _normalize(c.title)[:4] in h]
    if loose:
        return loose, "曲名の一部が会話に出ていた"
    if len(cands) == 1:
        # 曲そのものが1つしか無いなら「歌って」で通じる。
        # **2026-08-16 の事故は「stale な曲を候補から落として1曲に縮めた」ことが原因**で、
        # いまは _singable が stale も返す＝名指しされた曲は必ず候補に残る。
        # そのうえで stale なら後段が「書き出し直しが要る」と正しく断る
        return cands, "曲が1つだけ"
    return [], "手がかりから特定できなかった"


def _select_with_llm(hint: str, cands: list[SongCandidate]) -> tuple[SongCandidate | None, int | None, str]:
    """選曲 LLM に委ねる。戻り値は (曲, 版, 理由)。使えなければ (None, None, "") を返す。

    非同期関数を同期ツールから呼ぶため、**専用スレッドの asyncio.run** で回す
    （Gateway 側に既存のイベントループがあり、ここで run すると衝突するため）。
    """
    import asyncio
    import concurrent.futures

    from agent.subagents.song_select import run_song_select

    payload = [c.model_dump() for c in cands]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            out = ex.submit(lambda: asyncio.run(run_song_select(hint, payload))).result(timeout=20)
    except Exception as e:
        print(f"[play_song] 選曲 LLM を使えず文字列マッチへ: {type(e).__name__}: {e}")
        return None, None, ""
    if not out.song_id:
        return None, None, out.reason
    picked = next((c for c in cands if c.song_id == out.song_id), None)
    if picked is None:                                     # 実在しない song_id を返してきた
        return None, None, ""
    return picked, out.version, out.reason


def _finish_how(form: str, version: int) -> str:
    """その版を完成させるには何が要るか（形ごと。**インストでは VOCALOID の話をしない**・F8）。"""
    if form == "instrumental":
        return (f"演奏の仕上げは自分でできる（work_on_music で『v{version} の演奏を仕上げて』と頼む）。"
                "歌の入らない曲なので、VOCALOID もマスターの手も要らない。")
    return (f"歌声の書き出しは自分でできる（work_on_music に『v{version} の歌声を書き出して』と渡す）。"
            f"それが失敗したときだけ、マスターに {song_files.versioned(song_files.VPR, version)} を VOCALOID6 で開いて"
            "ミックスダウンし、そのままの名前で保存してもらう。")


def run_play_song(input_data: PlaySongInput) -> PlaySongResult:
    settings = load_settings()
    if not is_song_service_available(settings):
        return Failure(reason="いまは曲を歌える状態になっていない（設定でオフ、または準備が未完了）")

    songs_dir = get_songs_dir()                            # ★曲の置き場（2026-09-01〜 `miku/data/songs`）
    cands = _singable(songs_dir)
    _refresh_index_status(songs_dir, cands)                      # 書き出しが増えていれば台帳を実態へ合わせる
    if not cands:
        return Failure(reason=(
            "まだ聴かせられる曲が1つも無い。曲そのものは作れていても、完成品がまだできていない"
            "（歌の入る曲は歌声の書き出し、歌の入らない曲〔インスト〕は演奏の仕上げで完成する）。"
            "どちらも自分でできる（work_on_music で『この曲を仕上げて』と頼む）。"
            "歌の入る曲で歌声の書き出しが失敗したときだけ、マスターに vocal_v{N}.vpr（版番号つき。N はその曲の最新版）を"
            "VOCALOID6 で開いてミックスダウンし、曲のフォルダに保存してもらう（名前は既定のままでいい）。"))

    # ① 選曲は LLM に委ねる（指示語・版指定を解釈させる）。使えなければ文字列マッチへ
    picked, version, why = _select_with_llm(input_data.hint, cands)
    if picked is None:
        matched, why_str = _pick(cands, input_data.hint)
        if len(matched) == 1:
            picked, version = matched[0], None
        else:
            names = "／".join(_describe(c) for c in cands)
            return Failure(retryable=True, reason=(
                f"どの曲を流せばいいか決められなかった（{why or why_str}）。"
                f"いまの状況: {names}。どれがいいかマスターに聞く。"))

    # ② 版の指定が無ければ「鳴らせる版のうち最新」を選ぶ。
    #    固定名を見ないので、**名前と中身の食い違い**（固定名の写しが古い、等）に触れない
    if version is None and picked.playable_versions:
        version = max(picked.playable_versions)

    # ②' ★**まだ歌になっていない曲は、作りかけの音を聴かせる**（2026-09-23）。
    #    途中で手を止めたとき（`draft_melody(show_after=...)`）に `preview.wav` が置かれる。
    #    歌える版が1つも無いのにここで断ると、**せっかく作った途中経過を聴いてもらえない**。
    #    歌える版があるならそちらが優先（完成品のほうが聴きたいはず）。
    if not picked.playable_versions:
        prev = get_songs_dir() / picked.song_id / song_files.PREVIEW
        if prev.exists():
            reserve = get_song_playback_reserver()
            if reserve is None:
                return Failure(reason=("聴かせる仕組みが用意されていない（ミクの本体が動いていないときなど）。"
                                       "マスターに画面から起動してもらう。"))
            if not reserve(picked.song_id, picked.title, None,
                           audio=song_files.PREVIEW, mode="preview"):
                return Failure(retryable=True, reason=(
                    "届ける先が見つからない（マスターのブラウザが開いていないみたい）。"
                    "画面を開いてもらってから聴かせる。"))
            stage = _progress_stage(picked.song_id)
            return Success(
                value=PlaySongValue(
                    mode="preview", stage=stage, now_singing=False, title=picked.title,
                    note=("この返事をし終えたら、**作りかけの音**が流れる。"
                          "**まだ歌詞は乗っていない**ので、自分は歌わないし口も動かさない"
                          "——「ここまで作ったところ」を聴いてもらう。"
                          "聴いたうえで、**このまま進めていいか・直したいところがあるか**を聞くこと。"
                          "途中でやめてほしいときは、マイクのボタンか文字で話しかけてもらう。")),
                side_effects=[f"ブラウザで『{picked.title}』の作りかけを再生する"],
                summary=f"『{picked.title}』の{stage or '作りかけ'}を聴いてもらう")

    song_dir = get_songs_dir() / picked.song_id
    latest_form = _FORM_OF_LABEL.get(picked.latest_form, song_files.DEFAULT_FORM)

    # ③ いまの版の完成品がまだできていないなら、鳴らさずにそれを伝える。
    #    **別の曲に振り替えない**（宣言と違う曲が鳴る事故・2026-08-16）
    if version is None and picked.stale:
        have = "／".join(f"v{v}" for v in picked.playable_versions)
        return Failure(reason=(
            f"『{picked.title}』は作り直したばかりで（いまは v{picked.latest_version}）、"
            f"その版の完成品がまだできていない。"
            + (f"聴けるのは {have} まで。" if have else "")
            + _finish_how(latest_form, picked.latest_version)))

    # ④ 版を指定されたら、その版の完成品が実在するかを確かめる
    if version is not None and version not in picked.playable_versions:
        have = "／".join(f"v{v}" for v in picked.playable_versions) or "（1つも無い）"
        # ★2026-09-05: 書き出しは `export_vocal` が自分でやるようになったので、
        #   「マスターに書き出してもらう」前提の文面をやめた。
        #   ただし **V6 が無い環境では自分で書き出せない**（そのときは song_service が
        #   その旨を返してマスターに頼む形に落ちる）ので、ここでは「用意する」と言うに留める
        #   ——できると断言してから謝るより、やってみて報告するほうが誠実。
        form_v = _score_form(song_dir, version)
        kept = (f"版ごとの楽譜（{song_files.versioned(song_files.SCORE, version)}）"
                + ("" if form_v == "instrumental"
                   else f"と VOCALOID のデータ（{song_files.versioned(song_files.VPR, version)}）"))
        return Failure(reason=(
            f"『{picked.title}』の v{version} は、まだ音楽ファイルが見当たらない。"
            f"{kept}は残っているから、**仕上げればいつでも聴ける**の。"
            f"いま鳴らせるのは: {have}。その版を聴きたいなら言って——**用意するね**。"))

    # 鳴らす版の形は**完成品の名前**から読む（版ごとに形が違いうる・v1 インスト → v2 歌もの など）
    got = song_files.master_of(song_dir, version) if version is not None else None
    form = got.form if got else song_files.DEFAULT_FORM
    play_only = form == "instrumental"

    # 再生の予約はブラウザ接続を持つ Gateway の仕事。**gateway_service を import してはいけない**
    # （Gateway は `__main__` として起動されるので再 import すると状態が空の別インスタンスになり、
    #  繋がっていても「繋がっていない」と判定される。2026-08-15 の実害・host_bridge の docstring 参照）
    reserve = get_song_playback_reserver()
    if reserve is None:
        return Failure(reason=("曲を届ける仕組みが用意されていない（ミクの本体が動いていないときなど）。"
                               "マスターに画面から起動してもらう。"))
    sent = (reserve(picked.song_id, picked.title, version, mode="play") if play_only
            else reserve(picked.song_id, picked.title, version))
    if not sent:
        return Failure(retryable=True, reason=(
            "曲を届ける先が見つからない（マスターのブラウザが開いていないみたい）。"
            "画面を開いてもらってから流す。"))
    # ★**予約の関数が実際に決めた mode で言い方を選ぶ**（2026-10-03）。歌もの・アカペラでも口の形のファイルが
    #   無ければ Gateway は流すだけ（play）に落とす。頼んだ mode で「歌うね」と言うと、口が動かない食い違いになる
    #   （mode を返さない古い実装〔True〕なら頼んだとおり）
    mode = sent if sent in ("sing", "play") else ("play" if play_only else "sing")

    label = f"『{picked.title}』" + (f"の v{version}" if version else "")
    note = _NOTE if mode == "sing" else (_NOTE_PLAY if play_only else _NOTE_NO_LIP)
    if form == "acappella":
        note += "\nこの曲は伴奏の無い、声だけの曲（アカペラ）。"
    if version is not None and picked.latest_version and version < picked.latest_version:
        # 最新版はまだ完成品ができていないので、ひとつ前の版で鳴らす。
        # 音と口は同じ版で揃うのでズレないが、**最新ではないことは伝える**
        how = ("演奏の仕上げ" if latest_form == "instrumental"
               else f"{song_files.versioned(song_files.VPR, picked.latest_version)} の書き出し")
        note += (f"\nいま流すのは v{version}。最新の v{picked.latest_version} は"
                 f"まだ完成品ができていないので、ひとつ前の版にしている。"
                 f"最新で聴きたいなら{how}が要る、とマスターに一言伝える。")
        if got and latest_form != form:
            note += (f"（v{version} は{song_files.FORM_LABEL[form]}、最新の v{picked.latest_version} は"
                     f"{song_files.FORM_LABEL[latest_form]}——形が違うことも伝える）")
    return Success(
        value=PlaySongValue(mode=mode, now_singing=mode == "sing", title=picked.title, note=note),
        side_effects=[f"ブラウザで{label}の再生を開始する"],
        summary=f"これから{label}を" + ("歌う" if mode == "sing" else "流す"),
    )


def _describe(c: SongCandidate) -> str:
    """候補1件を「状態つき」で説明する（聞き返すときに何が問題かが伝わるように）。"""
    plain = all(f == song_files.FORM_LABEL["song"] for f in c.version_forms.values())
    vers = "／".join(f"v{v}" + ("" if plain else f"（{c.version_forms.get(v, '')}）")
                    for v in c.playable_versions)
    if c.in_progress:
        # ★「歌える・版: 」と出さない（版が無いので空欄になり、何のことか伝わらない）
        return f"{c.title}（まだ作りかけ・そこまでの音なら聴いてもらえる）"
    if c.stale:
        # 最新版はまだ完成品ができていないが、過去の版なら聴ける、という状態があり得る
        return (f"{c.title}（いま v{c.latest_version}・その版はまだ仕上げ前"
                + (f"。聴けるのは {vers}" if vers else "") + "）")
    if c.version_forms and all(f == song_files.FORM_LABEL["instrumental"] for f in c.version_forms.values()):
        return f"{c.title}（歌の入らない演奏だけの曲・流せる版: {vers}）"
    return f"{c.title}（歌える・版: {vers}）"
