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

"""セッション状態・トレース・直列化ロック・永続化（版管理/台帳/掃除）— coordinator/tools 共通の土台。
PoC coordinator_agent.py からの挙動同一の清書（対応は docs/plan/song_generation_study.md §6.3）。"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import song_files
from song_schema import SongScore

# ★**曲の置き場**（2026-09-01 に `miku/data/songs` へ移した）。
#   ここは `Path("output")` という**相対パス**を9箇所で使っており、曲の本体だけが
#   新しい場所へ移って**ログと台帳が取り残されていた**（実機で発覚）。
#   相対パスは「どこから起動したか」で行き先が変わるので、**絶対パスで1つ持つ**。
SONGS = Path(__file__).resolve().parents[2] / "data" / "songs"

#: ★**楽譜を書き直したあとの「次にやること」は、この1文だけから出す**（2026-09-26）。
#:   歌声の書き出しは 2026-09-05 から `export_vocal` が VOCALOID6 を自動で操作する。ところが道具の返事・
#:   コーディネーターの指示・申し送りの指示の**あちこちに「V6 でミックスダウンしてもらう」が残り**、
#:   書き出しが済んだ曲なのに、ミクがマスターへ「VOCALOID6 で書き出して」とメールした（『明日への足音』v1）。
#:   **マスターに手で頼むのは `export_vocal` が失敗したときだけ**（その文は `export_vocal` の失敗の返事が持つ）。
#:   文言の置き場の一覧は docs/components/song_service.md「歌声の書き出しは自動」。
#:   `tests/song_service/test_export_wording.py` が、ほかの場所に手の案内が戻ってきたら落とす。
NEXT_EXPORT = ("次は **export_vocal** で歌声を書き出す（VOCALOID6 を自動で操作する。マスターの手は要らない）。"
               "書き出したら曲は完成＝play_song で聴かせられる。")

#: ★**形ごとの「次にやること」**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §7-1）。
#:   アカペラも歌声を V6 で書き出すので歌ものと同じ1文。**インストには歌声の書き出しが無い**
#:   （完成品は `render_final` が伴奏を仕上げて書く）ので、V6 の話をしない。
NEXT_BY_FORM = {
    "song": NEXT_EXPORT,
    "acappella": NEXT_EXPORT,
    "instrumental": ("演奏だけの曲（インスト）なので歌声の書き出しは無い。完成品は **render_final** が演奏を仕上げて作る"
                     "（楽譜を直したら render_final をもう一度）。仕上がれば曲は完成＝play_song で流して聴かせられる。"),
}


def next_step(form: str | None) -> str:
    """楽譜を書き直したあとの「次にやること」（形ごと・空＝歌もの）。"""
    return NEXT_BY_FORM[song_files.as_form(form)]


#: ★**曲の状態（songs.md の状態列）の「完成」の言い方**（形ごと）。`play_song._refresh_index_status` が
#:   同じ言葉で台帳を直す（本体側の写しと突き合わせるテスト: `tests/tools/test_play_song.py`）
DONE_LABEL = {"song": "完成(実歌声)", "acappella": "完成(アカペラ)", "instrumental": "完成(インスト)"}
#: いまの版の完成品がまだ無い（前の版なら聴ける）ときの言い方。インストは歌声の書き出しではなく演奏の仕上げ
STALE_LABEL = {"song": "完成(編集あり・要再書き出し)", "acappella": "完成(編集あり・要再書き出し)",
               "instrumental": "完成(編集あり・要仕上げ直し)"}
#: ★**どの版にも完成品がまだ無い曲**の言い方（2026-10-07）。歌もの・アカペラは V6 で vpr を書き出せば完成
#:   （V6 の無い環境・書き出しに失敗したとき）。インストは V6 を使わないので「書き出し待ち」にしない。
#:   「生成中」と書かない——台帳に載るのは作業が終わった版で、ミクとマスターが「まだ作っている」と読む
AWAIT_LABEL = {"song": "歌声の書き出し待ち(VOCALOID6 で {vpr} を開いて書き出す)",
               "acappella": "歌声の書き出し待ち(VOCALOID6 で {vpr} を開いて書き出す)",
               "instrumental": "演奏の仕上げ待ち(完成品がまだ無い・仕上げ直しが要る)"}
#: 歌もの・アカペラで楽譜はあるが vpr がまだ無い（書き出しの前に止まった）
NO_VPR_LABEL = "書き出し前(楽譜だけ・vpr がまだ無い)"


def score_form(song_dir: Path, version: int) -> str:
    """その版の楽譜（`score_v{N}.json`）の形。楽譜が無い・欄が無い・読めないときは歌もの（古い曲は全部歌もの）。"""
    try:
        d = json.loads((song_dir / song_files.versioned(song_files.SCORE, version)).read_text(encoding="utf-8"))
        return song_files.as_form(d.get("form"))
    except Exception:
        return song_files.DEFAULT_FORM


@dataclass
class MusicSession:
    instruction: str
    song_id: str
    score: SongScore | None = None
    reference_json: str | None = None
    reference_midi: str | None = None        # 参照MIDIパス（忠実コピー用）
    melody_channel: int | None = None        # 参照MIDIの主旋律ch
    reference_audio_accomp: str | None = None  # 音声参照: demucs分離の実楽器伴奏wav（忠実コピー用）
    faithful: bool = False                    # 忠実コピーで作ったか（render時に伴奏を再生成しない）
    section: dict | None = None               # 工程間の中間状態（draft_melody が作る区間情報＋追跡ログ）
    lyrics: str | None = None                 # fetch_section_lyrics が取得した区間かな歌詞
    accomp_bounds: tuple | None = None        # 伴奏を最後に切り出した区間(start_sec,end_sec)。align再実行で区間が変わったら stale 検知用
    review_breaks: int = 0                    # 工程検査で「破綻あり」が出た累計（3回で自動修復を打ち切り安全停止へ＝ループ防止の命綱）
    recent_calls: list = field(default_factory=list)  # 直近ツールコール(名前,引数)＝同一引数の連続再実行ブロック用（§5.4b-2c）
    # ★**読み込んだ時点の楽譜**（2026-09-20）。命綱（coordinator の末尾）が
    #   「編集したのに書き出していない」と「読んだだけ」を区別するために使う。
    #   これが無かったころ、**調べただけのセッションで命綱が発火し、
    #   読んだ曲の vpr・伴奏・仮歌・口の動きを作り直した**（『ショートさせ』で実害）。
    #   作ったときの歌い方の選択はメモリにしか無いので、作り直すと失われる。
    #   新規作成では None のまま＝必ず書き出す（従来どおり）。
    score_at_load: str | None = None
    # ★**マスターが口ずさんだ音**（2026-09-23）。チャットに添付された音声のローカルパスと、
    #   それを読み取った**1枚のテキスト**（`engine/humming.to_sheet`）。
    #   **構造体で持たない**——音符の一覧にすると全部が同じ確からしさに見えてしまい、
    #   「この音は確か・ここは怪しい・ここはしゃくり」という差が消える。差こそが判断材料。
    #   渡し方は `song_context.CARRY`（前工程からの申し送り）に載せるだけ＝既存の仕組みに乗る。
    input_audio: str | None = None
    humming_sheet: str | None = None
    # ★**過去版を開いたときの、その版の番号**（2026-09-25・段階3）。`load_existing_song(version=N)` は
    #   楽譜を非破壊で読むだけで作業台（固定名）を書き換えない。記録も同じにして、
    #   **最初に記録を書くときに `record_v{N}.json` から始める**（書いたら None に戻す）。
    record_version: int | None = None
    # ★**曲フォルダに付けたいのに付けられなかった曲名**（2026-10-03）。曲名の作り直しで曲名が変わったのに、中のファイルを
    #   開いているもの（再生中・V6・ログを読み続けている）があってフォルダ名を変えられなかったときに覚えておき、
    #   次の書き出し（`render_final` / `export_vocal`・`_name_folder` を通る所）と会話の終わりにもう一度試す（`settle_folder_title`）
    folder_title: str = ""
    # ★**編集のあとの最終の点検**（2026-10-04・案C・`tools/editing.review_edits`）。大・中の編集の道具は、終わりにここへ1件積む
    #   （`tools/editing.note_edit`・区分の表は `tools/editing.EDIT_SIZE`）。書き出しの入口（`export_vocal`・インストの `render_final`
    #   と伴奏の編集）でまとめて1回最終の点検役を呼び、束を空にする。書き出さずに実行が終わったら `coordinator.run` が版の確定の前に呼ぶ。
    #   1件は dict（tool・size・what）。回数（`edit_gate_runs`）は1回の頼み（＝この実行＝1版）で最大2回
    pending_edits: list = field(default_factory=list)
    edit_gate_runs: int = 0
    edit_gate_log: list = field(default_factory=list)      # この実行で出した判断の文（2回目の点検へ「前に出した判断」として渡す）
    run_request_ids: list = field(default_factory=list)    # この実行で台帳に積んだ行の id（依頼文の【この編集】の印）
    edit_base_version: int | None = None                   # 編集の元にした版（`load_existing_song` が開いた版。前の版の伴奏 MIDI を引く）


def load_record(song_dir: Path, title: str = "", version: int | None = None):
    """曲の工程の記録（`Song`）を読む。無ければ None。

    探す順: 指定した版 `record_v{N}` → 作業台 `record.json` → 現在版 `record_v{cur}`
    → **途中保存** `_wip/<曲名>`（途中で止めた曲・落ちた曲。曲フォルダの記録は最後まで作れたときだけ書かれる。版の区別は無い）。
    """
    from song_model import Song
    cands = []
    if version:
        cands.append(song_dir / f"record_v{version}.json")
    cands.append(song_dir / "record.json")
    cur = _current_version(song_dir) if song_dir.exists() else 0
    if cur:
        cands.append(song_dir / f"record_v{cur}.json")
    for p in cands:
        if p.exists():
            return Song.load_record(p)
    if title:
        # 途中保存の名前は曲名の slug（2026-10-03・`song_flow._autosave`）。それより前の途中保存は曲名そのままなので後で見る
        for name in dict.fromkeys((song_files.slug(title), title)):
            if not name:
                continue
            legacy = song_dir.parent / "_wip" / f"{name}.json"   # 曲フォルダの親＝SONGS
            try:
                found = legacy.exists() or legacy.with_suffix(".steps.json").exists()
            except (OSError, ValueError):                  # 曲名そのまま（`?`・`:` 入り）はファイル名として引けない
                continue
            if found:
                return Song.load(legacy)
    return None


def save_record(song_dir: Path, song) -> None:
    """作業台の記録 `record.json` を書く（版は `end_session` が楽譜と一緒に切る）。"""
    song.save_record(song_dir / "record.json")

def _append_log(song_id: str, text: str) -> None:
    """**全ログを1本に統一**＝曲フォルダの作業記録（`song_files.BUILD_LOG`＝`log_build.txt`）へ即時追記（時系列・機械的・クラッシュ耐性）。
    工程の入出力（_trace）・カバー内部トレース・セッション要約をすべてここに集約する。"""
    try:
        d = SONGS / (song_id or "_unknown")
        d.mkdir(parents=True, exist_ok=True)
        with (d / song_files.BUILD_LOG).open("a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass

def _trace(song_id: str, tag: str, **fields) -> None:
    """各ツールの入出力・途中経過を作業記録（`log_build.txt`）へ即時追記（機械的・LLM不使用・工程ごとに即フラッシュ）。"""
    parts = "  ".join(f"{k}={v}" for k, v in fields.items())
    _append_log(song_id, f"[{datetime.now():%H:%M:%S}] {tag}  {parts}")

_TOOL_LOCK = threading.Lock()
"""全ツール直列化ロック。LLMが1応答で複数ツールを並列要求するとスレッドプールで同時実行され、
共有ファイル（vocal.vpr の書込→読直し等）と deps.score が競合する
（T10実測: replace_region×5並列 → 書きかけの vpr を別スレッドが読んで BadZipFile）。
読み書きの別を問わず単一ロックで直列化する（2026-07-09）。"""

def _repeat_guard(ctx, name: str, arg_s: str) -> str | None:
    """同一ツール・同一引数の**3連続**呼び出しを検知して人間語で打ち切る（§5.4b-2c・結果が変わらない
    再試行ループの命綱）。引数が変わる正当な繰り返し（区間ごとの replace 等）は妨げない。
    ※回数上限方式にしなかった理由: アレンジでは同一ツールを数十回呼ぶのが正当（T13実測98回）。"""
    sig = (name, arg_s)
    rc = ctx.deps.recent_calls
    if len(rc) >= 2 and rc[-1] == sig and rc[-2] == sig:
        _trace(ctx.deps.song_id, f"{name}.blocked", reason="同一引数の3連続再実行")
        return (f"同じ操作（{name}・同じ引数）を3回続けようとしているよ。同じやり方を繰り返しても結果は"
                "変わらないから、一旦止めるね。引数や対象区間・工程を変えるか、ここまでの成果でマスターに報告して終了して。")
    # draft_melody だけは**引数不問**の連打も止める（区間の言い回しを変えながら連打して
    # request_limit を使い切った実害・2026-08-09）。他ツールに広げないのは、アレンジでは
    # 同一ツールの正当な連続呼び出しが数十回ある（T13実測98回）ため
    if name == "draft_melody" and len(rc) >= 3 and all(t == "draft_melody" for t, _ in rc[-3:]):
        _trace(ctx.deps.song_id, f"{name}.blocked", reason="引数を変えた4連続の区間探し直し")
        return ("区間の探し直し（draft_melody）が4回続いているよ。直前の結果の『元曲の◯〜◯秒』を見て、"
                "指定との差が2秒程度までならそのまま fetch_section_lyrics へ進んで。"
                "大きくズレるなら『60秒から88秒』のように秒で明示指定して1回だけやり直す。"
                "それでも合わなければ、ここまでの状況をマスターに報告して終了して。")
    rc.append(sig)
    del rc[:-6]                                            # 直近だけ保持
    return None


def _locked(fn):
    """全ツール直列化デコレータ（_TOOL_LOCK）。_traced はロック内蔵なので併用しない（非再入・二重取得禁止）。"""
    import functools

    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        with _TOOL_LOCK:
            arg_s = ", ".join([repr(a)[:60] for a in args] + [f"{k}={repr(v)[:60]}" for k, v in kwargs.items()])
            blocked = _repeat_guard(ctx, fn.__name__, arg_s)
            if blocked:
                return blocked
            return fn(ctx, *args, **kwargs)
    return wrapper

#: **検査ごとの「見ているもの／見ていないもの」**（2026-09-20 本人指示）。
#:
#: ★★**OK の射程が分からないと、LLM は全体のお墨付きとして受け取る。**★★
#: 本人:「チェック用の関数が OK を返したとして、**それがどの観点については OK で、
#: 別にどの点については見てないのか**が分からないとだめ。チェックごとに
#: **何を見ていて、その範囲では OK なのか警告なのか NG なのか**を明示しないと LLM は分からない」。
#:
#: 実際に起きたこと（同日）: `review_progress` が「所見: 健全」、`check_consistency` が
#: 「食い違いなし。このまま進んでよい」と返した曲に、**早口もハモリの浮きも伴奏の復活も在った**。
#: どの検査も**自分が見た範囲**のことしか言っていないのに、**全体の判定として読まれた**。
#:
#: ★**ここに無い検査ツールはテストで落ちる**（`tests/song_service/test_check_scope.py`）。
#: 検査を足したら必ず1行足す——「何を見ているか」を言えない検査は、そもそも設計が曖昧。
_CHECK_SCOPE: dict[str, tuple[str, str]] = {
    # ツール名: (見ているもの, 見ていないもの)
    "review_progress": (
        "材料の揃い方（参照・区間・歌詞）と、歌の密度・音符の長さ・音域といった数値が常識の範囲か",
        "言葉と旋律の合い方・伴奏の刻み・音量の起伏・ハモリの重なり方"
        "（数値が常識内でも、耳で変なことは普通にある。→ inspect_song()）"),
    "check_consistency": (
        "音符とモーラの数／歌と伴奏の尺（区間の地図と照らす）／ハモリの音域／タイミング",
        "言葉と旋律の合い方・伴奏の刻み・音量の起伏 → inspect_song()"),
    "evaluate_song": (
        "拍のグリッドからの外れ・音符の重なり・歌と伴奏の調のずれ（**構造の壊れ**）",
        "**良い曲かどうかは一切見ていない**。詞・響き・抑揚・ハモリはこの検査の外。"
        "歌い出しが小節線からずれていても見ない（設計で拍の裏から入る歌は意図・2026-09-26）"),
    "review_manual_edits": (
        "マスターが手で直した vpr・MIDI と、その版の楽譜から作るものとの差（音符・表情・手で描いたカーブ・伴奏の小節ごとの音）",
        "直した結果が良いかどうか・楽譜への取り込み（この道具は差を出して台帳に残すだけ）"),
    "evaluate_timing": (
        "書き出した音での、歌と伴奏の時間のずれ（ドリフト・全体オフセット）",
        "音程・言葉・音量・編成"),
    "find_cramped_regions": (
        "音節が極端に詰まっている区間（音の間隔が閾値より狭いものが連続する所）",
        "詰まっていない種類の違和感——間延び・朗読調・言葉と旋律の合い方"),
    "find_misplaced_regions": (
        "無音の間と音節の位置がずれている区間（カバーで実演奏から外れた所）",
        "オリジナル曲では意味が薄い。音程・音量・編成は見ていない"),
    "check_mix_quality": (
        "書き出した音の、区間ごとの音量の起伏（抑揚が付いているか）",
        "音程・タイミング・言葉。**測れなかったときは『問題なし』ではなく未確認**"),
    "inspect_song": (
        "曲そのもの——歌の全音符＋伴奏の文脈（コード・和音での役割・拍位置・打点・強弱）と編成の推移",
        "**実際に鳴っている音**（声の質・こもり・滑舌）。楽譜に写らないものは見えない"),
    "inspect_region": (
        "指定した区間の現行ノートと、原曲ボーカルの実オンセット・推定音高",
        "区間の外・伴奏・音量"),
    "inspect_accompaniment": (
        "伴奏のトラック構成・音域・活動範囲と、指定区間のノートとコード推定",
        "歌・音量バランス・実際に鳴った音"),
    "recall_making": (
        "その曲を**どう作ったか**の記録（各工程で何をどんな理由で決めたか）",
        "**いまの姿は見ていない**（記録は作ったときのもの）。現物は inspect_song()"),
}


def with_scope(fn):
    """検査の戻り値に**見たもの／見ていないもの**を必ず添える（2026-09-20）。

    個別の関数に書かせると**次に足す検査でまた抜ける**ので、仕組みで付ける。
    判定（OK／注意／NG／未確認）は中身を知っている関数側が書き、
    **射程と死角はここが保証する**。
    """
    import functools

    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        out = fn(ctx, *args, **kwargs)
        looked, blind = _CHECK_SCOPE.get(fn.__name__, ("（未登録）", "（未登録）"))
        return (f"{out}\n──\n"
                f"【この検査が見たもの】{looked}\n"
                f"【見ていないもの】{blind}\n"
                f"※**上の判定は「見たもの」の範囲だけ**。ここで OK でも、"
                f"マスターの指摘がこの範囲の外なら**まだ何も確かめていない**。")

    return wrapper


def _traced(fn):
    """トレース自動付与デコレータ: 引数（ctx除く）と戻り値の先頭を .in/.out として build_log に記録。
    E2E検分（ツール順・入出力の検証）のため全ツールにトレースを保証する（T5で load_existing が
    ログから追えなかった問題への対処・2026-07-09）。専用の詳細トレースを持つツールには付けない。
    _TOOL_LOCK による直列化も兼ねる（.in→実行→.out がログ上も実行順になる）。"""
    import functools

    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        with _TOOL_LOCK:
            arg_s = ", ".join([repr(a)[:60] for a in args] + [f"{k}={repr(v)[:60]}" for k, v in kwargs.items()])
            blocked = _repeat_guard(ctx, fn.__name__, arg_s)
            if blocked:
                return blocked
            # ★★**どの曲に対して呼んだか／終わった時点でどの曲か**を必ず残す（2026-09-20）★★
            #   これが無かったせいで、**2曲目に切り替えたつもりが1曲目のままだった**事故の
            #   原因を追えなかった（`.out` の記録先が古い曲のログだったことだけが手掛かりで、
            #   「切り替えが効かなかったのか」「記録先が古いだけか」を分けられなかった）。
            #   `deps` の識別子も出す——呼び出しごとに**別インスタンス**になっていれば、
            #   関数の中で更新した song_id が次の呼び出しへ伝わらない。
            _sid0 = ctx.deps.song_id
            _dep = f"{id(ctx.deps) & 0xFFFFFF:06x}"
            _trace(_sid0, f"{fn.__name__}.in", args=arg_s or "(なし)", 対象=_sid0 or "(なし)", deps=_dep)
            out = fn(ctx, *args, **kwargs)
            _sid1 = ctx.deps.song_id
            _res = " ".join(str(out).split())[:140]
            if _sid1 != _sid0:
                # 対象が移った：**移った先**に記録し、**移る前**にも足跡を残す（追跡が切れないように）
                _trace(_sid1, f"{fn.__name__}.out", result=_res,
                       対象変更=f"{_sid0 or '(なし)'} → {_sid1}", deps=_dep)
                if _sid0:
                    _append_log(_sid0, f"[対象変更] {fn.__name__} でこの曲から『{_sid1}』へ移った")
            else:
                _trace(_sid1, f"{fn.__name__}.out", result=_res, 対象=_sid1 or "(なし)", deps=_dep)
            return out
    return wrapper

def _save_section(song_id: str, st: dict) -> None:
    """中間生成物（区間情報 section）を source/section.json に保存。
    編集時に load_existing_song が復元 → align_vocal を再実行できる（再fetch/再区間特定=LLM不要で再整列）。"""
    if not st or st.get("kind") != "audio":
        return                                             # 音声カバーのみ（MIDIは reference から再構成可）
    import json as _json
    try:
        sd = SONGS / song_id / "source"
        sd.mkdir(parents=True, exist_ok=True)
        serial = {k: v for k, v in st.items() if k != "log"}   # log は作業記録（log_build.txt）にある
        (sd / "section.json").write_text(_json.dumps(serial, ensure_ascii=False, default=float), encoding="utf-8")
    except Exception:
        pass

def _build_log_digest(song_dir: Path, max_chars: int = 4500) -> str:
    """作業記録（`log_build.txt`）から経緯のダイジェスト（要点）を返す。巨大行（全ノート列・階名列）は除外。
    編集時に LLM へ渡し、前回どう作ったか（区間/歌詞/再整列/整合/評価）を読んで修正に活かせるように。"""
    p = song_dir / song_files.BUILD_LOG
    if not p.exists():
        return ""
    skip = ("[5.最終ノート", "[2.区間 pyin", "[3.補正後 階名", "[2.区間 pyin階名")   # 巨大行は除外
    try:
        lines = [ln for ln in p.read_text(encoding="utf-8", errors="replace").splitlines()
                 if ln.strip() and not any(ln.lstrip().startswith(s) for s in skip)]
    except Exception:
        return ""
    txt = "\n".join(lines)
    return ("…(冒頭省略)…\n" + txt[-max_chars:]) if len(txt) > max_chars else txt

def _save_ref_index(src: Path, **kw) -> None:
    """採用した参照を source/reference.json に記録（修正時に再fetch=LLM不要で復元するため・basenameで保存）。"""
    import json as _json
    try:
        (src / "reference.json").write_text(_json.dumps({k: v for k, v in kw.items() if v is not None},
                                                        ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def _save_accomp_bounds(song_id: str, bounds: tuple | None) -> None:
    """伴奏を最後に切り出した区間(start_sec,end_sec)を source/accomp_bounds.json に永続化。
    load_existing_song が復元し、render_and_mix の stale 検知（区間変更→伴奏自動再生成）が
    **編集セッションをまたいでも**機能する（railgun: メモリのみ保持で None 素通り→歌と伴奏が約1秒ズレた対処）。"""
    import json as _json
    if not bounds:
        return
    try:
        sd = SONGS / song_id / "source"
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "accomp_bounds.json").write_text(
            _json.dumps({"start_sec": bounds[0], "end_sec": bounds[1]}), encoding="utf-8")
    except Exception:
        pass

def _save_section_bounds(song_id: str, st: dict) -> None:
    """音声カバーの区間の音源境界(秒)を source/reference.json に永続化（紛失時の再スライス/再解析用）。"""
    import json as _json
    p = SONGS / song_id / "source" / "reference.json"
    if not p.exists():
        return
    try:
        r = _json.loads(p.read_text(encoding="utf-8"))
        r["section_start_sec"] = round(float(st.get("start_sec", 0.0)), 2)
        r["section_end_sec"] = round(float(st.get("end_sec", 0.0)), 2)
        r["section_index"] = st.get("section_index")
        p.write_text(_json.dumps(r, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def _write_fetch_log(src: Path, song: str, *, midi_info: dict | None, audio_info: dict | None, adopted) -> None:
    """DL候補→ファイル名・URL・採点・同定・採用を source/fetch_log.txt に記録。
    『どのDLファイルを“正”として使ったか・なぜ選んだか・各候補の同定結果』を後から必ず追える。
    adopted: ("midi", file名, url) / ("audio", url) / None（取得失敗）。"""
    lines = [f"=== 参照取得ログ: 『{song}』 {datetime.now():%Y-%m-%d %H:%M} ==="]
    if midi_info is not None:
        lines.append(f"[MIDI検索] ok={midi_info.get('ok')} 失敗理由={midi_info.get('reason', '')}")
        for c in midi_info.get("candidates", []):
            mark = "★採用" if (adopted and adopted[0] == "midi" and c.get("file") == adopted[1]) else "  - "
            lines.append(f"  {mark} {c.get('file', '?')} ← {c.get('url')}  "
                         f"[{c.get('note_count')}音/ch{c.get('ch')}/歌メロらしさ{c.get('score')}/"
                         f"同定={c.get('match')}: {c.get('reason', '')}]")
    if audio_info is not None:
        lines.append(f"[音声検索(MIDI失敗時のF/B)] ok={audio_info.get('ok')} 失敗理由={audio_info.get('reason', '')}")
        for c in audio_info.get("candidates", []):           # 候補一覧（聴き比べ同定の結果）
            mk = "★採用" if (adopted and adopted[0] == "audio" and c.get("url") == adopted[1]) else "  - "
            lines.append(f"  {mk} {c.get('file', '?')} ← {c.get('url')}  [同定={c.get('match')}: {c.get('reason', '')}]")
    lines.append(f"[採用] {adopted if adopted else 'なし（取得失敗／要確認）'}")
    try:
        (src / "fetch_log.txt").write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass

def _clarify_msg(song: str, kind: str, info: dict) -> str:
    """同名異曲が複数で確定できない時、候補を提示してユーザーに聞き返す質問文を作る（Cの聞き返し）。"""
    lines = [f"【要確認】『{song}』は同名異曲が複数あり、どれが対象か確定できません（{kind}候補）。",
             f"理由: {info.get('reason', '')}", "候補:"]
    for i, c in enumerate(info.get("candidates", [])):
        lines.append(f"  {i}: {c.get('file') or c.get('url')} ← {c.get('url')}  "
                     f"[同定={c.get('match')}: {c.get('reason', '')}]")
    lines.append("→ アーティスト名や出所（例『魔王魂の』）を添えて、どの曲か教えてください。"
                 "その情報で再度 fetch_reference します（勝手に1つを選んで作成しない）。")
    return "\n".join(lines)

#: タイトルからフォルダ名用スラッグ。日本語等はそのまま残し、ファイル名に使えない文字だけ除去。
#: 命名揺れ対策（2026-07-10）: コードが必ず付ける区間「_全体」や「修正版」等のノイズ語は落とす
#: （「全体」以外の区間=「_1番」「_サビ」は別バージョンの区別に必要なので残す。版は版管理が担う）。
#: ★実体は `song_files.slug`（2026-10-03・再レビュー）。途中保存・差し戻しの保存・作り直しの作業場（エンジン側）も同じ1つを通す
_SLUG_NOISE = song_files.SLUG_NOISE
_slug = song_files.slug

_INDEX = SONGS / "songs.md"                    # 全曲の状態ログ（曲の置き場の直下・coordinator が起動時に読む）

def _master_versions(song_dir: Path) -> dict[int, str]:
    """完成品（`master_<形>_v{N}.wav`）がある版 → その形（＝実際に鳴らせる版）。

    完成の判定は**名前の決まりの1か所**（`song_files.masters`）から引く（2026-10-03・§7-1）。同じ判定を
    play_song・song_runner・handoff・Gateway の配信許可も読む（本体側は `song_service.song_files` として読む）。
    歌もの・アカペラは `export_vocal` が V6 の書き出しを仕上げて、インストは `render_final` が演奏を仕上げて書く。
    V6 の生の書き出し（`vocal_v{N}.wav`）は完成品ではない（仕上げのあと完成品へ付け替わる一時のもの）。
    """
    return {m.version: m.form for m in song_files.masters(song_dir)}


def _song_status(song_dir: Path) -> str:
    """曲の状態（songs.md に載る文言）。

    **判定は版付きファイルで行う**（2026-08-18）。固定名は作業中しか存在しないので、
    それを見ると「作業していない曲は全部スコアなし」になってしまう。
    完成は完成品 `master_<形>_v{N}.wav` があるかで決め、言い方を形で分ける（`DONE_LABEL`・2026-10-03）。
    """
    cur = _current_version(song_dir)
    done = _master_versions(song_dir)
    if done:
        if cur and cur not in done:
            # 作り直した後、その版の完成品がまだ無い（前の版なら聴ける）。言い方は**いまの版の形**で決める
            return STALE_LABEL[score_form(song_dir, cur)]
        return DONE_LABEL[done[cur] if cur in done else done[max(done)]]
    if not cur:
        # 版が1つも確定していない。作業台（固定名）があれば作業途中、無ければ残骸
        # （E2E T6 で「生成中」誤表示→誤案内を確認）
        if (song_dir / "score.json").exists():
            return "生成中"
        # ★**途中で止まった曲は「中止」ではない**（2026-09-25）。止めた目印（`_progress.json`）がある曲は
        #   済んだ工程が `_wip` に残っていて、`remake_from_step` で続きを作れる。
        #   「中止(スコアなし)」と出すと、ミクは**続けられない曲**と読んで作り直しを勧める。
        #   `show_after` で手を止めた曲も、ここまで同じ表示になっていた。
        try:
            pg = json.loads((song_dir / song_files.PROGRESS).read_text(encoding="utf-8"))
            # ★**形式の所見が直らずに止まった曲**（2026-09-30・`song_fix.StepUnfixed`）は、どの工程で止まったかを出す。
            #   「途中まで」と同じ表示だと、ミクは直らなかったことをマスターに伝えられない
            st = pg.get("stopped")
            if st:
                step = st.get("step", "") if isinstance(st, dict) else str(st)
                if step == "テーマ":                       # テーマは作り直しの起点にならない（済んだ工程が無い）
                    return "『テーマ』で止まった(直らなかった・作るなら最初から)"
                return f"『{step}』で止まった(直らなかった・続きから作れる)" if step else "工程で止まった(続きから作れる)"
            return ("失敗で止まった(続きから作れる)" if pg.get("failed")
                    else "途中まで(続きから作れる)")
        except Exception:
            return "中止(スコアなし)"
    # ★「下書き(仮歌)」は 2026-09-06 に廃止（draft_mix を作らなくなった）。
    # ★**版が確定していて完成品が無い曲は「生成中」ではない**（2026-10-07・VM の通し確認）。台帳を書くのは
    #   セッションの終わり（版を確定したあと）・止まったとき・編集のあとで、作業中の版はまだ `_CURRENT` に
    #   なっていない——ここに来る曲は**作業が終わっている**。以前は「楽譜がある＝生成中」と書いていたので、
    #   V6 の無い環境で作った曲（`_meta.json` は `awaiting_vocal_export`・報告も「vpr を V6 で開いて書き出せば完成」）が
    #   一覧ではいつまでも「生成中」のままで、ミクとマスターが「まだ作っている」と読んだ。
    #   判定は**その版の事実**（楽譜の形・vpr の有無）で行う。`_meta.json` は版を持たない（最後の書き出しの状態だけ）。
    if not (song_dir / song_files.versioned(song_files.SCORE, cur)).exists():
        return "中止(スコアなし)"
    form = score_form(song_dir, cur)
    if form == "instrumental":                            # V6 を使わない形。完成品は書き出しが直接書くので、無ければ仕上げが要る
        return AWAIT_LABEL["instrumental"]
    vpr = song_files.versioned(song_files.VPR, cur)
    if (song_dir / vpr).exists():
        return AWAIT_LABEL[form].format(vpr=vpr)
    return NO_VPR_LABEL

def _current_version(song_dir: Path) -> int:
    """現在版番号（`_CURRENT_v{N}.txt` マーカーの N）。無ければ 0。
    ＝固定名(vなし)ファイルが何版かを表す。過去版は `*_v{k}`（k<N）に残す（現在版の重複は作らない）。"""
    for p in song_dir.glob("_CURRENT_v*.txt"):
        if (m := re.match(r"_CURRENT_v(\d+)\.txt$", p.name)):
            return int(m.group(1))
    return 0

#: 版として永続する編集物（仮歌/伴奏MIDI/音量バランスまで含め復元を完全に）。並びは `song_files.WORKBENCH`
#: （名前の決まりの1か所・2026-10-03）。
#:
#: **記録（`record.json`＝工程の記録と頼まれごとの台帳）も入れる**（2026-09-25・段階3）。
#: それまでの記録は `_wip/<曲名>` の1本だけで、**版を区別できず**（「v3 から直して」でも最新の記録から
#: 作り直した）、**同じ曲名の別の曲が取り合っていた**。楽譜と同じ版で持てば、版の中で記録と楽譜が揃う。
#:
#: **実体は `*_v{N}` だけ。固定名は作業中しか存在しない**（2026-08-18・下の「作業台方式」参照）。
#:
#: **完成品（`master_<形>_v{N}.wav`）は入れない**。版番号つきで直接書く（作業台を通さない）ので、
#: 固定名からの写しで「名前だけ新しい古い音」ができる経路が無い（2026-08-17 に、固定名の完成ミックスを
#: 版ごとに写していたため全部の版が同じ中身になった事故があった）。V6 の生の書き出し（vpr と同じ名前の wav）も入れない。
#:
#: **口の動き（`lipsync.json`）は入れる**（2026-08-14）。score から決定論的に再生成できる派生物だが、
#: 版対象は「その版を復元したときに揃っているべきもの」で、完成品を歌って鳴らすなら同じ版の口が要る。
_VERSIONED = song_files.WORKBENCH


def _fixed_names_present(song_dir: Path) -> bool:
    """作業台（固定名ファイル）が残っているか＝**前回のセッションが片付けずに終わった**印。"""
    return any((song_dir / name).exists() for name in _VERSIONED)


def _spread_workbench(song_dir: Path, version: int) -> None:
    """指定版の `*_v{N}` を固定名へ展開する（＝作業台に広げる）。

    既存の生成・編集コードは固定名を読み書きする前提で書かれているので、**そこは触らずに済む**。
    版そのものを触るのはこの関数と `_settle_workbench` の2つだけ。
    """
    import shutil
    for name in _VERSIONED:
        stem, suf = name.rsplit(".", 1)
        src = song_dir / f"{stem}_v{version}.{suf}"
        if src.exists():
            shutil.copy2(src, song_dir / name)


def _workbench_unchanged(song_dir: Path, version: int) -> bool:
    """作業台（固定名）が指定版の `*_v{N}` と1バイトも違わないか（＝広げたまま何も書いていない）。"""
    for name in _VERSIONED:
        stem, suf = name.rsplit(".", 1)
        fixed, base = song_dir / name, song_dir / f"{stem}_v{version}.{suf}"
        if fixed.exists() != base.exists():
            return False
        if fixed.exists() and fixed.read_bytes() != base.read_bytes():
            return False
    return True


def _settle_workbench(song_dir: Path, version: int) -> None:
    """固定名の作業結果を `*_v{N}` へ確定し、**固定名を消す**（＝作業台を片付ける）。

    永続するのは版付きだけにする。固定名を残さないことで、
    「固定名が古いまま版を重ねて、名前と中身が食い違う」経路そのものを無くす。
    """
    import shutil
    for name in _VERSIONED:
        stem, suf = name.rsplit(".", 1)
        src = song_dir / name
        if not src.exists():
            continue
        shutil.copy2(src, song_dir / f"{stem}_v{version}.{suf}")
        src.unlink(missing_ok=True)

def _write_current_marker(song_dir: Path, n: int) -> None:
    """`_CURRENT_v{N}.txt` を書き直す（版の所在を人が読める形で残す）。

    並べるファイルと完成品の説明は**その版の形**で分ける（2026-10-03）。インストには vpr・仮歌・口の動きが無く、
    アカペラには伴奏が無い（`song_files.FORM_FILES`）。インストの完成品は V6 を通らない。
    """
    for old in song_dir.glob("_CURRENT_v*.txt"):
        old.unlink(missing_ok=True)
    form = score_form(song_dir, n)
    have = " / ".join(song_files.versioned(x, n) for x in song_files.FORM_FILES[form])
    if form == "instrumental":
        master = (f"完成品は {song_files.master(form, n)}＝伴奏（主旋律の楽器を含む）を仕上げた演奏"
                  "（歌の入らない曲なので VOCALOID6 は使わない）。再生はこれを流す（口は動かさない）。")
    else:
        master = (f"完成品は {song_files.master(form, n)}＝{song_files.versioned(song_files.VPR, n)} を VOCALOID6 で書き出して仕上げた"
                  + ("声だけの曲（アカペラ）" if form == "acappella" else "歌") + "。再生はこれを鳴らす。")
    (song_dir / song_files.current_mark(n)).write_text(
        f"現在版 v{n}（{song_files.FORM_LABEL[form]}）。**実体は版付きファイル**（{have}。\n"
        f"{song_files.versioned(song_files.RECORD, n)} は作り方の記録と頼まれたことの台帳）。\n"
        f"固定名（score.json 等）は**作業中だけ存在する作業台**で、セッションが終わると消える。\n"
        f"残っていれば前回が途中で終わった印＝次のセッションがその続きから作業する。\n"
        f"過去版は *_v1〜v{n}。\n※ {master}",
        encoding="utf-8")


def _working_version(song_dir: Path) -> int:
    """作業中の版（`_WORKING_v{N}.txt`）。無ければ 0。

    セッションの途中で `song_id` が変わりうる（新規発番→編集で別曲へ切替）ため、
    「いまどの版を作っているか」を**フォルダに置いて持ち回る**。異常終了しても残るので、
    次のセッションが続きから作業できる。
    """
    for p in song_dir.glob("_WORKING_v*.txt"):
        if (m := re.fullmatch(r"_WORKING_v(\d+)\.txt", p.name)):
            return int(m.group(1))
    return 0


def _set_working_version(song_dir: Path, n: int) -> None:
    for old in song_dir.glob("_WORKING_v*.txt"):
        old.unlink(missing_ok=True)
    (song_dir / f"_WORKING_v{n}.txt").write_text(
        f"作業中 v{n}。固定名ファイルはこの版の作業台で、確定すると *_v{n} になって消える。\n"
        "このファイルが残っていれば前回のセッションが途中で終わった印＝次回はこの版の続きから作業する。",
        encoding="utf-8")


def ensure_working_version(song_dir: Path) -> int:
    """作業中の版の印（`_WORKING_v{N}.txt`）が無ければ置き、その版を返す（2026-09-30・出-3）。

    ★**新しく作った曲には、作業中の印も確定の印も無かった**。印を置くのは `begin_session` だが、それは
      既存の曲を開くとき（`load_existing_song`）にしか呼ばれない。新しい曲は `render_final` まで固定名の作業台だけで進み、
      版は `end_session` が「作業中の印 or 現在版+1」で決めていた。そのため `render_final` の直後の `export_vocal` が
      「版が分からない」と断り、コーディネーターは render_final を繰り返したうえ曲を開き直して（＝`begin_session` で
      印が置かれて）ようやく書き出せた（9/27・9/29・9/30 の曲で毎回）。
    ここで置く版は `end_session`・依頼文の記録（`coordinator` の `log_prompts_v<版>`）と同じ決め方（作業中の印 or 現在版+1）。
    """
    n = _working_version(song_dir)
    if not n and song_dir.exists():
        n = _current_version(song_dir) + 1
        _set_working_version(song_dir, n)
    return n


def open_version_for_form(song_dir: Path, form: str) -> int:
    """書き出し（`song_pipeline.compose`）の前に、**その形の完成品を書いてよい作業中の版**にする（2026-10-03・再レビュー）。返すのは作業中の版（印が無ければ 0）。

    作業中の版（印 or 現在版+1・`end_session` と同じ決め方）に**別の形の完成品**がもうあるとき（完成のあと同じセッションで
    形を変えた）は、その版をいまの作業台のまま確定し（`end_session`）、次の版を開く（`begin_session`・前の版を作業台へ広げる）。
    ★以前は同じ版に `master_song_v1` と `master_inst_v1` が並び、`song_files.master_of`（最初の1つ）と `_master_versions`（後の1つ）で
      読む形が食い違った（状態と再生がずれる）。消すのでなく版を進めるのは、設計の「前の版は残る」（`change_form` の返事も
      「新しい版として保存したので、前の版は残っている」）に合わせるため——完成した前の形の版は、楽譜・記録ごと残る。
    同じ形の完成品なら何もしない（書き出し直しは同じ版で上書き）。印は置かない（インストの印は呼び手が `ensure_working_version` で置く）。
    """
    n = _working_version(song_dir) or (_current_version(song_dir) + 1)
    if any(m.version == n and m.form != song_files.as_form(form) for m in song_files.masters(song_dir)):
        end_session(song_dir)
        begin_session(song_dir)
    return _working_version(song_dir)


def begin_session(song_dir: Path) -> int:
    """作業する版を決めて、作業台（固定名）を用意する。**曲が決まった時点で呼ぶ。**

    版の粒度は「1回の作業＝1版」。ただし前回が途中で終わっていれば継続する:

    | 作業台の状態 | 意味 | 版 |
    |---|---|---|
    | 何も無い | 前回は正常に片付いた | **新版**（現在版+1）を切り、現在版を展開して作業 |
    | 固定名 or `_WORKING` が残っている | 前回が打ち切り・異常終了 | **継続**（その版のまま作業台を引き継ぐ） |

    判定に `_last_run.json` を読まないのは、**フォルダの状態そのものが事実**だから
    （記録と実体がずれる余地を作らない）。
    """
    if not song_dir.exists():
        return 1                                           # 新規曲。確定時に v1 になる
    working = _working_version(song_dir)
    if working or _fixed_names_present(song_dir):
        n = working or max(_current_version(song_dir), 1)
        _set_working_version(song_dir, n)                  # 継続（マーカーが無ければ補う）
        return n
    n = _current_version(song_dir)
    nxt = n + 1
    if n >= 1:
        _spread_workbench(song_dir, n)                     # 現在版を作業台へ展開してから編集させる
    _set_working_version(song_dir, nxt)
    return nxt


def end_session(song_dir: Path, asked: str = "") -> int:
    """作業結果を版として確定し、作業台を片付ける。**セッション終了時に1回だけ呼ぶ。**

    書いたものが何も無ければ版を切らない（空の版を増やさない）。`asked` はその回に頼まれた言葉（版の記録に残す）。
    """
    if not song_dir.exists() or not _fixed_names_present(song_dir):
        for old in song_dir.glob("_WORKING_v*.txt") if song_dir.exists() else []:
            old.unlink(missing_ok=True)                    # 何も書かずに終わった＝作業中の印も消す
        return _current_version(song_dir)
    n = _working_version(song_dir) or (_current_version(song_dir) + 1)
    # ★**開いて読んだだけなら版を切らない**（2026-10-09）。`begin_session` は開いた時点で現在版を作業台へ広げ、
    #   ここは作業台があれば必ず新しい版にしていた。歌詞を確かめただけの回で、現在版と1バイトも違わない
    #   v5・v7 ができ、完成品の無い版が最新になって「歌声の書き出しが失敗した」と伝わった（9/20 に作り直しは
    #   止めたが、版を切るところは残っていた）。作業台が現在版と同じで、作業中の版に直接書いたもの
    #   （完成品など `*_v{N}.*`）も無ければ、作業台を片付けて現在版のまま返す。
    cur = _current_version(song_dir)
    if cur >= 1 and n == cur + 1 and _workbench_unchanged(song_dir, cur) \
            and not any(p.is_file() and f"_v{n}." in p.name and not p.name.startswith("_WORKING_")
                        for p in song_dir.iterdir()):
        for name in _VERSIONED:
            (song_dir / name).unlink(missing_ok=True)
        for old in song_dir.glob("_WORKING_v*.txt"):
            old.unlink(missing_ok=True)
        return cur
    _settle_workbench(song_dir, n)
    _write_current_marker(song_dir, n)
    # ★**この版のファイルの指紋を控える**（2026-09-26・本人了承）。人が VOCALOID6 や MIDI の道具で直したとき、
    #   どのファイルが変わったかを知るため（突き合わせは `review_manual_edits` を呼んだときだけ）。
    try:
        from manual_diff import write_manifest
        write_manifest(song_dir, n)
    except Exception:
        pass
    # ★**版の記録を1行残す**（2026-10-11・本人了承・案B）。「朝の版」「さっき直す前の版」と言われたとき、
    #   ミクが版の番号でなく時刻と直した内容で当てられるように（`song_files.version_log` が読む）
    try:
        song_files.record_version(song_dir, n, cur, asked)
    except Exception:
        pass
    for old in song_dir.glob("_WORKING_v*.txt"):
        old.unlink(missing_ok=True)
    return n

def _snapshot_version_if_changed(song_dir: Path) -> int:
    """**互換のための入口**（2026-08-18 に作業台方式へ移行）。

    かつては「固定名と `*_v{N}` を全バイト比較し、違えば新版としてコピー」していた。
    いまは実体が `*_v{N}` だけで、固定名は作業台（作業中しか存在しない）ため比較が要らない。
    呼び出し側の互換のために残し、現在版を返すだけにする。
    **版を進める／確定するのは `begin_session` / `end_session`**。
    """
    return _current_version(song_dir)

def _read_index() -> str:
    return _INDEX.read_text(encoding="utf-8") if _INDEX.exists() else ""

def _update_index(song_id: str) -> None:
    """output直下 songs.md に当該曲の行を upsert（フォルダの score.json/ファイル有無から状態・現在版を導出）。"""
    import json as _json
    song_dir = SONGS / song_id
    if not song_dir.exists():
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    ver = _current_version(song_dir)
    # 曲名は「作業台 → 現在版 → 途中で止めた印」の順に探す。**固定名は作業中しか存在しない**ので、
    # そこだけを見ると作業していない曲の曲名が全部 "?" になる（2026-08-18）。
    # ★**止めた目印（`_progress.json`）を最後に見る**（2026-09-23）。途中で止めた曲は**まだ楽譜が無く、
    #   曲名がここにしか無い**ので、足さないと台帳に "?" と載る——「どれを聴く?」の
    #   聞き返しでも "?" が並び、選びようがなくなる。**楽譜があるときの挙動は変えない**（順番が後）。
    title = "?"
    for cand in (song_dir / "score.json", song_dir / f"score_v{ver}.json",
                 song_dir / song_files.PROGRESS):
        try:
            t = _json.loads(cand.read_text(encoding="utf-8")).get("title")
        except Exception:
            continue
        if t:
            title = t
            break
    row = f"| {song_id} | {title} | {_song_status(song_dir)} | v{ver} | {now} |"
    head = ("# 曲一覧（coordinator 状態ログ・自動更新）\n\n"
            "| song_id(フォルダ) | 曲名 | 状態 | 版 | 更新 |\n|---|---|---|---|---|\n")
    rows: dict[str, str] = {}
    if _INDEX.exists():
        for ln in _INDEX.read_text(encoding="utf-8").splitlines():
            if ln.startswith("| ") and "---" not in ln and not ln.startswith("| song_id"):
                rows[ln.split("|")[1].strip()] = ln
    rows[song_id] = row
    _INDEX.write_text(head + "\n".join(rows[k] for k in sorted(rows)) + "\n", encoding="utf-8")


def _rename_index_row(old_id: str, new_id: str) -> None:
    """台帳（songs.md）の行を古い song_id から新しい song_id へ書き換える（**行を2つにしない**・2026-10-03）。

    1列目の song_id が upsert の鍵なので、`_update_index(new_id)` だけだと古い行が残り、`work_on_music` の一覧と
    コーディネーターの【既存曲一覧】に消えた曲が出る。**古い行が無ければ何もしない**（まだ台帳に載っていない新しい曲は今のまま）。
    古い行の消し方は `_retire_scoreless_session` と同じ。
    """
    if not _INDEX.exists():
        return
    lines = _INDEX.read_text(encoding="utf-8").splitlines()
    kept = [ln for ln in lines if not ln.startswith(f"| {old_id} ")]
    if len(kept) == len(lines):
        return
    _INDEX.write_text("\n".join(kept) + "\n", encoding="utf-8")
    _update_index(new_id)


#: 曲フォルダの名前 `YYYYMMDD_HHMMSS[_曲名]`（頭の日時と、うしろの曲名の slug）
_STAMPED_ID = re.compile(r"^(\d{8}_\d{6})(?:_(.+))?$")


def _repoint_paths(deps: MusicSession, old_dir: Path, new_dir: Path) -> None:
    """同じ会話の中で持っている**曲フォルダの中を指す絶対パス**を新しいフォルダへ向け直す（2026-10-03）。

    参照（`reference_json` / `reference_midi` / `reference_audio_accomp`）と区間の中間状態（`section` の
    `vocals_wav` / `accomp_wav`）。曲フォルダの外を指すもの（チャットに添付された音など）は触らない。
    """
    old_s = str(old_dir)

    def _moved(p):
        if isinstance(p, str) and (p == old_s or p.startswith(old_s + "\\") or p.startswith(old_s + "/")):
            return str(new_dir) + p[len(old_s):]
        return p
    for name in ("reference_json", "reference_midi", "reference_audio_accomp", "input_audio"):
        setattr(deps, name, _moved(getattr(deps, name, None)))
    if isinstance(deps.section, dict):
        for k in ("vocals_wav", "accomp_wav"):
            if deps.section.get(k):
                deps.section[k] = _moved(deps.section[k])


def rename_song_folder(deps: MusicSession, new_id: str, *, tag: str, songs: Path | None = None) -> str:
    """曲フォルダを `new_id` へ付け替え、**同じ会話で曲を指しているもの**を一緒に直す（2026-10-03）。

    直すもの: `deps.song_id`・同じ会話で持っている絶対パス（`_repoint_paths`）・台帳の行（`_rename_index_row`）。
    直さなくてよいもの（読むときに basename で直す・古い名前のまま残っても曲を指せる）: `_meta.json`・`log_build.txt`・
    `source/` の中・KV の `background_tasks`・会話ログ。
    返り値: `"ok"`（付け替えた）／`"busy"`（中のファイルを開いているものがあって変えられない＝もう一度試す値打ちがある）／
    `"skip"`（元が無い・行き先が既にある・同じ名前）。**失敗しても例外は出さない**——曲づくりは止めない。
    """
    old_id = deps.song_id or ""
    if not old_id or not new_id or new_id == old_id:
        return "skip"
    songs = songs or SONGS
    old_dir, new_dir = songs / old_id, songs / new_id
    if not old_dir.exists():
        return "skip"
    if new_dir.exists():                                   # 同じ日時・同じ曲名のフォルダ（複製など）——上書きしない
        _trace(old_id, f"{tag}.warn", reason="行き先が既にある", new=new_id)
        return "skip"
    try:
        old_dir.rename(new_dir)
    except OSError as e:                                   # 致命ではない＝元の名前のまま続ける
        _trace(old_id, f"{tag}.warn", reason=f"{type(e).__name__}: {e}"[:160],
               hint="中のファイルを開いているものがあると名前を変えられない。次の書き出しでもう一度試す")
        return "busy"
    deps.song_id = new_id
    _repoint_paths(deps, old_dir, new_dir)
    try:
        _rename_index_row(old_id, new_id)
    except Exception as e:                                 # 台帳が書けなくても、フォルダは付け替わっている
        _trace(new_id, f"{tag}.warn", index=f"{type(e).__name__}: {e}"[:120])
    _trace(new_id, f"{tag}.out", old=old_id, new=new_id)
    return "ok"


def settle_folder_title(deps: MusicSession, songs: Path | None = None) -> None:
    """**付け替えたい曲名**（`deps.folder_title`）があれば、曲フォルダの名前を `YYYYMMDD_HHMMSS_<その曲名の slug>` にする（2026-10-03）。

    頭の日時は残し、`_` のうしろだけを替える。**うしろが既に同じ slug なら何もしない**（表記の揺れで付け直さない）。
    開いていて変えられなければ覚えたまま返す——次の書き出しと会話の終わりにもう一度ここを通る。
    """
    title = deps.folder_title
    if not title:
        return
    m = _STAMPED_ID.fullmatch(deps.song_id or "")
    s = song_files.slug(title)
    if not m or not s or (m.group(2) or "") == s:          # 日時で始まらない名前は触らない／既に同じ名前
        deps.folder_title = ""
        return
    if rename_song_folder(deps, f"{m.group(1)}_{s}", tag="rename_folder", songs=songs) != "busy":
        deps.folder_title = ""


def retitle_song_folder(deps: MusicSession, old_title: str, new_title: str, songs: Path | None = None) -> None:
    """**曲名の作り直しで曲名が実際に変わったら**、曲フォルダの名前も新しい曲名へ付け替える（2026-10-03・本人了承）。

    以前は「名前の付いたフォルダは付け直さない」だった（2026-07-12。編集のたびに slug を取り直すと、slug の作り方の差で
    フォルダ名が二重になる・付け替えに失敗する）。いまは slug が `song_files.slug` の1つで、付け替えるのは
    **曲名が変わったときだけ**、しかも**うしろを置き換える**（足さない）ので、その2つは起きない。
    """
    if not (new_title or "").strip() or (new_title or "").strip() == (old_title or "").strip():
        return
    deps.folder_title = new_title
    settle_folder_title(deps, songs)

_RETIRABLE_ID = re.compile(r"^\d{8}_\d{6}$")             # 曲名の付かない裸の発番フォルダだけが対象


def _retire_scoreless_session(song_id: str) -> None:
    """曲にならなかった**裸の発番フォルダ**を `output/_archive/` へ退避する（**削除はしない**）。

    一覧質問・既存曲の編集・聞き返し・ガード停止のたびに裸タイムスタンプフォルダが残る問題への
    対処（2026-07-10。新規/修正はLLMの判断後に決まるため、発番フォルダを事前には抑止できない）。

    **対象は名前で決める（ホワイトリスト）**。`YYYYMMDD_HHMMSS` に完全一致するものだけを畳み、
    曲名が付いていれば——中身が何であれ——触らない。「消してよい条件に当てはまらないものを畳む」
    と書くと、条件の側が変わった瞬間に無関係なフォルダを巻き込む。実際、作業台方式で `score.json`
    が確定時に消えるようになった結果、**完成した曲を「スコアなし」と誤判定して丸ごと削除した**
    （2026-08-18。『なついろのそら』の v2〜v5 を失い、output/ は git 管理外のため復旧できなかった）。

    削除ではなく退避なのも同じ事故への備え。判定を誤っても取り返しがつく。

    ★**途中で止めた曲も見送る**（2026-09-23 追加）。`draft_melody(show_after=...)` で
    工程の途中まで作って抜けた曲は、**曲名が止めた目印（`_progress.json`）にしか無く、楽譜もまだ無い**
    ——つまり「裸の発番フォルダ＋楽譜なし」に**そのまま当てはまる**。実際に退避され、
    **聴かせるために作った `preview.wav` を `play_song` が見つけられなくなった**（実機確認で検出）。
    `_wip/<曲名>.json` は触られないので続きは作れたが、**聴かせる側だけが壊れていた**。
    足すのは**見送る側の条件**なので、上のホワイトリスト設計は崩さない（安全側へ倒す）。
    """
    import shutil
    if not song_id or not _RETIRABLE_ID.fullmatch(song_id):
        return                                           # 曲名付き＝曲が確定した証拠。中身を問わず触らない
    d = SONGS / song_id
    if not d.exists() or (d / "score.json").exists() or any(d.glob("score_v*.json")):
        return                                           # 作業中（作業台あり）／版が確定済み＝曲がある
    if (d / song_files.PROGRESS).exists():
        return                                           # 途中で止めて人に見せている最中＝まだ生きている
    try:
        lp = d / song_files.BUILD_LOG
        log = lp.read_text(encoding="utf-8") if lp.exists() else "(ログなし)"
        with (SONGS / "_sessions.log").open("a", encoding="utf-8") as f:
            f.write(f"\n───── {song_id}（曲が確定しなかったセッション・_archive へ退避）─────\n{log}\n")
        dest = SONGS / "_archive" / song_id     # 名前は変えない（復元は移し戻すだけで済む）
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():                            # 既に退避済みなら出所を上書きしない（残すほうが安全）
            shutil.move(str(d), str(dest))
        if _INDEX.exists():
            lines = [ln for ln in _INDEX.read_text(encoding="utf-8").splitlines()
                     if not ln.startswith(f"| {song_id} ")]
            _INDEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass                                             # 退避失敗は致命でない（従来どおり残るだけ）
