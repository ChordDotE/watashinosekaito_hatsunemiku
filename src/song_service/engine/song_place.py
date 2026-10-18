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

"""**歌の組み立て**（記号→音）——詞を旋律の音へ当てる・移調と転調・曲頭の間・時間軸・歌の音符・コード・調声・楽譜にまとめる。

2026-09-30 に `song_compose._compose_once_body` から分けた。2026-10-03（アカペラ・インスト・`docs/plan/song_acappella_instrumental.md` §5）から
呼び順は **`build_timeline`（時間軸・調・コード・全部の形）→ `place_vocal`（歌もの・アカペラ）か `place_lead`（インスト・詞なし）**。
歌と主旋律の楽器の位置・長さの読み方は `_lay` の1か所。主旋律を楽器で弾くパートは `make_score` が楽譜（`SongScore.lead`）に入れる。
**組み立ては1か所に集める**（2本あると片方だけ直る）——
新しい曲（`place_vocal`）も編集の足し込み（`song_resize`）も、詞を音へ当てるのは `fit_phrase` の1か所を通す。
**組み立ては一切補正しない**（2026-09-28 本人方針）: 書かれたとおりに鳴らせる形へ訳すだけ。
**検査が保証している形しか来ない前提で書く**（2026-09-30）: 形式の所見が直らなければその工程で曲づくりが止まる
（`song_fix.StepUnfixed`）ので、詞と音の数のずれ・同じ位置に重なった音・コードの小節数のずれ・読めないコード名は来ない。
来たら検査と組み立てのずれ＝不具合なので `song_fix.AssemblyError` で止める（以前は鳴らさずに記録して進んでいた）。
"""
from __future__ import annotations

from typing import Callable

import lyric_display                                      # 見せる歌詞↔歌う歌詞の突き合わせ
import song_files                                         # 形の綴り（`as_form`）は1か所
import lyric_slots as _ls                                 # 詞の印を音ごとに分ける（タイの続きの音・詞の検査と同じ分け方）
import song_checks as ck
from song_design import CHORD_BAR, Q, bar_ticks, chord_readable, key_shift, transpose_chord
from song_fix import AssemblyError                         # 検査が保証しているはずの形でないものが来た（不具合）
from song_llm import solfa_to_midi
from song_lyrics import LYRIC_LINE
from song_model import StepRecord, made
from song_rhythm import GRID
from song_schema import ArrangementSpec, ChordSpan, LeadNote, LeadPart, SongNote, SongScore
from song_timeline import LEAD_IN, SectionTimeline         # 区間の位置はここが正
from song_tuning import apply_tuning                      # 調声はここだけ・カバーと共通


# ---- 詞を旋律の音へ当てる ------------------------------------------------------

def _hold_kana(ms: list[str], i: int) -> str:
    """`ms[i]` を**次の音符へ伸ばす**ときに置く仮名。

    ★★**「-」（前の音をそのまま伸ばす＝タイ）を返す**（2026-09-13 修正）★★

    以前は「ま」→「あ」のように**母音のかなを置いていた**。しかし V6 は歌詞「あ」を
    **新しい音として立ち上げる**ので、「ま・あ」と2音に聞こえる——本人が 2026-09-09 に
    指摘した「**『まあ』で『あ』を再発音してる**」そのもの。LLM 側の詞は（当時）`fix_sokuon` で直したが、
    **プログラム側のこの関数が同じことを作り続けていた**（『名前のない衝動』の「すれちが・あ・た」）。

    VOCALOID には**前の音を再発音せずに伸ばす歌詞「-」**がある（libresvip の `legato_chars`
    ＝「ー」「−」「-」→ 音素 "-"）。メリスマも促音の保ちも、本来これで書く。
    「ん」も「っ」の遡りも要らない——**何を伸ばすかは V6 が前の音から決める**。
    """
    return "-"


# 促音の「詰まり」＝音符の終わり際に置く無音の長さ（拍）。
# 日本語の促音の閉鎖はおおむね 0.1〜0.2秒。拍で持つと速い曲では短く、遅い曲では長くなり、
# どちらも歌として自然な範囲に収まる。**音符の長さの3割**を上限にして、短い音符が消えないようにする。
_SOKUON_GAP_BEATS = 0.25


def _sustain_sokuon(ms: list[str]) -> tuple[list[str], set[int]]:
    """**促音「っ」を音符に置かない**（2026-08-23 ユーザー指摘）。

    「っ」はモーラとして数えるので1音符を占めるが、**VOCALOID では無音**（声門閉鎖）。
    実測では『空白の夜明け』の「ず・っ・と」で **っ が1.5拍**あり、
    「ず …（無音1.5拍）… と」という**穴**になっていた。

    実際の歌では、促音の位置は——

      **直前の母音を保ったまま、その音符の「終わり際だけ」音を切る**

    切れた一瞬の無音が、そのまま「っ」の詰まりになる。
    母音をただ伸ばし切るのでも、まるごと無音にするのでもない。

        ず・っ・と  →  ず・-（前の音を保ったまま、終わり際を切る）・と

    ★「っ」の音符に置くのは母音のかなではなく **「-」（タイ）**（2026-09-13）。
    「う」と書くと V6 が**「う」を再発音**して「ず・う・と」の3音に聞こえた。
    戻り値は (置き換えた詞, **終わり際を切る位置**)。音符の数も高さも変えない。
    """
    if "っ" not in ms:
        return ms, set()
    out, cut = list(ms), set()
    for i, m in enumerate(out):
        if m != "っ" or i == 0:
            continue
        out[i] = _hold_kana(out, i)                        # 直前の母音（「ん」なら「ん」）
        cut.add(i)
    return out, cut


def _close_tie_gaps(out: list[tuple]) -> list[tuple]:
    """**「-」（前の音を伸ばす）の直前に休みを置かない**（2026-09-26・本人指摘）。

    旋律が音のあとに小休み（`gap_after`）を書いた音符の次が「っ」やメリスマだと、休みが
    「し」と「-」の**あいだ**に入る。V6 の「-」は直前の音を切らずに伸ばす記号なので、間に無音があると
    伸ばす相手がいなくなり、「しっかり」の「っ」が**伸ばしにならなかった**（『明日への足音』で 0.32秒の無音）。
    休みは「-」の**あと**へ回し、そのぶん「-」を伸ばす——次の音の位置は変わらない。
    要素は `(歌詞, 階名, オクターブ, 枠の長さ, 鳴る長さ, …)`（後ろの欄はそのまま通す）。
    """
    res = [list(x) for x in out]
    for k in range(1, len(res)):
        prev, cur = res[k - 1], res[k]
        if cur[0] != "-" or len(prev) < 5 or len(cur) < 5:
            continue
        g = prev[3] - prev[4]
        if g <= 1e-9:
            continue
        prev[3] = prev[4]                                  # 前の音は休まずに「-」へつなぐ
        cur[3] += g                                        # 休みは「-」のあとへ
        cur[4] += g                                        # 「-」をそのぶん伸ばす（促音の詰まりの長さは変えない）
    return [tuple(x) for x in res]


def fit_phrase(ss, notes: list["BareNote"],
               log: list[str] | None = None, label: str = "") -> list[tuple]:
    return _close_tie_gaps(_fit_phrase(ss, notes, log, label))


def _fit_phrase(ss, notes: list["BareNote"],
                log: list[str] | None = None, label: str = "") -> list[tuple]:
    """フレーズの詞を音符へ当てる。**印を1音ずつ、書いた順に置くだけ**（2026-09-28・組み立ては一切補正しない）。

    `ss` は `lyric_slots.slots(text)`（旋律の1音ごとの印・`Slot` の並び）。かなのリストを受ける古い呼び方は 2026-09-30 に外した。
    戻り値は `(歌詞, 階名, オクターブ, 枠の長さ, 鳴る長さ, 音の中の何番目か)` の並び。
    - `~` で詰めた別々の音節（`い~い`）: **その音を等分**して1音1モーラにする（V6 は1音に2モーラを歌えない・2026-09-29 実測）。
      分けた2つ目以降は「何番目か」が 1 以上——置く側は、音の頭だけを格子に乗せ、残りは頭からの距離のまま置く
    - `~` の無い区切りの長音（`にい`・`びょお`）: 前のモーラのまま1音で伸ばす（長音は音節の一部。`lyric_slots.Slot.parts`）
    - `_`（伸ばし）: 歌詞「-」（前の音を再発音せずに伸ばす・`_hold_kana`）
    - `*`（歌わない）: 歌詞を空にした1音——**時間は進めるが鳴らさない**

    ★以前は過不足を機械が吸収していた——音符が多ければ長い音の上でモーラを伸ばし（メリスマを機械が選ぶ）、
      詞が多ければ長い音を割って置き場を増やし（旋律のリズムが変わる）、割っても入らなければ末尾のモーラを捨て、
      詞が無ければ「あ」で歌っていた。どれも LLM の書いたものを変える補正で、詞の抜け・数のずれという不具合を隠していた。
      いまは詞の工程が詰め・伸ばしを印で書き、印の数と音の数のずれは形式の検査（`LyricsStep`）が差分で直させる。
      数が合わなければ: **詞の無い音は鳴らさない／音の無い印は歌わない**（`log` に書く）。新しい曲の組み立て（`place_vocal`）は
      `log` に書かれたら不具合として止める（2026-09-30。直らなければ詞の工程で止まるので来ない）。
    """
    if not notes:
        return []
    # ★**印の数は ABC で書いた音の数**（2026-10-01・判-H）。タイでまとめた音は `1 + ties` 個の印を持ち、続きの音の印は `_`
    #   （`lyric_slots.by_note`・詞の検査 `tie_problems` と同じ分け方）。続きの `_` は前の音に含める＝1つの長い音のまま鳴る
    n, m = _ls.slot_count(notes), len(ss)
    if m != n and log is not None:
        if not m:
            log.append(f"{label}詞が無いので、{n}音を鳴らさなかった")
        elif m < n:
            log.append(f"{label}{n}音に印{m}つ——詞の無い後ろの{n - m}音を鳴らさなかった")
        else:
            log.append(f"{label}{n}音に印{m}つ——音の無い末尾の「{' '.join(s.token for s in ss[n:])}」を歌わなかった")
    if log is not None:
        log += [f"{label}{w}" for w in _ls.tie_problems(ss, notes)]
    # 1音ずつ、印のとおりに分ける: (歌詞, 階名, 8ve, 枠, 鳴る長さ, 何番目か, タイでまとめた続きの音の数)
    pieces: list[list] = []
    for (s, conts), nt in zip(_ls.by_note(ss, notes), notes):
        if s is None:                                       # 詞の無い音（`log` に書いてある）
            break
        gap = max(0.0, nt.gap_after)
        ties = len(conts)
        if s.kind == "hold":
            pieces.append(["-", nt.solfa, nt.octave, nt.beats + gap, nt.beats, 0, ties])
        elif s.kind == "skip" or not s.moras:
            pieces.append(["", nt.solfa, nt.octave, nt.beats + gap, nt.beats, 0, ties])
        else:
            ps = s.parts or s.moras                         # 長音は前の音に含めてある（`lyric_slots.Slot`）
            k = len(ps)
            b = nt.beats / k
            for j, mo in enumerate(ps):
                last = j == k - 1
                pieces.append([mo, nt.solfa, nt.octave, b + (gap if last else 0.0), b, j, ties if last else 0])
    # 促音「っ」は前の音を保って終わり際を切る（V6 への訳し方）。分けたあとの1音ごとに見る
    ms, cut_at = _sustain_sokuon([p[0] for p in pieces])
    for i, p in enumerate(pieces):
        p[0] = ms[i]
        if i in cut_at:
            p[4] = max(0.05, p[4] - min(_SOKUON_GAP_BEATS, p[4] * 0.3))
    return [tuple(p) for p in pieces]


# ---- 歌を置く（移調・時間軸・音符・コード・調声）---------------------------------------

def section_keys(design) -> tuple[dict[str, str], "Callable[[str], int]"]:
    """設計 → (区間 → 調, 区間 → 移調量（半音・C メジャー／A マイナー基準から）)。**移調量はここ1か所で決める**。

    旋律・コード（`place_vocal`）・伴奏の `pitches`／リフ（`song_arrange.to_spec` の `shift_of`）・声部（`harmony.voices_for`・
    2026-09-30 の記-3 から）が同じ `shift_of` を使う——どれも C メジャー／A マイナー基準で書かせ、ここで区間ごとに移す。
    開発用の再生比較の変換（`dev_tools/song_legacy/convert_record`）も同じ関数を呼ぶ（2本目の計算を作らない）。
    区間名の無い所（曲頭の空き小節 `LEAD_IN` など）は設計の調の移調量。
    """
    # **転調**: セクションごとに移調量が変わる（2026-08-21 追加）。
    # 旋律もコードも C メジャー基準で書かせているので、ここで区間ごとに別々の量だけずらす。
    # 転調を「宣言されたセクション以降ずっと」効かせるため、sections の順に持ち回る。
    # 転調の区間名も同じ照らし方で設計の区間へ（大文字小文字の違いで転調が黙って消えないように・2026-09-28）
    _dnames = [s.kind for s in design.sections]
    mod_at = {(ck.resolve_section(m.at_section, _dnames) or m.at_section): m.key
              for m in design.modulations}
    key_of: dict[str, str] = {}
    cur_key = design.key
    for s in design.sections:
        cur_key = mod_at.get(s.kind, cur_key)
        key_of[s.kind] = cur_key
    return key_of, keys_shift(key_of, design.key)


def keys_shift(key_of: dict[str, str], default_key: str) -> "Callable[[str], int]":
    """区間 → 調（`key_of`）→ 区間 → 移調量（半音・C メジャー／A マイナー基準から）。**移調量の計算はここ1か所**。

    新しい曲は `section_keys`（設計の調と転調から区間の調を決める）がここを呼ぶ。**カバー**（2026-10-01・判-C）は
    原曲の音から推定した区間の調（`song_cover_load.add_sections` の `Song.key_of`）でここを呼ぶ——カバーも伴奏・声部を
    C 基準で書かせて、同じ量だけ移す（`song_arrange_cover`）。区間名の無い所は `default_key` の移調量。
    """
    sec_shift = {k: key_shift(v) for k, v in key_of.items()}

    # オクターブ補正は**曲全体で1つ**（区間ごとに変えると転調のたびに跳ぶ）。
    # ★★**曲全体のオクターブを機械が選ばない**（2026-09-27・目録の旋律-6）★★
    #   以前はここで「0 か −12 の、音域（58〜84）からのはみ出しが小さいほう」を機械が選んでいた。旋律には**移調前の空間での上限**
    #   （`_top_c`＝設計の最高音−いちばん上がる移調量）を渡し、`fix_melody` がその上限と最低音を検査して旋律に直させるので、
    #   移調しても設計の最高音を超えない。−12 が選ばれるのは旋律が上限を破ったときだけで、そのとき**曲全体が1オクターブ
    #   下がり、サビが開かなくなった**（『青色へと』で最低音 57）。旋律の決めた高さはそのまま使う。
    oct_adj = 0
    default = key_shift(default_key)                       # 転調が無い区間・区間名の無い所の既定

    def shift_of(kind: str) -> int:
        return sec_shift.get(kind, default) + oct_adj

    return shift_of


def place_chords(timeline: SectionTimeline, ch_map: dict, shift_of: "Callable[[str], int]",
                 key_of: dict[str, str], bar: int) -> list[ChordSpan]:
    """**コードを時間軸に置く**——区間の調へ移し（`shift_of`）・区間の調で綴り（`key_of`）・1小節に書いたコードは書いた数で小節を等分する。

    新しい曲の組み立て（`place_vocal`）と、旋律より前の段の試聴（`song_preview.placed_chords`・2026-10-01 判-E）が同じここを通る
    ——試聴用の置き方を別に作らない（2本あると片方だけ直る）。`ch_map` は区間 → 1小節ぶんのコードの並び（C 基準・コードの工程の形）。
    曲頭の空き小節（`LEAD_IN`）にはコードを置かない。
    """
    spans: list[ChordSpan] = []
    for kind, t0, t1 in timeline.key_ranges():
        if kind == LEAD_IN:                                # 曲頭の空き小節（組み立てが置く・コードも歌も無い）
            continue
        cs = ch_map.get(kind) or []
        _nb = (t1 - t0) // bar
        # ★**書いたコードを書いた小節に置くだけ**（2026-09-28）。以前は小節数が合わないと頭から繰り返し／後ろを捨てていた
        #   （時間-5）。全区間があるか・小節数は設計との照合（`ChordsStep`）が見ていて、直らなければコードの工程で止まる
        #   （`StepUnfixed`）ので、数が合わないことは無い。来たら不具合として止める（2026-09-30。以前は書いていない小節に
        #   和音を置かずに進んでいた）
        if len(cs) != _nb:
            raise AssemblyError(f"コードの組み立てに、検査を通ったはずのコードが来た: 区間『{kind}』は{_nb}小節なのに"
                                f"コードが{len(cs)}小節ぶん")
        for i in range(_nb):
            cell = str(cs[i])
            halves = CHORD_BAR.read(cell)                  # 検査と同じ記法の型で読む（空白区切り。`song_design.CHORD_BAR`）
            # ★**空の要素・読めないコード名は、移調量に関わらず止める**（2026-10-02）。以前は `or [cell]` で空の要素を
            #   コード名 `""` として置き、移調量 0 の区間では `transpose_chord` が読まずに返すので**黙って無音の和音**になった
            #   （移調する区間だけ ValueError で止まっていた）。どちらも記法の型（`CHORD_BAR`＝空の要素・`chord_problem`）が
            #   所見にし、直らなければコードの工程で止まるので来ない。来たら検査と組み立てのずれ＝不具合
            if not halves:
                raise AssemblyError(f"コードの組み立てに、検査を通ったはずのコードが来た: {kind} {i + 1}小節目が空"
                                    f"（『{cell}』。和音なしは `N.C.`）")
            bad = [c for c in halves if not chord_readable(c)]   # 読めるかは `song_design.chord_readable`（尺の変更・カバーと同じ部品）
            if bad:
                raise AssemblyError(f"コードの組み立てに、検査を通ったはずのコード名が来た: {kind} {i + 1}小節目"
                                    f"『{'・'.join(bad)}』（読めないコード名）")
            sh = shift_of(kind)                            # 転調はコードにも同じだけ効かせる
            at = t0 + i * bar
            # ★**1小節に書いたコードは、書いた数で小節を均等に分ける**（2026-09-28）。以前は前半・後半の2つにしか割れず、
            #   3つ目以降を捨てていた（コードの工程に「2つまで」と直させる検査もあった）。組み立てが書いたとおりに置けば検査は要らない
            # ★読めないコード名は移調できない（`transpose_chord` が ValueError）。読めないコード名は記法の型（`CHORD_BAR`＝
            #   `chord_problem`）が見ていて、直らなければコードの工程で止まるので、ここへは来ない。来たら不具合として止める
            #   （2026-09-30。2026-09-28 からは和音を置かずに記録して進み、その前は C 以外の調で作曲が落ちた）
            n_ch = len(halves)
            parts = [(c, at + h * bar // n_ch, (h + 1) * bar // n_ch - h * bar // n_ch) for h, c in enumerate(halves)]
            for c, st_, du in parts:
                try:
                    # 区間の調で綴る（F の曲の B♭ を `A#` と保存しない・2026-10-01 判-I）。鳴る音は同じ
                    spans.append(ChordSpan(chord=transpose_chord(c, sh, key_of.get(kind)), start_tick=st_, duration_tick=du))
                except ValueError as e:
                    raise AssemblyError(f"コードの組み立てに、検査を通ったはずのコード名が来た: {kind} {i + 1}小節目『{c}』"
                                        f"（{e}）") from e
    return spans


def build_timeline(st) -> None:
    """**時間軸・区間の調と移調量・コードの配置**（2026-10-03 に `place_vocal` から分けた・`docs/plan/song_acappella_instrumental.md` §5-1）。
    **全部の形で通る**（インストは詞が無いので、詞を音へ当てる `place_vocal` とは分けた）。歌の音符を置く前に呼ぶ。

    読む: `song.design`・`song.melody`（曲頭の間を決めるのに最初のフレーズの休みだけ）・`st.ch_map`（コード）。
    書く: `st.ts`・`st.bar`・`st.timeline`・`st.shift_of`・`st.key_of`・`st.spans`（コード）・`st.bars_of`。
    組み立てが決めたこと（曲頭の空き小節）は `song_fix.record_change` で曲の記録へ。
    検査が保証しているはずの形でないもの（コードの小節数・読めないコード名）が来たら `song_fix.AssemblyError`。
    """
    song, verbose = st.song, st.verbose
    design, melody = song.design, song.melody
    # 区間 → 調・移調量（転調を持ち回る）は `section_keys` の1か所（2026-09-30・声部も同じものを使う＝記-3）
    key_of, shift_of = section_keys(design)
    if design.modulations and verbose:
        print("  [4.5 転調] " + design.key
              + "".join(f" →（{m.at_section}）{m.key}" for m in design.modulations))

    # 旋律を tick へ置く。**イントロのぶんだけ後ろにずらす**（歌はイントロの後から始まる）。
    # 拍子から1小節の長さを決める（4/4=1920 / 3/4=6/8=1440）。以降 BAR ではなく bar を使う
    ts = design.time_signature                             # 選べる4つ（型 `TimeSignature`・2026-09-30）
    bar = bar_ticks(ts)
    # ---- ①測る → ②時間軸を組む → ③置く（2026-08-23）----
    #
    # 以前は「歌を並べながら tick を進め、並んだ結果から時間軸を測る」順だった。
    # 配置の都合（手前休符・小節への切り上げ）がそのまま時間軸に染み出し、
    # **伴奏とずれた**。位置ずれを直すために入れた切り上げが、その場で
    # **累積ずれ（最大4.1小節＝約12秒）**を生んだ。
    # 時間軸を**先に**決めれば、歌も伴奏も同じ座標から引くので**ずれ自体が起きない**。
    # 詳細は docs/plan/song_timeline.md。

    # **0秒ちょうどから声を出さない**（2026-08-22 指摘）。サビ始まりだと最初の区間から
    # 歌が始まり、再生した瞬間に声が出て「ぶつ切り」に聞こえる。息を吸う間を置く。
    # ★**間は組み立てが置く・曲全体をずらす**（2026-09-28 本人判断・同日に直し）。歌の中身の問題ではないので旋律へは返さない。
    #   旋律が書いた休みで足りなければ、**曲の頭に空の1小節**（`LEAD_IN`）を置き、歌・コード・伴奏を同じだけ後ろへずらす。
    #   ★以前（2026-08-22〜）は**最初の区間の歌だけ**を1拍ほど後ろへずらしていた——コードと伴奏は小節の頭のままなので、
    #   最初の区間の旋律が丸ごとコードに対して1拍ずれ、LLM が「この拍でこのコード」と書いた関係を崩していた（本人の指摘で発見）。
    #   1小節まるごと置くので、拍の頭もコードとの関係も変わらない。
    lead_bars = 0
    if design.sections and design.sections[0].has_vocal:
        beat_sec = 60.0 / design.bpm if design.bpm else 0.5
        want = int(round(max(1, -(-0.25 // beat_sec)) * Q))
        have = int(round((melody.phrases[0].rest_before if melody.phrases else 0.0) * Q))
        if have < want:
            lead_bars = 1
            from song_fix import record_change
            record_change(song, "melody", f"曲頭の間が {have / Q:g}拍しか無いので、曲の頭に空の1小節を置いた"
                                          "（歌・コード・伴奏とも同じだけ後ろへ。書いた関係は変えない）",
                          "0秒ちょうどから声が出ると、再生した瞬間にぶつ切りに聞こえる（2026-08-22）")
            if verbose:
                print(f"  [冒頭] 曲の頭に空の1小節を置いた（{bar / Q * beat_sec:.2f}秒・記録した）")

    # ★★**時間軸は設計の小節数のまま**（2026-09-28 本人方針「組み立ては一切補正しない」）。以前は歌のある区間の長さを旋律の
    #   実際の長さに合わせていた（目録の時間-1）——旋律の行が枠と合わないとき、区間の長さ・伴奏・コードの繰り返しまで変わる補正だった。
    #   行の長さは旋律の検査（`MelodyStep`）が枠と合わせさせ、直らなければ旋律の工程で止まる（2026-09-30）ので、旋律は区間に収まる
    timeline = SectionTimeline.build(design.sections, {}, bar, lead_bars=lead_bars)   # ②組む
    if verbose:
        print(f"  [時間軸] {timeline.total_bars}小節 = "
              f"{timeline.total_ticks / Q * 60 / design.bpm:.0f}秒\n"
              f"      {timeline.describe()}")
    # コードは**時間軸（設計の小節数）から**並べる。以前はここで小節を数え直して `bars_of` を作っていたが、
    # それが歌の時間軸とは別の3つ目の写しになっていた（ずれの温床）。
    # ※2026-08-20〜09-27 は歌のある区間の長さを旋律から逆算していた（歌が終わっても伴奏が25秒続いた事故の対策）。
    #   2026-09-28 から時間軸は設計どおりで、旋律の行の長さを旋律の検査が枠に合わせさせる（組み立ては補正しない）
    st.spans = place_chords(timeline, st.ch_map, shift_of, key_of, bar)   # コードを置く（試聴と同じ1か所・判-E）
    st.bars_of = timeline.bars_of()                        # **導出**（保存しない）
    st.ts, st.bar, st.timeline, st.shift_of, st.key_of = ts, bar, timeline, shift_of, key_of


def _need_timeline(st, who: str) -> None:
    """音符を置く前に時間軸ができているか（`build_timeline` を先に呼ぶ決まり）。無ければ呼び順の不具合として止める。"""
    if st.timeline is None or st.shift_of is None:
        raise AssemblyError(f"{who} の前に時間軸が無い——先に `song_place.build_timeline` を呼ぶ（呼び順の不具合）")


def _lay(phrases, cursor0, shift_of, grid: int, pieces_of) -> tuple[list[tuple], int]:
    """**旋律のフレーズを時間軸に置く**（歌 `place_vocal` と主旋律の楽器 `place_lead` が同じここを通る・2026-10-03）。

    `pieces_of(i, フレーズ)` はそのフレーズの `(詞, 階名, オクターブ, 枠の長さ, 鳴る長さ, 音の中の何番目か, タイの続きの数)` の並び
    （歌は `fit_phrase` の出力・主旋律の楽器は旋律の1音＝1つ）。詞が空の要素は時間だけ進めて置かない（`*`）。
    `cursor0(区間名)` は区間の頭の tick（`SectionTimeline.tick_from`）。
    戻り値は ((区間名, 詞, 高さ, 位置, 長さ, タイの続きの数, フレーズ頭か, フレーズ尾か) の並び, 次の音の手前で長さを切った数)。
    位置・長さの読み方は1つ——歌と主旋律の楽器で食い違わない（2本あると片方だけ直る）。
    """
    out: list[tuple] = []
    cursor: dict[str, int] = {}
    for i, p in enumerate(phrases, 1):
        if p.section not in cursor:                        # その区間の1フレーズ目
            cursor[p.section] = cursor0(p.section)          # 曲頭の間は時間軸の LEAD_IN（曲全体をずらす）
        tick = cursor[p.section] + int(round(p.rest_before * Q))
        sl = [list(x) for x in pieces_of(i, p)]            # (詞, 階名, 8ve, 枠, 鳴る長さ, 音の中の何番目か, タイの続きの数)
        # ---- ②位置を数える（まだ置かない）----
        raw: list[tuple] = []                               # (詞, 階名, 8ve, 位置, 鳴る長さ, 何番目か, タイの続きの数)
        at = tick
        for lyric, solfa, octv, beats, sound, sub, ties in sl:
            raw.append((lyric, solfa, octv, at, max(1, int(round(sound * Q))), sub, ties))
            at += max(1, int(round(beats * Q)))             # **枠ぶん進む**（差が促音の詰まりになる）
        # ---- ③グリッドへ乗せる（2026-09-08）----
        # ★**旋律が書く拍数は、拍のグリッドに乗るとは限らない。**
        #   実測（『琥珀色に灯る』BPM80）: `gap_after` が 0.1／0.2拍、`rest_before` が 1.2／1.3拍
        #   ——どの音符の長さでもない。位置は足し算で決まるので、**1つ入ると以降が全部ずれる**。
        #   結果 68音中50音（74%）がグリッドの外に出て「リズムに合ってない」と言われた。
        #   **その都度の絶対位置を丸める**ので、ずれは溜まらない（差分を丸めると累積する）。
        # ★**詰めて分けた音（何番目か≥1）は格子に乗せない**（2026-09-29）。4分に3モーラなら3連——16分の格子へ丸めると
        #   等分が崩れる。音の頭だけを丸め、分けた音は頭からの距離のまま置く（長さも格子で下限を取らない）。
        # ★**長さは丸めない**（2026-09-30・出-14③）。旋律の長さは ABC の16分の整数個なので、丸める必要があったのは
        #   **促音の詰まり**（鳴る長さを0.1〜0.25拍切る）だけで、そこを格子へ戻していた——0.5拍の音は予定の0.15拍が0.25拍の切りに、
        #   0.25拍の音は切れずに消えた。組み立ては補正しないので、書いたとおりの長さ（`fit_phrase` の鳴る長さ）のまま置く。
        #   位置（音の頭）は格子に乗せる（書いた位置は16分の倍数なので動かない）
        head_raw = head_st = 0
        for k, (lyric, solfa, octv, st_, du, sub, ties) in enumerate(raw):
            if sub == 0:
                head_raw, head_st = st_, int(round(st_ / grid)) * grid
                st_ = head_st
            else:
                st_ = head_st + (st_ - head_raw)
            raw[k] = (lyric, solfa, octv, st_, du, sub, ties)
        # ---- ④置く ----
        # `*`（歌わない印）は置かない——時間は上で進めてあるので、後ろの音はずれない
        sung = [j for j, r in enumerate(raw) if r[0]]
        for j, (lyric, solfa, octv, st_, du, sub, ties) in enumerate(raw):
            if not lyric:                                   # （全部 `*` なら sung は空で、ここで全部飛ぶ）
                continue
            if j + 1 < len(raw):                            # 次の音に食い込まない
                du = min(du, max(grid if raw[j + 1][5] == 0 else 1, raw[j + 1][3] - st_))
            out.append((p.section, lyric, solfa_to_midi(solfa, octv) + shift_of(p.section), st_, du, ties,
                        j == sung[0], j == sung[-1]))
        cursor[p.section] = (int(round(at / grid)) * grid) if raw else tick
    # ★★**最後に、音が重ならないことを保証する**（2026-09-09 追加）★★
    #
    #   上の `min(du, ...)` は**同じフレーズの中**しか見ていない。だが**区間の頭は
    #   設計の小節に固定**されている（`timeline.tick_from`）ので、前の区間の伸ばしが
    #   長いと**次の区間の1音目と同時に鳴る**——実際に2箇所で起きた（『穏やかな空白』）。
    #   旋律の工程で枠に収めたうえで（`MelodyStep`）、**ここでも必ず切る**（歌は単旋律で、
    #   重なりは V6 でも `score_validate` でも不正）。**位置は動かさない**（長さだけ切る）。
    #   ★**並べ替えてから見る**——区間ごとに積むので、前の区間の伸ばしが
    #   長いと**開始 tick の順に並んでいない**。隣どうしで比べると取りこぼす
    #   （実際に「2音を切り詰めた」と言いながら、検査は7箇所の重なりを見つけた）。
    #   ★**同じ位置に来た音は、不具合として止める**（2026-09-30）。旋律の行の長さは枠ちょうど（許容0）・区間ごとの行の数は
    #     グルーヴの行数（`MelodyStep`）で、枠の合計は区間の小節数（`song_melody._line_slots`）なので、旋律は区間の中に収まり、
    #     同じ位置に2つの音は来ない。直らなければ旋律の工程で止まる（`StepUnfixed`）。2026-09-28〜09-29 は後の音を鳴らさずに
    #     記録して進み、その前は1グリッド後ろへ送っていた（時間-4）。
    #     長さを次の音の手前で切るのは残す（伸ばしの丸めで次の音の頭にかかったときの、単旋律にする訳し方）
    order = sorted(range(len(out)), key=lambda k: out[k][3])
    durs = [x[4] for x in out]
    bumped = 0
    for a, b in zip(order, order[1:]):
        if out[a][3] + durs[a] > out[b][3]:
            if out[b][3] - out[a][3] < grid:
                raise AssemblyError(f"歌の組み立てで同じ位置に2つの音が来た: 「{out[a][1]}」と「{out[b][1]}」"
                                    f"（{out[b][3] / Q:g}拍目）——旋律の検査（行の長さ・行の数）が保証しているはずの形でない")
            durs[a] = max(grid, out[b][3] - out[a][3])
            bumped += 1
    return [x[:4] + (durs[k],) + x[5:] for k, x in enumerate(out)], bumped


def _melody_pieces(i, p) -> list[tuple]:
    """主旋律の楽器（詞なし）の1フレーズ: **旋律の1音＝1つ**（タイでまとめた音は1つの長い音のまま）。
    枠は音の長さ＋音のあとの小休み、鳴る長さは音の長さ（歌の `fit_phrase` と同じ読み方。詞の分け方・促音の詰まりは無い）。"""
    return [("♪", n.solfa, n.octave, n.beats + max(0.0, n.gap_after), n.beats, 0, 0) for n in p.notes]


def _lead_line(st) -> list[tuple]:
    """旋律を詞なしで時間軸に置く → `(区間名, LeadNote)` の並び（`st.lead_notes`）。歌と同じ置き方（`_lay`）。"""
    placed, _ = _lay(st.song.melody.phrases, st.timeline.tick_from, st.shift_of, GRID, _melody_pieces)
    return [(sec, LeadNote(key=key, start_tick=at, duration_tick=du)) for sec, _l, key, at, du, *_r in placed]


def place_lead(st) -> None:
    """**主旋律を詞なしで置く**（インスト・2026-10-03・§5-1）。旋律の音をそのまま時間軸に置く——位置・長さは歌（`place_vocal`）と
    同じ読み方（`_lay`）で、詞の印・調声は無い。歌の音符は無い（`st.notes` は空）。`build_timeline` のあとに呼ぶ。

    書く: `st.lead_notes`（`(区間名, LeadNote)`。楽器・音量・区間は `make_score` が編成と伴奏から付ける）・`st.notes`・`st.sec_of_note`（空）。
    """
    _need_timeline(st, "主旋律の組み立て（place_lead）")
    st.lead_notes = _lead_line(st)
    st.notes, st.sec_of_note = [], []
    st.log["lyric_fit"] = "（インスト・詞なし）"
    if st.verbose:
        print(f"  [主旋律] {len(st.lead_notes)}音を詞なしで置いた（楽器で弾く）")


def place_vocal(st) -> None:
    """**歌の組み立て**（2026-09-30 に `song_compose._compose_once_body` から切り出した）。歌もの・アカペラ。
    2026-10-03 から時間軸・調・コードは `build_timeline`（先に呼ぶ）が持ち、ここは**詞を音へ当てる・調声**だけ。

    読む: `song.design`・`song.melody`・`st.by_index`（詞）・時間軸（`build_timeline` が書いたもの）。
    書く: `st.notes`（調声つきの歌の音符）・`st.sec_of_note`・`st.lead_notes`（同じ旋律を詞なしで置いたもの。編成が主旋律を
    楽器でも重ねると決めたときに `make_score` が使う・F10②）。
    検査が保証しているはずの形でないものが来たら `song_fix.AssemblyError`（2026-09-30）。
    """
    _need_timeline(st, "歌の組み立て（place_vocal）")
    song, verbose = st.song, st.verbose
    design, melody = song.design, song.melody
    by_index, log = st.by_index, st.log
    timeline, shift_of, bar, bars_of = st.timeline, st.shift_of, st.bar, st.bars_of
    fit_log: list[str] = []                                # 詞と音符の過不足（来たら不具合・下で止める）

    def pieces(i, p):
        ss = LYRIC_LINE.read(by_index.get(i, ""))          # **旋律の1音ごとの印**（詰め・伸ばし・歌わない。検査と同じ記法の型で読む）
        got = fit_phrase(ss, p.notes, fit_log, f"フレーズ{i}: ")
        if fit_log:
            # ★印の数＝音の数・詞の無いフレーズは詞の検査（`LyricsStep`）が見ていて、直らなければ詞の工程で止まる
            #   （`StepUnfixed`）ので、ここで数が合わないことは無い。来たら不具合として止める（2026-09-30。
            #   以前は合う分だけ置き、詞の無い音は鳴らさず・音の無い印は歌わずに記録して進んでいた）
            raise AssemblyError("歌の組み立てに、詞の検査を通ったはずの詞が来た: " + "／".join(fit_log))
        # ---- ①母音単独のモーラへ「伸ばし」を渡す（2026-09-08）----
        # ★「つめた**い**」の「い」は独立した音ではなく**二重母音の尾**。
        #   旋律が「た」に長い音を書くと、**言葉の途中で伸びて「い」が遅れて出る**。
        #   本人の言葉:「**冷たいまで同じタイミングで言い切るべき**」——
        #   つまり つ・め・た・い は同じ間隔で、**伸ばすのは最後の「い」**。
        #   手前を伸ばして隙間を埋める実装を先に入れたが、それは**逆**だった
        #   （「た」が 0.94秒になり、かえって「い」が遅れた）。**長さを母音側へ渡す。**
        #   ★★**2026-09-27 から機械は長さを入れ替えない**（目録の詞-4）★★ 入れ替えは旋律の工程が書いたリズムと
        #   小節頭の音を変えていた（『風は海を越えて』で Am の頭がミ→レ、Em7 の頭がレ→ド）。しかも当たるのは
        #   「て|おもう」「か・お・り」のような語頭・語中の母音が多かった（保存済み47曲で音の4%・フレーズの25%）。
        #   尾になりうる所を所見にする案（`vowel_tail_findings`）も誤報が多く、同じ日に外した。
        return got

    # ③置く。**各区間の頭 tick は時間軸から引く**——自分で足し込まないので、ずれようがない。
    #
    # 同じ区間の続きのフレーズは、直前のフレーズの終わりから続ける。
    #
    # ★**「行を区間の中の割り当て位置へ直接置く」形も試したが、戻した**（2026-09-09）。
    #   区間の長さは**旋律の実長から決まる**（`SectionTimeline.build` の `needed_ticks`）ので、
    #   設計の小節数を前提に置き場所を計算すると**時間軸と食い違う**——
    #   実際に区間が 8小節→7小節に縮み、重なりが 2→4箇所に増えた。
    #   正しい直し方は**行が枠をぴったり埋めるようにする**ことで、そうすれば
    #   足し込みでも行は小節の頭から始まる（旋律の行は枠＝小節の整数倍で書かれる）。
    # ★**16分に固定**（2026-09-27・目録の時間-3）。旋律は ABC（1単位＝16分）で書くので、16分へ丸めても**何も動かない**
    #   （＝組み立て）。以前は `note_grid` が「16分の値が無ければ8分」を選び、8分へ丸めていた——書いた位置は8分の倍数でも、
    #   促音の詰まり（鳴る長さを0.1〜0.25拍切る）が8分へ丸められて消えていた（目録の詞-4 の隣・L4）。
    #   `note_grid` は旧形式（枠）で小数の拍が混ざった頃の保護で、測る側（`_div`）ではまだ使う。
    # ★位置・長さ・格子・重なりの読み方は `_lay` の1か所（主旋律の楽器 `place_lead` と同じ・2026-10-03）
    placed, _bump = _lay(melody.phrases, timeline.tick_from, shift_of, GRID, pieces)
    notes: list[SongNote] = []
    head: set[int] = set()
    tail: set[int] = set()
    for _sec, lyric, key, at, du, ties, is_head, is_tail in placed:
        if is_head:
            head.add(len(notes))
        if is_tail:
            tail.add(len(notes))
        notes.append(SongNote(lyric=lyric, key=key, start_tick=at, duration_tick=du, ties=ties))
    if _bump and verbose:
        print(f"  [重なり] {_bump}音の長さを、次の音の手前まで切り詰めた")
    # **音符 → 区間は導出する**（保存しない）。以前は配置のときに別途ためており、
    # 音符を足し引きすると長さが食い違った（実際に手で直す羽目になった）
    sec_of_note = timeline.sections_of(notes)
    # ※詞と音の数が合わない所・同じ位置に重なった音の記録（2026-09-27〜）は 2026-09-30 に外した（どちらも上で止める）
    log["lyric_fit"] = "（過不足なし）"

    # ---- 拍頭の音をコードへ寄せる——**2026-09-26 にやめた**（本人判断）----
    # 旋律は LLM がコードを見て書いており、構成音から外した音は**外した意図のある音**（経過音・かかり音）。
    # それを機械が±1半音動かしていた（『明日への足音』で19音・構成音 59%→70%）。歌い出しの位置を
    # 機械で動かしていたのと同じ形なので、同時にやめた。関数 `snap_to_chords` も 2026-09-28 に消した（本人承認・呼ばれていないコードの削除）。

    # ※「歌と伴奏の位置合わせ」の点検（組み上がった歌が区間から食い込む・はみ出す・コードが末尾まで無いを実測し、
    #   あれば点検役を呼ぶ）は 2026-09-29 に外した（本人「必要がいらないならステップとして消してもいい。シンプルにしたい」）。
    #   保存済み58曲で一度も所見が出ず、原因は各工程の形式の検査（旋律の行の長さ・行の数・グルーヴの行数・コードの小節数）が
    #   工程の中で止める。直らずに残った所見は旋律・コードの点検役に渡っていて、同じことを2回見ていた
    #   （2026-09-30 から、直らなければその工程で曲づくりが止まる）。
    #   古い記録（`alignment`）を読む所も消した（リリース前なので古い記録は読めなくてよい・本人）

    # 音量の起伏を付ける対象は「サビ」という**名前**ではなく、**最も熱い区間**（intensity 最大）。
    # 名前で探すと、Cメロが山の曲や独自の呼び方をした曲で当たらない
    peak = max((s for s in design.sections if s.has_vocal),
               key=lambda s: s.intensity, default=None)
    pk_bars = bars_of.get(peak.kind) if peak else None
    chorus_ticks = ((pk_bars[0] - 1) * bar, pk_bars[1] * bar) if pk_bars else None
    n_tuned = apply_tuning(notes, head, tail, chorus_ticks, tempo=float(design.bpm))   # 揺れは秒で決める（表情-7）
    log["tuning"] = (f"ビブラート/しゃくり/抜き/サビの音量を {n_tuned}/{len(notes)} 音に付与"
                     f"（フレーズ頭 {len(head)}・語尾 {len(tail)}）")
    song.record(StepRecord(step="tuning", reason=log["tuning"]))    # 機械処理なので LLM 申告は無い
    if verbose:
        print(f"  [5.5 調声] {log['tuning']}")
    st.notes, st.sec_of_note = notes, sec_of_note
    # 同じ旋律を詞なしで置いたもの（編成が主旋律を楽器でも重ねると決めたときに `make_score` が使う・F10②）
    st.lead_notes = _lead_line(st)


# ---- 楽譜にまとめる -------------------------------------------------------------

def make_score(st) -> None:
    """**楽譜（`SongScore`）にまとめる**（2026-09-30 に `song_compose._compose_once_body` から切り出した）。伴奏の関門のあと・声部の前に呼ぶ。

    読む: 歌の組み立て（`st.notes`・`st.spans`・時間軸・`st.lead_notes`）・伴奏の組み立て（`st.spec`）・編成（`st.inst_gms`・
    `song.instrumentation` の `lead`・`lead_sections`）・伴奏（`song.arrangement` の `lead_gain`）・`song.lyrics`・`song.form`。
    書く: `st.score`（形 `form`・主旋律の楽器 `lead` も）。
    ※以前は編集で工程を再実行するための構造化データ（`log["_meta"]`・旧 steps.json の形）もここで書いていた。読む所が無くなったので
      2026-09-30 に外した（曲の構造は `Song.to_json` の `context`＝時間軸・コード表・編成などが持つ）。
    """
    song = st.song
    design, words = song.design, song.lyrics
    form = song_files.as_form(getattr(song, "form", "song"))
    # ★**形と組み立ての中身が食い違っていたら止める**（2026-10-03・組み立ては補正しない）。アカペラに伴奏を足さない・インストに歌を足さない。
    #   どちらも走る工程の集合（`song_gate.FORM_STEPS`）が保証している——来たら工程の並びの不具合
    if form == "acappella" and st.spec is not None and (st.spec.parts or st.spec.hand_bars):
        raise AssemblyError("アカペラの楽譜に伴奏の組み立てが来た（伴奏の工程はアカペラでは走らないはず）")
    if form == "instrumental" and st.notes:
        raise AssemblyError(f"インストの楽譜に歌の音符が{len(st.notes)}音来た（インストは詞の工程が走らず、主旋律は楽器で弾く）")

    # ★**見せる歌詞も保存する**（2026-09-21）。`lyrics_full` は歌うためのかな表記なので、
    #   字幕・投稿・歌詞表示に使えない（`ぬるいかぜのなかうみぞいおあるく…`）。
    _disp = lyric_display.full_text(
        {p.index: p.text for p in words.phrases},
        {p.index: getattr(p, "display", "") or "" for p in words.phrases}) if made(words) else ""   # インストの詞は空（F1）
    # ★**楽譜の曲名は工程『曲名』の値を写す**（2026-10-03・`Song.step_title`＝曲名の工程 → テーマの仮の曲名）。組み立ては毎回走るので、
    #   曲名だけを作り直しても（ほかの工程は保存から戻る）楽譜の曲名は新しくなる。編成の見出しの値（`st.title`）には頼らない
    st.score = SongScore(title=song.step_title() or st.title, form=form, tempo=float(design.bpm), time_signature=st.ts,
                         lyrics_full="".join(n.lyric for n in st.notes), vocal_notes=st.notes,
                         lyrics_display=_disp,
                         extra_voices=[],                  # 声部は後の工程（`song_band.run_voices`）が書く
                         # 伴奏・主旋律の楽器が無い形（アカペラ・重ねない歌もの）も同じ欄を空で持つ（F1。機械は中身を作らない）
                         chords=st.spans, arrangement=st.spec if st.spec is not None else ArrangementSpec(),
                         lead=lead_part(st),
                         # 声の種類も楽譜に持つ（メモリだけだと、編成が走らない編集で抜けた・2026-09-26）。
                         # インストは歌わないので持たない（同じプロセスで前に作った曲の声を楽譜へ写さない）
                         voice_choice={} if form == "instrumental" else dict(__import__("voice_style").CHOICE))


def lead_part(st) -> LeadPart:
    """**主旋律を弾く楽器のパート**（2026-10-03・§5-2・F2・F10②・F19・F20）。編成が `lead` を書いていなければ中身が空の
    `LeadPart()`（F1・形でデータの形を変えない。以前は None）。

    音は旋律を詞なしで置いたもの（`st.lead_notes`・歌と同じ置き方）のうち、編成の `lead_sections` の区間の音だけ
    （空＝主旋律のある区間の全部）。**書いた範囲どおり**——組み立ては範囲を広げも狭めもしない。楽器は編成の `lead` 番目の音色
    （伴奏のパートと同じ `gm<bank>-<番号>`）、音量は伴奏の `lead_gain`（組み立ては既定値で埋めない）。
    形ごとの決まり（インストは必ず書く・空の区間だけ／アカペラは書けない／`lead` があれば `lead_gain` も）は編成・伴奏の検査が
    保証している。来たら不具合として `AssemblyError`。
    """
    song = st.song
    form = song_files.as_form(getattr(song, "form", "song"))
    ins = getattr(song, "instrumentation", None)
    n = ins.lead if ins is not None else None              # 欄は必須（2026-10-04・既定値をやめた）。編成の無い曲だけ None
    if n is None:
        if form == "instrumental":
            raise AssemblyError("インストなのに主旋律を弾く楽器（編成の lead）が無い——編成の検査が保証しているはずの形でない")
        return LeadPart()
    if form == "acappella":
        raise AssemblyError(f"アカペラなのに主旋律を弾く楽器（編成の lead={n}）が来た——編成の検査が保証しているはずの形でない")
    if not 1 <= int(n) <= len(st.inst_gms or []):
        raise AssemblyError(f"主旋律を弾く楽器の番号 {n} が編成の楽器（{len(st.inst_gms or [])}本）の範囲の外")
    gain = getattr(getattr(song, "arrangement", None), "lead_gain", None)
    if gain is None:
        raise AssemblyError("主旋律を弾く楽器があるのに、伴奏の lead_gain（その音量）が無い——伴奏の検査が保証しているはずの形でない")
    secs = list(ins.lead_sections)
    if form == "instrumental" and secs:
        raise AssemblyError(f"インストの主旋律の区間（lead_sections）が書かれた: {secs}（インストは全部の区間で弾く）")
    names = [s.kind for s in song.design.sections]
    if bad := [s for s in secs if s not in names]:
        raise AssemblyError(f"主旋律を弾く区間に設計に無い区間名: {bad}（設計の区間名を1字も違わずに・編成の検査が保証しているはず）")
    notes = [ln for sec, ln in st.lead_notes if not secs or sec in secs]
    if not notes:
        raise AssemblyError(f"主旋律を弾く楽器に置く音が1つも無い（区間 {secs or '全部'}）")
    bank, gm = st.inst_gms[int(n) - 1]
    return LeadPart(instrument=int(n), timbre=f"gm{bank}-{gm}", gain=float(gain), sections=secs, notes=notes)
