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

"""**旋律を測る側**——リズムの物差し（ヒット曲の実測）・旋律のリズムの数字・息継ぎの所見・区間の頭の間・入るモーラの数。

2026-09-30 に `song_compose` から分けた。どれも**旋律を読んで数えるだけ**で、旋律を書き換えない。
使う側: 旋律の工程（`song_melody_step.run_melody`＝点検役への所見と記録）・グルーヴの依頼文（`song_outline.run_groove`＝入るモーラ）・
旋律の依頼文（`song_melody_prompt.reference_text`＝ジャンルの実測）。
"""
from __future__ import annotations

import json
from pathlib import Path

import song_context as ctx
from song_design import Q, bar_ticks, split_bar_chords
from song_gate import Finding
from song_llm import solfa_to_midi


#: リズムの物差し（**POP909＝ヒット曲909曲の実測 p25〜p75**）。
#: **秒で持つ**のが要点——テンポと「1音の秒数」は無関係（相関 −0.08）、
#: 「1音の拍数」とは強い相関（+0.58）だった＝プロは速い曲ほど長い音価を使う。
RHYTHM_BAND: dict[str, tuple[float, float]] = {
    "休符率":        (0.25, 0.42),     # 休みの時間割合（中央 0.34）
    "ひと息の秒":     (2.5, 6.5),      # ひと息で歌う長さ・中央値（中央 3.7）
    "息継ぎの頻度":   (0.11, 0.32),    # 歌う1秒あたりの切れ目の回数（中央 0.20＝5秒に1回）
    "1音の秒":       (0.23, 0.36),    # 1音の長さ・中央値（中央 0.29）
    "音の密度":      (1.39, 1.83),    # 1秒あたりの音数（中央 1.58）
    "拍頭率":        (0.115, 0.329),  # 拍の頭から始まる音の割合（中央 0.246）
    "型の再出現":     (0.903, 0.975),  # 4音のリズム型が曲中で再び現れる割合（中央 0.951）
    # ---- 「伸ばし」（2026-09-07 追加）----
    # 本人の指摘:「単語の伸ばすとか、曲の中で普通どうやってるんだろう?
    #   **それが微妙に再現しきれてない感じがした**」
    # POP909 の旋律 27,044フレーズを、ひと息の塊ごとに「いちばん長い音」で測った。
    # **伸ばす場所は合っていた**（末尾が最長 68.3% / 自作 75.0%）。**違うのは量と、そのばらつき**。
    #   自作: 伸ばし 0.60秒・ばらつき 0.08秒（0.56〜0.64＝**毎回ほぼ同じ**）
    # ★**テンポで動かさない**——伸ばしの秒数とテンポの相関は **+0.07**（拍で見ると +0.28）。
    #   BPM帯ごとの中央値も 1.08 / 1.13 / 1.31 / 1.10秒でほぼ一定＝**速い曲ほど拍数を増やして**
    #   同じ「約1秒の伸び」を作っている。BPM200 なら 3〜4拍に相当する。
    "伸ばしの秒":     (0.67, 1.44),    # 各ひと息の最長音・その中央値（772曲の p25〜p75・中央 1.01）
    "伸ばしのばらつき": (0.38, 0.93),    # 同 p75−p25（中央 0.63）
    # ★**ばらつきより、こちらが本体**（2026-09-09）。
    #   「大きく区切るところほど長く伸ばす」という工夫が効いているか。
    #   実測: J-POP +0.297（正だった曲 88%）／中国ポップス +0.382（同 97%）。
    #   **ばらつきは工夫なしでも作れる**が、この対応は工夫がないと出ない。
    "伸ばしの必然性":  (0.05, 1.00),    # 末尾音の長さ と そのあとの休みの長さ の相関
    # ---- グリッド（2026-09-08 追加）----
    # ★**拍頭率は「グリッドに乗っているか」を見ていない。**
    #   拍頭率 0.235＝帯の中、なのに **68音中50音（74%）が16分グリッドの外**という曲が出荷され、
    #   本人に「全体的にリズムに合ってない」と言われた（『琥珀色に灯る』2026-09-08）。
    #   裏拍・16分裏に置くのは正しい——**グリッドのどこでもない場所**に置くのが事故。
    #   配置側（`GRID` への丸め）で必ず 1.0 になるが、**そう思って間違えた**ので測り続ける。
    "グリッド率":      (0.999, 1.0),    # 16分グリッドに乗っている音の割合
}


def _mora_budget(seconds: float, bpm: float, genre: str = "pop",
                 notes_per_beat: float = 0.0, tail_rest: float = 0.30) -> int:
    """その区間に入る**モーラの合計**（＝音の数）。**そのジャンルの実測の密度**で掛ける。

    ★**「秒数 ÷ 1音の長さ」ではない。** 1音 0.29秒は**鳴っている長さ**で、
    次の音までの間隔ではない——歌は時間の 25〜42% が休みなので、
    その式は**約2倍多く見積もる**。

    ★**`rhythm_band(bpm)` の真ん中を使う**（2026-09-11）。ここには
    **テンポごとの適正が既に入っている**——速い曲は参照が22曲しかないので、
    **本人の試聴で「2.0〜2.6 が適正」と詰めた**経緯がある（`rhythm_band` の説明を読むこと）。

    > ★**遠回りした記録**（同日）: いったん「テンポでは変えない」に変え、
    > `song_reference.json` の**ジャンル中央値**（pop 1.708）を使った。
    > 根拠は洋楽143曲で「テンポと密度の相関 −0.016」だったが、
    > **そのデータは 160BPM 超が6曲しかなく**、しかも旋律トラックの代用が粗かった。
    > 結果、BPM195 の曲で**予算が半分**になり、グルーヴが行を減らして
    > **1行4小節・伸ばし2.9秒・休符率0.074**の曲を作った。
    > **既にある `rhythm_band` を見ずに、弱いデータで上書きした**のが誤り。
    """
    # ★★**数字で受ける**（2026-09-16・本人「**パラメータ指定でいけないのかな**」）★★
    #   それまで `塊` / `まくし立て` という**名前を2つ**持っていたが、中身は全部数字だった。
    #   名前にすると**中間が作れない**（「行末を少しだけ空ける」が書けない）。
    #
    #       入るモーラ ＝ 拍数 × notes_per_beat   （`tail_rest` は掛けない・2026-09-18）
    #
    #   検算（BPM200・16小節＝64拍）:
    #     ・1.0 / 0.30（ふつう）    → 45 … 旧「塊」の 43 とほぼ同じ
    #     ・1.35 / 0.0（まくし立て） → 86 … 旧「まくし立て」の 87 とほぼ同じ
    #   参考曲（BPM204・作者配布の公式 MIDI）の実測は **4.50 音/秒 ＝ 1.32 音/拍**で、
    #   まくし立ての 1.35 はそこから取っている。
    if notes_per_beat:
        # ★★**`tail_rest` は掛けない**（2026-09-18・本人指示）★★
        #   当初は「行末を空けるぶん言葉が減る」として掛けていたが、**2つの軸を1つの数に
        #   掛けたせいで、片方を上げるともう片方で戻された**。実際、「Aメロの言葉を増やして」と
        #   頼んだら `notes_per_beat` 0.9→1.0 と同時に `tail_rest` 0.0→0.1 が上がり、
        #   **28.8 → 28.8 でぴったり相殺**——モーラ数が1つも増えなかった（2026-09-17）。
        #
        #   **行末の空きは「置き方」**（伸ばすか休むか）で、**入る量とは別の軸**。
        #   分けたので、`notes_per_beat` が**そのまま密度**になる:
        #
        #       1秒あたりの音数 ＝ notes_per_beat × BPM ÷ 60
        #
        #   人が「言葉を増やして」と言ったとき、**動かす数字が1つに決まる**。
        return max(4, int(round(seconds * bpm / 60.0 * notes_per_beat)))
    lo, hi = rhythm_band(bpm).get("音の密度", (1.4, 2.0))   # 指定が無ければ従来どおり帯の中央
    per_sec = (lo + hi) / 2
    return max(4, int(round(seconds * per_sec)))


_REFERENCE: dict | None = None


def reference(genre: str = "pop") -> dict:
    """**そのジャンルの実測値**を返す（2026-09-09 新設）。止めるための帯ではない。

    ★**単一の帯をやめた理由**（実測）:

    | 分かったこと | 数字 |
    |---|---|
    | p25〜p75 は**定義上ヒット曲の半分を外す** | 全帯に収まるヒット曲 **0/909曲**・平均 5.5/10項目が外れる |
    | 中央値は「正解」でなく**ひとつの型の位置** | 909曲は4つの型に分かれ、中央値は最大の型（40%）に寄っている |
    | 根拠が**中国のポップス**だった | J-POP は拍頭率 0.19（中国 0.12）・密度 1.71（1.56）・伸ばし 0.80秒（1.05）|
    | **ジャンルで大きく違う** | ひと息はジャンル間で **2.20〜10.22秒（4.6倍）** |

    値は `[p20, 中央, p80]`。**日本語の歌を作るので `pop` は J-POP 100曲**が根拠。
    出典は `song_reference.json` の `_出典`。
    """
    global _REFERENCE
    if _REFERENCE is None:
        p = Path(__file__).with_name("song_reference.json")
        _REFERENCE = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    g = (genre or "pop").strip().lower()
    return _REFERENCE.get(g) or _REFERENCE.get("pop") or {}


def rhythm_band(tempo: float) -> dict[str, tuple[float, float]]:
    """テンポに応じた帯を返す（2026-09-07 新設）。

    ★**`RHYTHM_BAND` は速い曲に合っていなかった**。素にした POP909 は
    **909曲中 872曲が 130BPM 未満**——バラード・ミドルテンポ中心で、
    BPM200 の元気な曲を代表していない。速い曲（130BPM 以上・22曲）だけ取り出すと
    **1音の中央値は 245ms**（全体 290ms より短い）。

    さらに本人の試聴で、**帯の中（密度 1.64音/秒・1音 300ms）なのに
    「言葉が少なくて間延びしてる」**と言われた（2026-09-07『青空めがけて』BPM200）。
    速い曲の参照が 22曲しかないので、**そこは本人の評価で詰める**:

      なついろのそら 3.57音/秒 … 息継ぎ0・詰まりすぎ
      飛沫を飛ばして 2.75音/秒 … 別の理由（拍への乗り方）で不合格
      青空めがけて   1.64音/秒 … **間延び**
      → 適正は **2.0〜2.6** あたり

    **遅い曲・中テンポの帯は動かさない**（そちらは 872曲ぶんの裏付けがある）。
    """
    band = dict(RHYTHM_BAND)
    if tempo >= 150:
        band["1音の秒"] = (0.19, 0.29)                     # 速い曲の実測 p25 0.195 / 中央 0.245
        band["音の密度"] = (1.9, 2.6)                      # 本人の評価で挟んだ範囲
    return band


def rhythm_metrics(melody, tempo: float, *, runs_at: list | None = None) -> dict[str, float]:
    """旋律のリズムを測る。**所見・やり直し・採否のすべてがこの1か所を見る**。

    ★**測る場所を分けると必ず食い違う**（2026-09-06 に実際にやった）。
    所見側にだけ「ひと息の長さ」があり、やり直し側に無かったため、
    やり直しが**フレーズを17秒に伸ばす改悪を「改善」と判定**した。

    返すのは `RHYTHM_BAND` と同じキー＋補助値（`ひと息の最長秒`・`音数`）。
    `runs_at` にリストを渡すと、ひと息ごとに (区間名, 始まりの行, 終わりの行, 拍) を入れる（行は区間の何行目か・1始まり。
    点検役に「何秒・どの行」を見せるため・2026-10-03）。

    ★**ひと息は区間の頭で閉じる**（2026-10-03・組G1）。位置は区間ごとに 0 から数え直すのに、塊は区間をまたいで続いていたので、
      前の区間の終わりと次の区間の頭に 0.25秒以上の休みが無いと、塊の長さが「新しい区間の位置 − 前の区間の位置」になり
      **負の値**が出ていた（中央値が下がって「ひと息が短い」と見える）。間奏など歌の無い区間が間に入るかは旋律だけでは分からないので、
      曲の時間軸でつなぐのでなく区間の頭で閉じる（区間の切れ目を休みなしで歌い続ける所は、2つのひと息に分かれて数える）。
    """
    import statistics as _st

    phrases = getattr(melody, "phrases", None)
    if not phrases:
        return {}
    spb = 60.0 / max(1.0, tempo)                           # 1拍の秒数

    runs: list[float] = []                                 # ひと息で歌う長さ（拍）
    gaps: list[float] = []                                 # 息継ぎ（拍）
    durs: list[float] = []                                 # 1音の長さ（拍）
    inner: list[float] = []                                # フレーズ内の小休み（拍）
    onsets: list[float] = []                               # **区間の頭（小節線）からの**発音位置（拍）
    # ★**区間ごとに位置を積み上げる**（2026-09-06 修正）。配置の実体（`song_place.place_vocal`）は
    #   `cursor[section]` を区間の頭（小節線）から始め、**2本目以降のフレーズは前の
    #   フレーズの終わりから続ける**。フレーズごとに 0 へ戻して測ると、
    #   **2本目以降の拍頭率が実際と食い違う**（実測で 0.453 と 0.240 に割れた）。
    #   ここは配置と同じ数え方でなければ、直させたいものを測れていないことになる。
    cursor: dict[str, float] = {}
    SPLIT = 0.25 / spb                                     # 0.25秒ぶんの拍数（POP909 と同じ区切り）
    run_from: float | None = None                          # いま歌っている塊の開始位置（拍）
    prev_end: float | None = None
    holds: list[float] = []                                # 各ひと息の**いちばん長い音**（拍）
    run_durs: list[float] = []                             # いま歌っている塊の音価
    # ★**「工夫があったところが長く伸ばされているか」**を測るための組（2026-09-09）。
    #   (その息の末尾音の長さ, そのあとの休みの長さ) —— 詳しくは `hold_fit` を見ること。
    tail_gap: list[tuple[float, float]] = []
    nth: dict[str, int] = {}                               # 区間の何行目か（フレーズ＝行）
    run_start: tuple[str, int] | None = None               # いまの塊の始まりの (区間, 行)
    run_last: tuple[str, int] | None = None                # いまの塊の最後の音の (区間, 行)
    prev_sec: str | None = None
    last_gap = 0.0                                         # 直前の音の後ろの空き（拍）

    def close_run(end: float, gap: float = 0.0) -> None:
        nonlocal run_from, run_start
        if run_from is not None:
            runs.append(end - run_from)
            if runs_at is not None and run_start is not None and run_last is not None:
                runs_at.append((run_start[0], run_start[1], run_last[1], end - run_from))
            run_from = None
            run_start = None
        if run_durs:                                       # **塊ごとの伸ばし**を1つ取る
            holds.append(max(run_durs))
            if gap > 0:
                tail_gap.append((run_durs[-1], gap))       # 末尾音と、そのあとの休み
            run_durs.clear()

    for ph in phrases:
        sec_k = str(getattr(ph, "section", "") or "_")
        nth[sec_k] = nth.get(sec_k, 0) + 1
        rb = float(ph.rest_before or 0.0)
        at = cursor.get(sec_k, 0.0) + rb
        if prev_sec is not None and sec_k != prev_sec and prev_end is not None:
            close_run(prev_end, last_gap + rb)             # 区間の頭で閉じる（上の説明）
        prev_sec = sec_k
        if rb:
            gaps.append(rb)
            if rb >= SPLIT and prev_end is not None:
                close_run(prev_end, rb)                    # 息継ぎで塊が切れる（休みの長さも渡す）
        for n in ph.notes:
            b = max(0.0, float(n.beats))
            gp = max(0.0, float(getattr(n, "gap_after", 0.0) or 0.0))
            if run_from is None:
                run_from = at
                run_start = (sec_k, nth[sec_k])
            run_last = (sec_k, nth[sec_k])
            last_gap = gp
            onsets.append(at)
            durs.append(b)
            run_durs.append(b)
            at += b
            prev_end = at
            if gp:
                inner.append(gp)
                if gp >= SPLIT:
                    close_run(at, gp)                      # 大きい間は息継ぎ扱い（同）
            at += gp
        cursor[sec_k] = at
    close_run(prev_end if prev_end is not None else 0.0)
    if not runs:
        runs = [0.0]

    # ★**休符率は「音が鳴っていない時間」全部**（2026-09-07）。
    #   それまでフレーズ間（`rest_before`）しか数えておらず、
    #   **フレーズ内の小休みを表現できない模型**だったので、
    #   帯（25〜42%）に構造的に届かなかった（息継ぎだけでは上限 21.7%）。
    note_time = sum(durs)
    rest = sum(gaps) + sum(inner)
    sung = sum(runs)                                       # 歌っている塊の合計（小休みを含む）
    total = note_time + rest
    if total <= 0 or not durs:
        return {}

    on_beat = sum(1 for x in onsets if abs(x - round(x)) < 0.05) / len(onsets)
    # **その曲のグリッドに乗っているか**（拍の頭かどうかとは別の話）。
    # ★**16分固定で測ってはいけない**——8分でできた曲が16分裏へ落ちても 1.0 と出てしまい、
    #   「歌の53%が伴奏の鳴らない位置にある」状態を見逃す（2026-09-08 実測）。
    _div = Q / note_grid(melody)                           # 8分なら 2・16分なら 4
    on_grid = sum(1 for x in onsets if abs(x * _div - round(x * _div)) < 0.02) / len(onsets)

    q = [max(1, round(b * 4)) for b in durs]               # 16分単位に丸めた音価
    grams = [tuple(q[i:i + 4]) for i in range(len(q) - 3)]
    if len(grams) >= 8:
        from collections import Counter
        again = sum(v for v in Counter(grams).values() if v >= 2) / len(grams)
    else:
        # 短すぎて測れない＝減点しない（帯の下限を返す）。
        # ★`rhythm_band(tempo)` を使うこと——ここを `_b` と書いていて未定義だった
        #   （音符8個未満の旋律で必ず落ちる。2026-09-07 の一括置換で混入）
        again = rhythm_band(tempo)["型の再出現"][0]

    # 息継ぎ＝**0.25秒以上の休み**（POP909 の数え方に合わせる）。
    # フレーズ間か中かは問わない——聴く側には同じ「間」に聞こえる。
    breaths = [g for g in (gaps + inner) if g >= SPLIT]

    # **伸ばし**——ひと息ごとの最長音。中央値（どれだけ伸ばすか）と
    # p75−p25（**伸ばし分けているか**）の2つを見る。
    # ★ばらつきのほうが効く。自作は 0.56〜0.64秒に固まっており、
    #   「伸ばしてはいるが、いつも同じだけ」＝**歌でなく朗読に聞こえる**原因になっていた。
    # ★★**「工夫があったところが長く伸ばされているか」**（2026-09-09・本人指示）★★
    #
    # それまで「伸ばしのばらつき（p75−p25）」だけを見ていた。**これが誤りだった。**
    # 本人の指摘:「**ばらつきってランダムな事象なの? 何かの工夫の結果ではないの?**
    #   もし工夫だったら、**工夫をしてるかしてないかに基づいて評価**しないとじゃない?」
    #
    # 実測して確かめた（RWC-P 95曲・POP909 719曲）:
    #
    # | 調べたこと | 結果 |
    # |---|---|
    # | 伸ばしの列は**周期的か**（4行ごとに長い等）| **ほぼ無相関**——混ぜても変わらない＝周期ではない |
    # | 末尾音の長さ ↔ **そのあとの休みの長さ** | **J-POP +0.297／中国 +0.382**。**正だった曲が 88%／97%** |
    #
    # つまり **「大きく区切るところほど長く伸ばす」**という一貫した工夫があり、
    # それが結果としてばらつきに見えていた。**ばらつきは工夫なしでも作れる**——
    # でたらめに長短を混ぜれば広がりは出るが、音楽にはならない。
    # だから測るのは**広がりでなく対応**にする。
    if len(tail_gap) >= 4:
        _t = [x[0] for x in tail_gap]
        _g = [x[1] for x in tail_gap]
        _mt, _mg = sum(_t) / len(_t), sum(_g) / len(_g)
        # ★変数名に注意——このスコープでは `_st` が `import statistics as _st` に使われている。
        #   同じ名前を使って**中央値の計算を壊した**（2026-09-09・実行時に発覚）。
        _dt = (sum((x - _mt) ** 2 for x in _t)) ** 0.5
        _dg = (sum((x - _mg) ** 2 for x in _g)) ** 0.5
        hold_fit = (sum((a - _mt) * (b - _mg) for a, b in zip(_t, _g)) / (_dt * _dg)
                    if _dt > 1e-9 and _dg > 1e-9 else 0.0)
    else:
        # 標本が足りない＝判定しない。**帯の中央を返して減点しない**
        hold_fit = sum(RHYTHM_BAND["伸ばしの必然性"]) / 2

    if len(holds) >= 4:
        _h = sorted(x * spb for x in holds)

        def _pct(p: float) -> float:
            i = (len(_h) - 1) * p
            lo, hi = int(i), min(int(i) + 1, len(_h) - 1)
            return _h[lo] + (_h[hi] - _h[lo]) * (i - lo)

        hold_mid, hold_spread = _pct(0.5), _pct(0.75) - _pct(0.25)
    else:                                                  # 短すぎて測れない＝減点しない
        hold_mid = sum(RHYTHM_BAND["伸ばしの秒"]) / 2
        hold_spread = sum(RHYTHM_BAND["伸ばしのばらつき"]) / 2

    return {
        "休符率": rest / total,
        "ひと息の秒": _st.median(runs) * spb,
        "息継ぎの頻度": (len(breaths) / (sung * spb)) if sung > 0 else 0.0,
        "1音の秒": _st.median(durs) * spb,
        "音の密度": len(durs) / (total * spb),
        "拍頭率": on_beat,
        "型の再出現": again,
        "伸ばしの秒": hold_mid,
        "伸ばしのばらつき": hold_spread,
        "伸ばしの必然性": hold_fit,
        "グリッド率": on_grid,
        # ---- 補助（帯を持たない。所見の文面に使う）----
        "ひと息の最長秒": max(runs) * spb,
        "音数": float(len(durs)),
        "歌う秒": sung * spb,
    }


#: 音を置くグリッドの**最小**＝16分音符（1拍＝`Q` なので `Q // 4`）。実際に使う値は `note_grid()`。
GRID = Q // 4

# ※二重母音・長音の尾の所見（`vowel_tail_findings`）は 2026-09-27 に足して、同じ日に外した（本人判断）。
#   最近13曲中10曲で36件出て、うち12件は語の切れ目をまたいだ誤報（これが｜いいや）。機械の長さ入れ替え（詞-4）も外したまま


def note_grid(melody) -> int:
    """**その曲が使っている音符の細かさ**に合わせたグリッドを返す（2026-09-08）。

    ★**16分に固定してはいけない。** 8分でできている曲を16分グリッドに乗せると、
    **8分の「裏の裏」**——伴奏が何も鳴らしていない場所——へ歌が落ちる。

    実際に起きた失敗（『琥珀色に灯る』BPM80）: 旋律の音価は **97% が8分の倍数**なのに、
    隙間を「最小16分」で丸めたせいで**68音中36音（53%）が奇数16分位置**へ移った。
    伴奏（23パート）が鳴らすのは 0/2/4/6/8… の偶数位置が中心で、
    **歌の半分以上が、伴奏が無音の瞬間に出ていた**。本人の評価は「リズムに乗り切れてない」。

    判定は**旋律が書いた音価そのもの**から取る（伴奏はこの時点でまだ無い）。
    8割が8分の倍数なら8分、そうでなければ16分。
    """
    vals = [float(n.beats) for p in getattr(melody, "phrases", []) or [] for n in p.notes]
    vals += [float(getattr(p, "rest_before", 0) or 0)
             for p in getattr(melody, "phrases", []) or []]
    vals = [v for v in vals if v > 0]
    if not vals:
        return GRID
    # ★**1つでも 16分の値があれば 16分**（2026-09-09 修正）。
    #   「8割が8分の倍数なら8分」としていたら、**16分の音が2つだけの曲を8分と判定**し、
    #   その2音が「グリッド外」に数えられて率が 0.452 まで落ちた。
    #   グリッドは**その曲がいちばん細かく置いた位置**で決まる——多数決ではない。
    for div, g in ((2, Q // 2), (4, Q // 4)):              # 粗いほうから試す（8分 → 16分）
        if all(abs(v * div - round(v * div)) < 1e-6 for v in vals):
            return g
    return GRID


def lead_gaps(melody, limit_bars: float = 2.0) -> list[str]:
    """区間ごとに、**区間の頭から歌い出しまで**が `limit_bars` 小節を超える所（点検役の材料・2026-09-29）。

    以前は組み上がった楽譜の実測（位置合わせの点検）で見ていた。旋律の1フレーズ目の手前の休み（`rest_before`）で同じことが測れる。
    """
    out: list[str] = []
    seen: set[str] = set()
    bar_beats = bar_ticks(ctx.METER) / Q
    for p in getattr(melody, "phrases", None) or []:
        if p.section in seen:
            continue
        seen.add(p.section)
        lead = float(getattr(p, "rest_before", 0) or 0) / bar_beats
        if lead > limit_bars:
            out.append(f"区間『{p.section}』は歌い出しまで **{lead:.1f}小節**空いている（意図した「間」ならそのままでよい）")
    return out


def breath_facts(melody, tempo: float) -> str:
    """**ひと息の事実**（旋律の点検役の材料・毎回・2026-10-03 組G1・`docs/plan/song_viewpoints_review.md` §1-2）。

    ひと息＝0.25秒以上の休みで切れるまで（`rhythm_metrics` と同じ数え方・区間の頭で区切る）。中央値・最長と最短の行・
    息継ぎの間隔・区間ごとの行と秒を並べる。本人の指摘がいちばん多い所（「リズムに乗れてない」＝1行ごとに息が切れていた）。
    ★ヒット曲の実測の範囲は添えず、出す・出さないも範囲で決めない——「判定ではない」と添えても帯の数字は点検役を寄せ、
      本人が気に入った曲がいちばん帯の外だった。毎回出る事実なので、所見（記録の warnings＝「うまくいかなかった工程」に並ぶ）にしない。
    """
    runs_at: list[tuple[str, int, int, float]] = []
    m = rhythm_metrics(melody, tempo, runs_at=runs_at)
    if not m or not runs_at:
        return ""
    spb = 60.0 / max(1.0, tempo)

    def _where(r) -> str:
        return f"{r[0]} {r[1]}行目" if r[1] == r[2] else f"{r[0]} {r[1]}〜{r[2]}行目"
    lng, sht = max(runs_at, key=lambda r: r[3]), min(runs_at, key=lambda r: r[3])
    per: dict[str, list[str]] = {}
    for r in runs_at:
        per.setdefault(r[0], []).append(f"{_where(r).split(' ', 1)[1]} {r[3] * spb:.1f}")
    freq = m["息継ぎの頻度"]
    return ("【ひと息】（0.25秒以上の休みで切れるまでの長さ。区間の頭で区切って数えた。機械が数えたもの）\n"
            f"  中央値 {m['ひと息の秒']:.1f}秒・最長 {lng[3] * spb:.1f}秒（{_where(lng)}）・最短 {sht[3] * spb:.1f}秒（{_where(sht)}）・"
            + (f"息継ぎは歌っている時間の {1 / freq:.1f}秒に1回" if freq > 0 else "息継ぎ（0.25秒以上の休み）なし") + "\n"
            + "\n".join(f"  {s}: " + "・".join(xs) + "（行・秒）" for s, xs in per.items()))


def inspect_breath(melody, tempo: float, groove=None, *, lines: list[tuple[str, str]] | None = None,
                   record_only: list[str] | None = None) -> list[Finding]:
    """**歌として成立するリズムか**を測って、関門の LLM に渡す所見にする。

    `lines` は旋律の工程が書いた行（区間名, ABC）の並び。区切りの記号（グルーヴの `breaks`・1行1文字）を**書いた行**に当てるのに使う
    （休みだけの行はフレーズにならないので、フレーズの順では行と記号がずれる）。ABC の記録が無い旋律（枠記法の頃）は None＝フレーズが行。

    ★★**点検役に渡すのは2つだけ**（2026-09-29 本人了承・調査は `docs/plan/song_no_silent_corrections.md`
    「息継ぎの所見の妥当性」）★★ 毎曲出ていて、18曲中8曲で旋律を作り直させ（1回で1曲の出力の約4割）、
    作り直しても別の数字が悪くなるだけで直る見込みが薄かった——「指標は検算に使い生成の駆動に使わない」とも食い違っていた。
    - 渡す: **区切る行と前の行との間の短さ**（グルーヴの宣言との照合）。測った事実（何秒・どの行）だけを書き、良し悪しと直し方は書かない。
      ヒット曲の実測の範囲は添えない（2026-10-03・組G1）。**ひと息**は 2026-10-03 から毎回の事実として材料に出す（`breath_facts`）
    - 記録だけ（`record_only` に文で入れる）: 休符率・ひと息の最長・音の密度・型の再出現
    - 外した: 1音の長さ（テンポで決まり直す手段がない）・拍頭率（2026-09-12 に代理指標と結論・好き嫌いと逆）

    ## なぜ「同時に」測るのか

    最初は**休符率と最長連続歌唱の2つだけ**を見ていた。それで
    「休み13.6%・6.9秒連続」の曲は直ったが、**次は逆側へ振れた**——
    **0.9秒の断片が22本**並び「リズムに乗っていない・詩を読み上げているよう」と言われた。
    片側の指標だけを見ると、必ずもう片方を壊す。

    ## 判定はしない

    切迫感を狙って詰めることはあるし、語り口の曲なら休みが少なくて正しい。
    ここは所見に留める（外れ具合でやり直させていた `rhythm_offenses` は、呼ばれなくなっていたので 2026-09-28 に消した）。
    """
    m = rhythm_metrics(melody, tempo)
    if not m:
        return []
    out: list[Finding] = []
    notes: list[str] = []                                  # 記録だけ（点検役には渡さない）

    _b = rhythm_band(tempo)
    spb = 60.0 / max(1.0, tempo)                           # 1拍の秒数（以下で使う）

    def band(k: str) -> tuple[float, float]:
        return _b[k]

    lo, hi = band("休符率")
    if m["休符率"] < lo:
        notes.append(f"休みが {m['休符率'] * 100:.0f}%（ヒット曲の実測 {lo * 100:.0f}〜{hi * 100:.0f}%）")

    # ★**「まくし立て」を宣言している曲では、ひと息の長さを咎めない**（2026-09-16）。
    #   帯（2.5〜6.5秒）は**人が歌うポップス**（POP909・RWC-P）の実測で、
    #   **ボカロは息継ぎが要らない**。参考曲（BPM204・作者配布の公式 MIDI）の実測は
    #   **ひと息 中央 8.97秒・最長 19.1秒・1息 最多 119音**で、帯を大きく外れたまま成立している。
    #   宣言していない曲では従来どおり所見を出す（黙って緩めない）。
    _rush = any(g.tail_rest is not None and g.tail_rest < 0.10   # null＝歌の無い区間（2026-10-04 組N2。以前は既定 0.30）
                for g in (getattr(groove, "sections", None) or []))
    lo, hi = band("ひと息の秒")
    if m["ひと息の最長秒"] > hi and not _rush:
        notes.append(f"{m['ひと息の最長秒']:.1f}秒続けて歌うフレーズがある（人の実測 {lo:.1f}〜{hi:.1f}秒）")
    # ※ひと息は 2026-10-03（組G1）から所見でなく、毎回の事実として点検役の材料に出す（`breath_facts`）。以前はここで、中央値と
    #   息継ぎの間隔がヒット曲の実測の範囲の外のときだけ、範囲の数字を添えて所見にしていた（その朝までは「ぶつ切りに聴こえる」と断じていた）

    # ★**区切りが区切りとして効いているか**（2026-09-07 追加）。
    #   `rest_before` が 0.25秒に満たないと**隣のフレーズと繋がる**——
    #   旋律は「12フレーズ」と書いたのに、出荷された楽譜は **6本・最長26.5秒の塊**だった。
    #   本人に渡した曲が実際にそうなっていた。**書いた形と鳴る形は違う。**
    _phr = getattr(melody, "phrases", None) or []
    # ★★**C（そのまま続く）と宣言した行は、繋がっているのが正しい**（2026-09-13 修正）★★
    #   `rest_before < 0.25秒` を一律に「区切りが効いていない」と数えていたため、
    #   C で繋いだ行（例: CMCS なら半分）が全部不良扱いになり、『この衝動』では
    #   「16/32本で効いていない＝破綻」と関門がやり直しを命じた。**繋ぐ設計と検査が衝突していた。**
    #   グルーヴの宣言を読み、**M・S なのに繋がっている行だけ**を数える。
    _before: dict[int, str] = {}                           # フレーズの番号 → その行の直前の行の区切りの記号
    if groove is not None:
        # ★**区切りの記号は記法の読み手（`song_design.BREAKS.read`）で1行1文字に**（2026-10-02）。以前は生の文字列を位置で引き
        #   （空白も1文字に数えた）、足りない行は黙って M とみなしていた。数（行数と同じ）はグルーヴの工程の検査が、
        #   旋律の行の数（グルーヴの行数と同じ）は旋律の工程の検査が保証しているので、足りないことは無い。来たら不具合として止める
        # ★★**記号は書いた行に当てる**（2026-10-02・仕様書 §7「1行1文字・行の順」）★★ 以前は「1行＝1フレーズ」として
        #   フレーズの順に当てていたが、休みだけの行（`z16`）はフレーズにならない（`abc_to_phrases`）ので、その区間の後ろの行の
        #   記号が1つずつずれていた。行はフレーズの素なので、音のある行とフレーズは順に1対1（1行に音があれば必ず1フレーズ）
        from abc_notation import notes_of                   # noqa: E402（旋律の行の読み手は1つ＝組み立てと同じ）
        from song_design import BREAKS                      # noqa: E402（区切りの記号の読み手は1つ）
        from song_fix import AssemblyError                  # noqa: E402
        _gv = {g.kind: BREAKS.read(g.breaks) for g in getattr(groove, "sections", [])}
        rows = ([(p.section, True) for p in _phr] if lines is None
                else [(sec, any(nm for nm, *_ in notes_of(abc))) for sec, abc in lines])
        _seen: dict[str, int] = {}
        row_mark: list[str] = []
        for sec, _sung in rows:
            _k = _seen.get(sec, 0)
            _seen[sec] = _k + 1
            _bk = _gv.get(sec, [])                         # ※外側の `_b`（帯）を上書きしない
            if _k >= len(_bk):
                raise AssemblyError(f"区切りの記号が足りない: 区間『{sec}』の{_k + 1}行目に当たる記号が無い"
                                    f"（グルーヴの区切り『{''.join(_bk)}』・検査を通ったはず）")
            row_mark.append(_bk[_k])
        sung_rows = [r for r, (_s, sung) in enumerate(rows) if sung]
        if len(sung_rows) != len(_phr) or any(rows[r][0] != p.section for r, p in zip(sung_rows, _phr)):
            raise AssemblyError(f"旋律の行とフレーズが対応しない: 音のある行 {len(sung_rows)}行・フレーズ {len(_phr)}本"
                                "（フレーズは音のある行から1つずつ起こすので、数と区間名が順に一致するはず）")
        # 行の頭の前の切れ目＝**直前の行の記号**（記号は「その行から次の行へどうつなぐか」・§7）。
        # ※休みだけの行は 2026-10-02 から旋律の工程の形式の所見（`song_melody_step.MelodyStep.check`・本人「おすすめで」）で、直らなければ
        #   旋律の工程で止まるので、ここへは来ない（どの行にも音があり、行とフレーズは1対1）。保存済みの記録にも0行。上の行の数え方は
        #   来た場合にも行と記号をずらさない形のまま残してある（休みだけの行の記号の意味は決めていない＝書けないので要らない）
        _before = {j: row_mark[r - 1] for j, r in enumerate(sung_rows) if r > 0}
    # 行の名前（区間の何行目か）。音のある行とフレーズは順に1対1（上の対応の検査）なので、区間ごとにフレーズを数えればよい
    _nth: dict[str, int] = {}
    _label: list[str] = []
    for p in _phr:
        _nth[p.section] = _nth.get(p.section, 0) + 1
        _label.append(f"{p.section}の{_nth[p.section]}行目")
    # ★★**前の行との間＝前の行の末尾の空き＋この行の頭の休み**で測り、0 も数える（2026-10-03・組G1）★★
    #   以前は「0 < 頭の休み < 0.25秒」だけを見ていた——①頭の休みが 0（前の行に末尾の休みも無い＝完全につながった行）を数えず、
    #   いちばん区切れていない行を見落とした ②区間の頭の行は、前の区間の末尾の休みが前の音の `gap_after` に入る
    #   （`song_melody.abc_sketch_to_melody`）ので頭の休みだけを見て、実際は空いていても短く出た。
    #   区間の中では行末の休みが次の行の `rest_before` に入っていて、前の行の最後の音の `gap_after` は行の中の空きだけ。どちらも足せば
    #   「前の行の最後の音が終わってから、この行の最初の音まで」になる。曲の最初の行は前の行が無いので数えない。
    #   間奏など歌の無い区間をはさむ区間の頭（グルーヴの区間の並びで隣どうしでない）は、その区間の長さが入らないので数えない
    _order = [str(getattr(g, "kind", "")) for g in (getattr(groove, "sections", None) or [])]

    def _between(i: int) -> float | None:
        if i == 0:
            return None
        prev, cur = _phr[i - 1], _phr[i]
        if prev.section != cur.section and prev.section in _order and cur.section in _order \
                and _order.index(cur.section) != _order.index(prev.section) + 1:
            return None                                    # 歌の無い区間をはさむ
        tail = float(getattr(prev.notes[-1], "gap_after", 0) or 0) if prev.notes else 0.0
        return tail + float(getattr(cur, "rest_before", 0) or 0)

    weak = [(_i, g) for _i in range(len(_phr))
            if (g := _between(_i)) is not None and g * spb < 0.25
            and _before.get(_i, "M") != "C"]               # 直前の行が C なら、この行の頭は繋がってよい
    # ★**事実だけを書く**（2026-10-03・本人「機械が検査して事実を出し、直し方は点検役に委ねる」）。以前は「区切りが効いていない
    #   ——隣と繋がって1つの塊になる」と断じ、休みの置き方まで指示していた。どの行が前の行と何秒空いているか・宣言は何かだけを並べる
    if weak:
        _rows = "・".join(f"{_label[i]}（{g * spb:.2f}秒）" for i, g in weak[:12])
        out.append(Finding(
            what=f"区切りが C（そのまま続く）でない行のうち、前の行との間（前の行の最後の音が終わってから、この行の最初の音まで）が "
                 f"**0.25秒未満**の行: {len(weak)}/{len(_phr)}本（{_rows}{' ほか' if len(weak) > 12 else ''}）",
            evidence=f"0.25秒（BPM{tempo:.0f} で {0.25 / spb:.2f}拍）は、ひと息を区切ると数えた休みの長さ（上のひと息と同じ）。"
                     "記号は、その行の前の行に書いた区切り（S/M）"))

    # ※「1音の長さ」と「拍頭率」の所見は 2026-09-29 に外した（上の説明）。数字は `rhythm_metrics` が出し続ける
    lo, hi = band("音の密度")
    if m["音の密度"] > hi:
        notes.append(f"1秒あたり {m['音の密度']:.1f}音（ヒット曲の実測 {lo:.1f}〜{hi:.1f}音/秒）")

    lo, hi = band("型の再出現")
    if m["型の再出現"] < lo:
        notes.append(f"4音のリズム型の再出現 {m['型の再出現'] * 100:.0f}%（ヒット曲の実測 {lo * 100:.0f}〜{hi * 100:.0f}%）")
    if record_only is not None:
        record_only.extend(notes)
    return out


def _hold_on_chord(melody, chords) -> str:
    """**伸ばす音が和音に乗っているか**を、旋律の段階で数字にする（2026-09-13 新設）。

    `song_pipeline` の同じ検査は**楽譜になってから**しか走らない。スケッチ（階名・拍）と
    コード（C 基準・小節ごと）から同じ比を出す。旋律もコードも C 基準なので移調は要らない。
    ※以前は旋律の選抜の札に載せていた。1案にした（2026-09-30）ので、旋律のリズムの数字と一緒に曲の記録に残す
    （`melody_numbers`・点検役には渡さない——J-POP の実測と比べる数字で、良し悪しを断じる帯と同じ扱い）。

    J-POP の実測は「長い音のほうがコード構成音である割合が **+8.7ポイント高い**」。
    本人の曲では 2曲続けて逆（57% 対 70%／69% 対 89%）だった。
    """
    from song_audio import chord_notes                     # 遅延 import（循環なし・重いだけ）

    bars = {s.kind: list(getattr(s, "chords_per_bar", []) or [])
            for s in getattr(chords, "sections", []) or []}
    cur: dict[str, float] = {}
    pairs: list[tuple[float, float]] = []
    for p in melody.phrases:
        names = bars.get(p.section)
        if not names:
            continue
        at = cur.get(p.section, 0.0) + float(p.rest_before)
        for n in p.notes:
            b = int(at // 4)
            if b < len(names):
                entry = names[b]
                _half = split_bar_chords(entry)            # 1小節に2つ → 前半／後半
                entry = _half[0 if (at % 4) < 2 or len(_half) < 2 else 1] if _half else entry
                tones = {k % 12 for k in chord_notes(entry.strip())}
                if tones:
                    pairs.append((float(n.beats), 1.0 if solfa_to_midi(n.solfa, n.octave) % 12 in tones else 0.0))
            at += float(n.beats) + float(n.gap_after)
        cur[p.section] = at
    if len(pairs) < 16:
        return ""
    pairs.sort(key=lambda x: x[0])
    q = len(pairs) // 4
    short = sum(r for _, r in pairs[:q]) / q
    long_ = sum(r for _, r in pairs[-q:]) / q
    return (f"  伸ばす音が和音に乗っている割合: 長い音 {long_ * 100:.0f}% / 短い音 {short * 100:.0f}%"
            f"（差 {(long_ - short) * 100:+.0f}pt。J-POP は +8.7pt。"
            f"**長い音のほうが低いと、伸ばすほど濁る**）")


def melody_numbers(melody, tempo: float, chords=None) -> list[str]:
    """旋律の**リズムと和音の数字**（曲の記録だけ・点検役には渡さない・2026-09-30）。

    以前は旋律3案の選抜の札（`_melody_card`）の後半で、選抜役はこれを主な材料に選んでいた。
    1案にしたので、息継ぎの記録（`melody_rhythm`）と同じ扱いで記録に残す。ヒット曲の実測と比べて読む数字で、
    「指標は検算に使い生成の駆動に使わない」（`generation_design.md`）ので、点検役の判断の材料にはしない。
    """
    out: list[str] = []
    m = rhythm_metrics(melody, tempo) if tempo > 0 else {}
    if m:
        out.append("リズム: " + " / ".join([
            f"拍頭率 {m['拍頭率']:.2f}",
            f"型の再出現 {m['型の再出現']:.2f}",
            f"ひと息 {m['ひと息の秒']:.1f}秒",
            f"休符率 {m['休符率']:.2f}",
            f"1音 {m['1音の秒'] * 1000:.0f}ミリ秒",
            f"音の密度 {m['音の密度']:.2f}音/秒",
            f"伸ばし {m['伸ばしの秒']:.2f}秒（ばらつき {m['伸ばしのばらつき']:.2f}）",
        ]))
    if chords is not None and (_hc := _hold_on_chord(melody, chords)):
        out.append(_hc.strip())                            # ★伸ばす音が和音に乗っているか（#13）
    return out
