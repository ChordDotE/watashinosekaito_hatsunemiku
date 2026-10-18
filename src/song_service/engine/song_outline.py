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

"""**曲の骨組みの工程**——テーマ・設計・コード・グルーヴ。

2026-09-30 に `song_compose._compose_once_body` から切り出した。LLM の型と指示書・工程の検査（`ChordsStep`・`GrooveStep`）は
`song_design` にあり、ここは**依頼文を組んで呼び、検査を通して曲に載せる**だけ。
"""
from __future__ import annotations

import song_context as ctx                                 # LLM の呼び出し口（`ctx._call`）・申し送り・拍子
from song_design import ChordsStep, DesignStep, GrooveStep, _chords, _design, _groove, _theme, beats_per_bar, grid_slots, meter_text, units_per_bar
from song_flow import _rec
from song_rhythm import _mora_budget


# ※設計の所見（`inspect_design`・区間名の重複）は 2026-09-30 に設計の型（`song_design.Design._unique_kinds`）へ移した。
#   「小節数が 0 以下」の所見は 2026-09-28 に外した（設計の型が 1〜32 小節しか受け付けない）
# ※グルーヴの検査と直し（1行が2小節を超える・`MAX_BARS_PER_LINE`）は、グルーヴの工程（`song_design.GrooveStep`）へ移した（2026-09-28）


def run_theme(st) -> None:
    """**テーマ**（構想）。書く: `song.theme`。"""
    song, verbose = st.song, st.verbose
    theme = ctx._call(_theme, st.instruction, "テーマ")
    st.log["theme"] = theme.reason
    song.theme = theme
    _rec(song, "theme", theme, verbose=verbose)


def run_design(st) -> None:
    """**設計**（構成・BPM・キー・区間の強さ）。読む: `song.theme`。書く: `song.design`。"""
    song, verbose = st.song, st.verbose
    theme = song.theme
    # ★**設計にも申し送りを渡す**（2026-09-09）。それまで設計が受け取っていたのは
    #   題材・情景・感情・曲名だけで、**様式も、心の動きも、声の高さも渡っていなかった**。
    #   構想は「様式がイントロの長さも構成も決める」と書き、`top_note` は
    #   「構想で決めた高さに合わせること」と書いているのに、**その両方が届いていなかった**。
    #   渡さなかったものは存在しないのと同じ（brief の項に同じ）。
    design = ctx._call(_design, (
        ctx.brief(theme) + f"\n\n仮の曲名『{theme.working_title}』。依頼: {st.instruction}"), "設計")
    st.log["design"] = design.reason
    # ★**区間名の重複は設計の型が差し戻す**（2026-09-30 本人承認・`song_design.Design._unique_kinds`）。区間名が重なると
    #   1番と2番の区別が付かず時間軸が組めない（本番の初回で `verse / … / verse / … / chorus ×3` が出て曲づくりが落ちた）。
    #   以前（2026-09-01〜）は機械が所見を出して設計の関門（点検役）がやり直させていた。答えが1つに決まる形式なので型へ移し、
    #   設計の関門（ほかに所見が無かった）ごと外した。どちらを1番・どれをラスサビにするかは、型の差し戻しを読んだ設計の LLM が決める
    # ★**曲の形で決まる設計の決まり**（2026-10-03・`docs/plan/song_acappella_instrumental.md` §3）。いまは終わり方だけ——アカペラ・
    #   インストで `ending: vocal`（伴奏が先に止む）は意味が無いので所見を返して差分で直させる。歌ものでは所見が出ない（以前と同じ）
    design, _ = DesignStep(song.form).run(design, song, verbose=verbose)
    song.design = design
    _rec(song, "design", design, verbose=verbose)
    st.log["concept"] = design.concept


def run_chords(st) -> None:
    """**コード**。読む: `song.theme`・`song.design`。書く: `song.chords`・`st.ch_map`（区間 → 小節ごとのコード）。"""
    song, verbose = st.song, st.verbose
    theme, design = song.theme, song.design
    chords = ctx._call(_chords, (
        ctx.brief(theme, design) +
        f"\n\n仮の曲名『{theme.working_title}』BPM{design.bpm} キー{design.key}\n構成: " +
        "\n".join(f"- {s.kind}: {s.bars}小節 音域={s.pitch_center}" for s in design.sections) +
        "\n\n**全体方針に沿った**進行を組んでください。"), "コード")
    chords, _ = ChordsStep(design.sections).run(chords, song, verbose=verbose)   # 区間名・小節数（設計と照合）・読めないコード名を返して直させる
    st.log["chords"] = chords.reason
    song.chords = chords
    _rec(song, "chords", chords, verbose=verbose)
    st.ch_map = {c.kind: c.chords_per_bar for c in chords.sections}
    if verbose:
        print(f"  [1-3] 『{theme.working_title}』{theme.emotion} / **{getattr(theme, 'era_style', '?')}** / "
              f"BPM{design.bpm} {design.key} / "
              + " → ".join(f"{s.kind}({s.bars})" for s in design.sections))
        print(f"      方針: {design.concept}")


def run_groove(st) -> None:
    """**グルーヴ**（ドラム・ベースの土台と、区間ごとの行数×モーラ数）。いまの曲の拍子（`ctx.METER`）もここで決まる。

    読む: `song.theme`・`song.design`・`song.chords`・`st.ch_map`。書く: `song.groove`・`ctx.METER`。
    """
    song, verbose = st.song, st.verbose
    theme, design, chords, ch_map = song.theme, song.design, song.chords, st.ch_map
    # ★**グルーヴを先に作る**（2026-09-07 新設）。プロは「ドラムパターンとコードだけでも良い」
    #   オケの上で鼻歌を歌ってメロディを作る（ドラム→ベース→仮ピアノ→仮メロディ）。
    #   それまで旋律工程は**コード名しか受け取っておらず、鳴っているリズムを知らなかった**——
    #   拍頭率 0.55（ヒット曲 0.12〜0.33）＝本人の「リズムに乗っていない・朗読のよう」の原因。
    #   実験でグルーヴを渡すと **型の再出現 0.57→0.88 / 拍頭率 0.55→0.36** と動いた。
    #   ここでは**対句の型（何行×何モーラ）**も決める——これが旋律と詞の骨格になる。
    # ★**拍子から1小節の拍数を引く**（2026-09-27）。`bars * 4` と 4/4 の決め打ちで、3/4 の曲に秒数・拍数・モーラが
    #   3分の1多く渡り、グルーヴが「4/4 で組んだ（妥協）」と書いた（『風に呼ばれたら』）
    ctx.METER = design.time_signature                    # 選べる4つ（型 `TimeSignature`・2026-09-30）
    sec_sec = {s.kind: s.bars * beats_per_bar(ctx.METER) * 60.0 / design.bpm for s in design.sections}
    groove = ctx._call(_groove, (
        ctx.brief(theme, design, chords)
        + f"\n\n仮の曲名『{theme.working_title}』BPM{design.bpm} 種類={theme.genre} **拍子={ctx.METER}**\n"
        # ★**拍子の伝え方は旋律と同じ1か所**（`song_design.meter_text`・2026-09-30・出-5）。以前はここに別の文があり、
        #   6/8 を「1小節＝3拍」と伝えていた（グルーヴが周回ごとに 3/4 と 6/8 を行き来した）
        + meter_text(ctx.METER) + "\n"
        + ("" if ctx.METER == "4/4" else
           f"※ドラムとベースは**8分音符 {grid_slots(ctx.METER)} 個**（16分なら {units_per_bar(ctx.METER)} 個）で1小節を書いてください"
           "（4/4 の8個ではありません）。\n")
        # ★**モーラ数もこちらで計算して渡す**（2026-09-09）。
        #   ここに「1モーラ0.29秒＋息継ぎぶん」と書いていたが、`song_groove.md` は
        #   **「秒数 ÷ 0.29秒 で計算しないでください（約2倍多く見積もる）」**と書いている
        #   ——**同じ依頼の中で真逆のことを言っていた**。0.29秒は「1音が鳴る長さ」で、
        #   **次の音までの間隔ではない**（休みが時間の 25〜42% を占める）。
        #   実際に BPM200 の曲で 4行×8モーラ＝密度 2.4音/秒（実測の中央 1.7）になった。
        + "\n".join(f"- {s.kind}: {s.bars}小節＝**{sec_sec[s.kind]:.1f}秒**"
                    f" 強さ{getattr(s, 'intensity', 3)}"
                    # ★**運びで入る量が変わるので、両方の数字を出す**（2026-09-16）。
                    #   `塊` は休みが 25〜42% ある前提の密度、`まくし立て` は休まない前提。
                    #   **どちらを選ぶかはグルーヴが決める**ので、こちらは座標だけ渡す。
                    + (f"　★**入るモーラ ＝ 拍数 × `notes_per_beat`**"
                       f"（{sec_sec[s.kind] * design.bpm / 60:.0f}拍・`tail_rest` は掛かりません）→ "
                       f"0.8 なら **{_mora_budget(sec_sec[s.kind], design.bpm, theme.genre, 0.8)}**・"
                       f"1.0 なら **{_mora_budget(sec_sec[s.kind], design.bpm, theme.genre, 1.0)}**・"
                       f"1.3 なら **{_mora_budget(sec_sec[s.kind], design.bpm, theme.genre, 1.3)}**"
                       if s.has_vocal else "")
                    # 小節の区切りは ` / `（コードの工程の `ChordsStep.show` と同じ）。以前は `-` でつなぎ、コード名の中の `-`
                    #   （`C7-9`）や1小節に2つ（`Am F`）と見分けにくかった（2026-10-01・仕様書 §3 の突き合わせ）
                    + f" コード: {' / '.join(ch_map.get(s.kind, []))}"
                    for s in design.sections)
        + "\n\n**歌のある区間は、行数×モーラ数を上の『入るモーラ』に合わせてください**"
          "（割り算はこちらで済ませてあります）。"
          "\n★**自分で決めた `notes_per_beat` と `tail_rest` から出る数に合わせること。**"
          "`tail_rest` を 0（休まない）にしておきながらモーラを少なくすると、"
          "**行末が休みだらけの、ただのぶつ切り**になります"
          "（2026-09-16 に実際に起きた——ひと息 1.65秒・1息5音）。"
          "\n★**1行のモーラ数が 12 を超えるときは、`moras_per_line` に 16 まで書けます**"
          "（`tail_rest=0` で1行2小節なら、8分で16モーラ）。"), "グルーヴ")
    # ★**1行が長すぎたら、グルーヴに差分で直させる**（2026-09-27・目録の時間-7）。直らなかった分だけ機械が行を増やし、記録する
    # 行数・区切りの記号・ドラムの行（伴奏と同じ打点の記法と打楽器名・2026-10-01 判-E）を検査して差分で直させる。直らなければ止まる
    groove, _ = GrooveStep(design.sections, design.time_signature, song.form).run(groove, song, verbose=verbose)
    st.log["groove"] = groove.reason
    song.groove = groove
    _rec(song, "groove", groove, verbose=verbose)
    if verbose:
        print(f"  [1-4 グルーヴ] {groove.feel}")
        for g in groove.sections:
            print(f"      {g.kind}: {g.lines}行 × {g.moras_per_line}モーラ / "
                  + " ".join(f"{d.inst}={d.pattern}" for d in g.drums[:3]))
