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

"""**表示用の歌詞**（漢字・カタカナ・英字あり）と、**歌う歌詞**（かな）を突き合わせる。

**なぜ2つ持つのか**（2026-09-21 本人指示）:
詞工程は最初から「ひらがなのみ」で書いていて、**漢字版はどこにも存在しなかった**——
保存されていないのではなく、一度も作られていない。そのため歌詞を人に見せようとすると
`ぬるいかぜのなかうみぞいおあるく…` という並びになり、読み物として成立しない。
字幕・投稿・一覧で使いたいので、**歌う用と見せる用を両方作って保存する**ことにした。

**優先は歌のほう**。`text`（かな）を先に決め、`display` はその読みとして書く。
どうしても使いたい語が音数に合わないときだけ、`text` 側を伸ばす／割って合わせる。
**`display` を削って詰めるのは禁止**——言いたかった言葉が消えるため。

**カタカナは歌える**（2026-09-21 実測）。`ヌ` も `ぬ` も音素は `n M` で同じ。
歌えないのは**漢字と英字**（ローマ字化されず、音素が付かない）。だから `text` は
**ひらがな または カタカナ**、`display` は自由、という切り方になっている。

---

★★**ここは読みを推定しない**（2026-09-22・本人「**そもそもパイカカシ精度低いのでいらないのでは**」）★★

いったん `pykakasi` で `display` の読みを起こして `text` と突き合わせていたが、やめた。
辞書ベースで文脈を見ないので、**同じ漢字で読みが変わる語**を外す——実測で
`南風`→「はえ」（歌なら「みなみかぜ」）・`風車`→「ふうしゃ」・`Tシャツ`→「Tしゃつ」。
歌詞によく出る語 27件では 26件が合ったが、**外すのは詩的な語**＝歌詞に出やすい側だった。

**この検査は曲を止めない**ので、外れても曲は壊れない。壊すのは**偽の警告**のほうで、
本物の所見を埋もれさせる（同じ型を `to_sung_kana` の連濁で既に踏んでいる）。

そこで**外れない検査だけを機械に残し、読みの判断は関門の LLM へ渡した**:

| 何を見るか | 外れるか |
|---|---|
| `display` のかな（送りがな・助詞）が `text` に**この順で出てくる**か | **外れない** |
| `display` の**最小モーラ数**が `text` のモーラ数を超えていないか | **外れない**（どの字も1モーラ以上）|
| 語がフレーズをまたいでいないか | **機械では見ない**。`［表示］` を関門へ渡して読ませる |
| 語の途中に伸ばし・小節線が入っていないか（`word_spans`・2026-10-04） | **外れうる**——字の種類の境目と助詞から語の切れ目を**推す**だけ。読みは推さず、表示に書かれたかなだけを手がかりに歌う詞へ当てる。だから**候補**として点検役へ出し、断じない |

> **★引ける原則**: 道具の精度が足りないと分かったら、精度を上げにいく前に
> **その道具に何を任せるかを狭める**。機械は外れない部分だけを断定し、
> 外れる部分は**証拠として差し出して判定を渡す**。
"""
from __future__ import annotations

from lyric_slots import sung_kana                      # 歌う詞の譜割りの印（空白・`_` `*` `~`）を外す

#: 詞に出てよいかな
_KANA_OK = set("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめも"
               "やゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ"
               "ゃゅょっぁぃぅぇぉ")

#: 前の字と繋がって1モーラになる小書き（「っ」は入れない＝促音は1モーラを占める）。
#: ★「ゎ」は入れない（2026-10-02・組み立ての `lyric_slots._SMALL_KANA` と同じ1つを使う・仕様書 §2）
from lyric_slots import _SMALL_KANA as _SMALL          # noqa: E402

#: **見せる歌詞**で前の字とつながる小書き（2026-10-04）。歌う詞はひらがなだけなので `_SMALL` で足りるが、見せる歌詞は
#: カタカナ語を書く。以前は `_SMALL`（ひらがなだけ）で数えていたので、「濡れたアスファルト」の「ァ」を1モーラに数えて
#: 9モーラ以上とし、8モーラの歌う詞に「表示のほうが長い」と誤って出した（『水たまりの青空』）。
#: 「ゎ」「ヮ」も見せる歌詞では前の字とつながる（歌う詞では書けない字＝`song_lyrics.check_kana` の所見なので、`_SMALL` には入れない）。
#: 「ッ」「ン」は1モーラを占めるので入れない
_SMALL_DISP = _SMALL + "ゎ" + "ァィゥェォャュョヮ"

#: 表示用に書かれがちな、音にならない字（数えない）
#: ※「ー」も数えない——歌う詞は長音を母音（`らあ`）でも `_`（前の音を伸ばす・モーラに数えない）でも書けるので、
#:   数えると `ら _ め ん`（3モーラ）に「ラーメン」で偽の所見が出る。下限は外れない側に倒す
_PUNCT = " 　\t\n\r、。,.!?！？「」『』（）()…‥・―—-ー~〜/"


def moras(kana: str) -> list[str]:
    """かな列 → モーラ列（拗音は2文字で1モーラ）。**1モーラ＝1音符**。"""
    out: list[str] = []
    for c in kana:
        if c in _SMALL and out:
            out[-1] += c
        else:
            out.append(c)
    return out


#: 比べるときだけ同じものとして扱うかな。**歌えば同じ音**になる組み合わせ。
#: ★実機で**偽の警告が4件**出て足した（2026-09-22・24フレーズ中）。ユニットテストでは出なかった:
#:   ・`僕らは歩く` ↔ `ぼくらわあるく` …… 助詞の **は→わ**（詞は必ず発音表記で書く）
#:   ・`明日へ進もう` ↔ `あしたえすすもお` …… **へ→え** と、**オ段の長音 う→お**
#: **両側に同じ変換を掛ける**ので、失うのは「う と お を見分ける力」だけ
#: （歌えば同じ音なので、見分けられなくて困らない）。
_SAME = {"は": "わ", "へ": "え", "を": "お", "う": "お"}


def _kana_in(s: str) -> str:
    """**かなだけ**を抜き、**歌えば同じ音になるものを同じ字へ寄せる**。

    ★**両側に同じものを掛ける**（2026-09-22）。最初は表示用にだけ掛けていたため、
      歌う歌詞に「へ」が残っている曲で**偽の警告**が出た。
      **片側だけ正規化すると、正規化そのものが食い違いを作る。**

    ★「は」も寄せる。当初は「語の一部のこともある」ので触らなかったが、
      **両側に掛けるなら取り違えようがない**——`花`（はな）は両側とも `わな` になるだけ。
      片側だけ寄せると壊れるが、両側なら**見分ける力が少し減る**だけで済む。
    """
    return "".join(_SAME.get(c, c) for c in (s or "") if c in _KANA_OK)


def min_moras(display: str) -> int:
    """`display` が**最低でも何モーラになるか**。

    漢字も英字も**1字につき1モーラ以上**なので、字を数えれば下限が出る
    （小書きは前の字と繋がるので数えない。カタカナの小書き「ァ」なども——`_SMALL_DISP`）。**辞書が要らず、外れない。**
    上限は出せない——1字が何モーラにもなる（「志」＝こころざし）。
    """
    n = 0
    for c in (display or ""):
        if c in _PUNCT or c in _SMALL_DISP:
            continue
        n += 1
    return n


def check_phrase(text: str, display: str) -> str:
    """1フレーズぶんを突き合わせ、**食い違っていたら申し送りを1行**返す（合っていれば空文字）。

    **止めない**——「合わないから伸ばした」が正当な場合があるので、所見として残すだけにする
    （判定は関門の LLM と人に委ねる。機械は証拠を並べる側）。
    **読みは推定しない**（このファイル冒頭の理由）。
    """
    if not (text or "").strip():
        return ""
    if not (display or "").strip():
        return f"「{text}」に表示用の歌詞が無い"
    kana = _kana_in(display)
    if kana and not _is_subsequence(kana, _kana_in(text)):
        # 事実だけを書く（2026-10-03・組G2）。以前は「送りがな・助詞が食い違っている」と原因まで断じていた——
        # 表示のかなは送りがな・助詞のことが多いが、どちらが違うかは機械には分からない
        return (f"表示「{display}」のかな「{kana}」が、歌う歌詞「{text}」に"
                f"この順で出てこない")
    lo, nt = min_moras(display), len(moras(sung_kana(text)))   # 譜割りの印・空白は数えない（2026-09-29）
    if lo > nt:
        return (f"表示「{display}」はどう読んでも {lo}モーラ以上あるが、"
                f"歌う歌詞「{text}」は {nt}モーラ（表示のほうが長い）")
    return ""


def _is_subsequence(small: str, big: str) -> bool:
    it = iter(big)
    return all(c in it for c in small)


def check_all(by_index: dict[int, str], display_by_index: dict[int, str]) -> list[str]:
    """全フレーズぶん。フレーズ番号を付けて返す。"""
    out: list[str] = []
    for i in sorted(by_index):
        if msg := check_phrase(by_index[i], display_by_index.get(i, "")):
            out.append(f"P{i}: {msg}")
    return out


# ---- 語の切れ目の推定（2026-10-04・詞の点検役の材料「語の途中の伸ばし・小節線」） -------------------------------
# ★**読みは推さない**（このファイル冒頭の理由）。見せる歌詞を**字の種類の塊**（漢字・ひらがな・カタカナ）に分け、
#   ひらがなとカタカナの塊を歌う詞の中へそのまま当て、漢字の塊はそのあいだの残りとして当てる（`re.fullmatch`）。
#   語の切れ目は次の目安で推す（外れうるので、使う側は**候補**として出す）:
#   - カタカナの塊は1つの語。漢字の塊は新しい語の頭（漢字が続く所＝「指先緩」の中の切れ目は分からない＝1つの語として扱う）
#   - 漢字のあとのひらがなは送りがな（同じ語）。ただし末尾の助詞（の が を へ に は と も や）と、かなで書く語（なる いる ある…）は別の語
#   - ひらがなだけの塊が助詞なら別の語
#   - 「射し込む」「踏み出す」のように、漢字＋1字の送りがな（イ段・エ段＝連用形）＋漢字＋送りがな は1つの語（複合動詞）
#   - 漢字もカタカナも含まない語（行頭のひらがなだけの所）は、切れ目を推せないので `anchored=False` で返す

_PARTICLES = ("から", "まで", "より", "の", "が", "を", "へ", "に", "は", "と", "も", "や", "で", "ね", "よ")
_TAIL_PARTICLE = "のがをへにはとも"
_KANA_WORDS = ("しまう", "なる", "いる", "ある", "する", "くる", "いく", "ゆく", "みる", "おく", "ない")
_RENYO_END = set("いきしちにひみりぎじびぴえけせてねへめれげぜでべぺ")

#: 濁点・半濁点を落とす対応表。表記ゆれを吸収して比べるためだけに使う（`song_lyrics._no_dakuten` もこれを読む）
_DAKUTEN = str.maketrans("がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ",
                         "かきくけこさしすせそたちつてとはひふへほはひふへほ")


def _ctype(c: str) -> str:
    if "ぁ" <= c <= "ゖ":
        return "H"
    if "ァ" <= c <= "ヺ" or c == "ー":
        return "T"
    if "一" <= c <= "鿿" or c in "々〆ヶ":
        return "K"
    return "O" if c.isalnum() else ""                       # 英字・数字は漢字と同じく読みの分からない塊。記号・空白は捨てる


def _runs(display: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for c in display or "":
        t = _ctype(c)
        if not t:
            if out and out[-1][0] != "|":
                out.append(("|", ""))                        # 記号・空白は塊の切れ目（語の切れ目にもする）
            continue
        t = "K" if t == "O" else t
        if out and out[-1][0] == t:
            out[-1] = (t, out[-1][1] + c)
        else:
            out.append((t, c))
    return out


def _norm(s: str) -> str:
    """比べるための形: カタカナ→ひらがな・歌えば同じ音（`_SAME`）・濁点を落とす。1字→1字（位置がずれない）。"""
    s = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)
    return "".join(_SAME.get(c, c) for c in s).translate(_DAKUTEN)


def _words(display: str) -> list[list[tuple[str, str]]]:
    """見せる歌詞 → **推した語**の並び（語＝字の種類の塊の並び）。"""
    runs = _runs(display)
    words: list[list[tuple[str, str]]] = []
    new = True                                              # 次の塊で新しい語を始めるか
    for k, (t, s) in enumerate(runs):
        if t == "|":
            new = True
            continue
        prev = runs[k - 1] if k else ("|", "")
        if t == "H":
            if s in _PARTICLES:
                words.append([(t, s)])
                new = True
                continue
            tail: list[str] = []
            if prev[0] in "KT":                             # 送りがな＋助詞・かなで書く語を切り離す
                for w in _KANA_WORDS:
                    if s.endswith(w) and len(s) > len(w):
                        s, tail = s[:-len(w)], [w]
                        break
                if not tail and len(s) > 1 and s[-1] in _TAIL_PARTICLE:
                    s, tail = s[:-1], [s[-1]]
            if new or not words:
                words.append([(t, s)])
            else:
                words[-1].append((t, s))
            for w in tail:
                words.append([("H", w)])
            new = bool(tail)
            if not tail and prev[0] not in "KT":
                new = True                                  # ひらがなだけの語は次の塊とつながない
            continue
        if t == "T":
            words.append([(t, s)])
            new = False                                     # カタカナ語のあとのひらがなは送りがな扱い（「スニーカーで」は助詞で切れる）
            continue
        # 漢字（英字・数字）の塊: 複合動詞（射し＋込んで）だけ前の語へつなぐ
        nxt = runs[k + 1] if k + 1 < len(runs) else ("|", "")
        compound = (not new and words and len(words[-1]) >= 2 and words[-1][-1][0] == "H"
                    and len(words[-1][-1][1]) == 1 and words[-1][-1][1] in _RENYO_END and words[-1][-2][0] == "K"
                    and nxt[0] == "H" and nxt[1] not in _PARTICLES)
        if compound:
            words[-1].append((t, s))
        else:
            words.append([(t, s)])
        new = False
    return [w for w in words if any(s for _t, s in w)]


def word_spans(display: str, sung: str) -> list[tuple[int, int, str, bool]] | None:
    """**推した語**ごとに、歌うかな（印を外したもの＝`lyric_slots.sung_kana`）の中の位置 `(始め, 終わり, 見せる字, anchored)`。
    当たらなければ None（見せる歌詞のかなが歌う詞に出てこない＝突き合わせの所見が別に出る）。

    `anchored` は漢字かカタカナを含む語か（ひらがなだけの語は切れ目を推せないので、使う側は見ない）。
    **推定**なので外れうる（このファイル冒頭の表）。使う側は候補として出し、断じない。
    """
    import re
    ws = _words(display)
    if not ws or not sung:
        return None
    pat = ""
    for w in ws:
        part = ""
        for t, s in w:
            if t == "K":
                part += ".+?"
            else:
                part += "".join("[あいえお]?" if c == "ー" else re.escape(c) for c in _norm(s))
        pat += f"({part})"
    m = re.fullmatch(pat, _norm(sung))
    if not m:
        return None
    return [(m.start(g), m.end(g), "".join(s for _t, s in w), any(t in "KT" for t, _s in w))
            for g, w in enumerate(ws, 1)]


def full_text(by_index: dict[int, str], display_by_index: dict[int, str]) -> str:
    """保存用の**表示歌詞**（1フレーズ＝1行）。字幕も投稿も行単位で使うのでこの形にする。

    ★**見せる詞は書いたとおり**（2026-10-02 本人決定・案2）。空は詞の型（`song_lyrics.PhraseLyric`）が差し戻すので、
      歌うかながあるのに見せる詞が無いフレーズが来たら不具合として止める（`AssemblyError`）。歌う音の無いフレーズ（印が全部 `*`）は空の行。
      以前は見せる詞の無いフレーズを**歌う歌詞（印を外したかな）で埋めていた**（読みにくいが欠けない、としていた）。
    """
    lack = [i for i in sorted(by_index) if not (display_by_index.get(i) or "").strip() and sung_kana(by_index.get(i, ""))]
    if lack:
        from song_fix import AssemblyError                  # noqa: E402（循環を避けてここで）
        raise AssemblyError(f"見せる詞の無いフレーズ（{', '.join(map(str, lack))}）が来た——詞の型が空を差し戻すはず")
    return "\n".join((display_by_index.get(i) or "").strip() for i in sorted(by_index))
