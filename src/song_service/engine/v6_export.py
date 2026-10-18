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

"""**VOCALOID6 に歌声を書き出させる**（人が開いて操作する工程を機械に渡す）。

## なぜこれがあるか

曲づくりは全部自動でも、**最後の「V6 で開いてミックスダウンして保存」だけが人手**だった。
本人（2026-09-05）:「いちいち操作した上で作って書き出して聞くのが面倒くさい」。
聴いて良し悪しを決めるのは人にしかできないが、**操作は機械にできる**。

## 何をしているか

VOCALOID6 Editor には CLI も公開 API も無い（最新の 6.13 まで確認済み）。
そこで **UI Automation** で操作する。要点は3つ:

- **実行時に LLM を挟まない**。画面を見て判断させる方式（Computer Use）ではなく、
  `AutomationId` で部品を名指しする。速い・無料・毎回同じ結果になる
- **座標を一切使わない**。解像度や DPI、ウィンドウ位置が変わっても効く
- V6 の書き出しダイアログは部品に ID が振ってある（`xAudioMixdownDlg` / `xRBMasterTrack` /
  `xOKButton`）ので、**表示言語や文言が変わっても壊れにくい**

実処理は `export_vocal.ps1`（PowerShell）。**別 venv を subprocess で呼ぶ既存の作法**
（YourMT3 の `venv_mt3`）と同じ形にしてある。

## 人手と同じものが出ることの確認（2026-09-05）

同じ vpr を **①自動（PowerShell 7）②自動（PowerShell 5.1）③人が手で操作**の3通りで書き出し、
**3本とも sha256 が完全一致**した（`63BA4999DCA2AAE5…` / 5,644,844 バイト）。
音質の劣化どころか**ファイルとして同一**で、V6 の書き出しが決定論的であることも同時に分かった。
→ 記録は [decisions.md 2026-09-05](../../../../docs/dev/decisions.md)

## 使えないとき

V6 は**別売・有償**なので、入っていない環境のほうが多い。`available()` が False なら
**従来どおり仮歌ミックスまで**で止め、人に書き出しを頼む（機能を消すのではなく落とす）。
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import song_files                                          # 完成品・ログの名前の決まりは1か所（song_service の直下・`_bootstrap` が通す）

SERVICE = Path(__file__).resolve().parents[1]
SCRIPT = SERVICE / "export_vocal.ps1"

#: `.vpr` の関連付け。**インストーラが実際の設置先で登録する**ので、ここが一番確かな手がかり。
_ASSOC_KEY = r"SOFTWARE\Classes\VOCALOID6.vpr\shell\open\command"


def find_v6_exe() -> Path | None:
    """VOCALOID6 の実体を探す。見つからなければ None。

    **パスを決め打ちにしない**（2026-09-06）。設置先は利用者が変えられるので、
    決め打ちだと**別のドライブに入れた人を「持っていない」と誤判定**して、
    作曲機能を導入させないことになる——いちばん避けたい外し方。

    ただし**深追いもしない**（本人判断）。見るのは2つだけ:

    1. **`.vpr` の関連付け**——インストーラが実際の場所で登録する。しかもこの機能は
       `VOCALOID6.exe <file.vpr>` で起動するので、**ここが正しければ確実に動く**
    2. 既定のパス（`%ProgramFiles%` 経由＝システムドライブが C: でなくても効く）

    **歌声ライブラリ（初音ミク V6）の有無は見ない**——別エントリで入るので調べられはするが、
    名前で照合することになり、かえって誤判定を生む。編集ソフトがあれば導入させる。
    """
    import os
    import re
    try:
        import winreg
    except ImportError:                                    # Windows 以外（テスト環境など）
        winreg = None

    if winreg is not None:
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _ASSOC_KEY) as k:
                cmd = winreg.QueryValueEx(k, "")[0]
            m = re.match(r'\s*"([^"]+)"', cmd) or re.match(r"\s*(\S+\.exe)", cmd, re.I)
            if m and Path(m.group(1)).exists():
                return Path(m.group(1))
        except OSError:
            pass

    for base in (os.environ.get("ProgramFiles"), r"C:\Program Files"):
        if base:
            p = Path(base) / "VOCALOID6" / "Editor" / "VOCALOID6.exe"
            if p.exists():
                return p
    return None

#: 書き出しの待ち上限。**長い曲でも足りる値**にしてある（32秒の曲でレンダリング6秒だった）。
#: ここで打ち切られても曲自体は残るので、人が V6 で開けば続きはできる。
RENDER_LIMIT_SEC = 1800


def available() -> bool:
    """V6 で書き出せる環境か（実体の有無だけを見る・数ミリ秒）。

    **起動して確かめない**——判定のたびに GUI が立ち上がるのは論外なので、
    GPU 判定を `nvidia-smi` の有無で済ませたのと同じ考え方にしている。
    """
    return find_v6_exe() is not None and SCRIPT.exists()


def is_busy() -> bool:
    """V6 が既に起動しているか。

    **V6 は単一インスタンス**。起動中に別の vpr を渡すと、こちらが起こしたプロセスは
    ファイルを既存側へ渡して終了してしまい、自動操作は相手を見失う。
    人が手で開いているときに割り込まないためにも見る。

    ⚠ **これは排他ではない**（2026-09-07・並行セッション `baf3d7bf` の指摘）。
    見てから起動するまでの隙に相手が掴めば、同じ衝突が起きる。
    **V6 を使うセッションが1本だけなら実害は無い**が、2本以上が同時に走るなら
    札（ロック）が要る。**同じ判定を各所に書き足さないこと**——確認はこの関数に寄せる。
    """
    try:
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq VOCALOID6.exe", "/NH"],
                           capture_output=True, text=True, timeout=15)
        return "VOCALOID6.exe" in (r.stdout or "")
    except Exception:
        return False                                   # 判定できないときは止めない（実行側で失敗する）


def target_vpr(song_dir: Path, version: int) -> Path | None:
    """その版の書き出し元 vpr を返す。

    版つき（`vocal_v{N}.vpr`）が実体で、固定名（`vocal.vpr`）は作業中しか無い
    ——どちらもありうるので**両方見る**（[session.py の「作業台方式」](../session.py)）。
    """
    for name in (song_files.versioned(song_files.VPR, version), song_files.VPR):
        p = song_dir / name
        if p.exists():
            return p
    return None


def score_form(song_dir: Path, version: int) -> str | None:
    """その版の楽譜の形（`song`／`acappella`／`instrumental`）。楽譜が無ければ None（2026-10-03・§5-4・§6-2）。

    vpr と同じ順（版つき `score_v{N}.json` → 作業台の `score.json`）で探す。欄の無い古い楽譜は `song`（`SongScore.form` の既定と同じ）。
    ★**書き出しの分かれ方は楽譜の形で決める**——フォルダに伴奏 wav があるか、では決めない（前の形のファイルが残りうる・§4-4）。
    """
    for name in (song_files.versioned(song_files.SCORE, version), song_files.SCORE):
        p = song_dir / name
        if p.is_file():
            try:
                return song_files.as_form(json.loads(p.read_text(encoding="utf-8")).get("form"))
            except (ValueError, OSError) as e:
                raise ValueError(f"{name} の形が読めない: {e}") from e
    return None


def finish_as_master(src: Path, song_dir: Path, form: str, version: int, *, copy: bool = False) -> tuple[Path, str]:
    """**仕上げ（`_finish_audio`：末尾の無音・−14 LUFS）を当てて完成品の名前にする**（2026-10-03・§6-2・F14①）。(完成品, 仕上げの説明)。

    - 歌もの・アカペラ（`copy=False`）: V6 の生の書き出し（`vocal_v{N}.wav`）を**仕上げてから** `master_<形>_v{N}.wav` へ付け替える。
      生の書き出しは残さない（同じ場所の中での名前の変更。生成物なので元の名前は残さない）。
      ★順番: `_fix_balance` は伴奏の音量を直して**同じ名前へ書き出し直す**ので、その後に仕上げ、最後に付け替える
    - インスト（`copy=True`）: 伴奏 wav（主旋律の楽器を含む）の**写し**を完成品の名前で書いてから仕上げる。伴奏 wav は途中の物として残す
    仕上げに失敗しても完成品は作る（曲は出来ている・説明に ★ で残す。以前の `vocal_v{N}.wav` と同じ扱い）。
    """
    import os
    import shutil
    dst = song_dir / song_files.master(form, version)
    if copy:
        shutil.copyfile(src, dst)
        note = _finish_audio(dst)
    else:
        note = _finish_audio(src)
        os.replace(src, dst)
    return dst, note


def export(vpr: Path, out_wav: Path, *, timeout_sec: int | None = None) -> tuple[bool, str]:
    """vpr を開いてミックスダウンし、wav を書き出す。`(成功したか, 説明)` を返す。

    **例外を投げない**——ここは「できたら儲けもの」の工程で、失敗しても曲そのものは
    出来上がっている。呼び出し側が人へ案内できるよう、理由を文で返す。
    """
    exe = find_v6_exe()
    if exe is None or not SCRIPT.exists():
        return False, "VOCALOID6 が見つからない（別売・有償のソフトなので、無い環境では書き出せない）"
    if is_busy():
        return False, "VOCALOID6 が既に開いている。閉じてからにする（同時にひとつしか動かせない）"
    if not vpr.exists():
        return False, f"書き出す vpr が無い: {vpr.name}"

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    limit = timeout_sec or RENDER_LIMIT_SEC
    # **`powershell`（Windows 標準の 5.1）で呼ぶ**。pwsh 7 は Windows に入っていないので
    # 配布物の前提にできない（`setup_common.ps1` も `powershell` で呼んでいる＝作法を揃える）。
    # ※スクリプトは **BOM 付き UTF-8** で保存すること。BOM が無いと 5.1 が cp932 として読み、
    #   日本語コメントが化けて構文エラーになる（2026-09-05 に実際に踏んだ）。
    # ★**確かめたパスをそのまま起動に渡す**（2026-09-06）。スクリプト側にも同じパスを
    #   決め打ちしていたので、既定以外の場所に入れた人は「判定は通るのに書き出しで失敗」
    #   という分かりにくい壊れ方をするところだった。**見つける場所は1つ**にする。
    cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
           "-Vpr", str(vpr), "-Out", str(out_wav), "-RenderSec", str(limit),
           "-Exe", str(exe)]
    try:
        # ★**読み方を明示する**（2026-09-06）。`text=True` だけだと Python は既定の
        #   cp932 で読もうとし、PowerShell が出す UTF-8 の日本語で `UnicodeDecodeError` になる。
        #   落ちるのは**出力を読むところ**なので、書き出し自体は成功しているのに
        #   「書き出せなかった」と報告される——原因が分かりにくい壊れ方をしていた。
        #   `errors="replace"` も付けて、化けても**止まらない**ようにする。
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=limit + 300,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, f"書き出しが {limit} 秒で終わらなかった"
    except Exception as e:
        return False, f"書き出しを起動できなかった: {type(e).__name__}: {e}"

    # ★**成功しても記録を残す**（2026-09-10）。PowerShell 側は段階ごとに時刻を出しているのに、
    #   ここで**失敗したときしか読んでいなかった**。本人に「ミックスダウンは GUI なら
    #   10秒で終わるのに、なぜ12分かかるのか」と聞かれて、**測った値が1つも無かった**。
    #   曲フォルダに置くので、あとから何段階でどれだけかかったかを追える。
    # ★名前は `log_v6export_v{N}.txt`（2026-10-03・F15②・F16①。以前は `vocal_v{N}.export.log`）。版の無い書き出し先なら名前の頭だけ揃える
    n = song_files.version_of(out_wav.name)
    log_name = song_files.v6export_log(n) if n else f"{song_files.LOG_PREFIX}v6export_{out_wav.stem}.txt"
    try:
        (out_wav.parent / log_name).write_text(
            (r.stdout or "") + (("\n--- stderr ---\n" + r.stderr) if r.stderr else ""),
            encoding="utf-8")
    except Exception:                                      # 記録は落としてよい（本筋ではない）
        pass

    if r.returncode != 0 or not out_wav.exists():
        tail = (r.stdout or "").strip().splitlines()[-3:] or [(r.stderr or "").strip()[-200:]]
        return False, "書き出せなかった: " + " / ".join(t.strip() for t in tail if t)
    return True, f"{out_wav.name}（{out_wav.stat().st_size / 1024 / 1024:.1f} MB）"


#: 狙い——**歌が伴奏より何 dB 前に出るか**の**既定**。曲ごとの指定（`mix.json` の `vocal_lead_db`）が
#: あればそちらを狙う（`lead_target_db`・2026-09-27）。`score_to_vpr._accomp_gain_db` の既定 4.0 と同じ値。
LEAD_DB = 4.0
#: この幅に収まっていれば直さない（測定の揺れぶん）
LEAD_TOL = 1.5
#: **これ以上の補正はしない**。ここを超えたら、ずれているのは音量ではなく**測定の前提**。
#: ★歯止めが無く **+50.3dB** を書き込んだ事故があった（2026-09-06『青空の境界線』）——
#: 伴奏が 2秒の無音だったため「歌が伴奏より +54.3dB」という値が出て、それをそのまま信じた。
#: **最初 12dB にしたら正当な補正（-14.4dB）まで弾いた**ので広げた（同日）。
#: 実測の必要量は -11.3dB / -14.4dB。壊れたときの値は +54dB なので、**24dB で両者は分けられる**。
#: なお**本当の歯止めは下の `_FADER_MIN/MAX`**（V6 が受け付ける範囲に収める）で、ここは前段の網。
MAX_SHIFT_DB = 24.0

#: vpr のフェーダーが取りうる範囲（**1/10 dB 単位**）。V6 の実機で確認できる上限は +6.0dB。
#: 下は無音に近づくだけなので広く取る。**範囲外の値を書かない**ことが最後の砦——
#: 事故のときは 538（+53.8dB）という、V6 が解釈できない値が vpr に入った。
_FADER_MIN, _FADER_MAX = -600, 60


def _measure_lead(mix_wav: Path, accomp_wav: Path) -> float | None:
    """完成ミックスと素の伴奏から「**歌が伴奏より何 dB 大きいか**」を出す。

    完成ミックスの中の伴奏は**素の伴奏を定数倍したもの**なので、最小二乗で倍率を求めて引けば
    歌成分が残る（V6 側のフェーダー値が分からなくても 測定 できる）。
    比べるのは**歌っている区間だけ**——間奏を含めると歌が 0 になり、不当に低く出る。
    """
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return None
    try:
        def mono(p: Path):
            d, sr = sf.read(p, dtype="float64")
            return (d.mean(axis=1) if d.ndim > 1 else d), sr

        mix, sr = mono(mix_wav)
        acc, _ = mono(accomp_wav)

        # ★**測れない伴奏なら測らない**（2026-09-06）。伴奏が無音・極端に短いと
        #   「歌が伴奏より +54dB」のような破綻した値が出て、それを信じた補正が暴走する。
        #   実際、伴奏が **2秒の無音**だった曲で **+50.3dB** の補正を書き込んだ。
        if float(np.sqrt(np.mean(acc ** 2))) < 1e-5:       # ほぼ無音
            return None
        if len(acc) < len(mix) * 0.5:                      # 完成ミックスの半分にも満たない
            return None

        n = min(len(mix), len(acc))
        if n < sr:                                         # 1秒未満は測らない
            return None
        mix, acc = mix[:n], acc[:n]
        denom = float(np.dot(acc, acc))
        if denom <= 0:
            return None
        g = float(np.dot(mix, acc) / denom)
        voc = mix - g * acc
        w = int(sr * 0.25)
        fr = [(i, i + w) for i in range(0, n - w, w)]
        rms = lambda x: float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0   # noqa: E731
        v = np.array([rms(voc[a:b]) for a, b in fr])
        a = np.array([rms((acc * g)[a:b]) for a, b in fr])
        sing = v > (v.max() * 0.08)
        if sing.sum() < 4:
            return None
        db = lambda x: 20 * np.log10(x) if x > 1e-12 else -99.0              # noqa: E731
        return db(float(np.median(v[sing]))) - db(float(np.median(a[sing])))
    except Exception:
        return None


def _accomp_wav(song_dir: Path, version: int) -> Path | None:
    for name in (song_files.versioned(song_files.ACCOMP_WAV, version), song_files.ACCOMP_WAV):
        p = song_dir / name
        if p.exists():
            return p
    return None


def _shift_accomp_fader(vpr: Path, delta_db: float) -> bool:
    """vpr の伴奏トラックの音量を `delta_db` だけ動かす（**1/10 dB 単位**で持っている）。"""
    import json
    import zipfile
    try:
        with zipfile.ZipFile(vpr) as z:
            names = z.namelist()
            blobs = {n: z.read(n) for n in names}
        seq = json.loads(blobs["Project/sequence.json"])
        hit = False
        for t in seq.get("tracks", []):
            if t.get("type") != 1:                         # InstrumentalTrack（伴奏）だけ
                continue
            ev = t.setdefault("volume", {}).setdefault("events", [{"pos": 0, "value": 0}])
            want = int(round(ev[0].get("value", 0) + delta_db * 10))
            # ★**V6 が解釈できない値を書かない**（2026-09-06）。事故のときは 538（+53.8dB）が入った。
            #   計算がどう転んでも、vpr に残るのは必ず正常な範囲の値にする。
            got = max(_FADER_MIN, min(_FADER_MAX, want))
            if got != want:
                return False                               # 収まらない＝そもそも要求がおかしい
            ev[0]["value"] = got
            hit = True
        if not hit:
            return False
        blobs["Project/sequence.json"] = json.dumps(seq, ensure_ascii=False).encode("utf-8")
        with zipfile.ZipFile(vpr, "w", zipfile.ZIP_DEFLATED) as z:
            for n in names:
                z.writestr(n, blobs[n])
        return True
    except Exception:
        return False


#: 配信の目安（Spotify / YouTube / TikTok）。Apple Music は −16。
TARGET_LUFS = -14.0
#: トゥルーピークの上限。超えると配信のエンコードで歪む。
TRUE_PEAK_CEIL = -1.0
#: 末尾に残してよい無音（秒）。余韻ぶんは残す。
TAIL_SILENCE_SEC = 2.0


def _finish_audio(wav: Path) -> str:
    """書き出した音を**製品として出せる形**に整える（2026-09-07 新設）。

    ## なぜ要るか

    それまで音量は「歌が伴奏より何 dB か」しか見ておらず、**曲全体の絶対音量**も
    **末尾の無音**も測っていなかった。実測すると:

    - **統合ラウドネス −23〜−27 LUFS**（配信の目安 −14）＝ **9〜13dB 小さい**
    - **末尾に 6〜75秒の無音**（本人の指摘「55秒で伴奏が終わって長い空白があるんだけど、それは何?」）

    V6 は vpr の長さぶんを書き出すので、伴奏 wav の余白がそのまま無音として残る。

    ## やること

    1. **末尾の無音を切る**（余韻ぶん 2秒は残す）
    2. **−14 LUFS へ揃える**（トゥルーピーク −1dBTP を超えないよう、必要なら控える）

    測れない・直せないときは**何もせず説明だけ返す**——曲自体は出来ているので。
    """
    try:
        import numpy as np
        import pyloudnorm as pyln
        import soundfile as sf
    except ImportError as e:
        # ★**黙って飛ばさない**（2026-09-07）。道具が無いだけなのに何もせず返すと、
        #   **末尾に80秒の無音がある曲がそのまま出荷される**（実際に起きた——
        #   検証を本体 venv で走らせており、そこに pyloudnorm が無かった）。
        return f"／★仕上げ（末尾の無音・音量）を飛ばした（{e.name} が無い）"
    try:
        data, sr = sf.read(wav, dtype="float64")
        if len(data) < sr:
            return ""
        mono = data.mean(axis=1) if data.ndim > 1 else data
        note = []

        # ---- ① 末尾の無音を切る ----
        w = int(sr * 0.25)
        thr = 10 ** (-60 / 20)                             # −60dBFS を無音とみなす
        frames = [float(np.sqrt(np.mean(mono[i:i + w] ** 2)))
                  for i in range(0, len(mono) - w, w)]
        live = [i for i, v in enumerate(frames) if v > thr]
        if live:
            end = min(len(data), int((live[-1] + 1) * w + sr * TAIL_SILENCE_SEC))
            cut = (len(data) - end) / sr
            if cut > TAIL_SILENCE_SEC:
                data = data[:end]
                mono = mono[:end]
                note.append(f"末尾の無音 {cut:.0f}秒を切った")

        # ---- ② 突出したピークを抑える（リミッター）----
        # ★**これが無いと音量が上がらない**（2026-09-07 追加）。
        #   ピークだけを見て控えると、**1箇所の突出が曲全体の音量を押し下げる**——
        #   実測でピーク対ラウドネス比 20.7dB、ピークが −1.0dBFS に当たって
        #   **−21.7 LUFS で頭打ち**になった（目安は −14）。
        #   プロはここでコンプ／リミッターを使う。やることは
        #   **大きい所だけを滑らかに抑える**（全体を下げるのではない）。
        meter = pyln.Meter(sr)
        before = meter.integrated_loudness(data)
        if np.isfinite(before):
            # **どこまで抑えれば目標に届くか**を分布から決める（決め打ちにしない）。
            #   実測: p99.9 で抑えても 0.1% しか触らず **−18.4 LUFS 止まり**。
            #   p95（全体の5%）まで下ろすと **−14.0 LUFS** に届いた。
            #   5% はコンプレッサーが常時かかる領域で、音楽的にも普通の範囲。
            thr = 0.0
            for q in (99.5, 99.0, 97.0, 95.0, 92.0):
                t = float(np.percentile(np.abs(data), q))
                if t <= 0:
                    continue
                sg, mg = np.sign(data), np.abs(data)
                trial = sg * np.where(mg > t, t * (1 + np.log1p((mg - t) / t)), mg)
                lv = meter.integrated_loudness(trial)
                pk = float(np.abs(trial).max()) or 1.0
                # この閾値で抑えたとき、天井まで上げてどこまで届くか
                reach = lv + 20 * np.log10(min(10 ** ((TARGET_LUFS - lv) / 20),
                                               10 ** (TRUE_PEAK_CEIL / 20) / pk))
                thr = t
                if reach >= TARGET_LUFS - 0.5:             # 目標に届くならこれ以上抑えない
                    break
            over = np.abs(data) > thr
            if over.any() and thr > 0:
                # 超えたぶんを対数的に圧縮（超えるほど強く抑える＝歪ませない）
                sign = np.sign(data)
                mag = np.abs(data)
                comp = np.where(mag > thr, thr * (1 + np.log1p((mag - thr) / thr)), mag)
                data = sign * comp
                note.append(f"突出したピークを抑えた（{float(over.mean()) * 100:.2f}% の点）")
                before = meter.integrated_loudness(data)

            gain = 10 ** ((TARGET_LUFS - before) / 20)
            peak = float(np.abs(data).max()) * gain
            ceil = 10 ** (TRUE_PEAK_CEIL / 20)
            if peak > ceil:                                # それでも天井を超えるなら控える
                gain *= ceil / peak
            data = data * gain
            got = meter.integrated_loudness(data)
            note.append(f"音量を {before:.1f} → **{got:.1f} LUFS** に揃えた"
                        f"（配信の目安 {TARGET_LUFS:.0f}）")

        if note:
            sf.write(wav, data, sr)
            return "／" + "・".join(note)
    except Exception as e:
        # ★**失敗も黙らない**（2026-09-27）。したことは説明に返しているのに、失敗したときだけ空を返しており、
        #   「整えた」のか「整えられなかった」のか区別がつかなかった。曲は出来ているので止めはしない。
        return f"／★仕上げ（末尾の無音・音量）ができなかった（{type(e).__name__}: {e}）"
    return ""


def lead_target_db(song_dir: Path) -> float:
    """**この曲の**「歌が伴奏より何 dB 前に出るか」の狙い。`mix.json` の `vocal_lead_db`、無ければ `LEAD_DB`。

    ★2026-09-27 まで `_fix_balance` は定数 `LEAD_DB` を狙っていた（目録 音量-2）。一方で vpr の初期値
      （`score_to_vpr._accomp_gain_db`）は `mix.json` の `vocal_lead_db` を読むので、
      `adjust_song(vocal_lead_db=8)` と指定しても、書き出しを測って**+4dB へ戻し**、その差を
      `accomp_trim_db` に覚えて**以後の初期値でも打ち消していた**（コードから導いた結論・実測はしていない）。
      **読む場所を1つにする**——初期値と同じ値を狙う。
    """
    try:
        mj = song_dir / "mix.json"
        if mj.exists():
            v = json.loads(mj.read_text(encoding="utf-8")).get("vocal_lead_db")
            if v is not None:
                return float(v)
    except Exception:
        pass
    return LEAD_DB


def _fix_balance(song_dir: Path, vpr: Path, out: Path) -> str:
    """歌と伴奏のバランスを実測し、外れていたら**1回だけ**直して書き出し直す。

    **繰り返さない**——フェーダーの dB と実際の音量は素直に対応するので、
    1回の補正で狙いに収まることを実測で確かめてある（-6.1dB → +4.0dB）。
    直せなくても**曲は出来ている**ので、失敗しても静かに諦める（戻り値は説明の追記だけ）。
    狙いは曲ごとの指定（`lead_target_db`）。**測って直すこと自体は残す**——2026-09-06 に歌が伴奏より
    6〜7dB 埋もれた事故を直したもので、仮歌（サイン波）からの見積もりだけでは当たらない。
    """
    acc = _accomp_wav(song_dir, song_files.version_of(out.name) or 1)   # 版を読むのは `song_files.version_of` の1つ（F16①）
    if acc is None:
        return ""
    lead = _measure_lead(out, acc)
    if lead is None:
        return ""
    target = lead_target_db(song_dir)
    if abs(lead - target) <= LEAD_TOL:
        return f"／音量は狙いどおり（歌が伴奏より {lead:+.1f}dB・狙い {target:+.1f}）"

    delta = -(target - lead)                               # 伴奏を動かす量（下げるので負）
    # ★**ありえない補正量は書かない**（2026-09-06）。フェーダーで足りる範囲を超える値が出たら、
    #   ずれているのは音量ではなく**測定の前提**（伴奏が壊れている等）。曲は出来ているので
    #   静かに諦め、**何が変か分かる説明**を返す——黙って +50dB 書き込むより害が小さい。
    if abs(delta) > MAX_SHIFT_DB:
        return (f"／音量が測れなかった（歌が伴奏より {lead:+.1f}dB という値になり、"
                f"必要な補正 {delta:+.1f}dB が上限 ±{MAX_SHIFT_DB:.0f}dB を超えた）。"
                f"**伴奏が正しく作られているか確認して**")
    if not _shift_accomp_fader(vpr, delta):
        return f"／音量が狙いから外れている（歌が伴奏より {lead:+.1f}dB・直せなかった）"
    # ★**測った残差を覚える**（2026-09-20）。仮歌はサイン波なので曲ごとに 1dB 強ずれるが、
    #   **同じ曲では以後ほぼ一定**（『加速させて』6回で幅 0.7dB）。覚えないと、版を作るたびに
    #   vpr が楽譜から作り直されて捨てられ、**毎回 V6 を2回起こす**ことになる（実測14/14回）。
    #   覚えるのは**フェーダーの絶対値ではなく残差**——絶対値は伴奏の音量に依存するので、
    #   楽器を抜くなど編成を変えると外れる。残差は「サイン波と実歌声の差」なので編成に依らない。
    try:
        from song_pipeline import set_mix_setting
        cur = 0.0
        mj = song_dir / "mix.json"
        if mj.exists():
            cur = float(json.loads(mj.read_text(encoding="utf-8")).get("accomp_trim_db", 0.0))
        set_mix_setting(song_dir, accomp_trim_db=round(cur + delta, 2))
    except Exception:                                      # 覚えられなくても曲は出来ている
        pass
    ok, _ = export(vpr, out)
    if not ok:
        return f"／音量を直したが書き出し直せなかった（歌が伴奏より {lead:+.1f}dB）"
    after = _measure_lead(out, acc)
    if after is None:
        return f"／音量を直した（伴奏を {delta:+.1f}dB）"
    return (f"／音量を直した: 歌が伴奏より {lead:+.1f}dB → **{after:+.1f}dB**"
            f"（伴奏を {delta:+.1f}dB・狙い {target:+.1f}）")


def export_song_vocal(song_dir: Path, version: int) -> tuple[bool, str]:
    """曲フォルダの「いま作っている版」の歌声を書き出し、**仕上げて完成品にする**（`master_<形>_v{N}.wav`・2026-10-03・§6-2）。

    V6 の生の書き出しは **`vocal_v{N}.wav`**——人が V6 で書き出したときと同じ名前（V6 は vpr と拡張子違いの同名で出す）。
    それを（歌ものだけ）伴奏との釣り合いを直し、仕上げ、**完成品の名前へ付け替える**（生の書き出しは残さない・F14①）。
    **再生も状態判定も完成品の名前（`song_files.master`）を見る**。
    形は楽譜（`score_form`）から読む——**インストは書き出さない**（歌が無い。完成品は `render_final` が伴奏から作る・V6 を起こさない）。
    **アカペラは伴奏との釣り合いを測らない**（伴奏が無い。前の版の伴奏 wav が残っていても測らない・§5-4）——黙って飛ばさず返事に書く。
    """
    form = score_form(song_dir, version)
    if form is None:
        return False, f"v{version} の楽譜が見つからない（先に書き出しの工程を通す）"
    if form == "instrumental":
        return False, (f"インストなので歌声の書き出しは無い。曲は render_final で完成している"
                       f"（{song_files.master(form, version)}）")
    vpr = target_vpr(song_dir, version)
    if not vpr:
        return False, f"v{version} の vpr が見つからない（先に書き出しの工程を通す）"
    out = song_dir / song_files.v6_raw(version)
    ok, detail = export(vpr, out)
    if not ok:
        return ok, detail

    # ★**書き出した音を測って、伴奏の音量を直す**（2026-09-06 追加）。
    #   vpr に焼き込む伴奏の音量は `score_to_vpr._accomp_gain_db` が**仮歌（サイン波）と
    #   伴奏の RMS 比**から決めている。**サイン波と実歌声では音量が違う**ので当たらない
    #   ——実測で**歌が伴奏より 6〜7dB 小さい**（狙いは歌が +4dB 大きい）＝10dB ずれていた。
    #   コード側も「実歌声の音量は書き出すまで不明なので、あくまで初期値（微調整は V6 の
    #   フェーダーで）」と認めていた。**人が触る前提が崩れた**（`export_vocal` で自動化した）ので、
    #   機械が実測して決め直す。
    if form == "song":
        fixed = _fix_balance(song_dir, vpr, out)
    else:
        fixed = "／アカペラなので伴奏との釣り合いは測らない（伴奏が無い）"
    # ★**最後に製品の形へ整え、完成品の名前へ付け替える**（2026-09-07・2026-10-03）。順番が大事——
    #   `_fix_balance` は伴奏の音量を直すために**同じ名前へ書き出し直す**ので、
    #   末尾を切ったり音量を揃えたりするのは**そのあと**でないと上書きされる。付け替えは仕上げの後（`finish_as_master`）
    dst, note = finish_as_master(out, song_dir, form, version)
    return True, f"{dst.name}（{dst.stat().st_size / 1024 / 1024:.1f} MB）" + fixed + note
