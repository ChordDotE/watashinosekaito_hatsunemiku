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

# 作曲機能（song_service）のセットアップ
#   実行: powershell -ExecutionPolicy Bypass -File setup_song_service.ps1
#   単独でも、インストーラ（setup_unattended.ps1）からでも実行できる（何度でも再実行可・不足分だけ取得）。
#
# 【隔離方針】本体のセットアップ（setup_common.ps1）と同じ:
#   - uv は miku\runtime\uv\uv.exe を**共用**（無ければ同じ場所へDL。バージョンも本体と揃える）
#   - Python 本体（3.10）は UV_PYTHON_INSTALL_DIR = miku\runtime\python に入る
#     → **グローバル環境・本体の Python 3.13 と競合しない**（ユーザーの AppData も汚さない）
#   - venv 3本・モデル・FluidSynth は song_service\ 配下
#   → アンインストールは「miku\runtime\ と song_service\ の該当フォルダを消す」だけで完結
#
# 【2段構え】(2026-09-01) **音源しか無い曲のカバーを使わないなら GPU も約15GB も要らない**:
#   -OriginalOnly … オリジナル曲・楽譜（MIDI）が見つかる曲のカバー・鼻歌の聞き取りに要るもの。
#                   名前は「オリジナル」だが、作れないのは**音源しか無い曲のカバーだけ**。
#                   **約3.9GB・十数分**（2026-09-23 に改訂）:
#                     ——うち 3.1GB は**文字起こしのモデル**（肉声の鼻歌から歌詞を採る。CPU で動く）。
#                     それ以前は約800MB だった（実測 2026-09-02）:
#                     venv 415MB（requirements_original.txt）＋ ffmpeg 307MB
#                     ＋ SoundFont 31MB ＋ FluidSynth 8MB
#                   **取りに行かないもの**: カバー用 venv 2本（3.4GB）・SOFA モデル 1.1GB・
#                     YourMT3 ckpt 0.5GB・事前取得のモデルのうち whisperx 整列2種と demucs・
#                     Python 3.10（3.10 は SOFA/YourMT3 専用）。
#                   統合 venv（8.2GB）との差は torch cu128・demucs・whisperx・pyannote で、
#                   **全部「音源しか無い曲のカバー」用**。
#                   オリジナルの経路は LLM＝クラウド／FluidSynth・numpy・libresvip＝CPU で、
#                   **GPU を1バイトも使わない**。
#   既定（フル）  … 音源しか無い曲のカバーも使う。GPU(CUDA)が要る。
# 所要: フルは DL合計 約15GB（ディスク使用 約30GB・uvキャッシュ込み）・40〜90分（回線次第）。
# モデルの重みもすべてここで明示DLする（初回の曲作成中に暗黙DLさせない・watchdog 対策）。
# 実歌声には VOCALOID6＋初音ミク V6（別売・有償）が必要（無い場合は仮歌ミックスまで）。
param([switch]$SkipVenv, [switch]$SkipModels, [switch]$OriginalOnly)
$ErrorActionPreference = "Continue"
$ProgressPreference    = "SilentlyContinue"   # PS5.1 の IWR は進捗表示でDLが大幅に遅くなる（別プロセスなので親の設定は継がない）

$SVC        = $PSScriptRoot                                  # miku\src\song_service
$MIKU_DIR   = Split-Path (Split-Path $SVC -Parent) -Parent   # miku
$TOOLS      = Join-Path $SVC "tools_bin"                     # FluidSynth / SoundFont（サービス専用）
$uvDir      = Join-Path $MIKU_DIR "runtime\uv"
$uvExe      = Join-Path $uvDir "uv.exe"
$UV_VERSION = "0.11.32"   # setup_common.ps1 と同じピン留め（更新時は両方を再検証）

# uv の Python/キャッシュ置き場を本体と共有（グローバルに撒かない）
$env:UV_PYTHON_INSTALL_DIR = Join-Path $MIKU_DIR "runtime\python"
$env:UV_CACHE_DIR          = Join-Path $MIKU_DIR "runtime\uv_cache"

# モデル類のキャッシュも本体と同じ場所へ（ユーザープロファイルに残さない。
# system_start.ps1 が実行時に同じ値を設定するので、ここで取得した実体がそのまま使われる）
$env:HF_HOME    = Join-Path $MIKU_DIR "runtime\models\hf"
$env:TORCH_HOME = Join-Path $MIKU_DIR "runtime\models\torch"

$svcPython = Join-Path $SVC "venv\Scripts\python.exe"

New-Item -ItemType Directory -Force $TOOLS | Out-Null

# どの段階を実行するかを先に決める。フルは 40〜90分かかるため、UI側（コントロールパネル）が
# 「[2/7] ...」を拾って進捗バーに出せるよう、総数を最初に確定させておく。
# 判定をここに集約しているので、下の if と二重管理にならない
# YourMT3 ckpt の配置先（mt3-infer が cwd 基準で参照するモデル別ディレクトリ名に合わせる）
$mt3CkptDir = Join-Path $SVC ".mt3_checkpoints\yourmt3\mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops"

$needUv        = -not (Test-Path $uvExe)
$needVenv      = -not $SkipVenv
$cover         = -not $OriginalOnly        # 音源しか無い曲のカバー用のものを取るか（MIDI のカバーは要らない）
# venv の「構築済み」判定は python.exe でなく torch の実体で見る（3本すべてに入る代表パッケージ）。
# uv venv は pip install の前に python.exe を作るため、インストール失敗した空 venv を
# 「済み」と誤判定して再実行で修復されなくなる（2026-08-09 のテストインストールで実害）
# ★カバー専用のものは **-OriginalOnly をこの判定に畳み込む**（下の if 側に足さない）。
#   以前は if 側にだけ書いていて**3か所で書き漏れ**、-OriginalOnly でも SOFA モデル 1.2GB と
#   YourMT3 ckpt 536MB を取りに行っていた（2026-09-02・クリーン VM で発覚）。
#   ここに集約すれば、**進捗の分母（$TotalSteps）も同時に正しくなる**——飛ばす工程を数えない。
# 統合 venv の「構築済み」の目印は**モードで変える**。カバー込みなら torch、オリジナルだけなら librosa
# （どちらもその構成の代表で、途中で失敗した空 venv を見逃さない）。
# これで **GPU を後から積んだ人**は再実行で統合版に作り直され（torch が無いので）、
# 逆に統合版が入っている人が -OriginalOnly を叩いても**わざわざ減らさない**（librosa はあるので素通り）。
#
# ★★**この目印で飛ばすのは「作り直し（uv venv --clear）」だけ。install は毎回通す**
#   （2026-09-23 本人指示）。**以前は install ごと飛ばしていて、依存を足しても入らなかった**
#   ——`librosa` が1つあれば「出来ている」と見なすので、2026-09-23 に足した6件
#   （faster-whisper・ctranslate2・av・onnxruntime・tokenizers・huggingface-hub）が
#   **既に入れている人には永久に入らない**。しかも最後に **selfcheck: PASS と表示される**
#   （クリーン VM で実測。`import faster_whisper` が全滅なのに PASS だった）。
#   ★**目印を足した依存に付け替える直し方は採らない**——**次に足したときに同じことが起きる**。
#   本人:「**インストール済みのは自動で飛ばされるだけだし、毎回インストールさせてもいい**」。
#   uv は満たされていれば数秒で抜けるので、再実行が重くならない。**更新し忘れる余地を無くす。**
$mainMarker    = if ($cover) { "torch" } else { "librosa" }
$haveMainVenv  = Test-Path (Join-Path $SVC "venv\Lib\site-packages\$mainMarker")
$needMainVenv  = $needVenv        # ★常に通す（中で「作り直すか」だけを $haveMainVenv で決める）
$needSofaVenv  = $needVenv -and $cover -and (-not (Test-Path (Join-Path $SVC "sofa_venv\Lib\site-packages\torch")))
$needMt3Venv   = $needVenv -and $cover -and (-not (Test-Path (Join-Path $SVC "venv_mt3\Lib\site-packages\torch")))
$needSofaRepo  = $cover -and (-not (Test-Path (Join-Path $SVC "sofa_repo\infer.py")))
$needSofaModel = $cover -and (-not $SkipModels) -and (-not (Test-Path (Join-Path $SVC "sofa_model\step.100000.ckpt")))
$needMt3Ckpt   = $cover -and (-not $SkipModels) -and (-not (Test-Path (Join-Path $mt3CkptDir "last.ckpt")))
$needFfmpeg    = -not (Test-Path (Join-Path $TOOLS "ffmpeg\bin\ffmpeg.exe"))
$needFluid     = -not (Test-Path (Join-Path $TOOLS "fluidsynth\bin\fluidsynth.exe"))
$needSf2       = -not (Test-Path (Join-Path $TOOLS "GeneralUser-GS.sf2"))
$needPrefetch  = $cover -and (-not $SkipModels)   # 取得済み分はライブラリ側キャッシュ判定で数秒で抜ける
# Python 3.10 は **SOFA と YourMT3 のためだけ**に要る（本体は 3.13）。オリジナルだけなら取らない
$needPy310     = $needVenv -and $cover

$TotalSteps = 0
foreach ($f in $needUv, $needPy310, $needMainVenv, $needSofaVenv, $needMt3Venv, $needSofaRepo,
               $needSofaModel, $needMt3Ckpt, $needFfmpeg, $needFluid, $needSf2, $needPrefetch) {
    if ($f) { $TotalSteps += 1 }
}
$StepNo = 0

function Step($name, $script) {
    $script:StepNo++
    Write-Host "[$script:StepNo/$script:TotalSteps] $name" -ForegroundColor Cyan
    try { & $script; Write-Host "   OK" -ForegroundColor Green }
    catch { Write-Host "   失敗: $_（続行。最後のselfcheckで確認）" -ForegroundColor Yellow }
}

# ── 1. uv（本体セットアップと共用。無ければ同じ場所へDL）──
if ($needUv) {
    Step "uv のダウンロード（miku\runtime\uv）" {
        New-Item -ItemType Directory -Force $uvDir | Out-Null
        $zip = Join-Path $uvDir "uv.zip"
        Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip
        Expand-Archive $zip $uvDir -Force
        Remove-Item $zip -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "== uv は既にあります: $(& $uvExe --version)" -ForegroundColor Cyan
}
if (-not (Test-Path $uvExe)) {
    Write-Host "uv を用意できませんでした。ネットワークを確認して再実行してください" -ForegroundColor Red
    exit 1
}

# ── 2. Python 3.10 取得 + venv 3本 ──
# 本体は Python 3.13（miku\venv）。SOFA と YourMT3 は 3.10 でしか動かないため別venvにする。
# uv が 3.10 を runtime\python へ取得するので、システムの Python には一切触れない。
if ($needVenv) {
    if ($needPy310) {
        Step "Python 3.10 の取得（miku\runtime\python 配下・グローバル非汚染）" {
            & $uvExe python install 3.10
            if ($LASTEXITCODE -ne 0) { throw "uv python install に失敗" }
        }
    }
    # ★**作り直しは「無いとき」だけ・install は毎回**（2026-09-23 に分けた。理由は上の $mainMarker）。
    #   `uv venv --clear` は既存 venv を捨てて作り直すので、これは目印で守る。
    #   `uv pip install -r` は**満たされていれば数秒で抜ける**ので、毎回通してよい
    #   ——**こうしておけば、あとで依存を足したときに何も直さなくても行き渡る。**
    if ($needMainVenv) {
        # ★オリジナルだけなら **requirements_original.txt**（415MB）。統合版（8.2GB）は
        #   torch cu128・demucs・whisperx・pyannote を含み、**どれもカバー専用**。
        #   両者は「オリジナル ⊂ 統合」かつ**版まで一致**させてある（片方でしか動かない状態を作らない）。
        #   後から GPU を積んだ人は、引数なしで再実行すれば統合版に作り直される（--clear があるので入れ直る）。
        Step $(if (-not $haveMainVenv) {
                   if ($cover) { "venv（統合・Python 3.13・torch cu128）の構築" }
                   else { "venv（オリジナル曲用・Python 3.13・GPU なし）の構築" }
               } else { "venv の依存を最新にそろえる（足りないものだけ入る）" }) {
            if (-not $haveMainVenv) {
                & $uvExe venv (Join-Path $SVC "venv") --clear --python 3.13
                if ($LASTEXITCODE -ne 0) { throw "venv の作成に失敗" }
            }
            if ($cover) {
                & $uvExe pip install --python $svcPython `
                    -r (Join-Path $SVC "requirements_full_main.txt") `
                    --extra-index-url https://download.pytorch.org/whl/cu128 `
                    --index-strategy unsafe-best-match
            } else {
                & $uvExe pip install --python $svcPython `
                    -r (Join-Path $SVC "requirements_original.txt")
            }
            if ($LASTEXITCODE -ne 0) { throw "統合venvのインストールに失敗" }
        }
    }
    if ($needSofaVenv) {
        Step "sofa_venv（SOFA歌用アライナ・Python 3.10・torch 2.4.1+cpu）の構築" {
            & $uvExe venv (Join-Path $SVC "sofa_venv") --clear --python 3.10
            & $uvExe pip install --python (Join-Path $SVC "sofa_venv\Scripts\python.exe") `
                -r (Join-Path $SVC "requirements_full_sofa.txt") `
                --extra-index-url https://download.pytorch.org/whl/cpu `
                --index-strategy unsafe-best-match
            if ($LASTEXITCODE -ne 0) { throw "sofa_venv のインストールに失敗" }
        }
    } else { Write-Host $(if ($cover) { "== sofa_venv は既にあります" }
                          else { "== sofa_venv は入れません（音源からのカバー専用）" }) -ForegroundColor Cyan }
    if ($needMt3Venv) {
        Step "venv_mt3（YourMT3伴奏採譜・Python 3.10）の構築" {
            & $uvExe venv (Join-Path $SVC "venv_mt3") --clear --python 3.10
            & $uvExe pip install --python (Join-Path $SVC "venv_mt3\Scripts\python.exe") `
                -r (Join-Path $SVC "requirements_full_mt3.txt") `
                --extra-index-url https://download.pytorch.org/whl/cpu `
                --index-strategy unsafe-best-match
            if ($LASTEXITCODE -ne 0) { throw "venv_mt3 のインストールに失敗" }
        }
    } else { Write-Host $(if ($cover) { "== venv_mt3 は既にあります" }
                          else { "== venv_mt3 は入れません（音源からのカバー専用）" }) -ForegroundColor Cyan }
}

# ── 3. SOFA リポジトリ（zip取得＝git不要）──
if ($needSofaRepo) {
    Step "SOFA リポジトリ（qiuqiao/SOFA）" {
        $zip = Join-Path $env:TEMP "sofa_repo.zip"
        Invoke-WebRequest "https://github.com/qiuqiao/SOFA/archive/refs/heads/main.zip" -OutFile $zip
        Expand-Archive $zip $env:TEMP -Force
        Move-Item (Join-Path $env:TEMP "SOFA-main") (Join-Path $SVC "sofa_repo") -Force
        Remove-Item $zip -ErrorAction SilentlyContinue
    }
}

# ── 4. モデル（公式配布元からDL代行＝再配布しない）──
# SOFA 日本語モデルの配布元は作者の GitHub Releases。
# ※ 旧配布元 HuggingFace (Greenleaf2001/SOFA_Models) は 2026-08 時点でアカウントごと消滅。
#   GitHub Releases 側は手元実体と全4ファイルのサイズ完全一致を確認済み（2026-08-09・decisions.md）
$sofaBase = "https://github.com/Greenleaf2001/SOFA_Models/releases/download/JPN_Test2"
if ($needSofaModel) {
    Step "SOFA 日本語モデル JPN Test2（GitHub Releases・約1.2GB）" {
        New-Item -ItemType Directory -Force (Join-Path $SVC "sofa_model") | Out-Null
        # readme/data_providers は訓練データ提供者のクレジット記録として一緒に保存する
        foreach ($f in "step.100000.ckpt", "japanese-extension-sofa.txt", "hparams.yaml", "readme.md", "data_providers.md") {
            Invoke-WebRequest "$sofaBase/$f" -OutFile (Join-Path $SVC "sofa_model\$f")
        }
        if (-not (Test-Path (Join-Path $SVC "sofa_model\step.100000.ckpt"))) { throw "SOFAモデルの取得に失敗" }
    }
    if (-not (Test-Path (Join-Path $SVC "sofa_model\step.100000.ckpt"))) {
        Write-Host "   手動手順: $sofaBase から" -ForegroundColor Yellow
        Write-Host "   step.100000.ckpt・japanese-extension-sofa.txt・hparams.yaml を sofa_model\ に置いてください" -ForegroundColor Yellow
    }
}

# YourMT3 の重み（約536MB）。mt3-infer の自動DLは git clone（git-lfs）前提でユーザー環境に
# git が無いと失敗するため、ここで明示取得する（配布元: HF Space mimbres/YourMT3。
# resolve URL の直DLが可能なことと手元実体とのサイズ一致 561,544,628 bytes を確認済み・2026-08-09）
if ($needMt3Ckpt) {
    Step "YourMT3 チェックポイント（伴奏採譜・約536MB）" {
        New-Item -ItemType Directory -Force $mt3CkptDir | Out-Null
        Invoke-WebRequest ("https://huggingface.co/spaces/mimbres/YourMT3/resolve/main/amt/logs/2024/" +
            "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt") `
            -OutFile (Join-Path $mt3CkptDir "last.ckpt")
        if (-not (Test-Path (Join-Path $mt3CkptDir "last.ckpt"))) { throw "YourMT3 ckpt の取得に失敗" }
    }
}

# ── 5. FluidSynth + GM SoundFont（伴奏の音源）──
if ($needFluid) {
    Step "FluidSynth（GitHub releases）" {
        $zip = Join-Path $env:TEMP "fluidsynth.zip"
        Invoke-WebRequest "https://github.com/FluidSynth/fluidsynth/releases/download/v2.4.5/fluidsynth-2.4.5-win10-x64.zip" -OutFile $zip
        Expand-Archive $zip (Join-Path $TOOLS "fluidsynth") -Force
        Remove-Item $zip -ErrorAction SilentlyContinue
        if (-not (Test-Path (Join-Path $TOOLS "fluidsynth\bin\fluidsynth.exe"))) {   # zip直下構成の差異を吸収
            $inner = Get-ChildItem (Join-Path $TOOLS "fluidsynth") -Directory | Select-Object -First 1
            if ($inner) { Get-ChildItem $inner.FullName | Move-Item -Destination (Join-Path $TOOLS "fluidsynth") -Force }
        }
    }
}
if ($needSf2) {
    Step "GeneralUser GS SoundFont（作成音源の配布制限なし）" {
        Invoke-WebRequest "https://github.com/mrbumpy409/GeneralUser-GS/raw/main/GeneralUser-GS.sf2" `
            -OutFile (Join-Path $TOOLS "GeneralUser-GS.sf2")
    }
}

# ── 5b. ffmpeg（whisperx が subprocess で呼ぶ必須CLI）──
# システムに要求せず static build を tools_bin\ に隔離配置し、song_runner が実行時に PATH 注入する
if ($needFfmpeg) {
    Step "ffmpeg（gyan.dev release-essentials・約110MB）" {
        $zip = Join-Path $env:TEMP "ffmpeg_essentials.zip"
        Invoke-WebRequest "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile $zip
        Expand-Archive $zip (Join-Path $TOOLS "ffmpeg") -Force
        Remove-Item $zip -ErrorAction SilentlyContinue
        if (-not (Test-Path (Join-Path $TOOLS "ffmpeg\bin\ffmpeg.exe"))) {   # zip直下の ffmpeg-<ver>-essentials_build\ を持ち上げる
            $inner = Get-ChildItem (Join-Path $TOOLS "ffmpeg") -Directory | Select-Object -First 1
            if ($inner) { Get-ChildItem $inner.FullName | Move-Item -Destination (Join-Path $TOOLS "ffmpeg") -Force }
        }
        if (-not (Test-Path (Join-Path $TOOLS "ffmpeg\bin\ffmpeg.exe"))) { throw "ffmpeg の配置に失敗" }
    }
}

# ── 5c. 学習済みモデルのプリフェッチ（whisper large-v3 / whisperx整列 / demucs・合計約5GB）──
# 初回の曲作成中の暗黙DLを防ぐ（規約: 重みはセットアップで明示DL。実行中DLは watchdog に殺されうる）。
# キャッシュ先は上で設定した HF_HOME / TORCH_HOME（miku\runtime\models\）。取得済み分は数秒で抜ける
if ($needPrefetch) {   # カバー向け（whisperx整列2種 / demucs）＋ whisper large-v3
    if (Test-Path $svcPython) {
        Step "音声モデルのプリフェッチ（whisper/整列/分離・約5GB）" {
            & $svcPython (Join-Path $SVC "prefetch_models.py")
            if ($LASTEXITCODE -ne 0) { throw "プリフェッチに失敗（selfcheck 後に再実行で再試行できます）" }
        }
    } else {
        Write-Host "== プリフェッチは統合venvが無いためスキップしました（venv構築後に再実行してください）" -ForegroundColor Yellow
    }
} elseif (-not $SkipModels) {
    # ★**GPU が無くても、文字起こしのモデルだけは取る**（2026-09-23・約3.1GB）。
    #   以前は「文字起こしはカバー専用＝カバーは GPU 必須」だったので軽い導入から外していたが、
    #   **鼻歌の肉声から歌詞を採る**という **CPU だけで成立する使い道**ができた。前提が変わった。
    #   実体は ctranslate2 で **torch は要らない**（torch 無しで動くことを実測で確認済み）。
    #   ★**ここで取る**（本人判断）——「**初回時になぜか時間がかかるのは不自然**」。
    if (Test-Path $svcPython) {
        Step "文字起こしモデルのプリフェッチ（whisper large-v3・約3.1GB）" {
            & $svcPython (Join-Path $SVC "prefetch_models.py") --whisper-only
            if ($LASTEXITCODE -ne 0) { throw "プリフェッチに失敗（selfcheck 後に再実行で再試行できます）" }
        }
    } else {
        Write-Host "== プリフェッチは venv が無いためスキップしました（venv構築後に再実行してください）" -ForegroundColor Yellow
    }
}

# ── 6. 受け入れ確認 ──
Write-Host ""
Write-Host "== selfcheck（設置確認）" -ForegroundColor Cyan
if (-not (Test-Path $svcPython)) {
    # venv 構築自体が失敗している。ここで exe を呼ぶと直前の $LASTEXITCODE が残って誤判定になる
    Write-Host "統合venv が作られていません（上のログでインストール失敗を確認してください）" -ForegroundColor Red
    $ok = $false
} else {
    & $svcPython (Join-Path $SVC "song_runner.py") selfcheck
    $ok = ($LASTEXITCODE -eq 0)
}
Write-Host ""
if ($ok) {
    Write-Host "作曲機能の準備が完了しました。settings.json の song_service.enabled を true にすると使えます。" -ForegroundColor Green
} else {
    Write-Host "不足している項目があります（上の ✗ 印）。このスクリプトを再実行すると不足分だけ取得します。" -ForegroundColor Yellow
}
Write-Host "実歌声にするには VOCALOID6＋初音ミク V6（別売・有償）が必要です。無い場合は仮歌ミックスまで作れます。"
exit ($(if ($ok) { 0 } else { 1 }))
