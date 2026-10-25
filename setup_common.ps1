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

# セットアップの共通処理ライブラリ（対話を一切含まない）
#
# 入口（setup_unattended.ps1・GUIインストーラから呼ばれる。コマンドから直接呼んでもよい）の
# 両方がこのファイルを dot-source して同じ関数を呼ぶ。処理を1箇所に集約し、
# 「片方だけ直して壊れる」事故を防ぐのが目的（2026-07-26）。
#
# 規約: 関数内で Read-Host を使わない／致命的失敗は $false を返す（exit しない）。
#       BOM付き UTF-8・PowerShell 5.1 互換構文で書くこと。

# uv のバージョン（クリーンWin11 VMで通し検証済み。更新時は再検証すること）
$script:MIKU_UV_VERSION = "0.11.32"

# ==========================================
#   入れ直しで「変わったところだけ」やるための印（2026-10-25）
# ==========================================
# 各段階が成功したら「何を元に作ったか」の照合値を runtime\stamps\<名前>.txt に残し、次の入れ直しで
# 同じなら飛ばす。本人「ハッシュ値が一致しないところだけ更新する動作にできないか」——0.2.1→0.2.2 の
# 入れ直しが約6分かかり、ほとんどが変わっていない段階の確かめ直しだった。
# 印は成功したときだけ書く（途中で止めたら次は必ずやり直す）。runtime はアンインストールで消える。

# ファイル群の中身から1つの照合値を作る（無いファイルは名前だけ混ぜる＝増減も差として拾う）
function Get-MikuInputHash {
    param([Parameter(Mandatory = $true)][string[]]$Files, [string]$Extra = "")
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append($Extra)
    foreach ($f in ($Files | Sort-Object)) {
        [void]$sb.Append("|" + [IO.Path]::GetFileName($f) + "=")
        if (Test-Path $f -PathType Leaf) { [void]$sb.Append((Get-FileHash $f -Algorithm SHA256).Hash) }
    }
    $bytes = [Text.Encoding]::UTF8.GetBytes($sb.ToString())
    return ([BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($bytes))).Replace("-", "")
}

function Test-MikuStamp {
    param([Parameter(Mandatory = $true)][string]$MikuDir, [Parameter(Mandatory = $true)][string]$Name,
          [Parameter(Mandatory = $true)][string]$Hash)
    $p = Join-Path $MikuDir "runtime\stamps\$Name.txt"
    return (Test-Path $p) -and ((Get-Content $p -Raw -ErrorAction SilentlyContinue).Trim() -eq $Hash)
}

function Set-MikuStamp {
    param([Parameter(Mandatory = $true)][string]$MikuDir, [Parameter(Mandatory = $true)][string]$Name,
          [Parameter(Mandatory = $true)][string]$Hash)
    $dir = Join-Path $MikuDir "runtime\stamps"
    New-Item -ItemType Directory -Force $dir | Out-Null
    Set-Content -Path (Join-Path $dir "$Name.txt") -Value $Hash -NoNewline -Encoding ASCII
}

# ==========================================
#   設定ファイルの準備
# ==========================================

# system_start.ps1 / settings.json / KV サンプルを用意する。
# 戻り値: @{ Ok = bool; StartPs1 = path; SettingsJson = path }
function Initialize-MikuConfigFiles {
    param([Parameter(Mandatory = $true)][string]$MikuDir)

    $srcDir = Join-Path $MikuDir "src"
    $result = @{
        Ok           = $false
        StartPs1     = Join-Path $MikuDir "system_start.ps1"
        SettingsJson = Join-Path $srcDir "settings.json"
    }

    $samplePs1 = Join-Path $MikuDir "system_start_sample.ps1"
    if (-not (Test-Path $samplePs1)) {
        Write-Host "NG エラー: system_start_sample.ps1 が見つかりません"
        return $result
    }
    if (-not (Test-Path $result.StartPs1)) {
        Copy-Item $samplePs1 $result.StartPs1
        Write-Host "OK system_start.ps1 を作成しました"
    } else {
        Write-Host "OK system_start.ps1 は既に存在します: $($result.StartPs1)"
    }

    $settingsSample = Join-Path $srcDir "settings_sample.json"
    if (-not (Test-Path $settingsSample)) {
        Write-Host "NG エラー: settings_sample.json が見つかりません"
        return $result
    }
    if (-not (Test-Path $result.SettingsJson)) {
        Copy-Item $settingsSample $result.SettingsJson
        Write-Host "OK settings.json を作成しました"
    } else {
        Write-Host "OK settings.json は既に存在します: $($result.SettingsJson)"
    }

    # KV 初期データ（長期記憶は 2026-06-01 に VDB collection へ移行済み）
    $kvDir = Join-Path $MikuDir "data\KV"
    @(
        @{ Sample = "this_conversation_sample.txt";                 Dest = "this_conversation.txt" },
        @{ Sample = "registration_waiting_conversation_sample.txt"; Dest = "registration_waiting_conversation.txt" }
    ) | ForEach-Object {
        $src  = Join-Path $kvDir $_.Sample
        $dest = Join-Path $kvDir $_.Dest
        if (Test-Path $src) {
            if (-not (Test-Path $dest)) {
                Copy-Item $src $dest
                Write-Host "OK $($_.Dest) を作成しました"
            } else {
                Write-Host "OK $($_.Dest) は既に存在します"
            }
        } else {
            Write-Host "NG エラー: $($_.Sample) が見つかりません"
        }
    }

    $result.Ok = $true
    return $result
}

# settings.json の現在の API キーを取得する（未設定・プレースホルダなら $null）
function Get-MikuApiKey {
    param(
        [Parameter(Mandatory = $true)][string]$SettingsJson,
        [Parameter(Mandatory = $true)][ValidateSet("google", "fish_audio")][string]$Name
    )
    try {
        $content = Get-Content $SettingsJson -Raw -Encoding UTF8 | ConvertFrom-Json
        $value = $content.api_keys.$Name
        if ($value -and ($value -notlike "put your*")) { return $value }
    } catch {}
    return $null
}

# API キーを settings.json に書き込む（空文字・$null のキーは触らない）。
function Set-MikuApiKeys {
    param(
        [Parameter(Mandatory = $true)][string]$SettingsJson,
        [string]$GoogleApiKey,
        [string]$FishApiKey
    )
    if (-not $GoogleApiKey -and -not $FishApiKey) { return $true }
    try {
        $content = Get-Content $SettingsJson -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        Write-Host "警告: settings.json の読み込みに失敗しました: $_"
        return $false
    }
    if ($GoogleApiKey) {
        $content.api_keys.google = $GoogleApiKey
        Write-Host "OK Google AI Studio (Gemini) APIキーを設定しました"
    }
    if ($FishApiKey) {
        $content.api_keys.fish_audio = $FishApiKey
        Write-Host "OK Fish Audio APIキーを設定しました"
        Write-Host "  （声は既定でミクに設定済み。変更はスタート画面の「設定」タブから）"
    }
    [System.IO.File]::WriteAllText($SettingsJson, ($content | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))
    return $true
}

# settings.json の system セクションを書く。
#
# **なぜ必要か**: コントロールパネルは外部アクセスの有無を
# settings.json の system セクションから読む。一方セットアップは長らく旧ランチャー
# （system_start.ps1）の変数にしか書いておらず、**書く場所と読む場所が食い違っていた**
# （セットアップで外部アクセスにチェックを入れてもパネルでは無効のまま・2026-08-02 修正）。
# 旧ランチャー向けの Set-MikuStartVars と対で呼ぶこと。
function Set-MikuSystemSettings {
    param(
        [Parameter(Mandatory = $true)][string]$SettingsJson,
        [bool]$TunnelEnabled = $false
    )
    try {
        $content = Get-Content $SettingsJson -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        Write-Host "警告: settings.json の読み込みに失敗しました: $_"
        return $false
    }
    if (-not $content.PSObject.Properties['system']) {
        $content | Add-Member -NotePropertyName "system" -NotePropertyValue ([PSCustomObject]@{})
    }
    $values = @{ tunnel_enabled = $TunnelEnabled }

    foreach ($key in $values.Keys) {
        if ($content.system.PSObject.Properties[$key]) {
            $content.system.$key = $values[$key]
        } else {
            $content.system | Add-Member -NotePropertyName $key -NotePropertyValue $values[$key]
        }
    }
    [System.IO.File]::WriteAllText($SettingsJson, ($content | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))
    Write-Host "OK settings.json（system）を更新しました"
    return $true
}

# system_start.ps1 のパス変数・トンネル設定を書き換える
function Set-MikuStartVars {
    param(
        [Parameter(Mandatory = $true)][string]$StartPs1,
        [Parameter(Mandatory = $true)][string]$MikuPath,
        [bool]$CloudflareEnabled = $false
    )
    $content = Get-Content $StartPs1 -Raw -Encoding UTF8
    $content = $content -replace '(?m)^\$MIKU_PATH\s*=\s*".*"', "`$MIKU_PATH     = `"$MikuPath`""
    $flag = '$false'
    if ($CloudflareEnabled) { $flag = '$true' }
    $content = $content -replace '(?m)^\$CLOUDFLARE_ENABLED\s*=.*', "`$CLOUDFLARE_ENABLED = $flag"
    # Windows PowerShell 5.1 で起動しても日本語コメントを誤読しないよう BOM付き UTF-8 で保存する
    [System.IO.File]::WriteAllText($StartPs1, $content, [System.Text.UTF8Encoding]::new($true))
    Write-Host "OK system_start.ps1 を更新しました"
    return $true
}

# ==========================================
#   外部コンポーネントの取得
# ==========================================

# cloudflared（外部アクセス用トンネル）をポータブル取得する。戻り値: 使える状態なら $true
function Install-Cloudflared {
    param([Parameter(Mandatory = $true)][string]$MikuDir)
    $cfDir = Join-Path $MikuDir "runtime\cloudflared"
    $cfExe = Join-Path $cfDir "cloudflared.exe"
    if (Test-Path $cfExe) { return $true }
    Write-Host "cloudflared をダウンロードしています（単体exe・約60MB）..."
    try {
        New-Item -ItemType Directory -Force $cfDir | Out-Null
        Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile $cfExe
    } catch {
        Write-Host "警告: cloudflared のダウンロードに失敗しました: $_"
        return $false
    }
    return (Test-Path $cfExe)
}

# Smart Turn v3 終話判定モデル（ONNX・約8.3MB・BSD-2）を Hugging Face から取得する。
# 使用箇所: src/stt_smart_turn.py（stt.smart_turn.enabled 時の意味的終話判定）
# モデルは data/（gitignore）配下のためコミットでは配布されず、ここが唯一の取得経路。
function Install-SmartTurnModel {
    param([Parameter(Mandatory = $true)][string]$MikuDir)
    $modelDir = Join-Path $MikuDir "data\models\smart_turn"
    $modelFile = Join-Path $modelDir "smart-turn-v3.2-cpu.onnx"
    if (Test-Path $modelFile) { return $true }
    Write-Host "終話判定モデル（Smart Turn v3・約8.3MB）をダウンロードしています..."
    try {
        New-Item -ItemType Directory -Force $modelDir | Out-Null
        Invoke-WebRequest -Uri "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx" -OutFile $modelFile
    } catch {
        Write-Host "警告: Smart Turn モデルのダウンロードに失敗しました: $_"
        Write-Host "  終話判定なしでも動作します（settings.json の stt.smart_turn.enabled を false にしてください）"
        return $false
    }
    return (Test-Path $modelFile)
}

# ブラウザ側・アイコン生成で使う第三者のライブラリとモデル。**配布物には同梱せず**、ここで
# 公式の配布元（npm レジストリ／Google の配布元）から版を固定して取得し、SHA256 で照合して置く
# （2026-10-08 本人決定。同梱していた時期の中身と同じ版・同じ SHA256 であることを確認済み）。
# 置き場所は同梱していたときと同じ（読む側のコードは変えていない）。どれも gitignore。
# 版を上げるときは Url と Sha256 を一緒に書き換える（照合が通らないものは置かない）。
#   Label   … 表示名  / Feature … 取れなかったときに使えなくなる機能（ほかは動く）
#   Url     … .tgz なら npm の公式 tarball（中の From を取り出す）。それ以外は単体ファイル
$script:MIKU_WEB_VENDOR = @(
    @{ Label = "PixiJS 6.5.10（MIT）"; Feature = "Live2D モデルからアイコンを作る（同梱のアイコンのまま）"
       Url = "https://registry.npmjs.org/pixi.js/-/pixi.js-6.5.10.tgz"
       Files = @(
           @{ From = "package/dist/browser/pixi.min.js"; To = "src\live2d\vendor\pixi.min.js"
              Sha256 = "403f2f2ee8145fa17f60c5c89403056efe2680e5096ec2762036486914ed19c5" },
           @{ From = "package/LICENSE"; To = "src\live2d\vendor\LICENSE.pixi.js.txt"
              Sha256 = "536a16e090f0b6c1454a7c0b47ee8414a39a2f1724a154f10ca1edc750a2dcfd" }) },
    @{ Label = "pixi-live2d-display 0.4.0（MIT・Live2D Cubism Framework を含む）"; Feature = "Live2D モデルからアイコンを作る（同梱のアイコンのまま）"
       Url = "https://registry.npmjs.org/pixi-live2d-display/-/pixi-live2d-display-0.4.0.tgz"
       Files = @(
           @{ From = "package/dist/cubism4.min.js"; To = "src\live2d\vendor\cubism4.min.js"
              Sha256 = "af1267e6d52759b245766c578d905bfa025b532d5c3cc727c370957c4409e21b" },
           @{ From = "package/LICENSE"; To = "src\live2d\vendor\LICENSE.pixi-live2d-display.txt"
              Sha256 = "8ccace668a041e78ed525d091b0b5daf8b623ff447f912616de535415e4f302f" }) },
    @{ Label = "MediaPipe Tasks Vision 1.0.1（Apache-2.0）"; Feature = "カメラでマスターの顔の方を向く（視線だけ動かない）"
       Url = "https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-1.0.1.tgz"
       Files = @(
           @{ From = "package/vision_bundle.mjs"; To = "src\templates\js\vendor\mediapipe\vision_bundle.mjs"
              Sha256 = "d885630c297c0b20b1fe86096cb06291c4c8080876f27852e724f24ac603713f" },
           @{ From = "package/wasm/vision_wasm_internal.js"; To = "src\templates\js\vendor\mediapipe\wasm\vision_wasm_internal.js"
              Sha256 = "e170ee67dd4e16c1a6fcd8840a206687e5a59b22c20e4a902bc445b095454d73" },
           @{ From = "package/wasm/vision_wasm_internal.wasm"; To = "src\templates\js\vendor\mediapipe\wasm\vision_wasm_internal.wasm"
              Sha256 = "8da277a733926eacd0474b8704b36742d6ec3231c57a860c5b889dff8f1df886" }) },
    @{ Label = "MediaPipe 顔検出モデル face_landmarker（Apache-2.0）"; Feature = "カメラでマスターの顔の方を向く（視線だけ動かない）"
       Url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
       Files = @(
           @{ To = "src\templates\js\vendor\mediapipe\face_landmarker.task"
              Sha256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff" }) },
    @{ Label = "Mediabunny 1.50.8（MPL-2.0）"; Feature = "大きい動画を送る前に縮める（元の動画のまま送る）"
       Url = "https://registry.npmjs.org/mediabunny/-/mediabunny-1.50.8.tgz"
       Files = @(
           @{ From = "package/dist/bundles/mediabunny.mjs"; To = "src\templates\js\vendor\mediabunny\mediabunny.mjs"
              Sha256 = "a7ae2036be704b7a3f03cbf0811d08501f59f014f2deaba959e4f0f0faba8bed" },
           @{ From = "package/LICENSE"; To = "src\templates\js\vendor\mediabunny\LICENSE"
              Sha256 = "3f3d9e0024b1921b067d6f7f88deb4a60cbe7a78e76c64e3f1d7fc3b779b9d04" }) }
)

function Test-MikuFileSha256 {
    param([string]$Path, [string]$Sha256)
    if (-not (Test-Path $Path)) { return $false }
    return ((Get-FileHash $Path -Algorithm SHA256).Hash -ieq $Sha256)
}

# 上の一覧を取得・照合・配置する。置き済みで SHA256 が合うものは取り直さない（冪等）。
# 取れなかったものは警告を出してその機能だけ使えない状態で続ける（導入は止めない）。
# 戻り値: 全部そろえば $true
# 開発者の手元で単独に呼ぶとき（リポジトリ直下で）:
#   powershell -NoProfile -ExecutionPolicy Bypass -Command ". .\miku\setup_common.ps1; Install-MikuWebVendor -MikuDir .\miku"
function Install-MikuWebVendor {
    param([Parameter(Mandatory = $true)][string]$MikuDir)
    # ★Git for Windows の tar が PATH で先に当たると "D:" をホスト名と読んで失敗する。Windows 標準の tar を名指しする
    $tar = Join-Path $env:SystemRoot "System32\tar.exe"
    $allOk = $true
    foreach ($pkg in $script:MIKU_WEB_VENDOR) {
        $missing = @($pkg.Files | Where-Object { -not (Test-MikuFileSha256 (Join-Path $MikuDir $_.To) $_.Sha256) })
        if ($missing.Count -eq 0) { continue }
        Write-Host "$($pkg.Label) を取得しています..."
        $tmp = Join-Path $env:TEMP ("miku_vendor_" + [guid]::NewGuid().ToString("N"))
        $ok = $true
        try {
            New-Item -ItemType Directory -Force $tmp | Out-Null
            $isTgz = $pkg.Url.EndsWith(".tgz")
            $dl = Join-Path $tmp $(if ($isTgz) { "pkg.tgz" } else { "file.bin" })
            Invoke-WebRequest -Uri $pkg.Url -OutFile $dl -UseBasicParsing
            if ($isTgz) {
                if (-not (Test-Path $tar)) { throw "tar.exe が見つかりません: $tar" }
                & $tar -xzf $dl -C $tmp
                if ($LASTEXITCODE -ne 0) { throw "展開に失敗しました（tar 終了コード $LASTEXITCODE）" }
            }
            foreach ($f in $missing) {
                $src = $(if ($isTgz) { Join-Path $tmp ($f.From -replace "/", "\") } else { $dl })
                if (-not (Test-MikuFileSha256 $src $f.Sha256)) {
                    throw "SHA256 が一致しません: $($f.To)（配布元の中身が想定と違うため置きません）"
                }
                $dest = Join-Path $MikuDir $f.To
                New-Item -ItemType Directory -Force (Split-Path -Parent $dest) | Out-Null
                Copy-Item $src $dest -Force
            }
        } catch {
            $ok = $false
            Write-Host "警告: $($pkg.Label) を用意できませんでした: $_"
            Write-Host "  使えなくなる機能: $($pkg.Feature)。ほかの機能はそのまま使えます"
            Write-Host "  あとから取り直すには、セットアップをもう一度実行してください"
        } finally {
            Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
        if (-not $ok) { $allOk = $false }
    }
    return $allOk
}

# VC++ 再頒布可能パッケージ（PyTorch の DLL が要求・クリーンな Windows には無い）を導入する
function Install-VcRedist {
    $vcRuntime1 = Join-Path $env:SystemRoot "System32\vcruntime140_1.dll"
    $vcRuntime2 = Join-Path $env:SystemRoot "System32\msvcp140.dll"
    if ((Test-Path $vcRuntime1) -and (Test-Path $vcRuntime2)) { return $true }

    Write-Host ""
    Write-Host "Microsoft Visual C++ 再頒布可能パッケージをインストールします（PyTorch の動作に必要・約25MB）"
    Write-Host "許可ダイアログ（UAC）が表示されたら「はい」を押してください"
    $vcExe = Join-Path $env:TEMP "vc_redist.x64.exe"
    try {
        Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile $vcExe
        Start-Process $vcExe -ArgumentList "/install /quiet /norestart" -Wait
        Remove-Item $vcExe -ErrorAction SilentlyContinue
    } catch {
        Write-Host "警告: VC++ 再頒布のインストールに失敗しました: $_"
        Write-Host "  手動でインストールしてください: https://aka.ms/vs/17/release/vc_redist.x64.exe"
        return $false
    }
    if ((Test-Path $vcRuntime1) -and (Test-Path $vcRuntime2)) {
        Write-Host "OK VC++ 再頒布可能パッケージをインストールしました"
        return $true
    }
    Write-Host "警告: VC++ 再頒布のインストールを確認できませんでした"
    Write-Host "  https://aka.ms/vs/17/release/vc_redist.x64.exe"
    return $false
}

# ==========================================
#   Python 環境
# ==========================================

# uv 取得 → venv 作成 → 有効化 → requirements2 → requirements の順に構築する。
# 戻り値: @{ Ok = bool; PythonExe = path; UvExe = path; UseUv = bool }
function Install-MikuPythonEnv {
    param([Parameter(Mandatory = $true)][string]$MikuDir)

    $srcDir   = Join-Path $MikuDir "src"
    $venvPath = Join-Path $MikuDir "venv"
    $uvDir    = Join-Path $MikuDir "runtime\uv"
    $uvExe    = Join-Path $uvDir "uv.exe"
    $result   = @{ Ok = $false; PythonExe = $null; UvExe = $uvExe; UseUv = $false }

    # 隔離方針: 自動取得物はすべて miku\runtime\ 配下に置き、
    # 「アンインストール＝フォルダ削除だけで完結」を保証する
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $MikuDir "runtime\python"
    $env:UV_CACHE_DIR          = Join-Path $MikuDir "runtime\uv_cache"

    # --- uv（ポータブル）の取得。管理者権限・winget・事前の Python 不要 ---
    if (-not (Test-Path $uvExe)) {
        Write-Host "uv（Python環境管理ツール）をダウンロードしています..."
        try {
            New-Item -ItemType Directory -Force $uvDir | Out-Null
            $uvZip = Join-Path $uvDir "uv.zip"
            Invoke-WebRequest -Uri "https://github.com/astral-sh/uv/releases/download/$script:MIKU_UV_VERSION/uv-x86_64-pc-windows-msvc.zip" -OutFile $uvZip
            Expand-Archive $uvZip $uvDir -Force
            Remove-Item $uvZip -ErrorAction SilentlyContinue
        } catch {
            Write-Host "警告: uv のダウンロードに失敗しました（従来方式で続行します）: $_"
        }
    }
    $useUv = Test-Path $uvExe
    if ($useUv) { Write-Host "OK uv: $(& $uvExe --version)" }

    Write-Host ""
    Write-Host "仮想環境を確認・作成しています..."
    # 環境を今回作ったなら、前回の印があっても部品は入れる（下の python_deps の判断に使う）
    $venvExisted = Test-Path $venvPath
    if (Test-Path $venvPath) {
        Write-Host "仮想環境は既に存在します: $venvPath"
    } elseif ($useUv) {
        # **venv は uv ではなく「本体の python -m venv」で作る**。
        # uv が作る venv の pythonw.exe は本物ではなく中継用の小さな exe で、
        # それが**コンソールアプリ**として作られているため、スタート画面を開くと
        # 黒いコンソールが付いてきて、閉じるとスタート画面ごと落ちる。
        # 2026-08-03 実測: uv venv 版 45,568B/コンソール、-m venv 版 237,056B/GUI。
        # uv 取得の Python は本体に python313.dll 等を同梱しているので、
        # -m venv で作った venv からも解決できる（実際に pythonw.exe が動くことを確認済み）。
        # uv は引き続き「Python の取得」と「パッケージの導入」に使う。
        #
        # 一過性のネットワーク/DNS失敗（Hyper-V NAT で実例）に備えて最大3回試行し、
        # エラー出力（stderr）を 2>&1 でログに残す
        # （残さないと「なぜ失敗したか」が追えない。2026-07-29 のVM検証で実害）
        $pyInstallDir = $env:UV_PYTHON_INSTALL_DIR
        for ($attempt = 1; $attempt -le 3; $attempt++) {
            Write-Host "  Python 3.13 を確認・取得しています..."
            & $uvExe python install 3.13 2>&1 | ForEach-Object { Write-Host "  $_" }

            # Windows 11 の新しいビルド（25H2 系で実害・2026-07-29 VM検証）では、uv が作る
            # バージョンリンク（ジャンクション）がセキュリティ機能にブロックされ
            # os error 448「信頼されていないマウントポイント」で失敗する。壊れたリンクが
            # 残ると以後の走査まで失敗するため、実体のフォルダだけを選んで直接使う。
            # ※ 実体の名前は `cpython-3.13-windows-x86_64-none` のように
            #   パッチ番号を含まないこともある（2026-08-03 実測）。`3.13.*` で絞ると外れる
            $realPy = Get-ChildItem $pyInstallDir -Directory -Filter "cpython-3.13*" -ErrorAction SilentlyContinue |
                Where-Object {
                    -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -and
                    (Test-Path (Join-Path $_.FullName "python.exe"))
                } | Select-Object -First 1

            if ($realPy) {
                Write-Host "  Python 本体: $($realPy.Name)"
                & (Join-Path $realPy.FullName "python.exe") -m venv $venvPath 2>&1 |
                    ForEach-Object { Write-Host "  $_" }
                if ($LASTEXITCODE -eq 0) { break }
            } else {
                Write-Host "  Python 本体が見つかりません（取得に失敗した可能性）"
            }
            if (Test-Path $venvPath) { Remove-Item $venvPath -Recurse -Force }

            if ($attempt -lt 3) {
                Write-Host "仮想環境の作成に失敗しました。5秒後にリトライします（$attempt/2回目）..."
                Start-Sleep -Seconds 5
            }
        }
        if (-not (Test-Path (Join-Path $venvPath "Scripts\python.exe"))) {
            Write-Host "警告: 仮想環境の作成に失敗しました。従来方式を試します"
            $useUv = $false
        } else {
            Write-Host "仮想環境を作成しました（Python 3.13 は uv が取得・venv は python -m venv）"
        }
    }

    if (-not (Test-Path $venvPath)) {
        # --- 従来方式（フォールバック）: インストール済みの Python 3.13 を探す ---
        $pythonCmd = $null
        foreach ($cmd in @("python", "python3.13")) {
            try {
                $ver = & $cmd --version 2>&1
                if ($ver -match "Python 3\.13") {
                    $pythonCmd = $cmd
                    Write-Host "Python 3.13 が見つかりました: $ver"
                    break
                } elseif ($cmd -eq "python" -and $ver -match "Python 3\.") {
                    Write-Host "警告: Python 3.13 以外のバージョンで続行します: $ver"
                    $pythonCmd = $cmd
                    break
                }
            } catch {}
        }
        if (-not $pythonCmd) {
            Write-Host "エラー: uv が使えず、Python 3.13 も見つかりません"
            Write-Host "  ネット接続を確認して再実行するか、Python 3.13 を手動インストールしてください"
            Write-Host "  https://www.python.org/downloads/release/python-3133/"
            return $result
        }
        & $pythonCmd -m venv $venvPath
        if ($LASTEXITCODE -ne 0) {
            Write-Host "エラー: 仮想環境の作成に失敗しました"
            return $result
        }
        Write-Host "仮想環境を作成しました"
    }

    # --- 有効化と検証（グローバル環境への誤インストールを防ぐ） ---
    & (Join-Path $venvPath "Scripts\Activate.ps1")
    if (-not $env:VIRTUAL_ENV) {
        Write-Host "エラー: 仮想環境の有効化に失敗しました"
        return $result
    }
    $pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $pythonExe) {
        Write-Host "エラー: Pythonパスの取得に失敗しました"
        return $result
    }
    if ($pythonExe -notmatch "venv") {
        Write-Host "エラー: Pythonがグローバル環境を使用しています: $pythonExe"
        return $result
    }
    Write-Host "OK 仮想環境のPythonを使用しています: $pythonExe"

    # --- 依存インストール ---
    # PyTorch（requirements2）→ 一般（requirements）の順。逆順だと facenet-pytorch / speechbrain が
    # PyPI 版 torch（数GB）を先に引き込み、直後に cu130 版で置き換える無駄が出る（実測）
    # ※ 2026-08-26 まで理由は sentence-transformers だったが、同パッケージは未使用のため削除した。
    #    torch を引き込むのは上記2つになったので、順序の規則自体は変わらず必要
    Set-Location $srcDir
    # 一覧（requirements）が前回入れたときと同じなら、確かめ直さない（入れ直しの短縮・2026-10-25）
    $reqHash = Get-MikuInputHash -Files @((Join-Path $srcDir "requirements2.txt"), (Join-Path $srcDir "requirements.txt")) -Extra $pythonExe
    if ($venvExisted -and (Test-MikuStamp -MikuDir $MikuDir -Name "python_deps" -Hash $reqHash)) {
        Write-Host "Python の部品は前回から変わっていないので、確認を飛ばします"
        $result.Ok = $true
        $result.PythonExe = $pythonExe
        $result.UseUv = $useUv
        return $result
    }
    foreach ($req in @(
        @{ Label = "PyTorchパッケージ（CUDA対応版）"; File = "requirements2.txt" },
        @{ Label = "一般的なパッケージ";              File = "requirements.txt" }
    )) {
        $reqPath = Join-Path $srcDir $req.File
        if (-not (Test-Path $reqPath)) {
            Write-Host "エラー: $($req.File) が見つかりません"
            return $result
        }
        Write-Host ""
        Write-Host "$($req.Label) をインストール中..."
        if ($useUv) {
            # 2>&1: uv は進捗・エラーを stderr に書くため、ログ（Transcript/インストーラ進捗）に残す
            & $uvExe pip install --python $pythonExe -r $reqPath 2>&1 | ForEach-Object { Write-Host "$_" }
        } else {
            pip install -r $reqPath
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Host "エラー: $($req.File) のインストールに失敗しました"
            return $result
        }
        Write-Host "$($req.Label) のインストールが完了しました"
    }
    Set-MikuStamp -MikuDir $MikuDir -Name "python_deps" -Hash $reqHash

    $result.Ok = $true
    $result.PythonExe = $pythonExe
    $result.UseUv = $useUv
    return $result
}

# 顔認識モデルの重み（約107MB）を明示ダウンロードする（初回起動時の暗黙DLを避ける）
function Install-MikuFaceWeights {
    param([Parameter(Mandatory = $true)][string]$MikuDir)
    Write-Host ""
    # 保存先はフォルダ内（runtime\models）に隔離。system_start.ps1 も同じ環境変数を設定する
    $env:TORCH_HOME = Join-Path $MikuDir "runtime\models\torch"
    # 取得済みなら確かめに読み込まない（PyTorch の読み込みだけで数十秒かかっていた・2026-10-25）
    if (Test-Path (Join-Path $env:TORCH_HOME "checkpoints\20180402-114759-vggface2.pt")) {
        Write-Host "顔認識モデルの重みは取得済みです"
        return $true
    }
    Write-Host "顔認識モデルの重みをダウンロードしています（約107MB・初回のみ）..."
    # 一過性のDNS/ネットワーク失敗（Hyper-V NAT等で実例あり）に備えて最大3回試行する
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        python -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2')"
        if ($LASTEXITCODE -eq 0) { return $true }
        if ($attempt -lt 3) {
            Write-Host "ダウンロードに失敗しました。5秒後にリトライします（$attempt/2回目）..."
            Start-Sleep -Seconds 5
        }
    }
    Write-Host "警告: 顔認識モデルの重みダウンロードに失敗しました（初回の顔認識時に再試行されます）"
    Write-Host "  ネットワーク／DNSを確認して再実行しても取得できます"
    return $false
}

# 話者照合モデルの重み（約80MB）を明示ダウンロードする（初回起動時の暗黙DLを避ける）
function Install-MikuVoiceWeights {
    param([Parameter(Mandatory = $true)][string]$MikuDir)
    Write-Host ""
    # 保存先はフォルダ内（runtime\models）に隔離。system_start.ps1 も同じ環境変数を設定する
    $env:HF_HOME = Join-Path $MikuDir "runtime\models\hf"
    $saveDir = Join-Path $MikuDir "runtime\models\speechbrain_ecapa"
    # 取得済みなら確かめに読み込まない（speechbrain の読み込みだけで数十秒かかっていた・2026-10-25）
    if ((Test-Path (Join-Path $saveDir "embedding_model.ckpt")) -and (Test-Path (Join-Path $saveDir "hyperparams.yaml"))) {
        Write-Host "話者照合モデルの重みは取得済みです"
        return $true
    }
    Write-Host "話者照合モデルの重みをダウンロードしています（約80MB・初回のみ）..."
    $py = "from speechbrain.inference.speaker import EncoderClassifier; " +
          "EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', " +
          "savedir=r'$saveDir', run_opts={'device':'cpu'})"
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        python -c $py
        if ($LASTEXITCODE -eq 0) { return $true }
        if ($attempt -lt 3) {
            Write-Host "ダウンロードに失敗しました。5秒後にリトライします（$attempt/2回目）..."
            Start-Sleep -Seconds 5
        }
    }
    Write-Host "警告: 話者照合モデルの重みダウンロードに失敗しました（初回の声の登録時に再試行されます）"
    Write-Host "  ネットワーク／DNSを確認して再実行しても取得できます"
    return $false
}

# ==========================================
#   作曲機能（任意・GPU あり 約15GB／GPU なし 約4GB）
# ==========================================

# NVIDIA GPU 名を返す（torch 未導入でも判定できるよう nvidia-smi で見る）。無ければ $null
function Get-NvidiaGpuName {
    try {
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            return (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
        }
    } catch {}
    return $null
}

# 作曲機能をセットアップし、成功したら settings.json で有効化する
# VOCALOID6 が入っているか。**作曲機能の導入条件**（2026-09-06 本人判断）。
#
# 作曲機能の値打ちは「**初音ミクが歌ってくれる**」ところにある。VOCALOID6 が無いと
# 曲は作れても**ミクの声で歌わせることが一生できない**（`play_song` は歌声 `vocal_v{N}.wav`
# しか鳴らさず、仮歌〔サイン波〕は対象外）。作った曲が全部聴けないまま溜まるので、
# **入れられるようにしない**——警告して選ばせるのではなく、条件を満たさなければ導入しない。
# ただし**無言で飛ばさない**。理由を必ず伝える。
#
# ★**パスを決め打ちにしない**（2026-09-06）。設置先は利用者が変えられるので、決め打ちだと
#   **別のドライブに入れた人を「持っていない」と誤判定**して導入させないことになる。
#   **ただし深追いもしない**（本人判断）。見るのは2つだけで、`v6_export.py` の `find_v6_exe()` と揃える:
#     ① `.vpr` の関連付け（実際の設置先が登録される。起動もこの経路なので確実）
#     ② 既定のパス（%ProgramFiles% 経由＝システムドライブが C: でなくても効く）
#   歌声ライブラリ（初音ミク V6）の有無は**見ない**——名前で照合することになり誤判定を生むため。
# ※ロジックの正は `miku/src/song_service/engine/v6_export.py` の `find_v6_exe()`。
#   変えるときは3つとも直す（ここ／同ファイル／`control_panel/supervisor.py`）
function Get-MikuVocaloid6Exe {
    $assoc = "HKLM:\SOFTWARE\Classes\VOCALOID6.vpr\shell\open\command"
    if (Test-Path $assoc) {
        $cmd = (Get-ItemProperty $assoc -EA SilentlyContinue).'(default)'
        if ($cmd -and $cmd -match '"([^"]+)"') {
            if (Test-Path $Matches[1]) { return $Matches[1] }
        }
    }
    foreach ($base in @($env:ProgramFiles, "C:\Program Files")) {
        if ($base) {
            $exe = Join-Path $base "VOCALOID6\Editor\VOCALOID6.exe"
            if (Test-Path $exe) { return $exe }
        }
    }
    return $null
}

function Test-MikuVocaloid6 {
    return ($null -ne (Get-MikuVocaloid6Exe))
}

function Install-MikuSongService {
    param(
        [Parameter(Mandatory = $true)][string]$MikuDir,
        [Parameter(Mandatory = $true)][string]$SettingsJson,
        # ★GPU が無い環境では**音源からのカバー用のものを除いて**入れる（2026-09-01）。
        #   カバー用の venv 2本と学習済みモデル（全部「音源しか無い曲のカバー」専用。
        #   サイズは setup_song_service.ps1 冒頭）を取りに行かない。オリジナル曲・MIDI のカバーは作れる。
        [switch]$OriginalOnly,
        # ★デバッグ用（2026-09-06）。V6 を入れられない環境で導入まで通したいときだけ。
        #   **飛ばすのは導入の可否だけ**で、実行時の書き出しは正直に失敗する
        [switch]$SkipVocaloidCheck
    )
    Write-Host ""
    Write-Host "========================================"
    Write-Host "  作曲機能のセットアップ"
    Write-Host "========================================"
    # ★入口でも守る（対話を通らない呼び出しがあるため）。**理由を言ってから断る**
    if ($SkipVocaloidCheck) {
        Write-Host "【デバッグ】VOCALOID6 の確認を飛ばして導入します（歌声の書き出しは失敗します）。" -ForegroundColor Magenta
    } elseif (-not (Test-MikuVocaloid6)) {
        Write-Host "VOCALOID6 が見つからないため、作曲機能は導入できません。" -ForegroundColor Yellow
        Write-Host "  この機能は『ミクが自分の声で歌う』ためのものです。VOCALOID6（＋初音ミク V6・別売）が"
        Write-Host "  無いと、曲は作れても歌声にできず、ミクに歌ってもらうことができません。"
        Write-Host "  VOCALOID6 を導入してから、もう一度この機能を追加してください。"
        return $false
    }
    # 材料（部品の一覧・セットアップの中身・取得するモデル）と構成が前回と同じで、環境もあれば、確かめ直さない
    # （入れ直しの短縮・2026-10-25。VOCALOID6 の起動確認も飛ぶので、本番の歌声の書き出しとぶつからない）
    $songDir = Join-Path $MikuDir "src\song_service"
    $songInputs = @(Get-ChildItem $songDir -Filter "requirements_*.txt" -File -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
    $songInputs += @((Join-Path $songDir "setup_song_service.ps1"), (Join-Path $songDir "prefetch_models.py"))
    $songHash = Get-MikuInputHash -Files $songInputs -Extra ("OriginalOnly=" + [bool]$OriginalOnly)
    if ((Test-Path (Join-Path $songDir "venv\Scripts\python.exe")) -and
        (Test-MikuStamp -MikuDir $MikuDir -Name "song_service" -Hash $songHash)) {
        Write-Host "作曲機能は前回から変わっていないので、確認を飛ばします"
        return $true
    }
    if ($OriginalOnly) {
        Write-Host "オリジナル曲と、楽譜（MIDI）が見つかる曲のカバーに要るものを取得します（約4GB・十数分）。"
        Write-Host "  うち約3GBは、歌声を聞き取るためのものです（鼻歌を歌詞ごと受け取れるようになります）。"
    } else {
        Write-Host "ここから 40〜90分かかります（約15GB・回線次第）。この間の操作は不要です。"
    }

    $songSetup = Join-Path $MikuDir "src\song_service\setup_song_service.ps1"
    if (-not (Test-Path $songSetup)) {
        Write-Host "[WARN] setup_song_service.ps1 が見つかりません: $songSetup"
        return $false
    }
    if ($OriginalOnly) {
        & powershell -ExecutionPolicy Bypass -File $songSetup -OriginalOnly
    } else {
        & powershell -ExecutionPolicy Bypass -File $songSetup
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "作曲機能のセットアップが完了しませんでした（settings.json では無効のままです）。"
        Write-Host "  不足分は setup_song_service.ps1 の再実行で取得できます。"
        return $false
    }
    # ★**導入の最後に一度だけ、実際に起動して確かめる**（2026-09-06 本人指定）。
    #   ミクは曲づくりの最後に VOCALOID6 を自分で開くので、**それがこの環境で通るか**を
    #   ここで見ておく。実体があってもライセンス認証で止まる等は、起動して初めて分かる。
    #   曲は作らない・5秒ほどで閉じる（`-CheckOnly`）。
    #   **失敗しても導入は止めない**——手で書き出す道は残っているので、伝えるだけにする。
    $checkScript = Join-Path $MikuDir "src\song_service\export_vocal.ps1"
    if (Test-Path $checkScript) {
        Write-Host ""
        Write-Host "動作確認のため VOCALOID6 を一度だけ起動します（数秒で閉じます）..."
        & powershell -ExecutionPolicy Bypass -File $checkScript -CheckOnly
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  OK: ミクが自分で歌声を書き出せます。" -ForegroundColor Green
        } else {
            Write-Host "  VOCALOID6 を起動できませんでした。" -ForegroundColor Yellow
            Write-Host "  作曲機能は使えますが、歌声の書き出しはご自身で行っていただくことになります"
            Write-Host "  （ミクが作った vpr を VOCALOID6 で開いてミックスダウンし、名前は既定のまま保存）。"
        }
    }
    try {
        # 準備できたので有効化（実体が無い間は settings が true でも実行時チェックで安全に無効扱いになる）
        $sc = Get-Content $SettingsJson -Raw -Encoding UTF8 | ConvertFrom-Json
        if (-not $sc.song_service) {
            $sc | Add-Member -NotePropertyName song_service -NotePropertyValue ([pscustomobject]@{ enabled = $true; max_runtime_minutes = 60 })
        } else {
            $sc.song_service.enabled = $true
        }
        [System.IO.File]::WriteAllText($SettingsJson, ($sc | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))
        Write-Host "OK settings.json の song_service.enabled を true にしました"
        Set-MikuStamp -MikuDir $MikuDir -Name "song_service" -Hash $songHash
        return $true
    } catch {
        Write-Host "警告: settings.json の更新に失敗しました（設定画面から作曲機能を ON にしてください）: $_"
        return $false
    }
}

# ==========================================
#   最終確認・仕上げ
# ==========================================

# インストール結果の自己診断を表示する
function Show-MikuFinalCheck {
    param(
        [Parameter(Mandatory = $true)][string]$StartPs1,
        [Parameter(Mandatory = $true)][string]$SettingsJson
    )
    Write-Host ""
    Write-Host "[設定ファイルの確認]"
    @($StartPs1, $SettingsJson) | ForEach-Object {
        if (Test-Path $_) { Write-Host "OK $_" } else { Write-Host "NG $_ が見つかりません" }
    }

    Write-Host ""
    Write-Host "[Python環境の確認]"
    python -c "import torch; print('OK PyTorch:', torch.__version__)" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host "NG PyTorchのインポートに失敗" }
    $cudaAvail = python -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ("$cudaAvail" -match "True") {
        Write-Host "OK CUDA(GPU) 利用可能"
    } else {
        Write-Host "注意 CUDA(GPU) が認識されていません（PyTorch は CUDA 版のまま CPU で動作します）"
        Write-Host "     GPU 非搭載・仮想マシンなら正常です。NVIDIA GPU 搭載機なら ドライバを確認してください"
    }
    foreach ($mod in @(
        @{ Import = "flask";           Label = "OK Flask" },
        @{ Import = "pydantic_ai";     Label = "OK pydantic_ai" },
        @{ Import = "chromadb";        Label = "OK chromadb" },
        @{ Import = "facenet_pytorch"; Label = "OK facenet_pytorch (顔認識)" },
        @{ Import = "speechbrain";     Label = "OK speechbrain (話者照合)" }
    )) {
        python -c "import $($mod.Import)" 2>$null
        if ($LASTEXITCODE -eq 0) { Write-Host $mod.Label } else { Write-Host "NG $($mod.Import)のインポートに失敗" }
    }

    Write-Host ""
    Write-Host "[APIキー設定の確認]"
    try {
        $sj = Get-Content $SettingsJson -Raw -Encoding UTF8
        if ($sj -match '"put your') {
            Write-Host "注意: settings.json に未設定のAPIキー（put your ...）が残っています"
            Write-Host "      使う機能のキーが設定済みか確認してください（キーの有効性はここでは検証されません）"
        } else {
            Write-Host "OK プレースホルダのままのAPIキーはありません"
        }
    } catch {}
}

# 起動ランチャー（Windows 標準の PowerShell 5.1・固定パス）。
# 裸の pwsh.exe を .lnk に入れると Store 版の実体パスに解決され、pwsh 更新で壊れる事故があったため
function Get-MikuLauncherExe {
    return (Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe")
}

# デスクトップショートカットを作成する
function New-MikuDesktopShortcut {
    param(
        [Parameter(Mandatory = $true)][string]$MikuDir,
        [Parameter(Mandatory = $true)][string]$StartPs1
    )
    $shortcutPath = Join-Path ([Environment]::GetFolderPath("Desktop")) "初音ミク_システム.lnk"
    # モデルから生成した版 ＞ 同梱の簡易マーク の優先順。
    # Live2D は必須なので通常は生成版が使われるが、この関数は生成より前にも呼ばれうる
    # （その場合は Live2D セットアップ側が後からショートカットのアイコンを差し替える）
    $iconPath = @(
        (Join-Path $MikuDir "runtime\miku_icon.ico"),
        (Join-Path $MikuDir "app_icon.ico")
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    # コントロールパネル（起動・停止のGUI）を指す。pythonw.exe を直接叩くのでシェルを経由しない
    $pythonw = Join-Path $MikuDir "venv\Scripts\pythonw.exe"
    try {
        $wsh = New-Object -ComObject WScript.Shell
        $sc  = $wsh.CreateShortcut($shortcutPath)
        if (Test-Path $pythonw) {
            $sc.TargetPath       = $pythonw
            # `-m control_panel` ではなくランチャーを指す。保護版ではパッケージが .pyd に
            # なり、.pyd に対して `python -m <パッケージ>` は使えないため（詳細は
            # docs/plan/distribution_protection.md）。素版でも同じ経路で動く
            $sc.Arguments        = "start_panel.py"
            $sc.WorkingDirectory = Join-Path $MikuDir "src"
        } else {
            # venv 未作成（セットアップ未完了）のときは従来の起動スクリプトを指しておく
            $sc.TargetPath       = Get-MikuLauncherExe
            $sc.Arguments        = "-ExecutionPolicy Bypass -File `"$StartPs1`""
            $sc.WorkingDirectory = $MikuDir
        }
        $sc.WindowStyle      = 1
        if (Test-Path $iconPath) { $sc.IconLocation = $iconPath }
        $sc.Save()
        if (Test-Path $shortcutPath) {
            Write-Host "ショートカットを作成しました: $shortcutPath"
            return $true
        }
    } catch {
        Write-Host "ショートカットの作成に失敗しました: $_"
    }
    return $false
}

# ミクのシステムを起動する
function Start-MikuSystem {
    param([Parameter(Mandatory = $true)][string]$StartPs1)
    Write-Host "初音ミクを起動しています..."
    Start-Process (Get-MikuLauncherExe) -ArgumentList "-ExecutionPolicy Bypass -File `"$StartPs1`""
}

# 完了メッセージ（オプション機能の案内つき）
function Show-MikuCompletionMessage {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "  セットアップ完了！"
    Write-Host "========================================"
    Write-Host ""
    Write-Host "[オプション機能の設定]"
    Write-Host "Web検索・メール送信・作曲などのオプション機能は、ミクのスタート画面の"
    Write-Host "「設定」タブから有効化できます（システム停止中でも変更できます）:"
    Write-Host "  - Web検索  : Tavily APIキー（https://tavily.com/ 無料枠あり）を入力して有効化"
    Write-Host "  - メール送信: 送信元Gmail・Gmailアプリパスワード・送信先を入力して有効化"
    Write-Host "  - そのほかの各種設定もここから変更できます（反映はシステム再起動時）"
    Write-Host ""
    Write-Host "[外出先から使うときは、パスワードの設定を]"
    Write-Host "外部アクセス（スマホなどからのアクセス）を使う場合は、スタート画面の"
    Write-Host "「設定」タブ →「ログインの保護」でパスワードを設定してください。"
    Write-Host "設定しないと、アクセス用のURLを知った人は誰でもミクと話せてしまいます"
    Write-Host "（会話の記録も読めます）。この PC からのアクセスには入力は要りません。"
    Write-Host ""
    Write-Host "次の手順:"
    Write-Host "1. start_miku.bat のダブルクリックでミクのスタート画面を開く"
    Write-Host "   （ショートカットを作成した場合はそちらをダブルクリック）"
    Write-Host "2. 「起動」ボタンを押す"
    Write-Host "3. ブラウザで http://localhost:5001 にアクセス"
    Write-Host "4. 初音ミクとの会話をお楽しみください！"
    Write-Host ""
}
