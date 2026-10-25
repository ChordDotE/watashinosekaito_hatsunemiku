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

# Live2D セットアップの共通処理ライブラリ（対話を一切含まない）
#
# 対話版（setup_live2d.ps1）と非対話版（miku\setup_unattended.ps1）の両方が
# このファイルを dot-source して同じ関数を呼ぶ。処理を1箇所に集約し、
# 「片方だけ直して壊れる」事故を防ぐのが目的（2026-07-26）。
#
# ※このファイルは BOM付き UTF-8・PowerShell 5.1 互換構文で書くこと
#   （dependencies_and_setup.md「セットアップスクリプト設計の原則」）

# Live2D 関連のパスをまとめて返す
function Get-Live2dPaths {
    param([Parameter(Mandatory = $true)][string]$Live2dDir)
    $sdk = Join-Path $Live2dDir "CubismWebSamples"
    $demo = Join-Path $sdk "Samples\TypeScript\Demo"
    $resources = Join-Path $sdk "Samples\Resources"
    return @{
        Live2dDir = $Live2dDir
        SdkDir    = $sdk
        DemoDir   = $demo
        DemoSrc   = Join-Path $demo "src"
        Resources = $resources
        MikuDirOld = Join-Path $resources "Miku"
        MikuDir    = Join-Path $resources "miku_sample_t04"
        WorkDir    = Join-Path $Live2dDir "_extract_tmp"
        CoreJs     = Join-Path $sdk "Core\live2dcubismcore.min.js"
    }
}

# Gateway の待ち受けポート（settings.json の services.gateway.port。読めなければ 5001）
# 以前は 5001 の直書きで、別フォルダに2つ目を入れてポートを変えると判定がずれた（2026-10-22）
function Get-MikuGatewayPort {
    $settingsPath = Join-Path (Split-Path $PSScriptRoot -Parent) "settings.json"
    try {
        $s = Get-Content -LiteralPath $settingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($s.services.gateway.port) { return [int]$s.services.gateway.port }
    } catch {}
    return 5001
}

# ミクのシステムが稼働中か（ビルドは配信中の dist/ を作り直すため、稼働中の実行は避けたい）
function Test-MikuGatewayRunning {
    try {
        $resp = Invoke-WebRequest -Uri ("http://127.0.0.1:" + (Get-MikuGatewayPort) + "/") -Method Head -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        if ($resp) { return $true }
    } catch {}
    return $false
}

function Remove-Live2dWorkDir {
    param([Parameter(Mandatory = $true)][hashtable]$Paths)
    if (Test-Path $Paths.WorkDir) {
        Remove-Item $Paths.WorkDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Node.js の確認（必要なら winget で導入）。戻り値: 使える状態なら $true
function Install-NodeJs {
    param([switch]$AutoInstall)
    $nv = $null
    try { $nv = node --version 2>$null } catch {}
    if ($nv) {
        Write-Host " [OK] Node.js $nv"
        return $true
    }
    if (-not $AutoInstall) {
        Write-Host " [SKIP] Node.js がありません（ビルドはスキップされます）"
        return $false
    }
    Write-Host " Node.js を winget でインストールしています（許可ダイアログが出たら「はい」）..."
    # Write-Host を通すこと: 出力を素通しするとパイプライン経由で戻り値に混入し、
    # 呼び出し側の [bool] 変換が System.Object[] エラーになる（2026-07-30 VM検証で実害）
    winget install --id OpenJS.NodeJS.LTS -e --accept-source-agreements --accept-package-agreements 2>&1 |
        ForEach-Object { Write-Host " $_" }
    # インストール直後は PATH がこのセッションに反映されないため、既定パスを追加する
    $nodeDir = Join-Path $env:ProgramFiles "nodejs"
    if (Test-Path (Join-Path $nodeDir "node.exe")) {
        $env:Path = "$nodeDir;$env:Path"
        Write-Host " [OK] Node.js $(node --version) をインストールしました"
        return $true
    }
    Write-Host " [NG] Node.js のインストールを確認できませんでした"
    Write-Host "      https://nodejs.org/ から LTS 版を入れて再実行してください"
    return $false
}

# Cubism SDK for Web の zip を展開して配置する。戻り値: 配置済みなら $true
function Install-Live2dSdk {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Paths,
        [string]$ZipPath
    )
    if (Test-Path $Paths.CoreJs) {
        Write-Host " [OK] Cubism SDK for Web は配置済みです"
        return $true
    }
    if (-not $ZipPath) {
        Write-Host " [SKIP] SDK の zip が指定されていません（後から再実行できます）"
        return $false
    }
    if (-not (Test-Path $ZipPath)) {
        Write-Host " [NG] 指定された zip が見つかりません: $ZipPath"
        return $false
    }
    Write-Host " SDK を展開しています..."
    try {
        Remove-Live2dWorkDir -Paths $Paths
        Expand-Archive -Path $ZipPath -DestinationPath $Paths.WorkDir -Force
        # zip 内の構造は「CubismSdkForWeb-X-r.X/Core,...」または直下に Core がある形の両対応
        $top = Get-ChildItem $Paths.WorkDir -Directory | Where-Object { Test-Path (Join-Path $_.FullName "Core") } | Select-Object -First 1
        if ($top) {
            Move-Item $top.FullName $Paths.SdkDir
            Remove-Live2dWorkDir -Paths $Paths
        } elseif (Test-Path (Join-Path $Paths.WorkDir "Core")) {
            Move-Item $Paths.WorkDir $Paths.SdkDir
        } else {
            Write-Host " [NG] zip 内に Core フォルダがありません。Cubism SDK for Web の zip か確認してください"
            Remove-Live2dWorkDir -Paths $Paths
            return $false
        }
    } catch {
        Write-Host " [NG] SDK の展開・配置に失敗しました: $_"
        Remove-Live2dWorkDir -Paths $Paths
        return $false
    }
    if (Test-Path $Paths.CoreJs) {
        Write-Host " [OK] $($Paths.SdkDir) に配置しました"
        return $true
    }
    return $false
}

# 初音ミクサンプルモデルの zip を展開して配置する。戻り値: 配置済みなら $true
function Install-Live2dModel {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Paths,
        [string]$ZipPath
    )
    $placed = (Test-Path (Join-Path $Paths.MikuDir "miku_sample_t04.model3.json")) -or
              (Test-Path (Join-Path $Paths.MikuDirOld "miku_sample_t04.model3.json"))
    if ($placed) {
        Write-Host " [OK] Miku モデルは配置済みです"
        return $true
    }
    if (-not (Test-Path $Paths.Resources)) {
        Write-Host " [SKIP] SDK が未配置のため、モデル配置をスキップします"
        return $false
    }
    if (-not $ZipPath) {
        Write-Host " [SKIP] モデルの zip が指定されていません（後から再実行できます）"
        return $false
    }
    if (-not (Test-Path $ZipPath)) {
        Write-Host " [NG] 指定された zip が見つかりません: $ZipPath"
        return $false
    }
    Write-Host " モデルを展開しています..."
    $ok = $false
    try {
        Remove-Live2dWorkDir -Paths $Paths
        Expand-Archive -Path $ZipPath -DestinationPath $Paths.WorkDir -Force
        # runtime フォルダを深さ問わず探す（zip 直下 or 1階層下の両対応）
        $runtime = Get-ChildItem $Paths.WorkDir -Directory -Recurse | Where-Object { $_.Name -eq "runtime" } | Select-Object -First 1
        if ($runtime) {
            New-Item -ItemType Directory -Force $Paths.MikuDirOld | Out-Null
            Copy-Item (Join-Path $runtime.FullName "*") $Paths.MikuDirOld -Recurse -Force
            Write-Host " [OK] $($Paths.MikuDirOld) に配置しました（パッチで自動リネームされます）"
            $ok = $true
        } else {
            Write-Host " [NG] zip 内に runtime フォルダがありません。miku_pro_jp.zip か確認してください"
        }
    } catch {
        Write-Host " [NG] モデルの展開・配置に失敗しました: $_"
    }
    Remove-Live2dWorkDir -Paths $Paths
    return $ok
}

# 制御コード（ws_control.ts）の配置と SDK へのパッチ適用
function Invoke-Live2dPatch {
    param([Parameter(Mandatory = $true)][hashtable]$Paths)
    if (-not (Test-Path $Paths.DemoSrc)) {
        Write-Host " [SKIP] Demo/src が見つかりません（SDK 未配置）"
        return $false
    }
    Copy-Item (Join-Path $Paths.Live2dDir "src\ws_control.ts") $Paths.DemoSrc -Force
    Write-Host " [OK] ws_control.ts を配置しました"
    Write-Host " パッチを適用しています..."
    try {
        & (Join-Path $Paths.Live2dDir "apply_patches.ps1") -DemoSrc $Paths.DemoSrc
        return $true
    } catch {
        Write-Host " [NG] パッチ適用に失敗しました: $_"
        return $false
    }
}

# npm install → npm run build（dist/ を生成）
function Invoke-Live2dBuild {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Paths,
        [bool]$NodeAvailable
    )
    if (-not ($NodeAvailable -and (Test-Path $Paths.DemoDir))) {
        Write-Host " [SKIP] Node.js または Demo ディレクトリが無いためビルドをスキップしました"
        return $false
    }
    Push-Location $Paths.DemoDir
    try {
        # Write-Host を通すこと: 戻り値への混入防止＋ログ（Transcript/インストーラ進捗）に残すため
        if (-not (Test-Path "node_modules")) {
            Write-Host " npm install を実行しています..."
            cmd /c "npm install" 2>&1 | ForEach-Object { Write-Host " $_" }
        }
        Write-Host " npm run build を実行しています..."
        cmd /c "npm run build" 2>&1 | ForEach-Object { Write-Host " $_" }
        if ($LASTEXITCODE -eq 0) {
            Write-Host " [OK] ビルド完了"
            return $true
        }
        Write-Host " [NG] ビルドに失敗しました。パッチの適用状況を確認してください"
        return $false
    } finally {
        Pop-Location
    }
}

# 入れ直し用: 展開済みの SDK・モデルはそのまま使い、制御コードの差し替えと組み立て直しだけ行う。
# 戻り値: 組み立て直せたら $true（失敗しても前の dist/ を戻すので、ミクの姿は前のまま出る）
#
# ★なぜ要るか（2026-09-27）: 入れ直しで「用意済みの Live2D を使う」を選ぶと、以前は dist/ をそのまま使い、
#   組み立て直さなかった。そのため ws_control.ts を変えても入れ直しでは届かなかった
#   （視線追従の `gaze` が黙って効かない状態になっていた）。パッチは適用済みなら飛ばす作りなので、何度流してもよい。
# ★vite は組み立ての最初に dist/ を空にするので、失敗に備えて dist/ を退避してから組み立て、失敗したら戻す
function Invoke-Live2dRebuild {
    param([Parameter(Mandatory = $true)][hashtable]$Paths)
    # 制御コードと組み立ての材料が前回組み立てたときと同じなら、組み立て直さない（入れ直しの短縮・2026-10-25）。
    # 印の関数は setup_common.ps1 にある（単体の setup_live2d.ps1 から呼ばれたときは無いので、毎回組み立てる）
    $mikuDir = Split-Path -Parent (Split-Path -Parent $Paths.Live2dDir)   # src\live2d → miku
    $stampHash = $null
    if (Get-Command Get-MikuInputHash -ErrorAction SilentlyContinue) {
        $inputs = @(Get-ChildItem -Path (Join-Path $Paths.Live2dDir "src"), (Join-Path $Paths.Live2dDir "dist") -File -Recurse -ErrorAction SilentlyContinue |
                    ForEach-Object { $_.FullName })
        $inputs += @("package.json", "package-lock.json", "esbuild.config.js", "tsconfig.json", "apply_patches.ps1", "live2d_common.ps1" |
                     ForEach-Object { Join-Path $Paths.Live2dDir $_ })
        $stampHash = Get-MikuInputHash -Files $inputs
        if ((Test-Path (Join-Path $Paths.DemoDir "dist\index.html")) -and
            (Test-MikuStamp -MikuDir $mikuDir -Name "live2d_build" -Hash $stampHash)) {
            Write-Host " [SKIP] 制御コードは前回の組み立てから変わっていないので、組み立て直しません"
            return $true
        }
    }
    $nodeOk = Install-NodeJs
    if (-not $nodeOk) {
        Write-Host " [SKIP] Node.js が無いため組み立て直しません（前の Live2D をそのまま使います）"
        return $false
    }
    if (-not (Invoke-Live2dPatch -Paths $Paths | Select-Object -Last 1)) {
        Write-Host " [SKIP] 制御コードを入れられなかったため組み立て直しません（前の Live2D をそのまま使います）"
        return $false
    }
    $dist = Join-Path $Paths.DemoDir "dist"
    $prev = Join-Path $Paths.DemoDir "dist_prev"
    if (Test-Path $prev) { Remove-Item $prev -Recurse -Force }
    if (Test-Path $dist) { Move-Item $dist $prev }
    $built = [bool](Invoke-Live2dBuild -Paths $Paths -NodeAvailable $true | Select-Object -Last 1)
    if ($built -and (Test-Path (Join-Path $dist "index.html"))) {
        if (Test-Path $prev) { Remove-Item $prev -Recurse -Force }
        Write-Host " [OK] 制御コードを最新にして組み立て直しました"
        if ($stampHash) { Set-MikuStamp -MikuDir $mikuDir -Name "live2d_build" -Hash $stampHash }
        return $true
    }
    Write-Host " [NG] 組み立て直しに失敗しました。前の Live2D をそのまま使います"
    if (Test-Path $prev) {
        if (Test-Path $dist) { Remove-Item $dist -Recurse -Force }
        Move-Item $prev $dist
    }
    return $false
}

# Live2D モデルからアプリのアイコンを生成し、作成済みショートカットへ反映する。
# 公式サンプルモデルは再配布できないため配布物には同梱できない。利用者が自分で
# ダウンロードしたモデルから、利用者の PC 上で作れば再配布に当たらない
# （生成できなければ同梱アイコンのままなので、失敗しても止めない）。
function New-MikuIconFromModel {
    param([Parameter(Mandatory = $true)][hashtable]$Paths)

    $mikuDir = Split-Path -Parent (Split-Path -Parent $Paths.Live2dDir)   # src\live2d → miku
    $python  = Join-Path $mikuDir "venv\Scripts\python.exe"
    $script  = Join-Path $Paths.Live2dDir "generate_icon.py"
    $icoPath = Join-Path $mikuDir "runtime\miku_icon.ico"

    if (-not (Test-Path $python)) {
        Write-Host " [SKIP] Python環境が未構築のため、アイコン生成をスキップしました"
        return $false
    }
    if (-not (Test-Path $script)) { return $false }

    # Python は UTF-8 で出力するが、PowerShell 5.1 は既定の cp932 として読むため化ける。
    # 成功時はどうでもよくても、**失敗したときに理由が読めない**ので揃えておく
    $prevEncoding = [Console]::OutputEncoding
    $prevPyIo     = $env:PYTHONIOENCODING
    try {
        [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
        $env:PYTHONIOENCODING = "utf-8"
        & $python $script 2>&1 | ForEach-Object { Write-Host " $_" }
    } finally {
        [Console]::OutputEncoding = $prevEncoding
        $env:PYTHONIOENCODING = $prevPyIo
    }
    if (-not (Test-Path $icoPath)) { return $false }

    $updated = Update-MikuShortcutIcon -MikuDir $mikuDir -IconPath $icoPath
    if ($updated -gt 0) {
        Write-Host " [OK] ショートカットのアイコンを更新しました（$updated 件）"
    } else {
        # 0 件でも黙って進むと「アイコンが変わらない」原因が分からない
        Write-Host " [WARN] 差し替え対象のショートカットが見つかりませんでした"
    }
    return $true
}

# ミクのフォルダを指すショートカットのアイコンを差し替える。
# インストーラが作るショートカットは同梱アイコンを指しているため、モデルから
# 生成できた場合はここで上書きする
function Update-MikuShortcutIcon {
    param(
        [Parameter(Mandatory = $true)][string]$MikuDir,
        [Parameter(Mandatory = $true)][string]$IconPath
    )
    # インストーラは管理者権限で動くため、Inno の {autodesktop} / {autoprograms} は
    # **全ユーザー共通**（C:\Users\Public\Desktop 等）に解決される。個人用だけを見ていると
    # デスクトップのショートカットが取り残される（2026-08-02: 更新1件＝スタートメニューのみ、
    # デスクトップは同梱アイコンのままという不具合になった）。4か所すべてを見る
    $targets = @(
        [Environment]::GetFolderPath("Desktop"),                  # 個人のデスクトップ
        [Environment]::GetFolderPath("CommonDesktopDirectory"),   # 全ユーザー共通のデスクトップ
        [Environment]::GetFolderPath("Programs"),                 # 個人のスタートメニュー
        [Environment]::GetFolderPath("CommonPrograms")            # 全ユーザー共通のスタートメニュー
    ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

    $wsh = New-Object -ComObject WScript.Shell
    $count = 0
    foreach ($dir in $targets) {
        Get-ChildItem $dir -Filter "*.lnk" -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
            try {
                $sc = $wsh.CreateShortcut($_.FullName)
                if ($sc.TargetPath -and $sc.TargetPath.StartsWith($MikuDir, [System.StringComparison]::OrdinalIgnoreCase)) {
                    $sc.IconLocation = "$IconPath,0"
                    $sc.Save()
                    $count++
                }
            } catch {}
        }
    }
    return $count
}

# 結果のチェックリストを表示する。戻り値: すべて OK なら $true
function Show-Live2dChecklist {
    param([Parameter(Mandatory = $true)][hashtable]$Paths)
    Write-Host ""
    Write-Host "========================================"
    Write-Host "  Live2D セットアップ結果"
    Write-Host "========================================"
    $checks = @(
        @{ Label = "Cubism Core";      Path = $Paths.CoreJs },
        @{ Label = "Cubism Framework"; Path = (Join-Path $Paths.SdkDir "Framework") },
        @{ Label = "Miku モデル";      Path = $Paths.MikuDir },
        @{ Label = "ws_control.ts";    Path = (Join-Path $Paths.DemoSrc "ws_control.ts") },
        @{ Label = "Demo ビルド";      Path = (Join-Path $Paths.DemoDir "dist") }
    )
    $allOk = $true
    foreach ($chk in $checks) {
        if (Test-Path $chk.Path) {
            Write-Host " [OK] $($chk.Label)"
        } else {
            Write-Host " [NG] $($chk.Label)"
            $allOk = $false
        }
    }
    Write-Host ""
    return $allOk
}

# SDK 配置 → モデル配置 → パッチ → ビルド → チェックリスト の一連を実行する。
# 対話版・非対話版のどちらからもこの1本を呼べばよい（zip パスは呼び出し側が用意する）
function Invoke-Live2dSetup {
    param(
        [Parameter(Mandatory = $true)][string]$Live2dDir,
        [string]$SdkZip,
        [string]$ModelZip,
        [switch]$AutoInstallNode
    )
    $paths = Get-Live2dPaths -Live2dDir $Live2dDir
    Write-Host ""
    Write-Host "[1/5] Node.js を確認しています..."
    $nodeOk = Install-NodeJs -AutoInstall:$AutoInstallNode

    Write-Host ""
    Write-Host "[2/5] Cubism SDK for Web を確認しています..."
    Install-Live2dSdk -Paths $paths -ZipPath $SdkZip | Out-Null

    Write-Host ""
    Write-Host "[3/5] Miku モデルを確認しています..."
    Install-Live2dModel -Paths $paths -ZipPath $ModelZip | Out-Null

    Write-Host ""
    Write-Host "[4/5] 制御コードを配置しています..."
    Invoke-Live2dPatch -Paths $paths | Out-Null

    Write-Host ""
    Write-Host "[5/5] ビルドしています..."
    Invoke-Live2dBuild -Paths $paths -NodeAvailable $nodeOk | Out-Null

    Write-Host ""
    Write-Host "アイコンを作成しています（ミクの姿から生成します）..."
    New-MikuIconFromModel -Paths $paths | Out-Null

    return (Show-Live2dChecklist -Paths $paths)
}
