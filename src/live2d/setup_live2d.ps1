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

# Live2D セットアップ（対話版・単体実行の入口）
#
# zip の場所をダイアログで聞き、実処理は live2d_common.ps1 の関数に委譲する。
# 非対話版（miku\setup_unattended.ps1）も同じ関数を呼ぶため、処理は1箇所にしかない。
#
# ※このファイルは BOM付き UTF-8 で保存すること（Windows PowerShell 5.1 が
#   BOM無しを Shift-JIS と誤読して構文エラーになるため。2026-07-25 事後規約）

$ErrorActionPreference = "Continue"
Add-Type -AssemblyName System.Windows.Forms

. (Join-Path $PSScriptRoot "live2d_common.ps1")

$LIVE2D_DIR = $PSScriptRoot

Write-Host "========================================"
Write-Host "  Live2D セットアップ"
Write-Host "========================================"
Write-Host ""
Write-Host "Live2Dディレクトリ: $LIVE2D_DIR"

# システム稼働中ガード: ビルドは配信中の dist/ を作り直すため、稼働中の実行は
# 一時的な表示不能・ビルド失敗時の破損リスクがある
if (Test-MikuGatewayRunning) {
    Write-Host ""
    Write-Host (" [警告] ミクのシステムが稼働中のようです（:" + (Get-MikuGatewayPort) + " 応答あり）。")
    Write-Host "        セットアップは Live2D の配信ファイルを作り直すため、システム終了後の実行を推奨します。"
    $c = Read-Host " それでも続行しますか？ (Y/N)"
    if ($c -ine "Y") {
        Write-Host " 中止しました。システムを終了してから再実行してください。"
        exit 0
    }
}

function Select-ZipFile {
    param([string]$Title)
    $dlg = New-Object System.Windows.Forms.OpenFileDialog
    $dlg.Title  = $Title
    $dlg.Filter = "ZIPファイル (*.zip)|*.zip"
    $downloads = Join-Path ([Environment]::GetFolderPath("UserProfile")) "Downloads"
    if (Test-Path $downloads) { $dlg.InitialDirectory = $downloads }
    if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { return $dlg.FileName }
    return $null
}

$paths = Get-Live2dPaths -Live2dDir $LIVE2D_DIR

# --- zip の場所を対話で集める（配置済みなら聞かない） ---
$sdkZip = $null
if (-not (Test-Path $paths.CoreJs)) {
    Write-Host ""
    Write-Host "Cubism SDK for Web がありません。以下からダウンロードしてください（利用規約への同意が必要）:"
    Write-Host "  https://www.live2d.com/sdk/download/web/"
    $c = Read-Host " zip をダウンロード済みですか？ (Y=選択する / N=スキップ)"
    if ($c -ieq "Y") { $sdkZip = Select-ZipFile -Title "CubismSdkForWeb-X-r.X.zip を選択してください" }
}

$modelZip = $null
$modelPlaced = (Test-Path (Join-Path $paths.MikuDir "miku_sample_t04.model3.json")) -or
               (Test-Path (Join-Path $paths.MikuDirOld "miku_sample_t04.model3.json"))
if (-not $modelPlaced) {
    Write-Host ""
    Write-Host "Miku モデルがありません。以下からダウンロードしてください（初音ミクサンプルモデル Pro版）:"
    Write-Host "  https://www.live2d.com/learn/sample/"
    $c = Read-Host " miku_pro_jp.zip をダウンロード済みですか？ (Y=選択する / N=スキップ)"
    if ($c -ieq "Y") { $modelZip = Select-ZipFile -Title "miku_pro_jp.zip を選択してください" }
}

# --- Node.js を自動導入してよいか（無い場合のみ聞く） ---
$autoNode = $false
$nodeExists = $false
try { if (node --version 2>$null) { $nodeExists = $true } } catch {}
if (-not $nodeExists) {
    Write-Host ""
    Write-Host "Node.js が見つかりません（Live2D のビルドに必要です）"
    $c = Read-Host " winget で自動インストールしますか？ 許可ダイアログが出たら「はい」を押してください (Y/N)"
    if ($c -ieq "Y") { $autoNode = $true }
}

Invoke-Live2dSetup -Live2dDir $LIVE2D_DIR -SdkZip $sdkZip -ModelZip $modelZip -AutoInstallNode:$autoNode | Out-Null
