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

param(
    # 指定すると起動時の記憶登録処理をスキップする(10秒のキー待ちも省略)
    # 例: powershell -ExecutionPolicy Bypass -File system_start.ps1 -SkipMemory
    [switch]$SkipMemory
)

# ==========================================
#   ユーザー設定部
# ==========================================
$MIKU_PATH     = "ここにmiku_agent-mainのディレクトリのフルパスを記入(例 C:\MyProgram\miku_agent-main)"
# Phase 7 で Pydantic AI に移行済みのため Dify / Docker は不要
# $DIFY_PATH   = "ここにDifyのdockerディレクトリのフルパスを記入(例 C:\MyProgram\dify\docker)"

# Cloudflare Quick Tunnel設定($false=OFF, $true=ON)
# 外部（スマホ・外出先）からアクセスする場合のみ $true にする。事前に cloudflared のインストールが必要:
#   winget install --id Cloudflare.cloudflared
$CLOUDFLARE_ENABLED = $false

# ==========================================
#   自動構築・自動実行パス
# ==========================================
$MIKU_BASE             = "$MIKU_PATH\miku"
$MIKU_SRC              = "$MIKU_BASE\src"

# モデル類（顔認識重み・whisper等）の保存先をフォルダ内に隔離する
# （アンインストール＝フォルダ削除で完結させる方針。runtime/ は消えても再生成できるもの置き場）
$env:TORCH_HOME = "$MIKU_BASE\runtime\models\torch"
$env:HF_HOME    = "$MIKU_BASE\runtime\models\hf"

# ==========================================
#   [1] Docker Desktop 確認  (Phase 7 で不要となり、切り替え用にコメントアウト)
# ==========================================
# Write-Host "[1/5] Docker Desktop を確認しています..."
# docker info 2>$null
# if ($LASTEXITCODE -ne 0) {
#     Write-Host "Docker Desktop が起動していません。起動しています..."
#     Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
#     Write-Host "Docker Desktop の起動待機しています(最大60秒)..."
#     $count = 0
#     do {
#         Start-Sleep -Seconds 5
#         docker info 2>$null
#         $count++
#     } while ($LASTEXITCODE -ne 0 -and $count -lt 12)
#     if ($LASTEXITCODE -ne 0) {
#         Write-Host "[警告] Docker Desktop の起動がタイムアウトしました"
#         Write-Host "手動で Docker Desktop を起動してから続行してください"
#         pause
#     }
# }
# Write-Host "Docker Desktop は起動しています"

# ==========================================
#   [2] Dify コンテナ確認  (Phase 7 で不要となり、切り替え用にコメントアウト)
# ==========================================
# Write-Host "[2/5] Dify コンテナを確認しています..."
# $difyRunning = docker ps --filter "name=docker-api-1" --format "{{.Names}}" 2>$null | Select-String "api"
# if ($difyRunning) {
#     Write-Host "Dify コンテナは既に起動済みです"
# } elseif (Test-Path "$DIFY_PATH\docker-compose.yaml") {
#     Write-Host "Dify コンテナを起動しています..."
#     Set-Location "$DIFY_PATH"
#     docker compose up -d
#     Write-Host "Dify コンテナを起動しました。起動待機しています(最大60秒)..."
#     $dify_count = 0
#     do {
#         Start-Sleep -Seconds 5
#         $difyRunning = docker ps --filter "name=docker-api-1" --format "{{.Names}}" 2>$null | Select-String "api"
#         $dify_count++
#     } while (-not $difyRunning -and $dify_count -lt 12)
#     if (-not $difyRunning) {
#         Write-Host "[警告] Dify コンテナの起動がタイムアウトしました"
#         Read-Host "Enterキーで続行"
#     } else {
#         Write-Host "Dify コンテナは起動しています"
#     }
# } else {
#     Write-Host "[警告] Dify パスが設定されていないか、存在しません: $DIFY_PATH"
#     Write-Host "Dify は手動で起動してください"
# }

# ==========================================
#   [1] Cloudflare Quick Tunnel 起動(オプション)
# ==========================================
if ($CLOUDFLARE_ENABLED) {
    Write-Host "[1/1] Cloudflare Quick Tunnel を起動しています..."
    $cloudflaredRunning = Get-Process -Name "cloudflared" -ErrorAction SilentlyContinue
    # セットアップが tools\cloudflared に自動DLしたポータブル版を優先し、無ければ PATH 上を探す
    $cloudflaredExe = Join-Path $MIKU_BASE "runtime\cloudflared\cloudflared.exe"
    if (-not (Test-Path $cloudflaredExe)) {
        $cfCmd = Get-Command cloudflared -ErrorAction SilentlyContinue
        if ($cfCmd) { $cloudflaredExe = $cfCmd.Source } else { $cloudflaredExe = $null }
    }
    if ($cloudflaredRunning) {
        Write-Host "Cloudflare トンネルは既に起動済みです"
    } elseif (-not $cloudflaredExe) {
        Write-Host "警告: cloudflared が見つからないため、外部アクセス用トンネルをスキップします"
        Write-Host "  使う場合はスタート画面の「設定」タブで外部アクセスを有効にしてください"
    } else {
        Start-Process "powershell.exe" -ArgumentList "-NoExit -ExecutionPolicy Bypass -Command `"& '$cloudflaredExe' tunnel --url http://localhost:5001`"" -WindowStyle Normal
        Write-Host "Cloudflare トンネルを起動しました"
        Write-Host "★ 別ウィンドウに表示されるURLでアクセスできます"
    }
} else {
    Write-Host "[1/1] Cloudflare Quick Tunnel はスキップ(無効化されています)"
}

# ==========================================
#   起動時記憶登録スキップ確認(10秒以内に全体スキップ)
# ==========================================
$MEMORY_SKIP_ARG = ""
Write-Host ""

if ($SkipMemory) {
    # -SkipMemory 指定時はキー待ちを行わない(非対話起動・自動化から使える)
    $MEMORY_SKIP_ARG = "--skip-memory"
    Write-Host "記憶登録をスキップします(-SkipMemory 指定)"
} else {
    Write-Host "起動時の記憶登録処理を実施します(予定)"
    Write-Host "10秒以内に [S] を押すと記憶登録をスキップします: " -NoNewline

    $skipMemoryByKey = $false
    $deadline = (Get-Date).AddSeconds(10)
    while ((Get-Date) -lt $deadline) {
        if ([Console]::KeyAvailable) {
            $key = [Console]::ReadKey($true)
            if ($key.Key -eq 'S') {
                $skipMemoryByKey = $true
                break
            } elseif ($key.Key -eq 'N') {
                break
            }
        }
        Start-Sleep -Milliseconds 100
    }
    Write-Host ""

    if ($skipMemoryByKey) {
        $MEMORY_SKIP_ARG = "--skip-memory"
        Write-Host "記憶登録をスキップします"
    }
}

# ==========================================
#   ミクの会話システム起動
# ==========================================
Write-Host ""
Write-Host "ミクの会話システムを起動しています..."
Set-Location $MIKU_BASE

# venv が無い＝セットアップ未完了。そのまま進むと「Python was not found（Store案内）」等の
# 分かりにくいエラーが連鎖するため、ここで明確に止める（2026-07-26 クリーンVMで実害）
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host ""
    Write-Host "エラー: Python環境（venv）が見つかりません。セットアップが完了していません。"
    Write-Host "  インストーラ（MikuAgentSetup.exe）をもう一度実行してセットアップを完了させてください。"
    Write-Host "  （GUIインストーラで失敗した場合の記録: $MIKU_BASE\runtime\setup_unattended.log）"
    Read-Host "Enterキーで終了"
    exit 1
}
& "venv\Scripts\Activate.ps1"
Set-Location $MIKU_SRC
$env:PYTHONUTF8 = "1"
if ($MEMORY_SKIP_ARG) {
    python gateway_service.py $MEMORY_SKIP_ARG
} else {
    python gateway_service.py
}

# ブラウザで会話システムを開く(Chromeの新しいウィンドウ)
# Start-Process "chrome" "--new-window http://localhost:5001"
