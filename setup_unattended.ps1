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

# 非対話セットアップ（GUIインストーラから呼ばれる入口・Live2D まで1本で完結）
#
# 対話は一切行わず、すべてパラメータで受け取る。実処理は setup_common.ps1 /
# live2d_common.ps1 の共通関数に委譲するため、対話版と処理が二重にならない。
#
# 使用例:
#   powershell -ExecutionPolicy Bypass -File setup_unattended.ps1 `
#       -GoogleApiKey "xxx" -FishApiKey "yyy" -EnableTunnel `
#       -Live2dSdkZip "C:\dl\CubismSdkForWeb-5-r.4.zip" -Live2dModelZip "C:\dl\miku_pro_jp.zip"
#
# 終了コード: 0=成功 / 1=失敗（インストーラ側はこれを見てメッセージを出し分ける）
#
# ※このファイルは BOM付き UTF-8・PowerShell 5.1 互換構文で書くこと

param(
    # APIキー（空なら settings.json の現在値を維持）
    [string]$GoogleApiKey = "",
    [string]$FishApiKey = "",

    # 外部アクセス（Cloudflare Quick Tunnel）を有効にする
    [switch]$EnableTunnel,

    # Live2D 素材の zip（省略するとアバター無しでセットアップし、後から追加できる）
    [string]$Live2dSdkZip = "",
    [string]$Live2dModelZip = "",

    # 作曲機能も導入する（GPU あり＝カバー込みで約15GB・40〜90分／GPU なし＝オリジナルのみ約4GB・十数分）
    [switch]$InstallSongService,

    # ★デバッグ用（2026-09-19）。VOCALOID6 を入れられない環境で**導入まで**通したいときだけ。
    #   飛ばすのは導入の可否だけ。
    #   実行時の歌声の書き出しは正直に失敗する。GUI インストーラからは渡さない
    [switch]$SkipVocaloidCheck,

    # デスクトップショートカットを作る（インストーラ側で作る場合は指定しない）
    [switch]$CreateShortcut
)

$ErrorActionPreference = "Continue"
# 5.1 の Invoke-WebRequest は進捗表示でダウンロードが大幅に遅くなるため抑止する
$ProgressPreference = "SilentlyContinue"

# APIキーはコマンドライン引数だと Start-Transcript のヘッダ（ホスト アプリケーション行）に
# 平文で記録されてしまうため、GUIインストーラは環境変数で渡してくる（2026-07-29 の実害から）。
# 引数指定があればそちらを優先（手動実行用）
if (-not $GoogleApiKey -and $env:MIKU_SETUP_GOOGLE_API_KEY) { $GoogleApiKey = $env:MIKU_SETUP_GOOGLE_API_KEY }
if (-not $FishApiKey -and $env:MIKU_SETUP_FISH_API_KEY) { $FishApiKey = $env:MIKU_SETUP_FISH_API_KEY }

# 出力がパイプへリダイレクトされている場合のみ UTF-8 で書く。
# GUIインストーラ（ExecAndLogOutput）は UTF-8 として読むため、PS5.1 既定の cp932 のままだと
# 進捗表示が文字化けする（2026-07-26 に実害・cp932コンソールで再現確認）。
# 逆に手動でコンソール実行するときは cp932 のままが正しく表示されるので切り替えない。
try {
    if ([Console]::IsOutputRedirected) {
        [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
    }
} catch {}

$MIKU_DIR = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
. (Join-Path $MIKU_DIR "setup_common.ps1")
. (Join-Path $MIKU_DIR "src\live2d\live2d_common.ps1")

# 全出力をログにも残す（GUIインストーラ経由だと画面が閉じた後に失敗原因を追えないため。
# 2026-07-26 のクリーンVM検証で「なぜ venv が無いのか分からない」実害から）
try {
    New-Item -ItemType Directory -Force (Join-Path $MIKU_DIR "runtime") | Out-Null
    Start-Transcript -Path (Join-Path $MIKU_DIR "runtime\setup_unattended.log") -Append | Out-Null
} catch {}
# 自分の PID を置く。GUI インストーラの「中止する」が、これを見て子プロセスごと止める（2026-09-26）
try {
    Set-Content -Path (Join-Path $MIKU_DIR "runtime\setup.pid") -Value $PID -Encoding ASCII
} catch {}

Write-Host "========================================"
Write-Host "  ワタシノセカイト初音ミク セットアップ"
Write-Host "========================================"
Write-Host "インストール先: $MIKU_DIR"

# ==========================================
#   [1/6] 設定ファイルの準備
# ==========================================
Write-Host ""
Write-Host "[1/6] 設定ファイルを準備しています..."
$config = Initialize-MikuConfigFiles -MikuDir $MIKU_DIR
if (-not $config.Ok) {
    Write-Host "エラー: 設定ファイルの準備に失敗しました"
    try { Stop-Transcript | Out-Null } catch {}
    exit 1
}

Set-MikuApiKeys -SettingsJson $config.SettingsJson -GoogleApiKey $GoogleApiKey -FishApiKey $FishApiKey | Out-Null

# ==========================================
#   [2/6] 外部アクセス（任意）
# ==========================================
Write-Host ""
Write-Host "[2/6] 外部アクセスの設定..."
$tunnelReady = $false
if ($EnableTunnel) {
    $tunnelReady = Install-Cloudflared -MikuDir $MIKU_DIR
    if ($tunnelReady) {
        Write-Host "OK 外部アクセスを有効にしました（URLはスタート画面の「アクセス先」に出ます）"
    } else {
        Write-Host "外部アクセスは無効のままにします（セットアップを再実行すれば設定できます）"
    }
} else {
    Write-Host "外部アクセスは使いません"
}

# コントロールパネルが読む先（settings.json の system）へ書き込む
Set-MikuSystemSettings -SettingsJson $config.SettingsJson `
    -TunnelEnabled $tunnelReady | Out-Null

# 旧ランチャー（start_miku.bat 経由）向けにも書き込む（従来の起動経路を壊さない）
Set-MikuStartVars -StartPs1 $config.StartPs1 `
    -MikuPath (Split-Path -Parent $MIKU_DIR) `
    -CloudflareEnabled $tunnelReady | Out-Null

# ==========================================
#   [3/6] Python 環境
# ==========================================
Write-Host ""
Write-Host "[3/6] Python環境を構築しています（数GBのダウンロードがあります）..."
Install-VcRedist | Out-Null
Install-SmartTurnModel -MikuDir $MIKU_DIR | Out-Null
$pyEnv = Install-MikuPythonEnv -MikuDir $MIKU_DIR
if (-not $pyEnv.Ok) {
    Write-Host "エラー: Python環境の構築に失敗しました"
    try { Stop-Transcript | Out-Null } catch {}
    exit 1
}

# ==========================================
#   [4/6] AIモデルの取得
# ==========================================
Write-Host ""
Write-Host "[4/6] AIモデルを取得しています..."
Install-MikuFaceWeights -MikuDir $MIKU_DIR | Out-Null
Install-MikuVoiceWeights -MikuDir $MIKU_DIR | Out-Null
# 画面側のライブラリ（視線・動画の圧縮・アイコン生成の描画）。Live2D のアイコン生成より前に置く
Install-MikuWebVendor -MikuDir $MIKU_DIR | Out-Null

# ==========================================
#   [5/6] Live2D（アバター）
# ==========================================
Write-Host ""
Write-Host "[5/6] Live2D（ミクの姿）をセットアップしています..."
# Live2D は必須。ミクの姿が出ない状態を「セットアップできた」とは扱わない。
# 失敗は $live2dOk に持ち、最後にセットアップ全体の成否へ反映する
$live2dOk = $false
if ($Live2dSdkZip -and $Live2dModelZip) {
    # Node.js が無ければ winget で自動導入する（UACが出る）
    # 最後の1個だけを見る: 関数内で拾い漏れた出力が混ざると配列になり、[bool] が常に真になるため
    $result = Invoke-Live2dSetup -Live2dDir (Join-Path $MIKU_DIR "src\live2d") `
        -SdkZip $Live2dSdkZip -ModelZip $Live2dModelZip -AutoInstallNode
    $live2dOk = [bool]($result | Select-Object -Last 1)
} elseif (Test-Path (Join-Path $MIKU_DIR "src\live2d\CubismWebSamples\Samples\TypeScript\Demo\dist\index.html")) {
    # 入れ直し（2026-09-26）: zip を渡されなくても、ミクの姿がもうできていればそれを使う。
    # ただし制御コード（ws_control.ts）は新しい版のものに差し替えて組み立て直す（2026-09-27）。
    # 組み立て直しに失敗しても前の姿がそのまま出るので、成否に関わらずセットアップは続ける
    Write-Host "Live2D はセットアップ済みの素材を使い、制御コードだけ最新にして組み立て直します"
    Invoke-Live2dRebuild -Paths (Get-Live2dPaths -Live2dDir (Join-Path $MIKU_DIR "src\live2d")) | Out-Null
    $live2dOk = $true
} else {
    Write-Host "エラー: Live2D の素材（zip 2つ）が指定されていません"
    Write-Host "  Cubism SDK for Web : https://www.live2d.com/sdk/download/web/"
    Write-Host "  初音ミク サンプルモデル Pro版 : https://www.live2d.com/learn/sample/"
    Write-Host "  ダウンロード後、miku\src\live2d\setup_live2d.bat を実行すると続きから導入できます"
}

# ==========================================
#   作曲機能（任意）
# ==========================================
# ★★**対話版と同じ判断にそろえる**（2026-09-19）★★
#   ここだけ 2026-09 の2つの変更が入っておらず、**GUI インストーラで入れた人にだけ古い挙動**が
#   届いていた（H6 の検証で発見。GUI は `miku_setup.iss` → この無人版を呼ぶ＝配布の本道）:
#     ① **GPU が無ければ丸ごとスキップ**していた。実際は 2026-09-01 から
#        **オリジナル曲だけなら GPU 無しで導入できる**（`-OriginalOnly`・約4GB／十数分。
#        うち約3GBは歌声の聞き取り用＝鼻歌を歌詞ごと受け取るためのもので、CPU で動く）。
#        README・CHANGELOG に「グラフィックボードが無くても作れる」と書いたのに、
#        **GUI で入れた人には一生届かない**状態だった。
#     ② **VOCALOID6 の確認をしていなかった**（2026-09-06 の判断「歌わせられないなら導入しない」）。
#        判断自体は `Install-MikuSongService` の入口にもあるので、ここは渡すだけでよい。
#   音源しか無い曲のカバーに GPU が要ることは変わらない（楽譜〔MIDI〕が見つかる曲のカバーは GPU 不要）。
#   **GPU の有無で「入れない」ではなく
#   「何を入れるか」が変わる**、が正しい形。
if ($InstallSongService) {
    $songOriginalOnly = (-not (Get-NvidiaGpuName))
    if ($songOriginalOnly) {
        Write-Host ""
        Write-Host "NVIDIA GPU（CUDA）が見つからないため、音源しか無い曲のカバーに使うものを除いて導入します"
        Write-Host "  オリジナル曲と、楽譜（MIDI）が見つかる曲のカバーは作れます。"
        Write-Host "  音源しか無い曲のカバーは、GPU を追加した後に"
        Write-Host "  miku\src\song_service\setup_song_service.ps1 を実行すると追加できます"
    }
    Install-MikuSongService -MikuDir $MIKU_DIR -SettingsJson $config.SettingsJson `
        -OriginalOnly:$songOriginalOnly -SkipVocaloidCheck:$SkipVocaloidCheck | Out-Null
}

# ==========================================
#   [6/6] 最終確認
# ==========================================
Write-Host ""
Write-Host "[6/6] 最終確認を行います..."
Show-MikuFinalCheck -StartPs1 $config.StartPs1 -SettingsJson $config.SettingsJson

if ($CreateShortcut) {
    New-MikuDesktopShortcut -MikuDir $MIKU_DIR -StartPs1 $config.StartPs1 | Out-Null
}

if (-not $live2dOk) {
    Write-Host ""
    Write-Host "Live2D のセットアップが完了していないため、セットアップは未完了です"
    Write-Host "  miku\src\live2d\setup_live2d.bat を実行するか、"
    Write-Host "  スタート画面の「設定」タブから入れ直してください"
    try { Stop-Transcript | Out-Null } catch {}
    exit 1
}

Show-MikuCompletionMessage
try { Stop-Transcript | Out-Null } catch {}
exit 0
