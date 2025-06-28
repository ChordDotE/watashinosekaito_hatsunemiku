@echo off
chcp 932 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   ワタシノセカイト初音ミク 対話型セットアップ
echo ========================================
echo.
echo このツールは以下の作業を対話形式で行います：
echo 1. 設定ファイルの準備
echo 2. パス設定の支援
echo 3. Python環境の構築
echo 4. 最終確認とテスト
echo.
echo 事前に以下が完了していることを確認してください：
echo - CUDA ドライバーのインストール^（必要に応じて^）
echo - VOICEVOX のセットアップ
echo - RVC のダウンロードとセットアップ
echo - OpenRouter APIキーの設定
echo.
set /p "continue=続行しますか？ (Y/N): "
if /i not "%continue%"=="Y" (
    echo セットアップを中止しました。
    pause
    exit /b 0
)

REM "現在のディレクトリを取得"
set "CURRENT_DIR=%~dp0"
set "SRC_DIR=%CURRENT_DIR%src"

echo.
echo ========================================
echo   [1/4] 設定ファイルの準備
echo ========================================
echo.

REM "設定ファイルのコピー"
echo 設定ファイルを準備しています...
if exist "%CURRENT_DIR%system_start_sample.bat" (
    if not exist "%CURRENT_DIR%system_start.bat" (
        copy "%CURRENT_DIR%system_start_sample.bat" "%CURRENT_DIR%system_start.bat" >nul
        echo OK system_start.bat を作成しました
    ) else (
        echo OK system_start.bat は既に存在します
    )
) else (
    echo NG エラー: system_start_sample.bat が見つかりません
    pause
    exit /b 1
)

if exist "%SRC_DIR%\settings_sample.json" (
    if not exist "%SRC_DIR%\settings.json" (
        copy "%SRC_DIR%\settings_sample.json" "%SRC_DIR%\settings.json" >nul
        echo OK settings.json を作成しました
    ) else (
        echo OK settings.json は既に存在します
    )
) else (
    echo NG エラー: settings_sample.json が見つかりません
    pause
    exit /b 1
)

echo.
echo ========================================
echo   [2/4] パス設定の支援
echo ========================================
echo.

echo system_start.bat に必要なパスを設定します。
echo.

REM "VOICEVOXパスの設定"
echo [VOICEVOX パスの設定]
echo VOICEVOXを展開したフォルダを選択してください。
echo 例: C:\MyProgram\voicevox-windows-directml-0.22.4
echo.
echo フォルダ選択ダイアログを開いています...
echo （ダイアログが表示されない場合は、タスクバーを確認してください）
echo.

REM "フォルダ選択ダイアログを使用"
for /f "delims=" %%i in ('powershell -command "Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.FolderBrowserDialog; $f.Description = 'VOICEVOXを展開したフォルダを選択してください（例: C:\sample\voicevox-windows-directml-0.22.4）'; $f.ShowNewFolderButton = $false; if($f.ShowDialog() -eq 'OK'){ $f.SelectedPath } else { '' }"') do set "VOICEVOX_PATH=%%i"

if "%VOICEVOX_PATH%"=="" (
    echo.
    echo フォルダが選択されませんでした。手動入力に切り替えます。
    set /p "VOICEVOX_PATH=VOICEVOXのフルパスを入力してください: "
)

if "%VOICEVOX_PATH%"=="" (
    echo  エラー: VOICEVOXパスが設定されませんでした
    pause
    exit /b 1
)

if not exist "%VOICEVOX_PATH%" (
    echo  警告: 指定されたVOICEVOXパスが存在しません: %VOICEVOX_PATH%
    set /p "continue_voicevox=続行しますか？ (Y/N): "
    if /i not "%continue_voicevox%"=="Y" exit /b 1
)

echo  VOICEVOXパス: %VOICEVOX_PATH%

echo.
REM "RVCパスの設定"
echo [RVC パスの設定]
echo RVCを展開したフォルダを選択してください。
echo 例: C:\MyProgram\vcclient_win_cuda_2.0.76-beta
echo.
echo フォルダ選択ダイアログを開いています...
echo （ダイアログが表示されない場合は、タスクバーを確認してください）
echo.

REM "フォルダ選択ダイアログを使用"
for /f "delims=" %%i in ('powershell -command "Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.FolderBrowserDialog; $f.Description = 'RVCを展開したフォルダを選択してください（例: C:\sample\vcclient_win_cuda_2.0.76-beta）'; $f.ShowNewFolderButton = $false; if($f.ShowDialog() -eq 'OK'){ $f.SelectedPath } else { '' }"') do set "RVC_PATH=%%i"

if "%RVC_PATH%"=="" (
    echo.
    echo フォルダが選択されませんでした。手動入力に切り替えます。
    set /p "RVC_PATH=RVCのフルパスを入力してください: "
)

if "%RVC_PATH%"=="" (
    echo  エラー: RVCパスが設定されませんでした
    pause
    exit /b 1
)

if not exist "%RVC_PATH%" (
    echo  警告: 指定されたRVCパスが存在しません: %RVC_PATH%
    set /p "continue_rvc=続行しますか？ (Y/N): "
    if /i not "%continue_rvc%"=="Y" exit /b 1
)

echo  RVCパス: %RVC_PATH%

echo.
REM "MIKUパスの設定"
echo [MIKU パスの設定]
echo miku_agent-mainフォルダのパスを自動設定します。
echo 現在のディレクトリの親フォルダを自動検出しています...

for %%i in ("%CURRENT_DIR%..") do set "MIKU_PATH=%%~fi"
echo 検出されたパス: %MIKU_PATH%
echo  自動設定されたパスを使用します

if "%MIKU_PATH%"=="" (
    echo  エラー: MIKUパスが設定されませんでした
    pause
    exit /b 1
)

echo  MIKUパス: %MIKU_PATH%

echo.
echo [APIキーの設定]

REM "既存のAPIキーをチェック"
for /f "tokens=*" %%i in ('powershell -command "$content = Get-Content '%SRC_DIR%\settings.json' -Encoding UTF8 | ConvertFrom-Json; $content.api.openrouter.api_key" 2^>nul') do set "EXISTING_API_KEY=%%i"

if not "!EXISTING_API_KEY!"=="" if not "!EXISTING_API_KEY!"=="put your OpenRouter API key" (
    echo 既存のAPIキーが設定されています: !EXISTING_API_KEY:~0,20!...
    set /p "update_api_key=APIキーを更新しますか？ (Y/N): "
    if /i not "!update_api_key!"=="Y" (
        echo  既存のAPIキーを使用します
        goto :skip_api_key
    )
)

set /p "OPENROUTER_API_KEY=OpenRouter APIキーを入力してください: "

if not "!OPENROUTER_API_KEY!"=="" (
    echo APIキーを設定しています...
    powershell -command "$content = Get-Content '%SRC_DIR%\settings.json' -Encoding UTF8 | ConvertFrom-Json; $content.api.openrouter.api_key = '!OPENROUTER_API_KEY!'; $json = $content | ConvertTo-Json -Depth 10; [System.IO.File]::WriteAllText('%SRC_DIR%\settings.json', $json, [System.Text.UTF8Encoding]::new($false))"
    echo  OpenRouter APIキーを設定しました
) else (
    echo  APIキーが入力されませんでした。後で手動で設定してください。
)

:skip_api_key

echo.
echo パス設定をsystem_start.batに書き込んでいます...

REM "system_start.batの更新"
powershell -command "$content = Get-Content '%CURRENT_DIR%system_start.bat' -raw -Encoding Default; $content = $content -replace 'set \"RVC_PATH=.*\"', 'set \"RVC_PATH=%RVC_PATH%\"'; $content = $content -replace 'set \"VOICEVOX_PATH=.*\"', 'set \"VOICEVOX_PATH=%VOICEVOX_PATH%\"'; $content = $content -replace 'set \"MIKU_PATH=.*\"', 'set \"MIKU_PATH=%MIKU_PATH%\"'; [System.IO.File]::WriteAllText('%CURRENT_DIR%system_start.bat', $content, [System.Text.Encoding]::GetEncoding('shift_jis'))"

echo  system_start.bat を更新しました

echo.
echo ========================================
echo   [3/4] Python環境の構築
echo ========================================
echo.

REM "Pythonのバージョン確認"
echo Python環境を確認しています...
python --version >nul 2>&1
if !errorlevel! neq 0 (
    python3.13 --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo  エラー: Python 3.13が見つかりません
        echo   Python 3.13をインストールしてから再実行してください
        echo   https://www.python.org/downloads/release/python-3133/
        pause
        exit /b 1
    ) else (
        set "PYTHON_CMD=python3.13"
        echo  Python 3.13が見つかりました
    )
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
    echo Pythonバージョン: !PYTHON_VERSION!
    echo !PYTHON_VERSION! | findstr /r "^3\.13\." >nul
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=python"
        echo  Python 3.13が見つかりました
    ) else (
        echo  警告: Python 3.13以外のバージョンが検出されました
        echo   Python 3.13の使用を推奨します
        set "PYTHON_CMD=python"
        set /p "continue_python=続行しますか？ (Y/N): "
        if /i not "!continue_python!"=="Y" exit /b 1
    )
)

echo.
echo 仮想環境を作成しています...
if exist "%CURRENT_DIR%venv" (
    echo  仮想環境は既に存在します
) else (
    !PYTHON_CMD! -m venv venv
    if !errorlevel! neq 0 (
        echo  エラー: 仮想環境の作成に失敗しました
        pause
        exit /b 1
    )
    echo  仮想環境を作成しました
)

echo.
echo 仮想環境を有効化しています...
call "%CURRENT_DIR%venv\Scripts\activate.bat"
if !errorlevel! neq 0 (
    echo  エラー: 仮想環境の有効化に失敗しました
    pause
    exit /b 1
)
echo  仮想環境を有効化しました

echo.
echo 仮想環境の確認を行っています...

REM "VIRTUAL_ENV環境変数の確認"
if "%VIRTUAL_ENV%"=="" (
    echo  エラー: 仮想環境が有効化されていません（VIRTUAL_ENV未設定）
    echo   → venv\Scripts\activate.bat を実行してください
    pause
    exit /b 1
) else (
    echo  仮想環境が有効です: %VIRTUAL_ENV%
)

REM "Pythonパスの確認"
for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_PATH=%%i"
if "%PYTHON_PATH%"=="" (
    echo  エラー: Pythonパスの取得に失敗しました
    pause
    exit /b 1
)

echo %PYTHON_PATH% | findstr "venv" >nul
if !errorlevel! neq 0 (
    echo  エラー: Pythonがグローバル環境を使用しています
    echo   現在のPython: %PYTHON_PATH%
    echo   → 仮想環境の有効化を確認してください
    pause
    exit /b 1
) else (
    echo  仮想環境のPythonを使用しています: %PYTHON_PATH%
)



echo  仮想環境の確認が完了しました。安全にライブラリをインストールできます。
REM set /p "continue=仮想環境が有効なことを確認してください (Y/N): "
REM if /i not "%continue%"=="Y" (
REM     echo セットアップを中止しました。
REM     pause
REM     exit /b 0
REM )
echo.
echo 必要なライブラリをインストールしています...
cd /d "%SRC_DIR%"

if exist "requirements.txt" (
    echo 一般的なパッケージをインストール中...
    pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo  エラー: requirements.txtのインストールに失敗しました
        pause
        exit /b 1
    )
    echo  一般的なパッケージのインストールが完了しました
) else (
    echo  エラー: requirements.txt が見つかりません
    pause
    exit /b 1
)

echo.
if exist "requirements2.txt" (
    echo PyTorchパッケージ（CUDA対応版）をインストール中...
    pip install -r requirements2.txt
    if !errorlevel! neq 0 (
        echo  エラー: requirements2.txtのインストールに失敗しました
        pause
        exit /b 1
    )
    echo  PyTorchパッケージのインストールが完了しました
) else (
    echo  エラー: requirements2.txt が見つかりません
    pause
    exit /b 1
)

echo.
echo ========================================
echo   [4/4] 最終確認とテスト
echo ========================================
echo.

echo 設定の最終確認を行います...
echo.

REM "設定ファイルの確認"
echo [設定ファイルの確認]
if exist "%CURRENT_DIR%system_start.bat" (
    echo  system_start.bat
) else (
    echo  system_start.bat が見つかりません
)

if exist "%SRC_DIR%\settings.json" (
    echo  settings.json
) else (
    echo  settings.json が見つかりません
)

echo.
echo [外部ソフトウェアの確認]
if exist "%RVC_PATH%\dist\main\start_http.bat" (
    echo  RVC実行ファイル
) else (
    echo  RVC実行ファイルが見つかりません: %RVC_PATH%\dist\main\start_http.bat
)

if exist "%VOICEVOX_PATH%\VOICEVOX\vv-engine\run.exe" (
    echo  VOICEVOX実行ファイル
) else (
    echo  VOICEVOX実行ファイルが見つかりません: %VOICEVOX_PATH%\VOICEVOX\vv-engine\run.exe
)

echo.
echo [Python環境の確認]
python -c "import torch; print(' PyTorch:', torch.__version__)" 2>nul || echo " PyTorchのインポートに失敗"
python -c "import flask; print(' Flask:', flask.__version__)" 2>nul || echo " Flaskのインポートに失敗"

echo.
echo ========================================
echo   セットアップ完了！
echo ========================================
echo.
echo  おめでとうございます！セットアップが完了しました。
echo.
echo 次の手順:
echo 1. system_start.bat をダブルクリックして起動
echo 2. ブラウザで以下のURLにアクセス
echo http://localhost:5001
echo 3. 初音ミクとの会話をお楽しみください！
echo.
echo  ヒント:
echo - このフォルダにsystem_start.batのショートカットを作成すると便利です
echo - アイコンには navi_icon.ico を使用できます
echo - 次回以降の起動には、このファイルではなくsystem_start.batまたはショートカットを使ってください
echo.

REM "ショートカット関連の変数を事前に設定（IFブロック外）"
set "SHORTCUT_PATH=%CURRENT_DIR%Start_HatsuneMiku_System.lnk"
if exist "%CURRENT_DIR%navi_icon.ico" (
    set "ICON_PATH=%CURRENT_DIR%navi_icon.ico"
) else (
    set "ICON_PATH="
)

set /p "create_shortcut=このフォルダにショートカットを作成しますか？ (Y/N): "
if /i "%create_shortcut%"=="Y" (
    echo ショートカットを作成しています...
    echo 作成先: %SHORTCUT_PATH%
    echo.
    
    if not "%ICON_PATH%"=="" (
        echo アイコンファイル: %ICON_PATH%
        powershell -command "try { $WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%CURRENT_DIR%system_start.bat'; $Shortcut.WorkingDirectory = '%CURRENT_DIR%'; $Shortcut.IconLocation = '%ICON_PATH%'; $Shortcut.Save(); Write-Host 'SUCCESS' } catch { Write-Host 'ERROR:' $_.Exception.Message }" 2>nul
    ) else (
        echo アイコンファイルが見つからないため、標準アイコンで作成します
        powershell -command "try { $WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%CURRENT_DIR%system_start.bat'; $Shortcut.WorkingDirectory = '%CURRENT_DIR%'; $Shortcut.Save(); Write-Host 'SUCCESS' } catch { Write-Host 'ERROR:' $_.Exception.Message }" 2>nul
    )
    
    echo.
    echo ショートカット作成結果を確認しています...
    if exist "%SHORTCUT_PATH%" (
        echo  ショートカットが正常に作成されました
        echo   場所: %SHORTCUT_PATH%
        echo   ショートカットはデスクトップなどに移動しておくと便利です
        
        for %%F in ("%SHORTCUT_PATH%") do (
            echo   サイズ: %%~zF bytes
            echo   作成日時: %%~tF
        )
    ) else (
        echo  ショートカットの作成に失敗しました
        echo   確認場所: %SHORTCUT_PATH%
        echo   手動でショートカットを作成してください
    )
)

echo.
set /p "start_now=今すぐ初音ミクを起動しますか？ (Y/N): "
if /i "%start_now%"=="Y" (
    echo 初音ミクを起動しています...
    start "" "%CURRENT_DIR%system_start.bat"
)

echo.
echo セットアップツールを終了します。
pause
