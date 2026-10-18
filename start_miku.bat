@echo off
rem
rem Watashinosekaito Hatsune Miku
rem Copyright (c) 2025-2026 ChordDotE. All rights reserved.
rem SPDX-License-Identifier: LicenseRef-ChordDotE-NonCommercial-2.0
rem
rem NON-COMMERCIAL USE ONLY. NO BUSINESS USE. NO ADAPTATION.
rem
rem Business use (any use by or for a company, organization, or sole
rem proprietor, WHETHER OR NOT ANY PAYMENT IS INVOLVED) and publishing or
rem distributing modified versions are PROHIBITED IN ALL CASES; no permission
rem is granted for them. Forks may be used only to propose improvements to the
rem Author (e.g. pull requests). Commercial use, redistribution, and Adaptation
rem are PROHIBITED without prior written permission. "Adaptation" includes
rem manual reimplementation, automated translation, and generation using large
rem language models or other machine learning, of software having equivalent or
rem similar functionality. Permission is granted individually, in recorded form.
rem Contact: https://x.com/ChordDotE
rem
rem 業務利用（対価の有無を問わず、法人・団体・個人事業主による利用と事業のための利用）と、
rem 改変版の公開・配布は、許可の余地なく禁止します。フォークは作者への改善の提案
rem （プルリクエスト等）のためだけに使えます。商用利用・再配布・翻案は、作者の許可が
rem ない限り禁止します。翻案には、再実装・自動変換・大規模言語モデル等を用いた生成に
rem よる、同等または類似の機能を持つソフトウェアの作成を含みます。
rem 許可が必要な場合は、記録の残る形でご連絡ください。
rem
rem See LICENSE for full terms. 詳細は LICENSE を参照してください。
rem

REM Thin launcher: opens the control panel (start/stop GUI).
REM Keep this file ASCII-only: cmd mis-parses multibyte comments after chcp.
if not exist "%~dp0venv\Scripts\pythonw.exe" (
    echo Python environment not found. Please run the installer (MikuAgentSetup.exe) again.
    pause
    exit /b 1
)
REM The package lives in src\control_panel, so run from src.
REM Use start_panel.py instead of "-m control_panel": the protected build compiles
REM the package to a .pyd, and "python -m <package>" does not work on a .pyd.
cd /d "%~dp0src"
start "" "%~dp0venv\Scripts\pythonw.exe" start_panel.py
