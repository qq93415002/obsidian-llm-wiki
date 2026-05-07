@echo off
chcp 65001 >nul

set "OLW_VAULT=D:\KNOWLEGE\wiki"
set "OLW_API_KEY=sk-cp-SZKQUWlP9Zceww2G67b1xa7y9djCykErpd9wTl1Q38kNv13wZlTRc4U_qGoiNKE-fmlZK9-oZqTkHk48eL3S2tuXkifrxSiC5ni3DEvS7l7rZWBllY3AkCo"

echo ==============================================
echo  OBSIDIAN-LLM-WIKI Runner
echo ==============================================
echo  Vault: %OLW_VAULT%
echo  Provider: MiniMax
echo ==============================================
echo.

if "%1"=="" (
    echo Usage: run_olw.bat [command]
    echo.
    echo Commands:
    echo   run          - Run full pipeline (ingest + compile)
    echo   status       - Show vault status
    echo   review       - Review drafts interactively
    echo   approve      - Approve all drafts
    echo   query "text" - Query the wiki
    echo   lint         - Check wiki health
    echo   maintain     - Maintain wiki
    echo   watch        - Watch for changes
    echo.
    pause
    exit /b
)

if "%1"=="run" (
    echo Running pipeline...
    olw run
) else if "%1"=="status" (
    echo Checking status...
    olw status
) else if "%1"=="review" (
    echo Starting review...
    olw review
) else if "%1"=="approve" (
    echo Approving all drafts...
    olw approve --all
) else if "%1"=="query" (
    echo Querying: %2
    olw query "%2"
) else if "%1"=="lint" (
    echo Running lint...
    olw lint
) else if "%1"=="maintain" (
    echo Running maintenance...
    olw maintain --fix
) else if "%1"=="watch" (
    echo Starting watcher...
    olw watch
) else (
    echo Unknown command: %1
    echo Type run_olw.bat for help
)

echo.
echo Press any key to exit...
pause >nul