@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "LAUNCHER_DIR=%~dp0"
set "LOG_FILE=%TEMP%\ecat-workbench-launch.log"
set "MPLCONFIGDIR=%TEMP%\ecat-matplotlib"

if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%" >nul 2>nul
echo eCAT Workbench launch started. > "%LOG_FILE%"
echo Launcher directory: %LAUNCHER_DIR% >> "%LOG_FILE%"
echo Matplotlib config: %MPLCONFIGDIR% >> "%LOG_FILE%"

where ecat-app >nul 2>nul
if not errorlevel 1 (
  echo Trying installed ecat-app command. >> "%LOG_FILE%"
  ecat-app --port 0 >> "%LOG_FILE%" 2>&1
  if not errorlevel 1 exit /b 0
  echo Installed ecat-app failed with status %ERRORLEVEL%. >> "%LOG_FILE%"
)

set "SEARCH_DIR=%LAUNCHER_DIR:~0,-1%"
:find_repo
if exist "%SEARCH_DIR%\apps\workbench\app.py" (
  set "REPO_ROOT=%SEARCH_DIR%"
  goto repo_found
)
for %%I in ("%SEARCH_DIR%\..") do set "PARENT_DIR=%%~fI"
if "%PARENT_DIR%"=="%SEARCH_DIR%" goto repo_missing
set "SEARCH_DIR=%PARENT_DIR%"
goto find_repo

:repo_found
echo Repository root: %REPO_ROOT% >> "%LOG_FILE%"

if defined ECAT_PYTHON (
  call :try_python "%ECAT_PYTHON%"
  if not errorlevel 1 exit /b 0
)

call :try_python "%REPO_ROOT%\.venv\Scripts\python.exe"
if not errorlevel 1 exit /b 0
call :try_python "%REPO_ROOT%\venv\Scripts\python.exe"
if not errorlevel 1 exit /b 0
call :try_python python
if not errorlevel 1 exit /b 0
call :try_python py -3
if not errorlevel 1 exit /b 0
call :try_python python3
if not errorlevel 1 exit /b 0

echo Repository launcher failed. >> "%LOG_FILE%"
goto failure

:repo_missing
echo Could not find repository root. >> "%LOG_FILE%"
goto failure

:try_python
echo Checking Python candidate: %* >> "%LOG_FILE%"
%* -c "import importlib.util, sys; missing=[m for m in ('numpy','dash','dash_ag_grid','webview') if importlib.util.find_spec(m) is None]; print('Missing modules: ' + ', '.join(missing)) if missing else None; sys.exit(1 if missing else 0)" >> "%LOG_FILE%" 2>&1
if errorlevel 1 exit /b 1
echo Launching with %* >> "%LOG_FILE%"
%* "%REPO_ROOT%\apps\workbench\app.py" --port 0 >> "%LOG_FILE%" 2>&1
exit /b %ERRORLEVEL%

:failure
call :write_install_instructions
call :show_failure_popup
echo eCAT Workbench could not start.
echo Log: %LOG_FILE%
pause
exit /b 1

:write_install_instructions
echo. >> "%LOG_FILE%"
echo Install or update the app dependencies from the eCAT repository folder: >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"
if defined REPO_ROOT echo cd /d "%REPO_ROOT%" >> "%LOG_FILE%"
echo python -m pip install -e . >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"
echo If you use a specific Python, set ECAT_PYTHON to that interpreter. >> "%LOG_FILE%"
exit /b 0

:show_failure_popup
powershell -NoProfile -ExecutionPolicy Bypass -Command "$repo=$env:REPO_ROOT; $lines=@('Install or update the app dependencies from the eCAT repository folder:',''); if ($repo) { $lines += 'cd /d \"' + $repo + '\"' }; $lines += 'python -m pip install -e .'; $lines += ''; $lines += 'Log: ' + $env:LOG_FILE; $lines += ''; $lines += 'If you use a specific Python, set ECAT_PYTHON to that interpreter.'; Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show(($lines -join [Environment]::NewLine), 'eCAT Workbench could not start', 'OK', 'Error')" >nul 2>nul
exit /b 0
