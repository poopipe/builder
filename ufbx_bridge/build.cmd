@echo off
setlocal EnableExtensions

rem Build ufbx_bridge.dll for Windows (cmd.exe).
rem Prefer cl (MSVC). Falls back to gcc if available.

rem %~dp0 has a trailing backslash; strip it so quoted -I"%ROOT%" does not
rem turn into -I"path\" and eat the rest of the command line.
set ROOT=%~dp0
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set OUT=%ROOT%\build
set UFBX_DIR=%ROOT%\ufbx
set BRIDGE=%ROOT%\ufbx_bridge.c

if not exist "%OUT%" mkdir "%OUT%"
if not exist "%UFBX_DIR%" mkdir "%UFBX_DIR%"

if not exist "%UFBX_DIR%\ufbx.c" (
  echo Downloading ufbx.c / ufbx.h ...
  curl -fsSL -o "%UFBX_DIR%\ufbx.c" https://raw.githubusercontent.com/ufbx/ufbx/master/ufbx.c
  if errorlevel 1 goto :download_fail
  curl -fsSL -o "%UFBX_DIR%\ufbx.h" https://raw.githubusercontent.com/ufbx/ufbx/master/ufbx.h
  if errorlevel 1 goto :download_fail
)

where cl >nul 2>nul
if %ERRORLEVEL%==0 goto :msvc

where gcc >nul 2>nul
if %ERRORLEVEL%==0 goto :gcc

echo Neither cl.exe nor gcc was found on PATH.
echo Open an "x64 Native Tools Command Prompt for VS" and re-run this script.
exit /b 1

:msvc
echo Building with MSVC...
cl /nologo /O2 /LD /DUFBX_BRIDGE_EXPORTS ^
  /I"%ROOT%" /I"%UFBX_DIR%" ^
  "%BRIDGE%" "%UFBX_DIR%\ufbx.c" ^
  /Fe"%OUT%\ufbx_bridge.dll" /Fo"%OUT%\\"
if errorlevel 1 exit /b 1
echo Built %OUT%\ufbx_bridge.dll
exit /b 0

:gcc
echo Building with gcc...
gcc -O2 -shared -DUFBX_BRIDGE_EXPORTS ^
  -I"%ROOT%" -I"%UFBX_DIR%" ^
  "%BRIDGE%" "%UFBX_DIR%\ufbx.c" ^
  -o "%OUT%\ufbx_bridge.dll"
if errorlevel 1 exit /b 1
echo Built %OUT%\ufbx_bridge.dll
exit /b 0

:download_fail
echo Failed to download ufbx sources. Check network / curl.
exit /b 1
