@echo off
REM Windows. Нужен Docker Desktop. Использование: run.bat [synth^|demo^|classic gs^|test^|build] [доп. аргументы]
chcp 65001 >nul
cd /d "%~dp0"
set IMAGE=doe-nn
where docker >nul 2>&1 || (echo Docker не найден: https://docs.docker.com/desktop/install/windows-install/ & exit /b 1)
if /I "%1"=="build" goto build
docker image inspect %IMAGE% >nul 2>&1 || goto build
goto run
:build
docker build -t %IMAGE% . || exit /b 1
if /I "%1"=="build" exit /b 0
:run
if not exist input mkdir input
if not exist output mkdir output
docker run --rm -v "%cd%\input:/data/input:ro" -v "%cd%\output:/data/output" %IMAGE% %*
