@echo off
rem Long Flint multiplier-training run on one Intel Arc GPU (Windows, PyTorch XPU + Triton).
rem Fill in the variables below (or set them in the environment before calling this script).
rem To stop cleanly: create an empty file named STOP in %RUN_DIR% (the trainer saves, then everything exits).
rem
rem   FLINT_GGUF  Flint-40B PQ2_0 GGUF file
rem   TOKENIZER   tokenizer.json for Flint
rem   DATA_DIR    folder with train.jsonl / val.jsonl
rem   RUN_DIR     output folder for saves and logs
rem   PYTHON      python.exe of the venv with torch+xpu and triton-xpu installed
rem   VCVARS      path to vcvars64.bat of Visual Studio 2022
rem   LEVEL_ZERO_SDK  folder of the unpacked Level Zero SDK (contains lib\ and include\)
rem   GPU_INDEX   GPU index for ZE_AFFINITY_MASK (default 0)
rem   EXTRA_JSONL optional extra training data in the same format (default: none)
if "%FLINT_GGUF%"=="" set FLINT_GGUF=PATH\TO\Ternary-Flint-40B-PQ2_0.gguf
if "%TOKENIZER%"==""  set TOKENIZER=PATH\TO\tokenizer.json
if "%DATA_DIR%"==""   set DATA_DIR=PATH\TO\data
if "%RUN_DIR%"==""    set RUN_DIR=PATH\TO\run
if "%PYTHON%"==""     set PYTHON=python
if "%VCVARS%"==""     set VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
if "%LEVEL_ZERO_SDK%"=="" set LEVEL_ZERO_SDK=PATH\TO\level-zero-win-sdk-1.35.0
if "%GPU_INDEX%"==""  set GPU_INDEX=0

if exist "%RUN_DIR%\STOP" (
  echo STOP file present in %RUN_DIR% - delete it to train. Exiting.
  exit /b 0
)
cd /d "%~dp0"
rem one growable memory pool instead of scattered blocks (stops the reserved-memory creep)
set PYTORCH_ALLOC_CONF=expandable_segments:True
rem Triton needs the MSVC compiler and the Level Zero SDK (ze_loader.lib, level_zero\ze_api.h)
call "%VCVARS%" >nul
set LIB=%LEVEL_ZERO_SDK%\lib;%LIB%
set INCLUDE=%LEVEL_ZERO_SDK%\include;%INCLUDE%
set CC=cl
rem experts per batched launch (16 was the best measured value)
if "%MG%"=="" set MG=16
set EXTRA=
if not "%EXTRA_JSONL%"=="" set EXTRA=--extra "%EXTRA_JSONL%"
"%PYTHON%" flint_supervisor.py --gpu %GPU_INDEX% --out "%RUN_DIR%" -- ^
  --gguf "%FLINT_GGUF%" ^
  --tokenizer "%TOKENIZER%" ^
  --data "%DATA_DIR%" %EXTRA% ^
  --chunk 1024 --lr 2e-4 --save-tokens 2500 --hours 2 ^
  --eval-now --eval-every 100 --eval-examples 12 --max-example-tokens 4096 --moe-group %MG%
