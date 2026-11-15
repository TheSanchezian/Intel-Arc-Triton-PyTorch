@echo off
rem Flint multiplier training split across TWO Intel Arc GPUs (model split, not data parallel):
rem layers 0..K-1 + embeddings on xpu:0, layers K..48 + final norm/head/loss on xpu:1.
rem Purpose: longer training examples (measured up to 16384 tokens at ~8.4 GB peak per A770), same speed as one GPU.
rem Saves use the same format as the single-GPU trainer, so a run can switch between the two.
rem Never run the single-GPU and split trainers on the same run folder at the same time.
rem To stop cleanly: create an empty file named STOP in %RUN_DIR% (the trainer saves, then everything exits).
rem
rem   FLINT_GGUF  Flint-40B PQ2_0 GGUF file
rem   TOKENIZER   tokenizer.json for Flint
rem   DATA_DIR    folder with train.jsonl / val.jsonl
rem   RUN_DIR     output folder for saves and logs
rem   PYTHON      python.exe of the venv with torch+xpu and triton-xpu installed
rem   VCVARS      path to vcvars64.bat of Visual Studio 2022
rem   LEVEL_ZERO_SDK  folder of the unpacked Level Zero SDK (contains lib\ and include\)
rem   MAXTOK      max tokens per training example (default 16384)
rem   SPLIT_AT    first layer on the second GPU (0 = automatic balance, measured K=27)
if "%FLINT_GGUF%"=="" set FLINT_GGUF=PATH\TO\Ternary-Flint-40B-PQ2_0.gguf
if "%TOKENIZER%"==""  set TOKENIZER=PATH\TO\tokenizer.json
if "%DATA_DIR%"==""   set DATA_DIR=PATH\TO\data
if "%RUN_DIR%"==""    set RUN_DIR=PATH\TO\run
if "%PYTHON%"==""     set PYTHON=python
if "%VCVARS%"==""     set VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
if "%LEVEL_ZERO_SDK%"=="" set LEVEL_ZERO_SDK=PATH\TO\level-zero-win-sdk-1.35.0
if "%MAXTOK%"==""     set MAXTOK=16384
if "%SPLIT_AT%"==""   set SPLIT_AT=0

if exist "%RUN_DIR%\STOP" (
  echo STOP file present in %RUN_DIR% - delete it to train. Exiting.
  exit /b 0
)
cd /d "%~dp0"
set PYTORCH_ALLOC_CONF=expandable_segments:True
call "%VCVARS%" >nul
set LIB=%LEVEL_ZERO_SDK%\lib;%LIB%
set INCLUDE=%LEVEL_ZERO_SDK%\include;%INCLUDE%
set CC=cl
if "%MG%"=="" set MG=16
"%PYTHON%" flint_supervisor_split.py --gpus 0,1 --out "%RUN_DIR%" -- ^
  --gguf "%FLINT_GGUF%" ^
  --tokenizer "%TOKENIZER%" ^
  --data "%DATA_DIR%" ^
  --chunk 1024 --lr 2e-4 --save-tokens 2500 --hours 2 ^
  --eval-now --eval-every 500 --eval-examples 12 --max-example-tokens %MAXTOK% --moe-group %MG% ^
  --devices xpu:0,xpu:1 --split-at %SPLIT_AT%
