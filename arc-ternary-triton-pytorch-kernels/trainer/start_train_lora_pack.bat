@echo off
rem LoRA training run for Flint on one Intel Arc GPU, with the fast path (folded multipliers + fused
rem RMSNorm/SwiGLU) and sequence packing. Restarts after each clean 2-hour stop or crash, resuming from the last save.
rem To stop cleanly: create an empty file named STOP in %RUN_DIR%.
rem
rem   FOLDED_GGUF  Flint GGUF with trained multipliers folded in (made by export_folded.py)
rem   TOKENIZER    tokenizer.json for Flint
rem   DATA_DIR     folder with val.jsonl (old-skills eval set)
rem   TRAIN_FILE   training jsonl
rem   VAL_NEW      optional eval jsonl for the new data (default: none)
rem   INIT_FROM    optional save.pt to start from (default: none)
rem   RUN_DIR      output folder for saves and logs
rem   PYTHON, VCVARS, LEVEL_ZERO_SDK, GPU_INDEX  as in start_train.bat
rem   MAXTOK       longest example in tokens (default 10240; falls back to 9216 after an out-of-memory crash)
if "%FOLDED_GGUF%"=="" set FOLDED_GGUF=PATH\TO\Flint-folded.gguf
if "%TOKENIZER%"==""   set TOKENIZER=PATH\TO\tokenizer.json
if "%DATA_DIR%"==""    set DATA_DIR=PATH\TO\data
if "%TRAIN_FILE%"==""  set TRAIN_FILE=PATH\TO\train.jsonl
if "%RUN_DIR%"==""     set RUN_DIR=PATH\TO\run
if "%PYTHON%"==""      set PYTHON=python
if "%VCVARS%"==""      set VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
if "%LEVEL_ZERO_SDK%"=="" set LEVEL_ZERO_SDK=PATH\TO\level-zero-win-sdk-1.35.0
if "%GPU_INDEX%"==""   set GPU_INDEX=0
if "%MAXTOK%"==""      set MAXTOK=10240

cd /d "%~dp0"
set ZE_AFFINITY_MASK=%GPU_INDEX%
set PYTORCH_ALLOC_CONF=expandable_segments:True
call "%VCVARS%" >nul
set LIB=%LEVEL_ZERO_SDK%\lib;%LIB%
set INCLUDE=%LEVEL_ZERO_SDK%\include;%INCLUDE%
set CC=cl
set OPT=
if not "%VAL_NEW%"=="" set OPT=--val-new "%VAL_NEW%" --val-new-examples 40 --val-new-every 500
if not "%INIT_FROM%"=="" set OPT=%OPT% --init-from "%INIT_FROM%"
set CRASHES=0
set LOG=%RUN_DIR%\trainer_console.log
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

:loop
if exist "%RUN_DIR%\STOP" (
  echo STOP file present in %RUN_DIR%. Exiting.
  exit /b 0
)
"%PYTHON%" -u flint_train_lora_pack.py --fast --pack --pack-max 1024 --pack-window 50 ^
  --gguf "%FOLDED_GGUF%" --tokenizer "%TOKENIZER%" --data "%DATA_DIR%" --train-file "%TRAIN_FILE%" %OPT% ^
  --out "%RUN_DIR%" --chunk 1024 --save-tokens 100000 --hours 2 --eval-every 500 --eval-examples 12 ^
  --max-example-tokens %MAXTOK% --moe-group 16 --freeze-mult ^
  --lora --lora-r 16 --lora-alpha 32 --lora-experts shared --lora-lr 1e-5 >> "%LOG%" 2>&1
if %ERRORLEVEL%==0 goto loop
set /a CRASHES+=1
findstr /i /c:"out of memory" /c:"UR_RESULT_ERROR_OUT_OF" /c:"OUT_OF_DEVICE_MEMORY" "%LOG%" >nul && set MAXTOK=9216
if %CRASHES% GEQ 6 (
  echo Too many crashes, giving up. >> "%LOG%"
  exit /b 1
)
timeout /t 60 /nobreak >nul
goto loop
