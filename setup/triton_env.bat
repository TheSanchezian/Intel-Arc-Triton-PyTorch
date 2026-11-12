@echo off
rem Environment for compiling Triton kernels on Windows (triton-xpu). Run with "call setup\triton_env.bat",
rem then start python in the same cmd window. Edit the two paths or set them beforehand.
rem   VCVARS          Visual Studio 2022 vcvars64.bat
rem   LEVEL_ZERO_SDK  unpacked level-zero-win-sdk-1.35.0 (contains lib\ze_loader.lib and include\level_zero\ze_api.h)
if "%VCVARS%"=="" set VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
if "%LEVEL_ZERO_SDK%"=="" set LEVEL_ZERO_SDK=PATH\TO\level-zero-win-sdk-1.35.0
call "%VCVARS%" >nul
set LIB=%LEVEL_ZERO_SDK%\lib;%LIB%
set INCLUDE=%LEVEL_ZERO_SDK%\include;%INCLUDE%
set CC=cl
rem optional: pick one GPU when several are installed
rem set ZE_AFFINITY_MASK=0
rem recommended for training runs
set PYTORCH_ALLOC_CONF=expandable_segments:True
echo Triton env ready: CC=%CC%, Level Zero SDK=%LEVEL_ZERO_SDK%
