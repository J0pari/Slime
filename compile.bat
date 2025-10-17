@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
nvcc -arch=sm_86 -std=c++17 -rdc=true -lcudadevrt -O3 --use_fast_math --expt-relaxed-constexpr main.cu -o slime.exe
echo Compilation complete!