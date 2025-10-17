@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
nvcc -arch=sm_86 -std=c++17 test_minimal.cu -o test.exe
test.exe