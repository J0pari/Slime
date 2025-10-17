@echo on
set PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\Installer
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
nvcc -arch=sm_86 -std=c++17 -rdc=true -lcudadevrt -O3 --use_fast_math --expt-relaxed-constexpr -Ic:/Slime/cuda_patches/include --include-path c:/Slime/cuda_patches/include main.cu -o slime.exe 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Compilation FAILED with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo Compilation complete!