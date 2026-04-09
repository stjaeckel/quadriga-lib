@echo off
setlocal enabledelayedexpansion
for /f "tokens=3 delims= " %%A in ('findstr "QUADRIGA_LIB_VERSION_STR" include\quadriga_lib.hpp') do (
    set V=%%A
    set V=!V:"=!
    echo QUADRIGA_VERSION=!V!> version.tmp
)