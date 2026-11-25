@echo off
setlocal
set "DXC="
for /f "delims=" %%F in ('where dxc.exe 2^>nul') do (
    set "DXC=%%F"
    goto :found
)
for /f "delims=" %%D in ('dir /b /ad /o-n "C:\Program Files (x86)\Windows Kits\10\bin\10.*" 2^>nul') do (
    if exist "C:\Program Files (x86)\Windows Kits\10\bin\%%D\x64\dxc.exe" (
        set "DXC=C:\Program Files (x86)\Windows Kits\10\bin\%%D\x64\dxc.exe"
        goto :found
    )
)
echo ERROR: dxc.exe not found in PATH or Windows 10 SDK.
exit /b 1

:found
echo Using DXC: "%DXC%"
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0bcpack_decode.dxil" "%~dp0bcpack_decode.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0bcpack_bc7_rans.dxil" "%~dp0bcpack_bc7_rans.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0bc16_pack.dxil" "%~dp0bc16_pack.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0gfx10_detile.dxil" "%~dp0gfx10_detile.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0bindless_paging.dxil" "%~dp0bindless_paging.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0crowd_skin.dxil" "%~dp0crowd_skin.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0hiz_build.dxil" "%~dp0hiz_build.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0hiz_cull.dxil" "%~dp0hiz_cull.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0aaa_vrs.dxil" "%~dp0aaa_vrs.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0aaa_rtgi_cs.dxil" "%~dp0aaa_rtgi_cs.hlsl" || exit /b 1
echo SDK DXIL shaders compiled successfully.
