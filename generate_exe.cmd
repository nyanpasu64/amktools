@echo off
rmdir /s /q dist\amktools
mkdir dist\amktools

rem TODO rewrite in bash or powershell or python

pyinstaller -y mmkparser.spec
xcopy /y /s /e dist\mmkparser dist\amktools\
dist\amktools\mmkparser.exe

pyinstaller -y wav2brr.spec
xcopy /y /s /e dist\wav2brr dist\amktools\
dist\amktools\wav2brr.exe
