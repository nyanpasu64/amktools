@echo off
rem pyinstaller -y amktools\mmkparser.py
pyinstaller -y mmkparser.spec
dist\mmkparser\mmkparser.exe

rem pyinstaller -y amktools\wav2brr\__main__.py -n wav2brr
pyinstaller -y wav2brr.spec
dist\wav2brr\wav2brr.exe
