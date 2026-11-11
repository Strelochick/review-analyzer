@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 1) Create an EMPTY public repository on https://github.com/new (without README).
echo 2) Paste its URL below, for example https://github.com/yourname/review-analyzer.git
set /p URL=Repository URL: 
git add .
git commit -m "Update: improved sentiment model, report" 
git branch -M main
git remote remove origin 2>nul
git remote add origin %URL%
git push -u origin main
pause
