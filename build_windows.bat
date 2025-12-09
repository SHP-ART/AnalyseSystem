@echo off
REM Build-Skript für Windows
REM Erstellt eine ausführbare .exe Datei ohne Python-Abhängigkeit

echo ========================================
echo Teilenummer-Analyse - Windows Build
echo ========================================
echo.

REM Prüfe ob Python installiert ist
where python >nul 2>&1
if errorlevel 1 (
    echo FEHLER: Python ist nicht installiert oder nicht im PATH.
    echo Bitte installiere Python von https://python.org
    pause
    exit /b 1
)

echo Python gefunden: 
python --version
echo.

REM Installiere PyInstaller falls nicht vorhanden
echo Installiere/Aktualisiere PyInstaller und Abhängigkeiten...
python -m pip install --upgrade pip
python -m pip install pyinstaller matplotlib --upgrade

echo.
echo Erstelle ausführbare Datei...
echo.

REM Erstelle die .exe mit Icon-Unterstützung (falls icon.ico vorhanden)
if exist icon.ico (
    python -m PyInstaller --onefile --windowed --name "Teilenummer-Analyse" --icon=icon.ico teilenummer_analyse.py
) else (
    python -m PyInstaller --onefile --windowed --name "Teilenummer-Analyse" teilenummer_analyse.py
)

echo.
echo ========================================
if exist "dist\Teilenummer-Analyse.exe" (
    echo BUILD ERFOLGREICH!
    echo.
    echo Die ausführbare Datei befindet sich in:
    echo   dist\Teilenummer-Analyse.exe
    echo.
    echo Diese Datei kann ohne Python-Installation
    echo auf jedem Windows-PC ausgeführt werden.
) else (
    echo BUILD FEHLGESCHLAGEN!
    echo Bitte überprüfe die Fehlermeldungen oben.
)
echo ========================================
echo.
pause
