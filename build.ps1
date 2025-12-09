# PowerShell Build-Skript für Teilenummer-Analyse
# Erstellt eine ausführbare .exe Datei

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Teilenummer-Analyse - Windows Build" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Prüfe Python
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "Python gefunden: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "FEHLER: Python ist nicht installiert oder nicht im PATH." -ForegroundColor Red
    Write-Host "Bitte installiere Python von https://python.org" -ForegroundColor Yellow
    Read-Host "Drücke Enter zum Beenden"
    exit 1
}

Write-Host ""
Write-Host "Installiere/Aktualisiere benötigte Pakete..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install pyinstaller matplotlib --upgrade

Write-Host ""
Write-Host "Erstelle ausführbare Datei..." -ForegroundColor Yellow
Write-Host ""

# Lösche alte Build-Dateien
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "*.spec") { Remove-Item -Force "*.spec" }

# Erstelle die .exe
$iconParam = if (Test-Path "icon.ico") { "--icon=icon.ico" } else { "" }

if ($iconParam) {
    python -m PyInstaller --onefile --windowed --name "Teilenummer-Analyse" $iconParam teilenummer_analyse.py
} else {
    python -m PyInstaller --onefile --windowed --name "Teilenummer-Analyse" teilenummer_analyse.py
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
if (Test-Path "dist\Teilenummer-Analyse.exe") {
    Write-Host "BUILD ERFOLGREICH!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Die ausführbare Datei befindet sich in:" -ForegroundColor Green
    Write-Host "  dist\Teilenummer-Analyse.exe" -ForegroundColor White
    Write-Host ""
    Write-Host "Diese Datei kann ohne Python-Installation" -ForegroundColor Cyan
    Write-Host "auf jedem Windows-PC ausgeführt werden." -ForegroundColor Cyan
    
    # Zeige Dateigröße
    $fileSize = (Get-Item "dist\Teilenummer-Analyse.exe").Length / 1MB
    Write-Host ""
    Write-Host "Dateigröße: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Yellow
} else {
    Write-Host "BUILD FEHLGESCHLAGEN!" -ForegroundColor Red
    Write-Host "Bitte überprüfe die Fehlermeldungen oben." -ForegroundColor Yellow
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Drücke Enter zum Beenden"
