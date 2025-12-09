#!/bin/bash
# Build-Skript für macOS
# Erstellt eine ausführbare .app Datei ohne Python-Abhängigkeit

echo "========================================"
echo "Teilenummer-Analyse - macOS Build"
echo "========================================"
echo ""

# Prüfe ob Python installiert ist
if ! command -v python3 &> /dev/null; then
    echo "FEHLER: Python3 ist nicht installiert."
    echo "Bitte installiere Python: brew install python"
    exit 1
fi

# Installiere PyInstaller falls nicht vorhanden
echo "Installiere/Aktualisiere PyInstaller..."
pip3 install pyinstaller --upgrade

echo ""
echo "Erstelle ausführbare Datei..."
echo ""

# Erstelle die .app
pyinstaller --onefile --windowed --name "Teilenummer-Analyse" teilenummer_analyse.py

echo ""
echo "========================================"
if [ -f "dist/Teilenummer-Analyse" ]; then
    echo "BUILD ERFOLGREICH!"
    echo ""
    echo "Die ausführbare Datei befindet sich in:"
    echo "  dist/Teilenummer-Analyse"
    echo ""
    echo "Diese Datei kann ohne Python-Installation"
    echo "auf jedem Mac ausgeführt werden."
else
    echo "BUILD FEHLGESCHLAGEN!"
    echo "Bitte überprüfe die Fehlermeldungen oben."
fi
echo "========================================"
