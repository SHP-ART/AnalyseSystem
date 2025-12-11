# Teilenummer-Analysesystem

Ein plattformübergreifendes Analysewerkzeug für Windows und macOS zur statistischen Auswertung von Teilenummern aus DMS Loco-Soft Textdateien.

## 💾 Download

### Windows (keine Python-Installation erforderlich!)

**[⬇️ Teilenummer-Analyse.exe v1.1.0 herunterladen](https://github.com/SHP-ART/AnalyseSystem/releases/download/v1.1.0/Teilenummer-Analyse.exe)** (38,92 MB)

Die EXE-Datei ist eigenständig und läuft auf jedem Windows-PC ohne zusätzliche Installation.

> **Neu in v1.1.0:** Zeitraum-Filter für Lagerhaltung-Analyse (1-12 Monate auswählbar)

### macOS / Linux / Python-Version

Siehe Installationsanleitung unten für die Python-basierte Version.

## Features

- 📊 **Statistische Analyse** - Zählt die Häufigkeit jeder Teilenummer
- 🔝 **Top-Listen** - Zeigt die am häufigsten verwendeten Teile
- 📈 **Mehrere Sortieroptionen** - Nach Vorgängen, Menge oder Umsatz
- 🔍 **Suchfunktion** - Schnelles Finden von Teilenummern oder Kunden
- 💾 **CSV-Export** - Exportiert Ergebnisse für Excel
- 📈 **Grafische Auswertungen** - Visualisierung mit Matplotlib
- 🖥️ **Plattformübergreifend** - Läuft auf Windows und macOS
- 🗄️ **SQLite-Unterstützung** - Für sehr große Datenmengen

## Schnellstart (Windows EXE)

1. **[Teilenummer-Analyse.exe v1.1.0 herunterladen](https://github.com/SHP-ART/AnalyseSystem/releases/download/v1.1.0/Teilenummer-Analyse.exe)**
2. **Doppelklick auf die heruntergeladene Datei**
3. **Fertig!** - Keine Installation, kein Python erforderlich

## Voraussetzungen (nur für Python-Version)

- Python 3.8 oder höher
- tkinter (in den meisten Python-Installationen enthalten)
- matplotlib (für Grafiken)

## Installation

### macOS
```bash
# Python installieren (falls nicht vorhanden)
brew install python

# In das Projektverzeichnis wechseln
cd /pfad/zu/AnalyseSystem

# Anwendung starten
python3 teilenummer_analyse.py
```

### Windows
```bash
# Python von python.org herunterladen und installieren
# Stelle sicher, dass "Add Python to PATH" aktiviert ist

# In das Projektverzeichnis wechseln
cd C:\pfad\zu\AnalyseSystem

# Anwendung starten
python teilenummer_analyse.py
```

## Verwendung

1. **Anwendung starten**: Doppelklick auf `teilenummer_analyse.py` oder Start über Terminal
2. **Datei öffnen**: Klicke auf "Datei öffnen..." oder nutze Cmd+O (macOS) / Strg+O (Windows)
3. **Analyse ansehen**: 
   - **Top Teilenummern** - Zeigt die meistverwendeten Teile
   - **Alle Daten** - Durchsuche alle Datensätze
   - **Zusammenfassung** - Detaillierter Analysebericht
4. **Exportieren**: Klicke auf "Exportieren..." um die Ergebnisse als CSV zu speichern

## Dateiformat

Das System erwartet Textdateien im DMS Loco-Soft Format mit Semikolon-Trennung:

```
DMS: Loco-Soft ;Händlernummer: 1710; Fakturierzeitraum: 01.10.25 - 99.99.99; ET-Nr.;Bezeichnung;...
016393;MOTOROELABLASS-SCHRAUBE;80/2,04;10.10.2025;...
016488;DICHTUNG ZYLINDERBLOCKSTOPFEN;4/1,02;01.10.2025;...
```

### Unterstützte Spalten

| Spalte | Beschreibung |
|--------|--------------|
| ET-Nr. | Teilenummer (wird analysiert) |
| Bezeichnung | Teilebeschreibung |
| Auftrag | Auftragsnummer |
| Abg.Datum | Abgabedatum |
| Menge | Verkaufte Menge |
| VK-Preis | Verkaufspreis |
| Kd.-Name | Kundenname |
| ... | weitere Felder |

## Screenshots

### Hauptansicht - Top Teilenummern
Die Anwendung zeigt eine sortierbare Liste der am häufigsten verwendeten Teilenummern.

### Zusammenfassung
Ein detaillierter Textbericht mit allen wichtigen Statistiken.

## Export

Der CSV-Export enthält:
- Teilenummer
- Bezeichnung
- Anzahl Vorgänge
- Gesamtmenge
- Gesamtumsatz
- Anzahl verschiedener Kunden

Die CSV-Datei verwendet Semikolon als Trennzeichen und ist direkt in Excel importierbar.

## Selbst eine EXE erstellen (für Entwickler)

### Windows

```powershell
# Methode 1: PowerShell-Skript
.\build.ps1

# Methode 2: Batch-Skript
.\build_windows.bat

# Methode 3: Manuell
python -m pip install pyinstaller matplotlib
python -m PyInstaller --onefile --windowed --name "Teilenummer-Analyse" teilenummer_analyse.py
```

Die fertige EXE befindet sich dann im `dist\`-Ordner.

### macOS

```bash
./build_macos.sh
```

## Technische Details

- **Python Version**: 3.14.2
- **Hauptbibliotheken**: tkinter, matplotlib, sqlite3
- **Build-Tool**: PyInstaller 6.17.0
- **Dateigröße (EXE)**: ~41 MB (enthält Python + alle Bibliotheken)

## Lizenz

MIT License - Frei verwendbar für private und kommerzielle Zwecke.

## Support

Bei Fragen oder Problemen erstellen Sie bitte ein Issue im Repository.
