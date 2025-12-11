# Release Notes - Teilenummer-Analysesystem

## Version 1.0.0 (10. Dezember 2025)

### ✨ Neue Features
- **Windows EXE-Datei**: Eigenständige ausführbare Datei ohne Python-Installation erforderlich
- **Beispieldaten**: Beispieldatei `beispiel_daten.txt` zum Testen hinzugefügt
- **Verbesserte Analyse**: Erweiterte statistische Auswertungen
- **Grafische Darstellung**: Matplotlib-Integration für Visualisierungen
- **SQLite-Support**: Unterstützung für sehr große Datenmengen

### 🎯 Hauptfunktionen
- Statistische Analyse von Teilenummern aus DMS Loco-Soft Dateien
- Top-Listen nach Vorgängen, Menge oder Umsatz
- Suchfunktion für Teilenummern und Kunden
- CSV-Export für Excel
- Zeitraum-Filter für gezielte Analysen
- Fortschrittsanzeige beim Import großer Dateien

### 📦 Installation
**Windows**: Einfach die EXE-Datei herunterladen und starten - keine Installation erforderlich!

**macOS/Linux**: Python 3.8+ erforderlich, siehe README für Details

### 🔧 Technische Details
- Python 3.14.2
- PyInstaller 6.17.0
- Matplotlib für Grafiken
- SQLite für große Datenmengen
- Tkinter GUI

### 📥 Download
Die ausführbare Windows-Version ist als Asset in diesem Release verfügbar.

### 🐛 Bekannte Einschränkungen
- Windows-EXE ist ca. 41 MB groß (enthält komplettes Python + Bibliotheken)
- Erste Ausführung kann etwas länger dauern (Windows Defender Scan)

### 💡 Hinweise
- Die EXE-Datei ist Code-signiert und sicher
- Für optimale Performance empfohlen: Windows 10/11, 4 GB RAM
- Bei sehr großen Dateien (>100.000 Zeilen) wird automatisch SQLite verwendet
