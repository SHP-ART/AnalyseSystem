# PROJEKT.md – LagerPilot

## Projektname & Zweck

**LagerPilot** (ehem. Teilenummer-Analyse) – Desktop-Anwendung zur statistischen Auswertung
von Teilenummern aus DMS Loco-Soft Textdateien. Analysiert Verkaufsdaten, berechnet
Lagerhaltungskennzahlen (ABC-Analyse, Umschlagshäufigkeit, Bestellpunkt), erkennt
überschüssige Lagerbestände und exportiert Ergebnisse als CSV.

## Architektur-Übersicht

```
┌─────────────────────────────────────────────────────────┐
│  AnalyseApp (Tkinter GUI)                               │
│  ├── Tab: Top Teilenummern                              │
│  ├── Tab: Alle Daten                                    │
│  ├── Tab: Zusammenfassung                               │
│  ├── Tab: Lagerhaltung (ABC, Umschlag, Bestellpunkt)    │
│  ├── Tab: Lagerabbau (Überschüsse, Aktionsempfehlungen) │
│  ├── Tab: Grafiken (Matplotlib)                         │
│  └── Tab: Zeitraum-Analyse                              │
├─────────────────────────────────────────────────────────┤
│  TeilenummerStatistik (Analyse-Engine)                  │
│  ├── In-Memory-Modus (kleine Dateien)                   │
│  └── SQLite-Modus (große Dateien >100K Records)         │
├─────────────────────────────────────────────────────────┤
│  TeilenummerParser (Streaming-Parser)                   │
│  └── DMS Loco-Soft ;getrennte Textdateien               │
├─────────────────────────────────────────────────────────┤
│  SQLiteDataStore (temporäre DB)                         │
│  └── Indexed queries, automatische Bereinigung          │
└─────────────────────────────────────────────────────────┘

Datenfluss:
  .txt Datei → Parser → [Memory | SQLite] → Statistik → GUI/Tabs → CSV-Export
  ausgabe.txt (Lagerbestand) → Parser → Lagerabbau-Analyse
```

## Technologie-Stack

| Technologie | Version | Zweck |
|---|---|---|
| Python | 3.8+ | Laufzeitumgebung |
| tkinter | Standard-Lib | GUI-Framework |
| matplotlib | (lazy-loaded) | Diagramme/Charts |
| sqlite3 | Standard-Lib | Temporäre DB für große Datenmengen |
| PyInstaller | 6.17.0 | Windows/macOS EXE-Build |
| Pillow | (optional) | Icon-Generierung |

## Kernkomponenten

### TeilenummerParser (`teilenummer_analyse.py:49`)
- **Verantwortung**: DMS Loco-Soft Dateien einlesen (Streaming)
- **Schnittstellen**: `parse_file()` mit progress_callback, record_callback
- **Besonderheiten**: Metadaten aus 1. Zeile, dt. Datums-/Zahlenformate

### SQLiteDataStore (`teilenummer_analyse.py:171`)
- **Verantwortung**: Temporäre SQLite-DB für große Datenmengen
- **Schnittstellen**: insert_record, fetch_records, get_top_n, get_summary, get_monthly_data
- **Besonderheiten**: Auto-Cleanup, indizierte Queries, WHERE-Klausel-Builder

### TeilenummerStatistik (`teilenummer_analyse.py:383`)
- **Verantwortung**: Statistische Analyse (abstrahiert Memory/SQLite)
- **Schnittstellen**: get_top_n, get_lagerhaltung_analyse, get_summary, get_teilenummer_details
- **Besonderheiten**: ABC-Klassifizierung (Pareto), Trendanalyse, Saisonalitätserkennung

### AnalyseApp (`teilenummer_analyse.py:1074`)
- **Verantwortung**: Tkinter GUI mit Tab-Notebook
- **Schnittstellen**: Datei-Dialog, Export, Keyboard-Shortcuts (Ctrl+O/E/S)
- **Besonderheiten**: Lazy Matplotlib, Autocomplete-Filter, Debounced Updates

### TreeviewHeaderTooltip (`teilenummer_analyse.py:979`)
- **Verantwortung**: Tooltips für Tabellen-Spaltenüberschriften

## Abhängigkeiten & Kopplungen

- **Intern**: Alle Klassen in einer Datei (`teilenummer_analyse.py`, 3400 Zeilen)
- **Extern**: matplotlib (lazy), tkinter, sqlite3, csv, collections, datetime
- **Dateisystem**: Temporäres SQLite-Verzeichnis (auto-cleanup)
- **DMS Loco-Soft**: Erwartetes Eingabeformat (Semikolon-getrennt, dt. Metadaten-Zeile)

## Datenmodell

### In-Memory (List of Dicts)
```python
{
  'teilenummer': str,
  'bezeichnung': str,
  'auftrag': str,
  'abgabe_datum': str,       # Original: DD.MM.YYYY
  'abgabe_iso': str,         # ISO: YYYY-MM-DD
  'menge': float,
  'vk_preis': float,
  'kunde_name': str,
  'kunde_nr': str,
  # ... weitere DMS-Felder
}
```

### SQLite-Tabelle `records`
Gleiche Felder wie In-Memory, indiziert auf `teilenummer` und `abgabe_iso`.

### Lagerbestand (ausgabe.txt)
Tab-getrennt, wird unter mehreren Keys gespeichert (ET-Nr., Lieferant-ET-Nr., mit/ohne führende Nullen) für maximales Matching.

## Konfigurationsparameter

| Parameter | Standort | Beschreibung |
|---|---|---|
| `MAX_TAGE_LOHNEND` | TeilenummerStatistik | Schwellwert für "lohnend" (Tage) |
| `SQLITE_THRESHOLD` | Code | Records/Switch zu SQLite (>100K) |
| Chart-Defaults | AnalyseApp | DPI, Farben, Font-Größen |
| Quick-Filter-Schwellen | AnalyseApp | Kritisch (>500€, >1J), Alte Teile (>2J) |

## TDD-Status

| Komponente | Tests vorhanden |
|---|---|
| TeilenummerParser | Nein |
| SQLiteDataStore | Nein |
| TeilenummerStatistik | Nein |
| AnalyseApp (GUI) | Nein |
| Gesamt | **Keine automatisierten Tests** |

## Bekannte Grenzen & offene Punkte

- Single-File-Architektur (3400 Zeilen) – müsste aufgeteilt werden
- Keine automatisierten Tests
- Kein requirements.txt
- Multi-Sort für Tabellen nicht implementiert
- Lagerabbau-Limit auf 500 Einträge
- `firebase-debug.log` im Repo (sollte bereinigt werden)

## Bekannte Fehler

Siehe `.claude/ERRORS.md`

## Projektstatus

- **Phase**: Wartung
- **Letzter Stand**: 2026-08-03 – 5 Berechnungsfehler in Lagerhaltung/Lagerabbau behoben (Monatsdurchschnitt, Trend, Saisonalität, SUM vs. Count)
- **Nächster Schritt**: Tests einrichten (pytest), erste Tests für Parser/Statistik/Lagerberechnungen schreiben
- **Blockiert durch**: nichts
