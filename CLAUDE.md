# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LagerPilot** (formerly Teilenummer-Analyse) is a cross-platform desktop application for analyzing part numbers from DMS Loco-Soft text files. Built with Python and Tkinter, it provides statistical analysis, visualization, and export capabilities for warehouse/inventory data.

**Primary Language**: German (UI, documentation, comments)
**Target Platforms**: Windows (primary, distributed as .exe), macOS, Linux
**Application Type**: Desktop GUI (Tkinter)

## Building the Application

### Windows Build
```powershell
# Option 1: PowerShell script
.\build.ps1

# Option 2: Batch script
.\build_windows.bat

# Option 3: Manual build
python -m pip install pyinstaller matplotlib
python -m PyInstaller --onefile --windowed --name "LagerPilot" --icon=icon.ico teilenummer_analyse.py
```

Output: `dist\LagerPilot.exe` (~38-41 MB, standalone executable)

### macOS Build
```bash
./build_macos.sh
```

Output: `dist/Teilenummer-Analyse` (standalone executable)

### Icon Generation
```bash
python create_icon.py
```

Creates `icon.ico` with a warehouse shelf icon design in multiple sizes (16x16 to 256x256).

## Running the Application

### From Source
```bash
# Windows
python teilenummer_analyse.py

# macOS/Linux
python3 teilenummer_analyse.py
```

### Dependencies
- Python 3.8+
- tkinter (usually bundled with Python)
- matplotlib (for charts/visualizations)
- sqlite3 (standard library, for large datasets)
- Pillow (for icon generation only)

## Architecture

### Core Components

**TeilenummerParser** (`teilenummer_analyse.py:49`)
- Parses DMS Loco-Soft semicolon-delimited text files
- Streaming parser with progress callback support
- Handles metadata extraction from first line (dealer number, billing period)
- Parses dates in German format (DD.MM.YYYY or DD.MM.YY)
- Converts monetary values (comma decimal separator to float)

**SQLiteDataStore** (`teilenummer_analyse.py:171`)
- Temporary SQLite database for large datasets (automatic mode selection)
- Enables analysis of files too large to fit in memory
- Created in temp directory, automatically cleaned up
- Provides indexed queries for performance

**TeilenummerStatistik** (`teilenummer_analyse.py:383`)
- Statistical analysis engine
- Supports both in-memory (list) and SQLite-backed data
- Provides filtering by date ranges (month, quarter, custom periods)
- Calculates:
  - Top N parts by operations, quantity, or revenue
  - Unique customer counts per part
  - Total quantities and revenue aggregations
  - Date range analysis

**AnalyseApp** (`teilenummer_analyse.py:1031`)
- Main Tkinter GUI application
- Tabbed interface:
  - **Top Teilenummern**: Sortable table of most-used parts
  - **Alle Daten**: Searchable complete dataset view
  - **Zusammenfassung**: Text report with statistics
  - **Lagerhaltung**: Inventory analysis with 1-12 month time filters
  - **Grafiken**: Matplotlib charts (bar charts, pie charts)
- Keyboard shortcuts: Ctrl+O (open), Ctrl+E (export), Ctrl+S (save)

### Data Flow

1. User selects `.txt` file via file dialog
2. `TeilenummerParser` streams file and extracts records
3. For large files (>100K records), data goes to `SQLiteDataStore`; otherwise kept in memory
4. `TeilenummerStatistik` provides analysis methods
5. GUI tabs display results via Tkinter widgets
6. Export to CSV uses semicolon delimiter (German Excel standard)

### File Format Expectations

DMS Loco-Soft format with first line containing metadata:
```
DMS: Loco-Soft ;Händlernummer: 1710; Fakturierzeitraum: 01.10.25 - 99.99.99; ET-Nr.;Bezeichnung;...
016393;MOTOROELABLASS-SCHRAUBE;80/2,04;10.10.2025;...
```

**Key Columns** (semicolon-separated):
- ET-Nr. (part number) - primary analysis key
- Bezeichnung (description)
- Abg.Datum (delivery date)
- Menge (quantity)
- VK-Preis (sales price)
- Kd.-Name (customer name)

## Key Features & Recent Changes

### Recent Version History
- **v1.1.0**: Time period filter for inventory analysis (1-12 months), renamed to LagerPilot, added custom icon
- **v1.0.0**: Initial release with Windows EXE, sortable columns, progress indicators, SQLite support

### Important Implementation Details

**Lazy Matplotlib Loading**: matplotlib is imported only when visualization tab is accessed to reduce startup time and memory usage.

**German Localization**: All UI strings, date formats, and number formats follow German conventions:
- Dates: DD.MM.YYYY
- Decimals: Comma separator (1,50 instead of 1.50)
- CSV: Semicolon delimiter for Excel compatibility

**Large File Handling**: Automatic switch to SQLite when file size or record count exceeds thresholds. Shows progress bar during import.

**Sortable Tables**: Click column headers to sort. Multi-column sort not implemented.

**Time Filters**: Lagerhaltung tab allows filtering by last 1-12 months from latest date in dataset.

## Git Workflow

- **Main Branch**: `master`
- **Ignored**: `Lager/` directory (contains local/sensitive warehouse data), standard Python build artifacts
- Recent commits follow pattern: Feature descriptions in German, version bumps noted

## Common Modifications

When adding features:
- Keep German language for all user-facing text
- Use semicolons for CSV export (German Excel standard)
- Test with both small and large datasets (SQLite mode)
- Update version numbers in README.md and code if applicable
- Rebuild EXE files after changes: run `build.ps1` or `build_windows.bat`

When working with data:
- Records are dictionaries with keys like 'teilenummer', 'bezeichnung', 'menge', etc.
- Dates are stored as both original string ('abgabe_datum') and ISO format ('abgabe_iso')
- All monetary calculations use float values converted from German comma notation

## Notes

- No automated tests currently exist
- No requirements.txt - dependencies are installed via build scripts
- Application name changed from "Teilenummer-Analyse" to "LagerPilot" in v1.1.0 (some filenames still reflect old name)
- EXE files are distributed via GitHub Releases
- Icon file (`icon.ico`) should be present for Windows builds to include custom icon
