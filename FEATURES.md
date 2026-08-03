# FEATURES.md – Feature Ledger

Vor jeder Erweiterung/Refactor: diese Datei lesen und pruefen ob die
Aenderung eine dort gelistete Funktion betrifft. Nach der Aenderung: per
grep bestaetigen dass alle betroffenen gelisteten Funktionen noch existieren.
Fehlt eine und war das nicht explizit Auftrag -> Nutzer fragen, nicht raten.
Neue/geaenderte Funktionen eintragen (Format siehe unten).

---

## Parser & Datenimport

- `_load_matplotlib()` (teilenummer_analyse.py:30) – Lazy-Loading von matplotlib mit TkAgg-Backend
- `TeilenummerParser.parse_file()` (teilenummer_analyse.py:90) – Streaming-Parser fuer DMS Loco-Soft Dateien mit progress/record_callback
- `TeilenummerParser._parse_metadata()` (teilenummer_analyse.py:159) – Extrahiert Key-Value-Paare aus Metadaten-Zeile

## Datenspeicher (SQLite)

- `SQLiteDataStore._setup_schema()` (teilenummer_analyse.py:181) – Erstellt records-Tabelle mit Indizes
- `SQLiteDataStore.insert_record()` (teilenummer_analyse.py:212) – Fuegt einzelnen Datensatz ein
- `SQLiteDataStore.fetch_records()` (teilenummer_analyse.py:265) – Holt gefilterte Records mit Limit
- `SQLiteDataStore.get_top_n()` (teilenummer_analyse.py:278) – Aggregiert Top-N-Teile nach Vorgaengen/Menge/Umsatz
- `SQLiteDataStore.get_summary()` (teilenummer_analyse.py:342) – Gesamtstatistik (Rows, Parts, Menge, Umsatz)
- `SQLiteDataStore.get_monthly_data()` (teilenummer_analyse.py:354) – Monatliche Aggregation
- `SQLiteDataStore.get_quarterly_data()` (teilenummer_analyse.py:375) – Quartalsweise Aggregation

## Statistik-Engine

- `TeilenummerStatistik.get_top_n()` (teilenummer_analyse.py:438) – Top-N-Teile sortiert nach Vorgaengen/Menge/Umsatz
- `TeilenummerStatistik.get_teilenummer_details()` (teilenummer_analyse.py:457) – Aggregiert alle Teilenummern mit Bezeichnung, Vorgaenge, Menge, Umsatz, Kunden
- `TeilenummerStatistik.get_lagerhaltung_analyse()` (teilenummer_analyse.py:570) – Kernfeature: Lagerhaltung mit ABC-Klassifizierung, Trend, Saisonalitaet, Prognose, Bestellpunkt
- `TeilenummerStatistik._berechne_trend()` (teilenummer_analyse.py:744) – Trendanalyse (Steigend/Fallend/Stabil)
- `TeilenummerStatistik._erkenne_saisonalitaet()` (teilenummer_analyse.py:764) – Saisonalitaetserkennung per Quartalsvergleich
- `TeilenummerStatistik.get_date_range()` (teilenummer_analyse.py:410) – Ermittelt min/max Datum
- `TeilenummerStatistik.get_summary()` (teilenummer_analyse.py:557) – Gesamtstatistik

## GUI – Tabs & Hauptfunktionen

- `AnalyseApp._build_ui()` (teilenummer_analyse.py:1123) – Baut gesamte Hauptstruktur mit Tabs
- `AnalyseApp._build_top_tab()` (teilenummer_analyse.py:1191) – Tab "Top Teilenummern" mit sortierbarer Tabelle
- `AnalyseApp._build_lagerhaltung_tab()` (teilenummer_analyse.py:1234) – Tab "Lagerhaltung" mit 16 Spalten, Autocomplete, Filtern, Tooltips
- `AnalyseApp._build_lagerabbau_tab()` (teilenummer_analyse.py:1385) – Tab "Lagerabbau" mit Quick-Buttons, ABC-Analyse, Prioritaets-Score, Aktionsempfehlungen
- `AnalyseApp._build_chart_tab()` (teilenummer_analyse.py:1536) – Tab "Grafiken" mit Diagramm-Auswahl
- `AnalyseApp._build_time_tab()` (teilenummer_analyse.py:1587) – Tab "Zeitraum-Analyse"
- `AnalyseApp._build_all_data_tab()` (teilenummer_analyse.py:1609) – Tab "Alle Daten" (durchsuchbar, limitiert 1000)
- `AnalyseApp._build_summary_tab()` (teilenummer_analyse.py:1638) – Tab "Zusammenfassung"
- `AnalyseApp._build_menu()` (teilenummer_analyse.py:1646) – Menueleiste mit Shortcuts (Ctrl+O/E/S)

## GUI – Datei-Import

- `AnalyseApp._open_file()` (teilenummer_analyse.py:1661) – Oeffnet Verkaufsdatei, streamed in SQLite, Fortschrittsbalken
- `AnalyseApp._open_lagerbestand_file()` (teilenummer_analyse.py:1709) – Laedt Lagerbestand (Tab-getrennt, auto-Encoding, mehrstufiges Matching)

## GUI – Lagerhaltung & Lagerabbau

- `AnalyseApp._update_lagerhaltung()` (teilenummer_analyse.py:2034) – Fuellt Lagerhaltungstabelle mit Reichweite, Umschlag, Bestellpunkt, Prognose, Filter
- `AnalyseApp._update_lagerabbau()` (teilenummer_analyse.py:2590) – Lagerabbau-Analyse: kreuzt Bestand mit Verkauf, berechnet Abbau-Menge, Aktionsempfehlung
- `AnalyseApp._quick_filter_kritisch()` (teilenummer_analyse.py:2481) – Quick-Filter: >500EUR Lagerwert, >1 Jahr
- `AnalyseApp._quick_filter_nie_verkauft()` (teilenummer_analyse.py:2491) – Quick-Filter: 0 Verkaeufe
- `AnalyseApp._quick_filter_top_wert()` (teilenummer_analyse.py:2501) – Quick-Filter: Top 20 Lagerwert
- `AnalyseApp._quick_filter_top_dauer()` (teilenummer_analyse.py:2512) – Quick-Filter: Top 20 Lagerdauer
- `AnalyseApp._quick_filter_alte_teile()` (teilenummer_analyse.py:2523) – Quick-Filter: Teile aelter als 2 Jahre
- `AnalyseApp._export_lagerabbau()` (teilenummer_analyse.py:2545) – Export Lagerabbau als CSV
- `AnalyseApp._export_lagerhaltung()` (teilenummer_analyse.py:3083) – Export Lagerhaltung als CSV

## GUI – Autocomplete & Filter

- `AnalyseApp._autocomplete_produkt()` (teilenummer_analyse.py:2337) – Autocomplete mit Priorisierung (Wortanfang > Enthaelt)
- `AnalyseApp._init_produkt_liste()` (teilenummer_analyse.py:2920) – Baut Produktliste nach Haeufigkeit
- `AnalyseApp._search_lager_list()` (teilenummer_analyse.py:2953) – Filtert Lagerhaltung nach Suchbegriff

## GUI – Diagramme

- `AnalyseApp._chart_top_bar()` (teilenummer_analyse.py:3222) – Horizontales Balkendiagramm Top-10
- `AnalyseApp._chart_top_pie()` (teilenummer_analyse.py:3240) – Kreisdiagramm Top-8
- `AnalyseApp._chart_time()` (teilenummer_analyse.py:3269) – Zeitverlaufs-Balkendiagramm mit Durchschnittslinie
- `AnalyseApp._chart_monthly_compare()` (teilenummer_analyse.py:3293) – Dual-Axis Monatsvergleich
- `AnalyseApp._save_chart()` (teilenummer_analyse.py:3314) – Speichert Diagramm als PNG/PDF/SVG

## GUI – Export & Sortierung

- `AnalyseApp._export_results()` (teilenummer_analyse.py:3351) – CSV-Export (Semikolon, dt. Format)
- `AnalyseApp._sort_treeview()` (teilenummer_analyse.py:2270) – Universelle Spaltensortierung (numerisch/alphabetisch)

## Hilfsklassen

- `TreeviewHeaderTooltip` (teilenummer_analyse.py:979) – Tooltips fuer Treeview-Spaltenueberschriften
- `main()` (teilenummer_analyse.py:3386) – Einstiegspunkt mit SQLite-Cleanup
