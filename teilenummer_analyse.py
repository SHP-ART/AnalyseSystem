#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teilenummer-Analysesystem (erweitert)
- Plattformübergreifend (Windows/macOS)
- Fortschrittsanzeige beim Import großer Dateien
- Optionaler SQLite-Modus für sehr große Datenmengen
- Grafische Auswertungen & Zeitraum-Filter
"""

import csv
import os
import sqlite3
import tempfile
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Matplotlib wird verzögert geladen
matplotlib = None
plt = None
FigureCanvasTkAgg = None
NavigationToolbar2Tk = None
Figure = None

def _load_matplotlib():
    global matplotlib, plt, FigureCanvasTkAgg, NavigationToolbar2Tk, Figure
    if matplotlib is None:
        import matplotlib as mpl
        mpl.use('TkAgg')
        import matplotlib.pyplot as _plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as _FigureCanvasTkAgg
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as _NavigationToolbar2Tk
        from matplotlib.figure import Figure as _Figure
        matplotlib = mpl
        plt = _plt
        FigureCanvasTkAgg = _FigureCanvasTkAgg
        NavigationToolbar2Tk = _NavigationToolbar2Tk
        Figure = _Figure


# -----------------------------------------------------------------------------
# Parser
# -----------------------------------------------------------------------------
class TeilenummerParser:
    """Parst DMS Loco-Soft Textdateien (streaming-fähig)."""

    EXPECTED_COLUMNS = [
        'ET-Nr.', 'Bezeichnung', 'Auftrag', 'Abg.Datum', 'Rg.-Art',
        'Rg.-Nr.', 'Rg.Datum', 'TA', 'Menge', 'Bestandsneutraler Teileabgang',
        'UPE', 'VK-Preis (rabattiert)', 'EW', 'H-Spanne', 'Lager', 'MA',
        'Kd.-Nr.', 'Kd.-Name', 'Kd.-Code 1', 'Kd.-Code 2', 'Kd.-Code 3',
        'Kd.-Code 4', 'Stellantis Zusatzcode ET-Herkunft', 'Stellantis Index',
        'Stellantis WKW-Teil', '', 'Zug.Datum'
    ]

    def __init__(self):
        self.metadata = {}
        self.headers = []
        self.data = []

    @staticmethod
    def _parse_number(value: str) -> float:
        if not value:
            return 0.0
        try:
            return float(value.strip().replace(',', '.'))
        except ValueError:
            return 0.0

    @staticmethod
    def _parse_date(value: str):
        if not value:
            return None
        for fmt in ('%d.%m.%Y', '%d.%m.%y'):
            try:
                return datetime.strptime(value.strip(), fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _iso_date(dt):
        return dt.strftime('%Y-%m-%d') if dt else None

    def parse_file(
        self,
        filepath: str,
        progress_callback=None,
        record_callback=None,
        store_records: bool = True,
    ):
        """Parst Datei, meldet Fortschritt und liefert optional Records zurück."""
        self.metadata = {}
        self.headers = []
        self.data = []

        total_bytes = max(os.path.getsize(filepath), 1)
        processed_bytes = 0

        with open(filepath, 'r', encoding='utf-8', errors='replace') as handle:
            first_line = handle.readline()
            processed_bytes += len(first_line.encode('utf-8', 'ignore'))

            if 'DMS:' in first_line:
                self._parse_metadata(first_line)

            header_start = first_line.find('ET-Nr.')
            if header_start != -1:
                header_part = first_line[header_start:]
                self.headers = [col.strip() for col in header_part.split(';')]
            else:
                self.headers = self.EXPECTED_COLUMNS.copy()

            for line in handle:
                processed_bytes += len(line.encode('utf-8', 'ignore'))
                line = line.strip()
                if not line or line.startswith('DMS:'):
                    continue
                fields = line.split(';')
                if len(fields) < 2 or not fields[0].strip():
                    continue

                abgabe_dt = self._parse_date(fields[3]) if len(fields) > 3 else None
                record = {
                    'teilenummer': fields[0].strip(),
                    'bezeichnung': fields[1].strip() if len(fields) > 1 else '',
                    'auftrag': fields[2].strip() if len(fields) > 2 else '',
                    'abgabe_datum': fields[3].strip() if len(fields) > 3 else '',
                    'abgabe_iso': self._iso_date(abgabe_dt),
                    'rg_art': fields[4].strip() if len(fields) > 4 else '',
                    'rg_nr': fields[5].strip() if len(fields) > 5 else '',
                    'rg_datum': fields[6].strip() if len(fields) > 6 else '',
                    'ta': fields[7].strip() if len(fields) > 7 else '',
                    'menge': self._parse_number(fields[8]) if len(fields) > 8 else 0.0,
                    'upe': self._parse_number(fields[10]) if len(fields) > 10 else 0.0,
                    'vk_preis': self._parse_number(fields[11]) if len(fields) > 11 else 0.0,
                    'ew': self._parse_number(fields[12]) if len(fields) > 12 else 0.0,
                    'h_spanne': self._parse_number(fields[13]) if len(fields) > 13 else 0.0,
                    'lager': fields[14].strip() if len(fields) > 14 else '',
                    'ma': fields[15].strip() if len(fields) > 15 else '',
                    'kd_nr': fields[16].strip() if len(fields) > 16 else '',
                    'kd_name': fields[17].strip() if len(fields) > 17 else '',
                }

                if store_records:
                    self.data.append(record)
                if record_callback:
                    record_callback(record)
                if progress_callback:
                    progress_callback(processed_bytes, total_bytes)

        return self.metadata, self.data

    def _parse_metadata(self, line: str):
        for segment in line.split(';'):
            if ':' in segment:
                key, _, value = segment.partition(':')
                key = key.strip()
                if key and key != 'ET-Nr.':
                    self.metadata[key] = value.strip()


# -----------------------------------------------------------------------------
# SQLite Store
# -----------------------------------------------------------------------------
class SQLiteDataStore:
    """Persistenter Speicher für sehr große Datenmengen."""

    def __init__(self, db_path=None):
        self._temp_dir = tempfile.mkdtemp(prefix='teilenummer_db_')
        self.db_path = db_path or os.path.join(self._temp_dir, 'analyse.db')
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._setup_schema()

    def _setup_schema(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                teilenummer TEXT,
                bezeichnung TEXT,
                auftrag TEXT,
                abgabe_datum TEXT,
                abgabe_iso TEXT,
                rg_art TEXT,
                rg_nr TEXT,
                rg_datum TEXT,
                ta TEXT,
                menge REAL,
                upe REAL,
                vk_preis REAL,
                ew REAL,
                h_spanne REAL,
                lager TEXT,
                ma TEXT,
                kd_nr TEXT,
                kd_name TEXT
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_records_teilenummer ON records(teilenummer);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_records_abgabe ON records(abgabe_iso);")
        self.conn.commit()

    def insert_record(self, record: dict):
        self.conn.execute(
            """
            INSERT INTO records (
                teilenummer, bezeichnung, auftrag, abgabe_datum, abgabe_iso,
                rg_art, rg_nr, rg_datum, ta, menge, upe, vk_preis, ew,
                h_spanne, lager, ma, kd_nr, kd_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record['teilenummer'], record['bezeichnung'], record['auftrag'],
                record['abgabe_datum'], record['abgabe_iso'], record['rg_art'],
                record['rg_nr'], record['rg_datum'], record['ta'], record['menge'],
                record['upe'], record['vk_preis'], record['ew'], record['h_spanne'],
                record['lager'], record['ma'], record['kd_nr'], record['kd_name'],
            ),
        )

    def finalize(self):
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        finally:
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    # --- Helper for filters -------------------------------------------------
    @staticmethod
    def build_filter_clause(filters):
        if not filters or filters.get('type') == 'alle':
            return '', []
        if filters['type'] == 'monat':
            return "WHERE strftime('%Y-%m', abgabe_iso) = ?", [filters['value']]
        if filters['type'] == 'quartal':
            return (
                "WHERE strftime('%Y', abgabe_iso) || '-Q' || (((CAST(strftime('%m', abgabe_iso) AS INTEGER)-1)/3)+1) = ?",
                [filters['value']],
            )
        return '', []

    # --- Queries -----------------------------------------------------------
    def get_record_count(self, filters=None, search=None):
        clause, params = self.build_filter_clause(filters)
        query = "SELECT COUNT(*) FROM records " + clause
        if search:
            query += " AND" if clause else " WHERE"
            query += " (teilenummer LIKE ? OR bezeichnung LIKE ? OR kd_name LIKE ?)"
            search_pattern = f"%{search}%"
            params.extend([search_pattern, search_pattern, search_pattern])
        cur = self.conn.execute(query, params)
        return cur.fetchone()[0]

    def fetch_records(self, filters=None, search=None, limit=1000):
        clause, params = self.build_filter_clause(filters)
        query = "SELECT teilenummer, bezeichnung, abgabe_datum, menge, kd_name FROM records " + clause
        if search:
            query += " AND" if clause else " WHERE"
            query += " (teilenummer LIKE ? OR bezeichnung LIKE ? OR kd_name LIKE ?)"
            search_pattern = f"%{search}%"
            params.extend([search_pattern, search_pattern, search_pattern])
        query += " ORDER BY COALESCE(abgabe_iso, '') LIMIT ?"
        params.append(limit)
        cur = self.conn.execute(query, params)
        return [dict(row) for row in cur.fetchall()]

    def get_top_n(self, by='vorgaenge', limit=20, filters=None):
        clause, params = self.build_filter_clause(filters)
        metric = {
            'vorgaenge': 'COUNT(*)',
            'menge': 'SUM(menge)',
            'umsatz': 'SUM(vk_preis)',
        }.get(by, 'COUNT(*)')
        query = f"""
            SELECT
                teilenummer,
                MAX(bezeichnung) AS bezeichnung,
                COUNT(*) AS anzahl_vorgaenge,
                SUM(menge) AS gesamtmenge,
                SUM(vk_preis) AS gesamtumsatz,
                COUNT(DISTINCT kd_name) AS anzahl_kunden
            FROM records
            {clause}
            GROUP BY teilenummer
            ORDER BY {metric} DESC
            LIMIT ?
        """
        params_with_limit = params + [limit]
        cur = self.conn.execute(query, params_with_limit)
        return [dict(row) for row in cur.fetchall()]

    def get_time_data(self, mode='monat'):
        if mode == 'quartal':
            period_expr = "strftime('%Y', abgabe_iso) || '-Q' || (((CAST(strftime('%m', abgabe_iso) AS INTEGER)-1)/3)+1)"
        else:
            period_expr = "strftime('%Y-%m', abgabe_iso)"

        query = f"""
            WITH aggregated AS (
                SELECT
                    {period_expr} AS periode,
                    COUNT(*) AS vorgaenge,
                    SUM(menge) AS menge,
                    SUM(vk_preis) AS umsatz
                FROM records
                WHERE abgabe_iso IS NOT NULL
                GROUP BY periode
            ), ranked AS (
                SELECT
                    {period_expr} AS periode,
                    teilenummer,
                    COUNT(*) AS cnt,
                    ROW_NUMBER() OVER (PARTITION BY {period_expr} ORDER BY COUNT(*) DESC, teilenummer) AS rn
                FROM records
                WHERE abgabe_iso IS NOT NULL
                GROUP BY periode, teilenummer
            )
            SELECT
                aggregated.periode,
                aggregated.vorgaenge,
                aggregated.menge,
                aggregated.umsatz,
                COALESCE(ranked.teilenummer || ' (' || ranked.cnt || 'x)', '-') AS top_teil
            FROM aggregated
            LEFT JOIN ranked ON aggregated.periode = ranked.periode AND ranked.rn = 1
            ORDER BY aggregated.periode
        """
        cur = self.conn.execute(query)
        return [dict(row) for row in cur.fetchall()]

    def get_summary(self):
        cur = self.conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT teilenummer), SUM(menge), SUM(vk_preis) FROM records"
        )
        total_rows, unique_parts, total_qty, total_revenue = cur.fetchone()
        return {
            'rows': total_rows or 0,
            'unique_parts': unique_parts or 0,
            'total_qty': total_qty or 0.0,
            'total_revenue': total_revenue or 0.0,
        }

    def get_monthly_data(self):
        data = self.get_time_data(mode='monat')
        return {item['periode']: item for item in data}

    def get_quarterly_data(self):
        data = self.get_time_data(mode='quartal')
        return {item['periode']: item for item in data}


# -----------------------------------------------------------------------------
# Statistik
# -----------------------------------------------------------------------------
class TeilenummerStatistik:
    def __init__(self, data=None, db_store: SQLiteDataStore | None = None):
        self.data = data or []
        self.db_store = db_store

    # --- Helpers -----------------------------------------------------------
    def _filter_data(self, dataset, filters):
        if not filters or filters.get('type') == 'alle':
            return dataset
        filtered = []
        for record in dataset:
            iso = record.get('abgabe_iso')
            if not iso:
                continue
            if filters['type'] == 'monat' and iso[:7] == filters['value']:
                filtered.append(record)
            elif filters['type'] == 'quartal':
                quarter = (int(iso[5:7]) - 1) // 3 + 1
                if f"{iso[:4]}-Q{quarter}" == filters['value']:
                    filtered.append(record)
        return filtered

    def get_record_count(self, filters=None):
        if self.db_store:
            return self.db_store.get_record_count(filters)
        return len(self._filter_data(self.data, filters))

    def fetch_records(self, filters=None, search=None, limit=1000):
        if self.db_store:
            return self.db_store.fetch_records(filters, search, limit)
        dataset = self._filter_data(self.data, filters)
        if search:
            search = search.lower()
            dataset = [
                r for r in dataset
                if search in r['teilenummer'].lower()
                or search in r['bezeichnung'].lower()
                or search in r['kd_name'].lower()
            ]
        return dataset[:limit]

    def get_top_n(self, n=20, by='vorgaenge', data=None, filters=None):
        if self.db_store:
            return self.db_store.get_top_n(by, n, filters)
        dataset = data if data is not None else self._filter_data(self.data, filters)
        details = self.get_teilenummer_details(dataset)
        key_map = {
            'vorgaenge': lambda item: item[1]['anzahl_vorgaenge'],
            'menge': lambda item: item[1]['gesamtmenge'],
            'umsatz': lambda item: item[1]['gesamt_umsatz'],
        }
        sort_key = key_map.get(by, key_map['vorgaenge'])
        return [
            {
                'teilenummer': tnr,
                **stats,
            }
            for tnr, stats in sorted(details.items(), key=sort_key, reverse=True)[:n]
        ]

    def get_teilenummer_details(self, data=None):
        if self.db_store:
            rows = self.db_store.get_top_n(by='vorgaenge', limit=10**6)
            return {row['teilenummer']: row for row in rows}
        dataset = data if data is not None else self.data
        details = defaultdict(lambda: {
            'bezeichnung': '',
            'anzahl_vorgaenge': 0,
            'gesamtmenge': 0.0,
            'gesamt_umsatz': 0.0,
            'anzahl_kunden': 0,
            'kunden': set(),
        })
        for record in dataset:
            block = details[record['teilenummer']]
            block['bezeichnung'] = record['bezeichnung']
            block['anzahl_vorgaenge'] += 1
            block['gesamtmenge'] += record['menge']
            block['gesamt_umsatz'] += record['vk_preis']
            if record['kd_name']:
                block['kunden'].add(record['kd_name'])
        for block in details.values():
            block['anzahl_kunden'] = len(block['kunden'])
            block.pop('kunden', None)
        return details

    def get_monthly_data(self):
        if self.db_store:
            return self.db_store.get_monthly_data()
        monthly = defaultdict(lambda: {
            'periode': '',
            'vorgaenge': 0,
            'menge': 0.0,
            'umsatz': 0.0,
            'top_teil': '-',
            'teilenummern': Counter(),
        })
        for record in self.data:
            iso = record.get('abgabe_iso')
            if not iso:
                continue
            key = iso[:7]
            monthly[key]['periode'] = key
            monthly[key]['vorgaenge'] += 1
            monthly[key]['menge'] += record['menge']
            monthly[key]['umsatz'] += record['vk_preis']
            monthly[key]['teilenummern'][record['teilenummer']] += 1
        for block in monthly.values():
            most_common = block['teilenummern'].most_common(1)
            block['top_teil'] = f"{most_common[0][0]} ({most_common[0][1]}x)" if most_common else '-'
            block.pop('teilenummern', None)
        return dict(monthly)

    def get_quarterly_data(self):
        if self.db_store:
            return self.db_store.get_quarterly_data()
        quarterly = defaultdict(lambda: {
            'periode': '',
            'vorgaenge': 0,
            'menge': 0.0,
            'umsatz': 0.0,
            'top_teil': '-',
            'teilenummern': Counter(),
        })
        for record in self.data:
            iso = record.get('abgabe_iso')
            if not iso:
                continue
            q = (int(iso[5:7]) - 1) // 3 + 1
            key = f"{iso[:4]}-Q{q}"
            quarterly[key]['periode'] = key
            quarterly[key]['vorgaenge'] += 1
            quarterly[key]['menge'] += record['menge']
            quarterly[key]['umsatz'] += record['vk_preis']
            quarterly[key]['teilenummern'][record['teilenummer']] += 1
        for block in quarterly.values():
            most_common = block['teilenummern'].most_common(1)
            block['top_teil'] = f"{most_common[0][0]} ({most_common[0][1]}x)" if most_common else '-'
            block.pop('teilenummern', None)
        return dict(quarterly)

    def get_summary(self):
        if self.db_store:
            return self.db_store.get_summary()
        total_qty = sum(r['menge'] for r in self.data)
        total_rev = sum(r['vk_preis'] for r in self.data)
        unique_parts = len({r['teilenummer'] for r in self.data})
        return {
            'rows': len(self.data),
            'unique_parts': unique_parts,
            'total_qty': total_qty,
            'total_revenue': total_rev,
        }

    def get_lagerhaltung_analyse(self, max_tage_lohnend=60):
        """
        Analysiert welche Teile sich lohnen im Lager zu halten.
        
        Kategorien:
        - "Lohnend": Verkauf alle 1-2 Monate (≤60 Tage zwischen Verkäufen)
        - "Grenzwertig": Verkauf alle 2-4 Monate (61-120 Tage)
        - "Nicht lohnend": Verkauf seltener als alle 4 Monate (>120 Tage)
        """
        if self.db_store:
            return self._get_lagerhaltung_from_db(max_tage_lohnend)
        
        # Sammle alle Verkaufsdaten pro Teilenummer
        teil_verkäufe = defaultdict(list)
        for record in self.data:
            iso = record.get('abgabe_iso')
            if iso:
                teil_verkäufe[record['teilenummer']].append({
                    'datum': iso,
                    'bezeichnung': record['bezeichnung'],
                    'menge': record['menge'],
                    'umsatz': record['vk_preis'],
                })
        
        ergebnis = []
        for teilenummer, verkäufe in teil_verkäufe.items():
            if len(verkäufe) < 2:
                # Nur ein Verkauf - kann Frequenz nicht berechnen
                avg_tage = 999
            else:
                # Sortiere nach Datum und berechne Durchschnitt der Abstände
                sorted_dates = sorted([v['datum'] for v in verkäufe])
                abstände = []
                for i in range(1, len(sorted_dates)):
                    d1 = datetime.strptime(sorted_dates[i-1], '%Y-%m-%d')
                    d2 = datetime.strptime(sorted_dates[i], '%Y-%m-%d')
                    abstände.append((d2 - d1).days)
                avg_tage = sum(abstände) / len(abstände) if abstände else 999
            
            # Kategorisierung
            if avg_tage <= 60:
                kategorie = "✅ Lohnend"
                empfehlung = "Im Lager halten"
            elif avg_tage <= 120:
                kategorie = "⚠️ Grenzwertig"
                empfehlung = "Bestand reduzieren"
            else:
                kategorie = "❌ Nicht lohnend"
                empfehlung = "Nicht bevorraten"
            
            gesamtmenge = sum(v['menge'] for v in verkäufe)
            gesamtumsatz = sum(v['umsatz'] for v in verkäufe)
            
            ergebnis.append({
                'teilenummer': teilenummer,
                'bezeichnung': verkäufe[0]['bezeichnung'],
                'anzahl_verkäufe': len(verkäufe),
                'durchschnitt_tage': avg_tage if avg_tage < 999 else None,
                'gesamtmenge': gesamtmenge,
                'gesamtumsatz': gesamtumsatz,
                'kategorie': kategorie,
                'empfehlung': empfehlung,
            })
        
        return sorted(ergebnis, key=lambda x: x['durchschnitt_tage'] or 9999)

    def _get_lagerhaltung_from_db(self, max_tage_lohnend=60):
        """SQLite-Version der Lagerhaltungsanalyse."""
        query = """
            WITH verkauf_daten AS (
                SELECT 
                    teilenummer,
                    MAX(bezeichnung) AS bezeichnung,
                    abgabe_iso,
                    SUM(menge) AS menge,
                    SUM(vk_preis) AS umsatz,
                    COUNT(*) AS anzahl
                FROM records
                WHERE abgabe_iso IS NOT NULL
                GROUP BY teilenummer, abgabe_iso
            ),
            teil_stats AS (
                SELECT 
                    teilenummer,
                    MAX(bezeichnung) AS bezeichnung,
                    COUNT(DISTINCT abgabe_iso) AS anzahl_verkaufstage,
                    SUM(menge) AS gesamtmenge,
                    SUM(umsatz) AS gesamtumsatz,
                    MIN(abgabe_iso) AS erster_verkauf,
                    MAX(abgabe_iso) AS letzter_verkauf
                FROM verkauf_daten
                GROUP BY teilenummer
            )
            SELECT 
                teilenummer,
                bezeichnung,
                anzahl_verkaufstage,
                gesamtmenge,
                gesamtumsatz,
                CASE 
                    WHEN anzahl_verkaufstage < 2 THEN NULL
                    ELSE CAST(julianday(letzter_verkauf) - julianday(erster_verkauf) AS INTEGER) / (anzahl_verkaufstage - 1)
                END AS durchschnitt_tage
            FROM teil_stats
            ORDER BY durchschnitt_tage NULLS LAST
        """
        cur = self.db_store.conn.execute(query)
        ergebnis = []
        for row in cur.fetchall():
            avg_tage = row[5]
            if avg_tage is None or avg_tage > 120:
                kategorie = "❌ Nicht lohnend"
                empfehlung = "Nicht bevorraten"
            elif avg_tage <= 60:
                kategorie = "✅ Lohnend"
                empfehlung = "Im Lager halten"
            else:
                kategorie = "⚠️ Grenzwertig"
                empfehlung = "Bestand reduzieren"
            
            ergebnis.append({
                'teilenummer': row[0],
                'bezeichnung': row[1],
                'anzahl_verkäufe': row[2],
                'durchschnitt_tage': avg_tage,
                'gesamtmenge': row[3],
                'gesamtumsatz': row[4],
                'kategorie': kategorie,
                'empfehlung': empfehlung,
            })
        return ergebnis


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------
class AnalyseApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Teilenummer-Analyse')
        self.geometry('1400x900')

        self.parser = TeilenummerParser()
        self.statistik = None
        self.data = []
        self.filtered_data = []
        self.metadata = {}
        self.current_file = None
        self.filter_params = {'type': 'alle', 'value': None}
        self.sqlite_store: SQLiteDataStore | None = None

        self._build_ui()
        self._build_menu()

    # --- UI ----------------------------------------------------------------
    def _build_ui(self):
        main = ttk.Frame(self, padding='10')
        main.grid(row=0, column=0, sticky='nsew')
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(3, weight=1)

        # Datei + Speicherwahl
        file_frame = ttk.LabelFrame(main, text='Datei', padding='5')
        file_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        file_frame.columnconfigure(2, weight=1)

        ttk.Button(file_frame, text='Datei öffnen...', command=self._open_file).grid(row=0, column=0, padx=(0, 10))

        ttk.Label(file_frame, text='Speicher:').grid(row=0, column=1)
        self.storage_var = tk.StringVar(value='memory')
        display_values = ['In-Memory (schnell)', 'SQLite (große Dateien)']
        self.storage_map = {
            'In-Memory (schnell)': 'memory',
            'SQLite (große Dateien)': 'sqlite',
        }
        storage_combo = ttk.Combobox(file_frame, values=display_values, state='readonly', width=22)
        storage_combo.current(0)
        storage_combo.bind('<<ComboboxSelected>>', lambda e: self.storage_var.set(self.storage_map[storage_combo.get()]))
        storage_combo.grid(row=0, column=2, padx=(0, 10), sticky='w')

        self.file_label = ttk.Label(file_frame, text='Keine Datei geladen')
        self.file_label.grid(row=0, column=3, sticky='w')

        # Zeitraum-Filter
        filter_frame = ttk.LabelFrame(main, text='Zeitraum-Filter', padding='5')
        filter_frame.grid(row=1, column=0, sticky='ew', pady=(0, 10))

        ttk.Label(filter_frame, text='Ansicht:').grid(row=0, column=0)
        self.period_var = tk.StringVar(value='alle')
        period_combo = ttk.Combobox(filter_frame, textvariable=self.period_var, values=['alle', 'monatlich', '3-monatlich (Quartal)'], width=22, state='readonly')
        period_combo.grid(row=0, column=1, padx=(5, 20))
        period_combo.bind('<<ComboboxSelected>>', self._on_period_change)

        ttk.Label(filter_frame, text='Zeitraum:').grid(row=0, column=2)
        self.time_filter_var = tk.StringVar(value='Alle')
        self.time_filter_combo = ttk.Combobox(filter_frame, textvariable=self.time_filter_var, values=['Alle'], width=18, state='readonly')
        self.time_filter_combo.grid(row=0, column=3, padx=(5, 20))
        self.time_filter_combo.bind('<<ComboboxSelected>>', self._on_time_filter_change)

        ttk.Button(filter_frame, text='Filter anwenden', command=self._apply_filter).grid(row=0, column=4, padx=(5, 10))
        self.filter_info_label = ttk.Label(filter_frame, text='Keine Filter aktiv')
        self.filter_info_label.grid(row=0, column=5, sticky='w')

        # Metadaten
        meta_frame = ttk.LabelFrame(main, text='Datei-Informationen', padding='5')
        meta_frame.grid(row=2, column=0, sticky='ew', pady=(0, 10))
        self.meta_label = ttk.Label(meta_frame, text='')
        self.meta_label.grid(row=0, column=0, sticky='w')

        # Notebook
        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=3, column=0, sticky='nsew')

        self._build_top_tab()
        self._build_lagerhaltung_tab()
        self._build_chart_tab()
        self._build_time_tab()
        self._build_all_data_tab()
        self._build_summary_tab()

        # Status
        status_frame = ttk.Frame(main)
        status_frame.grid(row=4, column=0, sticky='ew', pady=(10, 0))
        status_frame.columnconfigure(0, weight=1)
        self.status_label = ttk.Label(status_frame, text='Bereit')
        self.status_label.grid(row=0, column=0, sticky='w')
        ttk.Button(status_frame, text='Exportieren...', command=self._export_results).grid(row=0, column=1)

    def _build_top_tab(self):
        frame = ttk.Frame(self.notebook, padding='10')
        self.notebook.add(frame, text='Top Teilenummern')
        control = ttk.Frame(frame)
        control.grid(row=0, column=0, sticky='ew', pady=(0, 10))

        ttk.Label(control, text='Anzahl:').grid(row=0, column=0)
        self.top_n_var = tk.StringVar(value='20')
        ttk.Combobox(control, textvariable=self.top_n_var, values=['10', '20', '50', '100'], width=5).grid(row=0, column=1, padx=(5, 20))

        ttk.Label(control, text='Sortieren nach:').grid(row=0, column=2)
        self.sort_var = tk.StringVar(value='vorgaenge')
        ttk.Combobox(control, textvariable=self.sort_var, values=['vorgaenge', 'menge', 'umsatz'], width=12).grid(row=0, column=3, padx=(5, 20))

        ttk.Button(control, text='Aktualisieren', command=self._update_top_list).grid(row=0, column=4)

        columns = ('teilenummer', 'bezeichnung', 'vorgaenge', 'menge', 'umsatz', 'kunden')
        self.top_tree = ttk.Treeview(frame, columns=columns, show='headings')
        headings = {
            'teilenummer': 'Teilenummer',
            'bezeichnung': 'Bezeichnung',
            'vorgaenge': 'Vorgänge',
            'menge': 'Gesamtmenge',
            'umsatz': 'Umsatz (€)',
            'kunden': 'Anz. Kunden',
        }
        widths = {'teilenummer': 120, 'bezeichnung': 320, 'vorgaenge': 90, 'menge': 110, 'umsatz': 110, 'kunden': 100}
        aligns = {'vorgaenge': tk.E, 'menge': tk.E, 'umsatz': tk.E, 'kunden': tk.E}
        for col in columns:
            self.top_tree.heading(col, text=headings[col])
            self.top_tree.column(col, width=widths[col], anchor=aligns.get(col, tk.W))
        self.top_tree.grid(row=1, column=0, sticky='nsew')
        ttk.Scrollbar(frame, orient='vertical', command=self.top_tree.yview).grid(row=1, column=1, sticky='ns')
        self.top_tree.configure(yscrollcommand=lambda *args: None)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

    def _build_lagerhaltung_tab(self):
        """Tab für Lagerhaltungs-Analyse: Welche Teile lohnen sich im Lager?"""
        frame = ttk.Frame(self.notebook, padding='10')
        self.notebook.add(frame, text='📦 Lagerhaltung')
        
        # Info-Label
        info_frame = ttk.Frame(frame)
        info_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        
        info_text = (
            "Analyse der Verkaufsfrequenz: Teile die alle 1-2 Monate verkauft werden (≤60 Tage) "
            "lohnen sich im Lager. Teile mit >120 Tagen zwischen Verkäufen sind Lagerhüter."
        )
        ttk.Label(info_frame, text=info_text, wraplength=1200).grid(row=0, column=0, sticky='w')
        
        # Filter-Steuerung
        control = ttk.Frame(frame)
        control.grid(row=1, column=0, sticky='ew', pady=(0, 10))
        
        ttk.Label(control, text='Anzeigen:').grid(row=0, column=0)
        self.lager_filter_var = tk.StringVar(value='alle')
        ttk.Combobox(
            control, 
            textvariable=self.lager_filter_var,
            values=['alle', 'nur lohnend', 'nur grenzwertig', 'nur nicht lohnend'],
            width=18,
            state='readonly'
        ).grid(row=0, column=1, padx=(5, 20))
        
        ttk.Button(control, text='Aktualisieren', command=self._update_lagerhaltung).grid(row=0, column=2, padx=(5, 10))
        ttk.Button(control, text='Als CSV exportieren', command=self._export_lagerhaltung).grid(row=0, column=3)
        
        # Statistik-Labels
        self.lager_stats_label = ttk.Label(control, text='')
        self.lager_stats_label.grid(row=0, column=4, padx=(20, 0), sticky='w')
        
        # Tabelle
        columns = ('teilenummer', 'bezeichnung', 'kategorie', 'tage', 'verkäufe', 'menge', 'umsatz', 'empfehlung')
        self.lager_tree = ttk.Treeview(frame, columns=columns, show='headings')
        headings = {
            'teilenummer': 'Teilenummer',
            'bezeichnung': 'Bezeichnung',
            'kategorie': 'Kategorie',
            'tage': 'Ø Tage',
            'verkäufe': 'Verkäufe',
            'menge': 'Gesamtmenge',
            'umsatz': 'Umsatz (€)',
            'empfehlung': 'Empfehlung',
        }
        widths = {
            'teilenummer': 120, 'bezeichnung': 280, 'kategorie': 120, 
            'tage': 70, 'verkäufe': 70, 'menge': 100, 'umsatz': 100, 'empfehlung': 140
        }
        aligns = {'tage': tk.E, 'verkäufe': tk.E, 'menge': tk.E, 'umsatz': tk.E}
        for col in columns:
            self.lager_tree.heading(col, text=headings[col])
            self.lager_tree.column(col, width=widths[col], anchor=aligns.get(col, tk.W))
        
        self.lager_tree.grid(row=2, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=self.lager_tree.yview)
        scrollbar.grid(row=2, column=1, sticky='ns')
        self.lager_tree.configure(yscrollcommand=scrollbar.set)
        
        frame.rowconfigure(2, weight=1)
        frame.columnconfigure(0, weight=1)

    def _build_chart_tab(self):
        frame = ttk.Frame(self.notebook, padding='10')
        self.notebook.add(frame, text='📊 Grafiken')
        control = ttk.Frame(frame)
        control.grid(row=0, column=0, sticky='ew', pady=(0, 10))

        ttk.Label(control, text='Diagramm:').grid(row=0, column=0)
        self.chart_type_var = tk.StringVar(value='top_bar')
        ttk.Combobox(
            control,
            textvariable=self.chart_type_var,
            values=['top_bar', 'top_pie', 'zeitverlauf_vorgaenge', 'zeitverlauf_umsatz', 'monatlich_vergleich'],
            width=25,
            state='readonly',
        ).grid(row=0, column=1, padx=(5, 20))

        ttk.Button(control, text='Diagramm aktualisieren', command=self._update_chart).grid(row=0, column=2, padx=(5, 10))
        ttk.Button(control, text='Als Bild speichern', command=self._save_chart).grid(row=0, column=3)

        self.chart_container = ttk.Frame(frame)
        self.chart_container.grid(row=1, column=0, sticky='nsew')
        
        # Chart-Elemente werden erst bei Bedarf erstellt
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self._chart_frame = frame
        
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)
    
    def _ensure_chart_initialized(self):
        """Erstellt Chart-Elemente bei Bedarf (lazy loading)."""
        if self.figure is None:
            _load_matplotlib()
            try:
                plt.style.use('seaborn-v0_8-whitegrid')
            except Exception:
                pass
            self.figure = Figure(figsize=(12, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.figure, self.chart_container)
            self.canvas.get_tk_widget().pack(fill='both', expand=True)
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.chart_container)
            self.toolbar.update()

    def _build_time_tab(self):
        frame = ttk.Frame(self.notebook, padding='10')
        self.notebook.add(frame, text='📅 Zeitraum-Analyse')
        columns = ('zeitraum', 'vorgaenge', 'menge', 'umsatz', 'top_teil')
        self.time_tree = ttk.Treeview(frame, columns=columns, show='headings')
        labels = {
            'zeitraum': 'Zeitraum',
            'vorgaenge': 'Vorgänge',
            'menge': 'Gesamtmenge',
            'umsatz': 'Umsatz (€)',
            'top_teil': 'Top Teilenummer',
        }
        widths = {'zeitraum': 120, 'vorgaenge': 100, 'menge': 140, 'umsatz': 140, 'top_teil': 180}
        aligns = {'vorgaenge': tk.E, 'menge': tk.E, 'umsatz': tk.E}
        for col in columns:
            self.time_tree.heading(col, text=labels[col])
            self.time_tree.column(col, width=widths[col], anchor=aligns.get(col, tk.W))
        self.time_tree.grid(row=0, column=0, sticky='nsew')
        ttk.Scrollbar(frame, orient='vertical', command=self.time_tree.yview).grid(row=0, column=1, sticky='ns')
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

    def _build_all_data_tab(self):
        frame = ttk.Frame(self.notebook, padding='10')
        self.notebook.add(frame, text='Alle Daten')
        search_frame = ttk.Frame(frame)
        search_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        ttk.Label(search_frame, text='Suche:').grid(row=0, column=0)
        self.search_var = tk.StringVar()
        self.search_var.trace_add('write', self._on_search)
        ttk.Entry(search_frame, textvariable=self.search_var, width=30).grid(row=0, column=1, padx=(5, 20))

        columns = ('teilenummer', 'bezeichnung', 'datum', 'menge', 'kunde')
        self.all_tree = ttk.Treeview(frame, columns=columns, show='headings')
        labels = {
            'teilenummer': 'Teilenummer',
            'bezeichnung': 'Bezeichnung',
            'datum': 'Datum',
            'menge': 'Menge',
            'kunde': 'Kunde',
        }
        widths = {'teilenummer': 120, 'bezeichnung': 320, 'datum': 130, 'menge': 90, 'kunde': 240}
        aligns = {'menge': tk.E}
        for col in columns:
            self.all_tree.heading(col, text=labels[col])
            self.all_tree.column(col, width=widths[col], anchor=aligns.get(col, tk.W))
        self.all_tree.grid(row=1, column=0, sticky='nsew')
        ttk.Scrollbar(frame, orient='vertical', command=self.all_tree.yview).grid(row=1, column=1, sticky='ns')
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

    def _build_summary_tab(self):
        frame = ttk.Frame(self.notebook, padding='10')
        self.notebook.add(frame, text='Zusammenfassung')
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.summary_text = ScrolledText(frame, wrap='word', font=('Courier', 11))
        self.summary_text.grid(row=0, column=0, sticky='nsew')

    def _build_menu(self):
        menu = tk.Menu(self)
        self.config(menu=menu)
        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label='Datei', menu=file_menu)
        shortcut = 'Cmd+O' if os.name == 'darwin' else 'Strg+O'
        file_menu.add_command(label=f'Öffnen... ({shortcut})', command=self._open_file)
        file_menu.add_command(label='Ergebnisse exportieren...', command=self._export_results)
        file_menu.add_command(label='Diagramm speichern...', command=self._save_chart)
        file_menu.add_separator()
        file_menu.add_command(label='Beenden', command=self.destroy)
        self.bind('<Command-o>' if os.name == 'darwin' else '<Control-o>', lambda e: self._open_file())

    # --- Datei laden ------------------------------------------------------
    def _open_file(self):
        filetypes = [('Textdateien', '*.txt'), ('CSV-Dateien', '*.csv'), ('Alle Dateien', '*.*')]
        filepath = filedialog.askopenfilename(title='Textdatei öffnen', filetypes=filetypes)
        if not filepath:
            return

        storage_mode = self.storage_var.get()
        self.status_label.config(text='Datei wird geladen...')
        self.update()
        
        # Prüfe Dateigröße - nur bei großen Dateien (>1MB) Fortschrittsanzeige
        file_size = os.path.getsize(filepath)
        use_progress = file_size > 1_000_000  # 1 MB
        
        progress_ui = self._show_progress('Importiere Datei...') if use_progress else None

        try:
            if storage_mode == 'sqlite':
                if self.sqlite_store:
                    self.sqlite_store.close()
                self.sqlite_store = SQLiteDataStore()
                progress_cb = (lambda done, total: self._update_progress(progress_ui, done, total)) if use_progress else None
                metadata, _ = self.parser.parse_file(
                    filepath,
                    progress_callback=progress_cb,
                    record_callback=self.sqlite_store.insert_record,
                    store_records=False,
                )
                self.sqlite_store.finalize()
                self.data = []
                self.filtered_data = []
                self.statistik = TeilenummerStatistik(db_store=self.sqlite_store)
            else:
                progress_cb = (lambda done, total: self._update_progress(progress_ui, done, total)) if use_progress else None
                metadata, data = self.parser.parse_file(
                    filepath,
                    progress_callback=progress_cb,
                    record_callback=None,
                    store_records=True,
                )
                self.sqlite_store = None
                self.data = data
                self.filtered_data = data.copy()
                self.statistik = TeilenummerStatistik(data=self.data)

            self.metadata = metadata
            self.current_file = filepath
            self.file_label.config(text=Path(filepath).name)
            self._update_meta_label()
            self._refresh_after_load()
            self.status_label.config(text=f"Geladen: {self.statistik.get_record_count()} Datensätze")
        except Exception as exc:
            messagebox.showerror('Fehler', f'Die Datei konnte nicht geladen werden:\n{exc}')
            self.status_label.config(text='Fehler beim Laden')
        finally:
            if progress_ui:
                self._close_progress(progress_ui)

    def _show_progress(self, title):
        window = tk.Toplevel(self)
        window.title(title)
        window.geometry('360x100')
        window.resizable(False, False)
        window.transient(self)
        window.grab_set()  # Modal machen
        ttk.Label(window, text=title).pack(pady=(10, 5))
        progress = ttk.Progressbar(window, orient='horizontal', mode='determinate', length=300)
        progress.pack(pady=5)
        info = ttk.Label(window, text='0%')
        info.pack()
        window.update()
        return {'window': window, 'bar': progress, 'info': info, 'last_update': 0}

    def _update_progress(self, progress_ui, done, total):
        if not progress_ui:
            return
        # Nur alle 5% aktualisieren für bessere Performance
        percent = min(int(done / total * 100), 100)
        if percent - progress_ui.get('last_update', 0) < 5 and percent < 100:
            return
        progress_ui['last_update'] = percent
        bar = progress_ui['bar']
        label = progress_ui['info']
        bar['maximum'] = total
        bar['value'] = done
        label.config(text=f'{percent}%')
        progress_ui['window'].update()

    def _close_progress(self, progress_ui):
        if progress_ui and progress_ui.get('window'):
            try:
                progress_ui['window'].grab_release()
                progress_ui['window'].destroy()
            except:
                pass

    def _update_meta_label(self):
        meta_text = f"Datensätze: {self.statistik.get_record_count():,}".replace(',', '.')
        for key, value in self.metadata.items():
            meta_text += f"  |  {key}: {value}"
        self.meta_label.config(text=meta_text)

    def _refresh_after_load(self):
        self._update_time_filter_options()
        self._apply_filter()
        self._update_top_list()
        # Lagerhaltung wird manuell über Tab aktualisiert
        self._update_time_analysis()
        self._update_summary()
        # Chart wird erst bei Bedarf geladen

    # --- Filter -----------------------------------------------------------
    def _update_time_filter_options(self):
        options = ['Alle']
        if not self.statistik:
            self.time_filter_combo['values'] = options
            self.time_filter_var.set('Alle')
            return

        if self.period_var.get() == 'monatlich':
            options += sorted(self.statistik.get_monthly_data().keys())
        elif self.period_var.get() == '3-monatlich (Quartal)':
            options += sorted(self.statistik.get_quarterly_data().keys())
        self.time_filter_combo['values'] = options
        self.time_filter_var.set('Alle')

    def _on_period_change(self, _event=None):
        self._update_time_filter_options()
        self._apply_filter()

    def _on_time_filter_change(self, _event=None):
        self._apply_filter()

    def _apply_filter(self):
        if not self.statistik:
            return
        period = self.period_var.get()
        value = self.time_filter_var.get()

        if period == 'monatlich' and value != 'Alle':
            self.filter_params = {'type': 'monat', 'value': value}
        elif period == '3-monatlich (Quartal)' and value != 'Alle':
            self.filter_params = {'type': 'quartal', 'value': value}
        else:
            self.filter_params = {'type': 'alle', 'value': None}

        if self.sqlite_store:
            count = self.statistik.get_record_count(self.filter_params)
            self.filter_info_label.config(text=f"Gefiltert: {count} Datensätze")
        else:
            if self.filter_params['type'] == 'alle':
                self.filtered_data = self.data.copy()
            else:
                self.filtered_data = self.statistik._filter_data(self.data, self.filter_params)
            self.filter_info_label.config(text=f"Gefiltert: {len(self.filtered_data)} Datensätze")
        self._update_top_list()
        self._load_all_data()
        self._update_time_analysis()
        self._update_chart()

    # --- Tabellen & Anzeigen ----------------------------------------------
    def _update_top_list(self):
        if not self.statistik:
            return
        for item in self.top_tree.get_children():
            self.top_tree.delete(item)
        try:
            n = int(self.top_n_var.get())
        except ValueError:
            n = 20
        data = self.filtered_data if not self.sqlite_store else None
        top_items = self.statistik.get_top_n(n=n, by=self.sort_var.get(), data=data, filters=self.filter_params)
        for item in top_items:
            self.top_tree.insert('', 'end', values=(
                item['teilenummer'],
                item.get('bezeichnung', ''),
                item.get('anzahl_vorgaenge', 0),
                f"{item.get('gesamtmenge', 0.0):.2f}",
                f"{item.get('gesamtumsatz', 0.0):.2f}",
                item.get('anzahl_kunden', 0),
            ))

    def _update_lagerhaltung(self):
        """Aktualisiert die Lagerhaltungs-Analyse Tabelle."""
        if not self.statistik:
            return
        
        for item in self.lager_tree.get_children():
            self.lager_tree.delete(item)
        
        # Nur einmal berechnen!
        alle_analyse = self.statistik.get_lagerhaltung_analyse()
        
        # Statistiken berechnen
        lohnend = sum(1 for a in alle_analyse if 'Lohnend' in a['kategorie'])
        grenzwertig = sum(1 for a in alle_analyse if 'Grenzwertig' in a['kategorie'])
        nicht_lohnend = sum(1 for a in alle_analyse if 'Nicht lohnend' in a['kategorie'])
        
        self.lager_stats_label.config(
            text=f"✅ Lohnend: {lohnend}  |  ⚠️ Grenzwertig: {grenzwertig}  |  ❌ Nicht lohnend: {nicht_lohnend}"
        )
        
        # Filter anwenden
        filter_val = self.lager_filter_var.get()
        if filter_val == 'nur lohnend':
            analyse = [a for a in alle_analyse if 'Lohnend' in a['kategorie']]
        elif filter_val == 'nur grenzwertig':
            analyse = [a for a in alle_analyse if 'Grenzwertig' in a['kategorie']]
        elif filter_val == 'nur nicht lohnend':
            analyse = [a for a in alle_analyse if 'Nicht lohnend' in a['kategorie']]
        else:
            analyse = alle_analyse
        
        # Tabelle füllen
        for item in analyse:
            tage_str = f"{item['durchschnitt_tage']:.0f}" if item['durchschnitt_tage'] else "-"
            self.lager_tree.insert('', 'end', values=(
                item['teilenummer'],
                item['bezeichnung'],
                item['kategorie'],
                tage_str,
                item['anzahl_verkäufe'],
                f"{item['gesamtmenge']:.2f}",
                f"{item['gesamtumsatz']:.2f}",
                item['empfehlung'],
            ))

    def _export_lagerhaltung(self):
        """Exportiert die Lagerhaltungs-Analyse als CSV."""
        if not self.statistik:
            messagebox.showwarning('Hinweis', 'Keine Daten vorhanden.')
            return
        
        filepath = filedialog.asksaveasfilename(
            title='Lagerhaltungs-Analyse exportieren',
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv')],
            initialfile=f"lagerhaltung_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        if not filepath:
            return
        
        analyse = self.statistik.get_lagerhaltung_analyse()
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as handle:
                writer = csv.writer(handle, delimiter=';')
                writer.writerow([
                    'Teilenummer', 'Bezeichnung', 'Kategorie', 'Ø Tage zwischen Verkäufen',
                    'Anzahl Verkäufe', 'Gesamtmenge', 'Gesamtumsatz (€)', 'Empfehlung'
                ])
                for item in analyse:
                    tage_str = f"{item['durchschnitt_tage']:.0f}" if item['durchschnitt_tage'] else ""
                    writer.writerow([
                        item['teilenummer'],
                        item['bezeichnung'],
                        item['kategorie'].replace('✅ ', '').replace('⚠️ ', '').replace('❌ ', ''),
                        tage_str,
                        item['anzahl_verkäufe'],
                        f"{item['gesamtmenge']:.2f}".replace('.', ','),
                        f"{item['gesamtumsatz']:.2f}".replace('.', ','),
                        item['empfehlung'],
                    ])
            messagebox.showinfo('Export', f'Lagerhaltungs-Analyse gespeichert:\n{filepath}')
        except Exception as exc:
            messagebox.showerror('Fehler', f'Export fehlgeschlagen:\n{exc}')

    def _load_all_data(self, search_term=None):
        if not self.statistik:
            return
        for item in self.all_tree.get_children():
            self.all_tree.delete(item)
        records = self.statistik.fetch_records(self.filter_params, search_term, limit=1000)
        for record in records:
            self.all_tree.insert('', 'end', values=(
                record['teilenummer'],
                record.get('bezeichnung', ''),
                record.get('abgabe_datum', ''),
                f"{record.get('menge', 0.0):.2f}",
                record.get('kd_name', ''),
            ))

    def _on_search(self, *_):
        self._load_all_data(self.search_var.get())

    def _update_time_analysis(self):
        if not self.statistik:
            return
        for item in self.time_tree.get_children():
            self.time_tree.delete(item)
        period_mode = 'quartal' if self.period_var.get() == '3-monatlich (Quartal)' else 'monat'
        data = self.statistik.get_quarterly_data() if period_mode == 'quartal' else self.statistik.get_monthly_data()
        for key in sorted(data.keys()):
            entry = data[key]
            self.time_tree.insert('', 'end', values=(
                key,
                entry.get('vorgaenge', 0),
                f"{entry.get('menge', 0.0):.2f}",
                f"{entry.get('umsatz', 0.0):.2f}",
                entry.get('top_teil', '-'),
            ))

    # --- Diagramme -------------------------------------------------------
    def _update_chart(self):
        if not self.statistik:
            return
        self._ensure_chart_initialized()
        self.figure.clear()
        chart_type = self.chart_type_var.get()
        ax = self.figure.add_subplot(111)

        if chart_type == 'top_bar':
            self._chart_top_bar(ax)
        elif chart_type == 'top_pie':
            self._chart_top_pie(ax)
        elif chart_type == 'zeitverlauf_vorgaenge':
            self._chart_time(ax, metric='vorgaenge')
        elif chart_type == 'zeitverlauf_umsatz':
            self._chart_time(ax, metric='umsatz')
        elif chart_type == 'monatlich_vergleich':
            self._chart_monthly_compare()
        self.figure.tight_layout()
        self.canvas.draw()

    def _chart_top_bar(self, ax):
        data = self.filtered_data if not self.sqlite_store else None
        items = self.statistik.get_top_n(10, by='vorgaenge', data=data, filters=self.filter_params)
        if not items:
            ax.text(0.5, 0.5, 'Keine Daten', ha='center', va='center')
            return
        parts = [item['teilenummer'] for item in items]
        values = [item['anzahl_vorgaenge'] for item in items]
        bars = ax.barh(parts, values, color='steelblue')
        ax.set_xlabel('Anzahl Vorgänge')
        ax.invert_yaxis()
        for bar, value in zip(bars, values):
            ax.text(value + 0.3, bar.get_y() + bar.get_height() / 2, str(value), va='center')

    def _chart_top_pie(self, ax):
        data = self.filtered_data if not self.sqlite_store else None
        items = self.statistik.get_top_n(8, by='vorgaenge', data=data, filters=self.filter_params)
        if not items:
            ax.text(0.5, 0.5, 'Keine Daten', ha='center', va='center')
            return
        labels = [f"{item['teilenummer']}\n({item['anzahl_vorgaenge']})" for item in items]
        sizes = [item['anzahl_vorgaenge'] for item in items]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Top Teilenummern (Vorgänge)')

    def _chart_time(self, ax, metric='vorgaenge'):
        monthly = self.statistik.get_monthly_data()
        if not monthly:
            ax.text(0.5, 0.5, 'Keine Datumswerte', ha='center', va='center')
            return
        keys = sorted(monthly.keys())
        values = [monthly[k]['vorgaenge' if metric == 'vorgaenge' else 'umsatz'] for k in keys]
        color = 'steelblue' if metric == 'vorgaenge' else 'forestgreen'
        ax.bar(keys, values, color=color)
        ax.set_xticklabels(keys, rotation=45, ha='right')
        ax.set_ylabel('Anzahl Vorgänge' if metric == 'vorgaenge' else 'Umsatz (€)')
        avg = sum(values) / len(values) if values else 0
        ax.axhline(avg, color='red', linestyle='--', label=f'Durchschnitt {avg:.1f}')
        ax.legend()

    def _chart_monthly_compare(self):
        ax1 = self.figure.add_subplot(111)
        ax2 = ax1.twinx()
        monthly = self.statistik.get_monthly_data()
        if not monthly:
            ax1.text(0.5, 0.5, 'Keine Datumswerte', ha='center', va='center')
            return
        keys = sorted(monthly.keys())
        idx = list(range(len(keys)))
        vorgaenge = [monthly[k]['vorgaenge'] for k in keys]
        umsatz = [monthly[k]['umsatz'] for k in keys]
        width = 0.4
        ax1.bar([i - width / 2 for i in idx], vorgaenge, width, label='Vorgänge', color='steelblue')
        ax2.bar([i + width / 2 for i in idx], umsatz, width, label='Umsatz (€)', color='forestgreen')
        ax1.set_xticks(idx)
        ax1.set_xticklabels(keys, rotation=45, ha='right')
        ax1.set_ylabel('Vorgänge')
        ax2.set_ylabel('Umsatz (€)')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

    def _save_chart(self):
        if not self.statistik or self.figure is None:
            return
        filepath = filedialog.asksaveasfilename(
            title='Diagramm speichern',
            defaultextension='.png',
            filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg')],
        )
        if filepath:
            self.figure.savefig(filepath, dpi=150, bbox_inches='tight')
            messagebox.showinfo('Gespeichert', filepath)

    # --- Zusammenfassung --------------------------------------------------
    def _update_summary(self):
        if not self.statistik:
            return
        summary = self.statistik.get_summary()
        self.summary_text.delete(1.0, 'end')
        lines = [
            '=' * 70,
            'TEILENUMMER-ANALYSE - ZUSAMMENFASSUNG',
            '=' * 70,
            '',
            f"Datei: {Path(self.current_file).name if self.current_file else '-'}",
            f"Analysezeitpunkt: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            '',
            '-' * 70,
            'ALLGEMEINE STATISTIK:',
            '-' * 70,
            f"  Gesamtanzahl Datensätze:   {summary['rows']:>10}",
            f"  Verschiedene Teilenummern: {summary['unique_parts']:>10}",
            f"  Gesamtmenge:               {summary['total_qty']:>10.2f}",
            f"  Gesamtumsatz:              {summary['total_revenue']:>10.2f} €",
        ]
        self.summary_text.insert('end', '\n'.join(lines))

    # --- Export -----------------------------------------------------------
    def _export_results(self):
        if not self.statistik:
            messagebox.showwarning('Hinweis', 'Keine Daten vorhanden.')
            return
        filepath = filedialog.asksaveasfilename(
            title='Exportieren',
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv')],
            initialfile=f"teilenummer_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        if not filepath:
            return
        data = self.filtered_data if not self.sqlite_store else None
        details = self.statistik.get_teilenummer_details(data)
        try:
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as handle:
                writer = csv.writer(handle, delimiter=';')
                writer.writerow(['Teilenummer', 'Bezeichnung', 'Vorgänge', 'Gesamtmenge', 'Gesamtumsatz', 'Anz. Kunden'])
                for tnr, info in sorted(details.items(), key=lambda item: item[1]['anzahl_vorgaenge'], reverse=True):
                    writer.writerow([
                        tnr,
                        info.get('bezeichnung', ''),
                        info.get('anzahl_vorgaenge', 0),
                        f"{info.get('gesamtmenge', 0.0):.2f}".replace('.', ','),
                        f"{info.get('gesamtumsatz', 0.0):.2f}".replace('.', ','),
                        info.get('anzahl_kunden', 0),
                    ])
            messagebox.showinfo('Export', f'Datei gespeichert: {filepath}')
        except Exception as exc:
            messagebox.showerror('Fehler', f'Export fehlgeschlagen:\n{exc}')


# -----------------------------------------------------------------------------
# Start
# -----------------------------------------------------------------------------
def main():
    app = AnalyseApp()
    app.mainloop()


if __name__ == '__main__':
    main()
