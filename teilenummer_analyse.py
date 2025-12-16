#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teilenummer-Analysesystem (erweitert)
- Plattformübergreifend (Windows/macOS)
- Fortschrittsanzeige beim Import großer Dateien
- SQLite-Datenbank für performante Datenverarbeitung
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
from datetime import datetime, timedelta
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

    def get_monthly_data_filtered(self, teilenummern_filter, metric='vorgaenge'):
        """Monatliche Daten für spezifische Teilenummern."""
        placeholders = ','.join('?' * len(teilenummern_filter))
        query = f"""
            SELECT 
                substr(abgabe_iso, 1, 7) AS periode,
                COUNT(*) AS vorgaenge,
                SUM(menge) AS menge,
                SUM(vk_preis) AS umsatz
            FROM records
            WHERE abgabe_iso IS NOT NULL AND UPPER(teilenummer) IN ({placeholders})
            GROUP BY periode
            ORDER BY periode
        """
        rows = self.conn.execute(query, teilenummern_filter).fetchall()
        return {row[0]: {'periode': row[0], 'vorgaenge': row[1], 'menge': row[2], 'umsatz': row[3]} for row in rows}

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

    def get_date_range(self):
        """Ermittelt den Datumsbereich der Daten."""
        if self.db_store:
            query = "SELECT MIN(abgabe_iso) as min_date, MAX(abgabe_iso) as max_date FROM records WHERE abgabe_iso IS NOT NULL"
            result = self.db_store.conn.execute(query).fetchone()
            if result and result[0] and result[1]:
                return result[0], result[1]
            return None, None
        
        dates = [r.get('abgabe_iso') for r in self.data if r.get('abgabe_iso')]
        if dates:
            return min(dates), max(dates)
        return None, None

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

    def get_monthly_data_filtered(self, teilenummern_filter, metric='vorgaenge'):
        """Monatliche Daten für spezifische Teilenummern."""
        if self.db_store:
            return self.db_store.get_monthly_data_filtered(teilenummern_filter, metric)
        
        monthly = defaultdict(lambda: {'periode': '', 'vorgaenge': 0, 'menge': 0.0, 'umsatz': 0.0})
        for record in self.data:
            if record['teilenummer'].upper() not in teilenummern_filter:
                continue
            iso = record.get('abgabe_iso')
            if not iso:
                continue
            key = iso[:7]
            monthly[key]['periode'] = key
            monthly[key]['vorgaenge'] += 1
            monthly[key]['menge'] += record['menge']
            monthly[key]['umsatz'] += record['vk_preis']
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

    def get_lagerhaltung_analyse(self, max_tage_lohnend=60, monate=None):
        """
        Analysiert welche Teile sich lohnen im Lager zu halten.
        
        Kategorien:
        - "Lohnend": Verkauf alle 1-2 Monate (≤60 Tage zwischen Verkäufen)
        - "Grenzwertig": Verkauf alle 2-4 Monate (61-120 Tage)
        - "Nicht lohnend": Verkauf seltener als alle 4 Monate (>120 Tage)
        
        Args:
            max_tage_lohnend: Maximale Tage zwischen Verkäufen für "Lohnend"
            monate: Anzahl der Monate rückwärts vom letzten Datum (None = alle Daten)
        """
        if self.db_store:
            return self._get_lagerhaltung_from_db(max_tage_lohnend, monate)
        
        # Filtere Daten nach Zeitraum wenn angegeben
        dataset = self.data
        if monate is not None:
            # Finde das letzte Datum in den Daten (nicht heute!)
            alle_daten = [r.get('abgabe_iso', '') for r in self.data if r.get('abgabe_iso')]
            if alle_daten:
                letztes_datum_str = max(alle_daten)
                letztes_datum = datetime.strptime(letztes_datum_str, '%Y-%m-%d')
            else:
                letztes_datum = datetime.now()
            
            stichtag = letztes_datum - timedelta(days=monate * 30.44)
            stichtag_iso = stichtag.strftime('%Y-%m-%d')
            dataset = [r for r in self.data if r.get('abgabe_iso', '') >= stichtag_iso]
        
        # Sammle alle Verkaufsdaten pro Teilenummer
        teil_verkäufe = defaultdict(list)
        for record in dataset:
            iso = record.get('abgabe_iso')
            if iso:
                teil_verkäufe[record['teilenummer']].append({
                    'datum': iso,
                    'bezeichnung': record['bezeichnung'],
                    'menge': record['menge'],
                    'umsatz': record['vk_preis'],
                })
        
        # Berechne den GESAMTEN Zeitraum der Daten (für alle Teile gleich!)
        alle_daten_iso = [record.get('abgabe_iso') for record in dataset if record.get('abgabe_iso')]
        if len(alle_daten_iso) >= 2:
            erstes_datum_gesamt = datetime.strptime(min(alle_daten_iso), '%Y-%m-%d')
            letztes_datum_gesamt = datetime.strptime(max(alle_daten_iso), '%Y-%m-%d')
            gesamt_zeitraum_monate = max(1, ((letztes_datum_gesamt.year - erstes_datum_gesamt.year) * 12 + 
                                            letztes_datum_gesamt.month - erstes_datum_gesamt.month) + 1)
        else:
            gesamt_zeitraum_monate = 1
        
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
            
            # Berechne Monats-Durchschnitt
            # Zeitspanne ermitteln
            sorted_dates = sorted([v['datum'] for v in verkäufe])
            if len(sorted_dates) >= 1:
                d1 = datetime.strptime(sorted_dates[0], '%Y-%m-%d')
                d2 = datetime.strptime(sorted_dates[-1], '%Y-%m-%d')
                tage_gesamt = (d2 - d1).days
                monate_gesamt = max(1, tage_gesamt / 30.44)  # Durchschnittliche Tage pro Monat
                monatsdurchschnitt_menge = gesamtmenge / monate_gesamt
                monatsdurchschnitt_umsatz = gesamtumsatz / monate_gesamt
            else:
                monatsdurchschnitt_menge = gesamtmenge
                monatsdurchschnitt_umsatz = gesamtumsatz
            
            # 5. Kundenabhängigkeit
            unique_kunden = len({r.get('kd_name', '') for r in [rec for rec in dataset if rec['teilenummer'] == teilenummer]})
            
            # 4. Trendanalyse (erste vs. zweite Hälfte)
            trend = self._berechne_trend(verkäufe)
            
            # 3. Saisonalität
            saisonalität = self._erkenne_saisonalitaet(verkäufe)
            
            # 11. Verbrauchsprognose (nächste 3 Monate)
            prognose_3m = monatsdurchschnitt_menge * 3
            
            # Berechne in wie vielen verschiedenen Monaten verkauft wurde
            verkaufs_monate = set()
            for v in verkäufe:
                datum = datetime.strptime(v['datum'], '%Y-%m-%d')
                verkaufs_monate.add((datum.year, datum.month))
            anzahl_verkaufsmonate = len(verkaufs_monate)
            
            # Verwende den GESAMTEN Zeitraum der Daten (für alle Teile gleich!)
            zeitraum_monate = gesamt_zeitraum_monate
            
            # Lagerfähigkeit: Prozent der Monate mit Verkauf
            lagerfaehigkeit_prozent = (anzahl_verkaufsmonate / zeitraum_monate) * 100 if zeitraum_monate > 0 else 0
            
            # Lagerfähigkeits-Status
            if lagerfaehigkeit_prozent >= 80:
                lagerfaehig = "✅ Ja"
            elif lagerfaehigkeit_prozent >= 50:
                lagerfaehig = "⚠️ Bedingt"
            else:
                lagerfaehig = "❌ Nein"
            
            ergebnis.append({
                'teilenummer': teilenummer,
                'bezeichnung': verkäufe[0]['bezeichnung'],
                'anzahl_verkäufe': len(verkäufe),
                'durchschnitt_tage': avg_tage if avg_tage < 999 else None,
                'gesamtmenge': gesamtmenge,
                'gesamtumsatz': gesamtumsatz,
                'monatsdurchschnitt_menge': monatsdurchschnitt_menge,
                'monatsdurchschnitt_umsatz': monatsdurchschnitt_umsatz,
                'anzahl_kunden': unique_kunden,
                'trend': trend,
                'saisonalität': saisonalität,
                'prognose_3_monate': prognose_3m,
                'anzahl_verkaufsmonate': anzahl_verkaufsmonate,
                'zeitraum_monate': zeitraum_monate,
                'lagerfaehigkeit_prozent': lagerfaehigkeit_prozent,
                'lagerfaehig': lagerfaehig,
                'kategorie': kategorie,
                'empfehlung': empfehlung,
            })

        # ═══════════════════════════════════════════════════════════════
        # ABC-ANALYSE (nach Pareto-Prinzip)
        # ═══════════════════════════════════════════════════════════════
        # Sortiere nach Umsatz (höchster zuerst)
        ergebnis_sortiert = sorted(ergebnis, key=lambda x: x['gesamtumsatz'], reverse=True)
        gesamt_umsatz_alle = sum(item['gesamtumsatz'] for item in ergebnis)

        kumulativ_umsatz = 0
        kumulativ_prozent_teile = 0
        for i, item in enumerate(ergebnis_sortiert):
            kumulativ_umsatz += item['gesamtumsatz']
            kumulativ_prozent_teile = ((i + 1) / len(ergebnis_sortiert)) * 100 if ergebnis_sortiert else 0
            kumulativ_prozent_umsatz = (kumulativ_umsatz / gesamt_umsatz_alle * 100) if gesamt_umsatz_alle > 0 else 0

            # ABC-Klassifizierung nach Pareto
            if kumulativ_prozent_umsatz <= 80:
                item['abc'] = '🅰️'  # A-Teile: Top 20% machen 80% Umsatz
            elif kumulativ_prozent_umsatz <= 95:
                item['abc'] = '🅱️'  # B-Teile: Nächste 30% machen 15% Umsatz
            else:
                item['abc'] = '🅲️'  # C-Teile: Restliche 50% machen 5% Umsatz

        return sorted(ergebnis, key=lambda x: x['durchschnitt_tage'] or 9999)

    def _berechne_trend(self, verkäufe):
        """Berechnet den Trend: Steigend/Fallend/Stabil."""
        if len(verkäufe) < 4:
            return "Stabil"
        
        sorted_verkäufe = sorted(verkäufe, key=lambda x: x['datum'])
        mid = len(sorted_verkäufe) // 2
        
        erste_hälfte_menge = sum(v['menge'] for v in sorted_verkäufe[:mid])
        zweite_hälfte_menge = sum(v['menge'] for v in sorted_verkäufe[mid:])
        
        if zweite_hälfte_menge > erste_hälfte_menge * 1.2:
            prozent = int(((zweite_hälfte_menge / erste_hälfte_menge) - 1) * 100)
            return f"↗️ +{prozent}%"
        elif zweite_hälfte_menge < erste_hälfte_menge * 0.8:
            prozent = int((1 - (zweite_hälfte_menge / erste_hälfte_menge)) * 100)
            return f"↘️ -{prozent}%"
        else:
            return "→ Stabil"
    
    def _erkenne_saisonalitaet(self, verkäufe):
        """Erkennt Saisonalität nach Quartalen."""
        if len(verkäufe) < 6:
            return "k.A."
        
        # Verkäufe nach Quartal gruppieren
        quartal_verkäufe = defaultdict(float)
        for v in verkäufe:
            datum = datetime.strptime(v['datum'], '%Y-%m-%d')
            quartal = (datum.month - 1) // 3 + 1
            quartal_verkäufe[quartal] += v['menge']
        
        if len(quartal_verkäufe) < 2:
            return "k.A."
        
        max_q = max(quartal_verkäufe, key=quartal_verkäufe.get)
        max_menge = quartal_verkäufe[max_q]
        avg_menge = sum(quartal_verkäufe.values()) / len(quartal_verkäufe)
        
        if max_menge > avg_menge * 1.5:
            return f"Q{max_q}"
        return "Gleichmäßig"

    def _get_lagerhaltung_from_db(self, max_tage_lohnend=60, monate=None):
        """SQLite-Version der Lagerhaltungsanalyse."""
        # WHERE-Klausel für Zeitfilter - basierend auf dem letzten Datum in den Daten
        where_clause = ""
        if monate is not None:
            # Finde das letzte Datum in der Datenbank
            max_datum = self.db_store.conn.execute(
                "SELECT MAX(abgabe_iso) FROM records WHERE abgabe_iso IS NOT NULL"
            ).fetchone()[0]
            if max_datum:
                letztes_datum = datetime.strptime(max_datum, '%Y-%m-%d')
            else:
                letztes_datum = datetime.now()
            
            stichtag = letztes_datum - timedelta(days=monate * 30.44)
            stichtag_iso = stichtag.strftime('%Y-%m-%d')
            where_clause = f"WHERE abgabe_iso >= '{stichtag_iso}'"
        
        query = f"""
            WITH verkauf_daten AS (
                SELECT 
                    teilenummer,
                    MAX(bezeichnung) AS bezeichnung,
                    abgabe_iso,
                    SUM(menge) AS menge,
                    SUM(vk_preis) AS umsatz,
                    COUNT(*) AS anzahl
                FROM records
                WHERE abgabe_iso IS NOT NULL {(' AND ' + where_clause.replace('WHERE ', '')) if where_clause else ''}
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
                    MAX(abgabe_iso) AS letzter_verkauf,
                    CAST(julianday(MAX(abgabe_iso)) - julianday(MIN(abgabe_iso)) AS REAL) AS tage_gesamt
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
                END AS durchschnitt_tage,
                tage_gesamt,
                CASE 
                    WHEN tage_gesamt > 0 THEN gesamtmenge / (tage_gesamt / 30.44)
                    ELSE gesamtmenge
                END AS monatsdurchschnitt_menge,
                CASE 
                    WHEN tage_gesamt > 0 THEN gesamtumsatz / (tage_gesamt / 30.44)
                    ELSE gesamtumsatz
                END AS monatsdurchschnitt_umsatz
            FROM teil_stats
            ORDER BY durchschnitt_tage NULLS LAST
        """
        cur = self.db_store.conn.execute(query)
        
        # Berechne den GESAMTEN Zeitraum der Daten (für alle Teile gleich!)
        gesamt_query = """
            SELECT MIN(abgabe_iso), MAX(abgabe_iso) 
            FROM records 
            WHERE abgabe_iso IS NOT NULL
        """
        if monate is not None:
            # Mit Zeitfilter - nutze where_clause
            gesamt_query = f"""
                SELECT MIN(abgabe_iso), MAX(abgabe_iso) 
                FROM records 
                WHERE abgabe_iso IS NOT NULL {(' AND ' + where_clause.replace('WHERE ', '')) if where_clause else ''}
            """
        gesamt_result = self.db_store.conn.execute(gesamt_query).fetchone()
        if gesamt_result and gesamt_result[0] and gesamt_result[1]:
            erstes_datum_gesamt = datetime.strptime(gesamt_result[0], '%Y-%m-%d')
            letztes_datum_gesamt = datetime.strptime(gesamt_result[1], '%Y-%m-%d')
            gesamt_zeitraum_monate = max(1, ((letztes_datum_gesamt.year - erstes_datum_gesamt.year) * 12 + 
                                            letztes_datum_gesamt.month - erstes_datum_gesamt.month) + 1)
        else:
            gesamt_zeitraum_monate = 1
        
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
            
            # Berechne zusätzliche Metriken für SQLite-Daten
            teilenummer = row[0]
            
            # 5. Kundenanzahl
            kunden_query = f"SELECT COUNT(DISTINCT kd_name) FROM records WHERE teilenummer = ?"
            anzahl_kunden = self.db_store.conn.execute(kunden_query, (teilenummer,)).fetchone()[0]
            
            # 4. & 3. Trend und Saisonalität - hole Verkaufsdaten
            verkauf_query = """
                SELECT abgabe_iso, menge FROM records 
                WHERE teilenummer = ? AND abgabe_iso IS NOT NULL 
                ORDER BY abgabe_iso
            """
            verkäufe = [{'datum': r[0], 'menge': r[1]} for r in self.db_store.conn.execute(verkauf_query, (teilenummer,)).fetchall()]
            trend = self._berechne_trend(verkäufe) if len(verkäufe) >= 4 else "Stabil"
            saisonalität = self._erkenne_saisonalitaet(verkäufe) if len(verkäufe) >= 6 else "k.A."
            
            # Berechne in wie vielen verschiedenen Monaten verkauft wurde
            verkaufs_monate = set()
            for v in verkäufe:
                datum = datetime.strptime(v['datum'], '%Y-%m-%d')
                verkaufs_monate.add((datum.year, datum.month))
            anzahl_verkaufsmonate = len(verkaufs_monate)
            
            # Verwende den GESAMTEN Zeitraum der Daten (für alle Teile gleich!)
            zeitraum_monate = gesamt_zeitraum_monate
            
            # Lagerfähigkeit: Prozent der Monate mit Verkauf
            lagerfaehigkeit_prozent = (anzahl_verkaufsmonate / zeitraum_monate) * 100 if zeitraum_monate > 0 else 0
            
            # Lagerfähigkeits-Status
            if lagerfaehigkeit_prozent >= 80:
                lagerfaehig = "✅ Ja"
            elif lagerfaehigkeit_prozent >= 50:
                lagerfaehig = "⚠️ Bedingt"
            else:
                lagerfaehig = "❌ Nein"
            
            # 11. Prognose
            prognose_3m = row[7] * 3 if row[7] else 0
            
            ergebnis.append({
                'teilenummer': row[0],
                'bezeichnung': row[1],
                'anzahl_verkäufe': row[2],
                'durchschnitt_tage': avg_tage,
                'gesamtmenge': row[3],
                'gesamtumsatz': row[4],
                'monatsdurchschnitt_menge': row[7],
                'monatsdurchschnitt_umsatz': row[8],
                'anzahl_kunden': anzahl_kunden,
                'trend': trend,
                'saisonalität': saisonalität,
                'prognose_3_monate': prognose_3m,
                'anzahl_verkaufsmonate': anzahl_verkaufsmonate,
                'zeitraum_monate': zeitraum_monate,
                'lagerfaehigkeit_prozent': lagerfaehigkeit_prozent,
                'lagerfaehig': lagerfaehig,
                'kategorie': kategorie,
                'empfehlung': empfehlung,
            })

        # ═══════════════════════════════════════════════════════════════
        # ABC-ANALYSE (nach Pareto-Prinzip)
        # ═══════════════════════════════════════════════════════════════
        # Sortiere nach Umsatz (höchster zuerst)
        ergebnis_sortiert = sorted(ergebnis, key=lambda x: x['gesamtumsatz'], reverse=True)
        gesamt_umsatz_alle = sum(item['gesamtumsatz'] for item in ergebnis)

        kumulativ_umsatz = 0
        for i, item in enumerate(ergebnis_sortiert):
            kumulativ_umsatz += item['gesamtumsatz']
            kumulativ_prozent_umsatz = (kumulativ_umsatz / gesamt_umsatz_alle * 100) if gesamt_umsatz_alle > 0 else 0

            # ABC-Klassifizierung nach Pareto
            if kumulativ_prozent_umsatz <= 80:
                item['abc'] = '🅰️'  # A-Teile: Top 20% machen 80% Umsatz
            elif kumulativ_prozent_umsatz <= 95:
                item['abc'] = '🅱️'  # B-Teile: Nächste 30% machen 15% Umsatz
            else:
                item['abc'] = '🅲️'  # C-Teile: Restliche 50% machen 5% Umsatz

        return ergebnis


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

class TreeviewHeaderTooltip:
    """Tooltip für Treeview-Spaltenüberschriften mit Verzögerung."""
    
    def __init__(self, treeview, tooltips: dict, delay_ms: int = 3000):
        """
        Args:
            treeview: Das ttk.Treeview-Widget
            tooltips: Dict mit {spalten_id: tooltip_text}
            delay_ms: Verzögerung in Millisekunden (Standard: 3000 = 3 Sekunden)
        """
        self.treeview = treeview
        self.tooltips = tooltips
        self.delay_ms = delay_ms
        self.tooltip_window = None
        self.after_id = None
        self.current_column = None
        
        # Events binden
        self.treeview.bind('<Motion>', self._on_motion)
        self.treeview.bind('<Leave>', self._hide_tooltip)
    
    def _on_motion(self, event):
        """Prüft, ob Maus über Überschrift ist."""
        region = self.treeview.identify_region(event.x, event.y)
        
        if region == 'heading':
            column = self.treeview.identify_column(event.x)
            # Konvertiere #1, #2, etc. zu Spalten-ID
            if column:
                try:
                    col_idx = int(column.replace('#', '')) - 1
                    columns = self.treeview['columns']
                    if 0 <= col_idx < len(columns):
                        col_id = columns[col_idx]
                        
                        if col_id != self.current_column:
                            # Neue Spalte - Timer zurücksetzen
                            self._cancel_timer()
                            self._hide_tooltip()
                            self.current_column = col_id
                            
                            if col_id in self.tooltips:
                                self.after_id = self.treeview.after(
                                    self.delay_ms, 
                                    lambda: self._show_tooltip(event.x_root, event.y_root, col_id)
                                )
                except (ValueError, IndexError):
                    pass
        else:
            # Nicht über Überschrift
            self._cancel_timer()
            self._hide_tooltip()
            self.current_column = None
    
    def _show_tooltip(self, x, y, col_id):
        """Zeigt das Tooltip-Fenster."""
        if col_id not in self.tooltips:
            return
        
        self._hide_tooltip()
        
        self.tooltip_window = tk.Toplevel(self.treeview)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x + 10}+{y + 10}")
        
        # Tooltip-Stil
        frame = tk.Frame(self.tooltip_window, background='#ffffcc', borderwidth=1, relief='solid')
        frame.pack()
        
        label = tk.Label(
            frame, 
            text=self.tooltips[col_id],
            background='#ffffcc',
            foreground='#000000',
            font=('Segoe UI', 9),
            justify='left',
            wraplength=300,
            padx=8,
            pady=5
        )
        label.pack()
    
    def _hide_tooltip(self, event=None):
        """Versteckt das Tooltip-Fenster."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
    
    def _cancel_timer(self):
        """Bricht den Timer ab."""
        if self.after_id:
            self.treeview.after_cancel(self.after_id)
            self.after_id = None


class AnalyseApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('LagerPilot by Sven Hube')
        self.geometry('1400x900')

        self.parser = TeilenummerParser()
        self.statistik = None
        self.data = []
        self.filtered_data = []
        self.metadata = {}
        self.current_file = None
        self.filter_params = {'type': 'alle', 'value': None}
        self.sqlite_store: SQLiteDataStore | None = None
        self.alle_produkte_liste = []  # Für Autocomplete
        self.lagerbestand_data = {}  # Lagerbestand: {teilenummer: {bestand, verfuegbar, upe, ...}}

        self._build_ui()
        self._build_menu()

    def _show_auto_close_info(self, title, message, duration=3000):
        """Zeigt eine Info-Nachricht, die sich automatisch nach duration ms schließt."""
        popup = tk.Toplevel(self)
        popup.title(title)
        popup.geometry("400x150")
        popup.resizable(False, False)

        # Zentriere das Fenster über dem Hauptfenster
        popup.transient(self)
        popup.grab_set()

        # Icon und Nachricht
        frame = ttk.Frame(popup, padding=20)
        frame.pack(fill='both', expand=True)

        # Info-Icon (ℹ️) und Text
        ttk.Label(frame, text="ℹ️", font=('Segoe UI', 32)).pack(pady=(0, 10))
        ttk.Label(frame, text=message, wraplength=350, justify='center', font=('Segoe UI', 10)).pack()

        # Automatisches Schließen nach duration ms
        popup.after(duration, popup.destroy)

        # Position zentrieren
        popup.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() // 2) - (popup.winfo_width() // 2)
        y = self.winfo_y() + (self.winfo_height() // 2) - (popup.winfo_height() // 2)
        popup.geometry(f"+{x}+{y}")

    # --- UI ----------------------------------------------------------------
    def _build_ui(self):
        main = ttk.Frame(self, padding='10')
        main.grid(row=0, column=0, sticky='nsew')
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(3, weight=1)

        # Datei + Speicherwahl
        file_frame = ttk.LabelFrame(main, text='Dateien laden', padding='5')
        file_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        file_frame.columnconfigure(2, weight=1)

        ttk.Button(file_frame, text='📊 Verkaufsdaten...', command=self._open_file).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(file_frame, text='📦 Lagerbestand...', command=self._open_lagerbestand_file).grid(row=0, column=1, padx=(0, 15))

        # Immer SQLite verwenden
        self.storage_var = tk.StringVar(value='sqlite')

        self.file_label = ttk.Label(file_frame, text='Keine Dateien geladen')
        self.file_label.grid(row=0, column=2, sticky='w')

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
        self._build_lagerabbau_tab()
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
        
        # Suchfeld
        ttk.Label(control, text='Suche:').grid(row=0, column=5, padx=(20, 5))
        self.top_search_var = tk.StringVar()
        self.top_search_var.trace_add('write', lambda *args: self._search_top_list())
        ttk.Entry(control, textvariable=self.top_search_var, width=25).grid(row=0, column=6)

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
            self.top_tree.heading(col, text=headings[col], command=lambda c=col: self._sort_treeview(self.top_tree, c, False))
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
            "Prognose: 🔴 Zu wenig (<2 Mon.) | ✅ OK (2-12 Mon.) | ⚠️ Zu viel (>12 Mon.) | "
            "💡 Maus 3 Sek. auf Spalte = Tooltip"
        )
        ttk.Label(info_frame, text=info_text, wraplength=1200).grid(row=0, column=0, sticky='w')
        
        # Filter-Steuerung Zeile 1
        control = ttk.Frame(frame)
        control.grid(row=1, column=0, sticky='ew', pady=(0, 5))
        
        # Zeitraum-Variable (intern auf "Alle Daten" gesetzt)
        self.lager_monate_var = tk.StringVar(value='Alle Daten')
        self.lager_zeitraum_label = ttk.Label(control, text='', foreground='blue')
        self.lager_zeitraum_label.grid(row=0, column=0, padx=(0, 15))
        
        # Produkt-Filter (Combobox mit Autocomplete)
        ttk.Label(control, text='Produkte:').grid(row=0, column=1)
        self.lager_produkt_var = tk.StringVar(value='')
        self.lager_produkt_entry = ttk.Combobox(control, textvariable=self.lager_produkt_var, width=35)
        self.lager_produkt_entry.grid(row=0, column=2, padx=(5, 5))
        self.lager_produkt_entry.bind('<KeyRelease>', self._on_produkt_keyrelease)
        self.lager_produkt_entry.bind('<<ComboboxSelected>>', self._on_produkt_selected_and_update)
        self.lager_produkt_entry.bind('<Return>', lambda e: self._update_lagerhaltung())
        
        # Speicher für alle Produkte
        self.alle_produkte_liste = []
        
        ttk.Button(control, text='×', width=2, command=self._clear_produkt_filter_and_update).grid(row=0, column=3, padx=(0, 15))
        
        ttk.Label(control, text='Anzeigen:').grid(row=0, column=4)
        self.lager_filter_var = tk.StringVar(value='alle')
        self.lager_filter_combo = ttk.Combobox(
            control, 
            textvariable=self.lager_filter_var,
            values=['alle', 'nur lohnend', 'nur grenzwertig', 'nur nicht lohnend', 'nur lagerfähig'],
            width=18,
            state='readonly'
        )
        self.lager_filter_combo.grid(row=0, column=5, padx=(5, 15))
        self.lager_filter_combo.bind('<<ComboboxSelected>>', lambda e: self._update_lagerhaltung())
        
        ttk.Button(control, text='CSV Export', command=self._export_lagerhaltung).grid(row=0, column=6)
        
        # Filter-Steuerung Zeile 2 - Zusätzliche Filter
        control2 = ttk.Frame(frame)
        control2.grid(row=2, column=0, sticky='ew', pady=(0, 10))
        
        # Suchfeld (für Teilenummer/Bezeichnung)
        ttk.Label(control2, text='Suche:').grid(row=0, column=0, padx=(0, 5))
        self.lager_search_var = tk.StringVar()
        self.lager_search_var.trace_add('write', lambda *args: self._search_lager_list())
        ttk.Entry(control2, textvariable=self.lager_search_var, width=20).grid(row=0, column=1, padx=(0, 15))
        
        ttk.Label(control2, text='Min. Verkäufe:').grid(row=0, column=2)
        self.lager_min_verkaeufe_var = tk.StringVar(value='0')
        self.lager_min_verkaeufe_var.trace_add('write', lambda *args: self._delayed_update_lagerhaltung())
        ttk.Spinbox(control2, from_=0, to=100, textvariable=self.lager_min_verkaeufe_var, width=6).grid(row=0, column=3, padx=(5, 15))
        
        ttk.Label(control2, text='Min. Ø€/Mon.:').grid(row=0, column=4)
        self.lager_min_umsatz_var = tk.StringVar(value='0')
        self.lager_min_umsatz_var.trace_add('write', lambda *args: self._delayed_update_lagerhaltung())
        ttk.Spinbox(control2, from_=0, to=10000, increment=50, textvariable=self.lager_min_umsatz_var, width=8).grid(row=0, column=5, padx=(5, 15))
        
        ttk.Label(control2, text='Min. Kunden:').grid(row=0, column=6)
        self.lager_min_kunden_var = tk.StringVar(value='0')
        self.lager_min_kunden_var.trace_add('write', lambda *args: self._delayed_update_lagerhaltung())
        ttk.Spinbox(control2, from_=0, to=50, textvariable=self.lager_min_kunden_var, width=6).grid(row=0, column=7, padx=(5, 15))
        
        ttk.Label(control2, text='Min. Mon.%:').grid(row=0, column=8)
        self.lager_min_monatsprozent_var = tk.StringVar(value='0')
        self.lager_min_monatsprozent_var.trace_add('write', lambda *args: self._delayed_update_lagerhaltung())
        ttk.Spinbox(control2, from_=0, to=100, increment=10, textvariable=self.lager_min_monatsprozent_var, width=6).grid(row=0, column=9, padx=(5, 15))
        
        # Statistik-Labels
        self.lager_stats_label = ttk.Label(control2, text='')
        self.lager_stats_label.grid(row=0, column=10, padx=(20, 0), sticky='w')
        
        # Tabelle mit Lagerbestand-Spalten
        columns = ('teilenummer', 'bezeichnung', 'abc', 'bestand', 'umschlag', 'bestellpunkt', 'reichweite', 'prognose', 'verkäufe', 'verk_mon', 'kunden', 'lagerfaehig', 'ø_umsatz', 'trend', 'kategorie', 'empfehlung')
        self.lager_tree = ttk.Treeview(frame, columns=columns, show='headings')
        headings = {
            'teilenummer': 'Teilenummer',
            'bezeichnung': 'Bezeichnung',
            'abc': 'ABC',
            'bestand': 'Bestand',
            'umschlag': 'Umschlag',
            'bestellpunkt': 'Status',
            'reichweite': 'Reichweite',
            'prognose': 'Prognose',
            'verkäufe': 'Verkäufe',
            'verk_mon': 'Stk./Mon.',
            'kunden': 'Kunden',
            'lagerfaehig': 'Lagerfähig',
            'ø_umsatz': 'Ø €/Mon.',
            'trend': 'Trend',
            'kategorie': 'Kategorie',
            'empfehlung': 'Empfehlung',
        }
        widths = {
            'teilenummer': 100, 'bezeichnung': 160, 'abc': 35, 'bestand': 55, 'umschlag': 60, 'bestellpunkt': 50,
            'reichweite': 65, 'prognose': 80, 'verkäufe': 55, 'verk_mon': 60, 'kunden': 50,
            'lagerfaehig': 65, 'ø_umsatz': 65, 'trend': 50, 'kategorie': 85, 'empfehlung': 95
        }
        aligns = {'abc': tk.CENTER, 'bestand': tk.E, 'umschlag': tk.E, 'bestellpunkt': tk.CENTER, 'reichweite': tk.CENTER, 'verkäufe': tk.E, 'verk_mon': tk.E, 'kunden': tk.E, 'ø_umsatz': tk.E}
        for col in columns:
            self.lager_tree.heading(col, text=headings[col], command=lambda c=col: self._sort_treeview(self.lager_tree, c, False))
            self.lager_tree.column(col, width=widths[col], anchor=aligns.get(col, tk.W))
        
        self.lager_tree.grid(row=3, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=self.lager_tree.yview)
        scrollbar.grid(row=3, column=1, sticky='ns')
        self.lager_tree.configure(yscrollcommand=scrollbar.set)
        
        # Tooltips für Spaltenüberschriften (erscheinen nach 3 Sekunden)
        tooltip_texte = {
            'teilenummer': 'Die Artikelnummer des Teils aus dem DMS-System.',
            'bezeichnung': 'Die Bezeichnung/Beschreibung des Artikels.',
            'abc': 'ABC-Analyse (Pareto-Prinzip):\n🅰️ A-Teile = Top 20% der Teile machen 80% des Umsatzes\n    → Hohe Priorität, immer verfügbar halten\n🅱️ B-Teile = Nächste 30% machen 15% des Umsatzes\n    → Mittlere Priorität, regelmäßig prüfen\n🅲️ C-Teile = Restliche 50% machen 5% des Umsatzes\n    → Niedrige Priorität, ggf. abbauen',
            'bestand': 'Aktuelle Lagermenge aus Lagerbestand-Datei.\n- = Keine Lagerbestand-Datei geladen',
            'umschlag': 'Umschlagshäufigkeit (pro Jahr):\nBerechnung = Jahresverkäufe ÷ Ø Bestand\n🟢 >6/Jahr = Fast-Mover (sehr gut)\n🟡 2-6/Jahr = Normal-Mover (OK)\n🔴 <2/Jahr = Slow-Mover (kritisch)',
            'bestellpunkt': 'Bestellstatus:\n🟢 = Bestand OK\n🟡 = Nahe Bestellpunkt, bald bestellen\n🔴 = Unter Bestellpunkt, jetzt bestellen!\n⚪ = Kein Bestand/Bestellpunkt',
            'reichweite': 'Wie lange reicht der Bestand bei aktuellem Absatz?\nBerechnung: Bestand ÷ Stk./Mon.\nBeispiel: 12 Stück ÷ 2 Stk./Mon. = 6 Mon.',
            'prognose': 'Lagerprognose basierend auf Reichweite:\n🔴 Zu wenig = <2 Monate Reichweite\n✅ OK = 2-12 Monate Reichweite\n⚠️ Zu viel = >12 Monate Reichweite\n❌ Kein Abverkauf = Keine Verkäufe',
            'verkäufe': 'Gesamtanzahl aller Verkäufe im gewählten Zeitraum.',
            'verk_mon': 'Durchschnittliche STÜCKZAHL pro Monat.\nBerechnung: Gesamtmenge ÷ Gesamtmonate\nZeigt wie viele Teile im Schnitt pro Monat verkauft werden.',
            'kunden': 'Anzahl verschiedener Kunden, die dieses Teil gekauft haben.',
            'lagerfaehig': 'Bewertung der Lagerfähigkeit:\n✅ = ≥80% der Monate mit Verkauf (sehr lagerfähig)\n⚠️ = 50-79% der Monate (bedingt lagerfähig)\n❌ = <50% der Monate (wenig lagerfähig)',
            'ø_umsatz': 'Durchschnittlicher Umsatz in Euro pro Monat\n(über den gesamten Zeitraum).',
            'trend': 'Verkaufsentwicklung der letzten Monate:\n📈 steigend = mehr Verkäufe\n➡️ stabil = gleichbleibend\n📉 fallend = weniger Verkäufe',
            'kategorie': 'Gesamtbewertung basierend auf allen Faktoren:\n🟢 Lohnend = Empfohlen für Lager\n🟡 Grenzwertig = Einzelfallentscheidung\n🔴 Nicht lohnend = Nicht für Lager empfohlen',
            'empfehlung': 'Konkrete Handlungsempfehlung für diesen Artikel\nbasierend auf der Analyse aller Verkaufsdaten.',
        }
        TreeviewHeaderTooltip(self.lager_tree, tooltip_texte, delay_ms=3000)
        
        # 9. Farbliche Hervorhebung
        self.lager_tree.tag_configure('lohnend', background='#d4edda')  # Hellgrün
        self.lager_tree.tag_configure('grenzwertig', background='#fff3cd')  # Hellgelb
        self.lager_tree.tag_configure('nicht_lohnend', background='#f8d7da')  # Hellrot
        
        frame.rowconfigure(3, weight=1)
        frame.columnconfigure(0, weight=1)

    def _build_lagerabbau_tab(self):
        """Tab für Lagerabbau: Teile im Lager die nie verkauft wurden."""
        frame = ttk.Frame(self.notebook, padding='2')
        self.notebook.add(frame, text='🗑️ Lagerabbau')

        # ═══════════════════════════════════════════════════════════════════
        # QUICK-BUTTONS (Schnellfilter)
        # ═══════════════════════════════════════════════════════════════════
        quick_frame = ttk.LabelFrame(frame, text='⚡ Schnellfilter', padding='2')
        quick_frame.grid(row=0, column=0, sticky='ew', pady=(0, 2))
        
        ttk.Button(quick_frame, text='🔴 Kritische (>500€ + >1J)', 
                   command=self._quick_filter_kritisch).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(quick_frame, text='❌ Nie verkauft (0)', 
                   command=self._quick_filter_nie_verkauft).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(quick_frame, text='💰 Top 20 Lagerwert', 
                   command=self._quick_filter_top_wert).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(quick_frame, text='📅 Top 20 Lagerdauer', 
                   command=self._quick_filter_top_dauer).grid(row=0, column=3, padx=(0, 10))
        ttk.Button(quick_frame, text='⚠️ Alte Teile (>2 Jahre)', 
                   command=self._quick_filter_alte_teile).grid(row=0, column=4, padx=(0, 10))
        ttk.Button(quick_frame, text='🔄 Filter zurücksetzen', 
                   command=self._quick_filter_reset).grid(row=0, column=5, padx=(0, 10))
        
        # ═══════════════════════════════════════════════════════════════════
        # FILTER-STEUERUNG Zeile 1
        # ═══════════════════════════════════════════════════════════════════
        control = ttk.Frame(frame)
        control.grid(row=1, column=0, sticky='ew', pady=(0, 2))
        
        # Bezeichnung-Filter (Combobox mit Autocomplete wie bei Lagerhaltung)
        ttk.Label(control, text='Bezeichnung:').grid(row=0, column=0)
        self.abbau_bezeichnung_var = tk.StringVar(value='')
        self.abbau_bezeichnung_entry = ttk.Combobox(control, textvariable=self.abbau_bezeichnung_var, width=30)
        self.abbau_bezeichnung_entry.grid(row=0, column=1, padx=(5, 5))
        self.abbau_bezeichnung_entry.bind('<KeyRelease>', self._on_abbau_bezeichnung_keyrelease)
        self.abbau_bezeichnung_entry.bind('<<ComboboxSelected>>', lambda e: self._update_lagerabbau())
        self.abbau_bezeichnung_entry.bind('<Return>', lambda e: self._update_lagerabbau())
        self.abbau_bezeichnung_var.trace_add('write', lambda *args: self._delayed_update_lagerabbau())
        
        ttk.Button(control, text='×', width=2, command=self._clear_abbau_bezeichnung_filter).grid(row=0, column=2, padx=(0, 15))
        
        # Teilenummer-Filter
        ttk.Label(control, text='Suche (Nr./Bez.):').grid(row=0, column=3)
        self.abbau_search_var = tk.StringVar()
        self.abbau_search_var.trace_add('write', lambda *args: self._delayed_update_lagerabbau())
        ttk.Entry(control, textvariable=self.abbau_search_var, width=18).grid(row=0, column=4, padx=(5, 15))
        
        ttk.Button(control, text='CSV Export', command=self._export_lagerabbau).grid(row=0, column=5)
        
        # ═══════════════════════════════════════════════════════════════════
        # FILTER-STEUERUNG Zeile 2 - Numerische Filter
        # ═══════════════════════════════════════════════════════════════════
        control2 = ttk.Frame(frame)
        control2.grid(row=2, column=0, sticky='ew', pady=(0, 2))
        
        ttk.Label(control2, text='Min. Lagerwert €:').grid(row=0, column=0)
        self.abbau_min_wert_var = tk.StringVar(value='0')
        self.abbau_min_wert_var.trace_add('write', lambda *args: self._delayed_update_lagerabbau())
        ttk.Spinbox(control2, from_=0, to=10000, increment=50, textvariable=self.abbau_min_wert_var, width=8).grid(row=0, column=1, padx=(5, 15))
        
        ttk.Label(control2, text='Min. Bestand:').grid(row=0, column=2)
        self.abbau_min_bestand_var = tk.StringVar(value='1')
        self.abbau_min_bestand_var.trace_add('write', lambda *args: self._delayed_update_lagerabbau())
        ttk.Spinbox(control2, from_=0, to=100, textvariable=self.abbau_min_bestand_var, width=6).grid(row=0, column=3, padx=(5, 15))
        
        ttk.Label(control2, text='Min. Lagerdauer (Tage):').grid(row=0, column=4)
        self.abbau_min_lagerdauer_var = tk.StringVar(value='0')
        self.abbau_min_lagerdauer_var.trace_add('write', lambda *args: self._delayed_update_lagerabbau())
        ttk.Spinbox(control2, from_=0, to=3650, increment=30, textvariable=self.abbau_min_lagerdauer_var, width=8).grid(row=0, column=5, padx=(5, 15))
        
        ttk.Label(control2, text='Max. Verkäufe:').grid(row=0, column=6)
        self.abbau_max_verkaeufe_var = tk.StringVar(value='5')
        self.abbau_max_verkaeufe_var.trace_add('write', lambda *args: self._delayed_update_lagerabbau())
        ttk.Spinbox(control2, from_=0, to=100, textvariable=self.abbau_max_verkaeufe_var, width=6).grid(row=0, column=7, padx=(5, 15))
        
        ttk.Label(control2, text='Ziel-Reichweite:').grid(row=0, column=8)
        self.abbau_ziel_reichweite_var = tk.StringVar(value='6 Monate')
        self.abbau_ziel_reichweite_var.trace_add('write', lambda *args: self._delayed_update_lagerabbau())
        ttk.Combobox(control2, textvariable=self.abbau_ziel_reichweite_var, 
                     values=['3 Monate', '6 Monate', '9 Monate', '12 Monate'], 
                     width=10, state='readonly').grid(row=0, column=9, padx=(5, 15))
        
        # Treffer-Info (wird in _update_lagerabbau gefüllt)
        self.abbau_stats_label = ttk.Label(control2, text='', font=('Segoe UI', 9))
        self.abbau_stats_label.grid(row=0, column=10, padx=(20, 0), sticky='w')
        
        # Treeview für Lagerabbau mit Prioritäts-Score und Aktions-Empfehlung
        columns = ('prio', 'teilenummer', 'bezeichnung', 'abc', 'bestand', 'umschlag', 'bestellpunkt', 'verkaeufe', 'stueck_mon', 'reichweite', 'zielbestand', 'abbau', 'lagerwert', 'aktion')
        self.abbau_tree = ttk.Treeview(frame, columns=columns, show='headings')

        headings = {
            'prio': 'Prio',
            'teilenummer': 'Teilenummer',
            'bezeichnung': 'Bezeichnung',
            'abc': 'ABC',
            'bestand': 'Bestand',
            'umschlag': 'Umschl.',
            'bestellpunkt': 'Best.',
            'verkaeufe': 'Verk.',
            'stueck_mon': 'Ø/Mon',
            'reichweite': 'Reichw.',
            'zielbestand': 'Ziel',
            'abbau': 'Abbau',
            'lagerwert': 'Wert €',
            'aktion': 'Aktion',
        }
        widths = {
            'prio': 35, 'teilenummer': 100, 'bezeichnung': 160, 'abc': 35, 'bestand': 50, 'umschlag': 50,
            'bestellpunkt': 35, 'verkaeufe': 40, 'stueck_mon': 45, 'reichweite': 55, 'zielbestand': 40,
            'abbau': 45, 'lagerwert': 60, 'aktion': 185
        }
        aligns = {'prio': tk.CENTER, 'abc': tk.CENTER, 'bestand': tk.E, 'umschlag': tk.E, 'bestellpunkt': tk.CENTER,
                  'verkaeufe': tk.E, 'stueck_mon': tk.E, 'reichweite': tk.E, 'zielbestand': tk.E, 'abbau': tk.E, 'lagerwert': tk.E}
        for col in columns:
            self.abbau_tree.heading(col, text=headings[col], command=lambda c=col: self._sort_treeview(self.abbau_tree, c, False))
            self.abbau_tree.column(col, width=widths[col], anchor=aligns.get(col, tk.W))
        
        self.abbau_tree.grid(row=3, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=self.abbau_tree.yview)
        scrollbar.grid(row=3, column=1, sticky='ns')
        self.abbau_tree.configure(yscrollcommand=scrollbar.set)
        
        # Tooltips für Spaltenüberschriften
        tooltip_texte = {
            'prio': 'Priorität:\n🔴 ABBAU: Sofort (0 Verkäufe)\n🟠 ABBAU: Reduzieren\n🟡 ABBAU: Nicht nachbestellen\n🟢 OK',
            'teilenummer': 'Die Artikelnummer des Teils.',
            'bezeichnung': 'Die Bezeichnung/Beschreibung des Artikels.',
            'abc': 'ABC-Analyse nach Pareto-Prinzip:\n🅰️ A-Teile: Top 20% → 80% Umsatz\n🅱️ B-Teile: 30% → 15% Umsatz\n🅲️ C-Teile: 50% → 5% Umsatz',
            'bestand': 'Aktuelle Lagermenge.',
            'umschlag': 'Umschlagshäufigkeit pro Jahr.\nUmschlag = Jahresverkauf / Bestand\nHöher = bessere Lagerdrehung',
            'bestellpunkt': 'Bestellpunkt-Status:\n🟢 Grün: Bestand OK\n🟡 Gelb: Bald bestellen\n🔴 Rot: Jetzt bestellen\n⚪ Weiß: Keine Daten',
            'verkaeufe': 'Anzahl der Verkäufe im gesamten Zeitraum.\n0 = Nie verkauft (Ladenhüter)',
            'stueck_mon': 'Durchschnittlicher Monatsverbrauch.\nØ/Mon = Verkäufe / Anzahl Monate',
            'reichweite': 'Wie lange reicht der Bestand?\nReichweite = Bestand / Ø Monatsverbrauch\n∞ = Kein Verbrauch',
            'zielbestand': 'Optimaler Bestand basierend auf Ziel-Reichweite.\nZiel = Ø/Mon × Ziel-Monate',
            'abbau': 'Empfohlene Abbau-Menge.\nAbbau = MAX(0, Bestand - Zielbestand)',
            'lagerwert': 'Gebundenes Kapital: Bestand × UPE',
            'aktion': 'Empfohlene Aktion:\n🔴 ABBAU: Sofort - Nie verkauft\n🟠 ABBAU: X Stk reduzieren\n🟡 Nicht nachbestellen (<3 Verk.)\n🟢 OK: Bestand passt',
        }
        TreeviewHeaderTooltip(self.abbau_tree, tooltip_texte, delay_ms=3000)
        
        # Farbliche Hervorhebung nach Aktion
        self.abbau_tree.tag_configure('abbau_sofort', background='#f8d7da')  # Rot - sofort abbauen
        self.abbau_tree.tag_configure('abbau_reduzieren', background='#ffeeba')  # Orange - reduzieren
        self.abbau_tree.tag_configure('nicht_nachbestellen', background='#fff3cd')  # Gelb - nicht nachbestellen
        self.abbau_tree.tag_configure('ok', background='#d4edda')  # Grün - OK

        frame.rowconfigure(3, weight=1)
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
        
        # Filter für spezifische Teilenummern
        ttk.Label(control, text='Filter Teilenummern:').grid(row=0, column=4, padx=(20, 5))
        self.chart_filter_var = tk.StringVar()
        ttk.Entry(control, textvariable=self.chart_filter_var, width=30).grid(row=0, column=5, padx=(0, 5))
        ttk.Label(control, text='(Komma-getrennt, leer=alle)', font=('', 8)).grid(row=0, column=6, sticky='w')

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
        file_menu.add_command(label=f'Verkaufsdaten öffnen... ({shortcut})', command=self._open_file)
        file_menu.add_separator()
        file_menu.add_command(label='Ergebnisse exportieren...', command=self._export_results)
        file_menu.add_command(label='Diagramm speichern...', command=self._save_chart)
        file_menu.add_separator()
        file_menu.add_command(label='Beenden', command=self.destroy)
        self.bind('<Command-o>' if os.name == 'darwin' else '<Control-o>', lambda e: self._open_file())

    # --- Datei laden ------------------------------------------------------
    def _open_file(self):
        filetypes = [
            ('Textdateien', '*.txt'),
            ('Alle Dateien', '*.*')
        ]
        filepath = filedialog.askopenfilename(title='Verkaufsdaten öffnen', filetypes=filetypes)
        if not filepath:
            return

        self.status_label.config(text='Datei wird geladen...')
        self.update()
        
        # Prüfe Dateigröße - nur bei großen Dateien (>1MB) Fortschrittsanzeige
        file_size = os.path.getsize(filepath)
        use_progress = file_size > 1_000_000  # 1 MB
        
        progress_ui = self._show_progress('Importiere Datei...') if use_progress else None

        try:
            # Immer SQLite verwenden
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

            self.metadata = metadata
            self.current_file = filepath
            self._update_file_label()
            self._update_meta_label()
            self._refresh_after_load()
            self.status_label.config(text=f"Geladen: {self.statistik.get_record_count()} Datensätze")
        except Exception as exc:
            messagebox.showerror('Fehler', f'Die Datei konnte nicht geladen werden:\n{exc}')
            self.status_label.config(text='Fehler beim Laden')
        finally:
            if progress_ui:
                self._close_progress(progress_ui)

    def _open_lagerbestand_file(self):
        """Lädt eine Lagerbestand-Datei (ausgabe.txt aus Loco-Soft)."""
        filetypes = [
            ('Textdateien', '*.txt'),
            ('Alle Dateien', '*.*')
        ]
        filepath = filedialog.askopenfilename(
            title='Lagerbestand öffnen (ausgabe.txt)',
            filetypes=filetypes
        )
        if not filepath:
            return
        
        self.status_label.config(text='Lagerbestand wird geladen...')
        self.update()
        
        progress_ui = None
        try:
            # Zähle Zeilen für Fortschrittsbalken
            total_lines = 0
            encoding_found = None
            for encoding in ['cp1252', 'utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        total_lines = sum(1 for _ in f) - 1
                    encoding_found = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if not encoding_found:
                raise ValueError("Datei-Encoding konnte nicht erkannt werden")
            
            progress_ui = self._show_progress('Lagerbestand wird geladen...')
            self.lagerbestand_data = {}
            count = 0
            
            with open(filepath, 'r', encoding=encoding_found) as f:
                # Header lesen
                header = f.readline().strip().split('\t')
                
                # Spaltenindizes finden
                idx = {}
                for i, h in enumerate(header):
                    h_clean = h.strip().lower()
                    if 'et-nr' in h_clean:
                        idx['teilenummer'] = i
                    elif h_clean == 'et-bezeichnung':
                        idx['bezeichnung'] = i
                    elif h_clean == 'bestand':
                        idx['bestand'] = i
                    elif 'verfügbare menge' in h_clean or 'verfuegbare menge' in h_clean:
                        idx['verfuegbar'] = i
                    elif h_clean == 'upe':
                        idx['upe'] = i
                    elif 'verk. st. jahr' in h_clean:
                        idx['verk_jahr'] = i
                    elif 'verk. st. vorjahr' in h_clean:
                        idx['verk_vorjahr'] = i
                    elif h_clean == 'lager':
                        idx['lager'] = i
                    elif 'lagerdauer' in h_clean:
                        idx['lagerdauer'] = i
                
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split('\t')
                    if len(parts) > 5:  # Mindestens 6 Spalten
                        def parse_float(val):
                            try:
                                return float(val.replace(',', '.').strip())
                            except:
                                return 0.0
                        
                        def clean_teilenummer(tn):
                            """Bereinigt Teilenummer von ='...' Format"""
                            tn = tn.strip()
                            if tn.startswith('="') and tn.endswith('"'):
                                tn = tn[2:-1]
                            return tn
                        
                        # Hole BEIDE Teilenummern (Spalte 0 = ET-Nr., Spalte 1 = Lieferant-ET-Nr.)
                        et_nr = clean_teilenummer(parts[0]) if len(parts) > 0 else ''
                        lieferant_et_nr = clean_teilenummer(parts[1]) if len(parts) > 1 else ''
                        
                        # Bestand aus Spalte 5 lesen
                        bestand_idx = idx.get('bestand', 5)
                        bestand_val = parts[bestand_idx] if bestand_idx < len(parts) else '0'
                        
                        lager_eintrag = {
                            'bezeichnung': parts[idx.get('bezeichnung', 2)].strip() if idx.get('bezeichnung', 2) < len(parts) else '',
                            'bestand': parse_float(bestand_val),
                            'verfuegbar': parse_float(parts[idx.get('verfuegbar', 7)]) if idx.get('verfuegbar', 7) < len(parts) else 0,
                            'upe': parse_float(parts[idx.get('upe', 8)]) if idx.get('upe', 8) < len(parts) else 0,
                            'verk_jahr': parse_float(parts[idx.get('verk_jahr', 14)]) if idx.get('verk_jahr', 14) < len(parts) else 0,
                            'verk_vorjahr': parse_float(parts[idx.get('verk_vorjahr', 15)]) if idx.get('verk_vorjahr', 15) < len(parts) else 0,
                            'lager': parts[idx.get('lager', 4)].strip() if idx.get('lager', 4) < len(parts) else '',
                            'lagerdauer': parse_float(parts[idx.get('lagerdauer', 21)]) if idx.get('lagerdauer', 21) < len(parts) else 0,
                            'et_nr': et_nr,
                            'lieferant_et_nr': lieferant_et_nr,
                        }
                        
                        # Speichere unter ALLEN möglichen Schlüsseln für maximales Matching
                        keys_to_store = set()
                        
                        # ET-Nr. (Spalte 0) - EAN/Barcode
                        if et_nr:
                            keys_to_store.add(et_nr)
                            keys_to_store.add(et_nr.lstrip('0') or et_nr)
                        
                        # Lieferant-ET-Nr. (Spalte 1) - Hersteller-Teilenummer
                        if lieferant_et_nr:
                            keys_to_store.add(lieferant_et_nr)
                            keys_to_store.add(lieferant_et_nr.lstrip('0') or lieferant_et_nr)
                            # Auch mit führenden Nullen auf verschiedene Längen
                            keys_to_store.add(lieferant_et_nr.zfill(10))
                            keys_to_store.add(lieferant_et_nr.zfill(13))
                        
                        for key in keys_to_store:
                            if key:
                                self.lagerbestand_data[key] = lager_eintrag
                        
                        count += 1
                    
                    if line_num % 5000 == 0:
                        self._update_progress(progress_ui, line_num, total_lines)
            
            self._close_progress(progress_ui)
            progress_ui = None
            
            self._update_file_label()
            self.status_label.config(text=f"Lagerbestand: {count:,} Teile geladen".replace(',', '.'))
            
            # Initialisiere Bezeichnungsliste für Lagerabbau-Autocomplete
            self._init_abbau_bezeichnung_liste()
            
            # Aktualisiere Lagerhaltung-Tab wenn Verkaufsdaten vorhanden
            if self.statistik:
                self._update_lagerhaltung()
            
            # Aktualisiere Lagerabbau-Tab
            self._update_lagerabbau()
            
            self._show_auto_close_info('Lagerbestand geladen',
                f'{count:,} Teile aus Lagerbestand geladen.\n\n'.replace(',', '.') +
                f'Die Lagerhaltung-Tabelle zeigt nun Bestand, Reichweite und Prognose.')
                
        except Exception as exc:
            messagebox.showerror('Fehler', f'Lagerbestand konnte nicht geladen werden:\n{exc}')
            self.status_label.config(text='Fehler beim Laden des Lagerbestands')
        finally:
            if progress_ui:
                self._close_progress(progress_ui)

    def _update_file_label(self):
        """Aktualisiert das Label mit den geladenen Dateien."""
        parts = []
        if self.current_file:
            parts.append(f"Verkauf: {Path(self.current_file).name}")
        if self.lagerbestand_data:
            parts.append(f"Lager: {len(self.lagerbestand_data):,} Teile".replace(',', '.'))
        self.file_label.config(text=' | '.join(parts) if parts else 'Keine Dateien geladen')

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
        
        # Datumsbereich hinzufügen
        min_date, max_date = self.statistik.get_date_range()
        if min_date and max_date:
            # Konvertiere ISO-Format zu deutschem Format
            try:
                min_dt = datetime.strptime(min_date, '%Y-%m-%d').strftime('%d.%m.%Y')
                max_dt = datetime.strptime(max_date, '%Y-%m-%d').strftime('%d.%m.%Y')
                meta_text += f"  |  Zeitraum: {min_dt} - {max_dt}"
            except:
                meta_text += f"  |  Zeitraum: {min_date} - {max_date}"
        
        for key, value in self.metadata.items():
            meta_text += f"  |  {key}: {value}"
        self.meta_label.config(text=meta_text)

    def _refresh_after_load(self):
        try:
            self._update_time_filter_options()
        except Exception as e:
            print(f"Fehler in _update_time_filter_options: {e}")
        
        try:
            self._apply_filter()
        except Exception as e:
            print(f"Fehler in _apply_filter: {e}")
        
        try:
            self._update_top_list()
        except Exception as e:
            print(f"Fehler in _update_top_list: {e}")
        
        try:
            self._update_time_analysis()
        except Exception as e:
            print(f"Fehler in _update_time_analysis: {e}")
        
        try:
            self._update_summary()
        except Exception as e:
            print(f"Fehler in _update_summary: {e}")
        
        try:
            self._init_produkt_liste()
        except Exception as e:
            print(f"Fehler in _init_produkt_liste: {e}")
        
        try:
            self._update_lagerhaltung()
        except Exception as e:
            print(f"Fehler in _update_lagerhaltung: {e}")

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
        
        # Initialisiere Produktliste falls noch nicht vorhanden
        if not self.alle_produkte_liste:
            self._init_produkt_liste()
        
        # Ermittle Zeitraum
        monate_str = self.lager_monate_var.get()
        if monate_str == 'Alle Daten':
            monate = None
        else:
            monate = int(monate_str.split()[0])
        
        # Finde das letzte Datum in den Daten für korrekte Zeitraum-Anzeige
        alle_daten_iso = []
        if self.sqlite_store:
            # Hole Datumsbereich aus SQLite
            min_max = self.statistik.get_date_range()
            if min_max[0] and min_max[1]:
                erstes_datum_str = min_max[0]
                letztes_datum_str = min_max[1]
                alle_daten_iso = [erstes_datum_str, letztes_datum_str]
        else:
            alle_daten_iso = [r.get('abgabe_iso', '') for r in self.data if r.get('abgabe_iso')]

        if alle_daten_iso:
            letztes_datum_str = max(alle_daten_iso)
            erstes_datum_str = min(alle_daten_iso)
            letztes_datum = datetime.strptime(letztes_datum_str, '%Y-%m-%d')
            erstes_datum = datetime.strptime(erstes_datum_str, '%Y-%m-%d')
        else:
            letztes_datum = datetime.now()
            erstes_datum = datetime.now()
        
        # Berechne und zeige Zeitraum inkl. Anzahl Monate
        gesamt_monate = max(1, ((letztes_datum.year - erstes_datum.year) * 12 + 
                                letztes_datum.month - erstes_datum.month) + 1)
        if monate is None:
            self.lager_zeitraum_label.config(
                text=f"Daten: {erstes_datum.strftime('%d.%m.%Y')} - {letztes_datum.strftime('%d.%m.%Y')} ({gesamt_monate} Monate)"
            )
        else:
            von_datum = letztes_datum - timedelta(days=monate * 30.44)
            self.lager_zeitraum_label.config(
                text=f"({von_datum.strftime('%d.%m.%Y')} - {letztes_datum.strftime('%d.%m.%Y')})"
            )
        
        # Nur einmal berechnen!
        alle_analyse = self.statistik.get_lagerhaltung_analyse(monate=monate)
        
        # Zusätzliche Filter anwenden
        try:
            min_verkaeufe = int(self.lager_min_verkaeufe_var.get() or 0)
        except ValueError:
            min_verkaeufe = 0
        try:
            min_umsatz = float(self.lager_min_umsatz_var.get() or 0)
        except ValueError:
            min_umsatz = 0
        try:
            min_kunden = int(self.lager_min_kunden_var.get() or 0)
        except ValueError:
            min_kunden = 0
        try:
            min_monatsprozent = float(self.lager_min_monatsprozent_var.get() or 0)
        except ValueError:
            min_monatsprozent = 0
        
        # Produkt-Filter (kann mehrere Begriffe kommagetrennt enthalten)
        produkt_filter_text = self.lager_produkt_var.get().strip()
        produkt_filter_liste = []
        if produkt_filter_text:
            # Teile nach Komma und bereinige
            produkt_filter_liste = [p.strip().upper() for p in produkt_filter_text.split(',') if p.strip()]
        
        # Wende zusätzliche Filter an
        gefiltert = []
        for a in alle_analyse:
            if a['anzahl_verkäufe'] < min_verkaeufe:
                continue
            if a.get('monatsdurchschnitt_umsatz', 0) < min_umsatz:
                continue
            if a.get('anzahl_kunden', 0) < min_kunden:
                continue
            if a.get('lagerfaehigkeit_prozent', 0) < min_monatsprozent:
                continue
            # Produkt-Filter anwenden (Textsuche in Bezeichnung)
            if produkt_filter_liste:
                bez = a.get('bezeichnung', '').upper()
                # Prüfe ob einer der Suchbegriffe in der Bezeichnung vorkommt
                gefunden = False
                for suchbegriff in produkt_filter_liste:
                    if suchbegriff in bez:
                        gefunden = True
                        break
                if not gefunden:
                    continue
            gefiltert.append(a)
        
        alle_analyse = gefiltert
        
        # Statistiken berechnen
        lohnend = sum(1 for a in alle_analyse if 'Lohnend' in a['kategorie'])
        grenzwertig = sum(1 for a in alle_analyse if 'Grenzwertig' in a['kategorie'])
        nicht_lohnend = sum(1 for a in alle_analyse if 'Nicht lohnend' in a['kategorie'])
        lagerfaehig_count = sum(1 for a in alle_analyse if a.get('lagerfaehig', '').startswith('✅'))
        
        self.lager_stats_label.config(
            text=f"Gesamt: {len(alle_analyse)}  |  ✅ Lohnend: {lohnend}  |  ⚠️ Grenzwertig: {grenzwertig}  |  ❌ Nicht lohnend: {nicht_lohnend}  |  📦 Lagerfähig: {lagerfaehig_count}"
        )
        
        # Kategorie-Filter anwenden
        filter_val = self.lager_filter_var.get()
        if filter_val == 'nur lohnend':
            analyse = [a for a in alle_analyse if 'Lohnend' in a['kategorie']]
        elif filter_val == 'nur grenzwertig':
            analyse = [a for a in alle_analyse if 'Grenzwertig' in a['kategorie']]
        elif filter_val == 'nur nicht lohnend':
            analyse = [a for a in alle_analyse if 'Nicht lohnend' in a['kategorie']]
        elif filter_val == 'nur lagerfähig':
            analyse = [a for a in alle_analyse if a.get('lagerfaehig', '').startswith('✅')]
        else:
            analyse = alle_analyse
        
        # Tabelle füllen
        for item in analyse:
            tage_str = f"{item['durchschnitt_tage']:.0f}" if item['durchschnitt_tage'] else "-"
            
            # 9. Farbliches Tag bestimmen
            if 'Lohnend' in item['kategorie']:
                tag = 'lohnend'
            elif 'Grenzwertig' in item['kategorie']:
                tag = 'grenzwertig'
            else:
                tag = 'nicht_lohnend'
            
            # Stk./Mon.: Stückzahl (Menge) pro Monat (über alle Monate)
            zeitraum_monate = item.get('zeitraum_monate', 1)
            if zeitraum_monate and zeitraum_monate > 0:
                stueck_pro_mon = item['gesamtmenge'] / zeitraum_monate
            else:
                stueck_pro_mon = item['gesamtmenge']
            
            # Lagerbestand-Daten holen (versuche verschiedene Formate)
            teilenummer = item['teilenummer']
            lager_info = self.lagerbestand_data.get(teilenummer, {})
            
            # Falls nicht gefunden, versuche ohne führende Nullen
            if not lager_info:
                tn_stripped = teilenummer.lstrip('0') or teilenummer
                lager_info = self.lagerbestand_data.get(tn_stripped, {})
            
            # Falls immer noch nicht gefunden, versuche mit führenden Nullen (auf 13 Stellen)
            if not lager_info:
                tn_padded = teilenummer.zfill(13)
                lager_info = self.lagerbestand_data.get(tn_padded, {})
            
            bestand = lager_info.get('bestand', 0) if lager_info else 0
            
            # Bestand und Reichweite berechnen
            if self.lagerbestand_data:
                bestand_str = f"{bestand:.0f}" if bestand > 0 else "0"
                if stueck_pro_mon > 0 and bestand > 0:
                    reichweite_monate = bestand / stueck_pro_mon
                    reichweite_str = f"{reichweite_monate:.1f} Mon."
                    # Prognose basierend auf Reichweite
                    if reichweite_monate < 2:
                        prognose = "🔴 Zu wenig"
                    elif reichweite_monate <= 12:
                        prognose = "✅ OK"
                    else:
                        prognose = "⚠️ Zu viel"
                elif stueck_pro_mon == 0 and bestand > 0:
                    reichweite_str = "∞"
                    prognose = "❌ Kein Abverk."
                else:
                    reichweite_str = "-"
                    prognose = "-" if bestand == 0 else "❌ Kein Abverk."
            else:
                bestand_str = "-"
                reichweite_str = "-"
                prognose = "-"

            # ═══════════════════════════════════════════════════════════════
            # UMSCHLAGSHÄUFIGKEIT (Turnover Rate)
            # ═══════════════════════════════════════════════════════════════
            # Berechnung: Jahresverkäufe / Durchschnittsbestand
            if bestand > 0 and zeitraum_monate and zeitraum_monate > 0:
                jahresverkauf = (item['gesamtmenge'] / zeitraum_monate) * 12
                umschlagshaeufigkeit = jahresverkauf / bestand
                umschlag_str = f"{umschlagshaeufigkeit:.1f}x"
            else:
                umschlag_str = "-"
                umschlagshaeufigkeit = 0

            # ═══════════════════════════════════════════════════════════════
            # BESTELLPUNKT-EMPFEHLUNG
            # ═══════════════════════════════════════════════════════════════
            # Bestellpunkt = (Ø Tagesverbrauch × Lieferzeit) + Sicherheitsbestand
            # Vereinfacht: Bestellpunkt = 2 Monate Verbrauch (annahme 1 Monat Lieferzeit + 1 Monat Sicherheit)
            if stueck_pro_mon > 0 and bestand > 0:
                bestellpunkt = stueck_pro_mon * 2  # 2 Monate als Bestellpunkt
                if bestand < bestellpunkt:
                    bestellpunkt_status = "🔴"  # Unter Bestellpunkt - jetzt bestellen!
                elif bestand < bestellpunkt * 1.2:
                    bestellpunkt_status = "🟡"  # Nahe Bestellpunkt - bald bestellen
                else:
                    bestellpunkt_status = "🟢"  # Bestand OK
            else:
                bestellpunkt_status = "⚪"  # Kein Bestand/Bestellpunkt

            self.lager_tree.insert('', 'end', values=(
                item['teilenummer'],
                item['bezeichnung'],
                item.get('abc', '-'),
                bestand_str,
                umschlag_str,
                bestellpunkt_status,
                reichweite_str,
                prognose,
                item['anzahl_verkäufe'],
                f"{stueck_pro_mon:.1f}",
                item.get('anzahl_kunden', '-'),
                item.get('lagerfaehig', '-'),
                f"{item['monatsdurchschnitt_umsatz']:.0f}",
                item.get('trend', '-'),
                item['kategorie'],
                item['empfehlung'],
            ), tags=(tag,))

    def _sort_treeview(self, tree, col, reverse):
        """Sortiert eine Treeview-Tabelle nach der angeklickten Spalte."""
        # Hole alle Zeilen mit ihren Werten
        data_list = [(tree.set(child, col), child) for child in tree.get_children('')]
        
        # Erkenne ob numerische Sortierung nötig ist
        try:
            # Versuche als Zahl zu sortieren (entferne € und Tausender-Trennzeichen)
            data_list.sort(key=lambda t: float(t[0].replace(',', '.').replace('€', '').strip()), reverse=reverse)
        except (ValueError, AttributeError):
            # Sonst alphabetisch sortieren
            data_list.sort(key=lambda t: t[0].lower(), reverse=reverse)
        
        # Neuordnen der Zeilen
        for index, (val, child) in enumerate(data_list):
            tree.move(child, '', index)
        
        # Spalten-Überschrift aktualisieren um Sortierrichtung anzuzeigen
        for column in tree['columns']:
            current_heading = tree.heading(column)['text']
            # Entferne alte Pfeile
            if current_heading.endswith(' ▲') or current_heading.endswith(' ▼'):
                current_heading = current_heading[:-2]
            
            if column == col:
                # Füge Pfeil zur sortierten Spalte hinzu
                arrow = ' ▼' if reverse else ' ▲'
                tree.heading(column, text=current_heading + arrow, 
                           command=lambda c=col: self._sort_treeview(tree, c, not reverse))
            else:
                # Andere Spalten ohne Pfeil, aber mit Sortier-Command
                tree.heading(column, text=current_heading,
                           command=lambda c=column: self._sort_treeview(tree, c, False))

    def _search_top_list(self):
        """Filtert die Top-Liste nach Suchbegriff."""
        if not self.statistik:
            return
        
        search_term = self.top_search_var.get().lower().strip()
        
        # Lösche alle Einträge
        for item in self.top_tree.get_children():
            self.top_tree.delete(item)
        
        try:
            n = int(self.top_n_var.get())
        except ValueError:
            n = 20
        
        data = self.filtered_data if not self.sqlite_store else None
        top_items = self.statistik.get_top_n(n=n, by=self.sort_var.get(), data=data, filters=self.filter_params)
        
        # Filtere nach Suchbegriff
        for item in top_items:
            if not search_term or \
               search_term in item['teilenummer'].lower() or \
               search_term in item.get('bezeichnung', '').lower():
                self.top_tree.insert('', 'end', values=(
                    item['teilenummer'],
                    item.get('bezeichnung', ''),
                    item.get('anzahl_vorgaenge', 0),
                    f"{item.get('gesamtmenge', 0.0):.2f}",
                    f"{item.get('gesamtumsatz', 0.0):.2f}",
                    item.get('anzahl_kunden', 0),
                ))

    def _autocomplete_produkt(self, event=None):
        """Zeigt passende Vorschläge beim Tippen im Produkt-Feld."""
        if not hasattr(self, 'alle_produkte_liste') or not self.alle_produkte_liste:
            return
        
        # Hole aktuellen Text
        current_text = self.lager_produkt_var.get().strip().upper()
        
        # Wenn leer, zeige alle
        if not current_text:
            self.lager_produkt_entry['values'] = self.alle_produkte_liste
            return
        
        # Finde den letzten Begriff (nach Komma)
        if ',' in current_text:
            last_term = current_text.split(',')[-1].strip()
        else:
            last_term = current_text
        
        if not last_term:
            self.lager_produkt_entry['values'] = self.alle_produkte_liste
            return
        
        # Filtere passende Produkte - erst nach Anfang, dann überall
        startswith = [p for p in self.alle_produkte_liste if p.upper().startswith(last_term)]
        contains = [p for p in self.alle_produkte_liste if p not in startswith and last_term in p.upper()]
        gefiltert = startswith + contains

        if gefiltert:
            self.lager_produkt_entry['values'] = gefiltert
            # Öffne Dropdown automatisch bei wenigen Ergebnissen
            try:
                if event and event.keysym not in ('Return', 'Tab', 'Escape', 'Down', 'Up', 'Left', 'Right'):
                    self.lager_produkt_entry.event_generate('<Down>')
            except:
                pass

    def _on_produkt_keyrelease(self, event=None):
        """Wird bei Tastendruck im Produkt-Feld aufgerufen - Autocomplete + verzögerte Aktualisierung."""
        self._autocomplete_produkt(event)
        # Bei Enter sofort aktualisieren, sonst verzögert
        if event and event.keysym == 'Return':
            self._update_lagerhaltung()
        else:
            self._delayed_update_lagerhaltung()
    
    def _on_produkt_selected(self, event=None):
        """Wird aufgerufen wenn ein Produkt aus dem Dropdown ausgewählt wird."""
        selected = self.lager_produkt_var.get()
        if selected and '(' in selected:
            # Extrahiere nur den Produktnamen (ohne Anzahl)
            produkt_name = selected.split(' (')[0].strip()
            self.lager_produkt_var.set(produkt_name)
    
    def _on_produkt_selected_and_update(self, event=None):
        """Wird aufgerufen wenn ein Produkt aus dem Dropdown ausgewählt wird - mit Aktualisierung."""
        self._on_produkt_selected(event)
        self._update_lagerhaltung()

    def _clear_produkt_filter(self):
        """Löscht den Produkt-Filter."""
        self.lager_produkt_var.set('')
        if self.alle_produkte_liste:
            self.lager_produkt_entry['values'] = self.alle_produkte_liste

    def _clear_produkt_filter_and_update(self):
        """Löscht den Produkt-Filter und aktualisiert die Liste."""
        self._clear_produkt_filter()
        self._update_lagerhaltung()

    def _delayed_update_lagerhaltung(self):
        """Verzögerte Aktualisierung um nicht bei jedem Tastendruck zu aktualisieren."""
        # Abbrechen falls bereits ein Timer läuft
        if hasattr(self, '_update_timer') and self._update_timer:
            self.after_cancel(self._update_timer)
        # Neuen Timer starten (300ms Verzögerung)
        self._update_timer = self.after(300, self._update_lagerhaltung)

    def _delayed_update_lagerabbau(self):
        """Verzögerte Aktualisierung für Lagerabbau-Tab."""
        if hasattr(self, '_abbau_timer') and self._abbau_timer:
            self.after_cancel(self._abbau_timer)
        self._abbau_timer = self.after(300, self._update_lagerabbau)

    def _init_abbau_bezeichnung_liste(self):
        """Initialisiert die Bezeichnungsliste für Lagerabbau-Autocomplete."""
        if not self.lagerbestand_data:
            return

        # Sammle alle vollständigen Bezeichnungen mit Gesamtbestand
        bezeichnung_bestand = {}
        for eintrag in self.lagerbestand_data.values():
            bez = eintrag.get('bezeichnung', '').strip()
            bestand = eintrag.get('bestand', 0)
            if bez and bestand > 0:
                # Verwende vollständige Bezeichnung als Key
                bezeichnung_bestand[bez] = bezeichnung_bestand.get(bez, 0) + bestand

        # Sortiere nach Bestand und erstelle Liste
        sortiert = sorted(bezeichnung_bestand.items(), key=lambda x: -x[1])
        self.abbau_bezeichnung_liste = [name for name, count in sortiert[:200]]

        # Setze initiale Werte in Combobox
        self.abbau_bezeichnung_entry['values'] = self.abbau_bezeichnung_liste

    def _on_abbau_bezeichnung_keyrelease(self, event):
        """Autocomplete für Bezeichnung im Lagerabbau-Tab."""
        if not self.lagerbestand_data:
            return

        # Ignoriere spezielle Tasten
        if event.keysym in ('Return', 'Tab', 'Escape', 'Up', 'Down', 'Left', 'Right'):
            return

        typed = self.abbau_bezeichnung_var.get().strip().upper()
        if len(typed) < 1:
            # Zeige vollständige Liste
            if hasattr(self, 'abbau_bezeichnung_liste'):
                self.abbau_bezeichnung_entry['values'] = self.abbau_bezeichnung_liste[:50]
            return

        # Filtere Liste basierend auf Eingabe
        if hasattr(self, 'abbau_bezeichnung_liste'):
            # Priorität 1: Beginnt mit Eingabe
            startswith = [b for b in self.abbau_bezeichnung_liste if b.upper().startswith(typed)]

            # Priorität 2: Ein Wort in der Bezeichnung beginnt mit Eingabe
            word_startswith = [b for b in self.abbau_bezeichnung_liste
                              if b not in startswith and
                              any(word.upper().startswith(typed) for word in b.split())]

            # Priorität 3: Enthält die Eingabe irgendwo
            contains = [b for b in self.abbau_bezeichnung_liste
                       if b not in startswith and b not in word_startswith and typed in b.upper()]

            # Kombiniere die Ergebnisse in Prioritätsreihenfolge
            gefiltert = startswith + word_startswith + contains
            self.abbau_bezeichnung_entry['values'] = gefiltert[:30]

    def _clear_abbau_bezeichnung_filter(self):
        """Löscht den Bezeichnungs-Filter im Lagerabbau-Tab."""
        self.abbau_bezeichnung_var.set('')
        self._update_lagerabbau()

    def _quick_filter_kritisch(self):
        """Quick-Filter: Nur kritische Teile (>500€ UND >1 Jahr)."""
        self.abbau_min_wert_var.set('500')
        self.abbau_min_lagerdauer_var.set('365')
        self.abbau_min_bestand_var.set('1')
        self.abbau_max_verkaeufe_var.set('5')
        self.abbau_bezeichnung_var.set('')
        self.abbau_search_var.set('')
        self._update_lagerabbau()

    def _quick_filter_nie_verkauft(self):
        """Quick-Filter: Nur Teile die nie verkauft wurden (0 Verkäufe)."""
        self.abbau_min_wert_var.set('0')
        self.abbau_min_lagerdauer_var.set('0')
        self.abbau_min_bestand_var.set('1')
        self.abbau_max_verkaeufe_var.set('0')  # Nur 0 Verkäufe
        self.abbau_bezeichnung_var.set('')
        self.abbau_search_var.set('')
        self._update_lagerabbau()

    def _quick_filter_top_wert(self):
        """Quick-Filter: Top 20 nach Lagerwert."""
        self.abbau_min_wert_var.set('0')
        self.abbau_min_lagerdauer_var.set('0')
        self.abbau_min_bestand_var.set('1')
        self.abbau_max_verkaeufe_var.set('5')
        self.abbau_bezeichnung_var.set('')
        self.abbau_search_var.set('')
        self._abbau_sort_mode = 'lagerwert'
        self._update_lagerabbau()

    def _quick_filter_top_dauer(self):
        """Quick-Filter: Top 20 nach Lagerdauer."""
        self.abbau_min_wert_var.set('0')
        self.abbau_min_lagerdauer_var.set('0')
        self.abbau_min_bestand_var.set('1')
        self.abbau_max_verkaeufe_var.set('5')
        self.abbau_bezeichnung_var.set('')
        self.abbau_search_var.set('')
        self._abbau_sort_mode = 'lagerdauer'
        self._update_lagerabbau()

    def _quick_filter_alte_teile(self):
        """Quick-Filter: Teile älter als 2 Jahre."""
        self.abbau_min_wert_var.set('0')
        self.abbau_min_lagerdauer_var.set('730')  # 2 Jahre
        self.abbau_min_bestand_var.set('1')
        self.abbau_max_verkaeufe_var.set('5')
        self.abbau_bezeichnung_var.set('')
        self.abbau_search_var.set('')
        self._update_lagerabbau()

    def _quick_filter_reset(self):
        """Setzt alle Filter zurück."""
        self.abbau_min_wert_var.set('0')
        self.abbau_min_lagerdauer_var.set('0')
        self.abbau_min_bestand_var.set('1')
        self.abbau_max_verkaeufe_var.set('5')
        self.abbau_bezeichnung_var.set('')
        self.abbau_search_var.set('')
        if hasattr(self, '_abbau_sort_mode'):
            del self._abbau_sort_mode
        self._update_lagerabbau()

    def _export_lagerabbau(self):
        """Exportiert die Lagerabbau-Tabelle als CSV."""
        if not self.abbau_tree.get_children():
            messagebox.showwarning('Export', 'Keine Daten zum Exportieren vorhanden.')
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV Dateien', '*.csv'), ('Alle Dateien', '*.*')],
            title='Lagerabbau exportieren'
        )
        if not filepath:
            return
        
        try:
            with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
                # Header
                columns = ('Priorität', 'Teilenummer', 'Bezeichnung', 'ABC', 'Bestand', 'Umschlag/Jahr', 'Bestellpunkt Status', 'Verkäufe', 'Ø/Mon', 'Reichweite', 'Zielbestand', 'Abbau', 'Lagerwert €', 'Aktion')
                f.write(';'.join(columns) + '\n')

                # Daten
                for item in self.abbau_tree.get_children():
                    values = self.abbau_tree.item(item, 'values')
                    # Konvertiere Emojis zu Text für bessere Excel-Kompatibilität
                    values_converted = list(values)
                    # ABC-Spalte (Index 3)
                    if len(values_converted) > 3:
                        values_converted[3] = str(values_converted[3]).replace('🅰️', 'A').replace('🅱️', 'B').replace('🅲️', 'C')
                    # Bestellpunkt-Spalte (Index 6)
                    if len(values_converted) > 6:
                        bp = str(values_converted[6])
                        if '🔴' in bp:
                            values_converted[6] = 'Rot - Jetzt bestellen'
                        elif '🟡' in bp:
                            values_converted[6] = 'Gelb - Bald bestellen'
                        elif '🟢' in bp:
                            values_converted[6] = 'Grün - Bestand OK'
                        elif '⚪' in bp:
                            values_converted[6] = 'Keine Daten'
                    f.write(';'.join(str(v) for v in values_converted) + '\n')
            
            self._show_auto_close_info('Export erfolgreich', f'Daten wurden nach {filepath} exportiert.')
        except Exception as e:
            messagebox.showerror('Export-Fehler', f'Fehler beim Export: {e}')

    def _update_lagerabbau(self):
        """Aktualisiert die Lagerabbau-Tabelle mit nie verkauften Teilen."""
        if not self.lagerbestand_data:
            self.abbau_stats_label.config(text='⚠️ Bitte zuerst Lagerbestand laden!')
            return
        
        # Treeview leeren
        for item in self.abbau_tree.get_children():
            self.abbau_tree.delete(item)
        
        # Sammle alle verkauften Teilenummern MIT Anzahl der Verkäufe
        verkaufs_zaehler = {}  # Zählt Anzahl der Verkäufe pro Teilenummer

        # Hole Daten aus SQLite oder In-Memory
        if self.sqlite_store:
            # Daten aus SQLite-Datenbank holen
            cursor = self.sqlite_store.conn.execute("SELECT teilenummer FROM records WHERE teilenummer IS NOT NULL AND teilenummer != ''")
            for row in cursor:
                tn = row[0]
                if tn:
                    # Zähle unter VIELEN verschiedenen Formaten für besseres Matching
                    tn_upper = tn.upper().strip()
                    tn_stripped = tn.lstrip('0') or tn
                    varianten = [
                        tn,
                        tn_upper,
                        tn_stripped,
                        tn_stripped.upper(),
                        tn.zfill(10),
                        tn.zfill(13),
                        tn.zfill(15),
                    ]
                    for key in set(varianten):  # set() vermeidet Duplikate
                        verkaufs_zaehler[key] = verkaufs_zaehler.get(key, 0) + 1
        elif self.data:
            for record in self.data:
                tn = record.get('teilenummer', '')
                if tn:
                    # Zähle unter VIELEN verschiedenen Formaten für besseres Matching
                    tn_upper = tn.upper().strip()
                    tn_stripped = tn.lstrip('0') or tn
                    varianten = [
                        tn,
                        tn_upper,
                        tn_stripped,
                        tn_stripped.upper(),
                        tn.zfill(10),
                        tn.zfill(13),
                        tn.zfill(15),
                    ]
                    for key in set(varianten):  # set() vermeidet Duplikate
                        verkaufs_zaehler[key] = verkaufs_zaehler.get(key, 0) + 1
        
        # Filter-Werte holen
        try:
            min_wert = float(self.abbau_min_wert_var.get() or 0)
        except ValueError:
            min_wert = 0
        try:
            min_bestand = float(self.abbau_min_bestand_var.get() or 1)
        except ValueError:
            min_bestand = 1
        try:
            min_lagerdauer = float(self.abbau_min_lagerdauer_var.get() or 0)
        except ValueError:
            min_lagerdauer = 0
        try:
            max_verkaeufe = int(self.abbau_max_verkaeufe_var.get() or 5)
        except ValueError:
            max_verkaeufe = 5
        
        # Bezeichnung-Filter (kann mehrere Begriffe kommagetrennt enthalten)
        bezeichnung_filter_text = self.abbau_bezeichnung_var.get().strip()
        bezeichnung_filter_liste = []
        if bezeichnung_filter_text:
            # Entferne eventuelle Klammern mit Zahlen am Ende (z.B. "ÖLFILTER (123 Stk.)" -> "ÖLFILTER")
            for p in bezeichnung_filter_text.split(','):
                p = p.strip()
                if '(' in p:
                    p = p.split('(')[0].strip()
                if p:
                    bezeichnung_filter_liste.append(p.upper())
        
        # Such-Filter (sucht in Teilenummer UND Bezeichnung)
        search_filter = self.abbau_search_var.get().strip().upper()
        
        # Finde Teile im Lager, die nie verkauft wurden
        nie_verkauft = []
        gesehen = set()  # Vermeiden von Duplikaten
        
        for key, eintrag in self.lagerbestand_data.items():
            et_nr = eintrag.get('et_nr', '')
            if et_nr in gesehen:
                continue
            gesehen.add(et_nr)
            
            bestand = eintrag.get('bestand', 0)
            if bestand < min_bestand:
                continue
            
            bez = eintrag.get('bezeichnung', '')
            
            # Bezeichnung-Filter anwenden
            if bezeichnung_filter_liste:
                bez_upper = bez.upper()
                gefunden = False
                for suchbegriff in bezeichnung_filter_liste:
                    if suchbegriff in bez_upper:
                        gefunden = True
                        break
                if not gefunden:
                    continue
            
            # Such-Filter anwenden (Teilenummer ODER Bezeichnung)
            if search_filter:
                lieferant_nr = eintrag.get('lieferant_et_nr', '').upper()
                if search_filter not in et_nr.upper() and search_filter not in lieferant_nr and search_filter not in bez.upper():
                    continue
            
            # Ermittle Anzahl der Verkäufe - versuche verschiedene Formate
            anzahl_verkaeufe = 0
            lieferant_nr = eintrag.get('lieferant_et_nr', '')
            
            # Alle möglichen Schlüssel für dieses Teil
            such_keys = [
                et_nr,
                et_nr.upper(),
                et_nr.lstrip('0') or et_nr,
                key,
                key.upper() if isinstance(key, str) else key,
                lieferant_nr,
                lieferant_nr.upper() if lieferant_nr else '',
                lieferant_nr.lstrip('0') if lieferant_nr else '',
            ]
            
            for such_key in such_keys:
                if such_key and such_key in verkaufs_zaehler:
                    anzahl_verkaeufe = max(anzahl_verkaeufe, verkaufs_zaehler[such_key])
            
            # Filter nach Max. Verkäufe
            if anzahl_verkaeufe > max_verkaeufe:
                continue
            
            # Lagerwert berechnen
            upe = eintrag.get('upe', 0)
            lagerwert = bestand * upe
            
            if lagerwert < min_wert:
                continue
            
            lagerdauer = eintrag.get('lagerdauer', 0)
            if lagerdauer < min_lagerdauer:
                continue
            
            nie_verkauft.append({
                'teilenummer': et_nr,
                'bezeichnung': eintrag.get('bezeichnung', ''),
                'bestand': bestand,
                'verkaeufe': anzahl_verkaeufe,
                'upe': upe,
                'lagerwert': lagerwert,
                'lagerdauer': lagerdauer,
                'lager': eintrag.get('lager', ''),
            })
        
        # ═══════════════════════════════════════════════════════════════════
        # ZIEL-REICHWEITE UND ÜBERBESTAND BERECHNEN
        # ═══════════════════════════════════════════════════════════════════
        
        # Hole Ziel-Reichweite aus Dropdown (Standard: 6 Monate)
        ziel_reichweite_str = self.abbau_ziel_reichweite_var.get()
        ziel_monate = int(ziel_reichweite_str.split()[0]) if ziel_reichweite_str else 6
        
        # Berechne Zeitraum in Monaten aus Verkaufsdaten
        alle_daten_iso = []
        if self.sqlite_store:
            # Hole Datumsbereich aus SQLite
            min_max = self.statistik.get_date_range()
            if min_max[0] and min_max[1]:
                alle_daten_iso = [min_max[0], min_max[1]]
        elif self.data:
            alle_daten_iso = [r.get('abgabe_iso', '') for r in self.data if r.get('abgabe_iso')]

        if alle_daten_iso:
            from datetime import datetime
            erstes_datum = datetime.strptime(min(alle_daten_iso), '%Y-%m-%d')
            letztes_datum = datetime.strptime(max(alle_daten_iso), '%Y-%m-%d')
            gesamt_monate = max(1, ((letztes_datum.year - erstes_datum.year) * 12 +
                                    letztes_datum.month - erstes_datum.month) + 1)
        else:
            gesamt_monate = 12  # Fallback wenn keine Verkaufsdaten
        
        # Für jedes Teil: Berechne Stk/Mon, Reichweite, Zielbestand, Abbau, Umschlag, Bestellpunkt und Aktion
        import math
        for item in nie_verkauft:
            verkaeufe = item['verkaeufe']
            bestand = item['bestand']

            # Stück pro Monat (Monatsverbrauch)
            stueck_pro_monat = verkaeufe / gesamt_monate if gesamt_monate > 0 else 0
            item['stueck_mon'] = stueck_pro_monat

            # Reichweite = Bestand / Ø_Monatsverbrauch
            if stueck_pro_monat > 0:
                reichweite = bestand / stueck_pro_monat
                item['reichweite'] = reichweite
                item['reichweite_str'] = f"{reichweite:.1f} Mon"
            else:
                item['reichweite'] = float('inf')  # Unendlich
                item['reichweite_str'] = "∞" if bestand > 0 else "-"

            # Zielbestand = AUFRUNDEN(Ø_Monatsverbrauch × ZielMonate)
            zielbestand = math.ceil(stueck_pro_monat * ziel_monate)
            item['zielbestand'] = zielbestand

            # Abbau = MAX(0, Bestand - Zielbestand)
            abbau = max(0, bestand - zielbestand)
            item['abbau'] = abbau

            # UMSCHLAGSHÄUFIGKEIT (Turnover Rate)
            if bestand > 0 and gesamt_monate and gesamt_monate > 0:
                jahresverkauf = (verkaeufe / gesamt_monate) * 12
                umschlagshaeufigkeit = jahresverkauf / bestand
                item['umschlag_str'] = f"{umschlagshaeufigkeit:.1f}x"
            else:
                item['umschlag_str'] = "-"

            # BESTELLPUNKT-EMPFEHLUNG
            if stueck_pro_monat > 0 and bestand > 0:
                bestellpunkt = stueck_pro_monat * 2  # 2 Monate als Bestellpunkt
                if bestand < bestellpunkt:
                    item['bestellpunkt_status'] = "🔴"  # Unter Bestellpunkt - jetzt bestellen!
                elif bestand < bestellpunkt * 1.2:
                    item['bestellpunkt_status'] = "🟡"  # Nahe Bestellpunkt - bald bestellen
                else:
                    item['bestellpunkt_status'] = "🟢"  # Bestand OK
            else:
                item['bestellpunkt_status'] = "⚪"  # Kein Bestand/Bestellpunkt

            # Aktion bestimmen (nach Excel-Formel)
            if verkaeufe == 0 and bestand > 0:
                item['aktion'] = "🔴 ABBAU: Sofort (0 Verkäufe)"
                item['aktion_prio'] = 1
            elif bestand == 0:
                item['aktion'] = "✅ OK: kein Bestand"
                item['aktion_prio'] = 5
            elif verkaeufe < 3:
                item['aktion'] = "🟡 Nicht nachbestellen"
                item['aktion_prio'] = 3
            elif abbau > 0:
                item['aktion'] = f"🟠 ABBAU: {abbau:.0f} Stk reduzieren"
                item['aktion_prio'] = 2
            else:
                item['aktion'] = "✅ OK: Bestand passt"
                item['aktion_prio'] = 4
        
        # ═══════════════════════════════════════════════════════════════════
        # ABC-ANALYSE (nach Pareto-Prinzip basierend auf Lagerwert)
        # ═══════════════════════════════════════════════════════════════════
        # Sortiere nach Lagerwert (höchster zuerst) für ABC-Analyse
        nie_verkauft_sortiert_umsatz = sorted(nie_verkauft, key=lambda x: x['lagerwert'], reverse=True)
        gesamt_lagerwert_alle = sum(item['lagerwert'] for item in nie_verkauft)

        kumulativ_lagerwert = 0
        for i, item in enumerate(nie_verkauft_sortiert_umsatz):
            kumulativ_lagerwert += item['lagerwert']
            kumulativ_prozent_umsatz = (kumulativ_lagerwert / gesamt_lagerwert_alle * 100) if gesamt_lagerwert_alle > 0 else 0

            # ABC-Klassifizierung nach Pareto
            if kumulativ_prozent_umsatz <= 80:
                item['abc'] = '🅰️'  # A-Teile: Top 20% machen 80% Lagerwert
            elif kumulativ_prozent_umsatz <= 95:
                item['abc'] = '🅱️'  # B-Teile: Nächste 30% machen 15% Lagerwert
            else:
                item['abc'] = '🅲️'  # C-Teile: Restliche 50% machen 5% Lagerwert

        # Sortierung: Standardmäßig nach Aktion-Priorität, dann nach Lagerwert
        if hasattr(self, '_abbau_sort_mode') and self._abbau_sort_mode == 'lagerdauer':
            nie_verkauft.sort(key=lambda x: (-x.get('lagerdauer', 0), -x['lagerwert']))
        elif hasattr(self, '_abbau_sort_mode') and self._abbau_sort_mode == 'lagerwert':
            nie_verkauft.sort(key=lambda x: -x['lagerwert'])
        else:
            # Standard: Nach Aktion-Priorität, dann Lagerwert
            nie_verkauft.sort(key=lambda x: (x.get('aktion_prio', 5), -x['lagerwert']))

        # ═══════════════════════════════════════════════════════════════════
        # STATISTIKEN BERECHNEN (für Zusammenfassung)
        # ═══════════════════════════════════════════════════════════════════
        gesamt_teile = len(nie_verkauft)

        # Treffer-Info
        self.abbau_stats_label.config(text=f"(Zeige {min(len(nie_verkauft), 500)} von {gesamt_teile})")
        
        # ═══════════════════════════════════════════════════════════════════
        # TABELLE FÜLLEN mit Aktions-Empfehlung
        # ═══════════════════════════════════════════════════════════════════
        for item in nie_verkauft[:500]:  # Limit auf 500 für Performance
            # Tag basierend auf Aktion-Priorität
            aktion_prio = item.get('aktion_prio', 5)
            
            if aktion_prio == 1:
                tag = 'abbau_sofort'
                prio_symbol = '🔴'
            elif aktion_prio == 2:
                tag = 'abbau_reduzieren'
                prio_symbol = '🟠'
            elif aktion_prio == 3:
                tag = 'nicht_nachbestellen'
                prio_symbol = '🟡'
            else:
                tag = 'ok'
                prio_symbol = '🟢'
            
            self.abbau_tree.insert('', 'end', values=(
                prio_symbol,
                item['teilenummer'],
                item['bezeichnung'],
                item.get('abc', '-'),
                f"{item['bestand']:.0f}",
                item.get('umschlag_str', '-'),
                item.get('bestellpunkt_status', '⚪'),
                f"{item['verkaeufe']}",
                f"{item['stueck_mon']:.1f}",
                item['reichweite_str'],
                f"{item['zielbestand']}",
                f"{item['abbau']:.0f}" if item['abbau'] > 0 else '-',
                f"{item['lagerwert']:.0f}",
                item['aktion'],
            ), tags=(tag,))

    def _init_produkt_liste(self):
        """Initialisiert die Produktliste für Autocomplete nach dem Laden."""
        # Zähle Produkte nach Kategorie
        produkt_zaehler = {}

        # Hole Daten aus SQLite oder In-Memory
        if self.sqlite_store:
            # Daten aus SQLite-Datenbank holen
            cursor = self.sqlite_store.conn.execute("SELECT bezeichnung FROM records WHERE bezeichnung IS NOT NULL AND bezeichnung != ''")
            for row in cursor:
                bez = row[0].strip()
                if bez:
                    erste_worte = bez.split()[0] if bez.split() else bez
                    key = erste_worte.upper()
                    produkt_zaehler[key] = produkt_zaehler.get(key, 0) + 1
        elif self.data:
            for record in self.data:
                bez = record.get('bezeichnung', '').strip()
                if bez:
                    erste_worte = bez.split()[0] if bez.split() else bez
                    key = erste_worte.upper()
                    produkt_zaehler[key] = produkt_zaehler.get(key, 0) + 1
        else:
            return

        # Erstelle Liste mit Anzahl in Klammern, sortiert nach Anzahl
        self.alle_produkte_liste = [f"{name} ({count})" for name, count in
                                     sorted(produkt_zaehler.items(), key=lambda x: -x[1])]

        # Setze die Werte in die Combobox
        if hasattr(self, 'lager_produkt_entry'):
            self.lager_produkt_entry['values'] = self.alle_produkte_liste

    def _search_lager_list(self):
        """Filtert die Lagerhaltungs-Liste nach Suchbegriff."""
        if not self.statistik:
            return
        
        search_term = self.lager_search_var.get().lower().strip()
        
        # Lösche alle Einträge
        for item in self.lager_tree.get_children():
            self.lager_tree.delete(item)
        
        # Ermittle Zeitraum
        monate_str = self.lager_monate_var.get()
        monate = None if monate_str == 'Alle Daten' else int(monate_str.split()[0])
        
        # Hole die Analyse-Daten
        alle_analyse = self.statistik.get_lagerhaltung_analyse(monate=monate)
        
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
        
        # Filtere nach Suchbegriff und fülle Tabelle
        for item in analyse:
            if not search_term or \
               search_term in item['teilenummer'].lower() or \
               search_term in item['bezeichnung'].lower():
                tage_str = f"{item['durchschnitt_tage']:.0f}" if item['durchschnitt_tage'] else "-"
                
                # Farbliches Tag
                if 'Lohnend' in item['kategorie']:
                    tag = 'lohnend'
                elif 'Grenzwertig' in item['kategorie']:
                    tag = 'grenzwertig'
                else:
                    tag = 'nicht_lohnend'
                
                # Stk./Mon.: Stückzahl (Menge) pro Monat (über alle Monate)
                zeitraum_monate = item.get('zeitraum_monate', 1)
                if zeitraum_monate and zeitraum_monate > 0:
                    stueck_pro_mon = item['gesamtmenge'] / zeitraum_monate
                else:
                    stueck_pro_mon = item['gesamtmenge']
                
                # Lagerbestand-Daten holen (versuche verschiedene Formate)
                teilenummer = item['teilenummer']
                lager_info = self.lagerbestand_data.get(teilenummer, {})
                
                # Falls nicht gefunden, versuche ohne führende Nullen
                if not lager_info:
                    tn_stripped = teilenummer.lstrip('0') or teilenummer
                    lager_info = self.lagerbestand_data.get(tn_stripped, {})
                
                # Falls immer noch nicht gefunden, versuche mit führenden Nullen (auf 13 Stellen)
                if not lager_info:
                    tn_padded = teilenummer.zfill(13)
                    lager_info = self.lagerbestand_data.get(tn_padded, {})
                
                bestand = lager_info.get('bestand', 0) if lager_info else 0
                
                # Bestand und Reichweite berechnen
                if self.lagerbestand_data:
                    bestand_str = f"{bestand:.0f}" if bestand > 0 else "0"
                    if stueck_pro_mon > 0 and bestand > 0:
                        reichweite_monate = bestand / stueck_pro_mon
                        reichweite_str = f"{reichweite_monate:.1f} Mon."
                        # Prognose basierend auf Reichweite
                        if reichweite_monate < 2:
                            prognose = "🔴 Zu wenig"
                        elif reichweite_monate <= 12:
                            prognose = "✅ OK"
                        else:
                            prognose = "⚠️ Zu viel"
                    elif stueck_pro_mon == 0 and bestand > 0:
                        reichweite_str = "∞"
                        prognose = "❌ Kein Abverk."
                    else:
                        reichweite_str = "-"
                        prognose = "-" if bestand == 0 else "❌ Kein Abverk."
                else:
                    bestand_str = "-"
                    reichweite_str = "-"
                    prognose = "-"

                # Umschlagshäufigkeit berechnen
                if bestand > 0 and zeitraum_monate and zeitraum_monate > 0:
                    jahresverkauf = (item['gesamtmenge'] / zeitraum_monate) * 12
                    umschlagshaeufigkeit = jahresverkauf / bestand
                    umschlag_str = f"{umschlagshaeufigkeit:.1f}x"
                else:
                    umschlag_str = "-"

                # Bestellpunkt-Status
                if stueck_pro_mon > 0 and bestand > 0:
                    bestellpunkt = stueck_pro_mon * 2
                    if bestand < bestellpunkt:
                        bestellpunkt_status = "🔴"
                    elif bestand < bestellpunkt * 1.2:
                        bestellpunkt_status = "🟡"
                    else:
                        bestellpunkt_status = "🟢"
                else:
                    bestellpunkt_status = "⚪"

                self.lager_tree.insert('', 'end', values=(
                    item['teilenummer'],
                    item['bezeichnung'],
                    item.get('abc', '-'),
                    bestand_str,
                    umschlag_str,
                    bestellpunkt_status,
                    reichweite_str,
                    prognose,
                    item['anzahl_verkäufe'],
                    f"{stueck_pro_mon:.1f}",
                    item.get('anzahl_kunden', '-'),
                    item.get('lagerfaehig', '-'),
                    f"{item['monatsdurchschnitt_umsatz']:.0f}",
                    item.get('trend', '-'),
                    item['kategorie'],
                    item['empfehlung'],
                ), tags=(tag,))

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
        
        # Verwende gleichen Zeitraum wie aktuell angezeigt
        monate_str = self.lager_monate_var.get()
        monate = None if monate_str == 'Alle Daten' else int(monate_str.split()[0])
        
        analyse = self.statistik.get_lagerhaltung_analyse(monate=monate)
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as handle:
                writer = csv.writer(handle, delimiter=';')
                writer.writerow([
                    'Teilenummer', 'Bezeichnung', 'ABC', 'Bestand', 'Umschlag/Jahr',
                    'Bestellpunkt Status', 'Ø Tage', 'Verkäufe', 'Kunden',
                    'Ø Menge/Monat', 'Ø Umsatz/Monat (€)', 'Prognose 3 Monate',
                    'Trend', 'Saisonalität', 'Kategorie', 'Empfehlung'
                ])
                for item in analyse:
                    tage_str = f"{item['durchschnitt_tage']:.0f}" if item['durchschnitt_tage'] else ""

                    # ABC-Klassifizierung
                    abc = item.get('abc', '-')

                    # Bestand
                    bestand = item.get('bestand', 0)

                    # Umschlagshäufigkeit berechnen
                    if bestand > 0 and zeitraum_monate and zeitraum_monate > 0:
                        jahresverkauf = (item['gesamtmenge'] / zeitraum_monate) * 12
                        umschlagshaeufigkeit = jahresverkauf / bestand
                        umschlag_str = f"{umschlagshaeufigkeit:.1f}x".replace('.', ',')
                    else:
                        umschlag_str = "-"

                    # Bestellpunkt-Status berechnen
                    stueck_pro_mon = item['monatsdurchschnitt_menge']
                    if stueck_pro_mon > 0 and bestand > 0:
                        bestellpunkt = stueck_pro_mon * 2  # 2 Monate als Bestellpunkt
                        if bestand < bestellpunkt:
                            bestellpunkt_status = "Rot - Jetzt bestellen"
                        elif bestand < bestellpunkt * 1.2:
                            bestellpunkt_status = "Gelb - Bald bestellen"
                        else:
                            bestellpunkt_status = "Grün - Bestand OK"
                    else:
                        bestellpunkt_status = "Keine Daten"

                    writer.writerow([
                        item['teilenummer'],
                        item['bezeichnung'],
                        abc.replace('🅰️', 'A').replace('🅱️', 'B').replace('🅲️', 'C'),
                        bestand,
                        umschlag_str,
                        bestellpunkt_status,
                        tage_str,
                        item['anzahl_verkäufe'],
                        item.get('anzahl_kunden', ''),
                        f"{item['monatsdurchschnitt_menge']:.2f}".replace('.', ','),
                        f"{item['monatsdurchschnitt_umsatz']:.2f}".replace('.', ','),
                        f"{item.get('prognose_3_monate', 0):.2f}".replace('.', ','),
                        item.get('trend', ''),
                        item.get('saisonalität', ''),
                        item['kategorie'].replace('✅ ', '').replace('⚠️ ', '').replace('❌ ', ''),
                        item['empfehlung'],
                    ])
            self._show_auto_close_info('Export', f'Lagerhaltungs-Analyse gespeichert:\n{filepath}')
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
        
        # Filter anwenden
        items = self._apply_chart_filter(items)
        
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
        
        # Filter anwenden
        items = self._apply_chart_filter(items)
        
        if not items:
            ax.text(0.5, 0.5, 'Keine Daten', ha='center', va='center')
            return
        labels = [f"{item['teilenummer']}\n({item['anzahl_vorgaenge']})" for item in items]
        sizes = [item['anzahl_vorgaenge'] for item in items]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Top Teilenummern (Vorgänge)')

    def _apply_chart_filter(self, items):
        """Filtert Elemente nach eingegebenen Teilenummern."""
        filter_text = self.chart_filter_var.get().strip()
        if not filter_text:
            return items
        
        # Komma-getrennte Teilenummern
        filter_parts = [p.strip().upper() for p in filter_text.split(',') if p.strip()]
        if not filter_parts:
            return items
        
        # Filtere Items
        return [item for item in items if item['teilenummer'].upper() in filter_parts]
    
    def _chart_time(self, ax, metric='vorgaenge'):
        # Filter für spezifische Teilenummern
        filter_text = self.chart_filter_var.get().strip()
        
        if filter_text:
            # Wenn Filter aktiv: Zeige nur diese Teile im Zeitverlauf
            filter_parts = [p.strip().upper() for p in filter_text.split(',') if p.strip()]
            monthly = self.statistik.get_monthly_data_filtered(filter_parts, metric)
        else:
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
            self._show_auto_close_info('Gespeichert', filepath)

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
            self._show_auto_close_info('Export', f'Datei gespeichert: {filepath}')
        except Exception as exc:
            messagebox.showerror('Fehler', f'Export fehlgeschlagen:\n{exc}')


# -----------------------------------------------------------------------------
# Start
# -----------------------------------------------------------------------------
def main():
    app = AnalyseApp()
    
    # Aufräumen beim Beenden
    def on_closing():
        if app.sqlite_store:
            app.sqlite_store.close()
        app.destroy()
    
    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()


if __name__ == '__main__':
    main()
