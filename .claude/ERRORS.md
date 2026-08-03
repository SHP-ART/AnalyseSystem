# ERRORS.md – Bekannte Fehler & Lösungen

Format pro Eintrag:

```markdown
## [DATUM] – [Kurzbezeichnung]
**Fehler:** <exakte Fehlermeldung>
**Kontext:** <wo ist er aufgetreten, welche Aktion hat ihn ausgelöst>
**Ursache:** <was war der Grund>
**Lösung:** <was hat das Problem behoben>
**Prävention:** <wie vermeidet man diesen Fehler künftig>
```

---

## 2026-08-03 – Lagerabbau zählte Transaktionen statt Mengen
**Fehler:** `verkaufs_zaehler` zählte Zeilenanzahl (Transaktionen) statt SUM(menge) – alle Lagerabbau-Kennzahlen waren falsch
**Kontext:** `_update_lagerabbau()` – SQL `SELECT teilenummer FROM records` ohne SUM/GROUP BY, In-Memory `+ 1` statt `+ menge`
**Ursache:** Ursprüngliche Implementierung zählte Verkäufe als "Anzahl Vorgänge", nicht als "verkaufte Stückzahl"
**Lösung:** SQL auf `SELECT teilenummer, COALESCE(SUM(menge), 0) ... GROUP BY teilenummer` geändert, In-Memory `+ record.get('menge', 0)`
**Prävention:** Bei neuen Aggregationen: Immer explizit prüfen ob Mengen oder Counts gemeint sind

## 2026-08-03 – Monatsdurchschnitt nur über aktive Verkaufsperiode
**Fehler:** `monatsdurchschnitt_menge/umsatz` berechnet als `gesamtmenge / (tage_gesamt / 30.44)` – tage_gesamt war nur die Spanne zwischen erstem und letztem Verkauf, nicht der Gesamtzeitraum
**Kontext:** `get_lagerhaltung_analyse()` Memory-Modus Zeile 658-661, SQLite-Modus Zeile 843-848. Prognose basierte auf diesem überhöhten Wert.
**Ursache:** ursprünglich als "Durchschnitt während aktiver Periode" konzipiert, aber für Prognose/Reichweite wird der Durchschnitt über den Gesamtzeitraum benötigt
**Lösung:** `monatsdurchschnitt_menge = gesamtmenge / gesamt_zeitraum_monate` (Memory + SQLite)
**Prävention:** Monatsdurchschnitte immer über definierten Gesamtzeitraum berechnen, nicht über Per-teile-Spanne

## 2026-08-03 – Division durch Null in Trendberechnung
**Fehler:** `ZeroDivisionError` in `_berechne_trend()` bei `zweite_hälfte_menge / erste_hälfte_menge` wenn `erste_hälfte_menge == 0`
**Kontext:** Teile mit Verkäufen nur in der 2. Hälfte des Zeitraums (z.B. Neuteile)
**Ursache:** Keine Prüfung ob `erste_hälfte_menge > 0` vor Division
**Lösung:** `erste_hälfte_menge == 0 and zweite_hälfte_menge > 0` → "↗️ Neu", Guards vor allen Divisionen
**Prävention:** Vor jeder Division prüfen ob Nenner > 0

## 2026-08-03 – Saisonalität: Quartale über Jahre gemischt
**Fehler:** Saisonalitätserkennung gruppierte Q1 aus allen Jahren zusammen, was bei mehrjährigen Daten den Durchschnitt verzerrte
**Kontext:** `_erkenne_saisonalitaet()` – `quartal_verkäufe[quartal] += v['menge']` ohne Jahrstrennung
**Ursache:** Einfache Implementierung ohne Berücksichtigung von Mehrjahresdaten
**Lösung:** Pro (Jahr, Quartal) gruppieren, dann Durchschnitt pro Quartalnummer über alle Jahre
**Prävention:** Bei Zeitreihen-Aggregationen immer prüfen ob Jahrstrennung nötig ist
