# AGENTS.md

Zentraler Einstiegspunkt für alle KI-Coding-Tools in diesem Projekt
(Claude Code, Codex, Kilo Code, Kimi, Z.ai, etc.).

**Keine eigenen Konfigurations- oder Dokumentationsdateien anlegen.**
Alle Regeln und der Projektstatus stehen in den Dateien unten – dort lesen
und bei jeder Änderung aktualisieren, nicht duplizieren oder durch ein
eigenes Format ersetzen.

| Datei | Inhalt | Wann lesen |
|---|---|---|
| `CLAUDE.md` | Sprache, Stack, Konventionen, TDD-Regeln, kritische Regeln, Projektkontext | immer zuerst |
| `.claude/PROJEKT.md` | Architektur, Komponenten, Datenmodell, Projektstatus | vor jeder Änderung |
| `.claude/RETRO.md` | letzter Sessionstand, offene Punkte | vor jeder Session |
| `.claude/ERRORS.md` | bekannte Fehler & Lösungen | beim Debugging |
| `FEATURES.md` | bestehende Funktionen (Regressionsschutz) | vor/nach jeder Erweiterung |
| `.claude/SECRETS.md` | lokale Zugangsdaten/Server/Deploy-Infos (falls vorhanden, gitignored) | nur bei Bedarf |

Fehlt eine dieser Dateien: gemäß `CLAUDE.md` anlegen, nicht durch ein
tool-eigenes Format ersetzen. Ausnahme `.claude/SECRETS.md`: nur anlegen wenn
tatsächlich Zugangsdaten anfallen, siehe eigene Sektion dazu.
