LaTeX-Vorlage für Präsentationen
================================

Dies ist die Vorlage für Präsentationen am Lehrstuhl Software Design and
Quality (SDQ) am Institut für Datenorganisation und Programmstrukturen (IPD)
des Karlsruher Instituts für Technologie (KIT).

Version
=======

Version: 2.5
Autor: Dr.-Ing. Erik Burger (burger@kit.edu)
mit Beiträgen von Christian Hammer und Klaus Krogmann

Siehe https://sdqweb.ipd.kit.edu/wiki/Dokumentvorlagen

Verwendung
==========

Das vorliegende Paket dient als Vorlage für Präsentationen im KIT-Design (http://intranet.kit.edu/gestaltungsrichtlinien.php, Version 2.0).

Es basiert auf LaTeX Beamer (https://ctan.org/pkg/beamer).

Optionen der Dokumentklasse `sdqbeamer`
-----------------------------------------
Durch die folgenden Optionen kann das Seitenverhältnis der Folien bestimmt werden:

| Seitenverhältnis | Option              |
| ---------------- | ------------------- |
| 16:9             | `16:9`  (Standard)  |
| 4:3              | `4:3`               |

Die Plazierung der Navigationsleiste kann durch folgende Optionen beeinflußt werden:

| Position                                      | Option                    |
| --------------------------------------------- | ------------------------- |
| unterer Rand der weißen Fläche                | `navbarinline` (Standard) |
| in der grauen Fußzeile, stärker komprimiert   | `navbarinfooter`          |
| Seitenleiste am linken Rand der weißen Fläche | `navbarside`              |
| keine Navigationsleiste                       | `navbaroff`               |

Als Sprache sind Deutsch und Englisch verfügbar. Durch die Sprachwahl werden automatisch die passenden Logos und Formate (z.B. Datum) gewählt.

| Sprache  |                 |
| -------- |---------------- |
| Englisch | `en` (Standard) |
| Deutsch  | `de`            |

Beispiel: `\documentclass[en,16:9,navbarinline]{sdqbeamer}`

Titelbild
---------

Als Bild auf der Titelfolie wird standardmäßig die Datei `logos/bauplan_blau_wide.jpg` eigebunden. (Vielen Dank an Klaus Krogmann für die Fotografie.) Um ein eigenes Bild zu verwenden, bitte die Datei (z.B. `myimage.jpg`) ins `logos/`-Verzeichnis legen und den folgenden Befehl in die Präambel des Dokuments einfügen:

`\titleimage{myimage}` (ohne Dateiendung)

Für 16:9-Folien sollte das Verhältnis des Bildes 169:40 betragen, für 4:3-Folien 63:20. Es können auch breitere Bilder verwendet werden, da das Titelbild auf die Höhe des Rahmens skaliert wird.

Institutslogo
-------------

Standardmäßig wird das SDQ-Logo verwendet. Ein eigenes Logo (z.B. `mylogo.pdf`) kann in das Verzeichnis `logos/` gelegt und durch folgenden Aufruf aktiviert werden:

`\grouplogo{mylogo}` (ohne Dateiendung)

LaTeX allgemein
---------------
Siehe https://sdqweb.ipd.kit.edu/wiki/LaTeX

Dateistruktur
============
`presentation.tex`
------------------
Hauptdatei des LaTeX-Dokuments.

`presentation.bib`
-------------
Beispieldatei für BibTeX-Referenzen
https://sdqweb.ipd.kit.edu/wiki/BibTeX-Literaturlisten

`sdqbeamer.cls`
-----------------
Dokumentklasse für Präsentationen im KIT-Design.

`logos/`
--------
In diesem Verzeichnis befinden das das KIT- und SDQ-Logo als PDF sowie das Hintergrundbild der Titelfolie als JPG.

`templates/`
------------
In diesem Verzeichnis befindet sich das Farbschema des KIT (`beamercolorhemekit.sty`) sowie der Code für die halbgerundeten Aufzählungpunkte (`semirounded.sty`, vielen Dank an Christian Hammer).

`README.md`
-----------
Dieser Text.
