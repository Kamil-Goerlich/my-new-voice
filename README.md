# My New Voice

**Eine offene Spracherkennung für Menschen mit undeutlicher Sprache – auf Deutsch.**

Viele Menschen sprechen nach einem Schlaganfall, bei Parkinson, Multipler Sklerose, ALS oder aus anderen Gründen so undeutlich, dass weder ihre Mitmenschen noch Siri, Google oder Whisper sie verstehen. My New Voice lernt die Aussprache **einer einzelnen Person** aus wenigen eigenen Aufnahmen, schreibt das Gesagte als Text und lässt es von einer normalen Systemstimme vorlesen.

Alles läuft auf den eigenen Geräten. Sprachaufnahmen sind Gesundheitsdaten und verlassen das eigene Zuhause nicht.

## Stand (Oktober 2026): Prototyp

Erster Test mit meiner eigenen Stimme – 247 Alltagswörter, je 3 Aufnahmen mit dem iPhone-Mikrofon. Jede Aufnahme wurde erkannt, ohne dass der Erkenner sie vorher gehört hat:

| Methode | richtig | richtiges Wort unter Top 3 |
| --- | --- | --- |
| Whisper ohne Anpassung | 2,7 % | – |
| Abgleich mit eigenen Aufnahmen (ohne Training) | 33,3 % | 45,2 % |
| **Kleines Netz, nur auf meine Stimme trainiert** | **59,3 %** | **72,1 %** |

(Zufall: 0,4 %.) Der Erkenner hatte dabei nur 2 Lernaufnahmen pro Wort.

## Aufbau

| Ordner | Inhalt |
| --- | --- |
| `stimmtrainer/` | Web-App fürs Handy: zeigt ein Wort, nimmt es auf, Aufnahme in Runden, Sicherung als ZIP mit Tabelle. Läuft ohne Installation im Browser. |
| `training/erkenner_train.py` | Training eines kleinen Erkenners auf die eigene Stimme: Whisper-Encoder als Klangmerkmale, kleines Netz (unter 1 Mio. Parameter), ehrlicher Test (Kreuzvalidierung über die Aufnahmen). |
| `training/whisper_test.py` | Vergleichstest: Wie gut versteht normales Whisper die Aufnahmen? |

## Geplant

- Training mit einem Klick für Mac und Windows – ohne Programmierkenntnisse
- Erkennung ganzer Alltagssätze
- Handy-App für iPhone und Android: erkennt offline und liest mit der Systemstimme vor
- Tests mit weiteren Betroffenen

## Warum

Ich lebe seit 26 Jahren mit einer Sprechbehinderung, die immer stärker geworden ist. Logopädie hat mir nicht geholfen. Ich möchte wieder verstanden werden – und dass andere Betroffene das auch können, kostenlos und auf Deutsch.

## Datenschutz

Dieses Repository enthält **nur Code**. Sprachaufnahmen, Datensätze und trainierte Modelle gehören der jeweiligen Person und werden nie hochgeladen (siehe `.gitignore`).

## Lizenz

MIT – siehe [LICENSE](LICENSE). Whisper (OpenAI) steht ebenfalls unter MIT-Lizenz.

Kontakt: Kamil Goerlich · Insel Reichenau, Bodensee
