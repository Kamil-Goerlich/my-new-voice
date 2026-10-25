# Whisper-Test fuer den Stimmtrainer-Datensatz (25.10.2026)
# Start im Terminal (Mac Studio):
#   python training/whisper_test.py
# Drei Methoden:
#   A = Whisper frei (wie jede normale Spracherkennung)
#   B = Whisper mit Wortliste (darf nur eines der 247 Woerter waehlen)
#   C = Persoenlicher Abgleich (vergleicht mit deinen eigenen anderen Aufnahmen desselben Wortes)
import os, sys, csv, re, time, wave, math
import numpy as np, torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODELL = os.environ.get("MODELL", "openai/whisper-small")
DIR = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(DIR, "datensatz_woerter")
OUT_CSV = os.path.join(DS, "whisper_ergebnis.csv")
OUT_TXT = os.path.join(DS, "whisper_ergebnis.txt")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Modell {MODELL} auf {dev}", flush=True)

def lade(p):
    with wave.open(p) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return a

def norm(s):
    s = s.lower().strip()
    s = re.sub(r"[^\wäöüß ]+", "", s)
    return re.sub(r"\s+", " ", s).strip()

rows = [r for r in csv.DictReader(open(os.path.join(DS, "datensatz.csv"), encoding="utf-8"))
        if os.path.exists(os.path.join(DS, "wav", r["datei"]))]
woerter = sorted({r["wort"] for r in rows}, key=lambda w: int(next(x["nr"] for x in rows if x["wort"] == w)))
print(f"{len(rows)} Aufnahmen, {len(woerter)} Woerter", flush=True)

proc = WhisperProcessor.from_pretrained(MODELL)
model = WhisperForConditionalGeneration.from_pretrained(MODELL).to(dev).eval()
tok = proc.tokenizer
prefix = [model.config.decoder_start_token_id] + [i for _, i in proc.get_decoder_prompt_ids(language="de", task="transcribe", no_timestamps=True)]
eot = tok.eos_token_id

# Kandidaten fuer B: "Ja." usw.
cand = []
for w in woerter:
    t = w[0].upper() + w[1:] + "."
    cand.append(prefix + tok.encode(" " + t, add_special_tokens=False) + [eot])
L = max(len(c) for c in cand)
ids = torch.full((len(cand), L), eot, dtype=torch.long)
mask = torch.zeros((len(cand), L), dtype=torch.long)
for k, c in enumerate(cand):
    ids[k, :len(c)] = torch.tensor(c); mask[k, :len(c)] = 1
ids, mask = ids.to(dev), mask.to(dev)

ergebnis, embs = [], []
t0 = time.time()
with torch.no_grad():
    for n, r in enumerate(rows):
        a = lade(os.path.join(DS, "wav", r["datei"]))
        feat = proc(a, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
        # A: frei
        gen = model.generate(feat, language="de", task="transcribe", max_new_tokens=12)
        frei = tok.decode(gen[0], skip_special_tokens=True).strip()
        # Encoder (auf die Laenge der Aufnahme gekuerzt)
        enc = model.get_encoder()(feat).last_hidden_state
        fr = min(1500, max(50, int(len(a) / 16000 * 50) + 25))
        e = enc[:, :fr]
        # C: Fingerabdruck der Aufnahme
        v = torch.cat([e.mean(1), e.std(1)], -1)[0]
        embs.append(torch.nn.functional.normalize(v, dim=0).cpu().numpy())
        # B: Wortliste
        out = model(encoder_outputs=(e.expand(len(cand), -1, -1),), decoder_input_ids=ids[:, :-1], decoder_attention_mask=mask[:, :-1])
        lp = torch.log_softmax(out.logits.float(), -1)
        tgt = ids[:, 1:]; m = mask[:, 1:].clone(); m[:, :len(prefix) - 1] = 0
        sc = (lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * m).sum(1)
        b = woerter[int(sc.argmax())]
        ergebnis.append({"datei": r["datei"], "wort": r["wort"], "kategorie": r["kategorie"], "A_frei": frei,
                         "A_richtig": int(norm(frei) == norm(r["wort"])), "B_wortliste": b, "B_richtig": int(b == r["wort"])})
        if n % 25 == 0:
            el = time.time() - t0
            print(f"{n+1}/{len(rows)}  {r['wort']!r} -> frei {frei!r} | Liste {b!r}   ({el/(n+1)*(len(rows)-n-1)/60:.1f} Min. Rest)", flush=True)

# C: jede Aufnahme gegen alle anderen (eigene Aufnahme ausgenommen)
E = np.stack(embs); S = E @ E.T; np.fill_diagonal(S, -9)
W = np.array([r["wort"] for r in rows])
for k in range(len(rows)):
    best = {}
    for j in np.argsort(-S[k])[:40]:
        best.setdefault(W[j], []).append(S[k, j])
    rank = sorted(best, key=lambda w: -np.mean(sorted(best[w], reverse=True)[:2]))
    ergebnis[k]["C_abgleich"] = rank[0]; ergebnis[k]["C_richtig"] = int(rank[0] == W[k])
    ergebnis[k]["C_top3"] = int(W[k] in rank[:3])

with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(ergebnis[0].keys())); w.writeheader(); w.writerows(ergebnis)
N = len(ergebnis); pc = lambda k: 100 * sum(e[k] for e in ergebnis) / N
txt = [f"Whisper-Test {time.strftime('%d.%m.%Y %H:%M')} - Modell {MODELL} - {N} Aufnahmen, {len(woerter)} Woerter",
       f"A  Whisper frei:            {pc('A_richtig'):5.1f} % richtig",
       f"B  Whisper mit Wortliste:   {pc('B_richtig'):5.1f} % richtig",
       f"C  Persoenlicher Abgleich:  {pc('C_richtig'):5.1f} % richtig  (richtiges Wort unter den Top 3: {pc('C_top3'):.1f} %)",
       f"(Zufall bei {len(woerter)} Woertern: {100/len(woerter):.1f} %)", "", "Nach Kategorie (A / B / C):"]
for kat in dict.fromkeys(e["kategorie"] for e in ergebnis):
    g = [e for e in ergebnis if e["kategorie"] == kat]
    p = lambda k: 100 * sum(e[k] for e in g) / len(g)
    txt.append(f"  {kat:26s} {p('A_richtig'):5.1f}  {p('B_richtig'):5.1f}  {p('C_richtig'):5.1f}")
open(OUT_TXT, "w", encoding="utf-8").write("\n".join(txt) + "\n")
print("\n" + "\n".join(txt))
print(f"\nFertig. Details: {OUT_CSV}")
os.system("osascript -e 'display notification \"Whisper-Test fertig\" with title \"My New Voice\"'")
