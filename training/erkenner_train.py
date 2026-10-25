# Erkenner-Training (25.10.2026) – kleiner Worterkenner nur auf die eigene Stimme
#
# Start im Terminal (Mac Studio), eine Zeile:
#   python training/erkenner_train.py
#
# Was passiert (ca. 5-10 Minuten):
#   1. Jede Aufnahme aus datensatz_woerter/wav wird durch den Whisper-Encoder geschickt
#      (nur der "Hoer-Teil" von Whisper, nicht die Texterkennung). Zusaetzlich je eine
#      etwas langsamere und schnellere Fassung (Tempo 0,9 / 1,1) als Trainings-Futter.
#   2. Ehrlicher Test: Aufnahme 1 wird erkannt, gelernt wurde nur aus Aufnahme 2+3 usw.
#      (jede Aufnahme kommt einmal dran, der Erkenner hat sie vorher nie gehoert).
#      Verglichen werden:
#        M1 Durchschnitts-Abgleich (wie Methode C vom Whisper-Test, aber beste Encoder-Schicht)
#        M2 Verlaufs-Abgleich (DTW: vergleicht den zeitlichen Ablauf des Wortes)
#        M3 Kleines Netz, trainiert nur auf die eigene Stimme
#        M4 M2 + M3 kombiniert
#   3. Danach wird der Erkenner mit ALLEN Aufnahmen trainiert und gespeichert:
#        my_new_voice/erkenner/erkenner.pt        (kleines Netz, ein paar MB)
#        my_new_voice/erkenner/ergebnis.txt       (Zusammenfassung)
#        my_new_voice/erkenner/ergebnis.csv       (jede Aufnahme einzeln)
#        ~/my_new_voice_lokal/vorlagen.npz        (Vergleichs-Vorlagen fuer M2, bewusst NICHT in iCloud)
#
# Laeuft auch spaeter mit 10 Aufnahmen pro Wort einfach nochmal – es nimmt, was im Datensatz liegt.
import os, csv, time, json, math, random, wave, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

MODELL = os.environ.get("MODELL", "openai/whisper-small")
SCHICHTEN = [int(x) for x in os.environ.get("SCHICHTEN", "6,9,12").split(",")]  # Encoder-Schichten fuer M2/M3
EPOCHEN = int(os.environ.get("EPOCHEN", "60"))
TEMPI = [1.0, 0.9, 1.1]
KANDIDATEN = 40          # M2 prueft die 40 besten Woerter aus M1 genauer
AUSSCHLUSS = {"005_hallo_1.wav", "031_ich_1.wav"}   # fast stumm, werden nachaufgenommen
MIN_PEGEL = -55.0

DIR = os.path.dirname(os.path.abspath(__file__))
DS = os.environ.get("DATENSATZ", os.path.join(DIR, "datensatz_woerter"))
OUT = os.environ.get("AUSGABE", os.path.join(DIR, "erkenner"))
LOKAL = os.environ.get("LOKAL", os.path.expanduser("~/my_new_voice_lokal"))
os.makedirs(OUT, exist_ok=True); os.makedirs(LOKAL, exist_ok=True)
random.seed(7); np.random.seed(7); torch.manual_seed(7)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
T0 = time.time()
def log(*a):
    print(f"[{(time.time()-T0)/60:4.1f} Min]", *a, flush=True)

# ---------------------------------------------------------------- Audio
def lade(p):
    with wave.open(p) as w:
        sr = w.getframerate(); ch = w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    if ch > 1: a = a.reshape(-1, ch).mean(1)
    if sr != 16000:
        a = np.interp(np.arange(0, len(a), sr / 16000), np.arange(len(a)), a).astype(np.float32)
    return a

def pegel(a):
    r = math.sqrt(float(np.mean(a ** 2)) + 1e-12)
    a = a * (0.1 / r)                     # ca. -20 dBFS
    m = float(np.max(np.abs(a))) + 1e-9
    return (a * min(1.0, 0.99 / m)).astype(np.float32)

def tempo(a, f):
    if f == 1.0: return a
    n = int(len(a) / f)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)

# ---------------------------------------------------------------- Datensatz
rows = []
for r in csv.DictReader(open(os.path.join(DS, "datensatz.csv"), encoding="utf-8")):
    p = os.path.join(DS, "wav", r["datei"])
    if not os.path.exists(p) or r["datei"] in AUSSCHLUSS: continue
    try:
        if float(r.get("pegel_db") or 0) < MIN_PEGEL: continue
    except ValueError: pass
    rows.append(r)
woerter = sorted({r["wort"] for r in rows}, key=lambda w: int(next(x["nr"] for x in rows if x["wort"] == w)))
WI = {w: k for k, w in enumerate(woerter)}
KAT = {r["wort"]: r["kategorie"] for r in rows}
takes_alle = sorted({int(r["aufnahme"]) for r in rows})
log(f"{len(rows)} Aufnahmen, {len(woerter)} Woerter, Aufnahme-Nummern {takes_alle[0]}-{takes_alle[-1]}, Rechner: {dev}")

# ---------------------------------------------------------------- Merkmale (Whisper-Encoder)
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration
fe = WhisperFeatureExtractor.from_pretrained(MODELL)
enc = WhisperForConditionalGeneration.from_pretrained(MODELL).get_encoder().to(dev).eval()
NL = enc.config.encoder_layers + 1

items = []   # je Aufnahme x Tempo
jobs = [(r, f) for r in rows for f in TEMPI]
B = 8
with torch.no_grad():
    for s in range(0, len(jobs), B):
        part = jobs[s:s + B]
        auds = [tempo(pegel(lade(os.path.join(DS, "wav", r["datei"]))), f) for r, f in part]
        feat = fe(auds, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
        hs = enc(feat, output_hidden_states=True).hidden_states       # NL x [B,1500,D]
        for b, ((r, f), a) in enumerate(zip(part, auds)):
            fr = min(1500, max(10, int(len(a) / 16000 * 50) + 2))
            st = torch.stack([torch.cat([h[b, :fr].mean(0), h[b, :fr].std(0)]) for h in hs])   # [NL, 2D]
            fm = torch.stack([hs[l][b, :fr] for l in SCHICHTEN])                                # [S, fr, D]
            fm = F.avg_pool1d(fm.transpose(1, 2), 2, ceil_mode=True).transpose(1, 2)          # 25 Bilder/s
            items.append({"datei": r["datei"], "wort": r["wort"], "y": WI[r["wort"]], "take": int(r["aufnahme"]),
                          "tempo": f, "stats": F.normalize(st.float(), dim=-1).cpu().numpy(),
                          "frames": fm.half().cpu()})
        if (s // B) % 25 == 0:
            el = time.time() - T0; done = s + len(part)
            log(f"Merkmale {done}/{len(jobs)}  (noch ca. {el/done*(len(jobs)-done)/60:.1f} Min)")
orig = [it for it in items if it["tempo"] == 1.0]
Y = np.array([it["y"] for it in orig]); TK = np.array([it["take"] for it in orig])
log("Merkmale fertig")

def rang_aus_scores(sc_words):  # hoeher = besser
    return list(np.argsort(-sc_words))

def bewerte(ranks):
    top1 = np.mean([rk[0] == y for rk, y in zip(ranks, Y)]) * 100
    top3 = np.mean([y in rk[:3] for rk, y in zip(ranks, Y)]) * 100
    return top1, top3

# ---------------------------------------------------------------- M1 Durchschnitt, pro Schicht
KMAX = max(int((Y == w).sum()) for w in range(len(woerter)))
IDX = np.full((len(woerter), KMAX), -1)
for w in range(len(woerter)):
    j = np.where(Y == w)[0]; IDX[w, :len(j)] = j

def m1_scores(layer):
    E = np.stack([it["stats"][layer] for it in orig]); S = E @ E.T
    V = S[:, np.maximum(IDX, 0)]                                     # [N, W, K]
    bad = (IDX[None] < 0) | (TK[np.maximum(IDX, 0)][None] == TK[:, None, None])
    V = np.where(bad, -np.inf, V)
    V = -np.sort(-V, axis=2)[:, :, :2]                               # beste 2 je Wort
    n_ok = np.isfinite(V).sum(2)
    out = np.where(n_ok > 0, np.where(np.isfinite(V), V, 0).sum(2) / np.maximum(n_ok, 1), -9.0)
    return out

m1 = {}
for l in range(NL):
    sc = m1_scores(l); m1[l] = (sc, bewerte([rang_aus_scores(s) for s in sc]))
L1 = max(m1, key=lambda l: m1[l][1][0])
log("M1 je Schicht (Top1): " + "  ".join(f"{l}:{m1[l][1][0]:.0f}%" for l in range(NL)) + f"  -> beste Schicht {L1}")
M1SC = m1[L1][0]; M1RK = [rang_aus_scores(s) for s in M1SC]

# ---------------------------------------------------------------- M2 Verlauf (DTW)
try:
    from numba import njit
    @njit(cache=False)
    def _dtw(C):
        n, m = C.shape; D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        return D[n, m] / (n + m)
except Exception:
    def _dtw(C):
        n, m = C.shape; D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        return D[n, m] / (n + m)

def frames_np(it, si):
    x = it["frames"][si].float().numpy(); x = x - x.mean(0, keepdims=True)
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float64)

def m2_ranks(si):
    FR = [frames_np(it, si) for it in orig]
    ranks, scores = [], []
    for i in range(len(orig)):
        kand = M1RK[i][:KANDIDATEN]; sc = {}
        for w in kand:
            ds = [_dtw(1.0 - FR[i] @ FR[j].T) for j in np.where((Y == w) & (TK != TK[i]))[0]]
            if ds: sc[w] = min(ds)
        rk = sorted(sc, key=lambda w: sc[w]) + [w for w in M1RK[i] if w not in sc]
        ranks.append(rk); scores.append(sc)
    return ranks, scores

m2 = {}
for si, l in enumerate(SCHICHTEN):
    rk, sc = m2_ranks(si); m2[si] = (rk, sc, bewerte(rk))
    log(f"M2 Schicht {l}: Top1 {m2[si][2][0]:.1f} %  Top3 {m2[si][2][1]:.1f} %")
S2 = max(m2, key=lambda s: m2[s][2][0]); M2RK = m2[S2][0]

# ---------------------------------------------------------------- M3 kleines Netz
class Netz(nn.Module):
    def __init__(s, ns, d, nk, h=192):
        super().__init__()
        s.lw = nn.Parameter(torch.zeros(ns)); s.ln = nn.LayerNorm(d); s.inp = nn.Linear(d, h)
        s.c1 = nn.Conv1d(h, h, 5, padding=2); s.c2 = nn.Conv1d(h, h, 5, padding=2)
        s.att = nn.Linear(h, 1); s.drop = nn.Dropout(0.3); s.out = nn.Linear(2 * h, nk)
    def forward(s, x, mask):                      # x [B,S,T,D]  mask [B,T] bool
        x = (x * torch.softmax(s.lw, 0)[None, :, None, None]).sum(1)
        x = F.gelu(s.inp(s.ln(x))) * mask[..., None]
        x = x.transpose(1, 2)
        x = x + F.gelu(s.c1(x)); x = x * mask[:, None]
        x = x + F.gelu(s.c2(x)); x = x * mask[:, None]
        x = x.transpose(1, 2)
        a = torch.softmax(s.att(x).squeeze(-1).masked_fill(~mask, -1e4), 1)[..., None]
        mu = (a * x).sum(1); sd = torch.sqrt(((a * (x - mu[:, None]) ** 2).sum(1)).clamp_min(1e-6))
        return s.out(s.drop(torch.cat([mu, sd], -1)))

def stapel(lst, aug):
    xs = []
    for it in lst:
        x = it["frames"].float()
        if aug:
            T = x.shape[1]
            if T > 8:
                L = max(6, int(T * random.uniform(0.8, 1.0))); st = random.randint(0, T - L); x = x[:, st:st + L].clone()
            T = x.shape[1]
            for _ in range(2):
                w = random.randint(0, max(1, int(T * 0.15))); st = random.randint(0, max(0, T - w)); x[:, st:st + w] = 0
            x = x + 0.05 * x.std() * torch.randn_like(x)
        xs.append(x)
    T = max(x.shape[1] for x in xs); S, D = xs[0].shape[0], xs[0].shape[2]
    X = torch.zeros(len(xs), S, T, D); M = torch.zeros(len(xs), T, dtype=torch.bool)
    for k, x in enumerate(xs): X[k, :, :x.shape[1]] = x; M[k, :x.shape[1]] = True
    return X.to(dev), M.to(dev)

def trainiere(train, tag=""):
    torch.manual_seed(7); random.seed(7)
    D = train[0]["frames"].shape[2]
    net = Netz(len(SCHICHTEN), D, len(woerter)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    steps = EPOCHEN * math.ceil(len(train) / 64); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.1)
    for ep in range(EPOCHEN):
        net.train(); random.shuffle(train); tot = 0
        for s in range(0, len(train), 64):
            b = train[s:s + 64]; X, M = stapel(b, True)
            y = torch.tensor([it["y"] for it in b], device=dev)
            loss = F.cross_entropy(net(X, M), y, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
        if ep % 20 == 19 or ep == EPOCHEN - 1: log(f"  Netz{tag} Runde {ep+1}/{EPOCHEN}  Fehler {tot/len(train):.2f}")
    return net.eval()

def vorhersage(net, lst):
    out = []
    with torch.no_grad():
        for s in range(0, len(lst), 64):
            X, M = stapel(lst[s:s + 64], False); out.append(torch.log_softmax(net(X, M).float(), -1).cpu().numpy())
    return np.concatenate(out)

M3LP = np.zeros((len(orig), len(woerter)))
for t in takes_alle:
    test_idx = [i for i, it in enumerate(orig) if it["take"] == t]
    if not test_idx: continue
    train = [it for it in items if it["take"] != t]
    log(f"M3: lerne ohne Aufnahme {t} ({len(train)} Beispiele), teste {len(test_idx)} Aufnahmen")
    net = trainiere(train, f" (ohne {t})")
    M3LP[test_idx] = vorhersage(net, [orig[i] for i in test_idx])
M3RK = [rang_aus_scores(s) for s in M3LP]

# ---------------------------------------------------------------- M4 Kombination (Rang-Mischung)
def rrf(*rks, k=5):
    out = []
    for parts in zip(*rks):
        sc = np.zeros(len(woerter))
        for rk in parts:
            for pos, w in enumerate(rk): sc[w] += 1.0 / (k + pos)
        out.append(rang_aus_scores(sc))
    return out
M4RK = rrf(M2RK, M3RK)

ERG = {"M1 Durchschnitt (Schicht %d)" % L1: bewerte(M1RK),
       "M2 Verlauf/DTW (Schicht %d)" % SCHICHTEN[S2]: bewerte(M2RK),
       "M3 Kleines Netz": bewerte(M3RK),
       "M4 Verlauf + Netz": bewerte(M4RK)}
best_name = max(ERG, key=lambda k: ERG[k][0])
BESTRK = {"M1": M1RK, "M2": M2RK, "M3": M3RK, "M4": M4RK}[best_name[:2]]

# ---------------------------------------------------------------- Bericht
pro_wort = {}
for i, it in enumerate(orig): pro_wort.setdefault(it["wort"], []).append(woerter[BESTRK[i][0]])
txt = [f"Erkenner-Test {time.strftime('%d.%m.%Y %H:%M')} – {len(orig)} Aufnahmen, {len(woerter)} Woerter, "
       f"ca. {len(orig)/len(woerter):.1f} Aufnahmen pro Wort",
       "Ehrlicher Test: jede Aufnahme wird erkannt, ohne dass der Erkenner sie vorher gehoert hat.", "",
       f"{'Methode':34s} {'richtig':>8s} {'Top 3':>8s}",
       f"{'Whisper frei (Test vom 25.10.)':34s} {'2.7 %':>8s} {'':>8s}",
       f"{'Persoenlicher Abgleich C (25.10.)':34s} {'33.3 %':>8s} {'45.2 %':>8s}"]
for k, (a, b) in ERG.items(): txt.append(f"{k:34s} {a:6.1f} % {b:6.1f} %")
txt += [f"(Zufall: {100/len(woerter):.1f} %)", "", f"Beste Methode: {best_name}", "", "Nach Kategorie (beste Methode):"]
for kat in dict.fromkeys(KAT[it["wort"]] for it in orig):
    idx = [i for i, it in enumerate(orig) if KAT[it["wort"]] == kat]
    txt.append(f"  {kat:26s} {100*np.mean([BESTRK[i][0]==Y[i] for i in idx]):5.1f} %")
schwer = sorted(pro_wort, key=lambda w: (sum(p == w for p in pro_wort[w]) / len(pro_wort[w]), w))[:25]
txt += ["", "Schwierigste Woerter (erkannt als ...):"]
for w in schwer:
    txt.append(f"  {w:16s} {sum(p == w for p in pro_wort[w])}/{len(pro_wort[w])}   -> " + ", ".join(pro_wort[w]))
ver = {}
for w, ps in pro_wort.items():
    for p in ps:
        if p != w: ver[(w, p)] = ver.get((w, p), 0) + 1
txt += ["", "Haeufigste Verwechslungen:"] + [f"  {a} -> {b}  ({n}x)" for (a, b), n in sorted(ver.items(), key=lambda x: -x[1])[:15]]

with open(os.path.join(OUT, "ergebnis.csv"), "w", newline="", encoding="utf-8") as fh:
    wr = csv.writer(fh); wr.writerow(["datei", "wort", "M1", "M2", "M3", "M4", "M4_top3", "M4_richtig"])
    for i, it in enumerate(orig):
        wr.writerow([it["datei"], it["wort"], woerter[M1RK[i][0]], woerter[M2RK[i][0]], woerter[M3RK[i][0]],
                     woerter[M4RK[i][0]], " | ".join(woerter[w] for w in M4RK[i][:3]), int(M4RK[i][0] == Y[i])])

# ---------------------------------------------------------------- Endgueltiger Erkenner (alle Aufnahmen)
log("Endgueltiger Erkenner: lerne aus allen Aufnahmen")
net = trainiere(list(items), " (alle)")
torch.save({"state": {k: v.half() for k, v in net.state_dict().items()}, "woerter": woerter, "modell": MODELL,
            "schichten": SCHICHTEN, "dtw_schicht": SCHICHTEN[S2], "m1_schicht": L1, "datum": time.strftime("%Y-%m-%d %H:%M"),
            "aufnahmen": len(orig)}, os.path.join(OUT, "erkenner.pt"))
np.savez_compressed(os.path.join(LOKAL, "vorlagen.npz"),
                    frames=np.array([it["frames"][S2].numpy() for it in orig], dtype=object),
                    woerter=np.array([it["wort"] for it in orig]), dateien=np.array([it["datei"] for it in orig]))
gew = torch.softmax(net.lw.detach().float().cpu(), 0).numpy()
txt += ["", "Gespeichert: erkenner/erkenner.pt (Netz) + ~/my_new_voice_lokal/vorlagen.npz (Vorlagen)",
        "Gewichtung der Encoder-Schichten im Netz: " + ", ".join(f"{l}: {g:.2f}" for l, g in zip(SCHICHTEN, gew)),
        f"Dauer: {(time.time()-T0)/60:.1f} Minuten"]
open(os.path.join(OUT, "ergebnis.txt"), "w", encoding="utf-8").write("\n".join(txt) + "\n")
print("\n" + "\n".join(txt))
if sys.platform == "darwin":
    os.system("osascript -e 'display notification \"Erkenner-Training fertig\" with title \"My New Voice\"'")
