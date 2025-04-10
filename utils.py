# utils.py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, Markdown
import random

def fade(t):
    """Fadekurve nach Ken Perlin"""
    return t**3 * (t * (t * 6 - 15) + 10)

def lerp(a, b, t):
    """Lineare Interpolation zwischen a und b"""
    return a + t * (b - a)

def gradient(h, x, y):
    """Bestimmt den Gradientenvektor und berechnet das Skalarprodukt."""
    vectors = np.array([[1,1], [-1,1], [1,-1], [-1,-1], [1,0], [-1,0], [0,-1], [0,1], [0,0]])
    g = vectors[h % len(vectors)]
    return g[:, :, 0] * x + g[:, :, 1] * y

def perlin_noise(width, height, scale=10, seed=0):
    """Generiert ein Perlin-Noise-Bild."""
    # Gitterpunkte
    x = np.linspace(0, scale, width, endpoint=False)
    y = np.linspace(0, scale, height, endpoint=False)
    X, Y = np.meshgrid(x, y)

    x0 = np.floor(X).astype(int)
    x1 = x0 + 1
    y0 = np.floor(Y).astype(int)
    y1 = y0 + 1

    # Zufällige Gradienten an den Gitterpunkten
    np.random.seed(seed)
    gradient_grid = np.random.randint(0, 9, (scale+1, scale+1))

    # Skalarprodukte
    sx, sy = X - x0, Y - y0
    n0 = gradient(gradient_grid[x0 % scale, y0 % scale], sx, sy)
    n1 = gradient(gradient_grid[x1 % scale, y0 % scale], sx - 1, sy)
    ix0 = lerp(n0, n1, fade(sx))

    n0 = gradient(gradient_grid[x0 % scale, y1 % scale], sx, sy - 1)
    n1 = gradient(gradient_grid[x1 % scale, y1 % scale], sx - 1, sy - 1)
    ix1 = lerp(n0, n1, fade(sx))

    return lerp(ix0, ix1, fade(sy))

def find_idx(mat, besucht):
    minval = np.inf
    idx = [0,0]
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if minval>=mat[i,j] and not besucht[i,j]:
                minval = mat[i,j]
                idx[0], idx[1] = i,j
    return np.array(idx)

def dijkstra(karte, start, ziel, heuristik=None):
    return a_stern(karte, start, ziel, heuristik=None)

def a_stern(karte, start, ziel, heuristik=None):
    map = karte.copy()
    if heuristik is None:
        heuristik = np.zeros_like(map)

    # Nachbarschaft, hier Achternachbarschaft
    dxdy = np.array([[0,1], [1,0], [-1,0], [0,-1], [1,1], [1,-1], [-1,1], [-1,-1]])

    # Initialisiere die Kostenmatrix
    costmat = np.inf*np.ones_like(map)
    costmat[start[0], start[1]] = 0

    # Initialisiere die Besuchtmatrix
    besucht = np.zeros_like(map).astype(bool)

    # Initialisiere die Vorgängermatrix:
    vorgaengermat = -np.ones_like(map)

    # A*-Algorithmus
    while not besucht[ziel[0], ziel[1]]:
        idx = find_idx(costmat+heuristik, besucht)
        besucht[idx[0], idx[1]] = True
        for n in range(len(dxdy)):
            nn = idx+dxdy[n]
            if any(nn<0) or any(nn>(np.array(map.shape)-1)) or besucht[nn[0], nn[1]]:
                continue
            kosten = -(map[idx[0], idx[1]]-map[nn[0], nn[1]])
            #if kosten < 0: kosten=0
            if costmat[nn[0], nn[1]] > costmat[idx[0], idx[1]]+kosten:
                costmat[nn[0], nn[1]] = costmat[idx[0], idx[1]]+kosten
                vorgaengermat[nn[0], nn[1]] = n

    # Backtracking
    pfad = []
    hier = np.array(ziel.copy())
    pfad.append(hier.copy())
    while vorgaengermat[hier[0], hier[1]]>=0:
        hier -= dxdy[int(vorgaengermat[hier[0], hier[1]])]
        pfad.append(hier.copy())
    return np.array(pfad)[::-1], costmat, besucht

def a_stern_img(karte, start, ziel, heuristik=None):
    map = karte.copy()
    if heuristik is None:
        heuristik = np.zeros_like(map)

    # Nachbarschaft, hier Achternachbarschaft
    dxdy = np.array([[0,1], [1,0], [-1,0], [0,-1], [1,1], [1,-1], [-1,1], [-1,-1]])

    # Initialisiere die Kostenmatrix
    costmat = np.inf*np.ones_like(map)
    costmat[start[0], start[1]] = 0

    # Initialisiere die Besuchtmatrix
    besucht = np.zeros_like(map).astype(bool)

    # Initialisiere die Vorgängermatrix:
    vorgaengermat = -np.ones_like(map)

    # A*-Algorithmus
    while not besucht[ziel[0], ziel[1]]:
        idx = find_idx(costmat+heuristik, besucht)
        besucht[idx[0], idx[1]] = True
        for n in range(len(dxdy)):
            nn = idx+dxdy[n]
            if any(nn<0) or any(nn>(np.array(map.shape)-1)) or besucht[nn[0], nn[1]]:
                continue
            kosten = map[nn[0], nn[1]]
            if costmat[nn[0], nn[1]] > costmat[idx[0], idx[1]]+kosten:
                costmat[nn[0], nn[1]] = costmat[idx[0], idx[1]]+kosten
                vorgaengermat[nn[0], nn[1]] = n

    # Backtracking
    pfad = []
    hier = np.array(ziel.copy())
    pfad.append(hier.copy())
    while vorgaengermat[hier[0], hier[1]]>=0:
        hier -= dxdy[int(vorgaengermat[hier[0], hier[1]])]
        pfad.append(hier.copy())
    return np.array(pfad)[::-1], costmat, besucht


def draw_path(path, c="black"):
    for p1, p2 in zip(path[1:], path[:-1]):
        plt.plot([p1[1], p2[1]], [p1[0], p2[0]], c=c)



def load_img(path):
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32)
    img -= np.min(img)
    img /= np.max(img)

    img = img@np.array([0.2126, 0.7152, 0.0722])
    return img

def kantenbild(img):
    #print(img.shape)
    padded = np.zeros((img.shape[0]+2, img.shape[1]+2))
    padded[1:-1, 1:-1] = img.copy()
    dx = 1/3*(padded[:,2:]+padded[:,1:-1]+padded[:,:-2])
    dx = dx[2:]-dx[:-2]
    dy = 1/3*(padded[2:]+padded[1:-1]+padded[:-2])
    dy = dy[:,2:]-dy[:,:-2]
    #print(dx.shape, dy.shape)
    out = np.sqrt(dx**2+dy**2)
    out = 1-(out/(out.max()+1e-8))
    out[0] = 1
    out[-1] = 1
    out[:,0] = 1
    out[:,-1] = 1
    return out

class Knoten():
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos
        self.nachbarn = []
        self.vorg = None
        self.kosten = float("inf")

    def copy(self):
        out = Knoten(self.name, self.pos)

        out.nachbarn = self.nachbarn.copy()
        out.vorg = self.vorg
        out.kosten = self.kosten
        return out

    def set_nachbarn(self, nachbar):
        if isinstance(nachbar, list):
            self.nachbarn.extend(nachbar)
        else:
            self.nachbarn.append(nachbar)

    def __repr__(self):
        return f"Knoten {self.name} hat Position {self.pos} und Verbindungen {self.nachbarn}."

    def display(self):
        text = f"Knoten **{self.name}** hat Position **{self.pos}**. Verbindungen:\n"
        table = "| Knoten | Kosten |\n|---|---|\n"
        for eintrag in self.nachbarn:
            table += f"|{eintrag[0]}|{eintrag[1]}|\n"
        display(Markdown(text))
        display(Markdown(table))

class Knotenliste():
    def __init__(self):
        self.liste = []

    def append(self, item):
        self.liste.append(item)

    def __getitem__(self, idx):
        return self.liste[idx]

    def get_knoten(self, name):
        for idx in range(len(self.liste)):
            if self.liste[idx].name == name:
                return self.liste[idx]

    def display_knoten(self):
        aktuell = self.liste[0]

        while True:
            aktuell.display()
            naechster = input("Gib einen Knotennamen ein. ")
            for eintrag in self.liste:
                if naechster.upper()==eintrag.name:
                    aktuell = self.get_knoten(naechster)
                    break
            else:
                print(f"{naechster} ist leider nicht enthalten.")

    def verbinde(self, verb):
        k1 = verb[0]
        k2 = verb[1]
        val = float(verb[2])
        self._verbinde(k1, k2, val)

    def _verbinde(self, k1, k2, val):
        for k in self.liste:
            if k.name==k1:
                k.set_nachbarn((k2, val))
            elif k.name==k2:
                k.set_nachbarn((k1,val))

    def draw(self):
        plt.figure()
        for k in self.liste:
            plt.scatter(k.pos[0], k.pos[1], zorder=3, c="tab:blue")
            plt.text(k.pos[0], k.pos[1], k.name, zorder=4)

        for k1 in self.liste:
            for k2name, v in k1.nachbarn:
                plt.plot([k1.pos[0], self.get_knoten(k2name).pos[0]], [k1.pos[1], self.get_knoten(k2name).pos[1]], "k")
                plt.text((k1.pos[0]+self.get_knoten(k2name).pos[0])/2, (k1.pos[1]+self.get_knoten(k2name).pos[1])/2, str(v), zorder=3)

        plt.axis("scaled")
        plt.show()

class Pfad(Knotenliste):
    def __init__(self):
        super().__init__()
        self.kostenliste = []
        self.pfad = []

    def _verbinde(self, k1, k2, kosten):
        ziel = self.get_knoten(k2)
        ziel.nachbarn = [(k1, kosten)]

    def menschenwalk(self, graph, start="S", end="E"):
        display(Markdown(f"### Wir suchen den Pfad von Knoten {start} zu Knoten {end}."))

        aktuell = graph.get_knoten(start)

        while aktuell.name!=end:
            display(Markdown(f"Wir sind aktuell bei Knoten **{aktuell.name}**. Momentan sind unsere Kosten **{sum(self.kostenliste)}**. Er ist verbunden mit den folgenden Knoten:"))
            aktuell.display()
            weiter = True
            while weiter:
                weiter = False; naechster = input("Wo wollen wir hingehen? ").upper()
                if naechster=="": break
                for eintrag in aktuell.nachbarn:
                    if naechster.upper()==eintrag[0]:
                        self.kostenliste.append(eintrag[1]); self.pfad.append(naechster); break
                else:
                    print(f"{naechster} ist leider nicht enthalten."); weiter = True
            aktuell = graph.get_knoten(naechster)
        display(Markdown(f"Sehr gut, wir sind am Zielknoten {end} angekommen! Unser Pfad hat Kosten von {sum(self.kostenliste)}."))

    def randomwalk(self, graph, start="S", end="E"):
        knoten = graph.get_knoten(start); self.pfad.append(knoten.name)
        while knoten.name!=end:
            nachbarliste = knoten.nachbarn
            naechster = random.choice(nachbarliste)
            knoten = graph.get_knoten(naechster[0])
            self.pfad.append(knoten.name)
            self.kostenliste.append(naechster[1])

    def prune(self):
        for i in range(len(self.pfad), 0, -1):
            if self.pfad[i-1] in self.pfad[:i-1]:
                neueliste = self.pfad[:self.pfad.index(self.pfad[i-1])]+self.pfad[i-1:]
                neuekosten = self.kostenliste[:self.pfad.index(self.pfad[i-1])]+self.kostenliste[i-1:]
                self.pfad, self.kostenliste = neueliste, neuekosten
                break
        else:
            return
        self.prune()

    def __repr__(self):
        return f"Der Pfad {self.pfad} hatte die Kosten {sum(self.kostenliste)}."

def dijkstra_graph(adj, start=0, ziel=5):
    besucht = []
    vor = []
    keys = "SABCDEFGHIJKL"
    nodelist = list(range(len(adj)))
    nachbarn = [[i, float("inf"), -1] for i in nodelist]
    nachbarn[start][1] = 0
    nachbarn[start][2] = 0

    while nodelist[ziel] not in besucht:
        nachbarn = sorted(nachbarn, key=lambda x: x[1])
        node = nachbarn[0]
        for j, (i, val, _) in enumerate(nachbarn):
            if nv:=adj[node[0], i]:
                new_val = node[1]+nv
                if new_val<nachbarn[j][1]:
                    #print(f"Update Schritt {len(besucht)}: [{keys[nachbarn[j][0]]},{nachbarn[j][1]}]->[{keys[nachbarn[j][0]]},{int(new_val)}]")
                    nachbarn[j][1] = new_val
                    nachbarn[j][-1] = node[0]
        last = nachbarn.pop(0)
        besucht.append(node[0])
        vor.append(node[-1])

    vor = list(zip([keys[i] for i in besucht], [keys[i] for i in vor]))
    weg = [vor[-1][0]]
    while weg[-1] != keys[start]:
        for i in range(len(vor)):
            if vor[i][0] == weg[-1]:
                weg.append(vor[i][1])
                break
    
    print("Kürzester Weg:", weg[::-1])
    print("Kosten:", last[1])
