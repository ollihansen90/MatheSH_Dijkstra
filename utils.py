# utils.py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
            kosten = np.abs(map[idx[0], idx[1]]-map[nn[0], nn[1]])
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
    print(img.shape)
    padded = np.zeros((img.shape[0]+2, img.shape[1]+2))
    padded[1:-1, 1:-1] = img.copy()
    dx = 1/3*(padded[:,2:]+padded[:,1:-1]+padded[:,:-2])
    dx = dx[2:]-dx[:-2]
    dy = 1/3*(padded[2:]+padded[1:-1]+padded[:-2])
    dy = dy[:,2:]-dy[:,:-2]
    print(dx.shape, dy.shape)
    out = np.sqrt(dx**2+dy**2)
    out = 1-(out/(out.max()+1e-8))
    out[0] = 1
    out[-1] = 1
    out[:,0] = 1
    out[:,-1] = 1
    return out