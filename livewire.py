import matplotlib.pyplot as plt
from utils import load_img, kantenbild, a_stern_img, draw_path
import numpy as np

every = 4
img = load_img("bilder/eule.jpg")[::every,::every]
kanten = kantenbild(img)
plt.figure()
plt.imshow(img)
eckpunkte = np.array(plt.ginput(n=100, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)).round().astype(int)[:,::-1]
plt.show()
h, w = img.shape


pfad = []
c = 0
for start, ziel in zip(eckpunkte[:-1], eckpunkte[1:]):
    distmat = np.linalg.norm(np.stack(np.meshgrid(np.arange(0,h), np.arange(0,w))).transpose(2,1,0)-ziel, axis=-1)
    #distmat /= distmat[start[0], start[1]]
    heuristik = np.log(distmat+1e-8)
    c += 1
    print(f"{c}/{len(eckpunkte)-1}")
    p,*_ = a_stern_img(kanten, start, ziel, heuristik)
    pfad.append(p)

#print(pfad)
plt.figure()
plt.imshow(img)
for p in pfad:
    draw_path(p, "red")
plt.show()
