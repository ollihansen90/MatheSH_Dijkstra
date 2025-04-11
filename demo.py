import matplotlib.pyplot as plt
from utils import load_img, kantenbild, a_stern_img, draw_path
import numpy as np

img = "marienkaefer.jpg"
#img = "eule.jpg"
#img = "pantoffeltierchen.jpg"
img = "holstentor.jpg"

img = load_img("bilder/"+img)
every = np.min(img.shape[:2])//150
img = img[::every,::every]
kanten = kantenbild(img)
plt.figure()
plt.imshow(img)
eckpunkte = np.array(plt.ginput(n=100, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)).round().astype(int)[:,::-1]
plt.show()
h, w, _ = img.shape

pfad = []
c = 0
for start, ziel in zip(eckpunkte[:-1], eckpunkte[1:]):
    distmat = np.linalg.norm(np.stack(np.meshgrid(np.arange(0,h), np.arange(0,w))).transpose(2,1,0)-ziel, axis=-1)
    heuristik = np.log(distmat+1e-8)
    c += 1
    print(f"Eckpunkt {c}/{len(eckpunkte)-1}, Distanz: {np.linalg.norm(start-ziel):.2f}")
    p,*_ = a_stern_img(kanten, start, ziel, heuristik)
    pfad.append(p)

#print(pfad)
plt.figure()
plt.imshow(img)
for p in pfad:
    draw_path(p, "red")
plt.show()
