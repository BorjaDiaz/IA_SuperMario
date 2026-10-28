"""Ver a varios clones jugar a la vez, en cuadrícula, con el último checkpoint (no entrena ni cambia nada).

Uso (desde la carpeta "SuperMario"):
    .venv\\Scripts\\python.exe ver_clones.py                  # 6 clones con el último checkpoint
    .venv\\Scripts\\python.exe ver_clones.py --clones 9 --velocidad 2
Cada clon elige sus acciones con el mismo azar que en el entrenamiento, así que cada uno juega distinto.
Se puede usar mientras se entrena: solo lee ia/checkpoints (cada CHECKPOINT_FREQ pasos) o ia/modelo.zip.
"""
import argparse
import glob
import math
import os
import tempfile
import time
import tkinter as tk

os.environ["MARIO_SIN_RECORDS"] = "1"  # mirar no cambia los récords
os.environ["MARIO_RUN_DIR"] = tempfile.mkdtemp(prefix="ver_mario_")

import numpy as np
from PIL import ImageTk

from mario import config
from mario.env import MarioEnv

FPS = 59.73


def ultimo_modelo():
    checkpoints = glob.glob(os.path.join(config.CHECKPOINTS_DIR, "*.zip"))
    candidatos = checkpoints + ([config.MODELO] if os.path.exists(config.MODELO) else [])
    if not candidatos:
        raise SystemExit("Aún no hay modelo ni checkpoints: lanza antes train.py.")
    return max(candidatos, key=os.path.getmtime)


class Visor:
    def __init__(self, n, velocidad, escala, determinista):
        from stable_baselines3 import PPO
        self.ruta = ultimo_modelo()
        self.model = PPO.load(self.ruta, device="cpu")
        self.envs = [MarioEnv(rank=110 + i) for i in range(n)]
        self.pilas = np.zeros((n, config.FRAME_STACK, 72, 80), dtype=np.uint8)
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            self.pilas[i] = obs[0]
        self.determinista = determinista
        self.escala = escala
        self.espera = config.FRAMES_POR_ACCION / FPS / velocidad
        self.contador = {"meta": 0, "muerte": 0, "atasco": 0}
        self.pausa = [0] * n  # pasos que se queda congelado el final de cada partida

        self.raiz = tk.Tk()
        self.raiz.configure(bg="black")
        columnas = math.ceil(math.sqrt(n))
        self.pantallas, self.textos = [], []
        for i in range(n):
            marco = tk.Frame(self.raiz, bg="black")
            marco.grid(row=i // columnas, column=i % columnas, padx=3, pady=3)
            pantalla = tk.Label(marco, bg="black")
            pantalla.pack()
            texto = tk.Label(marco, fg="white", bg="black", font=("Consolas", 10))
            texto.pack()
            self.pantallas.append(pantalla)
            self.textos.append(texto)
        self.raiz.after(0, self.paso)

    def titulo(self):
        c = self.contador
        total = sum(c.values()) or 1
        self.raiz.title(f"{os.path.basename(self.ruta)} · metas {c['meta']}/{total} "
                        f"({100 * c['meta'] / total:.0f} %) · muertes {c['muerte']} · atascos {c['atasco']}")

    def paso(self):
        inicio = time.time()
        acciones, _ = self.model.predict(self.pilas, deterministic=self.determinista)
        for i, env in enumerate(self.envs):
            if self.pausa[i]:
                self.pausa[i] -= 1
                if not self.pausa[i]:
                    obs, _ = env.reset()
                    self.pilas[i] = obs[0]
                continue
            obs, _, fin, _, info = env.step(int(acciones[i]))
            self.pilas[i] = np.roll(self.pilas[i], -1, axis=0)
            self.pilas[i][-1] = obs[0]
            if fin:
                self.contador[info["motivo"]] += 1
                marca = {"meta": "🏁", "muerte": "💀", "atasco": "⏸"}[info["motivo"]]
                self.textos[i].config(text=f"clon {i} · {marca} {info['motivo']} · {info['frames'] / FPS:.2f} s")
                self.pausa[i] = 25  # ~1,5 s para verlo antes de empezar otra partida
                self.titulo()
            else:
                self.textos[i].config(text=f"clon {i} · x {env.x():4d} · {env.frames / FPS:5.2f} s")
            imagen = env.pb.screen.image
            imagen = imagen.resize((160 * self.escala, 144 * self.escala))
            self.pantallas[i].imagen = ImageTk.PhotoImage(imagen)  # guardar la referencia o tkinter la borra
            self.pantallas[i].config(image=self.pantallas[i].imagen)
        restante = self.espera - (time.time() - inicio)
        self.raiz.after(max(1, int(restante * 1000)), self.paso)

    def cerrar(self):
        for env in self.envs:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clones", type=int, default=6)
    parser.add_argument("--velocidad", type=float, default=1, help="1 = tiempo real")
    parser.add_argument("--escala", type=int, default=2, help="tamaño de cada pantalla (1 = 160x144)")
    parser.add_argument("--determinista", action="store_true", help="siempre la acción más probable (todos iguales)")
    args = parser.parse_args()
    visor = Visor(args.clones, args.velocidad, args.escala, args.determinista)
    print(f"▶️ {args.clones} clones con {visor.ruta}")
    visor.titulo()
    try:
        visor.raiz.mainloop()
    finally:
        visor.cerrar()


if __name__ == "__main__":
    main()
