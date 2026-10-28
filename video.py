"""Graba en MP4 la partida récord de un nivel, repitiendo sus acciones (no entrena ni cambia nada).

Uso (desde la carpeta "SuperMario"):
    .venv\\Scripts\\python.exe video.py              # récord del 1-1 -> videos/1-1_41.92s.mp4
    .venv\\Scripts\\python.exe video.py --nivel 1-2
El emulador es determinista: el vídeo no hace falta grabarlo mientras se entrena, se saca después del récord.
train.py lo lanza solo, en segundo plano, cada vez que hay un récord nuevo.
"""
import argparse
import os
import sys
import tempfile

os.environ["MARIO_SIN_RECORDS"] = "1"
os.environ["MARIO_RUN_DIR"] = tempfile.mkdtemp(prefix="video_mario_")

import imageio.v2 as imageio
import numpy as np

from mario import config
from mario.env import MarioEnv, cargar_records

VIDEOS_DIR = os.path.join(config.BASE_DIR, "videos")
FPS = 60
ESCALA = 3                    # 160x144 -> 480x432
FOTOGRAMAS_TRAS_LA_META = 90  # 1,5 s más para ver la bajada por el mástil


def grabar(nivel):
    nombre = config.nombre_nivel(nivel)
    record = cargar_records().get(nombre)
    if not record:
        sys.exit(f"Aún no hay récord en el {nombre}.")
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    ruta = os.path.join(VIDEOS_DIR, f"{nombre}_{record['segundos']:.2f}s.mp4")
    tmp = ruta + ".tmp.mp4"
    env = MarioEnv(rank=120, nivel=nivel)
    try:
        env.reset()
        # Igual que env.step (mismos botones, mismos fotogramas), pero dibujando y guardando cada fotograma
        fotogramas, meta = 0, None
        with imageio.get_writer(tmp, fps=FPS, codec="libx264", quality=8, macro_block_size=1) as video:
            def fotograma():
                imagen = np.asarray(env.pb.screen.image.convert("RGB"))
                video.append_data(imagen.repeat(ESCALA, axis=0).repeat(ESCALA, axis=1))
            for botones in record["botones"]:
                env._pulsar(botones)
                for _ in range(record["frames_por_accion"]):
                    env.pb.tick(1, True)
                    fotogramas += 1
                    fotograma()
                    if meta is None and env.pb.memory[config.ADDR_ESTADO] == config.ESTADO_META:
                        meta = fotogramas
            env._soltar_todo()
            for _ in range(FOTOGRAMAS_TRAS_LA_META):
                env.pb.tick(1, True)
                fotograma()
    finally:
        env.close()
    if meta != record["frames"]:
        os.remove(tmp)
        sys.exit(f"Al repetirlo la meta llega en el fotograma {meta}, no en el {record['frames']}: vídeo descartado")
    os.replace(tmp, ruta)
    print(f"🎬 {ruta}")
    return ruta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nivel", default="1-1", help="mundo-nivel, p. ej. 1-2")
    args = parser.parse_args()
    grabar(tuple(map(int, args.nivel.split("-"))))


if __name__ == "__main__":
    main()
