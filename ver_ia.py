"""Ver a la IA jugar, o repetir exactamente la partida récord de un nivel (no entrena ni cambia nada).

Uso (desde la carpeta "Super Mario"):
    ..\\Pokemon\\.venv\\Scripts\\python.exe ver_ia.py --record        # repite el récord del nivel
    ..\\Pokemon\\.venv\\Scripts\\python.exe ver_ia.py                 # juega el modelo entrenado
    ..\\Pokemon\\.venv\\Scripts\\python.exe ver_ia.py --velocidad 3   # 3 veces más rápido
"""
import argparse
import os
import sys
import tempfile

os.environ["MARIO_SIN_RECORDS"] = "1"  # mirar no cambia los récords
os.environ["MARIO_RUN_DIR"] = tempfile.mkdtemp(prefix="ver_mario_")

import numpy as np

from mario import config
from mario.env import MarioEnv, cargar_records


def repetir_record(env):
    nivel = "-".join(map(str, config.NIVEL))
    record = cargar_records().get(nivel)
    if not record:
        sys.exit(f"Aún no hay récord en el {nivel}.")
    if record["frames_por_accion"] != config.FRAMES_POR_ACCION:
        sys.exit("El récord se grabó con otro FRAMES_POR_ACCION: no se puede repetir igual.")
    print(f"▶️ Récord del {nivel}: {record['segundos']} s ({record['frames']} fotogramas, clon {record['clon']}, {record['run']})")
    env.reset(timer_div=record["timer_div"])
    for accion in record["acciones"]:
        _, _, fin, _, info = env.step(accion)
    print(f"Resultado al repetirlo: {info.get('motivo')} en {env.frames / 59.73:.2f} s")


def jugar_modelo(env, episodios, determinista):
    from stable_baselines3 import PPO
    model = PPO.load(config.MODELO, device="cpu")
    pila = np.zeros((config.FRAME_STACK, 72, 80), dtype=np.uint8)
    for _ in range(episodios):
        obs, _ = env.reset()
        pila[:] = obs[0]
        while True:
            pila = np.roll(pila, -1, axis=0)
            pila[-1] = obs[0]
            accion, _ = model.predict(pila[None], deterministic=determinista)
            obs, _, fin, _, info = env.step(int(accion[0]))
            if fin:
                print(f"🏁 {info['motivo']} · avance {info['progreso']} px · {info['frames'] / 59.73:.2f} s")
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", action="store_true", help="repetir la partida récord del nivel")
    parser.add_argument("--velocidad", type=int, default=1, help="1 = tiempo real, 0 = sin límite")
    parser.add_argument("--episodios", type=int, default=5)
    parser.add_argument("--determinista", action="store_true", help="siempre la acción más probable")
    args = parser.parse_args()
    env = MarioEnv(rank=100, ventana=True, velocidad=args.velocidad)
    try:
        repetir_record(env) if args.record else jugar_modelo(env, args.episodios, args.determinista)
    finally:
        env.close()


if __name__ == "__main__":
    main()
