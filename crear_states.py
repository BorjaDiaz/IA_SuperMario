"""Crea states/<mundo>-<nivel>.state: el primer fotograma jugable de cada nivel.

Uso (desde la carpeta "Super Mario"), una sola vez:
    ..\\Pokemon\\.venv\\Scripts\\python.exe crear_states.py
Para llegar a cada nivel, PyBoy parchea en memoria el menú de inicio de ESTE proceso. El parche no se guarda en
el archivo: el entrenamiento carga los states con la ROM original y sin escribir nada en la RAM.
"""
import os

from pyboy import PyBoy

from mario import config


def main():
    os.makedirs(config.STATES_DIR, exist_ok=True)
    for mundo, nivel in config.NIVELES:
        pb = PyBoy(config.ROM_PATH, window="null", sound_emulated=False)
        pb.game_wrapper.start_game(timer_div=0, world_level=(mundo, nivel))
        if pb.game_wrapper.world != (mundo, nivel):
            raise SystemExit(f"Se esperaba el {mundo}-{nivel} y el juego está en {pb.game_wrapper.world}")
        ruta = config.state_path((mundo, nivel))
        with open(ruta, "wb") as f:
            pb.save_state(f)
        pb.stop(save=False)
        print(f"✅ {mundo}-{nivel} → {os.path.relpath(ruta, config.BASE_DIR)}")


if __name__ == "__main__":
    main()
