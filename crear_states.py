"""Crea los states de inicio de los niveles (states/<mundo>-<nivel>.state) jugando, sin tocar RAM ni ROM.

Uso (desde la carpeta "Super Mario"):
    .venv\\Scripts\\python.exe crear_states.py
Crea el 1-1 desde los menús y, por orden, el de cada nivel cuyo anterior ya tiene récord en records.json
(repitiendo esa partida hasta el mástil). El entrenamiento también lo hace solo al agotar un nivel.
"""
import os

from mario import config, states
from mario.env import cargar_records


def main():
    if not os.path.exists(config.state_path(config.NIVELES[0])):
        states.crear_primero()
        print(f"✅ 1-1 desde los menús")
    records = cargar_records()
    for nivel in config.NIVELES[:-1]:
        record = records.get(config.nombre_nivel(nivel))
        if not record or not os.path.exists(config.state_path(nivel)):
            break
        siguiente = config.NIVELES[config.NIVELES.index(nivel) + 1]
        if os.path.exists(config.state_path(siguiente)):
            continue
        if states.crear_siguiente(nivel, record["acciones"]) is None:
            raise SystemExit(f"El récord del {config.nombre_nivel(nivel)} ya no llega a la meta")
        print(f"✅ {config.nombre_nivel(siguiente)} repitiendo el récord del {config.nombre_nivel(nivel)}")
    print("States:", ", ".join(sorted(f[:-6] for f in os.listdir(config.STATES_DIR) if f.endswith(".state"))))


if __name__ == "__main__":
    main()
