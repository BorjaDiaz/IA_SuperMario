"""States de inicio de cada nivel, conseguidos jugando: nunca se escribe en la RAM ni en la ROM.

- 1-1: desde el encendido, por los menús (Original 1985 → archivo nuevo → mapa → A).
- Nivel siguiente: se repite la partida que pasa el nivel anterior hasta el mástil y se entra al nuevo desde el
  mapa. Así cada nivel empieza como lo haría en una partida real (con el tamaño de Mario que tenga al llegar).
"""
import os

from pyboy import PyBoy

from mario import config

# Pulsaciones desde el encendido hasta el mapa del 1-1 (medidas con capturas): título → START, menú → A en
# "Original 1985", selección de archivo → A, archivo nuevo → A
MENUS = [(700, "start"), (90, "a"), (90, "a"), (90, "a"), (90, None)]
MAX_FOTOGRAMAS_HASTA_MAPA = 3000


def _guardar(pb, nivel):
    os.makedirs(config.STATES_DIR, exist_ok=True)
    with open(config.state_path(nivel), "wb") as f:
        pb.save_state(f)


def _x(pb):
    return pb.memory[config.ADDR_X_PAGINA] * 256 + pb.memory[config.ADDR_X]


def entrar_desde_el_mapa(pb):
    """Pulsa A en el mapa hasta que el nivel se carga: la X de Mario pasa a 32 (su posición inicial). Vuelve
    True en ese fotograma, que es desde el que Mario responde."""
    for _ in range(MAX_FOTOGRAMAS_HASTA_MAPA // 30):
        x_antes = _x(pb)
        pb.button("a")
        for _ in range(30):
            pb.tick(1, False)
            if _x(pb) == 32 and x_antes != 32:
                return True
            x_antes = _x(pb)
    return False


def crear_primero():
    pb = PyBoy(config.ROM_PATH, window="null", sound_emulated=False)
    try:
        for espera, boton in MENUS:
            pb.tick(espera, False)
            if boton:
                pb.button(boton)
                pb.tick(2, False)
        if not entrar_desde_el_mapa(pb) or pb.memory[config.ADDR_ESTADO] != 0:
            raise RuntimeError("No se llegó al 1-1 desde los menús")
        _guardar(pb, config.NIVELES[0])
    finally:
        pb.stop(save=False)


def crear_siguiente(nivel, botones, frames_por_accion):
    """Repite `botones` (una partida que pasa `nivel`) y guarda el state del nivel que viene después.
    Devuelve ese nivel, o None si la partida no llega a la meta."""
    from mario.env import MarioEnv  # evita la importación circular (env no depende de este módulo)
    siguiente = config.NIVELES[config.NIVELES.index(tuple(nivel)) + 1]
    env = MarioEnv(rank=98, nivel=nivel)
    try:
        info = env.repetir(botones, frames_por_accion)
        if info.get("motivo") != "meta":
            return None
        env._soltar_todo()
        if not entrar_desde_el_mapa(env.pb):
            return None
        _guardar(env.pb, siguiente)
        return siguiente
    finally:
        env.close()


def asegurar(nivel):
    """True si el nivel tiene state o se puede crear ahora con el récord del nivel anterior."""
    if os.path.exists(config.state_path(nivel)):
        return True
    i = config.NIVELES.index(tuple(nivel))
    if i == 0:
        crear_primero()
        return True
    from mario.env import cargar_records
    anterior = config.NIVELES[i - 1]
    record = cargar_records().get(config.nombre_nivel(anterior))
    return bool(record) and os.path.exists(config.state_path(anterior)) \
        and crear_siguiente(anterior, record["botones"], record["frames_por_accion"]) == tuple(nivel)
