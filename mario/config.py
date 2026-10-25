import os

# Carpeta del proyecto (mario/config.py -> dos niveles arriba)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROM_PATH = os.path.join(BASE_DIR, "roms", "Super Mario Land (World).gb")
IA_DIR = os.path.join(BASE_DIR, "ia")
MODELO = os.path.join(IA_DIR, "modelo.zip")
CHECKPOINTS_DIR = os.path.join(IA_DIR, "checkpoints")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
RECORDS_PATH = os.path.join(BASE_DIR, "records.json")
STATES_DIR = os.path.join(BASE_DIR, "states")
# Qué niveles se dan por agotados (sin bajar más el tiempo); train.py sigue por el primero pendiente
NIVELES_PATH = os.path.join(BASE_DIR, "niveles.json")
# Carpeta de la ejecución en curso (la fija train.py para los clones, que van en procesos aparte)
RUN_DIR = os.environ.get("MARIO_RUN_DIR", os.path.join(BASE_DIR, "runs", "pruebas"))
# MARIO_SIN_RECORDS=1: no escribe records.json (pruebas, ver_ia.py). Solo el entrenamiento debe cambiarlo
GUARDAR_RECORDS = os.environ.get("MARIO_SIN_RECORDS") != "1"
STOP_FILE = os.path.join(BASE_DIR, "PARAR")

# ==========================================
# JUEGO
# ==========================================
# Cada nivel es un speedrun aparte que empieza en su state (crear_states.py). Se entrenan en este orden
NIVELES = [(m, n) for m in range(1, 5) for n in range(1, 4)]


def nombre_nivel(nivel):
    return "-".join(map(str, nivel))


def state_path(nivel):
    return os.path.join(STATES_DIR, f"{nombre_nivel(nivel)}.state")


# Un nivel se da por agotado (imposible bajar más el tiempo con lo aprendido) cuando llega a la meta en al
# menos META_MINIMA de los últimos 100 episodios y lleva PASOS_SIN_MEJORA pasos sin récord. Entonces se pasa al
# siguiente con el mismo modelo
META_MINIMA = 0.8
PASOS_SIN_MEJORA = 3_000_000
# Acciones = combinaciones de botones mantenidas FRAMES_POR_ACCION fotogramas. Pocos fotogramas por
# acción = control fino: saltos precisos y trucos de movimiento que un control grueso no permitiría.
# El salto es más alto cuanto más se mantiene A; para volver a saltar hay que soltarlo (otra acción)
ACCIONES = [
    (),                       # nada
    ("right",),
    ("right", "a"),           # saltar hacia delante
    ("right", "b"),           # correr
    ("right", "a", "b"),      # salto largo corriendo
    ("left",),
    ("left", "a"),
    ("a",),
    ("down",),                # agacharse / entrar en tuberías
]
FRAMES_POR_ACCION = 4
# Estado del juego (0xFFB3), medido en el emulador: 0 jugando (13 en submarino y avión), 7 toca la meta (luego
# 5 recuento, 6, 8 nivel siguiente), 3 golpe (luego 4 animación, 1 caída), 1 directamente al caer a un foso o
# quedar aplastado, 2 reaparecer, 57-58 GAME OVER y 14-15-17 vuelta al título (con el mundo puesto a 1-1)
ADDR_ESTADO = 0xFFB3
ESTADO_META = 7
ESTADOS_MUERTE = (1, 3, 4)
ESTADOS_GAME_OVER = (14, 15, 17, 57, 58)

# ==========================================
# RECOMPENSAS (speedrun: llegar a la meta en el menor número de fotogramas)
# ==========================================
# Solo se indica QUÉ conseguir, nunca POR DÓNDE: así cualquier atajo o truco que lleve antes a la meta
# se premia solo
REWARD_PIXEL = 0.01           # por cada píxel más lejos que nunca en el episodio (1-1 ≈ 2.600 px ≈ +26)
MAX_PIXELES_POR_PASO = 40     # un salto mayor en un paso es una lectura rara (cambio de zona): no se premia
REWARD_PASO = -0.01           # cada paso cuesta: cuanto antes llegue, mejor
REWARD_META = 50.0            # nivel completado...
REWARD_META_RAPIDEZ = 50.0    # ...más hasta esto según lo rápido (lineal hasta FRAMES_REFERENCIA)
FRAMES_REFERENCIA = 6000      # ~100 s de juego: tardar esto o más no da bonus de rapidez
PENALTY_MUERTE = -10.0
PENALTY_ATASCO = -1.0
MAX_PASOS_SIN_AVANCE = 300    # ~20 s de juego sin pasar de su marca: fin del episodio

# ==========================================
# ENTRENAMIENTO (PPO)
# ==========================================
NUM_CLONES = 12
FRAME_STACK = 4               # ver la velocidad y la dirección de Mario y de los enemigos
TOTAL_TIMESTEPS = 20_000_000
PPO_LEARNING_RATE = 0.00025
PPO_N_STEPS = 1024
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 4
PPO_GAMMA = 0.99
PPO_ENT_COEF = 0.01
PPO_TARGET_KL = 0.03
CHECKPOINT_FREQ = 500_000     # pasos totales entre copias de seguridad
MAX_MINUTOS = 180             # de entrenamiento activo (sin contar suspensiones del PC)
