import os

# Carpeta del proyecto (mario/config.py -> dos niveles arriba)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROM_PATH = os.path.join(BASE_DIR, "roms", "Super Mario Bros. Deluxe (USA, Europe) (Rev A).gbc")
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
# Super Mario Bros. Deluxe (Game Boy Color), modo "Original 1985": los 32 niveles del Super Mario Bros. de NES.
# Cada nivel es un speedrun aparte que empieza en su state. El del 1-1 sale de los menús; el de cada nivel
# siguiente, de repetir la partida que pasa el anterior (mario/states.py): nunca se escribe en la RAM ni en la ROM
NIVELES = [(m, n) for m in range(1, 9) for n in range(1, 5)]


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
# A salta (más alto cuanto más se mantiene; para volver a saltar hay que soltarlo) y B corre y lanza fuego.
# Para buscar trucos y glitches: todas las combinaciones con sentido, aunque una persona casi nunca las use
ACCIONES = [
    (),                       # nada
    ("right",),
    ("right", "a"),           # saltar hacia delante
    ("right", "b"),           # correr
    ("right", "a", "b"),      # salto largo corriendo
    ("left",),
    ("left", "a"),
    ("left", "b"),            # correr hacia atrás
    ("left", "a", "b"),
    ("a",),
    ("b",),                   # bola de fuego sin moverse
    ("a", "b"),               # saltar sin dirección manteniendo la carrera
    ("down",),                # agacharse / entrar en tuberías
    ("right", "down"),        # agacharse en carrera (deslizar)
    ("up",),                  # trepar por las enredaderas
]
# Con 2 fotogramas (para trucos de precisión) un modelo nuevo no pasó de las tuberías del 1-1 (x=675) en 78 min:
# saltarlas pide ~10 decisiones iguales seguidas y a ~1.700 fotogramas/s de juego, 3 veces menos que con 4. Con 4
# las pasaba en ~10 min. Los récords guardan sus propios fotogramas por acción, así que todos se siguen repitiendo
FRAMES_POR_ACCION = 4
# RAM medida en el emulador (capturas en cada cambio). X de Mario en el nivel = página * 256 + x
ADDR_X_PAGINA = 0xC1CB
ADDR_X = 0xC1CA
ADDR_VIDAS = 0xC17F
# Estado de Mario (0xC1C1): 0 en el suelo, 1 en el aire, 19 rebota al pisar a un enemigo, 16 entra andando
# (inicio del 1-2), 5 se agarra al mástil (meta), 3 golpe mortal, 17 cae a un foso. La muerte y la meta se ven
# en su fotograma exacto; las vidas bajan cientos de fotogramas después, ya en el mapa
ADDR_ESTADO = 0xC1C1
ESTADO_META = 5
ESTADOS_MUERTE = (3, 17)

# ==========================================
# RECOMPENSAS (speedrun: llegar a la meta en el menor número de fotogramas)
# ==========================================
# Solo se indica QUÉ conseguir, nunca POR DÓNDE: así cualquier atajo o truco que lleve antes a la meta
# se premia solo
REWARD_PIXEL = 0.01           # por cada píxel más lejos que nunca en el episodio (mástil del 1-1 ≈ x 3.130 ≈ +31)
MAX_PIXELES_POR_PASO = 40     # un salto mayor en un paso es una lectura rara (cambio de zona): no se premia
REWARD_PASO = -0.01           # cada paso cuesta: cuanto antes llegue, mejor
REWARD_META = 50.0            # nivel completado...
# Cuando ya llega casi siempre (97 % en el 1-1), lo que queda es recortar tiempo: con +50 repartidos en 6.000
# fotogramas cada segundo ahorrado valía ~0,65 puntos (menos que el ruido entre partidas) y probar algo raro no
# compensaba. Con +200 en 3.000 fotogramas vale ~4,1 puntos por segundo
REWARD_META_RAPIDEZ = 200.0   # ...más hasta esto según lo rápido (lineal hasta FRAMES_REFERENCIA)
FRAMES_REFERENCIA = 3000      # ~50 s de juego: tardar esto o más no da bonus de rapidez
PENALTY_MUERTE = -10.0
# Atascarse tiene que salir más caro que morir AUN DESCONTADO (llega al final, gamma^pasos): con -1 y 300 pasos la
# IA se quedaba quieta ante las tuberías del 1-1, y con -10 seguía parada al borde del primer foso (84 % de atascos;
# -10 a 75 pasos pesa ~-6,9 frente a -10 de caer). -15 pesa ~-10,3: saltar ya no parece peor que esperar
PENALTY_ATASCO = -15.0
# El castigo del atasco llega al final y PPO lo descuenta (gamma^pasos): con 150 pasos y gamma 0,99 pesaba ~-2 desde
# el borde del primer foso frente a -10 de caer, y la IA se quedaba parada (78 % de atascos en x≈1.000-1.100).
# Con 75 pasos de 4 fotogramas y gamma 0,995 pesaba ~-6,9 (con -15, ~-10,3). No menos de ~5 s: hay niveles donde
# se espera a plataformas móviles. Se mide en fotogramas para que no cambie con FRAMES_POR_ACCION
MAX_FRAMES_SIN_AVANCE = 300   # ~5 s de juego sin pasar de su marca: fin del episodio
MAX_PASOS_SIN_AVANCE = MAX_FRAMES_SIN_AVANCE // FRAMES_POR_ACCION

# Práctica: cada clon guarda un state en un fotograma al azar de cada partida (si llega a la meta, se queda) y, en
# PRACTICA_PROB de los episodios, empieza desde uno de ellos en vez de desde el principio. Así repite más los tramos
# donde se ganan décimas. No le dice por dónde ir (son puntos de sus propias partidas); el tiempo cuenta desde el
# inicio real del nivel y estos episodios nunca dan récords
PRACTICA_PROB = 0.25
PRACTICA_CADA_FRAMES = 120    # al sembrar con el récord: un punto cada ~2 s de juego
# Guardar un state en PyBoy cuesta ~35 ms: con un punto cada 2 s en todas las partidas un clon iba a la mitad de
# velocidad (434 frente a 915 pasos/s). Con uno por partida, en un fotograma al azar, la lista se llena igual
PRACTICA_MAX = 60             # puntos guardados por clon (los más recientes: de sus partidas más rápidas)

# ==========================================
# ENTRENAMIENTO (PPO)
# ==========================================
# Todos los clones juegan el mismo state. Medido en un i5-11600KF (6 núcleos, 12 hilos): el emulador sube hasta
# ~2.600 pasos/s con 10-12 clones y baja con 16 (se pisan en la CPU); cada clon ocupa ~250 MB de RAM. Un clon por
# hilo es el máximo que acelera
NUM_CLONES = os.cpu_count() or 12
FRAME_STACK = 4               # ver la velocidad y la dirección de Mario y de los enemigos
TOTAL_TIMESTEPS = 200_000_000  # tope por lanzamiento; lo normal es parar con PARAR o --minutos
PPO_LEARNING_RATE = 0.00025
PPO_N_STEPS = 1024
PPO_BATCH_SIZE = 512           # con 12 clones: 1.140 pasos/s frente a 1.020 con 256 (aprender 2,0 s frente a 3,3 s)
PPO_N_EPOCHS = 4
# Horizonte largo: valora el atasco y la meta lejanos. Con 4 fotogramas el atasco (-15) queda a 75 pasos:
# 0,998^75 = 0,86, pesa ~-12,9 desde donde empieza, peor que morir (-10)
PPO_GAMMA = 0.998
# Exploración: con 0,01 la entropía bajó a 1,69 y dejó de probar a saltar el foso; con 0,03, una vez dominado el
# 1-1 bajó a ~1,0 y dejó de probar cosas nuevas. Para buscar trucos hace falta que siga probando
PPO_ENT_COEF = 0.05
PPO_TARGET_KL = 0.03
CHECKPOINT_FREQ = 500_000     # pasos totales entre copias de seguridad
MAX_MINUTOS = 180             # de entrenamiento activo (sin contar suspensiones del PC)
