"""Pulidor de récords: búsqueda local tipo TAS que afina un récord fotograma a fotograma.

La IA (PPO) encuentra una ruta y la afina hasta un tope; los fotogramas que quedan se ganan como en un TAS:
volviendo a momentos concretos de la partida (states del emulador) y probando variantes de las entradas.

En cada iteración se retoca la mejor ruta conocida (una entrada por fotograma):
- se elige un momento y se cambian unos pocos fotogramas (otros botones, soltar o mantener uno, quitar fotogramas);
- después se sigue con el resto de la ruta original, saltando unos fotogramas si el cambio adelanta a Mario
  (si no, iría desincronizado con su propia ruta);
- si llega antes al mástil sin morir, pasa a ser la mejor ruta.

Para ir rápido se guardan states de la mejor ruta cada CADA_STATE fotogramas y se corta un intento en cuanto va
claramente por detrás de ella. Solo busca llegar antes (el QUÉ), nunca se le dice por dónde, y no toca la RAM:
cada mejora es una partida real, que además se comprueba repitiéndola entera desde el inicio del nivel.
(Probado antes: un Go-Explore con casillas y trozos al azar adelantaba casillas, pero en 600 iteraciones no
encadenó ninguna llegada antes al mástil, ni siquiera sobre una ruta lenta.)
"""
import io
import math
import random
import time

from pyboy import PyBoy

from mario import config

CADA_STATE = 16               # fotogramas entre los states guardados de la mejor ruta
VENTANA_MAX = 20              # fotogramas que se retocan como mucho en cada intento
SALTO_MAX = 8                 # fotogramas de la ruta original que se pueden saltar tras el retoque
RETRASO_MAX = 48              # píxeles por detrás de la mejor ruta (en el mismo fotograma) para cortar el intento
BOTONES = ("right", "left", "a", "b", "down", "up")
OPCIONES = list(config.ACCIONES) + [("a", "b"), ("right", "down")]


class Pulidor:
    def __init__(self, nivel, record, semilla=None):
        self.nivel = tuple(nivel)
        self.rng = random.Random(semilla)
        self.pb = PyBoy(config.ROM_PATH, window="null", sound_emulated=False)
        with open(config.state_path(self.nivel), "rb") as f:
            self.inicio = f.read()
        self.pulsados = set()
        self.iteraciones = 0
        entradas = [tuple(b) for b in record["botones"] for _ in range(record["frames_por_accion"])]
        frames = self._jugar_desde(0, entradas, guardar=True)
        if frames is None:
            raise ValueError("El récord no llega a la meta al repetirlo fotograma a fotograma")

    # ---------------------------------------------------------------- emulador
    def _pulsar(self, botones):
        deseados = set(botones)
        for b in self.pulsados - deseados:
            self.pb.button_release(b)
        for b in deseados - self.pulsados:
            self.pb.button_press(b)
        self.pulsados = deseados

    def _cargar(self, state, pulsados):
        self.pb.load_state(io.BytesIO(state))
        for b in BOTONES:
            self.pb.button_release(b)
        self.pulsados = set()
        self._pulsar(pulsados)

    def _x(self):
        return self.pb.memory[config.ADDR_X_PAGINA] * 256 + self.pb.memory[config.ADDR_X]

    def _jugar_desde(self, desde, entradas, guardar=False):
        """Juega `entradas` (la ruta completa) desde el fotograma `desde`, cargando el state guardado de la mejor
        ruta. Devuelve el fotograma de la meta, o None si muere o va por detrás. Con guardar=True (la ruta nueva
        mejor) guarda sus states y su posición en cada fotograma."""
        if desde == 0:
            self._cargar(self.inicio, ())
        else:
            self._cargar(self.states[desde], self.entradas[desde - 1])
        if guardar:
            states, xs = {0: self.inicio}, [self._x()]
        limite = self.mejor_frames if hasattr(self, "mejor_frames") and not guardar else math.inf
        for frame in range(desde, len(entradas)):
            if frame + 1 >= limite:
                return None
            self._pulsar(entradas[frame])
            self.pb.tick(1, False)
            estado = self.pb.memory[config.ADDR_ESTADO]
            if estado in config.ESTADOS_MUERTE:
                return None
            if estado == config.ESTADO_META:
                if guardar:
                    self.entradas, self.states, self.xs = list(entradas[:frame + 1]), states, xs
                    self.mejor_frames = frame + 1
                return frame + 1
            x = self._x()
            if guardar:
                xs.append(x)
                if (frame + 1) % CADA_STATE == 0:
                    guardado = io.BytesIO()
                    self.pb.save_state(guardado)
                    states[frame + 1] = guardado.getvalue()
            elif frame + 1 < len(self.xs) and x < self.xs[frame + 1] - RETRASO_MAX:
                return None
        return None

    # ------------------------------------------------------------- búsqueda
    def _retocar(self, t):
        """Una variante de la mejor ruta que cambia las entradas a partir del fotograma t."""
        e = self.entradas
        largo = self.rng.randint(1, VENTANA_MAX)
        tipo = self.rng.random()
        if tipo < 0.4:      # otras combinaciones, mantenidas de 1 a 6 fotogramas
            nuevas = []
            while len(nuevas) < largo:
                nuevas += [self.rng.choice(OPCIONES)] * self.rng.randint(1, 6)
            ventana = nuevas[:largo]
        elif tipo < 0.7:    # soltar o pulsar un botón durante la ventana
            boton = self.rng.choice(BOTONES)
            ventana = [tuple(sorted(set(x) ^ {boton})) for x in e[t:t + largo]]
        elif tipo < 0.85:   # mantener más tiempo lo que se pulsaba en t
            ventana = [e[t]] * largo
        else:               # quitar fotogramas: la ruta sigue antes
            ventana, largo = [], self.rng.randint(1, 4)
        salto = self.rng.randint(0, SALTO_MAX)
        return e[:t] + ventana + e[t + largo + salto:]

    def iteracion(self):
        """Prueba una variante. Devuelve los fotogramas de la meta si mejora la mejor ruta."""
        self.iteraciones += 1
        t = self.rng.randrange(len(self.entradas) - 1)
        variante = self._retocar(t)
        desde = (t // CADA_STATE) * CADA_STATE
        frames = self._jugar_desde(desde, variante)
        if frames is None or frames >= self.mejor_frames:
            return None
        self._jugar_desde(0, variante, guardar=True)  # nueva mejor ruta: guarda sus states
        return frames

    @property
    def mejor(self):
        return self.mejor_frames, self.entradas

    def buscar(self, segundos=None, iteraciones=None, al_mejorar=None):
        fin = time.time() + segundos if segundos else math.inf
        while time.time() < fin and (iteraciones is None or self.iteraciones < iteraciones):
            frame = self.iteracion()
            if frame is not None and al_mejorar:
                al_mejorar(*self.mejor)
        return self.mejor

    def cerrar(self):
        self.pb.stop(save=False)


def comprobar(nivel, entradas):
    """Repite la partida desde el inicio del nivel en un entorno limpio (un fotograma por paso).
    Devuelve los fotogramas de la meta, o None si no llega."""
    from mario.env import MarioEnv
    env = MarioEnv(rank=121, nivel=nivel, guardar_records=False)  # el récord lo guarda pulir.py como "pulidor"
    try:
        info = env.repetir(entradas, 1)
        return info["frames"] if info.get("motivo") == "meta" else None
    finally:
        env.close()
