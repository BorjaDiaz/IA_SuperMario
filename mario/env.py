"""Entorno de Gymnasium para el speedrun de un nivel de Super Mario Bros. Deluxe (PyBoy, leyendo la RAM)."""
import io
import json
import os
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyboy import PyBoy

from mario import config


class MarioEnv(gym.Env):
    def __init__(self, rank=0, ventana=False, velocidad=0, nivel=None, practica=False):
        """practica=True (solo el entrenamiento): algunos episodios empiezan en puntos de partidas propias que
        llegaron a la meta (ver PRACTICA_PROB en config.py)."""
        super().__init__()
        self.rank = rank
        self.practica = practica
        self.puntos_practica = deque(maxlen=config.PRACTICA_MAX)
        self.rng = random.Random(time.time_ns() + rank)
        self.pb = PyBoy(config.ROM_PATH, window="SDL2" if ventana else "null", sound_emulated=False)
        self.pb.set_emulation_speed(velocidad)
        self.nivel = None
        self.cambiar_nivel(nivel or config.NIVELES[0])
        self.action_space = spaces.Discrete(len(config.ACCIONES))
        self.observation_space = spaces.Box(0, 255, shape=(1, 72, 80), dtype=np.uint8)
        self.pulsados = set()
        self.episodio = 0
        # Un archivo por clon: escribiendo todos en el mismo, a veces se mezclaban líneas
        os.makedirs(os.path.join(config.RUN_DIR, "metricas"), exist_ok=True)
        self.metricas_path = os.path.join(config.RUN_DIR, "metricas", f"clon_{rank}.jsonl")

    # ------------------------------------------------------------------ gym
    def reset(self, seed=None, options=None):
        """Cada episodio empieza en el mismo fotograma: el azar del juego va dentro del state (cambiar el
        temporizador DIV, como en Super Mario Land, no altera nada), así que un récord se repite con sus acciones."""
        super().reset(seed=seed)
        if self.nivel_pendiente != self.nivel:
            self._cargar_state(self.nivel_pendiente)
        self._soltar_todo()
        punto = None
        if self.practica and self.puntos_practica and self.rng.random() < config.PRACTICA_PROB:
            punto = self.rng.choice(self.puntos_practica)
        self.es_practica = punto is not None
        if punto:
            estado, frames, botones = punto
            self.pb.load_state(io.BytesIO(estado))
            self._pulsar(botones)
        else:
            self.state.seek(0)
            self.pb.load_state(self.state)
        self.episodio += 1
        self.pasos = 0
        self.frames = punto[1] if punto else 0  # el tiempo cuenta desde el inicio real del nivel
        self.candidatos_practica = []
        self.proximo_punto = self.frames + config.PRACTICA_CADA_FRAMES
        self.pasos_sin_avance = 0
        self.vidas = self.pb.memory[config.ADDR_VIDAS]
        self.frame_meta = None
        self.progreso_inicio = self.progreso_max = self.x()
        self.progreso_anterior = self.progreso_max
        self.recompensa_total = 0.0
        self.acciones = []
        return self._obs(), {}

    def step(self, action):
        return self.paso(config.ACCIONES[action])

    def paso(self, botones, frames_por_accion=None):
        """Mantiene `botones` frames_por_accion fotogramas. step() lo usa con las acciones de la IA; las
        repeticiones de récords, con los botones y los fotogramas por acción con que se grabaron."""
        frames_por_accion = frames_por_accion or config.FRAMES_POR_ACCION
        self.frames_por_accion = frames_por_accion
        self._pulsar(botones)
        # Fotograma a fotograma para saber en cuál exacto toca la meta; solo se dibuja el que se observa
        for i in range(frames_por_accion):
            self.pb.tick(1, i == frames_por_accion - 1)
            if self.frame_meta is None and self.pb.memory[config.ADDR_ESTADO] == config.ESTADO_META:
                self.frame_meta = self.frames + i + 1
        self.pasos += 1
        self.frames += frames_por_accion
        self.acciones.append(list(botones))
        if self.practica and self.frames >= self.proximo_punto:
            guardado = io.BytesIO()
            self.pb.save_state(guardado)
            self.candidatos_practica.append((guardado.getvalue(), self.frames, tuple(self.pulsados)))
            self.proximo_punto += config.PRACTICA_CADA_FRAMES
        reward = config.REWARD_PASO

        # Avance: solo cuenta pasar de la mejor marca del episodio (ir y volver no paga)
        progreso = self.x()
        salto = progreso - self.progreso_anterior
        if progreso > self.progreso_max and 0 < salto <= config.MAX_PIXELES_POR_PASO:
            reward += config.REWARD_PIXEL * (progreso - self.progreso_max)
            self.progreso_max = progreso
            self.pasos_sin_avance = 0
        else:
            self.pasos_sin_avance += 1
            if progreso > self.progreso_max:  # lectura rara: fija la marca sin premiar
                self.progreso_max = progreso
        self.progreso_anterior = progreso

        # El estado de Mario cambia en el fotograma de la meta o de la muerte; las vidas bajan cientos de
        # fotogramas después, ya en el mapa: solo quedan de respaldo
        estado = self.pb.memory[config.ADDR_ESTADO]
        vidas, self.vidas = self.vidas, self.pb.memory[config.ADDR_VIDAS]
        motivo = None
        if estado in config.ESTADOS_MUERTE or self.vidas < vidas:
            motivo = "muerte"
            reward += config.PENALTY_MUERTE
        elif self.frame_meta is not None:
            motivo = "meta"
            self.frames = self.frame_meta
            reward += recompensa_meta(self.frames)
            self.puntos_practica.extend(self.candidatos_practica)  # solo puntos de partidas que llegan
            if not self.es_practica:
                self._guardar_si_es_record()
        elif self.pasos_sin_avance >= config.MAX_PASOS_SIN_AVANCE:
            motivo = "atasco"
            reward += config.PENALTY_ATASCO

        self.recompensa_total += reward
        info = {}
        if motivo:
            info = {"motivo": motivo, "nivel": config.nombre_nivel(self.nivel), "frames": self.frames,
                    "progreso": self.progreso_max - self.progreso_inicio, "practica": self.es_practica}
            self._registrar_episodio(motivo)
        return self._obs(), reward, motivo is not None, False, info

    def repetir(self, botones, frames_por_accion):
        """Juega una partida guardada (p. ej. un récord) desde el inicio del nivel. Devuelve el info final."""
        self.reset()
        info = {}
        for b in botones:
            _, _, fin, _, info = self.paso(b, frames_por_accion)
            if fin:
                break
        return info

    def cambiar_nivel(self, nivel):
        """El nivel cambia en el siguiente reset: el episodio en curso termina en el suyo."""
        nivel = tuple(nivel)
        if not os.path.exists(config.state_path(nivel)):
            raise FileNotFoundError(f"Falta {config.state_path(nivel)}: ejecuta crear_states.py")
        self.nivel_pendiente = nivel

    def x(self):
        """Posición de Mario en el nivel, en píxeles."""
        return self.pb.memory[config.ADDR_X_PAGINA] * 256 + self.pb.memory[config.ADDR_X]

    def close(self):
        self.pb.stop(save=False)

    # ------------------------------------------------------------ interno
    def _cargar_state(self, nivel):
        """El primer fotograma jugable del nivel (crear_states.py), con la ROM sin tocar."""
        with open(config.state_path(nivel), "rb") as f:
            self.state = io.BytesIO(f.read())
        self.nivel = nivel
        if self.practica:
            self.puntos_practica.clear()
            self._sembrar_practica()

    def _sembrar_practica(self):
        """Primeros puntos de práctica: los del récord del nivel, si lo hay (una partida de la propia IA, quizá
        con otros fotogramas por acción). Así practica tramos de todo el nivel antes de saber llegar."""
        record = cargar_records().get(config.nombre_nivel(self.nivel))
        if not record:
            return
        self.state.seek(0)
        self.pb.load_state(self.state)
        frames, proximo = 0, config.PRACTICA_CADA_FRAMES
        for botones in record["botones"]:
            self._pulsar(botones)
            for _ in range(record["frames_por_accion"]):
                self.pb.tick(1, False)
                frames += 1
                if self.pb.memory[config.ADDR_ESTADO] == config.ESTADO_META:
                    self._soltar_todo()
                    return
            if frames >= proximo:
                guardado = io.BytesIO()
                self.pb.save_state(guardado)
                self.puntos_practica.append((guardado.getvalue(), frames, tuple(self.pulsados)))
                proximo += config.PRACTICA_CADA_FRAMES
        self._soltar_todo()

    def _pulsar(self, botones):
        """Mantiene pulsados exactamente esos botones (suelta los demás)."""
        deseados = set(botones)
        for b in self.pulsados - deseados:
            self.pb.button_release(b)
        for b in deseados - self.pulsados:
            self.pb.button_press(b)
        self.pulsados = deseados

    def _soltar_todo(self):
        for b in self.pulsados:
            self.pb.button_release(b)
        self.pulsados = set()

    def _obs(self):
        # Grises y media de cada bloque 2x2: 144x160 -> 72x80
        rgb = self.pb.screen.ndarray[:, :, :3].astype(np.uint16)
        gris = (rgb[:, :, 0] * 77 + rgb[:, :, 1] * 150 + rgb[:, :, 2] * 29) >> 8
        gris = (gris[0::2, 0::2] + gris[1::2, 0::2] + gris[0::2, 1::2] + gris[1::2, 1::2]) >> 2
        return gris.astype(np.uint8)[None]

    def _registrar_episodio(self, motivo):
        fila = {"rank": self.rank, "episodio": self.episodio, "motivo": motivo, "pasos": self.pasos,
                "frames": self.frames, "progreso": self.progreso_max - self.progreso_inicio,
                "recompensa": round(self.recompensa_total, 3), "practica": self.es_practica,
                "nivel": config.nombre_nivel(self.nivel), "timestamp": round(time.time(), 1)}
        with open(self.metricas_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(fila) + "\n")

    def _guardar_si_es_record(self):
        """records.json guarda, por nivel, la partida más rápida: fotogramas y los botones de cada paso (no el
        número de acción, para que siga valiendo si cambia la lista de ACCIONES). El emulador es determinista,
        así que ver_ia.py --record la repite exactamente."""
        if not config.GUARDAR_RECORDS:
            return
        nivel = config.nombre_nivel(self.nivel)
        records = cargar_records()
        anterior = records.get(nivel, {}).get("frames")
        if anterior is not None and self.frames >= anterior:
            return
        records[nivel] = {"frames": self.frames, "segundos": round(self.frames / 59.73, 2),
                          "frames_por_accion": self.frames_por_accion,
                          "botones": self.acciones, "clon": self.rank,
                          "run": os.path.basename(config.RUN_DIR), "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}
        tmp = f"{config.RECORDS_PATH}.{self.rank}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(records, f)
        os.replace(tmp, config.RECORDS_PATH)
        antes = f" (antes {anterior / 59.73:.2f} s)" if anterior else ""
        print(f"🏆 [Clon {self.rank}] ¡Récord en el {nivel}! {self.frames / 59.73:.2f} s{antes}")


def recompensa_meta(frames):
    """Llegar da REWARD_META; llegar antes, hasta REWARD_META_RAPIDEZ más (lineal hasta FRAMES_REFERENCIA)."""
    rapidez = max(0.0, 1 - frames / config.FRAMES_REFERENCIA)
    return config.REWARD_META + config.REWARD_META_RAPIDEZ * rapidez


def cargar_records():
    if not os.path.exists(config.RECORDS_PATH):
        return {}
    with open(config.RECORDS_PATH, encoding="utf-8") as f:
        return json.load(f)
