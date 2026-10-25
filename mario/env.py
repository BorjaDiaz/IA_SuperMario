"""Entorno de Gymnasium para el speedrun de un nivel de Super Mario Land (PyBoy + su game wrapper)."""
import io
import json
import os
import random
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyboy import PyBoy

from mario import config


class MarioEnv(gym.Env):
    def __init__(self, rank=0, ventana=False, velocidad=0, nivel=None):
        super().__init__()
        self.rank = rank
        self.pb = PyBoy(config.ROM_PATH, window="SDL2" if ventana else "null", sound_emulated=False)
        self.pb.set_emulation_speed(velocidad)
        self.juego = self.pb.game_wrapper
        self.nivel = None
        self.cambiar_nivel(nivel or config.NIVELES[0])
        self.rng = random.Random(time.time_ns() + rank)
        self.action_space = spaces.Discrete(len(config.ACCIONES))
        self.observation_space = spaces.Box(0, 255, shape=(1, 72, 80), dtype=np.uint8)
        self.pulsados = set()
        self.episodio = 0
        os.makedirs(config.RUN_DIR, exist_ok=True)
        self.metricas_path = os.path.join(config.RUN_DIR, "metrics.jsonl")

    # ------------------------------------------------------------------ gym
    def reset(self, seed=None, options=None, timer_div=None):
        """timer_div: semilla del temporizador del juego (cambia el azar de enemigos). Al azar en cada
        episodio para no memorizar una sola partida; se guarda con el récord para poder repetirlo."""
        super().reset(seed=seed)
        self.timer_div = self.rng.randrange(256) if timer_div is None else timer_div
        if self.nivel_pendiente != self.nivel:
            self._cargar_state(self.nivel_pendiente)
        self._soltar_todo()
        # Lo mismo que reset_game del wrapper, pero desde el state del nivel en vez del de start_game
        self.state.seek(0)
        self.pb.load_state(self.state)
        self.juego._set_timer_div(self.timer_div)
        self.juego.post_tick()
        self.episodio += 1
        self.pasos = self.frames = 0
        self.pasos_sin_avance = 0
        self.vidas_inicio = self.juego.lives_left
        self.mundo_inicio = self.juego.world
        self.progreso_inicio = self.progreso_max = self.juego.level_progress
        self.progreso_anterior = self.progreso_max
        self.recompensa_total = 0.0
        self.acciones = []
        return self._obs(), {}

    def step(self, action):
        self._pulsar(config.ACCIONES[action])
        self.pb.tick(config.FRAMES_POR_ACCION - 1, False)
        self.pb.tick(1, True)  # solo se dibuja el fotograma que se observa
        self.pasos += 1
        self.frames += config.FRAMES_POR_ACCION
        self.acciones.append(int(action))
        reward = config.REWARD_PASO

        # Avance: solo cuenta pasar de la mejor marca del episodio (ir y volver no paga)
        progreso = self.juego.level_progress
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

        motivo = None
        if self.juego.world != self.mundo_inicio:
            motivo = "meta"
            rapidez = max(0.0, 1 - self.frames / config.FRAMES_REFERENCIA)
            reward += config.REWARD_META + config.REWARD_META_RAPIDEZ * rapidez
            self._guardar_si_es_record()
        elif self.juego.lives_left < self.vidas_inicio or self.juego.game_over():
            motivo = "muerte"
            reward += config.PENALTY_MUERTE
        elif self.pasos_sin_avance >= config.MAX_PASOS_SIN_AVANCE:
            motivo = "atasco"
            reward += config.PENALTY_ATASCO

        self.recompensa_total += reward
        info = {}
        if motivo:
            info = {"motivo": motivo, "nivel": config.nombre_nivel(self.nivel), "frames": self.frames,
                    "progreso": self.progreso_max - self.progreso_inicio}
            self._registrar_episodio(motivo)
        return self._obs(), reward, motivo is not None, False, info

    def cambiar_nivel(self, nivel):
        """El nivel cambia en el siguiente reset: el episodio en curso termina en el suyo."""
        nivel = tuple(nivel)
        if not os.path.exists(config.state_path(nivel)):
            raise FileNotFoundError(f"Falta {config.state_path(nivel)}: ejecuta crear_states.py")
        self.nivel_pendiente = nivel

    def close(self):
        self.pb.stop(save=False)

    # ------------------------------------------------------------ interno
    def _cargar_state(self, nivel):
        """El primer fotograma jugable del nivel (crear_states.py), con la ROM sin tocar."""
        with open(config.state_path(nivel), "rb") as f:
            self.state = io.BytesIO(f.read())
        self.nivel = nivel

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
        # Grises y media de cada bloque 2x2: 144x160 -> 72x80 (como en el proyecto de Pokémon)
        rgb = self.pb.screen.ndarray[:, :, :3].astype(np.uint16)
        gris = (rgb[:, :, 0] * 77 + rgb[:, :, 1] * 150 + rgb[:, :, 2] * 29) >> 8
        gris = (gris[0::2, 0::2] + gris[1::2, 0::2] + gris[0::2, 1::2] + gris[1::2, 1::2]) >> 2
        return gris.astype(np.uint8)[None]

    def _registrar_episodio(self, motivo):
        fila = {"rank": self.rank, "episodio": self.episodio, "motivo": motivo, "pasos": self.pasos,
                "frames": self.frames, "progreso": self.progreso_max - self.progreso_inicio,
                "recompensa": round(self.recompensa_total, 3), "timer_div": self.timer_div,
                "nivel": config.nombre_nivel(self.nivel), "timestamp": round(time.time(), 1)}
        with open(self.metricas_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(fila) + "\n")

    def _guardar_si_es_record(self):
        """records.json guarda, por nivel, la partida más rápida: fotogramas, timer_div y la lista de
        acciones. El emulador es determinista, así que ver_ia.py --record la repite exactamente."""
        if not config.GUARDAR_RECORDS:
            return
        nivel = config.nombre_nivel(self.nivel)
        records = cargar_records()
        anterior = records.get(nivel, {}).get("frames")
        if anterior is not None and self.frames >= anterior:
            return
        records[nivel] = {"frames": self.frames, "segundos": round(self.frames / 59.73, 2),
                          "timer_div": self.timer_div, "frames_por_accion": config.FRAMES_POR_ACCION,
                          "acciones": self.acciones, "clon": self.rank,
                          "run": os.path.basename(config.RUN_DIR), "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}
        tmp = f"{config.RECORDS_PATH}.{self.rank}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(records, f)
        os.replace(tmp, config.RECORDS_PATH)
        antes = f" (antes {anterior / 59.73:.2f} s)" if anterior else ""
        print(f"🏆 [Clon {self.rank}] ¡Récord en el {nivel}! {self.frames / 59.73:.2f} s{antes}")


def cargar_records():
    if not os.path.exists(config.RECORDS_PATH):
        return {}
    with open(config.RECORDS_PATH, encoding="utf-8") as f:
        return json.load(f)
