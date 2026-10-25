import json
import os
import shutil
import time
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback

from mario import config, states
from mario.env import cargar_records


def quitar_parar():
    if os.path.isdir(config.STOP_FILE):
        shutil.rmtree(config.STOP_FILE)
    elif os.path.exists(config.STOP_FILE):
        os.remove(config.STOP_FILE)


class ParadaCallback(BaseCallback):
    """Para de forma ordenada si existe el archivo PARAR o se superan MAX_MINUTOS de entrenamiento
    activo (un hueco de más de 5 min entre comprobaciones es el PC suspendido: no cuenta)."""

    def __init__(self):
        super().__init__()
        self.motivo = None
        self.segundos = 0.0
        self.ultimo = None

    def _on_training_start(self):
        self.ultimo = time.time()

    def _on_step(self):
        if self.n_calls % 100:
            return True
        ahora = time.time()
        if ahora - self.ultimo < 300:
            self.segundos += ahora - self.ultimo
        self.ultimo = ahora
        if os.path.exists(config.STOP_FILE):
            self.motivo = "parado por el usuario (PARAR)"
        elif config.MAX_MINUTOS and self.segundos > config.MAX_MINUTOS * 60:
            self.motivo = f"tiempo máximo ({config.MAX_MINUTOS} min)"
        if self.motivo:
            print(f"\n🛑 {self.motivo}: deteniendo el entrenamiento y guardando...")
            return False
        return True


class EpisodiosCallback(BaseCallback):
    """Lleva a TensorBoard el avance, las metas y el tiempo de los episodios terminados."""

    def __init__(self, ventana=100):
        super().__init__()
        self.metas = deque(maxlen=ventana)

    def _on_step(self):
        for info in self.locals["infos"]:
            if "motivo" not in info:
                continue
            meta = info["motivo"] == "meta"
            self.metas.append(meta)
            self.logger.record_mean("mario/progreso", info["progreso"])
            self.logger.record_mean("mario/muere", float(info["motivo"] == "muerte"))
            self.logger.record("mario/metas_ultimos_100", sum(self.metas) / len(self.metas))
            if meta:
                self.logger.record_mean("mario/segundos_meta", info["frames"] / 59.73)
        return True


def cargar_niveles():
    if not os.path.exists(config.NIVELES_PATH):
        return {}
    with open(config.NIVELES_PATH, encoding="utf-8") as f:
        return json.load(f)


def primer_nivel_pendiente():
    agotados = cargar_niveles()
    return next((n for n in config.NIVELES if config.nombre_nivel(n) not in agotados), None)


def nivel_agotado(metas, pasos_sin_mejora):
    """Agotado = ya llega a la meta con fiabilidad y hace mucho que no baja el tiempo."""
    return (len(metas) == metas.maxlen and sum(metas) / len(metas) >= config.META_MINIMA
            and pasos_sin_mejora >= config.PASOS_SIN_MEJORA)


class NivelesCallback(BaseCallback):
    """Repite un nivel hasta agotarlo, lo apunta en niveles.json y sigue con el siguiente pendiente usando el
    mismo modelo. Cuando no quedan niveles, para el entrenamiento."""

    def __init__(self, nivel, parada):
        super().__init__()
        self.parada = parada
        self.empezar(nivel)

    def empezar(self, nivel):
        self.nivel = nivel
        self.metas = deque(maxlen=100)
        record = cargar_records().get(config.nombre_nivel(nivel), {})
        self.mejor = record.get("frames")
        self.ultima_mejora = None  # num_timesteps de la última mejora (o del inicio del nivel)

    def _on_step(self):
        if self.ultima_mejora is None:
            self.ultima_mejora = self.num_timesteps
        nombre = config.nombre_nivel(self.nivel)
        for info in self.locals["infos"]:
            if info.get("nivel") != nombre:  # episodios del nivel anterior que aún terminaban
                continue
            meta = info["motivo"] == "meta"
            self.metas.append(meta)
            if meta and (self.mejor is None or info["frames"] < self.mejor):
                self.mejor = info["frames"]
                self.ultima_mejora = self.num_timesteps
        self.logger.record("mario/pasos_sin_mejora", self.num_timesteps - self.ultima_mejora)
        if not nivel_agotado(self.metas, self.num_timesteps - self.ultima_mejora):
            return True

        agotados = cargar_niveles()
        segundos = round(self.mejor / 59.73, 2) if self.mejor else None
        agotados[nombre] = {"frames": self.mejor, "segundos": segundos,
                            "pasos": int(self.num_timesteps), "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(config.NIVELES_PATH, "w", encoding="utf-8") as f:
            json.dump(agotados, f, indent=1)
        siguiente = primer_nivel_pendiente()
        print(f"\n✅ {nombre} agotado: {segundos} s sin mejorar en {config.PASOS_SIN_MEJORA:,} pasos")
        if siguiente is None:
            self.parada.motivo = "todos los niveles agotados"
            return False
        if not states.asegurar(siguiente):
            self.parada.motivo = f"no se pudo crear el state del {config.nombre_nivel(siguiente)}"
            return False
        print(f"➡️ Siguiente nivel: {config.nombre_nivel(siguiente)}")
        self.training_env.env_method("cambiar_nivel", siguiente)
        self.empezar(siguiente)
        return True
