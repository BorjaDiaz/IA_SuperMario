import os
import shutil
import time
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback

from mario import config


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
