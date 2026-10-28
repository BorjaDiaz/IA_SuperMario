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

    def __init__(self, max_minutos=None):
        super().__init__()
        self.max_minutos = config.MAX_MINUTOS if max_minutos is None else max_minutos
        self.motivo = None
        self.segundos = 0.0
        self.ultimo = None

    def _on_training_start(self):
        self.ultimo = time.time()

    def _on_step(self):
        if self.motivo:  # puesto desde fuera (Ctrl+C en train.py)
            print(f"\n🛑 {self.motivo}: deteniendo el entrenamiento y guardando...")
            return False
        if self.n_calls % 100:
            return True
        ahora = time.time()
        if ahora - self.ultimo < 300:
            self.segundos += ahora - self.ultimo
        self.ultimo = ahora
        if os.path.exists(config.STOP_FILE):
            self.motivo = "parado por el usuario (PARAR)"
        elif self.max_minutos and self.segundos > self.max_minutos * 60:
            self.motivo = f"tiempo máximo ({self.max_minutos} min)"
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
            if info.get("practica"):  # empezó a medio nivel: no cuenta para las metas ni el avance
                self.logger.record_mean("mario/practica_meta", float(info["motivo"] == "meta"))
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
    mismo modelo. Cuando no quedan niveles, para el entrenamiento. Con quedarse=True no pasa nunca al siguiente:
    sigue exprimiendo el nivel mientras se quiera."""

    def __init__(self, nivel, parada, quedarse=False):
        super().__init__()
        self.parada = parada
        self.quedarse = quedarse
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
            if info.get("nivel") != nombre or info.get("practica"):  # otro nivel, o empezó a medio nivel
                continue
            meta = info["motivo"] == "meta"
            self.metas.append(meta)
            if meta and (self.mejor is None or info["frames"] < self.mejor):
                self.mejor = info["frames"]
                self.ultima_mejora = self.num_timesteps
        self.logger.record("mario/pasos_sin_mejora", self.num_timesteps - self.ultima_mejora)
        if self.quedarse or not nivel_agotado(self.metas, self.num_timesteps - self.ultima_mejora):
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


class VideoRecordCallback(BaseCallback):
    """Cuando records.json cambia, graba en otro proceso (prioridad baja, de uno en uno) el vídeo del récord del
    nivel que se entrena. Los clones no se paran: video.py repite las acciones del récord por su cuenta."""

    def __init__(self, nivel):
        super().__init__()
        self.nivel = nivel
        self.proceso = None
        self.version_grabada = self._version()  # los récords previos ya tienen su vídeo (o se graban a mano)

    def _version(self):
        return os.path.getmtime(config.RECORDS_PATH) if os.path.exists(config.RECORDS_PATH) else None

    def _on_step(self):
        if self.n_calls % 200 or (self.proceso and self.proceso.poll() is None):
            return True
        version = self._version()
        if version == self.version_grabada:
            return True
        import subprocess
        import sys
        self.version_grabada = version
        # Prioridad baja y su propio grupo de procesos: un Ctrl+C en la terminal del entrenamiento no corta el vídeo
        prioridad_baja = (getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
                          | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        self.proceso = subprocess.Popen(
            [sys.executable, os.path.join(config.BASE_DIR, "video.py"), "--nivel", config.nombre_nivel(self.nivel)],
            cwd=config.BASE_DIR, creationflags=prioridad_baja,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True


class SeguridadCallback(BaseCallback):
    """Explorar sin olvidar: guarda MODELO_SEGURO mientras llega a la meta casi siempre y, si la exploración lo
    estropea (baja de SEGURIDAD_MINIMA), vuelve a esa copia. Solo cuentan las partidas normales del nivel."""

    def __init__(self, nivel):
        super().__init__()
        self.nivel = config.nombre_nivel(nivel)
        self.metas = deque(maxlen=config.SEGURIDAD_VENTANA)
        self.ultimo_guardado = 0.0
        self.vueltas = 0

    def _on_step(self):
        for info in self.locals["infos"]:
            if "motivo" in info and not info.get("practica") and info.get("nivel") == self.nivel:
                self.metas.append(info["motivo"] == "meta")
        if len(self.metas) < self.metas.maxlen:
            return True
        tasa = sum(self.metas) / len(self.metas)
        self.logger.record("mario/tasa_seguridad", tasa)
        if tasa >= config.SEGURIDAD_BUENA:
            if time.time() - self.ultimo_guardado >= config.SEGURIDAD_MINUTOS * 60:
                self.model.save(config.MODELO_SEGURO)
                self.ultimo_guardado = time.time()
        elif tasa < config.SEGURIDAD_MINIMA and os.path.exists(config.MODELO_SEGURO):
            self.model.set_parameters(config.MODELO_SEGURO, device=self.model.device)
            self.vueltas += 1
            self.logger.record("mario/vueltas_al_seguro", self.vueltas)
            print(f"\n🛟 Llegaba a la meta solo el {100 * tasa:.0f} %: vuelvo al último modelo seguro")
            self.metas.clear()
        return True
