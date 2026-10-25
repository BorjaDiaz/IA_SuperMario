"""Entrena la IA del speedrun de Super Mario Land. Continúa ia/modelo.zip si existe.

Uso (desde la carpeta "Super Mario"):
    ..\\Pokemon\\.venv\\Scripts\\python.exe train.py
Para pararla de forma ordenada, crea un archivo llamado PARAR en esta carpeta.
"""
import json
import os
import sys
import time

# La carpeta de la ejecución se fija antes de importar config: los clones (procesos aparte) la heredan
RUN_ID = time.strftime("%Y-%m-%d_%H%M%S")
os.environ["MARIO_RUN_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", RUN_ID)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from mario import config
from mario.callbacks import EpisodiosCallback, ParadaCallback, quitar_parar
from mario.env import MarioEnv


def crear_clon(rank):
    return lambda: Monitor(MarioEnv(rank=rank))


def hiperparametros():
    return dict(learning_rate=config.PPO_LEARNING_RATE, n_steps=config.PPO_N_STEPS, batch_size=config.PPO_BATCH_SIZE,
                n_epochs=config.PPO_N_EPOCHS, gamma=config.PPO_GAMMA, ent_coef=config.PPO_ENT_COEF,
                target_kl=config.PPO_TARGET_KL)


def main():
    quitar_parar()
    os.makedirs(config.RUN_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    env = VecFrameStack(SubprocVecEnv([crear_clon(i) for i in range(config.NUM_CLONES)]),
                        n_stack=config.FRAME_STACK, channels_order="first")
    tb = os.path.join(config.RUN_DIR, "tensorboard")
    if os.path.exists(config.MODELO):
        print("🧠 Cargando el modelo existente...")
        model = PPO.load(config.MODELO, env=env, tensorboard_log=tb, **hiperparametros())
        nuevo = False
    else:
        print("👶 Creando un modelo desde cero...")
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=tb, **hiperparametros())
        nuevo = True
    print(f"🖥️ Entrenando en: {model.device} · nivel {config.NIVEL} · {config.NUM_CLONES} clones")

    info = {"inicio": time.strftime("%Y-%m-%d %H:%M:%S"), "nivel": config.NIVEL, "modelo": "nuevo" if nuevo else "continuado",
            "pasos_al_empezar": int(model.num_timesteps),
            "config": {k: getattr(config, k) for k in dir(config) if k.isupper() and not k.endswith(("_DIR", "_PATH", "_FILE"))
                       and k not in ("MODELO", "ROM_PATH")}}
    parada = ParadaCallback()
    checkpoint = CheckpointCallback(save_freq=max(1, config.CHECKPOINT_FREQ // config.NUM_CLONES),
                                    save_path=config.CHECKPOINTS_DIR, name_prefix="mario")
    try:
        model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=[checkpoint, EpisodiosCallback(), parada],
                    reset_num_timesteps=nuevo, tb_log_name="PPO")
    except KeyboardInterrupt:
        parada.motivo = "interrumpido (Ctrl+C)"
    finally:
        model.save(config.MODELO)
        print(f"💾 Modelo guardado en {config.MODELO}")
        info.update(fin=time.strftime("%Y-%m-%d %H:%M:%S"), pasos_al_terminar=int(model.num_timesteps),
                    motivo_fin=parada.motivo or "terminado")
        with open(os.path.join(config.RUN_DIR, "run.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1, ensure_ascii=False)
        env.close()
        quitar_parar()
        print("📊 Análisis: python analisis.py")


if __name__ == "__main__":
    sys.exit(main())
