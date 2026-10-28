# Speedrun de Super Mario Bros. Deluxe con IA

PPO (stable-baselines3) + PyBoy, en el modo "Original 1985" de Super Mario Bros. Deluxe (Game Boy Color):
los 32 niveles del Super Mario Bros. de NES. La IA aprende a pasar un nivel en el menor número de fotogramas.
Solo se le dice **qué** conseguir (avanzar y llegar a la meta cuanto antes), nunca **por dónde**: así
cualquier atajo o truco que encuentre se premia solo.

Tiene su propio entorno de Python en `.venv` (no se sube a git). Para crearlo, desde esta carpeta:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Después, desde esta carpeta:

| Qué | Comando |
|---|---|
| Crear el state del 1-1 y los de los niveles ya pasados | `.venv\Scripts\python.exe crear_states.py` |
| Entrenar (continúa `ia/modelo.zip` si existe) | `.venv\Scripts\python.exe train.py [--nivel 1-2]` |
| Parar de forma ordenada (guarda el modelo) | Ctrl+C en la terminal del entrenamiento (un segundo Ctrl+C corta sin esperar), o crear un archivo `PARAR` en esta carpeta |
| Resumen de la última ejecución | `.venv\Scripts\python.exe analisis.py` |
| Ver jugar a la IA | `.venv\Scripts\python.exe ver_ia.py` |
| Ver una partida guardada (p. ej. la del test, que llega al mástil del 1-1) | `.venv\Scripts\python.exe ver_ia.py --acciones tests\datos\meta_1-1.json` |
| Ver varios clones a la vez (último checkpoint, sirve mientras entrena) | `.venv\Scripts\python.exe ver_clones.py --clones 6` |
| Vídeo MP4 del récord (train.py lo graba solo en `videos/` con cada récord nuevo) | `.venv\Scripts\python.exe video.py --nivel 1-1` |
| Repetir exactamente el récord | `.venv\Scripts\python.exe ver_ia.py --record --nivel 1-1` |
| Tests | `.venv\Scripts\python.exe -m unittest discover -s tests -t .` |
| TensorBoard | `.venv\Scripts\tensorboard.exe --logdir runs` |

- `mario/config.py`: niveles, acciones, recompensas y PPO.
- Cada nivel empieza en `states/<mundo>-<nivel>.state`, conseguido jugando (`mario/states.py`): el del 1-1 desde
  los menús y el de cada nivel siguiente repitiendo el récord del anterior hasta el mástil. `train.py` repite un nivel hasta agotarlo (llega a la
  meta en el 80 % de los últimos 100 episodios y lleva `PASOS_SIN_MEJORA` pasos sin récord), lo apunta en
  `niveles.json` y sigue con el siguiente usando el mismo modelo. Al relanzarlo continúa por el primero pendiente.
- Búsqueda de trucos: acciones de 2 fotogramas con 15 combinaciones de botones, exploración alta y **práctica**:
  cada clon guarda puntos de sus partidas que llegan a la meta y en un 25 % de los episodios empieza en uno de ellos
  (con el tiempo contado desde el inicio real). Esas partidas no dan récords ni cuentan en las estadísticas.
- `mario/env.py`: el entorno. Lee de la RAM la X de Mario (avance) y su estado (5 = mástil, 3/17 = muerte), medidos
  en el emulador y anotados en `config.py`. Cada récord guarda en `records.json` sus acciones; el emulador es
  determinista y el azar del juego va en el state, así que `ver_ia.py --record` lo repite igual.
- `runs/<fecha>/`: métricas por episodio, TensorBoard, `resumen.txt` y `progreso.png`.
- La ROM va en `roms/Super Mario Bros. Deluxe (USA, Europe) (Rev A).gbc` (no se sube a git).
