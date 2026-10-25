# Speedrun de Super Mario Land con IA

PPO (stable-baselines3) + PyBoy. La IA aprende a pasar un nivel en el menor número de fotogramas.
Solo se le dice **qué** conseguir (avanzar y llegar a la meta cuanto antes), nunca **por dónde**: así
cualquier atajo o truco que encuentre se premia solo.

Usa el entorno de Python del proyecto de Pokémon (`..\Pokemon\.venv`). Desde esta carpeta:

| Qué | Comando |
|---|---|
| Crear los states de los 12 niveles (una vez) | `..\Pokemon\.venv\Scripts\python.exe crear_states.py` |
| Entrenar (continúa `ia/modelo.zip` si existe) | `..\Pokemon\.venv\Scripts\python.exe train.py [--nivel 1-2]` |
| Parar de forma ordenada | crear un archivo `PARAR` en esta carpeta |
| Resumen de la última ejecución | `..\Pokemon\.venv\Scripts\python.exe analisis.py` |
| Ver jugar a la IA | `..\Pokemon\.venv\Scripts\python.exe ver_ia.py` |
| Repetir exactamente el récord | `..\Pokemon\.venv\Scripts\python.exe ver_ia.py --record --nivel 1-1` |
| Tests | `..\Pokemon\.venv\Scripts\python.exe -m unittest discover -s tests -t .` |
| TensorBoard | `..\Pokemon\.venv\Scripts\tensorboard.exe --logdir runs` |

- `mario/config.py`: niveles, acciones, recompensas y PPO.
- Cada nivel empieza en `states/<mundo>-<nivel>.state`. `train.py` repite un nivel hasta agotarlo (llega a la
  meta en el 80 % de los últimos 100 episodios y lleva `PASOS_SIN_MEJORA` pasos sin récord), lo apunta en
  `niveles.json` y sigue con el siguiente usando el mismo modelo. Al relanzarlo continúa por el primero pendiente.
- `mario/env.py`: el entorno. Cada récord guarda en `records.json` sus acciones y la semilla del
  temporizador; el emulador es determinista, así que `ver_ia.py --record` lo repite igual.
- `runs/<fecha>/`: métricas por episodio, TensorBoard, `resumen.txt` y `progreso.png`.
- La ROM va en `roms/Super Mario Land (World).gb` (no se sube a git).
