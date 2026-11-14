"""Pule el récord de un nivel con una búsqueda tipo TAS (mario/pulidor.py), en varios procesos a la vez.

Uso (desde la carpeta "SuperMario"):
    .venv\\Scripts\\python.exe pulir.py                        # 1-1, 60 minutos, un proceso por hilo de la CPU
    .venv\\Scripts\\python.exe pulir.py --minutos 180 --procesos 6
El tiempo que se pule es el del nivel hasta que empieza el siguiente (cuentan los fuegos artificiales).
Trabaja por rondas: en cada una, todos los procesos parten del mejor tiempo hasta ese momento. Cada mejora se
comprueba repitiendo la partida desde el inicio del nivel, se guarda en records.json y se graba su vídeo.
Ctrl+C para parar (lo ya mejorado queda guardado). Si a la vez se entrena, conviene usar menos procesos.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

os.environ.setdefault("MARIO_RUN_DIR", tempfile.mkdtemp(prefix="pulir_mario_"))

from mario import config
from mario.env import cargar_records, guardar_record
from mario.pulidor import Pulidor, comprobar


def trabajar(nivel, record, semilla, segundos):
    """Una ronda de un proceso. Devuelve (fotogramas, entradas por fotograma, iteraciones)."""
    pulidor = Pulidor(nivel, record, semilla)
    try:
        frames, entradas = pulidor.buscar(segundos=segundos)
        return frames, entradas, pulidor.iteraciones
    finally:
        pulidor.cerrar()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nivel", default="1-1")
    parser.add_argument("--minutos", type=float, default=60)
    parser.add_argument("--procesos", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--ronda", type=float, default=5, help="minutos de cada ronda antes de compartir el mejor")
    args = parser.parse_args()
    nivel = tuple(map(int, args.nivel.split("-")))
    nombre = config.nombre_nivel(nivel)
    record = cargar_records().get(nombre)
    if not record:
        sys.exit(f"Aún no hay récord en el {nombre}: primero hay que pasarlo (train.py).")
    print(f"🔎 Puliendo el {nombre}: récord {record['segundos']} s ({record['frames']} fotogramas) · "
          f"{args.procesos} procesos · {args.minutos:g} min", flush=True)
    fin = time.time() + args.minutos * 60
    total, ronda = 0, 0
    with Pool(args.procesos) as pool:
        try:
            while time.time() < fin:
                ronda += 1
                segundos = min(args.ronda * 60, fin - time.time())
                semillas = [ronda * 1000 + i for i in range(args.procesos)]
                resultados = pool.starmap(trabajar, [(nivel, record, s, segundos) for s in semillas])
                total += sum(r[2] for r in resultados)
                frames, entradas, _ = min(resultados, key=lambda r: r[0])
                mejora = frames < record["frames"]
                print(f"  ronda {ronda}: {total:,} iteraciones · mejor {frames} fotogramas"
                      + (" (¡mejora!)" if mejora else ""), flush=True)
                if not mejora:
                    continue
                if comprobar(nivel, entradas) != frames:
                    print("  ⚠️ la mejora no se repite igual desde el inicio: descartada", flush=True)
                    continue
                guardar_record(nivel, frames, len(entradas), [list(e) for e in entradas], 1, "pulidor")
                record = cargar_records()[nombre]
                subprocess.Popen([sys.executable, os.path.join(config.BASE_DIR, "video.py"), "--nivel", nombre],
                                 cwd=config.BASE_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except KeyboardInterrupt:
            print("\n🛑 Parado: lo mejorado ya está guardado", flush=True)
    print(f"✅ Fin: récord del {nombre} {cargar_records()[nombre]['segundos']} s · {total:,} iteraciones", flush=True)


if __name__ == "__main__":
    main()
