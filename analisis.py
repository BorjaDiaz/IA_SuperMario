"""Resumen de una ejecución del entrenamiento (por defecto, la más reciente).

Uso (desde la carpeta "Super Mario"):
    ..\\Pokemon\\.venv\\Scripts\\python.exe analisis.py
    ..\\Pokemon\\.venv\\Scripts\\python.exe analisis.py --run 2026-10-25
Guarda runs/<id>/resumen.txt y runs/<id>/progreso.png.
"""
import argparse
import glob
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mario import config
from mario.env import cargar_records


def tendencia(valores):
    k = max(1, len(valores) // 5)
    return float(np.mean(valores[:k])), float(np.mean(valores[-k:]))


def archivos_de_metricas(run):
    """metrics.jsonl (ejecuciones antiguas, todos los clones en un archivo) o metricas/clon_<n>.jsonl."""
    return glob.glob(os.path.join(run, "metrics.jsonl")) + glob.glob(os.path.join(run, "metricas", "*.jsonl"))


def leer_episodios(run):
    """Se salta las líneas rotas: en metrics.jsonl los clones escribían a la vez y a veces se mezclaban."""
    eps, rotas = [], 0
    for ruta in archivos_de_metricas(run):
        with open(ruta, encoding="utf-8") as f:
            for l in f:
                if not l.strip():
                    continue
                try:
                    eps.append(json.loads(l))
                except json.JSONDecodeError:
                    rotas += 1
    return sorted(eps, key=lambda r: r["timestamp"]), rotas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="ejecución concreta (basta parte del nombre)")
    args = parser.parse_args()
    runs = sorted(d for d in glob.glob(os.path.join(config.RUNS_DIR, "*")) if archivos_de_metricas(d))
    if args.run:
        runs = [d for d in runs if args.run in os.path.basename(d)]
    if not runs:
        raise SystemExit("No hay ejecuciones con datos.")
    run = runs[-1]
    eps, rotas = leer_episodios(run)

    L = [f"RESUMEN DE LA EJECUCIÓN {os.path.basename(run)}", "=" * 60]
    if os.path.exists(os.path.join(run, "run.json")):
        with open(os.path.join(run, "run.json"), encoding="utf-8") as f:
            info = json.load(f)
        L.append(f"Inicio {info['inicio']} · fin {info.get('fin', 'en curso')} · nivel {info['nivel']} · modelo {info['modelo']}")
        if "motivo_fin" in info:
            L.append(f"Motivo de fin: {info['motivo_fin']} · pasos {info['pasos_al_empezar']:,} → {info['pasos_al_terminar']:,}")
    motivos = Counter(r["motivo"] for r in eps)
    L.append(f"\nEPISODIOS: {len(eps)} · " + " · ".join(f"{m} {100 * n / len(eps):.0f}%" for m, n in motivos.most_common())
             + (f" · {rotas} líneas ilegibles ignoradas" if rotas else ""))
    p0, p1 = tendencia([r["progreso"] for r in eps])
    L.append(f"  Avance medio (píxeles): primer 20% {p0:.0f} → último 20% {p1:.0f} · máximo {max(r['progreso'] for r in eps)}")
    m0, m1 = tendencia([r["motivo"] == "meta" for r in eps])
    L.append(f"  Llega a la meta: primer 20% {100 * m0:.0f}% → último 20% {100 * m1:.0f}%")
    metas = [r for r in eps if r["motivo"] == "meta"]
    if metas:
        s = [r["frames"] / 59.73 for r in metas]
        L.append(f"  Tiempo hasta la meta: mejor {min(s):.2f} s · mediana primeras 10 {np.median(s[:10]):.2f} s → "
                 f"últimas 10 {np.median(s[-10:]):.2f} s")
    L.append("\nRÉCORDS (de todas las ejecuciones; ver_ia.py --record los repite)")
    for nivel, r in cargar_records().items():
        L.append(f"  {nivel}: {r['segundos']} s ({r['frames']} fotogramas, clon {r['clon']}, {r['run']}, {r['fecha']})")
    texto = "\n".join(L)
    with open(os.path.join(run, "resumen.txt"), "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print(texto)

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    ax[0].plot([r["progreso"] for r in eps], ".", alpha=0.3)
    ax[0].set_ylabel("avance (px)")
    if metas:
        idx = [eps.index(r) for r in metas]
        ax[1].plot(idx, [r["frames"] / 59.73 for r in metas], ".", color="tab:green")
    ax[1].set_ylabel("segundos hasta la meta")
    ax[1].set_xlabel("episodio")
    fig.savefig(os.path.join(run, "progreso.png"), dpi=100)


if __name__ == "__main__":
    main()
