"""Pulidor de récords: retoca una ruta, encuentra una más rápida y es una partida real y repetible."""
import json
import os
import tempfile
import unittest

os.environ["MARIO_SIN_RECORDS"] = "1"
os.environ["MARIO_RUN_DIR"] = tempfile.mkdtemp(prefix="test_mario_")

from mario import config
from mario.pulidor import Pulidor, comprobar

META_1_1 = os.path.join(os.path.dirname(__file__), "datos", "meta_1-1.json")


@unittest.skipUnless(os.path.exists(config.state_path((1, 1))), "Falta el state del 1-1 (crear_states.py)")
class PulidorTests(unittest.TestCase):
    def setUp(self):
        with open(META_1_1, encoding="utf-8") as f:
            self.record = {"botones": json.load(f), "frames_por_accion": 4}

    def test_seeds_with_the_record_at_its_exact_frame(self):
        pulidor = Pulidor((1, 1), self.record, semilla=1)
        try:
            self.assertEqual(pulidor.mejor_frames, 1418)
            self.assertEqual(len(pulidor.entradas), 1418)  # una entrada por fotograma
        finally:
            pulidor.cerrar()

    def test_finds_a_faster_route_that_replays_the_same_from_the_level_start(self):
        pulidor = Pulidor((1, 1), self.record, semilla=1)
        try:
            frames, entradas = pulidor.buscar(iteraciones=60)
        finally:
            pulidor.cerrar()
        self.assertLess(frames, 1418)  # una ruta lenta (búsqueda a lo bruto) siempre tiene fotogramas que ganar
        self.assertEqual(comprobar((1, 1), entradas), frames)

    def test_a_record_that_does_not_reach_the_goal_is_rejected(self):
        with self.assertRaises(ValueError):
            Pulidor((1, 1), {"botones": [["right"]] * 300, "frames_por_accion": 4})


if __name__ == "__main__":
    unittest.main()
