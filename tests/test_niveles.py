"""Cuándo se da un nivel por agotado y se pasa al siguiente."""
import os
import tempfile
import unittest
from collections import deque
from unittest import mock

os.environ["MARIO_SIN_RECORDS"] = "1"
os.environ["MARIO_RUN_DIR"] = tempfile.mkdtemp(prefix="test_mario_")

from mario import config
from mario.callbacks import nivel_agotado, primer_nivel_pendiente


def metas(llegadas, total=100):
    return deque([True] * llegadas + [False] * (total - llegadas), maxlen=100)


class NivelAgotadoTests(unittest.TestCase):
    def test_reliable_and_without_records_for_long_is_exhausted(self):
        self.assertTrue(nivel_agotado(metas(80), config.PASOS_SIN_MEJORA))

    def test_still_improving_is_not_exhausted(self):
        self.assertFalse(nivel_agotado(metas(100), config.PASOS_SIN_MEJORA - 1))

    def test_not_reaching_the_goal_reliably_is_not_exhausted(self):
        self.assertFalse(nivel_agotado(metas(79), 10 * config.PASOS_SIN_MEJORA))

    def test_needs_a_full_window_of_episodes(self):
        self.assertFalse(nivel_agotado(metas(50, total=50), 10 * config.PASOS_SIN_MEJORA))


class PrimerNivelPendienteTests(unittest.TestCase):
    def test_skips_exhausted_levels_in_order(self):
        with mock.patch("mario.callbacks.cargar_niveles", return_value={"1-1": {}, "1-2": {}}):
            self.assertEqual(primer_nivel_pendiente(), (1, 3))

    def test_none_when_all_are_exhausted(self):
        todos = {config.nombre_nivel(n): {} for n in config.NIVELES}
        with mock.patch("mario.callbacks.cargar_niveles", return_value=todos):
            self.assertIsNone(primer_nivel_pendiente())


if __name__ == "__main__":
    unittest.main()
