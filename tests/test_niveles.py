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


class QuedarseTests(unittest.TestCase):
    """Con --quedarse el nivel no cambia aunque esté agotado: se sigue exprimiendo."""

    def callback(self, quedarse):
        from mario.callbacks import NivelesCallback, ParadaCallback
        with mock.patch("mario.callbacks.cargar_records", return_value={"1-1": {"frames": 1300}}):
            cb = NivelesCallback((1, 1), ParadaCallback(), quedarse=quedarse)
        cb.model = mock.Mock(num_timesteps=0)  # también da cb.logger y cb.training_env
        cb.locals = {"infos": [{"motivo": "meta", "nivel": "1-1", "frames": 1400}] * 100}
        cb.on_step()  # 100 metas sin récord
        cb.model.num_timesteps = config.PASOS_SIN_MEJORA
        cb.locals = {"infos": []}
        return cb

    def test_exhausted_level_moves_on_without_quedarse(self):
        cb = self.callback(quedarse=False)
        with mock.patch("mario.callbacks.states.asegurar", return_value=True), \
                mock.patch("mario.callbacks.cargar_niveles", return_value={}),                 mock.patch("mario.callbacks.cargar_records", return_value={}), \
                mock.patch("builtins.open", mock.mock_open()):
            cb.on_step()
        self.assertEqual(cb.nivel, (1, 2))

    def test_exhausted_level_stays_with_quedarse(self):
        cb = self.callback(quedarse=True)
        self.assertTrue(cb.on_step())
        self.assertEqual(cb.nivel, (1, 1))
        cb.model.get_env.return_value.env_method.assert_not_called()


class RecompensaMetaTests(unittest.TestCase):
    def test_each_second_saved_near_the_current_record_is_worth_several_points(self):
        from mario.env import recompensa_meta
        fps = 59.73
        diferencia = recompensa_meta(21 * fps) - recompensa_meta(22 * fps)
        self.assertGreaterEqual(diferencia, 3)

    def test_reaching_the_goal_always_pays_at_least_reward_meta(self):
        from mario.env import recompensa_meta
        self.assertEqual(recompensa_meta(10 * config.FRAMES_REFERENCIA), config.REWARD_META)


class SeguridadTests(unittest.TestCase):
    """Explorar sin olvidar: guarda la copia segura si va bien y vuelve a ella si se estropea."""

    def callback(self):
        from mario.callbacks import SeguridadCallback
        cb = SeguridadCallback((1, 1))
        cb.model = mock.Mock(num_timesteps=0, device="cpu")
        return cb

    def jugar(self, cb, metas, total):
        infos = [{"motivo": "meta" if i < metas else "muerte", "nivel": "1-1", "practica": False}
                 for i in range(total)]
        cb.locals = {"infos": infos}
        cb.on_step()

    def test_saves_the_safe_model_while_it_reaches_the_goal_almost_always(self):
        cb = self.callback()
        self.jugar(cb, 290, config.SEGURIDAD_VENTANA)  # 97 %
        cb.model.save.assert_called_once_with(config.MODELO_SEGURO)
        cb.model.set_parameters.assert_not_called()

    def test_goes_back_to_the_safe_model_when_exploring_breaks_it(self):
        cb = self.callback()
        with mock.patch("mario.callbacks.os.path.exists", return_value=True):
            self.jugar(cb, 240, config.SEGURIDAD_VENTANA)  # 80 %
        cb.model.set_parameters.assert_called_once()
        self.assertEqual(cb.model.set_parameters.call_args.args[0], config.MODELO_SEGURO)
        self.assertEqual(len(cb.metas), 0)  # vuelve a medir desde cero

    def test_practice_games_do_not_count(self):
        cb = self.callback()
        cb.locals = {"infos": [{"motivo": "muerte", "nivel": "1-1", "practica": True}] * config.SEGURIDAD_VENTANA}
        cb.on_step()
        self.assertEqual(len(cb.metas), 0)
