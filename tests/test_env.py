"""Entorno de Mario con el emulador: avance, muerte, meta, cambio de nivel y repetición exacta de una partida."""
import json
import os
import random
import shutil
import tempfile
import unittest
from unittest import mock

os.environ["MARIO_SIN_RECORDS"] = "1"
os.environ["MARIO_RUN_DIR"] = tempfile.mkdtemp(prefix="test_mario_")

from mario import config, states
from mario.env import MarioEnv

DERECHA, SALTO_CORRIENDO = 1, 4
# Acciones que llegan al mástil del 1-1 (encontradas con una búsqueda a lo bruto, no con la IA): se agarra en
# el fotograma 1418; el juego no vuelve al mapa hasta cientos de fotogramas después
META_1_1 = os.path.join(os.path.dirname(__file__), "datos", "meta_1-1.json")


@unittest.skipUnless(os.path.exists(config.state_path((1, 1))), "Falta el state del 1-1 (crear_states.py)")
class MarioEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MarioEnv(rank=99)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def jugar(self, acciones):
        self.env.reset()
        recompensas, info = [], {}
        for a in acciones:
            _, r, fin, _, info = self.env.step(a)
            recompensas.append(r)
            if fin:
                break
        return recompensas, info, self.env.progreso_max

    def test_observation_is_a_small_grayscale_frame(self):
        obs, _ = self.env.reset()
        self.assertEqual(obs.shape, (1, 72, 80))

    def test_moving_right_is_rewarded_and_standing_still_costs(self):
        quieto, _, _ = self.jugar([0] * 20)
        derecha, _, _ = self.jugar([DERECHA] * 20)
        self.assertAlmostEqual(sum(quieto), 20 * config.REWARD_PASO)
        self.assertGreater(sum(derecha), sum(quieto))

    def test_walking_into_the_first_goomba_is_a_death_at_the_hit(self):
        # Termina en el paso del golpe; las vidas no bajan hasta volver al mapa, cientos de fotogramas después
        _, info, _ = self.jugar([DERECHA] * 2000)
        self.assertEqual(info["motivo"], "muerte")
        self.assertEqual(info["frames"], 164)

    def test_goal_is_counted_at_the_frame_mario_grabs_the_flagpole(self):
        with open(META_1_1, encoding="utf-8") as f:
            acciones = json.load(f)
        recompensas, info, _ = self.jugar(acciones)
        self.assertEqual(info["motivo"], "meta")
        self.assertEqual(info["frames"], 1418)
        self.assertEqual(len(recompensas), len(acciones))
        self.assertGreater(recompensas[-1], config.REWARD_META)

    def test_same_actions_give_the_same_game(self):
        # ver_ia.py --record se basa en esto: el emulador es determinista y el azar del juego va en el state
        rng = random.Random(1)
        acciones = [rng.choice([DERECHA, SALTO_CORRIENDO, 2, 3]) for _ in range(400)]
        self.assertEqual(self.jugar(acciones), self.jugar(acciones))


@unittest.skipUnless(os.path.exists(config.state_path((1, 1))), "Falta el state del 1-1 (crear_states.py)")
class NivelSiguienteTests(unittest.TestCase):
    """El state del 1-2 sale de repetir una partida que pasa el 1-1 (en una carpeta temporal)."""

    def setUp(self):
        self.carpeta = tempfile.mkdtemp(prefix="test_states_")
        shutil.copy(config.state_path((1, 1)), self.carpeta)
        parche = mock.patch.object(config, "STATES_DIR", self.carpeta)
        parche.start()
        self.addCleanup(parche.stop)
        with open(META_1_1, encoding="utf-8") as f:
            self.acciones = json.load(f)

    def test_beating_a_level_creates_the_next_state_and_the_env_can_switch_to_it(self):
        self.assertEqual(states.crear_siguiente((1, 1), self.acciones), (1, 2))
        env = MarioEnv(rank=99, nivel=(1, 1))
        try:
            env.reset()
            env.cambiar_nivel((1, 2))
            env.step(DERECHA)
            self.assertEqual(env.nivel, (1, 1))  # el episodio en curso sigue en su nivel
            env.reset()
            self.assertEqual(env.nivel, (1, 2))
            self.assertEqual(env.x(), 32)
        finally:
            env.close()

    def test_a_game_that_dies_creates_nothing(self):
        self.assertIsNone(states.crear_siguiente((1, 1), [DERECHA] * 2000))
        self.assertFalse(os.path.exists(config.state_path((1, 2))))


if __name__ == "__main__":
    unittest.main()
