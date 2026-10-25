"""Entorno de Mario con el emulador: avance, muerte, meta y repetición exacta de una partida."""
import json
import os
import random
import tempfile
import unittest

os.environ["MARIO_SIN_RECORDS"] = "1"
os.environ["MARIO_RUN_DIR"] = tempfile.mkdtemp(prefix="test_mario_")

from mario import config
from mario.env import MarioEnv

DERECHA, SALTO_CORRIENDO = 1, 4
# Acciones que llegan a la meta del 1-1 con timer_div=0 (encontradas con una búsqueda a lo bruto, no con la IA).
# Toca la meta en el fotograma 2301; el mundo no cambia hasta el 2986, tras el recuento del tiempo
META_1_1 = os.path.join(os.path.dirname(__file__), "datos", "meta_1-1.json")


@unittest.skipUnless(os.path.exists(config.ROM_PATH) and os.path.isdir(config.STATES_DIR),
                     "ROM o states no disponibles (crear_states.py)")
class MarioEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MarioEnv(rank=99)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def jugar(self, acciones, timer_div):
        self.env.reset(timer_div=timer_div)
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
        quieto, _, _ = self.jugar([0] * 20, timer_div=0)
        derecha, _, _ = self.jugar([DERECHA] * 20, timer_div=0)
        self.assertAlmostEqual(sum(quieto), 20 * config.REWARD_PASO)
        self.assertGreater(sum(derecha), sum(quieto))

    def test_walking_into_the_first_goomba_is_a_death_at_the_hit(self):
        # El golpe es en el fotograma 125 y termina al acabar ese paso; las vidas no bajan hasta el 336
        _, info, _ = self.jugar([DERECHA] * 2000, timer_div=0)
        self.assertEqual(info["motivo"], "muerte")
        self.assertEqual(info["frames"], 128)

    def test_goal_is_counted_at_the_frame_mario_touches_it(self):
        with open(META_1_1, encoding="utf-8") as f:
            acciones = json.load(f)
        recompensas, info, _ = self.jugar(acciones, timer_div=0)
        self.assertEqual(info["motivo"], "meta")
        self.assertEqual(info["frames"], 2301)
        self.assertEqual(len(recompensas), len(acciones))
        self.assertGreater(recompensas[-1], config.REWARD_META)

    def test_each_level_starts_from_its_state(self):
        self.env.cambiar_nivel((2, 3))
        self.env.reset(timer_div=0)
        self.assertEqual(self.env.juego.world, (2, 3))
        self.env.cambiar_nivel((1, 1))
        self.env.reset(timer_div=0)
        self.assertEqual(self.env.juego.world, (1, 1))

    def test_level_change_waits_for_the_next_reset(self):
        self.env.reset(timer_div=0)
        self.env.cambiar_nivel((1, 2))
        self.env.step(DERECHA)
        self.assertEqual(self.env.juego.world, (1, 1))
        self.env.reset(timer_div=0)
        self.assertEqual(self.env.juego.world, (1, 2))
        self.env.cambiar_nivel((1, 1))

    def test_same_actions_and_timer_give_the_same_game(self):
        # ver_ia.py --record se basa en esto: el emulador es determinista
        rng = random.Random(1)
        acciones = [rng.choice([DERECHA, SALTO_CORRIENDO, 2, 3]) for _ in range(400)]
        a = self.jugar(acciones, timer_div=37)
        b = self.jugar(acciones, timer_div=37)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
