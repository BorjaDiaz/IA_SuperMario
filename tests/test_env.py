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

NADA, DERECHA, SALTO, CORRER, SALTO_CORRIENDO = (), ("right",), ("right", "a"), ("right", "b"), ("right", "a", "b")
# Botones de una partida que llega al mástil del 1-1 con pasos de 4 fotogramas (encontrada con una búsqueda a lo
# bruto, no con la IA): se agarra en el fotograma 1418; el juego no vuelve al mapa hasta cientos de fotogramas después
META_1_1 = os.path.join(os.path.dirname(__file__), "datos", "meta_1-1.json")


@unittest.skipUnless(os.path.exists(config.state_path((1, 1))), "Falta el state del 1-1 (crear_states.py)")
class MarioEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MarioEnv(rank=99)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def jugar(self, botones, frames_por_accion=None):
        self.env.reset()
        recompensas, info = [], {}
        for b in botones:
            _, r, fin, _, info = self.env.paso(b, frames_por_accion)
            recompensas.append(r)
            if fin:
                break
        return recompensas, info, self.env.progreso_max

    def test_observation_is_a_small_grayscale_frame(self):
        obs, _ = self.env.reset()
        self.assertEqual(obs.shape, (1, 72, 80))

    def test_moving_right_is_rewarded_and_standing_still_costs(self):
        quieto, _, _ = self.jugar([NADA] * 20)
        derecha, _, _ = self.jugar([DERECHA] * 20)
        self.assertAlmostEqual(sum(quieto), 20 * config.REWARD_PASO)
        self.assertGreater(sum(derecha), sum(quieto))

    def test_standing_still_is_never_cheaper_than_dying(self):
        # Si quedarse quieto saliera más barato, la IA aprendería a esperar en vez de arriesgarse
        quieto, info, _ = self.jugar([NADA] * 2000)
        muerte, _, _ = self.jugar([DERECHA] * 2000)
        self.assertEqual(info["motivo"], "atasco")
        self.assertEqual(len(quieto), config.MAX_PASOS_SIN_AVANCE)
        self.assertLessEqual(sum(quieto), sum(muerte))

    def test_walking_into_the_first_goomba_is_a_death_at_the_hit(self):
        # Termina en el paso del golpe; las vidas no bajan hasta volver al mapa, cientos de fotogramas después
        _, info, _ = self.jugar([DERECHA] * 2000)
        self.assertEqual(info["motivo"], "muerte")
        self.assertEqual(info["frames"], 164)

    def test_goal_is_counted_at_the_frame_mario_grabs_the_flagpole(self):
        with open(META_1_1, encoding="utf-8") as f:
            acciones = json.load(f)
        recompensas, info, _ = self.jugar(acciones, 4)
        self.assertEqual(info["motivo"], "meta")
        self.assertEqual(info["frames"], 1418)
        self.assertEqual(len(recompensas), len(acciones))
        self.assertGreater(recompensas[-1], config.REWARD_META)

    def test_same_actions_give_the_same_game(self):
        # ver_ia.py --record se basa en esto: el emulador es determinista y el azar del juego va en el state
        rng = random.Random(1)
        acciones = [rng.choice([DERECHA, SALTO_CORRIENDO, SALTO, CORRER]) for _ in range(400)]
        self.assertEqual(self.jugar(acciones), self.jugar(acciones))


@unittest.skipUnless(os.path.exists(config.state_path((1, 1))), "Falta el state del 1-1 (crear_states.py)")
class PracticaTests(unittest.TestCase):
    """Episodios que empiezan en puntos de partidas propias que llegaron a la meta."""

    def setUp(self):
        with open(META_1_1, encoding="utf-8") as f:
            self.acciones = json.load(f)
        parche = mock.patch.object(config, "PRACTICA_PROB", 1.0)
        parche.start()
        self.addCleanup(parche.stop)

    def test_off_by_default_so_viewers_and_records_always_start_at_the_beginning(self):
        env = MarioEnv(rank=99)
        try:
            env.repetir(self.acciones, 4)
            env.reset()
            self.assertFalse(env.es_practica)
            self.assertEqual((env.frames, env.x()), (0, 32))
        finally:
            env.close()

    def test_after_a_game_that_reaches_the_goal_it_practises_from_its_points(self):
        env = MarioEnv(rank=99, practica=True)
        try:
            env.reset()
            self.assertFalse(env.es_practica)  # aún no hay puntos
            info = env.repetir(self.acciones, 4)
            self.assertEqual(info["motivo"], "meta")
            self.assertEqual(len(env.puntos_practica), 1418 // config.PRACTICA_CADA_FRAMES)
            env.reset()
            self.assertTrue(env.es_practica)
            self.assertGreater(env.frames, 0)  # el tiempo sigue contando desde el inicio real del nivel
            self.assertGreater(env.x(), 32)
        finally:
            env.close()

    def test_a_practice_game_keeps_the_real_time_and_never_sets_a_record(self):
        env = MarioEnv(rank=99, practica=True)
        try:
            env.repetir(self.acciones, 4)
            env.rng.choice = lambda puntos: puntos[-1]  # el punto más avanzado: fotograma 1320 = paso 330
            env.reset()
            self.assertEqual(env.frames, 1320)
            with mock.patch.object(env, "_guardar_si_es_record") as guardar:
                info = {}
                for b in self.acciones[1320 // 4:]:  # el resto de la misma partida
                    _, _, fin, _, info = env.paso(b, 4)
                    if fin:
                        break
            self.assertEqual((info["motivo"], info["frames"], info["practica"]), ("meta", 1418, True))
            guardar.assert_not_called()
        finally:
            env.close()


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
        self.assertEqual(states.crear_siguiente((1, 1), self.acciones, 4), (1, 2))
        env = MarioEnv(rank=99, nivel=(1, 1))
        try:
            env.reset()
            env.cambiar_nivel((1, 2))
            env.paso(DERECHA)
            self.assertEqual(env.nivel, (1, 1))  # el episodio en curso sigue en su nivel
            env.reset()
            self.assertEqual(env.nivel, (1, 2))
            self.assertEqual(env.x(), 32)
        finally:
            env.close()

    def test_a_game_that_dies_creates_nothing(self):
        self.assertIsNone(states.crear_siguiente((1, 1), [DERECHA] * 2000, 4))
        self.assertFalse(os.path.exists(config.state_path((1, 2))))


if __name__ == "__main__":
    unittest.main()
