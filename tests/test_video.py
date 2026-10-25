"""El vídeo del récord se graba en otro proceso cuando records.json cambia, de uno en uno."""
import os
import tempfile
import unittest
from unittest import mock

os.environ["MARIO_SIN_RECORDS"] = "1"
os.environ["MARIO_RUN_DIR"] = tempfile.mkdtemp(prefix="test_mario_")

from mario import config
from mario.callbacks import VideoRecordCallback


class VideoRecordCallbackTests(unittest.TestCase):
    def setUp(self):
        self.records = os.path.join(tempfile.mkdtemp(prefix="test_records_"), "records.json")
        parche = mock.patch.object(config, "RECORDS_PATH", self.records)
        parche.start()
        self.addCleanup(parche.stop)
        self.popen = mock.patch("subprocess.Popen").start()
        self.addCleanup(mock.patch.stopall)
        self.cb = VideoRecordCallback((1, 1))

    def pasos(self, n=200):
        for _ in range(n):  # lo que hace on_step() de stable-baselines3, sin necesitar un modelo
            self.cb.n_calls += 1
            self.cb._on_step()

    def nuevo_record(self, version):
        with open(self.records, "w") as f:
            f.write("{}")
        os.utime(self.records, (version, version))

    def test_nothing_is_recorded_until_there_is_a_new_record(self):
        self.pasos(1000)
        self.popen.assert_not_called()
        self.nuevo_record(1000)
        self.pasos()
        self.assertEqual(self.popen.call_count, 1)
        self.assertIn("video.py", self.popen.call_args.args[0][1])

    def test_only_one_video_at_a_time_then_the_latest_record(self):
        self.nuevo_record(1000)
        self.pasos()
        self.popen.return_value.poll.return_value = None  # sigue grabando
        self.nuevo_record(2000)
        self.pasos(1000)
        self.assertEqual(self.popen.call_count, 1)
        self.popen.return_value.poll.return_value = 0     # terminó: graba el más reciente
        self.pasos()
        self.assertEqual(self.popen.call_count, 2)


if __name__ == "__main__":
    unittest.main()
