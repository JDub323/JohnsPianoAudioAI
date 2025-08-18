import unittest
from hydra import initialize, compose
from pathlib import Path
from omegaconf import DictConfig
import librosa
import os 
import torch
from hydra.core.global_hydra import GlobalHydra

from .utils import plot_spec, prep_spectrogram_test

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        # Initialize Hydra (must reset per test suite)
        self._hydra_ctx = initialize(config_path="../../configs", job_name="test_app", version_base=None)
        self._hydra_ctx.__enter__()   # enter context


    def tearDown(self):
        # remove the hydra instance each time to avoid re-init errors
        self._hydra_ctx.__exit__(None, None, None)
        GlobalHydra.instance().clear()

    def test_rows_override(self):
        # Load config and override some parameters for testing
        self.cfg: DictConfig = compose(config_name="smalldata.yaml")

        self.assertEqual(self.cfg.dataset.row_limit, 10)

    def test_spectrogram(self):
        self.cfg: DictConfig = compose(config_name="smalldata.yaml")
        prep_spectrogram_test(self, regenerate_data=False)

        # get the spectrogram, arbitrarily from the test sub-directory
        directory = os.path.join(self.cfg.dataset.export_root, "test")
        first_tensor = next(f for f in os.listdir(directory) if f.endswith('.pt'))
        path = os.path.join(directory, first_tensor)
        segment = torch.load(path, weights_only=False)
        spec = segment.model_input.numpy()

        plot_spec(spec, sr=self.cfg.dataset.sample_rate)

    def test_custom_audio_file_spectrograms(self):
        self.cfg: DictConfig = compose(config_name="config.yaml", overrides=['dataset=testdata'])

        prep_spectrogram_test(self, regenerate_data=False)

        # get dir for easy access
        directory = os.path.join(self.cfg.dataset.export_root, "test")

        # loop through all items in my dataset
        for f in os.listdir(directory):
            # skip if not a pytorch saved file
            if not f.endswith('.pt'): continue

            # get spectrogram
            path = os.path.join(directory, f)
            segment = torch.load(path, weights_only=False)
            spec = segment.model_input.numpy()

            # plot spectrogram
            plot_spec(spec, sr=self.cfg.dataset.sample_rate, filename=f)


