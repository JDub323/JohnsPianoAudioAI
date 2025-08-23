import unittest
from hydra import initialize, compose
from omegaconf import DictConfig
import os 
import torch
from hydra.core.global_hydra import GlobalHydra
from src.corpus.DataAugmenter import DataAugmenter
import sounddevice as sd
from . import utils
from ..corpus.utils import print_augs

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
        self.cfg: DictConfig = compose(config_name="smalldata.yaml", overrides=['verbose=False'])

        self.assertEqual(self.cfg.dataset.row_limit, 10)
        self.assertEqual(self.cfg.verbose, False)

    def test_spectrogram(self):
        self.cfg: DictConfig = compose(config_name="smalldata.yaml")
        utils.prep_spectrogram_test(self, regenerate_data=False)

        # get the spectrogram, arbitrarily from the test sub-directory
        directory = os.path.join(self.cfg.dataset.export_root, "test")
        first_tensor = next(f for f in os.listdir(directory) if f.endswith('.pt'))
        path = os.path.join(directory, first_tensor)
        segment = torch.load(path, weights_only=False)
        spec = segment.model_input.numpy()

        utils.plot_spec(spec, sr=self.cfg.dataset.sample_rate)

    def test_custom_audio_file_spectrograms(self):
        self.cfg: DictConfig = compose(config_name="config.yaml", overrides=['dataset=testdata'])

        utils.prep_spectrogram_test(self, regenerate_data=True)

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
            utils.plot_spec(spec, sr=self.cfg.dataset.sample_rate, filename=f)

    def test_audio_augmentations(self):
        # get configs
        self.cfg: DictConfig = compose(config_name="smalldata.yaml")

        # gets all the wavs from the raw data (following the row limit)
        all_wavs = utils.get_raw_wavs(self.cfg, segment_length=10) # in seconds

        # make data augmenter object 
        my_augmenter = DataAugmenter(self.cfg)

        # loop through all wavs
        for index, raw_wav in enumerate(all_wavs):
            # generate augmentations
            aug_wav, augs = my_augmenter.augment_data(raw_wav)

            # play the original clip 
            print(f"\n\nPlaying clip number {index + 1}\n")
            sd.play(raw_wav, self.cfg.dataset.sample_rate)
            sd.wait()

            # play the new version and print the augmentations
            print("Now playing the augmented version")
            print("Augmentations: ")
            print_augs(augs)
            sd.play(aug_wav, self.cfg.dataset.sample_rate)
            sd.wait()


    # plays audio and video from before and after audio augmentations were done for 10 test tracks
    # does not run the build_corpus function, despite its name
    def test_corpus_build_quality(self):
        # get configs
        self.cfg: DictConfig = compose(config_name="smalldata.yaml", overrides=['verbose=False'])
        if self.cfg.verbose: print("OVERRIDE DIDN'T WORK: VERBOSE IS TRUE")

        # define how long each segment should be:
        sl = 30 # pneumonic: 'segment length'

        # get configs
        self.cfg: DictConfig = compose(config_name="smalldata.yaml")

        # get random indices so the same songs aren't played each run (I was going crazy)
        # bad code: this is the third time I download the df during this test
        max_val = len(utils.get_raw_data_df(self.cfg)) - 1 
        indices = utils.generate_rand_row_indices(count=self.cfg.dataset.row_limit, max_val=max_val)

        # generate data manually, since processed files lose audio data, making them unable to be played back
        # this returns an array of 30-second segments from each row I am interested in 
        raw_wavs = utils.get_raw_wavs(self.cfg, segment_length=sl, row_indices=indices)

        # download the labels, and cut them to their correct size
        note_labels = utils.download_cut_labels(self.cfg, segment_length=sl, row_indices=indices)

        # make sure the two arrays have the same number of songs
        assert len(raw_wavs) == len(note_labels)

        # make data augmenter object 
        my_augmenter = DataAugmenter(self.cfg)

        # for all audio files 
        for i, raw_wav in enumerate(raw_wavs):
            # play and display the raw version
            utils.play_and_display(configs=self.cfg, wav=raw_wav, labels=note_labels[i])

            # calculate the processed version
            aug_wav, augs = my_augmenter.augment_data(wav=raw_wav)

            # play and display the processed version, printing the augmentations to the terminal
            print_augs(augs) 
            utils.play_and_display(configs=self.cfg, wav=aug_wav, labels=note_labels[i])


