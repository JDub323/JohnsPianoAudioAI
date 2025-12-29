import os
import shutil
import tempfile
import unittest
import torch
import pandas as pd

# ---------------------------------------------------------
# Test DataFrame factory (you were right — this was missing)
# ---------------------------------------------------------

def make_test_dataframe(num_shards, shard_len):
    rows = []
    sample_id = 0

    for shard_id in range(num_shards):
        for shard_index in range(shard_len):
            rows.append({
                "sample_id": sample_id,
                "shard_number": shard_id,
                "shard_index": shard_index,
            })
            sample_id += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Assertions
# ---------------------------------------------------------

def assert_global_integrity(tc, df, total_samples):
    ids = list(df["sample_id"])

    tc.assertEqual(
        len(ids),
        total_samples,
        "DataFrame size changed — accumulation or loss occurred"
    )

    tc.assertEqual(
        set(ids),
        set(range(total_samples)),
        "Sample IDs missing or duplicated"
    )


def assert_shard_integrity(tc, df, num_shards, shard_len):
    for shard_id in range(num_shards):
        shard_df = df[df["shard_number"] == shard_id]

        tc.assertEqual(
            len(shard_df),
            shard_len,
            f"Shard {shard_id} has wrong number of samples"
        )

        tc.assertEqual(
            set(shard_df["shard_index"]),
            set(range(shard_len)),
            f"Shard {shard_id} has invalid shard_index values"
        )


# ---------------------------------------------------------
# Test Case
# ---------------------------------------------------------

class TestSuperShuffleShards(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.num_shards = 3
        self.shard_len = 5
        self.total = self.num_shards * self.shard_len

        # Create shards on disk
        for shard_id in range(self.num_shards):
            shard = []
            for i in range(self.shard_len):
                sample_id = shard_id * self.shard_len + i
                shard.append({"sample_id": sample_id})
            torch.save(shard, os.path.join(self.tmpdir, f"train_shard_{shard_id}.pt"))

        # Create DataFrame
        df = make_test_dataframe(self.num_shards, self.shard_len)
        self.df_path = os.path.join(self.tmpdir, "index.csv")
        df.to_csv(self.df_path, index=False)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    # -----------------------------------------------------
    # TEST 1: No accumulation (this is the failure you see)
    # -----------------------------------------------------

    def test_no_accumulation_bug(self):
        super_shuffle_shards(self.tmpdir, "index.csv", self.shard_len)

        df = pd.read_csv(self.df_path)

        self.assertEqual(
            len(df),
            self.total,
            "Accumulation bug: rows were appended instead of reassigned"
        )

    # -----------------------------------------------------
    # TEST 2: Global permutation preserved
    # -----------------------------------------------------

    def test_all_samples_present_exactly_once(self):
        super_shuffle_shards(self.tmpdir, "index.csv", self.shard_len)

        df = pd.read_csv(self.df_path)
        assert_global_integrity(self, df, self.total)

    # -----------------------------------------------------
    # TEST 3: Shard structure preserved
    # -----------------------------------------------------

    def test_shard_structure_preserved(self):
        super_shuffle_shards(self.tmpdir, "index.csv", self.shard_len)

        df = pd.read_csv(self.df_path)
        assert_shard_integrity(self, df, self.num_shards, self.shard_len)


# ---------------------------------------------------------
# Import target function LAST (avoids circular imports)
# ---------------------------------------------------------

from src.corpus.build_corpus import super_shuffle_shards
