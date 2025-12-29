# I needed separate scripts for building and shuffling the corpus for bug testing, so here we are

import torch
from pathlib import Path

from .datatypes import NoteLabels, ProcessedAudioSegment
import pandas as pd
import numpy as np
import os
import logging 

def shuffle_corpus(configs):
    # shuffle shards 
    logging.info('Cleaning Corpus')
    clean_corpus(configs)

# function to shuffle the corpus after it was built so dataloader can used cached shards
# "clean corpus" in case I would like to do other post-processing, like removing outliers
def clean_corpus(configs):
    # allow the loading of my shards
    torch.serialization.add_safe_globals([ProcessedAudioSegment, NoteLabels])

    # gather variables
    ptd = os.path.join(configs.dataset.export_root, "train") # only need to shuffle the training dataset
    name = configs.dataset.processed_csv_name
    sz = configs.dataset.sharding.samples_per_shard

    # shuffle elements among each shard
    super_shuffle_shards(ptd, name, sz)

# currently not used since I suspect the shuffle was not strong enough
# path is a directory which houses a dataframe which points to all the shards, and all the shards
# shuffles items within each shard, then shuffles the shards
def shuffle_shards(path_to_data, df_name):
    # download the dataframe 
    dataframe_abs_name = os.path.join(path_to_data, df_name)
    df = pd.read_csv(dataframe_abs_name)

    # get a list of shard filenames 
    shards = sorted(Path.glob(Path(path_to_data), "*_shard*.pt")) # the star in beginning lets me shuffle anything

    # shuffle the shards themselves
    num_shards = len(shards)
    shard_perm = np.random.permutation(num_shards)
    shards = [shards[i] for i in shard_perm]

    # for each shard 
    for new_idx, shard_filename in enumerate(shards):
        # get the id 
        shard_id = int(shard_filename.stem.split("_shard")[1])

        # load the shard 
        shard = torch.load(shard_filename)
        n = len(shard)

        # shuffle the shard 
        perm = np.random.permutation(n)
        shard_shuffled = [shard[i] for i in perm]
        torch.save(shard_shuffled, shard_filename)
        
        # apply the same shuffle to the rows of the dataframe
        mask = df["shard_number"] == shard_id
        shard_rows = df.loc[mask].copy().reset_index(drop=True)
        shard_rows = shard_rows.iloc[perm].reset_index(drop=True)
        
        # apply the shard-level shuffle
        shard_rows["shard_number"] = new_idx

        # put the shuffled rows back into the dataframe
        df.loc[mask] = shard_rows.values

    # save the dataframe with the same filename (overwrite it)
    df.to_csv(dataframe_abs_name, index=False)
    logging.info(f"Shuffling complete. Updated dataframe saved to {dataframe_abs_name}")

# path is a directory which houses a dataframe which points to all the shards, and all the shards
# shard_len is size of all shards except last one
# VERY VERY IMPORTANT: I THINK THERE IS A BUG IN MY CODE WHICH MAKES SOME SHARDS ACCUMULATE SAMPLES RATHER THAN REPLACING them
# this would explain the first two shards being multiple gigabytes of length and would explain the runtime error where 
# I get Killed by the OS due to oom issues (trying to load a tensor which I cannot load)
# all the shards after the first two have a bit over double their expected size
# def super_shuffle_shards(path_to_data, df_name, shard_len):
#     # download the dataframe 
#     dataframe_abs_name = os.path.join(path_to_data, df_name)
#     df = pd.read_csv(dataframe_abs_name)
#
#     # get a list of shard filenames 
#     shards = sorted(Path.glob(Path(path_to_data), "*_shard*.pt")) # the star in beginning lets me shuffle anything
#
#     # get the total number of samples
#     shard_info = [] # used for global index mapping
#     global_idx = 0
#
#     for shard_path in shards:
#         shard_id = int(shard_path.stem.split("_shard")[1])
#         shard_size = len(torch.load(shard_path))
#         shard_info.append((shard_id, shard_size, global_idx, shard_path))
#         global_idx += shard_size
#
#     total_samples = global_idx
#     # shard_info.sort(key=lambda x: x[2]) # sort shard info by global index.
#
#     # create random permutation of data
#     permutation = np.arange(total_samples)
#
#     swaps = []
#     for i in range(total_samples - 1):
#         j = np.random.randint(i, total_samples)
#         if i != j:
#             swaps.append((i, j))
#             permutation[i], permutation[j] = permutation[j], permutation[i]
#
#     logging.info(f"Generated {len(swaps)} swaps")
#
#     # Helper functions
#     def swap_within_shard(shard, rows, local_i, local_j):
#         """Swap two elements within the same shard and dataframe rows."""
#         # Swap samples
#         shard[local_i], shard[local_j] = shard[local_j], shard[local_i]
#
#         # Swap dataframe rows (entire rows, preserving all columns)
#         rows.iloc[local_i], rows.iloc[local_j] = rows.iloc[local_j].copy(), rows.iloc[local_i].copy()
#
#         # Update shard_index to reflect new positions
#         rows.at[local_i, 'shard_index'] = local_i
#         rows.at[local_j, 'shard_index'] = local_j
#
#     def swap_between_shards(shard_a, rows_a, local_a, shard_a_id,
#                            shard_b, rows_b, local_b, shard_b_id):
#         """Swap elements between two different shards and their dataframe rows."""
#         # Swap samples
#         shard_a[local_a], shard_b[local_b] = shard_b[local_b], shard_a[local_a]
#
#         # Get the rows to swap
#         row_a = rows_a.iloc[local_a].copy()
#         row_b = rows_b.iloc[local_b].copy()
#
#         # Update shard_number and shard_index for the swapped rows
#         row_a['shard_number'] = shard_b_id
#         row_a['shard_index'] = local_b
#         row_b['shard_number'] = shard_a_id
#         row_b['shard_index'] = local_a
#
#         # Perform the swap
#         rows_a.iloc[local_a] = row_b
#         rows_b.iloc[local_b] = row_a
#
#     # Group swaps by shard pairs
#     shard_pair_swaps = {}
#     for i, j in swaps:
#         shard_i = i // shard_len
#         shard_j = j // shard_len
#         pair = tuple(sorted([shard_i, shard_j]))
#
#         if pair not in shard_pair_swaps:
#             shard_pair_swaps[pair] = []
#         shard_pair_swaps[pair].append((i, j))
#
#
#     # apply the swaps to the dataframe and shards
#     # the number of swap_list should be equal to (n+1)n/2 (where n is the number of shards)
#     for pair_idx, (pair, swap_list) in enumerate(shard_pair_swaps.items()):
#         # print(f"Shuffling swap list number {pair_idx}")
#         shard_a_id, shard_b_id = pair
#
#         # Get shard paths
#         shard_a_path = next(sp for sid, ss, si, sp in shard_info if sid == shard_a_id)
#         shard_a = torch.load(shard_a_path)
#         same_shard = (shard_a_id == shard_b_id)
#         if same_shard:
#             shard_b = shard_a
#             shard_b_path = shard_a_path
#         else:
#             shard_b_path = next(sp for sid, ss, si, sp in shard_info if sid == shard_b_id)
#             shard_b = torch.load(shard_b_path)
#
#         # Load dataframe rows
#         mask_a = df["shard_number"] == shard_a_id
#         rows_a = df.loc[mask_a].copy().reset_index(drop=True)
#         if same_shard:
#             rows_b = rows_a
#         else:
#             mask_b = df["shard_number"] == shard_b_id
#             rows_b = df.loc[mask_b].copy().reset_index(drop=True)
#
#         # for each index i, j to swap
#         for i, j in swap_list:
#             shard_i_id, local_i = (i // shard_len, i % shard_len)
#             shard_j_id, local_j = (j // shard_len, j % shard_len)
#
#             if same_shard:
#                 # Both in same shard
#                 swap_within_shard(shard_a, rows_a, local_i, local_j)
#             else:
#                 # Between different shards
#                 if shard_i_id == shard_a_id:
#                     swap_between_shards(shard_a, rows_a, local_i, shard_a_id,
#                                        shard_b, rows_b, local_j, shard_b_id)
#                 else:
#                     swap_between_shards(shard_b, rows_b, local_j, shard_b_id,
#                                        shard_a, rows_a, local_i, shard_a_id)
#
#         # save shards
#         torch.save(shard_a, shard_a_path)
#         if not same_shard:
#             torch.save(shard_b, shard_b_path)
#
#         # write dataframe rows back
#         df.loc[mask_a] = rows_a.values
#         if not same_shard:
#             mask_b = df["shard_number"] == shard_b_id
#             df.loc[mask_b] = rows_b.values
#
#         logging.info(f"Processed shard pair {pair_idx+1}/{len(shard_pair_swaps)}: "
#                     f"shards {shard_a_id}, {shard_b_id} ({len(swap_list)} swaps)")
#
#         # if shard_a != shard_b:
#         #     del shard_b
#         # del shard_a 
#
#     # Save the dataframe ONCE at the end
#     df.to_csv(dataframe_abs_name, index=False)
#     logging.info(f"Shuffling complete. Updated dataframe saved to {dataframe_abs_name}")

# super shuffle were I just bring all the shards into memory and then apply a transformation to the rows and 
# to the shards, while leaving the "global_idx" and "local_idx" the same. hopefully I have enough memory lol
def super_shuffle_shards(path_to_data, df_name, shard_len):
    # get the dataframe 
    df = pd.read_csv(os.path.join(path_to_data, df_name))

    # get a list of shard filenames 
    shard_names = sorted(Path.glob(Path(path_to_data), "*_shard*.pt")) # the star in beginning lets me shuffle anything

    # download all the shards 
    samples = []
    for fname in shard_names:
        shard = torch.load(fname)
        samples.extend(shard)

    # get the total number of samples
    total_samples = len(samples)
    print(f"Total Samples: {total_samples}")

    # create random permutation of data
    permutation = np.arange(total_samples)
    np.random.shuffle(permutation)

    # apply the permutation to the shards 
    permute_in_place(samples, permutation)

    # apply the permutation to the dataframe WHILE KEEPING GLOBAL AND LOCAL INDEX THE SAME 
    df_shuffled = df.iloc[permutation].reset_index(drop=True)
    df_shuffled['shard_number'] = df['shard_number'].values
    df_shuffled['shard_index']  = df['shard_index'].values

    # save the shards (OVERWRITING THE PREVIOUS ONES WITH THE SAME NAME. DO NOT APPEND THESE SHARDS)
    # to do this, I will reverse the order of the shards to utilize O(1) popping
    samples.reverse() # I love this

    temp = []
    shards_saved = 0
    while samples:
        temp.append(samples.pop())
        if len(temp) >= shard_len:
            final_path = shard_names[shards_saved]
            tmp_path = str(final_path) + ".tmp"

            torch.save(temp, tmp_path)
            os.replace(tmp_path, final_path)  # atomic overwrite

            shards_saved += 1
            print("saving:")
            print(temp)
            temp = [] # clear temp

    # make sure to save the last partial shard
    if len(temp) != 0:
        final_path = shard_names[shards_saved]
        tmp_path = str(final_path) + ".tmp"

        torch.save(temp, tmp_path)
        os.replace(tmp_path, final_path)  # atomic overwrite

    del temp # clear temp

    # save the dataframe (OVERWRITING THE PREVIOUS ONE WITH THE SAME NAME)
    df.to_csv("data.csv", index=False)

    # done :)

# hopefully this helper function works correctly
def permute_in_place(a, perm):
    visited = np.zeros(len(a), dtype=bool)

    for i in range(len(a)):
        if visited[i]:
            continue

        j = i
        temp = a[i]
        while not visited[j]:
            visited[j] = True
            k = perm[j]
            if k == i:
                a[j] = temp
            else:
                a[j] = a[k]
            j = k

