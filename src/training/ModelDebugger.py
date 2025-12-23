# (c) Meta Platforms, Inc. and affiliates. 
# I build on the script that they wrote, making an object that I can use vs just a test script
# found here: https://pytorch.org/blog/understanding-gpu-memory-1/
import socket
from datetime import datetime, timedelta
import torch

class ModelDebugger:
    def __init__(self, configs):
        self.use_db = configs.training.debug_model
        self.cycles_before_stopping = configs.training.cycles_before_quit_debug
        self.num_cycles = 0

        self.TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

        # Keep a max of 100,000 alloc/free events in the recorded history
        # leading up to the snapshot.
        self.MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

    def start_record_memory_history(self) -> None:
        if not self.use_db:
            return

        if not torch.cuda.is_available():
            print("CUDA unavailable. Not recording memory history")
            return

        print("snapshot record_memory_history")
        torch.cuda.memory._record_memory_history(
            max_entries=self.MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
        )

    def check_cycle_limit(self):
        if not self.use_db:
            return

        self.num_cycles += 1
        if self.num_cycles > self.cycles_before_stopping:
            print("reached cycle limit. Disabling future saves. Saving memory info and quitting debugger")
            self.export_memory_snapshot()
            self.stop_record_memory_history()
            self.use_db = False

    def stop_record_memory_history(self) -> None:
        if not self.use_db:
            return

        if not torch.cuda.is_available():
            print("CUDA unavailable. Not recording memory history")
            return

        print("Stopping snapshot record_memory_history")
        torch.cuda.memory._record_memory_history(enabled=None)

    def export_memory_snapshot(self) -> None:
        if not self.use_db:
            return

        if not torch.cuda.is_available():
           print("CUDA unavailable. Not exporting memory snapshot")
           return

        # Prefix for file names.
        host_name = socket.gethostname()
        timestamp = datetime.now().strftime(self.TIME_FORMAT_STR)
        file_prefix = f"{host_name}_{timestamp}"

        try:
           print(f"Saving snapshot to local file: {file_prefix}.pickle")
           torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
        except Exception as e:
           print(f"Failed to capture memory snapshot {e}")
           return

