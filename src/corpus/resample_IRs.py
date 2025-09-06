import os
import soundfile as sf
from scipy.signal import resample_poly

# --- CONFIG ---
root = "/mnt/c/Users/jwhal/Projects/PythonProjects/JohnsPianoAudioAI/data/augmentations"
directories = [
        "environment/esc50-environment"
]
target_sr = 16000  # new sampling rate

def resample_wav(in_path, out_path, target_sr):
    """
    Resamples a single WAV file.
    """
    try:
        data, sr = sf.read(in_path)

        if sr == target_sr:
            print(f"Skipping {in_path}, already {sr} Hz")
            return

        # Compute resampling ratio
        gcd = __import__('math').gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd

        resampled = resample_poly(data, up, down, axis=0)
        sf.write(out_path, resampled, target_sr)
        print(f"Resampled: {in_path} -> {out_path}")

    except Exception as e:
        print(f"Error processing {in_path}: {e}")

# Check if the root directory exists
if not os.path.isdir(root):
    print(f"Error: Root directory not found at {root}")
    exit()

for d in directories:
    # Construct the full path to the source directory
    source_dir = os.path.join(root, d)

    # Construct the full path to the output directory
    out_dir = os.path.join(root, d + "_resampled")

    # Use os.walk on the full source directory path
    for dirpath, _, filenames in os.walk(source_dir):
        # Determine the relative path from the source directory
        rel_path = os.path.relpath(dirpath, source_dir)
        
        # Construct the full path for the new directory
        new_dir = os.path.join(out_dir, rel_path)
        
        # Create the new directory if it doesn't exist
        os.makedirs(new_dir, exist_ok=True)

        for f in filenames:
            if f.lower().endswith(".wav"):
                in_path = os.path.join(dirpath, f)
                out_path = os.path.join(new_dir, f)
                resample_wav(in_path, out_path, target_sr)
