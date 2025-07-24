import numpy as np
import librosa

def take_transform(audio: np.ndarray, transform_type: str) -> np.ndarray:
    print()
    if transform_type == 'cqt':
        return librosa.feature.cqt(audio)
