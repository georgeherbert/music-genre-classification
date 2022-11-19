import numpy as np
import pickle
import librosa
import torch

DATASET = pickle.load(open("data/train.pkl", 'rb'))


def create_mel(wave):
    mel = librosa.feature.melspectrogram(
        y=wave,
        n_fft=1024,
        hop_length=512,
        n_mels=80
    )
    time_intervals = mel.shape[1]
    mel = mel[:, :min(time_intervals, 80)]
    mel = librosa.power_to_db(mel)
    padding_right = max(0, 80 - time_intervals)
    mel = np.pad(
        mel,
        [[0, 0], [0, padding_right]],
        mode='edge'
    )
    return torch.Tensor([mel])


def augment():
    new_dataset = []
    # new_dataset = DATASET
    for i, sample in enumerate(DATASET):
        file = sample[0]
        mel = sample[1]
        label = sample[2]
        wave = sample[3]
        new_dataset.append((file, mel, label, ""))
        for change in (-1, 1):
            # new_wave = librosa.effects.time_stretch(wave, rate=change)
            new_wave = librosa.effects.pitch_shift(
                wave,
                sr=22050,
                n_steps=change,
                bins_per_octave=12
            )
            new_mel = create_mel(new_wave)
            new_dataset.append((file, new_mel, label, ""))
        print(i + 1, len(new_dataset))
    with open("data/augment.pkl", "wb") as file:
        pickle.dump(new_dataset, file)


if __name__ == "__main__":
    augment()
