import numpy as np
import pickle
import librosa
import torch
import random
import torchaudio
from matplotlib import pyplot as plt

DATASET = pickle.load(open("data/train.pkl", 'rb'))


def create_mel(wave):
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        n_mels=80,
        power=1,
    )
    mel = transform(torch.Tensor(wave))
    mel = (mel + 1e-8).log().numpy()

    time_intervals = mel.shape[1]
    mel = mel[:, :min(time_intervals, 80)]
    padding_right = max(0, 80 - time_intervals)
    mel = np.pad(
        mel,
        [[0, 0], [0, padding_right]],
        mode='edge',
    )

    return torch.Tensor([mel])


def augment():
    aug_dataset = DATASET.copy()
    for i, (file, mel, label, _) in enumerate(aug_dataset):
        aug_dataset[i] = (file, mel, label, np.array([]))
    for i in range(0, len(DATASET), 15):
        for stretch in (0.2, 0.5, 1.2, 1.5):
        # for shift in (-1, 1):
        # for shift in (-5, -2, 2, 5):
            for j in random.sample(range(15), 3):
                sample = DATASET[i + j]
                # new_wave = torchaudio.functional.pitch_shift(
                #     torch.Tensor(sample[3]),
                #     sample_rate=22050,
                #     n_steps=shift
                # )
                new_wave = librosa.effects.time_stretch(
                    sample[3],
                    rate=stretch
                )
                # new_wave = librosa.effects.pitch_shift(
                #     sample[3],
                #     sr=22050,
                #     n_steps=shift,
                #     bins_per_octave=12
                # )
                new_mel = create_mel(new_wave)
                aug_dataset.append(
                    (sample[0], new_mel, sample[2], np.array([]))
                )
        print(i // 15, len(aug_dataset))
    with open("data/augment.pkl", "wb") as file:
        pickle.dump(aug_dataset, file)


if __name__ == "__main__":
    augment()
