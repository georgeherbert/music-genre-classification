import numpy as np
import pickle
import librosa
import torch
import random

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
    aug_dataset = DATASET.copy()
    for i, (file, mel, label, _) in enumerate(aug_dataset):
        aug_dataset[i] = (file, mel, label, np.array([]))
    for i in range(0, len(DATASET), 15):
        for stretch in (0.95, 1.05):
            for shift in (-0.5, 0.5):
                for j in random.sample(range(15), 3):
                    sample = DATASET[i + j]
                    new_wave = librosa.effects.time_stretch(
                        sample[3],
                        rate=stretch
                    )
                    new_wave = librosa.effects.pitch_shift(
                        new_wave,
                        sr=22050,
                        n_steps=shift,
                        bins_per_octave=12
                    )
                    new_mel = create_mel(new_wave)
                    aug_dataset.append(
                        (sample[0], new_mel, sample[2], np.array([]))
                    )
        print(i // 15, len(aug_dataset))
    with open("data/augment.pkl", "wb") as file:
        pickle.dump(aug_dataset, file)


if __name__ == "__main__":
    augment()
