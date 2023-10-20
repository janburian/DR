import numpy as np
from pathlib import Path
import os
from scipy.io import wavfile
from scipy.signal import correlate


def load_data(path: Path):
    data = []
    filenames = os.listdir(path)
    for filename in filenames:
        audio = wavfile.read(os.path.join(path, filename))
        data.append(audio)

    return data


def Fourier_transform(audio):
    fs = audio[0]
    audio = audio[1]

    audio_microphone_1 = audio[:, 0]
    audio_microphone_2 = audio[:, 1]

    spectrum_microphone1 = np.fft.fft(audio_microphone_1)
    spectrum_microphone2 = np.fft.fft(audio_microphone_2)

    return [spectrum_microphone1, spectrum_microphone2]


def count_correlation(spectres):
    spectrum_microphone_1 = spectres[0]
    spectrum_microphone_2 = spectres[1]

    R = np.conj(spectrum_microphone_1) * spectrum_microphone_2  # Correlation theorem

    return R


def count_distance(time_delay):
    SPEED_OF_SOUND = 343  # in meters per second
    distance = time_delay * SPEED_OF_SOUND   # Calculate the distance based on the time delay

    return distance


if __name__=="__main__":
    path = os.path.join(Path(__file__).parent, "data")
    data = load_data(path)

    for audio in data:
        spectres = Fourier_transform(audio)
        R = count_correlation(spectres)
        r = np.fft.ifft(R)
        time_delay_samples = np.argmax(r)

        fs = audio[0] # Calculate the time delay in seconds
        time_delay = time_delay_samples / fs

        distance = count_distance(time_delay)

        print(f"Time delay: {time_delay} seconds")
        print(f"Estimated Distance between Microphones: {distance} meters")