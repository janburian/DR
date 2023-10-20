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


def count_correlation(spectrums):
    spectrum_microphone_1 = spectrums[0]
    spectrum_microphone_2 = spectrums[1]

    R = spectrum_microphone_1 * spectrum_microphone_2 # Correlation theorem

    return R


def calculate_distance():
    pass


if __name__=="__main__":
    path = os.path.join(Path(__file__).parent, "data")
    data = load_data(path)

    for audio in data:
        # correlation = count_correlation(audio)
        spectrums = Fourier_transform(audio)
        R = count_correlation(spectrums)
        r = np.fft.ifft(R)
        t = np.argmax(r)

        # Calculate the time delay in seconds
        time_delay = t / audio[0]

        # Assuming the speed of sound in air is 343 meters per second
        speed_of_sound = 343  # in meters per second

        # Calculate the distance based on the time delay
        distance = time_delay * speed_of_sound

        print(f"Time delay: {time_delay} seconds")
        print(f"Estimated Distance between Microphones: {distance} meters")