import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate

# Load the audio files captured by both microphones
fs, audio = wavfile.read('data/hit03.wav')


audio_microphone1 = audio[:, 0]  # Left channel
audio_microphone2 = audio[:, 1]  # Right channel

# Perform cross-correlation to find the time delay
correlation = correlate(audio_microphone1, audio_microphone2, mode='full')
delay = np.argmax(correlation) - len(audio_microphone1) + 1

# Calculate the Fourier Transform of the signals
spectrum_microphone1 = np.fft.fft(audio_microphone1)
spectrum_microphone2 = np.fft.fft(audio_microphone2)

# Choose the frequency component with the most phase difference
phase_difference = np.angle(spectrum_microphone2) - np.angle(spectrum_microphone1)
max_freq_index = np.argmax(np.abs(phase_difference))
max_freq = max_freq_index / len(phase_difference) * fs

# Calculate the distance based on the time delay and frequency
speed_of_sound = 343  # Speed of sound in meters per second
distance = (delay / fs) * speed_of_sound / (2 * np.pi * max_freq)

print(f"Time delay: {delay} samples")
print(f"Max Phase Difference Frequency: {max_freq} Hz")
print(f"Estimated Distance between Microphones: {distance} meters")