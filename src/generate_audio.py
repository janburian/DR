import os.path
import numpy as np
from scipy.io import wavfile

# Audio parameters
sample_rate = 44100  # Sample rate in Hz
duration = 2  # Duration in seconds
frequency = 500  # Hz
OUTPUT_filename = "generated_audio.wav"

# Generate a mono audio signal
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
audio_signal = t * np.sin(2 * np.pi * frequency * t)

# Normalize the audio signal to the range [-1, 1]
audio_signal = audio_signal / np.max(np.abs(audio_signal))

# Create a delay
num_delay_samples = int(0.02 * sample_rate)  # Number of delay samples
delayed_audio_signal = np.roll(audio_signal, num_delay_samples)  # Shifting audio signal

# Create a stereo audio signal
stereo_audio_signal = np.column_stack((audio_signal, delayed_audio_signal))

# Save the stereo audio signal as a WAV file
wavfile.write(os.path.join("./data/", OUTPUT_filename), sample_rate, stereo_audio_signal.astype(np.float32))
# wavfile.write(os.path.join(OUTPUT_filename), sample_rate, stereo_audio_signal.astype(np.float32))