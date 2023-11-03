import numpy as np
from scipy.io import wavfile

# Specify audio parameters
sample_rate = 44100  # Sample rate in Hz
duration = 1  # Duration of the snap sound in seconds

# Generate a mono snap sound (centered)
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)  # Time array

# Generate a snap sound for the left channel
snap_left = np.exp(-t) * np.sin(2 * np.pi * 1000 * t)  # Adjust the frequency as needed

# Generate a snap sound for the right channel
snap_right = np.exp(-t) * np.sin(2 * np.pi * 1200 * t)  # Adjust the frequency as needed

# Normalize the audio signals to the range [-1, 1]
snap_left = snap_left / np.max(np.abs(snap_left))
snap_right = snap_right / np.max(np.abs(snap_right))

# Create a stereo audio signal by combining the left and right channels
stereo_snap_sound = np.column_stack((snap_left, snap_right))

# Save the stereo snap sound as a WAV file
wavfile.write("stereo_snap_sound.wav", sample_rate, stereo_snap_sound.astype(np.float32))