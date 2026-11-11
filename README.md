# IT
import os
import subprocess
import imageio_ffmpeg
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import noisereduce as nr
import soundfile as sf
import numpy as np

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

video_file = "long_video.mp4"
audio_file = "long_audio.wav"

command = [ffmpeg_path, '-y', '-i', video_file, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '1', audio_file]
subprocess.run(command, check=True)

y, sr = librosa.load(audio_file, sr=None, duration=60) 

plt.figure(figsize=(15, 6))
D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Спектрограмма (первые 60 секунд)')
plt.tight_layout()
plt.savefig("long_spectrogram.png")

data, rate = sf.read(audio_file)

chunk_size = 1000000
cleaned_data = []
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    if len(chunk) < 1000:
        break
    cleaned_chunk = nr.reduce_noise(y=chunk, sr=rate, stationary=True, prop_decrease=0.8)
    cleaned_data.append(cleaned_chunk)

cleaned_audio = np.concatenate(cleaned_data)
sf.write("long_clean_audio.wav", cleaned_audio, rate)
