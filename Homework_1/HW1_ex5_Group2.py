import subprocess

performance = ['sudo', 'sh', '-c', 'echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']
powersave = ['sudo', 'sh', '-c', 'echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']

#subprocess.check_call(performance)
subprocess.check_call(powersave)

import os
import argparse
from io import BytesIO
import time as t

import pyaudio
import numpy as np
from scipy import signal
from scipy.io import wavfile

from tensorflow import convert_to_tensor, float32, tensordot
from tensorflow import abs as tfabs
from tensorflow.io import serialize_tensor, write_file
from tensorflow.math import log
from tensorflow.signal import linear_to_mel_weight_matrix,mfccs_from_log_mel_spectrograms, stft

from numpy import uint16, frombuffer
from scipy.signal import resample_poly

# Reading arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num-samples", help="num samples", type=int, default=5)
parser.add_argument("--output", help="output folder", type=str, default="./HW1_ex5_Group2_output/")
args = parser.parse_args()

number_of_sample = args.num_samples
output_folder = args.output
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
else:
    os.system(f"rm -r {output_folder}")
    #raise NameError('output folder already exists, choose another folder')

sample_format = pyaudio.paInt16 # Number of bits per sample
sample_rate = 48000 # Number of samples per second after recording
resample_rate = 16000 # Number of samples per second after resampling
downsample = int(sample_rate / resample_rate) # Ratio needed for resampling

num_chunks = 10 # Number of chunks
chunk = int(sample_rate / num_chunks) # Chunk size
channels = 1 # Mono
# seconds = 1

# Frame lenght and frame step, needed for the stft
l = 0.040
frame_length = int(resample_rate * l)

s = 0.020
frame_step = int(resample_rate * s)

# This matrix is always the same for every iteration, so we can already create it now
linear_to_mel_weight_matrix = linear_to_mel_weight_matrix(40, 321, 16000, 20, 4000)

# Create an interface to PortAudio
p = pyaudio.PyAudio()

# Searching the microphone index among all devices
# The index is usually 0
microphone_name = "USB Microphone: Audio"
dev_index = -0 # USB microphone index
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if microphone_name in dev['name']:
        dev_index = i
        break
print(f"Microphone index: {dev_index}\n")

# Opening the stream.
# The stream is initially stopped, it will be opened only when needed
stream = p.open(format = sample_format,
                channels = channels,
                rate = sample_rate,
                frames_per_buffer = chunk,
                input_device_index = dev_index,
                input = True,
                start = False)

# Cleaning the cpu usage
subprocess.check_call(['sudo', 'sh', '-c', 'echo 1 > /sys/devices/system/cpu/cpufreq/policy0/stats/reset'])

for n in range(number_of_sample):
    start = t.time()
    
    ###### Record
    stream.start_stream()
    
    # Once that the stream is strarted, we can set powersaves mode
    subprocess.Popen(powersave)
    frames = stream.read(chunk * (num_chunks - 1))
    
    # Before reading the last chunk, we set performance mode.
    # subprocess.Popen is not instantaneous so we need to call it a bit
    # before than when we actually need its effects.
    # In this case, we need the performance mode for preprocessing the audio.
    subprocess.Popen(performance)
    frames += stream.read(chunk)
    
    stream.stop_stream()
    
    ###### Resample
    #frame = np.frombuffer(io.BytesIO(b''.join(frames)).getbuffer(), dtype=np.uint16)
    frame = frombuffer(BytesIO(frames).getbuffer(), dtype=uint16)
    audio = resample_poly(frame, 1, downsample)
    tf_audio = convert_to_tensor(audio, dtype=float32) / 32767 - 1
    
    ###### STFT
    stft__ = stft(tf_audio, frame_length=frame_length, frame_step=frame_step, fft_length=frame_length)
    spectrogram = tfabs(stft__)
    
    ###### MFCCs
    mel_spectrogram = tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = log(mel_spectrogram + 1e-6)
    mfccs = mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :10]
    
    ###### Saving the output
    f_res = f'{output_folder}/mfccs{n}.bin'
    mfccs_ser = serialize_tensor(mfccs)
    write_file(f_res, mfccs_ser)
    
    ###### Printing execution time
    t_savefile = t.time()
    print(t_savefile - start)

# After reading all the samples we can set the powersaves mode
subprocess.Popen(powersave)

# Terminate the PortAudio interface
stream.close()
p.terminate()

# Reading the cpu usages
cpu_usage = subprocess.check_output(['cat', '/sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state']).decode('ascii')

print(cpu_usage)
