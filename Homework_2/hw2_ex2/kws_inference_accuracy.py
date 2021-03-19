import argparse
import numpy as np
from subprocess import call
import tensorflow as tf
import time
from scipy import signal
import os

if False:
    print("Culo")

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=False,
        help='model full path')
parser.add_argument('--rate', type=int, default=16000,
        help='sampling rate after resampling')
parser.add_argument('--mfcc', action='store_true',
        help='use mfcc')
parser.add_argument('--resize', type=int, default=32,
        help='input size after resize')
parser.add_argument('--length', type=int, default=640,
        help='stft window legnth in number of samples')
parser.add_argument('--stride', type=int, default=320,
        help='stft window stride in number of samples')
parser.add_argument('--bins', type=int, default=40,
        help='number of mel bins')
parser.add_argument('--coeff', type=int, default=10,
        help='number of MFCCs')
args = parser.parse_args()


call('sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
            shell=True)

rate = args.rate
length = args.length
stride = args.stride
resize = args.resize
num_mel_bins = args.bins
num_coefficients = args.coeff


#stride = 160
#length = 320
#rate = 8000
#args.mfcc = True
#num_coefficients = 8
#args.model = "pruned16000_2_mfccs_optimized.tflite"



num_frames = (rate - length) // stride + 1
num_spectrogram_bins = length // 2 + 1

linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, rate, 20, 4000)

if args.model is not None:
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

f = open("kws_test_split.txt", "r")
test_set = f.readlines()
f.close()

data_dir = os.path.join('.','data', 'mini_speech_commands')
LABELS = np.array(tf.io.gfile.listdir(str(data_dir))) 
LABELS = np.sort(LABELS[LABELS != 'README.md'])
print(LABELS)


inf_latency = []
tot_latency = []
num = 0
num_corr = 0
for audio_path in test_set:
    num += 1
    if num%100 == 0:
        print(num)
    
    #sample = np.array(np.random.random_sample(48000), dtype=np.float32)
    parts = tf.strings.split(audio_path, os.path.sep)
    label = parts[-2]
    label = tf.argmax(label == LABELS)
    label = label.numpy()
    
    audio_binary = tf.io.read_file(audio_path.replace("\n",''))
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=1)
    zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio,zero_padding],0)
    sample = audio.numpy()
    
    start = time.time()

    # Resampling
    #sample = signal.resample_poly(sample, 1, 48000 // rate)
    sample = signal.resample_poly(sample, 1, 16000 // rate)
    #sample = sample[::2]
    
    sample = tf.convert_to_tensor(sample, dtype=tf.float32)

    # STFT
    stft = tf.signal.stft(sample, length, stride,
            fft_length=length)
    spectrogram = tf.abs(stft)
    
    
    if args.mfcc is False and args.resize > 0:
        # Resize (optional)
        spectrogram = tf.reshape(spectrogram, [1, num_frames, num_spectrogram_bins, 1])
        spectrogram = tf.image.resize(spectrogram, [resize, resize])
        input_tensor = spectrogram
    else:
        # MFCC (optional)
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :num_coefficients]
        mfccs = tf.reshape(mfccs, [1, num_frames, num_coefficients, 1])
        input_tensor = mfccs
    
    '''
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mfccs_ = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs_ = mfccs_[..., :num_coefficients]
    mfccs_ = tf.expand_dims(mfccs_, -1)
    mfccs_ = tf.expand_dims(mfccs_, 0)
    input_tensor = mfccs_
    '''
    
    if args.model is not None:
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        start_inf = time.time()
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        if label == np.argmax(output_data):
            num_corr += 1

    end = time.time()
    tot_latency.append(end - start)

    if args.model is None:
        start_inf = end

    inf_latency.append(end - start_inf)
    time.sleep(0.1)

print('Inference Latency {:.2f}ms'.format(np.mean(inf_latency)*1000.))
print('Total Latency {:.2f}ms'.format(np.mean(tot_latency)*1000.))

print('Accuracy', num_corr/num)
