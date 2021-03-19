import os
import datetime as dt

import pyaudio
import wave

from board import D4
import adafruit_dht

number_of_sample = 10

chunk = 1024 # Record in chunks of 1024 samples
channels = 1
seconds = 1

output_folder = "raw_data"
os.mkdir(output_folder)

dht_device = adafruit_dht.DHT11(D4) # Sensor variable

sample_format = pyaudio.paInt16 # Number of bits per sample
sample_rate = 48000 # Number of samples per second

p = pyaudio.PyAudio()  # Create an interface to PortAudio

microphone_name = "USB Microphone: Audio"
dev_index = -1 # USB microphone index

# Searching the microphone index among all devices
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if microphone_name in dev['name']:
        dev_index = i
        break
print(f"Microphone index: {dev_index}\n")

csv_output = []
for n in range(1, 1 + number_of_sample):
    # Reading data from DHT11
    reading_time = dt.datetime.now().strftime("%d/%m/%Y,%H:%M:%S")
    try:
        temperature = dht_device.temperature
    except:
        temperature = -1000
    
    try:
        humidity = dht_device.humidity
    except:
        humidity = -1000
    
    # Reading data from microphone
    print('Recording')
    
    stream = p.open(format = sample_format,
                channels = channels,
                rate = sample_rate,
                frames_per_buffer = chunk,
                input_device_index = dev_index,
                input = True)
    
    frames = [] # Initialize array to store frames
    
    # Store data in chunks for 1 seconds
    for i in range(0, int(sample_rate * seconds / chunk)):
        data = stream.read(chunk)
        frames.append(data)
    
    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    
    print('Finished recording')
    
    # Save the recorded data as a WAV file
    audio_file_name = f"audio{n}.waw"
    wf = wave.open(f"{output_folder}/{audio_file_name}", 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    csv_output.append(f'{reading_time},{temperature},{humidity},{audio_file_name}\n')

# Terminate the PortAudio interface
p.terminate()

# Saving the csv output
f = open(output_folder + "/samples.csv", "w")
f.writelines(csv_output)
f.close()


print("End.")