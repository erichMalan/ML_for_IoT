import argparse
import os
import time as t
import json
import datetime
import base64
import paho.mqtt.client as PahoMQTT
import numpy as np
import pandas as pd
import tensorflow as tf

class Processor():
    def __init__(self, clientID, dataf, preprocess, broker="mqtt.eclipseprojects.io", port=1883):
        self._isSubscriber = True
        self.clientID = clientID
        self.broker = broker
        self.port = port
        
        self._pub_topic = f"/PoliTO/ML4IOT/Group2/{clientID}/data/"
        self._sub_topic = f"/PoliTO/ML4IOT/Group2/{clientID}/results/+"
        
        self._records = pd.DataFrame(columns = ['audio', 'model', 'label', 'score'])
        self._test_ds = os.path.join(dataf, "kws_test_split.txt")
        self._ground_truth = []
        self._LABELS = os.path.join(dataf, "labels.txt")
        self._dataf = dataf
        self._preprocess = preprocess
        
        # create an instance of paho.mqtt.client
        self._paho_mqtt = PahoMQTT.Client(clientID, False) 
        
        # register the callback
        self._paho_mqtt.on_connect = self.myOnConnect
        self._paho_mqtt.on_message = self.myOnMessageReceived
        
        self.models = set()
        self.num_audio = None # 800
        self.num_messages_received = 0
    
    
    def myOnConnect(self, paho_mqtt, userdata, flags, rc):
        print(f"Connected to {self.broker} with result code: {rc}")
        print(f"User data: {userdata}, Flags: {flags}")
    
    def myOnMessageReceived(self, paho_mqtt, userdata, msg):
        self.store_record(msg.topic, msg.payload)
        '''
        if msg.topic != self._pub_topic:
            self.store_record(msg.topic, msg.payload)
        '''
    
    def myPublish (self, msg):
        self._paho_mqtt.publish(self._pub_topic, msg, 2)
    
    
    def start(self):
        self._paho_mqtt.connect(self.broker, self.port)
        self._paho_mqtt.loop_start()
        
        print(f"Subscribing to {self._sub_topic}")
        self._paho_mqtt.subscribe(self._sub_topic, 2)
        
    def stop(self):
        if (self._isSubscriber):
            # remember to unsuscribe if it is working also as subscriber 
            self._paho_mqtt.unsubscribe(self._sub_topic)
        
        self._paho_mqtt.loop_stop()
        self._paho_mqtt.disconnect()
    
    
    def store_record(self, topic, data):
        # data = {'bn': audio_id, "label": label, "ts": trust_score}
        tops = str(topic).split(os.path.sep)
        data = json.loads(data)
        
        network_data = base64.b64decode(data['data'])
        network_data = tf.io.decode_raw(network_data, tf.float32)
        
        nump_sorted = np.argsort(network_data)
        label = nump_sorted[-1]
        #label = int(str(label))
        second = nump_sorted[-2]
        trust_score = network_data[label] - network_data[second]
        #trust_score = float(str(trust_score))
        
        row = {
            'model': tops[-1],
            'audio': data['bn'],
            'label': label,
            'score': trust_score
        }
        
        self.models.add(tops[-1])
        self._records = self._records.append(row, ignore_index=True)
        self.num_messages_received += 1
    
    def print_result(self):
        recs = self._records.groupby(['audio','label'])['score'].sum().reset_index()
        
        idx = recs.groupby(['audio'])['score'].transform(max) == recs['score']
        predictions = recs[idx].sort_values(by=['audio'])['label'].to_numpy()
        
        correct_predictions = predictions[predictions == self._ground_truth]
        accuracy = len(correct_predictions) * 100.0 / len(predictions)
        print(f'Accuracy: {accuracy}%')
        
        self.stop()
    
    def preprocess(self, audio_path):
        audio_path = os.path.join(self._dataf, audio_path)
        parts = tf.strings.split(audio_path, os.path.sep)
        #idx = audio_path.split(os.path.sep)[-1]
        idx = parts[-1] # Filename
        label = parts[-2]
        label = tf.argmax(label == self._LABELS)
        label = label.numpy()
        
        audio_binary = tf.io.read_file(audio_path.replace("\n",''))
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)
        zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
        sample = tf.concat([audio,zero_padding],0)
        
        # STFT
        stft = tf.signal.stft(sample, self._preprocess['frame_length'], self._preprocess['frame_step'], fft_length=self._preprocess['frame_length'])
        spectrogram = tf.abs(stft)
        
        # MFCC
        mel_spectrogram = tf.tensordot(spectrogram, self._preprocess['linear_to_mel_weight_matrix'], 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self._preprocess['num_coefficients']]
        mfccs = tf.reshape(mfccs, [1, self._preprocess['num_frames'], self._preprocess['num_coefficients'], 1])
        data = mfccs
        
        return idx, data, label
    
    
    def read(self):
        f = open(self._LABELS, "r")
        self._LABELS = f.read().split(' ')
        self._LABELS = np.array(self._LABELS)
        f.close()
        
        f = open(self._test_ds, "r")
        test_set = f.readlines()
        self.num_audio = len(test_set)
        f.close()
        
        for i, audio_path in enumerate(test_set):
            now = datetime.datetime.now()
            timestamp = now.timestamp()
            
            _, data, label = self.preprocess(audio_path)
            
            data_64 = base64.b64encode(data)
            data_string = data_64.decode()
            
            data_shape_bytes = int.from_bytes(data.shape, byteorder ='big')
            
            msg = {
                'bn': i,
                'bt': timestamp,
                'e': [
                    {'n': 'audio', 'u': '/', 't': 0, 'vd': data_string},
                    {'n': 'shape', 'u': '/', 't': 0, 'vd': data_shape_bytes},
                    {'n': 'shape_len', 'u': '/', 't': 0, 'v': len(data.shape)}
                ]
            }
            msg = json.dumps(msg)
            
            self.myPublish(msg)
            self._ground_truth.append(label)
            
            if (i + 1) % 100 == 0:
                print(f"Sent {i + 1} / {len(test_set)}")
            
            t.sleep(0.1)
        
        self._ground_truth = np.array(self._ground_truth)
        if len(test_set) % 100 != 0:
            print(f"Sent {len(test_set)} / {len(test_set)}")
        print(f"End publication.")
    
    def wait_all_results(self, i):
        num_messages_to_receive = i * len(self.models)
        
        while self.num_messages_received < num_messages_to_receive:
            t.sleep(1)
            print(f"Received {self.num_messages_received} messages", end=' ')
            print(f"over {num_messages_to_receive}.")


### Reading arguments
parser = argparse.ArgumentParser()
parser.add_argument('--id', default="id1234", type=str, help='ID of the speech processor')
parser.add_argument('--maindir', default="./", type=str, help='Path to dataset, label.txt and test splits')
args = parser.parse_args()
clientID = args.id
maindir = args.maindir

data_dir = os.path.join(maindir,'data', 'mini_speech_commands')
if not os.path.exists(data_dir):
    zip_path = tf.keras.utils.get_file(
        origin='http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip',
        fname='mini_speech_commands.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')

preprocess = {
    'sampling_rate'     :   16000,
    'frame_length'      :   640,
    'frame_step'        :   320,
    'num_mel_bins'      :   80,
    'lower_freq'        :   20,
    'upper_freq'        :   4000,
    'num_coefficients'  :   20,
    'mfccs'             :   True
}
preprocess['num_frames'] = (preprocess['sampling_rate'] - preprocess['frame_length']) // preprocess['frame_step'] + 1
preprocess['num_spectrogram_bins'] = preprocess['frame_length'] // 2 + 1
preprocess['linear_to_mel_weight_matrix'] = tf.signal.linear_to_mel_weight_matrix(
                                                        preprocess['num_mel_bins'], 
                                                        preprocess['num_spectrogram_bins'], 
                                                        preprocess['sampling_rate'], 
                                                        preprocess['lower_freq'], 
                                                        preprocess['upper_freq'])


proc = Processor(clientID, maindir, preprocess)
proc.start()
proc.read()
proc.wait_all_results(proc.num_audio)
proc.print_result()

