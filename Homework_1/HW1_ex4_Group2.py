import argparse
import pandas as pd
import tensorflow as tf
import time
from datetime import datetime
import os
from scipy.io import wavfile

def _bytes_feature(value):                                                      
  """Returns a bytes_list from a string / byte. Best method"""                              
  if isinstance(value, type(tf.constant(0))):                                   
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  
#def _int_feature(value):                                                      
  #return tf.train.Feature(int64_list=tf.train.Int64List(value=value.tolist())) 
  
#def _float_feature(value):                                                      
  #return tf.train.Feature(float_list=tf.train.FloatList(value=audio.numpy().flatten().tolist()))  

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input path", type=str, default='./raw_data')
parser.add_argument("--output", help="output file", type=str, default='.HW1_ex4_Group2_output.tfrecord')
parser.add_argument('-v',nargs='?', default=False, const=True)
args = parser.parse_args()

out_filename = args.output
in_filename = args.input
verbose = args.v

df = pd.read_csv(in_filename+'/samples.csv', header = None)

with tf.io.TFRecordWriter(out_filename) as writer:
    for i in range(df.iloc[:,0].size):
        raw_date = ",".join([df.iloc[i,0],df.iloc[i,1]])
        date = datetime.strptime(raw_date, '%d/%m/%Y,%H:%M:%S')
        posix_date = time.mktime(date.timetuple())

        raw_audio = tf.io.read_file(in_filename+'/'+df.iloc[i,4])
        
        """no processing needed for bytesList format (takes string tensor)"""
        audio = raw_audio 
        
        """decode_wav to get float32 tensor"""
        # ~ audio, sample_rate = tf.audio.decode_wav(
                    # ~ raw_audio,
                    # ~ desired_channels=1,  # mono
                    # ~ desired_samples=48000 * 1) 
        
        """wavefile.read to get int tensor"""
        # ~ sample_rate, audio = wavfile.read(in_filename+'/'+df.iloc[i,4])
        
        datetime_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(posix_date)])) 
        temperature = tf.train.Feature(int64_list=tf.train.Int64List(value=[df.iloc[i,2]]))
        humidity = tf.train.Feature(int64_list=tf.train.Int64List(value=[df.iloc[i,3]]))

        mapping = {'datetime': datetime_feature,
                   'temperature': temperature,
                   'humidity': humidity,
                   'audio': _bytes_feature(audio)}
                   
        example = tf.train.Example(features=tf.train.Features(feature=mapping))
        writer.write(example.SerializeToString())
        
if verbose:
    print(os.path.getsize(out_filename))