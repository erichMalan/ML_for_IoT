import argparse
import json
import base64
import paho.mqtt.client as PahoMQTT
import tensorflow as tf

class InferEngine:
    def __init__(self, version, broker="mqtt.eclipseprojects.io", port=1883):
        self._isSubscriber = True
        self.broker = broker
        self.port = port
        
        self.model = f'{version}.tflite'
        self.interpreter = tf.lite.Interpreter(model_path=self.model)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.clientID = "inference-" + self.model
        
        self._sub_topic = "/PoliTO/ML4IOT/Group2/+/data/"
        self._pub_topic_base = f"/PoliTO/ML4IOT/Group2/+/results/{version}"
        
        # Create an instance of paho.mqtt.client
        self._paho_mqtt = PahoMQTT.Client(self.clientID, False) 
        
        # Register the callback
        self._paho_mqtt.on_connect = self.myOnConnect
        self._paho_mqtt.on_message = self.myOnMessageReceived
        self._paho_mqtt.notify = self.myNotify
    
    def myNotify(self, topic, msg):
        print(topic)
    
    def myOnConnect(self, paho_mqtt, userdata, flags, rc):
        print("Connected to %s with result code: %d" % (self.broker, rc))
    
    def myOnMessageReceived(self, paho_mqtt, userdata, msg):
        self._paho_mqtt.notify(msg.topic, msg.payload)
        self.run_inference(msg.topic, msg.payload)
    
    def myPublish(self, topic, msg):
        self._paho_mqtt.publish(topic, msg, 2)
    
    def mySubscribe(self, topic):
        print(f"Subscribing to {topic}")
        self._paho_mqtt.subscribe(topic, 2)
        print("Subscription complete")
    
    
    def start(self):
        print("Connecting to:", self.broker, self.port)
        self._paho_mqtt.connect(self.broker, self.port)
        self._paho_mqtt.loop_start()
        
        self.mySubscribe(self._sub_topic)
    
    def stop (self):
        if (self._isSubscriber):
            self._paho_mqtt.unsubscribe(self._sub_topic)
        
        self._paho_mqtt.loop_stop()
        self._paho_mqtt.disconnect()
     
    def run_inference(self, topic, data):
        sensorID = topic.split("/")[4]
        pubtopic = self._pub_topic_base.replace('+', sensorID)
        
        ''' msg = {
                    'bn': i,
                    'bt': timestamp,
                    'e': [
                        {'n': 'audio', 'u': '/', 't': 0, 'vd': data_string},
                        {'n': 'shape', 'u': '/', 't': 0, 'v': data_shape_bytes},
                        {'n': 'shape_len', 'u': '/', 't': 0, 'v': len(data.shape)}
                    ]
                }
        '''
        msg = json.loads(data)
        audio_id = msg['bn']
        
        shape_len = msg['e'][2]['v']
        
        shape = msg['e'][1]['vd']
        shape = tuple(shape.to_bytes(shape_len, byteorder ='big'))
        
        audio = msg['e'][0]['vd']
        audio = base64.b64decode(audio)
        audio = tf.io.decode_raw(audio, tf.float32)
        audio = tf.reshape(audio, shape)
        
        input_tensor = audio
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        '''
        nump_sorted = np.argsort(output_data)
        label = nump_sorted[-1]
        label = int(str(label))
        second = nump_sorted[-2]
        trust_score = output_data[label] - output_data[second]
        trust_score = float(str(trust_score))
        '''
        output_data_64 = base64.b64encode(output_data)
        output_data_string = output_data_64.decode()
        
        msg = {'bn': audio_id, "data": output_data_string}
        msg = json.dumps(msg)
        print(msg)
        self.myPublish(pubtopic, msg)

### Reading arguments
parser = argparse.ArgumentParser()
parser.add_argument('--version', default='1', type=str, help='Model version')
args = parser.parse_args()
version = args.version

#inf = InferEngine(model_name, broker='192.168.1.195')
inf = InferEngine(version)

try:
    inf.start()
    
    while (input()!=''):
        pass
    
    inf.stop()
    
except Exception as e:
    print(e)
