from audio.AudioThread import *

# import matplotlib.pyplot as plt
import math
import crepe
import pyaudio
from scipy.io import wavfile

import numpy as np
np.set_printoptions(threshold=np.inf)

def test(arg):
    time, frequency, confidence, activation = crepe.predict(arg, 44100, viterbi=False)
    print("time: ", str(time), "freq: ", str(frequency))
    print("test")
    return

def test_wav():
    sr, audio = wavfile.read('eqt-major-sc.wav')
    audio = audio[:, 0]
    print(audio.shape, type(audio[0]))
    time, frequency, confidence, activation = crepe.predict(audio[0:10000], 44100, viterbi=True)
    # print(frequency)
    
def main():
    test_wav()
    return
    # p = pyaudio.PyAudio()
    # info = p.get_host_api_info_by_index(0)
    # numdevices = info.get('deviceCount')

    # for i in range(0, numdevices):
    #     if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
    #         print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

    my_thread = AudioThreadWithBuffer('my thread', rate=44100, starting_chunk_size=1024, process_func=test)
    try: 
        my_thread.start()
        print("Start thread")
        while True:
            print(my_thread.input_on)
            time.sleep(1.5)
    except KeyboardInterrupt:
        my_thread.stop_request = True

if __name__ == "__main__":
    main()