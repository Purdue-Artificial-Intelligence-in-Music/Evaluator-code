from AudioThreadWithBufferPorted import *
import pyaudio
from librosa import *
import matplotlib.pyplot as plt
import math
import crepe

from music21 import *
import numpy
import pandas as pd

def test(arg):
    a = 0
    a += 1
    #print(type(arg[0]))
    #print("arg: ", arg)
    # numpy_array = np.frombuffer(arg, dtype=np.float64)
    # f0, voiced_flag, voiced_probs = pyin(y=arg,
    #                                          fmin=note_to_hz('C2'),
    #                                          fmax=note_to_hz('C7'), sr=44100)
    # if (len(f0) > 0):
    #     f0 = f0[~np.isnan(f0)]
    
    # times = times_like(f0)
    # if len(f0) > 0:
    #     notes = hz_to_note(f0)
    #     print("librosa freq: ", f0)
    #     print("librosa out: ", notes)
    # time, frequency, confidence, activation = crepe.predict(arg, 16000, viterbi=False)
    # print(frequency)
    # if len(frequency) > 0:
    #     notes = hz_to_note(frequency)
    #     print("crepe out: ", notes)
    return
def main():
    
    #Create and start PyAudio thread
    my_thread = AudioThreadWithBufferPorted('my thread', rate=44100, starting_chunk_size=1024, process_func=test)
    start = time.time()
    try: 
        my_thread.start()
        print("Start thread")
        while True:
            print(my_thread.input_on)
            time.sleep(1.5)
            if time.time() - start > 5:
                break
                
    except KeyboardInterrupt:
        my_thread.stop_request = True
    buffer = my_thread.audio_buffer
    my_thread.stop_request = True
    
    #Librosa calculations
    numpy_array = np.frombuffer(buffer, dtype=np.float64)
    f0, voiced_flag, voiced_probs = pyin(y=buffer,
                                             fmin=note_to_hz('C2'),
                                             fmax=note_to_hz('C7'), sr=44100)
    #out = get_duration(y=buffer, sr=44100)
    
    if (len(f0) > 0):
        f0 = f0[~np.isnan(f0)]
    
    times = times_like(f0)
    notes = []
    if len(f0) > 0:
        notes = hz_to_note(f0)
        print("librosa freq: ", f0)
        print("librosa out: ", notes)
    
    # rms = feature.rms(y=buffer)
    # x = np.arange(0, len(rms[0]))
    # print(len(rms[0]))
    # print("rms: ", rms)
    # plt.title("Line graph") 
    # plt.xlabel("X axis") 
    # plt.ylabel("Y axis") 
    # plt.plot(x, rms[0], color ="green") 
    # plt.show()
    onset_frames = onset.onset_detect(y=buffer, sr=44100)
    print('onset_frames:', onset_frames)
    rms = feature.rms(y=buffer)
    x = np.arange(0, len(rms[0]))
    print(len(rms[0]))
    print("rms: ", rms)
    plt.title("Line graph") 
    plt.xlabel("X axis") 
    plt.ylabel("Y axis") 
    plt.plot(x, rms[0], color ="green") 
    plt.show()

    
   
    #Create dataframe
    my_dict = {'Note Name': notes, 'Frequency': f0, 'Times': times_like(f0)}
    df = pd.DataFrame(data=my_dict)
    print(df)


if __name__ == "__main__":
    main()