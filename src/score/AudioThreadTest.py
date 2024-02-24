'''
Testing different libraries to generate a dataframe from microphone input
Using PyAudio and Librosa. Using rms to calculate when a new note or rest occurs

'''

from AudioThreadWithBufferPorted import *
import pyaudio
from librosa import *
import matplotlib.pyplot as plt
import math
import crepe

from music21 import *
import numpy
import pandas as pd

from parsing.generate_new_score import AudioAnalysis

'''
callback function calls this and passes the most recent parts of the buffer
Will be useful to calculate frequencies and rms in real time.
Currently does nothing.
Will use this once calculations and dataframe generation are done

'''
def test(arg):
    
    return

'''
main function to create and run Audio Thread.
calculates frequencies and rms after Audio Thread finishes.
contents will be organized into functions once we achieve desired results.

'''
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
            #only let thread run for 5 seconds
            if time.time() - start > 10:
                break
                
    except KeyboardInterrupt:
        my_thread.stop_request = True
    buffer = my_thread.audio_buffer
    my_thread.stop_request = True
    
    #Librosa calculations
    numpy_array = np.frombuffer(buffer, dtype=np.float64)
    f0, voiced_flag, voiced_probs = pyin(y=buffer,
                                             fmin=note_to_hz('A0'),
                                             fmax=note_to_hz('C7'), sr=44100)
    #out = get_duration(y=buffer, sr=44100)
    
    #replace NaN with 0s
    if (len(f0) > 0):
        f0 = np.nan_to_num(f0, nan=0, posinf=10000, neginf=-10000)
        
    #get the time each entry is recorded    
    times = times_like(f0)
    
    notes = note_names_from_freqs(f0, 0) #C2 is the lowest note on a Cello (62 Hz)
    
    #The points in the array where a new note begins
    onset_frames = onset.onset_detect(y=buffer, sr=44100)
    print('onset_frames:', onset_frames)
    
    #Notes based on onsets
    onset_freqs = []
    onset_notes = []
    onset_times = []
    for my_onset in onset_frames: #note sure about the +1 here. But possibly onset is the note right before the switch
        freq = f0[my_onset]
        time_pos = times[my_onset]
        onset_freqs.append(freq)
        onset_times.append(time_pos)
    
    onset_notes = note_names_from_freqs(onset_freqs, 0)
    
    #Calculates RMS value of each entry
    rms = feature.rms(y=buffer)
    
    graph_rms(rms[0])

    
   
    #Create dataframe
    if onset_notes[0] == 'rest': #check to deal with starting time
        onset_notes.pop(0)
        onset_freqs.pop(0)
        onset_times.pop(0)
    my_dict = {'Note Name': onset_notes, 'Frequency': onset_freqs, 'Times': onset_times}
    #my_dict = {'Note Name': notes, 'Frequency': f0, 'Times': times}
    df = pd.DataFrame(data=my_dict)
    print(df)
    
    # testing = AudioAnalysis(df, "C:\\Users\\brian\\Desktop\\VIP\\Evaluator-code\\src\\score\\twinkle.musicxml")
   
    # testing.generate_overlay_score()

'''
Calculate the Note corresponding to the frequency 
Will set note name to Rest if frequency is below a certain frequency
Will return an array of note names corresponding to each entry in given array
'''
def note_names_from_freqs(f0: np.ndarray, rest_threshold:int=0):
    notes = []
    if len(f0) > 0:
        for freq in f0:
            if freq <= rest_threshold:
                notes.append('rest')
            else:
                notes.append(hz_to_note(freq, cents=True))
        print("librosa freq: ", f0)
        print("librosa out: ", notes)
    return notes


'''
Takes in rms array and graphs it
'''
def graph_rms(rms, color="green"):
    #Graph rms
    x = np.arange(0, len(rms))
    #print("rms: ", rms)
    plt.title("Line graph") 
    plt.xlabel("X axis") 
    plt.ylabel("Y axis") 
    plt.plot(x, rms, color =color) 
    plt.show()

if __name__ == "__main__":
    main()
    
    
