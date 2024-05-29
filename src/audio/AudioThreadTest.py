'''
Testing different libraries to generate a dataframe from microphone input
Using PyAudio and Librosa. Using rms to calculate when a new note or rest occurs

'''

from AudioThread import *
import pyaudio
from librosa import *
import matplotlib.pyplot as plt
import math
import re
import time
# import crepe

from music21 import *
import numpy as np
import pandas as pd

from generate_new_score import AudioAnalysis

'''
callback function calls this and passes the most recent parts of the buffer
Will be useful to calculate frequencies and rms in real time.
Currently does nothing.
Will use this once calculations and dataframe generation are done

'''
my_buffer = np.empty(0)
def test(arg):
    #Building my_buffer by appending data each time
    #Caution: Will keep taking up memory until no more.
    global my_buffer
    #print("here")
    # print("my_buffer: ", my_buffer)
    # print("arg: ", arg)
    my_buffer = np.append(my_buffer, arg)
    # calculate(my_buffer)
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
            print("listening")
            time.sleep(1.5)
            #only let thread run for time seconds
            timer = 10
            
            if time.time() - start > timer:
                break
                
    except KeyboardInterrupt:
        my_thread.stop_request = True
    in_buffer = my_thread.audio_buffer
    print('stopped')
    my_thread.stop() #manually stopping it is also causing the same problems??
    
    #calculate duration for each note
    #maybe see what vibrato looks like?
    #try recording with actual mic.
    
    # print('buffer calculation')
    # calculate(in_buffer, False) #This one fails (means in_buffer is changing still)
    
    #my_buffer is built by appending data each time
    print('my_buffer calculation')
    begin = time.time()
    calculate(my_buffer, False)
    ending = time.time()
    
    print("Calculations took", ending - begin, "seconds")
    

def calculate(buffer, rms_graph=False, fast=True):
    start = time.time()
    #Librosa calculations
    numpy_array = np.frombuffer(buffer, dtype=np.float64)
    if fast:
        f0 = yin(y=buffer,
            fmin=note_to_hz('A0'),
            fmax=note_to_hz('C7'), sr=44100)
        
    else:
        f0, voiced_flag, voiced_probs = pyin(y=buffer,
                                                fmin=note_to_hz('A0'),
                                                fmax=note_to_hz('C7'), sr=44100)
    
    end_librosa = time.time()
    
    print("Librosa took", end_librosa - start, "seconds")

    #replace NaN with 0s
    if (len(f0) > 0):
        f0 = np.nan_to_num(f0, nan=0, posinf=10000, neginf=-10000)
        
    #get the time each entry is recorded    
    times = times_like(f0, sr=44100)
    
    # notes = note_names_from_freqs(f0, 0) #C2 is the lowest note on a Cello (62 Hz)
    
    #The points in the array where a new note begins
    onset_frames = onset.onset_detect(y=buffer, sr=44100)
    
    #Notes based on onsets
    onset_freqs = []
    onset_notes = []
    onset_times = []
    for my_onset in onset_frames: 
        #not sure about the +3 here. But possibly onset is the note right before the switch
        #Adding 3 because frequency at onset isn't accurate
        #Definitely need something better
        if not my_onset + 3 >= len(f0):
            my_onset += 3
        freq = f0[my_onset]
        time_pos = times[my_onset]
        onset_freqs.append(freq)
        onset_times.append(time_pos)
    
    onset_notes = note_names_from_freqs(onset_freqs, 0)
    
    #Calculate durations:
    
    durs = get_durations(onset_times, times[len(times) - 1])
    
    #Calculates RMS value of each entry
    
    if rms_graph:
        rms = feature.rms(y=buffer)
        graph_rms(rms[0])

    #Create dataframe
    
    # my_dict = {'Note Name': notes, 'Frequency': f0, 'Times': times}
    # df = pd.DataFrame(data=my_dict)
    # print(df)
    # df.to_csv("out.csv")
    
    my_dict = {'Note Name': onset_notes, 'Frequency': onset_freqs, 'Start Time': onset_times, 'Duration': durs}
    note_names = list(my_dict['Note Name'])
    note_names = [note.replace('♯', '#') for note in note_names]
    my_dict['Note Name'] = note_names
    df = pd.DataFrame(data=my_dict)
    

    df.to_csv("out.csv")
    # print(df)

    offs = []
    for i in range (len(df['Note Name'])):
        note_str = df['Note Name'][i]
        note = note_str
        offset = 0
        if (note_str != "rest"):
            print(note_str)
            note, offset = [x for x in re.split('([A-G][#-]?[0-9]+)([-+][0-9]+)', note_str) if x]
            if offset[0] == '+':
                offset = int(offset[1:])
            else:
                offset = int(offset)
            df.loc[i, 'Note Name'] = note
            offs.append(offset)
        else:
            df.drop(i, inplace=True)
        
    df.insert(4, "Cents", offs)
 
    print(df)

    # end = time.time()
    
    # testing = AudioAnalysis(df, "C:\\Users\\brian\\Desktop\\VIP\\Evaluator-code\\src\\score\\cscale.xml")
   
    # testing.generate_overlay_score()

    # end2 = time.time()

    # print("Calculation took " + str(end - start) + " seconds. Comparison took " + str(end2 - end) + " seconds.\n")

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
        #print("librosa freq: ", f0)
        #print("librosa out: ", notes)
    return notes

'''
Given times at each index and time of ended recording, return duration of each note
'''
def get_durations(onset_times, end_time):
    durs = []
    for i in range(len(onset_times)):
        if i < len(onset_times) - 1:
            durs.append(onset_times[i + 1] - onset_times[i])
        else:
            durs.append(end_time - onset_times[i])
    return durs

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
    