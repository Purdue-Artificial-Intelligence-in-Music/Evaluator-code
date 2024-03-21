import numpy as np
import os

# Parameters to determine duration of a 16th note
#-----------------------------------------#
#   CHANGE THE BPM ACCORDING TO THE SCORE #
#-----------------------------------------#
BPM = 115                   # user defined
lengthOf16th = 15 / BPM     # fixed formula

# current directory
dirname = os.path.dirname(__file__)

# load in data of note and times
noteAndTimesFull = np.loadtxt(os.path.join(dirname, 'Analysis/noteAndTimesFull.csv'), delimiter= ",", dtype='str', encoding='utf-8')
noteAndRhythm = []
noteAndDuration = []


current_note = None
start_time = 0
end_time = 0
restCheck = True    #restCheck is continuously on, ready for whenever the confidence drops below 90%


for note, duration, confidence in noteAndTimesFull:
    current_duration = float(duration)
    current_confidence = float(confidence)

    # catches first low confidence time
    if (current_confidence < 0.75 and restCheck):
        current_note = "rest"
        start_time = current_duration
        restCheck = False   # restCheck false so start_time won't be overwritten
    # caatches repeated low confidence
    elif (current_confidence < 0.75):
        if current_duration > end_time:
            end_time = current_duration

    # catches new notes
    elif current_note is None:
        # First note encountered
        current_note = note
        start_time = current_duration
    elif current_note == note:
        # Consecutive occurrence of the same note
        if current_duration > end_time:
            end_time = current_duration
    else:
        # Different note encountered
        if  end_time > start_time:
                end_time = current_duration
                deltat = end_time - start_time

                #numSixteenths = round(deltat / lengthOf16th, 3)
                #noteAndRhythm.append([current_note, numSixteenths])
                if not(restCheck):
                    noteAndDuration.append(["rest", round(deltat, 3)])
                else:
                    noteAndDuration.append([current_note, round(deltat, 3)])
                print("start_time: " + str(start_time))
                print("end_time: " + str(end_time))
    
        current_note = None
        start_time = end_time 
        end_time = 0
        restCheck = True
# Check if the last note is part of the result and has a non-zero duration
if start_time is not None and end_time >= start_time:
    deltat = end_time - start_time
    #numSixteenths = round(deltat / lengthOf16th, 3)
    #noteAndRhythm.append([current_note, numSixteenths])
    noteAndDuration.append([current_note, round(deltat, 3)])

    
print(noteAndDuration)

# Convert the result to a 2D NumPy array
result_array = np.array(noteAndRhythm)
np.savetxt(os.path.join(dirname, 'Transcribed', 'noteAndNum16ths.csv'), noteAndRhythm, fmt='%s', delimiter=', ', encoding='utf-8', header = 'note, time')
#np.savetxt(os.path.join(dirname, 'Transcribed', 'noteAndDurations.csv'), noteAndDuration, fmt='%s', delimiter=', ', encoding='utf-8', header = 'note, time')