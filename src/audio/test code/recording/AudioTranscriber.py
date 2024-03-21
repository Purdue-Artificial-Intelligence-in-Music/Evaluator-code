import crepe
from scipy.io import wavfile
import numpy as np
import librosa
import os

# current directory
dirname = os.path.dirname(__file__)

# read in data and get frequencies
#--------------------------------------------------------------------------#
#  PUT IN THE AUDIO FILE NAME HERE WITH THE FILE EXTENSION (i.e audio.wav) #
#--------------------------------------------------------------------------#
audioFileName = "violin.wav"

sr, audio = wavfile.read(os.path.join(dirname, 'Audio Input', audioFileName))

# change step_size to be lower for greater precision
time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True, step_size = 30)
a = np.column_stack((time, frequency, confidence))

# Save entire analysis
np.savetxt(os.path.join(dirname, 'Analysis', 'FullAnalysis.csv'), a,
              ['%.3f', '%.3f', '%.6f'],
              header='time,frequency,confidence',delimiter=',')

# Take only confident frequencies
filteredFreqs = a[np.where(a[:, 2] > 0.80)]

np.savetxt(os.path.join(dirname, 'Analysis', 'ConfidentAnalysis.csv'), filteredFreqs,
              ['%.3f', '%.3f', '%.6f'],
              header='time,frequency,confidence',delimiter=',')

# 1D array of just frequencies
frequencies_arr = filteredFreqs[:, 1]

#1d array of frequencies from full analysis
frequencies_arr2 = a[:, 1]

# Arrays for time and notes (filtered)
time_arr = filteredFreqs[:, 0]
notes_arr = np.empty((0, 0))

# arrays for time and notes (unfiltered)
time_arr2 = a[:, 0]
notes_arr2 = notes_arr

# Convert frequencies into notes and put in notes_arr
for freq in frequencies_arr:
    note = librosa.hz_to_note(freq)
    notes_arr = np.append(notes_arr, note)

for freq in frequencies_arr2:
    note = librosa.hz_to_note(freq)
    notes_arr2 = np.append(notes_arr2, note)

# Combines arrays into a 2D array with 2 columns
noteAndTimes = np.column_stack((notes_arr, time_arr))
noteAndTimes2 = np.column_stack((notes_arr2, time_arr2, a[:, 2]))

# Save notes and time array
np.savetxt(os.path.join(dirname, 'Analysis', 'noteAndTimes.csv'), noteAndTimes, fmt='%s', delimiter=', ', encoding='utf-8', header = 'note, time')
np.savetxt(os.path.join(dirname, 'Analysis', 'noteAndTimesFull.csv'), noteAndTimes2, fmt='%s', delimiter=', ', encoding='utf-8', header = 'note, time')

