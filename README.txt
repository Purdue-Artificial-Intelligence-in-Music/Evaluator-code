A few notes about AudioToScore.py

- In order to display the score and be able to play audio from it, MuseScore4 must be installed.
    - https://musescore.org/en
-On Line 2 and Line 3, the 2nd parameter of enviroment.set requires the path to MuseScore4.exe
    - MuseScore4.exe can be found in 'Musescore install directory'/MuseScore/bin/MuseScore4.exe

- UseScale can be a parameter which could affect the accuracy of the generated score when compared to the original score
    - More info here: https://web.mit.edu/music21/doc/moduleReference/moduleScale.html

- Taking input from an audio file:
    - 'useMic' parameter in the runTranscribe('parameters') function must be: useMic = False
        - To take input from the mic, useMic == True

    - 'saveFile' parameter in the runTranscribe('parameters') function must be: saveFile = True
        - This will allow the program to transcribe the audio file specificed by setting a parameter (more below)
        - if saveFile = False, then recorded audio is saved to the disk.
    - saveFile argument:
        - Line 34, variable WAVE_FILENAME should contain the path to the audio file which you want transcribed

- In summary:
    - When transcribing from an audio file, input parameters 'useMic = False' and 'saveFile = True'
        - Make sure to put the path to your audio file in WAVE_FILENAME (Line 34)

    - When transcribing in real time such as from a microphone, input parameters 'useMic =True' and 'saveFile = False'
        - The parameter 'seconds = ___' is considered when useMic = True. This determines the duration of time the program will use
          your microphone to generate the transcribed score 

