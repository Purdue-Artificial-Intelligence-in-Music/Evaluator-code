from music21 import *
musescoreDirectPNGPath : environment.set("musescoreDirectPNGPath",  "YOUR PATH TO 'MuseScore4.exe") 
musicxmlPath: environment.set("musicxmlPath", "YOUR PATH TO 'MuseScore4.exe'")

def runTranscribe(show=True, plot=True, useMic=True,
                  seconds=15, useScale=None, saveFile=True):
    '''
    runs all the methods to record from audio for `seconds` length (default 10.0)
    and transcribe the resulting melody returning a music21.Score object
    
    if `show` is True, show the stream.  
    
    if `plot` is True then a Tk graph of the frequencies will be displayed.
    
    if `useMic` is True then use the microphone.  If False it will load the file of `saveFile`
    or the default temp file to run transcriptions from.
        
    a different scale besides the chromatic scale can be specified by setting `useScale`.
    See :ref:`moduleScale` for a list of allowable scales. (or a custom one can be given).
    Microtonal scales are totally accepted, as are retuned scales where A != 440hz.

    if `saveFile` is False then then the recorded audio is saved to disk.  If
    set to `True` then `environLocal.getRootTempDir() + os.path.sep + 'ex.wav'` is
    used as the filename.  If set to anything else then it will use that as the
    filename. 
    '''
    from music21 import audioSearch as audioSearchBase

    if useScale is None:
        useScale = scale.ChromaticScale('C4')
    #beginning - recording or not
    if saveFile != False:
        if saveFile == True:
            WAVE_FILENAME = 'YOUR PATH TO YOUR AUDIO FILE'
        else:
            WAVE_FILENAME = saveFile
    else:
        WAVE_FILENAME = False
    
    # the rest of the score
    if useMic is True:
        freqFromAQList = audioSearchBase.getFrequenciesFromMicrophone(length=seconds, storeWaveFilename=WAVE_FILENAME)
    else:
        freqFromAQList = audioSearchBase.getFrequenciesFromAudioFile(waveFilename=WAVE_FILENAME)
        
    detectedPitchesFreq = audioSearchBase.detectPitchFrequencies(freqFromAQList, useScale)
    detectedPitchesFreq = audioSearchBase.smoothFrequencies(detectedPitchesFreq)
    (detectedPitchObjects, listplot) = audioSearchBase.pitchFrequenciesToObjects(detectedPitchesFreq, useScale)
    (notesList, durationList) = audioSearchBase.joinConsecutiveIdenticalPitches(detectedPitchObjects)
    myScore, unused_length_part = audioSearchBase.notesAndDurationsToStream(notesList, durationList, removeRestsAtBeginning=True)    

    # This shows the score in MuseScore
    if show == True:
        myScore.show()        
    
    if plot == True:
        try:
            import matplotlib.pyplot # for find
        except ImportError:
            raise audioSearchBase.AudioSearchException("Cannot plot without matplotlib installed.")
        matplotlib.pyplot.plot(listplot)
        matplotlib.pyplot.show()
    print("END")    
        
    return myScore

# This runs the function with specificed parameters
runTranscribe(show=True, plot=True, useMic=False,
                  seconds=20.0, useScale=None, saveFile=True)