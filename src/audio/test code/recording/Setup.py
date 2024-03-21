import os

dirname = os.path.dirname(__file__)

# Setup folder to make the Analysis csv files
analysisFolder = os.path.join(dirname, 'Analysis')
if ((os.path.exists(analysisFolder)) == False):         # Check if folder does not exist before creating one (prevents overwriting contents in folder)
    os.makedirs(analysisFolder)

# Setup folder to make the input wav file
audioFolder = os.path.join(dirname, 'Audio Input')
if ((os.path.exists(audioFolder)) == False):             # Check if folder does not exist before creating one (prevents overwriting contents in folder)
    os.makedirs(audioFolder)

transcribedFolder = os.path.join(dirname, 'Transcribed')
if ((os.path.exists(transcribedFolder)) == False):      # Check if folder does not exist before creating one (prevents overwriting contents in folder)
    os.makedirs(transcribedFolder)