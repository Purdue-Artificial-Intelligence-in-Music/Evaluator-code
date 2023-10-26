# Installs the hand_landmarker model
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "$SCRIPT_DIR"
cd ../../..
if [ ! -d models ]; then
  mkdir models || exit 1
fi
cd models || exit 2
wget 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' \
  -O hand_landmarker.task