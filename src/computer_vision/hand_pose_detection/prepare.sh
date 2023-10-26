#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd ../../..
if [ ! -d models ]; then
  mkdir models || exit 1
fi
cd models || exit 2
if [ -f hand_landmarker.task ]; then
  echo "The model already exists";
  exit 0;
else
  wget 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' \
    -O hand_landmarker.task
fi