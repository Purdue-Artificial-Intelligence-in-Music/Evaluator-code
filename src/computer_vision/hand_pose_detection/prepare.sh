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
  curl 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' \
    -o hand_landmarker.task
  curl 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task' \
    -o pose_landmarker.task
fi