## Prerequisites
### Required Software

1. **Node.js & npm**
- [Node.js Official Website](https://nodejs.org/en/download); [npm Documentation](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)
- **Verify installation**:
  ```bash
  node --version
  npm --version
  ```


2. **Java Development Kit (JDK 17)**
- Download: OpenJDK (or Oracle JDK)
- Verify installation:
  ```bash
  java -version  # Should show Java 17.x.x
  ```


3. **Android Studio with Android SDK (For testing on Android)**
- Download: Android Studio Official Website

### Android Setup

- Setup Android Studio and Android SDK (API level 33 or higher)
- Set up Android Virtual Device (AVD) *or* connect to physical device
- For physical device: Enable Developer Options and USB Debugging

## Setup Instructions
1. Git pull from ```UI_branch_new```
2. Install packages by running ```npm install```
3. Connect to Android device - Start Virtual device or connect to physical device via USB
   
   Run ```adb devices``` to verify connection

## NPU + QNN SETUP (works with emulator)
- Download the entire Qualcomm AI SDK from here https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
- In modules\camerax\android\build.gradle, replace flatDir{} path 
with the path of aarch64-android in your Qualcomm AI SDK folder.
Path in SDK folder should be roughly "QualcommSDK/qairt/2.38.0.250901/lib/aarch64-android", but verify.
- create two folders "jniLibs" and "arm64-v8a" within that folder if they don't exist in the
folder path "modules/camerax/android/src/main/jniLibs"
- Download all files except the ones in the model folder from here  https://drive.google.com/drive/folders/1qA3KZxN9eVqKu0i0XrEHumIE49JOg6mx?usp=sharing
- Move all files into arm64-v8a folder
- IMPORTANT: Everything should be set, after running the first successful build 
delete all files within arm64-v8a (but not the folder), because Android Studio
creates a copy in a cache directory and will get confused if you build again
without deleting the ones in arm64-v8a because there's two of them.

## Download Models
- Download tflite models from here: https://drive.google.com/drive/u/2/folders/15l-dIwOizXFY3Cfk1p3PRpNmih1y8u_T
- Move them to modules/camerax/android/src/main/assets in the project explorer

## Pushing to Git
- Our project uses Firebase to distribute stable apks to user
- Don't need to do anything in order to run the software
- For pushing changes to Git will require certain Firebase files only members have access to
  - Get assistance from AIM member to acquire files and move them to android/app in the project.

## Run Application
5. Build and Run on Android
  ```bash
  npx expo run:android # This may take a while
  ```
6. You should see 3 buttons on the home page. Try testing with "Open Camera" or "Choose Video".


## Overview
- Frontend file: app/index.tsx (home page), app/CameraComponent.tsx (camera page)
- Native Module: modules/expo-video-analyzer (for video processing), modules/camerax (for live inference)
