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
5. Build and Run on Android
  ```bash
  npx expo run:android # This may take a while
  ```
6. Testing: press "Fetch data" button, you should see "Status: Video Analyzer Module Connected"


## Overview
- Frontend file: app/index.tsx
- Native Module: modules/expo-video-analyzer
