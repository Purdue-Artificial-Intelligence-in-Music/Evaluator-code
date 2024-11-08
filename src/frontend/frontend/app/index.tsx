import React, { useState } from 'react';
import { Button, View, StyleSheet, Text } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { ResizeMode, Video } from 'expo-av'; // Import Video
import { CameraView, useCameraPermissions } from 'expo-camera';

export default function App() {
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false); // state to toggle camera visibility
  const [permission, requestPermission] = useCameraPermissions();

  if (!permission) {
    // Camera permissions are still loading.
    return <View />;
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet.
    return (
      <View style={styles.container}>
        <Text style={styles.message}>We need your permission to show the camera</Text>
        <Button onPress={requestPermission} title="Grant Permission" />
      </View>
    );
  }

  const pickVideo = async () => {
    // Request permission to access the media library
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (permissionResult.granted === false) {
      alert('Permission to access media library is required!');
      return;
    }

    // Launch the image picker to allow the user to select a video
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
    });

    if (!result.canceled) {
      // Store the selected video URI
      const selectedVideoUri = result.assets[0].uri;
      setVideoUri(selectedVideoUri);

      // Close the camera when a video is selected
      setIsCameraOpen(false);
    }
  };

  function toggleCamera() {
    // Clear the video URI if camera is opened
    if (!isCameraOpen) {
      setVideoUri(null); // Clear video when switching to camera view
    }
    setIsCameraOpen(prev => !prev);
  }

  return (
    <View style={styles.container}>
      {/* Buttons to toggle between live camera and video selection */}
      <Button title="Choose Video" onPress={pickVideo} />
      <Button
        title={isCameraOpen ? 'Close Camera' : 'Open Camera'}
        onPress={toggleCamera}
      />

      {/* Render the camera feed if it's open */}
      {isCameraOpen && (
        <CameraView style={styles.camera} facing="front">
          {/* Camera feed will show here */}
        </CameraView>
      )}

      {/* Render the selected video if a video URI is set */}
      {videoUri ? (
        <Video
          source={{ uri: videoUri }}
          shouldPlay
          isLooping
          resizeMode={ResizeMode.CONTAIN} // Ensure the video is contained within the box
          style={styles.video}
        />
      ) : (
        <Text style={styles.placeholderText}>No video selected</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  message: {
    textAlign: 'center',
    paddingBottom: 10,
  },
  camera: {
    width: 300,  // Set width of the camera view to create a rectangle
    height: 400, // Set height to make it a rectangle
    marginBottom: 20,
    borderRadius: 10, // Optional: Adds rounded corners to the camera view
  },
  video: {
    width: 300,  // Fixed width for the video
    height: 200, // Fixed height for the video
    alignSelf: 'center',  // Ensure the video is centered
  },
  placeholderText: {
    color: '#000',
    fontSize: 16,
    marginTop: 20,
  },
  toggleButton: {
    marginBottom: 20,
  },
});
