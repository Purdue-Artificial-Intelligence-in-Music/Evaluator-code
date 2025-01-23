import React, { useState } from 'react';
import { Button, View, StyleSheet, Text } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { ResizeMode, Video } from 'expo-av'; // Import Video
import { CameraView, useCameraPermissions } from 'expo-camera';

export default function App() {
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [videoDimensions, setVideoDimensions] = useState<{ width: number; height: number } | null>(null); // Store video dimensions
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

      // Fetch the video dimensions from the asset metadata
      const { width, height } = result.assets[0];
      setVideoDimensions({ width, height });

      // Close the camera when a video is selected
      setIsCameraOpen(false);
    }
  };

  function toggleCamera() {
    // Clear the video URI if camera is opened
    if (!isCameraOpen) {
      setVideoUri(null); // Clear video when switching to camera view
      setVideoDimensions(null); // Clear video dimensions when switching to camera
    }
    setIsCameraOpen(prev => !prev);
  }

  // Resize logic to maintain aspect ratio
  const containerWidth = 300;
  const containerHeight = 400;

  let videoWidth = containerWidth;
  let videoHeight = containerHeight;

  if (videoDimensions) {
    const aspectRatio = videoDimensions.width / videoDimensions.height;

    // If the video is wider than the container, scale it based on width
    if (aspectRatio > containerWidth / containerHeight) {
      videoWidth = containerWidth;
      videoHeight = containerWidth / aspectRatio;
    } else {
      // If the video is taller than the container, scale it based on height
      videoHeight = containerHeight;
      videoWidth = containerHeight * aspectRatio;
    }
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
          resizeMode={ResizeMode.COVER}
          style={{width: 300, height:300}}
        />
      ) : (
        <Text style={styles.placeholderText}>No video selected</Text>
      )}

      {videoDimensions && (
        <Text style={styles.videoDimensionsText}>
          Video Dimensions: {videoDimensions.width}x{videoDimensions.height}
        </Text>
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
    width: 300, // Set width of the camera view to create a rectangle
    height: 400, // Set height to make it a rectangle
    marginBottom: 20,
    borderRadius: 10, // Optional: Adds rounded corners to the camera view
  },
  video: {
    alignSelf: 'center', // Ensure the video is centered (can be removed, as we use absolute positioning)
  },
  placeholderText: {
    color: '#000',
    fontSize: 16,
    marginTop: 20,
  },
  toggleButton: {
    marginBottom: 20,
  },
  videoDimensionsText: {
    marginTop: 10,
    fontSize: 16,
    color: '#333',
  },
});
