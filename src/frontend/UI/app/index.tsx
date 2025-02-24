import React, { useState, useEffect, useCallback, useRef } from 'react';
import { SafeAreaView, Button, Text, StyleSheet, View } from 'react-native';
import { ResizeMode, Video } from 'expo-av';
import * as ImagePicker from 'react-native-image-picker';
import * as ImagePickerExpo from 'expo-image-picker';
import * as Network from 'expo-network';
import { Camera, CameraView } from 'expo-camera';


// Define Point type for clarity
type Point = {
  x: number;
  y: number;
};

export default function App() {
  const [videoDimensions, setVideoDimensions] = useState<{ width: number; height: number } | null>(null);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [ipAddress, setIpAddress] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [pickerResponse, setPickerResponse] = useState<ImagePicker.ImagePickerResponse | null>(null);
  const [hasPermission, setHasPermission] = useState(false);  // State for camera permission
  const cameraRef = useRef<React.RefObject<typeof Camera>>(null); // Ref to the Camera component

  // CameraComponent to handle camera view
const CameraComponent = ({ cameraRef }: { cameraRef: React.RefObject<Camera> }) => {
  const [photoUri, setPhotoUri] = useState<string | null>(null);

  const takePicture = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync();
      sendImageToBackend(photo.base64);
      setPhotoUri(photo.uri);
      console.log('Photo URI:', photo.uri);
    }
  };

  return (
    <View style={styles.cameraContainer}>
      <CameraView ref={cameraRef} style={styles.camera} />
      <Button title="Take Picture" onPress={takePicture} />
      {photoUri && <Text>Photo taken! URI: {photoUri}</Text>}
    </View>
  );
};

  // Fetch IP address on mount
  useEffect(() => {
    const fetchIpAddress = async () => {
      try {
        const ip = await Network.getIpAddressAsync();
        setIpAddress(ip);
      } catch (error) {
        console.error("Error fetching IP address:", error);
      }
    };
    fetchIpAddress();

    // Request camera permission
    const getCameraPermission = async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    };
    getCameraPermission();
  }, []);

  // Function to handle video selection
  const pickVideo = async () => {
    const permissionResult = await Camera.requestCameraPermissionsAsync();

    if (!permissionResult.granted) {
      alert('Permission to access media library is required!');
      return;
    }

    const result = await ImagePickerExpo.launchImageLibraryAsync({
      mediaTypes: ['videos']

    });

    if(!result.canceled && result.assets && result.assets[0]) {
      const selectedVideoUri = result.assets[0].uri;
      const { width = 0, height = 0 } = result.assets[0];
      setVideoDimensions({ width, height });
      if (selectedVideoUri) {
        setVideoUri(selectedVideoUri);
      }
      setIsCameraOpen(false);
    }

  };

  const returnBack = async () => {

      setIsCameraOpen(false)
      setVideoUri(null)

  };
  // Send captured image to backend API
  const sendImageToBackend = async (imageBase64: string) => {
    console.log(imageBase64)
    const jsonData = {
      "title": "Test Image",
      "content": "This is a test image",
      "image": imageBase64,
    }

    try {
      setLoading(true);
      const response = await fetch('http://127.0.0.1:8000/api/upload/', {
        method: 'POST',
        body: JSON.stringify(jsonData),
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        console.log(response)
        //const pointArray: Point[] = await response.json();
        //console.log('Points:', pointArray);
      }

    } catch (error) {
      console.error('Error uploading image:', error);
      alert('Error: Failed to upload image.');

    } finally {
      setLoading(false);
    }
  };
  
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.buttonStyle}>
      <Button title="Choose Video" onPress={pickVideo}/>
      <Button title={isCameraOpen ? 'Close Camera' : 'Open Camera'} onPress={() => {setIsCameraOpen(!isCameraOpen); setVideoUri(null)} } />
      <Button title="Fetch Data from API" disabled={loading} />
      <Button title="Back" onPress={returnBack}/>
      </View>

      {isCameraOpen && hasPermission && <CameraComponent cameraRef={cameraRef} />}
      
      <Text>IP Address: {ipAddress || 'Fetching IP...'}</Text>

      {videoUri ? (
        <Video
          source={{ uri: videoUri }}
          shouldPlay
          resizeMode={ResizeMode.COVER}
          style={{ width: videoDimensions?.width, height: videoDimensions?.height }}
        />
      ) : (
        <Text style={styles.placeholderText}>No video selected</Text>
      )}

      {videoDimensions && (
        <Text style={styles.videoDimensionsText}>
          Video Dimensions: {videoDimensions.width}x{videoDimensions.height}
        </Text>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'flex-start',
    alignItems: 'center',
    backgroundColor: '#FFF',
  },
  buttonStyle: {

    flexDirection: 'row',
    justifyContent: 'space-evenly',
    alignItems: 'center',
    paddingTop: 15,
    paddingBottom: 15,
    backgroundColor: '#AAA',
    width: '100%',

  },
  cameraContainer: {
    width: 300,
    height: 400,
    marginBottom: 20,
    borderRadius: 10,
    backgroundColor: 'lightgray',
    justifyContent: 'center',
    alignItems: 'center',
  },
  camera: {
    width: '100%',
    height: '100%',
    borderRadius: 10,
  },
  placeholderText: {
    color: '#000',
    fontSize: 16,
    marginTop: 20,
  },
  videoDimensionsText: {
    marginTop: 10,
    fontSize: 16,
    color: '#333',
  },
});

