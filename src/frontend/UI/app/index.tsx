import React, { useState, useEffect } from 'react';
import { Button, View, StyleSheet, Text } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { ResizeMode, Video } from 'expo-av';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Network from 'expo-network';

export default function App() {
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [videoDimensions, setVideoDimensions] = useState<{ width: number; height: number } | null>({ width: 300, height: 300 });
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [permission, requestPermission] = useCameraPermissions();
  const [apiData, setApiData] = useState<string | null>(null);
  const [loading, setLoading] = useState(false); // Loading state for the API request
  const [ipAddress, setIpAddress] = useState<string | null>(null);

  // Fetch IP address on mount
  useEffect(() => {
    const fetchIp = async () => {
      try {
        const ip = await Network.getIpAddressAsync();
        setIpAddress(ip);
      } catch (error) {
        console.error("Error fetching IP address:", error);
      }
    };

    fetchIp();
  }, []); // Empty dependency array means this runs only once when the app loads

  if (!permission) {
    return <View />; // Return a blank view while permission is loading
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>We need your permission to show the camera</Text>
        <Button onPress={requestPermission} title="Grant Permission" />
      </View>
    );
  }

  const pickVideo = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (permissionResult.granted === false) {
      alert('Permission to access media library is required!');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
    });

    if (!result.canceled) {
      const selectedVideoUri = result.assets[0].uri;
      const { width, height } = result.assets[0];
      setVideoDimensions({ width, height });
      setVideoUri(selectedVideoUri);
      setIsCameraOpen(false);
    }
  };

  const returnBack = async () => {

      setIsCameraOpen(false)
      setVideoUri(null)


  };

  const fetchDataFromAPI = async () => {
    if (!ipAddress) {
      console.log("IP address not available yet.");
      alert('IP address not available yet.');
      return;
    }

    setLoading(true);
    try {
      console.log("Making API request...");
      console.log(`http://${ipAddress}:8000/api/hello`);
      //assuming that django is running locally as well
      const response = await fetch(`http://localhost:8000/api/hello`, {
        method: 'GET',  
      });
      const json = await response.json();
      console.log("API Response:", json);
      setApiData(JSON.stringify(json));
    } catch (error) {
      console.error("Error fetching data:", error);
      setApiData('Failed to fetch data');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.buttonStyle}>
      <Button title="Choose Video" onPress={pickVideo}/>
      <Button title={isCameraOpen ? 'Close Camera' : 'Open Camera'} onPress={() => {setIsCameraOpen(!isCameraOpen); setVideoUri(null)} } />
      <Button title="Fetch Data from API" onPress={fetchDataFromAPI} />
      <Button title="Back" onPress={returnBack}/>
      </View>

      {isCameraOpen && (
        <CameraView style={styles.camera} facing="front" />
      )}

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

      {loading ? (
        <Text style={styles.apiDataText}>Fetching data...</Text>
      ) : apiData ? (
        <Text style={styles.apiDataText}>API Response: {apiData}</Text>
      ) : null}
    </View>
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
  message: {
    textAlign: 'center',
    paddingBottom: 10,
  },
  camera: {
    width: 300,
    height: 400,
    marginBottom: 20,
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
  apiDataText: {
    marginTop: 20,
    fontSize: 14,
    color: 'blue',
  },
});
