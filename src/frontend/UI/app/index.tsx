import React, { useState, useEffect, useRef } from 'react';
import { SafeAreaView, Button, Text, Image, StyleSheet, View, Dimensions } from 'react-native';

import { ResizeMode, Video } from 'expo-av';
import * as ImagePickerExpo from 'expo-image-picker';
import { Camera, CameraView } from 'expo-camera';

import { Svg, Circle, Line} from 'react-native-svg';

import * as Network from 'expo-network';

type Point = {
  x: number;
  y: number;
};

type ResponseData = {
  [key: string]: { x: number; y: number };
};

const windowWidth = Dimensions.get('window').width;
const windowHeight = Dimensions.get('window').height;
const aspectRatio = windowWidth + "x" + windowHeight;
// console.log(aspectRatio)

export default function App() {
  const [videoDimensions, setVideoDimensions] = useState<{ width: number; height: number } | null>(null);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [ipAddress, setIpAddress] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [hasPermission, setHasPermission] = useState(false);  // State for camera permission
  
  const cameraComponentDelays = [0, 500]

  
  const [points, setPoints] = useState([{x: 0, y: 0}]); // FOR THE POINTS WE NEED TO RENDER ON THE SCREEN
  const [linePoints, setLinePoints] = useState([{start: {x: 0, y: 0}, end: {x: 0, y: 0}}]);
  const intervalRef = useRef<NodeJS.Timeout>();
  const [recording, setRecording] = useState<Boolean>(false);

  const [supinating, setSupinating] = useState<String>("none");
  
  

  // CameraComponent to handle camera view
interface CameraComponentProps {
  startDelay: number;
}

const CameraComponent: React.FC<CameraComponentProps> = ({ startDelay }) => {
  const cameraRef = useRef<Camera | null>(null); // Ref to the Camera component

  const takePicture = async () => {
    console.log("picture taken", startDelay);
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync();
      sendImageToBackend(photo.base64 || '');
      // console.log("photo dimensions: ", photo.width, photo.height);
    }
  };

  useEffect(() => {
    if (recording && !loading) {
      setTimeout(() => {
        intervalRef.current = setInterval(() => {
          takePicture();
        }, 1000);
      }, startDelay)
    } else {
      clearInterval(intervalRef.current);
    }

    return () => clearInterval(intervalRef.current);
  }, [recording, loading]);

  return (
    <View style={styles.cameraContainer}>
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        pictureSize={aspectRatio}
        mirror={true}
        onCameraReady={() => setLoading(false)}
      />
      <Button title="RECORD" onPress={() => setRecording(!recording)} />
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

  function processPoints(responseData: ResponseData) {
    let newLines = [{start: responseData["box bow top left"], end: responseData["box bow top right"]},
                    {start: responseData["box bow top right"], end: responseData["box bow bottom right"]},
                    {start: responseData["box bow bottom right"], end: responseData["box bow bottom left"]},
                    {start: responseData["box bow bottom left"], end: responseData["box bow top left"]},

                    {start: responseData["box string top left"], end: responseData["box string top right"]},
                    {start: responseData["box string top right"], end: responseData["box string bottom right"]},
                    {start: responseData["box string bottom right"], end: responseData["box string bottom left"]},
                    {start: responseData["box string bottom left"], end: responseData["box string top left"]},
                  ];

    setLinePoints(newLines);
    console.log(newLines)
    setPoints(Object.values(responseData))
    console.log("pts")
  }

  

  const returnBack = async () => {
      setIsCameraOpen(false)
      setVideoUri(null)
  };
  // Send captured image to backend API
  const sendImageToBackend = async (imageBase64: string) => {
    const jsonData = {
      "title": "Test Image",
      "content": "This is a test image",
      "image": imageBase64,
    }

    try {
      setLoading(true);
      //django used /api/upload
      const response = await fetch('http://127.0.0.1:8000/upload/', {
        method: 'POST',
        body: JSON.stringify(jsonData),
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        // console.log(response)

        const responseData: ResponseData = await response.json(); // Type casting here

        // console.log('Response Data:', responseData);
        // console.log("All values in list: ", Object.values(responseData));

        processPoints(responseData)
        console.log('Points:', points);

        
        setSupinating(responseData["supination"].toString())
      }

    } catch (error) {
      console.error('Error uploading image:', error);
      alert('Error: Failed to upload image.');

    } finally {
      setLoading(false);
    }
  };

  async function demoVideo() {
    const jsonData = {
      "title": "demo video",
      "videouri": "this is a test",
    }
    const response = await fetch('http://127.0.0.1:8000/api/upload/', {
      method: 'POST',
      body: JSON.stringify(jsonData),
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }
  
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.buttonStyle}>
      <Button title="Choose Video" onPress={pickVideo}/>
      <Button title={isCameraOpen ? 'Close Camera' : 'Open Camera'} onPress={() => {setIsCameraOpen(!isCameraOpen); setVideoUri(null)} } />
      <Button title="Fetch Data from API" disabled={loading} onPress={demoVideo} />
      <Button title="Back" onPress={returnBack}/>
      </View>

      {(isCameraOpen && hasPermission) ? (
       cameraComponentDelays.map((delay, index) => (
          <CameraComponent startDelay={delay} key={index}/>
       ))
      ) : (<></>)
      }    
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

      <Svg style={{ ...styles.cameraContainer, height: 440 - 20 }}>
        {points.map((item, index) => (
          <Circle r={5} 
                  cx={item.x} 
                  cy={item.y} 
                  fill={"rgb(" + (255 - index * 30) +"," + index * 30 + "," + (255 - index * 30) + ")"} 
                  key={index}/>
        ))}
        {linePoints.map((item, index) => (
          <Line x1={item.start.x} 
                y1={item.start.y} 
                x2={item.end.x}
                y2={item.end.y}
                strokeWidth={1}
                stroke="red"
                key={index}
          />
        ))}
      </Svg>
        
      
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
    flex: 1,
    position: 'absolute',
    marginVertical: 150,
    width: 640,
    height: 440,
    marginBottom: 20,
    borderRadius: 10,
    backgroundColor: 'transparent',
    justifyContent: 'center',
    alignItems: 'center',
  },
  point: {
    flex: 1,
    position: 'absolute',
    width:5,
    height: 5,
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
