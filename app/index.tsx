import React, { useState, useEffect, useRef } from 'react';
import * as VideoAnalyzer from '../modules/expo-video-analyzer/src/ExpoVideoAnalyzer';


import { TouchableOpacity} from 'react-native';
import ChooseVideoIcon from '../assets/images/ChooseVideo.png';
import CloseCamera from '../assets/images/CloseCamera.png';
import FetchData from '../assets/images/FetchData.png';
import OpenCamera from '../assets/images/OpenCamera.png';
import Back from '../assets/images/Back.png';
import Record from '../assets/images/Record.png';
import Recording from '../assets/images/Recording.png';
import SendVideo from '../assets/images/SendVideo.png';
import Download from '../assets/images/Download156x41.png';

import { useWindowDimensions } from 'react-native';
import { Platform } from 'react-native';

import * as MediaLibrary from 'expo-media-library';

import { SafeAreaView, Button, Text, Image, StyleSheet, View, Dimensions, ScrollView, ActivityIndicator, Alert } from 'react-native';

import { ResizeMode, Video } from 'expo-av';
import * as ImagePickerExpo from 'expo-image-picker';

import { Camera, useCameraDevice, useCameraPermission, useCameraFormat } from 'react-native-vision-camera';
import RNFS from 'react-native-fs';

import { Svg, Circle, Line} from 'react-native-svg';

import * as Network from 'expo-network';

import * as FileSystem from 'expo-file-system';

type Point = {
  x: number;
  y: number;
};

type ResponseData = {
  [key: string]: { x: number; y: number };
};

// const windowWidth = Dimensions.get('window').width;
// const windowHeight = Dimensions.get('window').height;
// const aspectRatio = windowWidth + "x" + windowHeight;

const width = Dimensions.get('window').width;
const height = Dimensions.get('window').height;
const aspectRatio = width + "x" + height;
console.log(width)
console.log(height)
let factor = 0.9;
let factorTwo = 1;
let factorThree = 0.9;
if (Platform.OS === 'web') {
  factor = 0.5;
  factorTwo = 0.7;
}


export default function HomePage() {
  const [videoDimensions, setVideoDimensions] = useState<{ width: number; height: number } | null>(null);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [ipAddress, setIpAddress] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [hasPermission, setHasPermission] = useState(false);  // State for camera permission



  const [points, setPoints] = useState([{x: 0, y: 0}]); // FOR THE POINTS WE NEED TO RENDER ON THE SCREEN
  const [linePoints, setLinePoints] = useState([{start: {x: 0, y: 0}, end: {x: 0, y: 0}}]);
  const intervalRef = useRef<number | null>(null);
  const [recording, setRecording] = useState<boolean>(false);
  const [sendButton, setsendButton] = useState(false)
  const [sendVideo, setsendVideo] = useState(false)
  const [videofile, setvideofile] = useState<string | null>(null);

  const [supinating, setSupinating] = useState<String>("none");
  
  const [imageWidth, setImageWidth] = useState<number | null>(0);
  const [imageHeight, setImageHeight] = useState<number | null>(0);
  const socketRef = useRef<WebSocket | null>(null);
  const isSending = useRef(false);


  const [analysisResult, setAnalysisResult] = useState<string>('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const testConnection = () => {
    try {
      const moduleStatus = VideoAnalyzer.getStatus();
      Alert.alert('Connection Test', `Status: ${moduleStatus}`);
    } catch (error) {
      Alert.alert('Connection Failed', `Error: ${error.message}`);
    }
  };

const requestCameraPermission = async () => {
  const status = await Camera.requestCameraPermission();
  if (status === 'granted') {
    setHasPermission(true);
  } else {
    setHasPermission(false);
  }
  // console.log("camera permission status: ", status);
};

type CameraComponentProps = {
  startDelay: number;
};

useEffect(() => {
  console.log('Points updated:', points);
}, [points]);

const CameraComponent: React.FC<CameraComponentProps> = ({ startDelay }) => {
  const device = useCameraDevice('back');
  if (device == null) {
    console.log("device = null");
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <Text>Loading camera device...</Text>
      </View>
    );
  }
  const cameraRef = useRef<Camera>(null);
  const isTakingPhoto = useRef(false);
  // const format = useCameraFormat(device, [
  //   { photoResolution: { width: 640, height: 480 } }
  // ]);
 

  return (
    <View style={styles.cameraContainer}>
      <TouchableOpacity
          style={styles.closeCameraButton}
          onPress={() => {
          
            setRecording(false); 
            closeCamera();       // close the overlay
            
          }}
        
          activeOpacity = {1} // Prevents the button from being pressed when recording
        >
        <Image source={CloseCamera} style={styles.closeCameraButton} resizeMode="contain" />
      </TouchableOpacity>
      <View style={styles.cameraWrapper}>
        <Camera 
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        // style={{ marginTop: 50, width: width, height: height * 0.7 }}
        device={device}
        isActive={true}
        photoQualityBalance="speed"
        photo={true}
        video={false}
        // format={format}
        />
      
        <Svg
          viewBox={`0 0 ${imageWidth || 4080} ${imageHeight || 3060}`}
          preserveAspectRatio="xMidYMid slice"
          style={[StyleSheet.absoluteFill, { zIndex: 10 }]}
        >
          {recording && (
            <>
            {points.map((item, index) => (
            <Circle
              r={20}
              cx={item.x}
              cy={item.y}
              fill={`rgb(${255 - index * 30}, ${index * 30}, ${255 - index * 30})`}
              key={index}
            />
            ))}
            {linePoints.map((item, index) => (
              <Line
                x1={item.start.x}
                y1={item.start.y}
                x2={item.end.x}
                y2={item.end.y}
                strokeWidth={5}
                stroke="red"
                key={index}
              />
            ))}
            </>
          )}
        </Svg>
      </View>
      <Text style={styles.placeholderText}> Forearm posture: {supinating} </Text> 
      <TouchableOpacity
      onPress={() => {

        if (recording) {
          setRecording(false);
          // reset old dots and lines
          setPoints([{x: 0, y: 0}]);
          setLinePoints([{start: {x: 0, y: 0}, end: {x: 0, y: 0}}]);
        } else {
          setRecording(true);
        }
      }} >

      <Image
        source = {recording ? Recording : Record}
        style = {{width: 140, height: 40, marginTop: 10}}
      />
      </TouchableOpacity>
      
    </View>
  )
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
  }, []);

  // Function to handle video selection
  // base64 + url
  const pickVideo = async () => {
    try {
      const result = await ImagePickerExpo.launchImageLibraryAsync({
        mediaTypes: ['videos'],
        base64: true
      });

      if(!result.canceled && result.assets && result.assets[0]) {
        const selectedVideoUri = result.assets[0].uri;
        const { width = 0, height = 0 } = result.assets[0];
        setVideoDimensions({ width, height });
        setVideoUri(selectedVideoUri);

        
        setvideofile(selectedVideoUri);
        setIsCameraOpen(false);
        setsendButton(true);
      }
    } catch (error) {
      console.error('Error picking video:', error);
      alert('Failed to pick video. Please try again.');
    }
  };


  function processPoints(responseData: any) {
    const coordList = responseData.coord_list || {};
    const classificationList = responseData.classification_list || {};
  
    // transform box/box string points into coordinates {x, y}
    function arrToPoint(arr: number[] | undefined) {
      if (!arr || arr.length !== 2) return { x: 0, y: 0 };
      return { x: arr[0], y: arr[1] };
    }
  
    // get lines
    const newLines = [
      { start: arrToPoint(coordList["box bow top left"]), end: arrToPoint(coordList["box bow top right"]) },
      { start: arrToPoint(coordList["box bow top right"]), end: arrToPoint(coordList["box bow bottom right"]) },
      { start: arrToPoint(coordList["box bow bottom right"]), end: arrToPoint(coordList["box bow bottom left"]) },
      { start: arrToPoint(coordList["box bow bottom left"]), end: arrToPoint(coordList["box bow top left"]) },
  
      { start: arrToPoint(coordList["box string top left"]), end: arrToPoint(coordList["box string top right"]) },
      { start: arrToPoint(coordList["box string top right"]), end: arrToPoint(coordList["box string bottom right"]) },
      { start: arrToPoint(coordList["box string bottom right"]), end: arrToPoint(coordList["box string bottom left"]) },
      { start: arrToPoint(coordList["box string bottom left"]), end: arrToPoint(coordList["box string top left"]) },
    ];
    setLinePoints(newLines);
  
    const boxPoints = [
      "box bow top left", "box bow top right", "box bow bottom right", "box bow bottom left",
      "box string top left", "box string top right", "box string bottom right", "box string bottom left"
    ].map(key => arrToPoint(coordList[key]));

    const handPoints = Array.isArray(coordList["hand points"])
      ? coordList["hand points"].map(arrToPoint)
      : [];

    setPoints([...boxPoints, ...handPoints]);

    if (classificationList["wrist posture"]) {
      setSupinating(classificationList["wrist posture"]);
    }

    console.log("New points:", points, "; lines: ", linePoints);
    console.log("New supination: ", classificationList ? classificationList["wrist posture"] : 'None');
  }

  function modifyPoint(point: { x: number; y: number }, yFactor: number) {
    return { ...point, y: point.y * yFactor };
  }

  function correctPoints(responseData: ResponseData, yFactor: number): ResponseData {
    const correctedData = Object.entries(responseData).reduce((acc, [key, point]) => {
      acc[key] = modifyPoint(point, yFactor);
      return acc;
    }, {} as ResponseData);
  
    return correctedData;
  }

  

  const returnBack = async () => {
      setIsCameraOpen(false)
      setVideoUri(null)
      setsendButton(false)
      setAnalysisResult('')
  };
  // Send captured image to backend API
  const sendImageToBackend = async (imageBase64: string) => {
    if ((socketRef.current) && (socketRef.current.readyState === WebSocket.OPEN) && (!isSending.current)) {
      isSending.current = true;
      const message = {
        type: "frame",
        image: imageBase64,
      };
      socketRef.current.send(JSON.stringify(message));
    }
  };
  


  const sendVideoBackend = async () => {
    if (!videofile) {
      Alert.alert('Error', 'Please select a video');
      return;
    }

    setIsAnalyzing(true);
    try {
      console.log('try:', videofile);
      const result = await VideoAnalyzer.analyzeVideo(videofile);
      console.log('Result:', result);
      
      if (result.success) {
        Alert.alert('Success', `Video opened\nLength: ${result.duration}ms\nResolution: ${result.width}x${result.height}`);
        setAnalysisResult(JSON.stringify(result, null, 2));
      } else {
        Alert.alert('Error', `Video could not be opened: ${result.error}`);
      }
      
    } catch (error) {
      console.error('Failed to open video:', error);
      Alert.alert('Error', `Error: ${error.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  

  const downloadVideo = async (videoUrl: string) => {
    // Temporary part preventing crash on web platform
    if (Platform.OS === 'web') {
      alert("Video download is only supported on mobile devices.");
      return;
    }

    try {
      // Request permission
      const { status } = await MediaLibrary.requestPermissionsAsync();
      if (status !== 'granted') {
        alert("Permission to access media library is required.");
        return;
      }

      // Download video to local temporary folder
      const fileUri = FileSystem.documentDirectory + 'downloaded_video.mp4';
      const downloadResumable = FileSystem.createDownloadResumable(
        videoUrl,
        fileUri
      );

      const downloadResult = await downloadResumable.downloadAsync();

      if (!downloadResult || !downloadResult.uri) {
        throw new Error("Download failed");
      }
      const uri = downloadResult.uri;

      // Save to gallery
      await MediaLibrary.saveToLibraryAsync(uri);

      alert("Video has been saved to your gallery!");
    } catch (error) {
      console.error("Download error:", error);
      alert("Failed to download video.");
    }
  };

 
  
  const openCamera = () => {
    requestCameraPermission()
    setIsCameraOpen(true);
    setVideoUri(null);
    setVideoDimensions(null);
    setsendButton(false);
  };

  const closeCamera = () => {
    console.log("Closing camera");
    setIsCameraOpen(false);
    setRecording(false);
    setPoints([{ x: 0, y: 0 }]);
    setLinePoints([{ start: { x: 0, y: 0 }, end: { x: 0, y: 0 } }]);
    setVideoUri(null);
    setVideoDimensions(null);
    setsendButton(false);
  };
  return (
    <SafeAreaView style={styles.container}>


      <View style={styles.buttonStyle}>
      

      <TouchableOpacity
        onPress={pickVideo}
        style={{
          width: 320,    // smaller width to allow multiple buttons per row
          height: 60,
          margin: 5,     // margin to add space between buttons
        }}
        activeOpacity={0.7}
      >
      <Image
        source={ChooseVideoIcon}
        style={{ width: '100%', height: '100%' }}
        resizeMode="cover"
      />
    </TouchableOpacity>

      <TouchableOpacity
        onPress={() => { 
          console.log("Opening camera");
          if (!isCameraOpen) {
            openCamera();
          }
         
        }}
        activeOpacity={0.7}
        style={{
          width: 320,
          height: 60,
          marginVertical: 5,
        }}
      >
        <Image
          source={OpenCamera}
          style={{ width: '100%', height: '100%' }}
          resizeMode="cover"
        />
      </TouchableOpacity>

     <TouchableOpacity
      onPress={testConnection}
      disabled={loading}
      style={{
        width:320,
        height: 60,
        marginVertical: 5,
      }}
      activeOpacity={0.7}
    >
      <Image
        source={FetchData}
        style={{
          width: '100%',
          height: '100%',
        }}
        resizeMode="cover" // or 'contain' if you want to keep aspect ratio
      />
      </TouchableOpacity>
      
      <TouchableOpacity
      onPress={returnBack}
      disabled={loading}
      style={{
        width:320,
        height: 60,
        marginVertical: 5,
      }}
      activeOpacity={0.7}
    >
      <Image
        source={Back}
        style={{
          width: '100%',
          height: '100%',
        }}
        resizeMode="cover" // or 'contain' if you want to keep aspect ratio
      />
      </TouchableOpacity>

      </View>

      
      {(isCameraOpen && hasPermission) ? (
        <View style={styles.cameraOverlay}>
          <CameraComponent startDelay={0} />


        </View>
      ) : null}

      
      <Text style={styles.ipAddressText}>
        IP Address: {ipAddress || 'Fetching IP...'}
      </Text>



      {videoUri ? (
        <View
        //  contentContainerStyle={{ flexGrow: 1 }} 
        //  showsVerticalScrollIndicator={true}
        >
        <Video
          source={{ uri: videoUri }}
          shouldPlay
          resizeMode={ResizeMode.COVER}
          style={{
            width: videoDimensions ? videoDimensions.width * 0.2 : undefined,
            height: videoDimensions ? videoDimensions.height * 0.2 : undefined
          }}
        />
        </View>
      ) : (
        !sendVideo && <Text style={styles.placeholderText}>No video selected</Text>
      )}

      {videoDimensions && (
        <Text style={styles.videoDimensionsText}>
          Video Dimensions: {videoDimensions.width}x{videoDimensions.height}
        </Text>
      )}


      {sendVideo && <ActivityIndicator size="large" color="#0000ff" />} 

      <View style={{opacity: sendButton ? 1: 0}}>
        <TouchableOpacity
          onPress = {sendVideoBackend}
          style = {{
            alignItems: 'center',
            justifyContent: 'center',
            padding: 8,
          }}
        >
          <Image
            source = {SendVideo}
            style = {{width: 140, height: 40}}

           />
        </TouchableOpacity>
        
      </View>

      {/* download video */}
      {(videoUri && !sendButton) && <Button
        title="Download Video"
        onPress={() => downloadVideo(videoUri)}
        disabled={!videoUri}
      />}

      {/* only show dots and lines when camera is open and recording*/}
      {/* {(isCameraOpen && recording) && <Svg style={{ ...styles.cameraContainer, height: 440 - 20 }}>
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
      </Svg>} */}
        
      
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
    flexDirection: 'column',
    justifyContent: 'flex-start',
    alignItems: 'center',
    paddingTop: 15,
    paddingBottom: 15,
    backgroundColor: '#FFFFFF',
    width: '100%',

  },
  button: {
    width: '90%',
    height: 60,
    marginVertical: 10,
  },

  closeCameraButton: {
    position: 'absolute',
    backgroundColor: 'transparent',
    top: 10,
    alignSelf: 'center',
    width: 160,
    height: 30,
    zIndex: 10, 
  },

  cameraContainer: {
    // flex: 1,
    position: 'absolute',
   // marginVertical: 100,
    width: width * factor, //can be changed
    height: height * factorThree, //can be changed
   // marginBottom: 20,
    borderRadius: 10,
    backgroundColor: 'transparent',
    justifyContent: 'center',
    alignItems: 'center',
  },
  cameraWrapper: {
    width: width,
    height: height * 0.7,
    position: 'relative',
    marginTop: 50,
    overflow: 'hidden',
  },
  point: {
    flex: 1,
    position: 'absolute',
    width:5,
    height: 5,
  },
  camera: {
    flex: 1,
    width: width * factorTwo,
    height: height,
    borderRadius: 10,
  },
  placeholderText: {
    color: 'red',
    fontSize: 16,
    fontWeight: 'bold',
    marginTop: 10,
  },

  videoDimensionsText: {
    marginTop: 10,
    marginBottom: 10,
    fontSize: 16,
    color: '#333',
  },
  ipAddressText : {
    marginBottom: 20,
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginTop: 2,
    fontFamily: 'System',

  },

  cameraOverlay: {
    // flex: 1,
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    backgroundColor: 'rgba(0,0,0,0.9)', // Optional: Dim the background behind camera
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 100,
  },
});