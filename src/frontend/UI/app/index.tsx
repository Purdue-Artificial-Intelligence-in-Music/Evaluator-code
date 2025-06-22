import React, { useState, useEffect, useRef } from 'react';


import { TouchableOpacity} from 'react-native';
import ChooseVideoIcon from '../assets/images/ChooseVideo.png';
import CloseCamera from '../assets/images/CloseCamera.png';
import FetchData from '../assets/images/FetchData.png';
import OpenCamera from '../assets/images/OpenCamera.png';
import Back from '../assets/images/Back.png';
import Record from '../assets/images/Record.png';
import Recording from '../assets/images/Recording.png';

import { useWindowDimensions } from 'react-native';
import { Platform } from 'react-native';

import * as MediaLibrary from 'expo-media-library';

import { SafeAreaView, Button, Text, Image, StyleSheet, View, Dimensions, ScrollView, ActivityIndicator } from 'react-native';

import { ResizeMode, Video } from 'expo-av';
import * as ImagePickerExpo from 'expo-image-picker';
import { Camera, CameraView } from 'expo-camera';

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

const windowWidth = Dimensions.get('window').width;
const windowHeight = Dimensions.get('window').height;
const aspectRatio = windowWidth + "x" + windowHeight;

// TODO: use ip address of your computer (the backend) here (use ipconfig or ifconfig to look up)


export default function App() {
  const [serverIP, setServerIP] = useState("192.168.68.52"); // Change this to your server's IP address
  const [videoDimensions, setVideoDimensions] = useState<{ width: number; height: number } | null>(null);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [ipAddress, setIpAddress] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [hasPermission, setHasPermission] = useState(false);  // State for camera permission
  

  
  const [points, setPoints] = useState([{x: 0, y: 0}]); // FOR THE POINTS WE NEED TO RENDER ON THE SCREEN
  const [linePoints, setLinePoints] = useState([{start: {x: 0, y: 0}, end: {x: 0, y: 0}}]);
  const intervalRef = useRef<NodeJS.Timeout>();
  const [recording, setRecording] = useState<boolean>(false);
  const [sendButton, setsendButton] = useState(false)
  const [sendVideo, setsendVideo] = useState(false)
  const [videofile, setvideofile] = useState<string | null>(null);

  const [supinating, setSupinating] = useState<String>("none");
  
  

  // CameraComponent to handle camera view
interface CameraComponentProps {
  startDelay: number;
}

const CameraComponent: React.FC<CameraComponentProps> = ({ startDelay }) => {
  const cameraRef = useRef<Camera | null>(null); // Ref to the Camera component

  const takePicture = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync({
        qualityPrioritization: 'speed',
        quality: 85,
        skipProcessing: true,
        skipMetadata: true,
        shutterSound: false,
    });
      sendImageToBackend(photo.base64 || '');
      // console.log("photo dimensions: ", photo.width, photo.height);
    }
  };

  useEffect(() => {
    if (recording && !(loading)) {
      setTimeout(() => {
        intervalRef.current = setInterval(() => {
          takePicture();
        }, 500);
      }, startDelay)
    } else {
      clearInterval(intervalRef.current);
    }
 
    if (!isCameraOpen) {
      console.log("camera off");
    } else {
      console.log("camera on:", isCameraOpen);
    }

    return () => clearInterval(intervalRef.current);
  }, [recording, loading]);

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
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        pictureSize={aspectRatio}
        mirror={true}
        onCameraReady={() => setLoading(false)}
      />

      <Text style={styles.placeholderText}> Forearm posture: {supinating} </Text> 
      <Button title={recording ? "STOP" : "RECORD"} onPress={() => {
        if (recording) {
          setRecording(false);
          // reset old dots and lines
          setPoints([{x: 0, y: 0}]);
          setLinePoints([{start: {x: 0, y: 0}, end: {x: 0, y: 0}}]);
        } else {
          setRecording(true);
        }
      }} />
      
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


  function processPoints(responseData: ResponseData) {

    //modify the points such that the screen adjusts
    responseData = correctPoints(responseData, 0.75)

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
      const response = await fetch(`http://${serverIP}:8000/upload/`, {
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

        console.log(responseData)
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

  // base64 + URL
  async function sendVideoBackend() {
    try {
      if (!videofile) {
        alert('No video selected');
        return;
      }

      // Extract base64 data from the URI
      
       const base64Data = await FileSystem.readAsStringAsync(videofile, {
         encoding: FileSystem.EncodingType.Base64,
       });

      const jsonData = {
        video: base64Data
      };

      console.log("Sending video data:", {
        hasData: !!jsonData.video,
        dataLength: jsonData.video?.length
      });
      console.log("Raw videofile URI:", videofile);

      setsendVideo(true);
      setVideoUri(null);
      setsendButton(false);

      const response = await fetch(`http://${serverIP}:8000/send-video`, {
        method: "POST",
        body: JSON.stringify(jsonData),
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Server error:', errorData);
        throw new Error(`Server error: ${response.status}`);
      }

      const result = await response.json();
      console.log("Response from Backend", result);

      if (!result.Video) {
        throw new Error('No video data in response');
      }

      
      setVideoUri(result.Video);

      
      if (result.Width && result.Height) {
        setVideoDimensions({ width: result.Width, height: result.Height });
      }

      setsendVideo(false);
    } catch (error) {
      console.error('Error sending video:', error);
      alert(`Failed to send video: ${error.message}`);
      setsendVideo(false);
    }
  }

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

 

  async function demoVideo() {
    const jsonData = {
      "title": "demo video",
      "videouri": "this is a test",
    }
    const response = await fetch(`http://${serverIP}:8000/api/upload/`, {
      method: 'POST',
      body: JSON.stringify(jsonData),
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }
  
  const openCamera = () => {
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
      onPress={demoVideo}
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

      
      {isCameraOpen && hasPermission && (
        <View style={styles.cameraOverlay}>
          <CameraComponent startDelay={0} />

         
        </View>
      )}

      
      <Text style={styles.ipAddressText}>
        IP Address: {ipAddress || 'Fetching IP...'}
      </Text>



      {videoUri ? (
        <ScrollView 
          contentContainerStyle={{ flexGrow: 1 }} 
          showsVerticalScrollIndicator={true}
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
        </ScrollView>
      ) : (
        !sendVideo && <Text style={styles.placeholderText}>No video selected</Text>
      )}

      {videoDimensions && (
        <Text style={styles.videoDimensionsText}>
          Video Dimensions: {videoDimensions.width}x{videoDimensions.height}
        </Text>
      )}


      {sendVideo && <ActivityIndicator size="large" color="#0000ff" />} 

      <View style={{marginTop: 10,opacity: sendButton ? 1: 0}}>
        <Button title="Send Video" onPress={sendVideoBackend} />
      </View>

      {/* download video */}
      {(videoUri && !sendButton) && <Button
        title="Download Video"
        onPress={() => downloadVideo(videoUri)}
        disabled={!videoUri}
      />}

      {/* only show dots and lines when camera is open and recording*/}
      {(isCameraOpen && recording) && <Svg style={{ ...styles.cameraContainer, height: 440 - 20 }}>
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
      </Svg>}
        
      
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
    flex: 1,
    position: 'absolute',
    marginVertical: 130,
    top: 250,
    width: 640 * 0.9, //can be changed
    height: 440 * 0.9,
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
    flex: 1,
    width: '100%',
    height: '100%',
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
    fontSize: 16,
    color: '#333',
  },
  ipAddressText : {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginTop: 2,
    fontFamily: 'System',

  },

  cameraOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: windowWidth,
    height: windowHeight,
    backgroundColor: 'rgba(0,0,0,0.9)', // Optional: Dim the background behind camera
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 5,
  },
});