import React, { useState, useEffect, useRef } from 'react';


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

import { SafeAreaView, Button, Text, Image, StyleSheet, View, Dimensions, ScrollView, ActivityIndicator } from 'react-native';

import { ResizeMode, Video } from 'expo-av';
import * as ImagePickerExpo from 'expo-image-picker';

import { Camera, useCameraDevice, useCameraPermission, CameraRef } from 'react-native-vision-camera';
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


// TODO: use ip address of your computer (the backend) here (use ipconfig or ifconfig to look up)
export default function App() {
  const [serverIP, setServerIP] = useState("10.186.25.110"); // Change this to your server's IP address
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

useEffect(() => {
    const socket = new WebSocket(`ws://${serverIP}:8000/ws`);
    socketRef.current = socket;

    socket.onopen = () => console.log("WebSocket connected");
    socket.onmessage = (event) => {
      try {
        const responseData = JSON.parse(event.data);
        console.log("Received:", responseData);
        isSending.current = false;

        processPoints(responseData);
        setSupinating(responseData.supination.toString());
      } catch (e) {
        console.error("Error parsing server message:", e);
      }
    };
    socket.onerror = (err) => {
      console.error("WebSocket error:", err);
      isSending.current = false;
      socket.close();
    };

    socket.onclose = (event) => {
      console.log("WebSocket closed", event.code, event.reason);
    };
}, []);

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
  const takePicture = async () => {
    if (cameraRef.current == null || isTakingPhoto.current || (socketRef.current?.readyState !== WebSocket.OPEN)) return;
    isTakingPhoto.current = true;
    try {
      const photo = await cameraRef.current.takePhoto({
        flash: 'off',
        
      });
      setImageWidth(photo.width);
      setImageHeight(photo.height);
      console.log("Photo taken, width: ", photo.width, "height: ", photo.height);
      const base64 = await RNFS.readFile(photo.path, 'base64');
      sendImageToBackend(`data:image/jpeg;base64,${base64}`);
    } catch (err: any) {
      if (err?.message?.includes('Camera is closed')) {
        console.warn('Camera was closed before photo could be taken.');
      } else {
        console.error("Error taking photo:", err);
      }
    } finally {
      isTakingPhoto.current = false;
    }
  };
  useEffect(() => {
    if (recording && !loading) {
      const timeout = setTimeout(() => {
        intervalRef.current = setInterval(() => {
          takePicture();
        }, 500) as unknown as number;
      }, startDelay);
      return () => {
        clearTimeout(timeout);
        if (intervalRef.current !== null) clearInterval(intervalRef.current);
      };
    } else {
      if (intervalRef.current !== null) clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current !== null) clearInterval(intervalRef.current);
    };
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
      <View style={styles.cameraWrapper}>
        <Camera 
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        // style={{ marginTop: 50, width: width, height: height * 0.7 }}
        device={device}
        isActive={true}
        photo={true}
        video={false}
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
    console.log("newLines: ", newLines);
    const safePoints = Object.values(responseData).filter(p => {
      return (
        typeof p === 'object' &&
        p !== null &&
        typeof p.x === 'number' &&
        typeof p.y === 'number' &&
        !isNaN(p.x) &&
        !isNaN(p.y)
      );
    });
    setPoints(safePoints);
    // console.log('Points:', safePoints);
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
    if ((socketRef.current) && (socketRef.current.readyState === WebSocket.OPEN) && (!isSending.current)) {
      isSending.current = true;
      const message = {
        type: "frame",
        image: imageBase64,
      };
      socketRef.current.send(JSON.stringify(message));
    }
  };

  // const sendImageToBackend = async (imageBase64: string) => {
  //   const jsonData = {
  //     "title": "Test Image",
  //     "content": "This is a test image",
  //     "image": imageBase64,
  //   }
  //   try {
  //     setLoading(true);
  //     const response = await fetch(`http://${serverIP}:8000/upload/`, {
  //       method: 'POST',
  //       body: JSON.stringify(jsonData),
  //       headers: {
  //         'Content-Type': 'application/json',
  //       },
  //     });

  //     if (response.ok) {
  //       const responseData: ResponseData = await response.json(); // Type casting here

  //       console.log(responseData);
  //       processPoints(responseData);  
  //       setSupinating(responseData["supination"].toString());
  //     }

  //   } catch (error) {
  //     console.error('Error uploading image:', error);
  //     alert('Error: Failed to upload image.');

  //   } finally {
  //     setLoading(false);
  //   }
  // };

  
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
    console.log("Opening camera, isCameraOpen:", isCameraOpen);
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
          console.log("Pressed open camera");
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