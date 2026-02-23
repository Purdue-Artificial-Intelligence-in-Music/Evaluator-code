import React, { useState } from 'react';
import * as VideoAnalyzer from '../modules/expo-video-analyzer/src/ExpoVideoAnalyzer';
import CameraComponent from './CameraComponent';

import LogoutButton from '../components/LogoutButton';

import { TouchableOpacity} from 'react-native';
import ChooseVideoIcon from '../assets/images/ChooseVideo.png';
import OpenCamera from '../assets/images/OpenCamera.png';
import SessionHistory from '../assets/images/SessionHistory.png';
import Back from '../assets/images/Back.png';
import SendVideo from '../assets/images/SendVideo.png';
import { Platform } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

import * as MediaLibrary from 'expo-media-library';

import { SafeAreaView, Button, Text, Image, StyleSheet, View, Dimensions, ScrollView, ActivityIndicator, Alert, Modal } from 'react-native';

import { ResizeMode, Video } from 'expo-av';
import * as ImagePickerExpo from 'expo-image-picker';

import { Camera } from 'react-native-vision-camera';

import * as FileSystem from 'expo-file-system';

const width = Dimensions.get('window').width;
const height = Dimensions.get('window').height;
console.log(width)
console.log(height)


export default function HomePage() {
  const [videoDimensions, setVideoDimensions] = useState<{ width: number; height: number } | null>(null);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [userId, setUserId] = useState('default_user');
  const [loading, setLoading] = useState(false);
  const [hasPermission, setHasPermission] = useState(false);  // State for camera permission

  const [sendButton, setsendButton] = useState(false)
  const [sendVideo, setsendVideo] = useState(false)
  const [videofile, setvideofile] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Session History flag
  const [openHistoryOnCamera, setOpenHistoryOnCamera] = useState(false);

  const requestCameraPermission = async () => {
    const status = await Camera.requestCameraPermission();
    if (status === 'granted') {
      setHasPermission(true);
    } else {
      setHasPermission(false);
    }
  };

  // Function to handle video selection
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

  const returnBack = async () => {
    setIsCameraOpen(false);
    setVideoUri(null);
    setVideoDimensions(null);
    setsendButton(false);
    setIsAnalyzing(false);
  };

  const cancelVidProc = async () => {
    if (isAnalyzing) {
      try {
        await VideoAnalyzer.cancelProcessing();
        Alert.alert('Cancelled', 'Video processing cancelled.');
        console.log("Video processing cancelled.");
      } catch (err) {
        console.error("Failed to cancel processing:", err);
      } finally {
        setIsAnalyzing(false);
        setsendVideo(false);
        setvideofile(null);
        setVideoUri(null);
      }
    } else {
      setIsCameraOpen(false);
      setVideoUri(null);
      setsendButton(false);
      setvideofile(null);
      setsendVideo(false);
    }
  };

  const loadUserId = async () => {
      try {
        const email = await AsyncStorage.getItem('userEmail');
        if (email) {
          setUserId(email);
          console.log('User ID loaded for video processing:', email);
          return email;
        } else {
          console.warn('No user email found, using default');
          return 'default_user';
        }
      } catch (error) {
        console.error('Error loading user ID:', error);
        return 'default_user';
      }
  };

  const sendVideoBackend = async () => {
    if (!videofile) {
      Alert.alert('Error', 'Please select a video');
      return;
    }

    setIsAnalyzing(true);
    try {
      // 1. initialize video analyzer
      const initResult = await VideoAnalyzer.initialize();
      if (!initResult.success) {
        Alert.alert('Error: Initialization fail', 'Initialization Failed');
        return;
      }

      // 2. process video
      console.log('Start frame extraction...');
      //console.log(VideoAnalyzer);
      const currentUserId = await loadUserId();
      const proc = await VideoAnalyzer.analyzeVideo(videofile, currentUserId);
      /*
      const proc = await VideoAnalyzer.processVideoComplete(videofile);
      console.log('Processing complete:', proc);
      console.log('Frame processing result:', proc);

      if (!proc.success) {
        Alert.alert('Processing Error', 'An error occured processing.');
        return;
      }
      setvideofile(proc.outputPath);
      setVideoUri(proc.outputPath);
      setsendVideo(false);


      // 3. give user option of saving resulting video to photos
      Alert.alert(
        "Processing complete",
        `Do you want to save the processed video to Photos?`,
        [
          {
            text: "No",
            style: "destructive",
            onPress: async () => {
              try {
                await FileSystem.deleteAsync(proc.outputPath, { idempotent: true });
                console.log("Temporary file deleted:", proc.outputPath);
                setvideofile(null);
                setVideoUri(null);
            } catch (err) {
                console.error("Failed to delete temp file:", err);
            }
            },
          },
          {
            text: "Yes",
            onPress: async () => {
              const { status } = await MediaLibrary.requestPermissionsAsync();
              if (status === "granted") {
                await MediaLibrary.saveToLibraryAsync(proc.outputPath);
                Alert.alert("Saved", "Video saved to Photos.");
              } else {
                Alert.alert("Permission denied", "Could not save video.");
                return;
              }
            },
          },
        ]
      );
  */
    } catch (error: any) {
      console.error('sendVideoBackend failed:', error);
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

 
  
  const openCamera = async (requestPerm: boolean = true) => {
    if (requestPerm) {
      await requestCameraPermission();
    }
    requestCameraPermission()
    setIsCameraOpen(true);
    setVideoUri(null);
    setVideoDimensions(null);
    setsendButton(false);
  };

  const closeCamera = () => {
    setIsCameraOpen(false);
    setVideoUri(null);
    setVideoDimensions(null);
    setsendButton(false);

    // reset Session History flag
    setOpenHistoryOnCamera(false);

  };
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.buttonStyle}>

      <View style={styles.logoutContainer}>
        <LogoutButton />
      </View>

      <TouchableOpacity
        onPress={pickVideo}
        style={{
          width: 320,    // smaller width
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

      {/* Open Camera now explicitly clears history flag */}
      <TouchableOpacity
        onPress={() => { 
          if (!isCameraOpen) {
            setOpenHistoryOnCamera(false); // normal camera mode
            openCamera(true);
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

      {/* Session History button styled like the others (no extra image) */}
      <TouchableOpacity
        onPress={() => {
          console.log("Opening session history (no camera)");
          if (!isCameraOpen) {
            setOpenHistoryOnCamera(true); // tell CameraComponent to open history modal
            openCamera(false);
          }
        }}
        activeOpacity={0.7}
        style={{
          width: 320,
          height: 60,
          marginVertical: 5,
          borderRadius: 8,
          overflow: 'hidden',
        }}
      >
        <Image
          source={SessionHistory}
          style={{ width: '100%', height: '100%' }}
          resizeMode="cover"
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

      
      {(isCameraOpen) ? (
        <View style={styles.cameraOverlay}>
          {/* tell CameraComponent whether to auto-open Session History */}
          <CameraComponent
            startDelay={0}
            onClose={closeCamera}
            initialHistoryOpen={openHistoryOnCamera}
          />
        </View>
      ) : null}

      {videoUri ? (
        <ScrollView
          style={{ flex: 1 }}
          contentContainerStyle={{
            justifyContent: 'center',
            alignItems: 'center',
          }}
        >
          <Video
            source={{ uri: videoUri }}
            shouldPlay
            resizeMode={ResizeMode.CONTAIN}
            style={{
              width: videoDimensions ? Math.min(videoDimensions.width, width) : width,
              height: videoDimensions ? Math.min(videoDimensions.height, height * 0.6) : height * 0.6,
            }}
          />
        </ScrollView>
      ) : (
        <View style={{ alignItems: 'center', paddingHorizontal: 24 }}>
          {!sendVideo && (
            <Text style={styles.placeholderText}>No video selected</Text>
          )}
          <Text style={styles.noticeText}>
            This app is a supplementary tool for checking overall posture. Posture standards may vary by school, so always follow your instructor's advice first.
          </Text>
        </View>
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

      {/* activityindicator: full-screen overlay during upload/analysis */}
      <Modal transparent visible={isAnalyzing} animationType="fade" statusBarTranslucent>
        <View style={styles.overlay}>
          <View style={styles.spinnerCard}>
            <ActivityIndicator size="large" />
            <Text style={styles.spinnerText}>Processing video…</Text>
            <TouchableOpacity
              onPress={cancelVidProc}
              style={styles.cancelButton}
              activeOpacity={0.7}
            >
              <Text style={styles.cancelButtonText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

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
  placeholderText: {
    color: '#000',
    fontSize: 16,
    fontWeight: 'bold',
    marginTop: 10,
  },
  noticeText: {
    color: '#666',
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 20,
    marginTop: 8,
  },

  videoDimensionsText: {
    marginTop: 10,
    marginBottom: 10,
    fontSize: 16,
    color: '#333',
  },
  cameraOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'transparent',
    zIndex: 1000,
  },

  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.35)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 2000,
  },
  spinnerCard: {
    backgroundColor: '#fff',
    paddingVertical: 18,
    paddingHorizontal: 22,
    borderRadius: 16,
    minWidth: 220,
    alignItems: 'center',
  },
  spinnerText: {
    marginTop: 8,
    fontSize: 16,
    fontWeight: '600',
  },
  cancelButton: {
  marginTop: 15,
  paddingVertical: 10,
  paddingHorizontal: 20,
  borderRadius: 8,
  backgroundColor: '#d9534f',
  },
  cancelButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  logoutContainer: {
    width: '100%',
    alignItems: 'flex-end',
    paddingRight: 30,
    paddingTop: 5,
    marginBottom: 5,
  },
});