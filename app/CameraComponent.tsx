import React, { useState, useEffect } from 'react';
import { View, TouchableOpacity, Text, StyleSheet, Dimensions, Platform } from 'react-native';
import { requireNativeViewManager } from 'expo-modules-core';
import AsyncStorage from '@react-native-async-storage/async-storage';

const CameraxView = requireNativeViewManager('Camerax');

const { width: W, height: H } = Dimensions.get('window');
const BODY_W = W - 32;
const BODY_H = Math.min(H * 0.78, (W - 32) * 1.9);
const BODY_TOP = H * 0.08;

const CameraComponent = ({ startDelay, onClose }) => {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isDetectionEnabled, setIsDetectionEnabled] = useState(false);
  const [lensType, setLensType] = useState('back'); // use front or back camera
  const [userId, setUserId] = useState('default_user');
  const [showSetupOverlay, setShowSetupOverlay] = useState(true);


  useEffect(() => {
    const timer = setTimeout(() => {
      setIsCameraActive(true);
    }, startDelay || 100);

    return () => clearTimeout(timer);
  }, []);

  const toggleCamera = () => {
    setLensType(prev => prev === 'back' ? 'front' : 'back');
  };

  const handleReady = () => {
    setShowSetupOverlay(false);
    setIsDetectionEnabled(true);
  };

  // load user id
  useEffect(() => {
    const loadUserId = async () => {
      try {
        const email = await AsyncStorage.getItem('userEmail');
        if (email) {
          setUserId(email);
          console.log('User ID loaded for camera:', email);
        } else {
          console.warn('No user email found, using default');
        }
      } catch (error) {
        console.error('Error loading user ID:', error);
      }
    };
    loadUserId();
  }, []);

  return (
    <View style={styles.container}>
      <CameraxView
        style={styles.camera}
        userId={userId}
        cameraActive={isCameraActive}
        detectionEnabled={isDetectionEnabled}
        lensType={lensType}
      />
      
      <TouchableOpacity
        style={styles.closeButton}
        onPress={onClose}
        activeOpacity={0.7}
      >
        <Text style={styles.closeButtonText}>✕</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.flipButton}
        onPress={toggleCamera}
        activeOpacity={0.7}
      >
        <Text style={styles.flipButtonText}>🔄</Text>
      </TouchableOpacity>
      
      <TouchableOpacity
        style={styles.detectionButton}
          onPress={() => {
            if (showSetupOverlay) setShowSetupOverlay(false);
            setIsDetectionEnabled(!isDetectionEnabled);
          }}
      >
        <Text style={styles.buttonText}>
          {isDetectionEnabled ? 'Stop Detection' : 'Start Detection'}
        </Text>
      </TouchableOpacity>

      {showSetupOverlay && (
        <>
          {/* dark overlay */}
          <View pointerEvents="none" style={styles.vignette} />

          {/* cello silhouette */}
          <View pointerEvents="none" style={styles.silhouetteWrap}>
            <View style={styles.celloBody} />
            <View style={styles.bridgeGuide} />
            <View style={styles.endpinGuide} />
          </View>

          {/* setup instructions */}
          <View style={styles.instructionsCard}>
            <Text style={styles.cardTitle}>Set up your camera & cello</Text>
            <View style={{ height: 6 }} />
            <Bullet>Hold phone upright (portrait), ~2–3 ft (60–90 cm) away</Bullet>
            <Bullet>Center yourself and the cello inside the outline</Bullet>
            <Bullet>Keep the bridge near the dotted line</Bullet>
            <Bullet>Ensure the endpin is visible and background is clear</Bullet>

            <TouchableOpacity style={styles.readyBtn} onPress={handleReady} activeOpacity={0.9}>
              <Text style={styles.readyText}>Ready</Text>
            </TouchableOpacity>
          </View>
        </>
      )}
    </View>
  );
};

function Bullet({ children }) {
  return (
    <View style={styles.bulletRow}>
      <View style={styles.bulletDot} />
      <Text style={styles.bulletText}>{children}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  closeButton: {
    position: 'absolute',
    top: 50,
    right: 20,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  closeButtonText: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
  },
  flipButton: {
    position: 'absolute',
    top: 100,
    right: 20,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  flipButtonText: {
    fontSize: 22,
  },
  detectionButton: {
    position: 'absolute',
    bottom: 50,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  vignette: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.18)',
  },
  silhouetteWrap: {
    position: 'absolute',
    top: BODY_TOP,
    left: 16,
    right: 16,
    alignItems: 'center',
  },
  celloBody: {
    width: BODY_W,
    height: BODY_H,
    borderRadius: BODY_W * 0.28,
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.9)',
    backgroundColor: 'rgba(255,255,255,0.05)',
  },
  // dashed line near the bridge (now relative to celloBody)
  bridgeGuide: {
    position: 'absolute',
    top: BODY_H * 0.46,
    left: BODY_W * 0.15,
    width: BODY_W * 0.7,
    borderTopWidth: 2,
    borderColor: 'white',
    borderStyle: 'dashed',
    opacity: 0.85,
  },
  // short dashed vertical near the endpin (relative to celloBody)
  endpinGuide: {
    position: 'absolute',
    top: BODY_H * 0.9,
    left: BODY_W * 0.5 - 1,
    height: BODY_H * 0.12,
    borderLeftWidth: 2,
    borderColor: 'white',
    borderStyle: 'dashed',
    opacity: 0.85,
  },
  instructionsCard: {
    position: 'absolute',
    bottom: Platform.select({ ios: 20, android: 16 }),
    left: 16,
    right: 16,
    padding: 14,
    borderRadius: 16,
    backgroundColor: 'rgba(0,0,0,0.55)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.12)',
  },
  cardTitle: {
    color: 'white',
    fontWeight: '700',
    fontSize: 16,
    letterSpacing: 0.2,
  },
  bulletRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginTop: 6,
  },
  bulletDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginTop: 7,
    marginRight: 8,
    backgroundColor: 'white',
    opacity: 0.9,
  },
  bulletText: {
    color: 'white',
    opacity: 0.95,
    fontSize: 14,
    lineHeight: 19,
    flexShrink: 1,
  },
  readyBtn: {
    alignSelf: 'flex-end',
    marginTop: 10,
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 999,
    backgroundColor: 'white',
  },
  readyText: {
    color: '#111',
    fontWeight: '700',
    fontSize: 13,
    letterSpacing: 0.3,
  },
});

export default CameraComponent;