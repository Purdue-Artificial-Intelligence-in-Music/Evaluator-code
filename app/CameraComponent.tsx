import React, { useState, useEffect } from 'react';
import { View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import { requireNativeViewManager } from 'expo-modules-core';

const CameraxView = requireNativeViewManager('Camerax');

const CameraComponent = ({ startDelay, onClose }) => {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isDetectionEnabled, setIsDetectionEnabled] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsCameraActive(true);
    }, startDelay || 100);

    return () => clearTimeout(timer);
  }, []);

  return (
    <View style={styles.container}>
      <CameraxView
        style={styles.camera}
        cameraActive={isCameraActive}
        detectionEnabled={isDetectionEnabled}
        lensType="back"
        onDetectionResult={(event) => console.log('Detection:', event.nativeEvent)}
        onNoDetection={(event) => console.log('No detection:', event.message)}
      />
      
      <TouchableOpacity
        style={styles.closeButton}
        onPress={onClose}
        activeOpacity={0.7}
      >
        <Text style={styles.closeButtonText}>✕</Text>
      </TouchableOpacity>
      
      <TouchableOpacity
        style={styles.detectionButton}
        onPress={() => setIsDetectionEnabled(!isDetectionEnabled)}
      >
        <Text style={styles.buttonText}>
          {isDetectionEnabled ? 'Stop Detection' : 'Start Detection'}
        </Text>
      </TouchableOpacity>
    </View>
  );
};

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
});

export default CameraComponent;