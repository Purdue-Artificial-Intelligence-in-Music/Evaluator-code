import React, { useState, useEffect } from 'react';
import { View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import { requireNativeViewManager } from 'expo-modules-core';

const CameraxView = requireNativeViewManager('Camerax');

const CameraComponent = (
  { startDelay, onClose }: { startDelay?: number; onClose?: () => void }
  ) => {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isDetectionEnabled, setIsDetectionEnabled] = useState(false);
  const [rhPronationText, setRhPronationText] = useState<string | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsCameraActive(true);
    }, startDelay || 100);

    return () => clearTimeout(timer);
  }, []);

// Pull the right-hand pronation label from the native event payload
const pullRightHandText = (nativeEvent: any) => {
  // Try common locations. Adjust only if your payload uses a different key.
  const raw =
    nativeEvent?.rightHand?.pronationLabel ??
    nativeEvent?.hands?.right?.pronationLabel ??
    nativeEvent?.pronationLabel ??
    null;

  if (typeof raw === 'string') {
    // normalize a few common variants into your three phrases
    const s = raw.toLowerCase();
    let label = raw;
    if (s.includes('supin')) label = 'Supination';
    else if (s.includes('too') || s.includes('over')) label = 'Too much pronation';
    else if (s.includes('good') || s.includes('neutral')) label = 'Good pronation';

    setRhPronationText(label);
  }
};

useEffect(() => {
  const originalLog = console.log;
  console.log = (...args: any[]) => {
    try {
      // You already log: console.log('Detection:', event.nativeEvent)
      if (args[0] === 'Detection:' && args[1]) {
        pullRightHandText(args[1]);
      }
      // You already log: console.log('No detection:', event.message)
      if (args[0] === 'No detection:') {
        setRhPronationText(null);
      }
    } catch {}
    originalLog(...args);
  };
  return () => {
    console.log = originalLog; // restore console.log
  };
}, []);

  return (
    <View style={styles.container}>
      <CameraxView
        style={styles.camera}
        cameraActive={isCameraActive}
        detectionEnabled={isDetectionEnabled}
        lensType="back"
        onDetectionResult={(event: any) => console.log('Detection:', event.nativeEvent)}
        onNoDetection={(event: any) => console.log('No detection:', event.message)}
      />
      
      {/* Right-hand pronation text overlay */}
      {isDetectionEnabled && rhPronationText && (
        <View style={styles.pronationPill}>
          <Text style={styles.pronationText}>{rhPronationText}</Text>
        </View>
      )}

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
  pronationPill: {
    position: 'absolute',
    top: 50,
    left: 20,
    right: 20,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    backgroundColor: 'rgba(0,0,0,0.7)',
    alignItems: 'center',
  },
  pronationText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default CameraComponent;