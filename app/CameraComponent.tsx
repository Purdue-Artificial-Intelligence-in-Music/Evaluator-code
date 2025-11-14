import React, { useState, useEffect } from 'react';
import { View, TouchableOpacity, Modal, Text, StyleSheet, ScrollView, Button } from 'react-native';
import { requireNativeViewManager } from 'expo-modules-core';
import AsyncStorage from '@react-native-async-storage/async-storage';

const CameraxView = requireNativeViewManager('Camerax');

interface SummaryData {
  heightBreakdown?: {
    Top?: number;
    Middle?: number;
    Bottom?: number;
    Unknown?: number;
  };
  angleBreakdown?: {
    Correct?: number;
    Wrong?: number;
    Unknown?: number;
  };
  handPresenceBreakdown?: {
    Detected?: number;
    None?: number;
  };
  handPostureBreakdown?: {
    Correct?: number;
    Supination?: number;
    'Too much pronation'?: number;
    Unknown?: number;
  };
  posePresenceBreakdown?: {
    Detected?: number;
    None?: number;
  };
  elbowPostureBreakdown?: {
    Correct?: number;
    'Low elbow'?: number;
    'Elbow too high'?: number;
    Unknown?: number;
  };
  userId?: string;
  timestamp?: string;
}

const CameraComponent = ({ startDelay, onClose }) => {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isDetectionEnabled, setIsDetectionEnabled] = useState(false);
  const [lensType, setLensType] = useState('back'); // use front or back camera
  const [userId, setUserId] = useState('default_user');

  const [summaryVisible, setSummaryVisible] = useState(false);
  const [summaryData, setSummaryData] = useState<SummaryData | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsCameraActive(true);
    }, startDelay || 100);

    return () => clearTimeout(timer);
  }, []);

  const toggleCamera = () => {
    setLensType(prev => prev === 'back' ? 'front' : 'back');
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

  const handleSessionEnd = (event: any) => {
  const {
    heightBreakdown,
    angleBreakdown,
    handPresenceBreakdown,
    handPostureBreakdown,
    posePresenceBreakdown,
    elbowPostureBreakdown,
    userId: eventUserId,
    timestamp
  } = event.nativeEvent;

  const newSummaryData = {
    heightBreakdown,
    angleBreakdown,
    handPresenceBreakdown,
    handPostureBreakdown,
    posePresenceBreakdown,
    elbowPostureBreakdown,
    userId: eventUserId,
    timestamp
  };

  setSummaryData(newSummaryData);
  setSummaryVisible(true);
};

  const closeSummary = () => {
    setSummaryVisible(false);
    setSummaryData(null);
  };

  return (
    <View style={styles.container}>
      <Modal visible={summaryVisible} animationType="slide" transparent={true}>
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <ScrollView>
              <Text style={styles.title}>Session Summary</Text>

              {summaryData ? (
                <>
                  <View style={styles.section}>
                    <Text style={styles.sectionTitle}>Bow Height</Text>
                    <Text>Top: {summaryData.heightBreakdown?.Top?.toFixed(1) || 0}%</Text>
                    <Text>Middle: {summaryData.heightBreakdown?.Middle?.toFixed(1) || 0}%</Text>
                    <Text>Bottom: {summaryData.heightBreakdown?.Bottom?.toFixed(1) || 0}%</Text>
                  </View>

                  <View style={styles.section}>
                    <Text style={styles.sectionTitle}>Bow Angle</Text>
                    <Text>Correct: {summaryData.angleBreakdown?.Correct?.toFixed(1) || 0}%</Text>
                    <Text>Wrong: {summaryData.angleBreakdown?.Wrong?.toFixed(1) || 0}%</Text>
                  </View>

                  <View style={styles.section}>
                    <Text style={styles.sectionTitle}>Hand Posture</Text>
                    <Text>Correct: {summaryData.handPostureBreakdown?.Correct?.toFixed(1) || 0}%</Text>
                    <Text>Supination: {summaryData.handPostureBreakdown?.Supination?.toFixed(1) || 0}%</Text>
                    <Text>Too much pronation: {summaryData.handPostureBreakdown?.['Too much pronation']?.toFixed(1) || 0}%</Text>
                  </View>

                  <View style={styles.section}>
                    <Text style={styles.sectionTitle}>Elbow Posture</Text>
                    <Text>Correct: {summaryData.elbowPostureBreakdown?.Correct?.toFixed(1) || 0}%</Text>
                    <Text>Low elbow: {summaryData.elbowPostureBreakdown?.['Low elbow']?.toFixed(1) || 0}%</Text>
                    <Text>Elbow too high: {summaryData.elbowPostureBreakdown?.['Elbow too high']?.toFixed(1) || 0}%</Text>
                  </View>

                  <View style={styles.section}>
                    <Text style={styles.timestamp}>Time: {summaryData.timestamp}</Text>
                  </View>
                </>
              ) : (
                <Text>No data available</Text>
              )}

              <Button title="Close" onPress={closeSummary} />
            </ScrollView>
          </View>
        </View>
      </Modal>


      <CameraxView
        style={styles.camera}
        userId={userId}
        cameraActive={isCameraActive}
        detectionEnabled={isDetectionEnabled}
        lensType={lensType}
        onSessionEnd={handleSessionEnd}
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
  modalContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)'
  },
  modalContent: {
    width: '90%',
    maxHeight: '80%',
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 20
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center'
  },
  section: {
    marginBottom: 20
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10
  },
  timestamp: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center'
  }
});

export default CameraComponent;