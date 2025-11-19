import React, { useState, useEffect } from 'react';
import {
  View,
  TouchableOpacity,
  Modal,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  Dimensions,
  Platform,
} from 'react-native';
import { requireNativeViewManager } from 'expo-modules-core';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FileSystem from 'expo-file-system';

const CameraxView = requireNativeViewManager('Camerax');

interface SummaryData {
  heightBreakdown?: { Top?: number; Middle?: number; Bottom?: number; Unknown?: number };
  angleBreakdown?: { Correct?: number; Wrong?: number; Unknown?: number };
  handPresenceBreakdown?: { Detected?: number; None?: number };
  handPostureBreakdown?: { Correct?: number; Supination?: number; 'Too much pronation'?: number; Unknown?: number };
  posePresenceBreakdown?: { Detected?: number; None?: number };
  elbowPostureBreakdown?: { Correct?: number; 'Low elbow'?: number; 'Elbow too high'?: number; Unknown?: number };
  userId?: string;
  timestamp?: string;
}

const { width: W, height: H } = Dimensions.get('window');
const BODY_W = W - 32;
const BODY_H = Math.min(H * 0.78, (W - 32) * 1.9);
const BODY_TOP = H * 0.08;

const CameraComponent = ({ startDelay, onClose }: { startDelay?: number; onClose: () => void }) => {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isDetectionEnabled, setIsDetectionEnabled] = useState(false);
  const [lensType, setLensType] = useState('back');
  const [userId, setUserId] = useState('default_user');
  const [showSetupOverlay, setShowSetupOverlay] = useState(true);

  const [summaryVisible, setSummaryVisible] = useState(false);
  const [summaryData, setSummaryData] = useState<SummaryData | null>(null);

  const [detailVisible, setDetailVisible] = useState(false);
  const [detailKey, setDetailKey] = useState<string | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsCameraActive(true);
    }, startDelay || 100);
    return () => clearTimeout(timer);
  }, [startDelay]);

  const toggleCamera = () => setLensType(prev => (prev === 'back' ? 'front' : 'back'));
  const handleReady = () => setShowSetupOverlay(false);

  useEffect(() => {
    const loadUserId = async () => {
      try {
        const email = await AsyncStorage.getItem('userEmail');
        if (email) setUserId(email);
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
      timestamp,
    } = event.nativeEvent;

    setSummaryData({
      heightBreakdown,
      angleBreakdown,
      handPresenceBreakdown,
      handPostureBreakdown,
      posePresenceBreakdown,
      elbowPostureBreakdown,
      userId: eventUserId,
      timestamp,
    });
    setSummaryVisible(true);
  };

  const closeSummary = async () => {
    setSummaryVisible(false);
    setSummaryData(null);

    try {
      const dir = `${FileSystem.cacheDirectory}summary_images`;
      const info = await FileSystem.getInfoAsync(dir);

      if (info.exists) {
        await FileSystem.deleteAsync(dir, { idempotent: true });
      }

      await FileSystem.makeDirectoryAsync(dir, { intermediates: true });
    } catch (e) {
      console.error("Error clearing cache:", e);
    }
  };

  const openDetail = (key: string) => {
    setDetailKey(key);
    setDetailVisible(true);
  };
  const closeDetail = () => {
    setDetailVisible(false);
    setDetailKey(null);
  };

  const imageMap: Record<string, string> = {
    'Correct elbow': 'good_elbow.png',
    'Elbow too high': 'high_elbow.png',
    'Low elbow': 'low_elbow.png',
    'Supination': 'supination.png',
    'Too much pronation': 'too_much_pronation.png',
    'Correct hand posture': 'good_pronation.png',
    'Correct angle': 'correct_angle.png',
    'Incorrect angle': 'incorrect_angle.png',
    'Middle': 'correct_bow.png',
    'Top': 'bow_too_high.png',
    'Bottom': 'bow_too_low.png',
  };

  const getImageFromCache = async (filename: string) => {
    if (!filename) return null;
    const path = `${FileSystem.cacheDirectory}summary_images/${filename}`;
    const info = await FileSystem.getInfoAsync(path);
    return info.exists ? `file://${path}` : null;
  };

  const AsyncImage = ({ filename }: { filename: string }) => {
    const [uri, setUri] = useState<string | null>(null);

    useEffect(() => {
      let mounted = true;
      const load = async () => {
        if (!filename) return;
        const path = await getImageFromCache(filename);
        if (mounted) setUri(path);
      };
      load();
      return () => {
        mounted = false;
      };
    }, [filename]);

    if (!filename) return <Text>No image available</Text>;
    if (!uri) return <Text>No image available</Text>;
    return <Image source={{ uri }} style={{ width: '100%', height: 150, resizeMode: 'contain' }} />;
  };

  const renderDetailContent = () => {
    if (!detailKey || !summaryData) return <Text>No data</Text>;

    let items: { label: string; value?: number }[] = [];

    switch (detailKey) {
      case 'height':
        items = [
          { label: 'Top', value: summaryData.heightBreakdown?.Top },
          { label: 'Middle', value: summaryData.heightBreakdown?.Middle },
          { label: 'Bottom', value: summaryData.heightBreakdown?.Bottom },
        ];
        break;
      case 'angle':
        items = [
          { label: 'Correct angle', value: summaryData.angleBreakdown?.Correct },
          { label: 'Incorrect angle', value: summaryData.angleBreakdown?.Wrong },
        ];
        break;
      case 'handPosture':
        items = [
          { label: 'Correct hand posture', value: summaryData.handPostureBreakdown?.Correct },
          { label: 'Supination', value: summaryData.handPostureBreakdown?.Supination },
          { label: 'Too much pronation', value: summaryData.handPostureBreakdown?.['Too much pronation'] },
        ];
        break;
      case 'elbow':
        items = [
          { label: 'Correct elbow', value: summaryData.elbowPostureBreakdown?.Correct },
          { label: 'Low elbow', value: summaryData.elbowPostureBreakdown?.['Low elbow'] },
          { label: 'Elbow too high', value: summaryData.elbowPostureBreakdown?.['Elbow too high'] },
        ];
        break;
      default:
        return <Text>No data</Text>;
    }

    return (
      <>
        <Text style={styles.sectionTitle}>{detailKey}</Text>
        {items.map(item => (
          <View key={item.label} style={{ marginBottom: 16 }}>
            <Text style={{ fontWeight: 'bold', marginBottom: 6 }}>
              {item.label}: {item.value?.toFixed(1) || 0}%
            </Text>
            <AsyncImage filename={imageMap[item.label] || ''} />
          </View>
        ))}
      </>
    );
  };

  const renderBreakdownRow = (label: string, value?: number) => (
    <Text key={label} style={{ fontWeight: 'bold', marginBottom: 4 }}>
      {label}: {value?.toFixed(1) || 0}%
    </Text>
  );

  return (
    <View style={styles.container}>
      {/* SUMMARY MODAL */}
      <Modal visible={summaryVisible} animationType="slide" transparent={true}>
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <ScrollView>
              <Text style={styles.title}>Session Summary</Text>
              {summaryData ? (
                <>
                  {['height', 'angle', 'handPosture', 'elbow'].map(section => {
                    let items: { label: string; value?: number }[] = [];
                    switch (section) {
                      case 'height':
                        items = [
                          { label: 'Top', value: summaryData.heightBreakdown?.Top },
                          { label: 'Middle', value: summaryData.heightBreakdown?.Middle },
                          { label: 'Bottom', value: summaryData.heightBreakdown?.Bottom },
                        ];
                        break;
                      case 'angle':
                        items = [
                          { label: 'Correct angle', value: summaryData.angleBreakdown?.Correct },
                          { label: 'Incorrect angle', value: summaryData.angleBreakdown?.Wrong },
                        ];
                        break;
                      case 'handPosture':
                        items = [
                          { label: 'Correct', value: summaryData.handPostureBreakdown?.Correct },
                          { label: 'Supination', value: summaryData.handPostureBreakdown?.Supination },
                          { label: 'Too much pronation', value: summaryData.handPostureBreakdown?.['Too much pronation'] },
                        ];
                        break;
                      case 'elbow':
                        items = [
                          { label: 'Correct', value: summaryData.elbowPostureBreakdown?.Correct },
                          { label: 'Low elbow', value: summaryData.elbowPostureBreakdown?.['Low elbow'] },
                          { label: 'Elbow too high', value: summaryData.elbowPostureBreakdown?.['Elbow too high'] },
                        ];
                        break;
                    }

                    return (
                      <View key={section} style={styles.section}>
                        <Text style={styles.sectionTitle}>
                          {section === 'height'
                            ? 'Bow Height'
                            : section === 'angle'
                            ? 'Bow Angle'
                            : section === 'handPosture'
                            ? 'Hand Posture'
                            : 'Elbow Posture'}
                        </Text>

                        {items.map(item => renderBreakdownRow(item.label, item.value))}

                        <View style={{ marginTop: 10 }}>
                          <TouchableOpacity
                            style={styles.viewMoreBtn}
                            onPress={() => openDetail(section)}
                          >
                            <Text style={styles.viewMoreText}>View More →</Text>
                          </TouchableOpacity>
                        </View>
                      </View>
                    );
                  })}
                  <View style={styles.section}>
                    <Text style={styles.timestamp}>Time: {summaryData.timestamp}</Text>
                  </View>
                  <TouchableOpacity style={[styles.viewMoreBtn, { marginTop: 6 }]} onPress={closeSummary}>
                    <Text style={styles.viewMoreText}>Close</Text>
                  </TouchableOpacity>
                </>
              ) : (
                <Text>No data available</Text>
              )}
            </ScrollView>
          </View>
        </View>
      </Modal>

      {/* DETAIL MODAL */}
      <Modal visible={detailVisible} animationType="slide" transparent={true}>
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <ScrollView>
              <TouchableOpacity style={[styles.viewMoreBtn, { marginBottom: 8 }]} onPress={closeDetail}>
                <Text style={styles.viewMoreText}>← Back</Text>
              </TouchableOpacity>
              {renderDetailContent()}
            </ScrollView>
          </View>
        </View>
      </Modal>

      {/* CAMERA VIEW */}
      <CameraxView
        style={styles.camera}
        userId={userId}
        cameraActive={isCameraActive}
        detectionEnabled={isDetectionEnabled}
        lensType={lensType}
        onSessionEnd={handleSessionEnd}
      />

      {/* Setup overlay */}
      {showSetupOverlay && (
        <>
          <View pointerEvents="none" style={styles.vignette} />
          <View pointerEvents="none" style={styles.silhouetteWrap}>
            <View style={styles.celloBody} />
            <View style={styles.bridgeGuide} />
            <View style={styles.endpinGuide} />
          </View>
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

      {/* UI Buttons */}
      <TouchableOpacity style={styles.closeButton} onPress={onClose} activeOpacity={0.7}>
        <Text style={styles.closeButtonText}>✕</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.flipButton} onPress={toggleCamera} activeOpacity={0.7}>
        <Text style={styles.flipButtonText}>🔄</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.detectionButton}
        onPress={() => setIsDetectionEnabled(prev => !prev)}
      >
        <Text style={styles.buttonText}>{isDetectionEnabled ? 'Stop Detection' : 'Start Detection'}</Text>
      </TouchableOpacity>
    </View>
  );
};

function Bullet({ children }: { children: React.ReactNode }) {
  return (
    <View style={styles.bulletRow}>
      <View style={styles.bulletDot} />
      <Text style={styles.bulletText}>{children}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  camera: { flex: 1 },
  closeButton: { position: 'absolute', top: 50, right: 20, width: 40, height: 40, borderRadius: 20, backgroundColor: 'rgba(0,0,0,0.7)', justifyContent: 'center', alignItems: 'center' },
  closeButtonText: { color: 'white', fontSize: 20, fontWeight: 'bold' },
  flipButton: { position: 'absolute', top: 100, right: 20, width: 40, height: 40, borderRadius: 20, backgroundColor: 'rgba(0,0,0,0.7)', justifyContent: 'center', alignItems: 'center' },
  flipButtonText: { fontSize: 22 },
  detectionButton: { position: 'absolute', bottom: 50, left: 20, right: 20, backgroundColor: 'rgba(0,0,0,0.7)', padding: 15, borderRadius: 8, alignItems: 'center' },
  buttonText: { color: 'white', fontSize: 16, fontWeight: 'bold' },

  modalContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.5)' },
  modalContent: { width: '90%', maxHeight: '80%', backgroundColor: 'white', borderRadius: 10, padding: 20 },
  title: { fontSize: 24, fontWeight: 'bold', marginBottom: 20, textAlign: 'center' },
  section: { marginBottom: 20 },
  sectionTitle: { fontSize: 18, fontWeight: 'bold', marginBottom: 10 },
  timestamp: { fontSize: 12, color: '#666', textAlign: 'center' },

  viewMoreBtn: { backgroundColor: 'white', padding: 10, borderRadius: 8, alignItems: 'center' },
  viewMoreText: { color: 'black', fontWeight: '700' },

  vignette: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,0,0,0.18)' },
  silhouetteWrap: { position: 'absolute', top: BODY_TOP, left: 16, right: 16, alignItems: 'center' },
  celloBody: { width: BODY_W, height: BODY_H, borderRadius: BODY_W * 0.28, borderWidth: 2, borderColor: 'rgba(255,255,255,0.9)', backgroundColor: 'rgba(255,255,255,0.05)' },
  bridgeGuide: { position: 'absolute', top: BODY_H * 0.46, left: BODY_W * 0.15, width: BODY_W * 0.7, borderTopWidth: 2, borderColor: 'white', borderStyle: 'dashed', opacity: 0.85 },
  endpinGuide: { position: 'absolute', top: BODY_H * 0.9, left: BODY_W * 0.5 - 1, height: BODY_H * 0.12, borderLeftWidth: 2, borderColor: 'white', borderStyle: 'dashed', opacity: 0.85 },
  instructionsCard: { position: 'absolute', bottom: Platform.select({ ios: 20, android: 16 }), left: 16, right: 16, padding: 14, borderRadius: 16, backgroundColor: 'rgba(0,0,0,0.55)', borderWidth: 1, borderColor: 'rgba(255,255,255,0.12)' },
  cardTitle: { color: 'white', fontWeight: 'bold', fontSize: 16 },
  bulletRow: { flexDirection: 'row', alignItems: 'flex-start', marginTop: 6 },
  bulletDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: 'white', marginTop: 6, marginRight: 6 },
  bulletText: { color: 'white', flex: 1, fontSize: 14 },
  readyBtn: { marginTop: 14, padding: 12, borderRadius: 8, backgroundColor: '#FFF', alignItems: 'center' },
  readyText: { color: 'black', fontWeight: 'bold', fontSize: 16 },
});

export default CameraComponent;
