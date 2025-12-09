import React, { useState, useEffect } from 'react';

import { View, TouchableOpacity, Modal, Text, ScrollView, Button, TextInput, Alert } from 'react-native';

import { requireNativeViewManager, requireNativeModule } from 'expo-modules-core';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { styles } from './CameraComponentStyles';

const CameraxView = requireNativeViewManager('Camerax');
const CameraxModule = requireNativeModule('Camerax')


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
  sessionDuration?: string;
}

const CameraComponent = ({ startDelay, onClose }) => {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isDetectionEnabled, setIsDetectionEnabled] = useState(false);
  const [lensType, setLensType] = useState('back'); // use front or back camera
  const [userId, setUserId] = useState('default_user');
  const [showSetupOverlay, setShowSetupOverlay] = useState(true);

  const [summaryVisible, setSummaryVisible] = useState(false);
  const [summaryData, setSummaryData] = useState<SummaryData | null>(null);
  const [maxAngle, setMaxAngle] = useState(15);
  
  // Settings modal state
  const [settingsVisible, setSettingsVisible] = useState(false);
  const [tempMaxAngle, setTempMaxAngle] = useState(15);

  // History modal state
  const [historyVisible, setHistoryVisible] = useState(false);
  const [historySessions, setHistorySessions] = useState<SummaryData[]>([]);
  const [selectedHistoryIndex, setSelectedHistoryIndex] = useState<number | null>(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [currentPage, setCurrentPage] = useState(0); // current page on "Session Summary History"

  const SESSIONS_PER_PAGE = 5;
  const TOTAL_SESSIONS = 15;
  // Total playing time
  const [sessionStartTime, setSessionStartTime] = useState<Date | null>(null);

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
  };

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

  const loadSessionHistory = async () => {
    setIsLoadingHistory(true);
    setCurrentPage(0); // reset to first page
    try {
      // load history session
      const sessions = await CameraxModule.getRecentSessions(userId, TOTAL_SESSIONS);
      
      if (sessions && sessions.length > 0) {
        setHistorySessions(sessions as SummaryData[]);
        setHistoryVisible(true);
      } else {
        setHistorySessions([]);
        setHistoryVisible(true);
      }
    } catch (error) {
      console.error('Error loading session history:', error);
      Alert.alert('Error', 'Failed to load session history. Please try again.');
      setHistorySessions([]);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const handleSessionEnd = async (event: any) => {
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

    let finalDuration = "0s";
    if (sessionStartTime) {
      const endTime = new Date();
      const diffMs = endTime.getTime() - sessionStartTime.getTime();
      finalDuration = formatDuration(diffMs);
    }

    const newSummaryData = {
      heightBreakdown,
      angleBreakdown,
      handPresenceBreakdown,
      handPostureBreakdown,
      posePresenceBreakdown,
      elbowPostureBreakdown,
      userId: eventUserId,
      timestamp,
      sessionDuration: finalDuration,
    };

    setSummaryData(newSummaryData);
    setSummaryVisible(true);
  };

  const closeSummary = () => {
    setSummaryVisible(false);
    setSummaryData(null);
  };

  const closeHistory = () => {
    setHistoryVisible(false);
    setSelectedHistoryIndex(null);
    setCurrentPage(0);
  };

  const openSettings = () => {
    setTempMaxAngle(maxAngle);
    setSettingsVisible(true);
  };

  const closeSettings = () => {
    setSettingsVisible(false);
  };

  const saveSettings = () => {
    setMaxAngle(tempMaxAngle);
    setSettingsVisible(false);
  };


  const getCurrentPageSessions = () => {
    const startIndex = currentPage * SESSIONS_PER_PAGE;
    const endIndex = startIndex + SESSIONS_PER_PAGE;
    return historySessions.slice(startIndex, endIndex);
  };

  const totalPages = Math.ceil(historySessions.length / SESSIONS_PER_PAGE);

  const goToNextPage = () => {
    if (currentPage < totalPages - 1) {
      setCurrentPage(currentPage + 1);
    }
  };

  const goToPrevPage = () => {
    if (currentPage > 0) {
      setCurrentPage(currentPage - 1);
    }
  };

  const renderSummaryContent = (data: SummaryData | null) => {
    if (!data) {
      return <Text>No data available</Text>;
    }

    return (
      <>
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Bow Height</Text>
          <Text>Top: {data.heightBreakdown?.Top?.toFixed(1) || 0}%</Text>
          <Text>Middle: {data.heightBreakdown?.Middle?.toFixed(1) || 0}%</Text>
          <Text>Bottom: {data.heightBreakdown?.Bottom?.toFixed(1) || 0}%</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Bow Angle</Text>
          <Text>Correct: {data.angleBreakdown?.Correct?.toFixed(1) || 0}%</Text>
          <Text>Wrong: {data.angleBreakdown?.Wrong?.toFixed(1) || 0}%</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Hand Posture</Text>
          <Text>Correct: {data.handPostureBreakdown?.Correct?.toFixed(1) || 0}%</Text>
          <Text>Supination: {data.handPostureBreakdown?.Supination?.toFixed(1) || 0}%</Text>
          <Text>Too much pronation: {data.handPostureBreakdown?.['Too much pronation']?.toFixed(1) || 0}%</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Elbow Posture</Text>
          <Text>Correct: {data.elbowPostureBreakdown?.Correct?.toFixed(1) || 0}%</Text>
          <Text>Low elbow: {data.elbowPostureBreakdown?.['Low elbow']?.toFixed(1) || 0}%</Text>
          <Text>Elbow too high: {data.elbowPostureBreakdown?.['Elbow too high']?.toFixed(1) || 0}%</Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.timestamp}>
            <Text style={styles.subTitle}>Total Playing Time: </Text>
            {summaryData?.sessionDuration || "0s"}
          </Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.timestamp}>Completed On: {formattedTimestamp}</Text>
        </View>
      </>
    );
  };

  let formattedTimestamp = "";

  if (summaryData?.timestamp) {
    const dt = new Date(summaryData.timestamp.replace(" ", "T"));

    const weekday = dt.toLocaleString([], { weekday: "short" });
    const date = dt.toLocaleDateString("en-CA");
    const time = dt.toLocaleString([], { hour: "numeric", minute: "2-digit" });

    formattedTimestamp = `${weekday}, ${date}, ${time}`;
  }

  function formatDuration(ms: number): string {
    const totalSeconds = Math.floor(ms / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
    if (minutes > 0) return `${minutes}m ${seconds}s`;
    return `${seconds}s`;
  }


  return (
    <View style={styles.container}>
      {/* Current Session Summary Modal */}
      <Modal visible={summaryVisible} animationType="slide" transparent={true}>
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <ScrollView>
              <Text style={styles.title}>Session Summary</Text>
              {renderSummaryContent(summaryData)}
              <Button title="Close" onPress={closeSummary} />
            </ScrollView>
          </View>
        </View>
      </Modal>

      {/* History Modal */}
      <Modal visible={historyVisible} animationType="slide" transparent={true}>
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <Text style={styles.title}>Session History</Text>
            
            {isLoadingHistory ? (
              <Text style={styles.loadingText}>Loading history...</Text>
            ) : historySessions.length === 0 ? (
              <Text style={styles.noHistoryText}>No previous sessions found</Text>
            ) : (
              <ScrollView>
                {/* Session List */}
                {selectedHistoryIndex === null ? (
                  <>
                    <Text style={styles.historySubtitle}>
                      Showing {currentPage * SESSIONS_PER_PAGE + 1}-{Math.min((currentPage + 1) * SESSIONS_PER_PAGE, historySessions.length)} of {historySessions.length} Sessions
                    </Text>
                    
                    {getCurrentPageSessions().map((session, index) => {
                      const actualIndex = currentPage * SESSIONS_PER_PAGE + index;
                      return (
                        <TouchableOpacity
                          key={actualIndex}
                          style={styles.historyItem}
                          onPress={() => setSelectedHistoryIndex(actualIndex)}
                        >
                          <Text style={styles.historyItemTitle}>
                            Session {actualIndex + 1}
                          </Text>
                          <Text style={styles.historyItemDate}>
                            {session.timestamp}
                          </Text>
                          <Text style={styles.historyItemArrow}>→</Text>
                        </TouchableOpacity>
                      );
                    })}

                    {/* Pagination Controls */}
                    {totalPages > 1 && (
                      <View style={styles.paginationContainer}>
                        <TouchableOpacity
                          onPress={goToPrevPage}
                          disabled={currentPage === 0}
                          style={styles.paginationArrowButton}
                        >
                          <Text style={[
                            styles.paginationArrow,
                            currentPage === 0 && styles.paginationArrowDisabled
                          ]}>
                            ‹
                          </Text>
                        </TouchableOpacity>
                        
                        <Text style={styles.paginationText}>
                          Page {currentPage + 1} of {totalPages}
                        </Text>
                        
                        <TouchableOpacity
                          onPress={goToNextPage}
                          disabled={currentPage === totalPages - 1}
                          style={styles.paginationArrowButton}
                        >
                          <Text style={[
                            styles.paginationArrow,
                            currentPage === totalPages - 1 && styles.paginationArrowDisabled
                          ]}>
                            ›
                          </Text>
                        </TouchableOpacity>
                      </View>
                    )}
                  </>
                ) : (
                  /* Session Detail View */
                  <>
                    <TouchableOpacity
                      style={styles.backButton}
                      onPress={() => setSelectedHistoryIndex(null)}
                    >
                      <Text style={styles.backButtonText}>← Back to List</Text>
                    </TouchableOpacity>
                    
                    <Text style={styles.detailTitle}>
                      Session {selectedHistoryIndex + 1}
                    </Text>
                    
                    {renderSummaryContent(historySessions[selectedHistoryIndex])}
                  </>
                )}
              </ScrollView>
            )}
            
            <View style={{ marginTop: 16 }}>
              <Button title="Close" onPress={closeHistory} />
            </View>
          </View>
        </View>
      </Modal>

      {/* Settings Modal */}
      <Modal visible={settingsVisible} animationType="slide" transparent={true}>
        <View style={styles.modalContainer}>
          <View style={styles.settingsModalContent}>
            <Text style={styles.title}>Settings</Text>
            
            <View style={styles.settingsSection}>
              <Text style={styles.settingsLabel}>Maximum Bow Angle Tolerance (0-90°)</Text>
              <TextInput
                style={styles.settingsInput}
                keyboardType="numeric"
                placeholder="Enter angle (0-90)"
                value={tempMaxAngle.toString()}
                onChangeText={(text) => {
                  const num = parseInt(text) || 0;
                  if (num >= 0 && num <= 90) {
                    setTempMaxAngle(num);
                  }
                }}
              />
              <Text style={styles.settingsHint}>
                Current value: {tempMaxAngle}° (Default: 15°)
              </Text>
            </View>

            <View style={styles.settingsFooter}>
              <TouchableOpacity 
                style={styles.cancelButton} 
                onPress={closeSettings}
              >
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={styles.saveButton} 
                onPress={saveSettings}
              >
                <Text style={styles.saveButtonText}>Save</Text>
              </TouchableOpacity>
            </View>
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
        maxBowAngle={maxAngle}
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
        style={styles.settingsButton}
        onPress={openSettings}
        activeOpacity={0.7}
      >
        <Text style={styles.settingsButtonText}>⚙️</Text>
      </TouchableOpacity>

      {/* History Button */}
      <TouchableOpacity
        style={styles.historyButton}
        onPress={loadSessionHistory}
        activeOpacity={0.7}
        disabled={isLoadingHistory}
      >
        <Text style={styles.historyButtonText}>
          {isLoadingHistory ? '...' : '📋'}
        </Text>
      </TouchableOpacity>
      
      <TouchableOpacity
        style={styles.detectionButton}
          onPress={() => {
            if (showSetupOverlay) setShowSetupOverlay(false);
            if (!isDetectionEnabled) {
              setSessionStartTime(new Date());
            }
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
            <Bullet>Hold phone upright (portrait), ~1-2 ft (30-60 cm) away</Bullet>
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

export default CameraComponent;