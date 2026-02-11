import React, { useState, useEffect } from 'react';
import { View, TouchableOpacity, Modal, Text, ScrollView, Button, TextInput, Alert, Image } from 'react-native';
import { requireNativeViewManager, requireNativeModule } from 'expo-modules-core';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { styles } from '../styles/CameraComponent.styles';

const CameraxView = requireNativeViewManager('Camerax');
const CameraxModule = requireNativeModule('Camerax');

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

interface CategorizedImages {
  bowHeight: string[];
  bowAngle: string[];
  handPosture: string[];
  elbowPosture: string[];
  all: string[];
}

interface CameraComponentProps {
  startDelay?: number;
  onClose: () => void;
  initialHistoryOpen?: boolean;
}

const SESSIONS_PER_PAGE = 5;
const TOTAL_SESSIONS = 15;

const CameraComponent: React.FC<CameraComponentProps> = ({
  startDelay,
  onClose,
  initialHistoryOpen,
}) => {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isDetectionEnabled, setIsDetectionEnabled] = useState(false);
  const [lensType, setLensType] = useState('back'); // use front or back camera
  const [userId, setUserId] = useState('default_user');
  const [showSetupOverlay, setShowSetupOverlay] = useState(false);

  const [summaryVisible, setSummaryVisible] = useState(false);
  const [summaryData, setSummaryData] = useState<SummaryData | null>(null);
  const [maxAngle, setMaxAngle] = useState(18);
  
  // Settings modal state
  const [settingsVisible, setSettingsVisible] = useState(false);
  const [tempMaxAngle, setTempMaxAngle] = useState(18);

  // History modal state
  const [historyVisible, setHistoryVisible] = useState(false);
  const [historySessions, setHistorySessions] = useState<SummaryData[]>([]);
  const [selectedHistoryIndex, setSelectedHistoryIndex] = useState<number | null>(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [currentPage, setCurrentPage] = useState(0); // current page on "Session Summary History"

  // Toolbar state
  const [toolbarExpanded, setToolbarExpanded] = useState(false);

  // Mirror toggle (pass to native view if supported)
  const [isMirrored, setIsMirrored] = useState(false);

  // Learn Postures modal
  const [learnVisible, setLearnVisible] = useState(false);

  // Exit confirmation modal
  const [exitConfirmVisible, setExitConfirmVisible] = useState(false);

  // Detail modal state
  const [selectedDetailSection, setSelectedDetailSection] = useState<string | null>(null);

  const [sessionStartTime, setSessionStartTime] = useState<Date | null>(null);
  const [sessionImages, setSessionImages] = useState<CategorizedImages>({
    bowHeight: [],
    bowAngle: [],
    handPosture: [],
    elbowPosture: [],
    all: []
  });
  const [isLoadingImages, setIsLoadingImages] = useState(false);

  useEffect(() => {
    if (initialHistoryOpen && userId !== 'default_user') {
      loadSessionHistory();
    }
  }, [initialHistoryOpen, userId]);

  useEffect(() => {
    if (initialHistoryOpen) {
      setHistoryVisible(true);
    }
  }, [initialHistoryOpen]);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsCameraActive(true);
    }, startDelay || 100);

    return () => clearTimeout(timer);
  }, [startDelay]);

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

  const toggleCamera = () => {
    setLensType(prev => prev === 'back' ? 'front' : 'back');
  };

  const handleCalibrated = async (event: any) => {
    console.log("Calibration", "Calibration successful");
    setShowSetupOverlay(false);
    setSessionStartTime(new Date());
  };

  const skipCalibration = () => {
    console.log("Calibration", "Skipping Calibration");
    setShowSetupOverlay(false);
    setSessionStartTime(new Date());
  };

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

  const categorizeImages = (imagePaths: string[]): CategorizedImages => {
    const categorized: CategorizedImages = {
      bowHeight: [],
      bowAngle: [],
      handPosture: [],
      elbowPosture: [],
      all: imagePaths
    };

    console.log('=== CATEGORIZING IMAGES ===');
    imagePaths.forEach((path, index) => {
      const fileName = path.toLowerCase();
      console.log(`Image ${index}: ${path.split('/').pop()}`);
      
      // Bow Height: bow_too_high, correct_bow
      if (fileName.includes('bow_too_high') || fileName.includes('correct_bow')) {
        categorized.bowHeight.push(path);
        console.log(`  -> Categorized as bowHeight`);
      }
      
      // Bow Angle: correct_angle, incorrect_angle
      if (fileName.includes('correct_angle') || fileName.includes('incorrect_angle')) {
        categorized.bowAngle.push(path);
        console.log(`  -> Categorized as bowAngle`);
      }
      
      // Hand Posture: good_pronation, supination
      if (fileName.includes('good_pronation') || fileName.includes('supination')) {
        categorized.handPosture.push(path);
        console.log(`  -> Categorized as handPosture`);
      }
      
      // Elbow Posture: good_elbow, high_elbow, low_elbow
      if (fileName.includes('good_elbow') || fileName.includes('high_elbow') || fileName.includes('low_elbow')) {
        categorized.elbowPosture.push(path);
        console.log(`  -> Categorized as elbowPosture`);
      }
    });

    console.log('=== CATEGORIZATION SUMMARY ===');
    console.log('Bow Height images:', categorized.bowHeight.length);
    console.log('Bow Angle images:', categorized.bowAngle.length);
    console.log('Hand Posture images:', categorized.handPosture.length);
    console.log('Elbow Posture images:', categorized.elbowPosture.length);
    console.log('Total images:', categorized.all.length);

    return categorized;
  };

  const getImagesForSection = (sectionKey: string): string[] => {
    switch (sectionKey) {
      case 'bowHeight':
        return sessionImages.bowHeight;
      case 'bowAngle':
        return sessionImages.bowAngle;
      case 'handPosture':
        return sessionImages.handPosture;
      case 'elbowPosture':
        return sessionImages.elbowPosture;
      default:
        return sessionImages.all;
    }
  };

  const loadSessionImages = async (session: SummaryData) => {
    if (!session.timestamp || !session.userId) {
      console.error('=== MISSING SESSION INFO ===');
      console.log('Session data:', JSON.stringify(session, null, 2));
      Alert.alert('Error', 'Missing session information');
      return;
    }

    console.log('=== LOADING IMAGES ===');
    console.log('Session userId:', session.userId);
    console.log('Session timestamp:', session.timestamp);
    console.log('Session timestamp type:', typeof session.timestamp);
    console.log('Full session data:', JSON.stringify(session, null, 2));

    setIsLoadingImages(true);
    try {
      const images = await CameraxModule.getSessionImages(
        session.userId,
        session.timestamp
      );
      
      console.log('=== RAW IMAGES LOADED ===');
      console.log('Number of images:', images.length);
      if (images.length > 0) {
        console.log('First 3 image paths:');
        images.slice(0, 3).forEach((img: string, idx: number) => {
          console.log(`  ${idx + 1}. ${img}`);
        });
        
        const categorized = categorizeImages(images);
        setSessionImages(categorized);
      } else {
        console.warn('No images found for this session');
        console.log('Expected path format: /Documents/sessions/{userId}/{timestamp}/*.png');
        setSessionImages({
          bowHeight: [],
          bowAngle: [],
          handPosture: [],
          elbowPosture: [],
          all: []
        });
      }
    } catch (error) {
      console.error('=== ERROR LOADING IMAGES ===');
      console.error('Error:', error);
      Alert.alert('Error', 'Failed to load session images');
      setSessionImages({
        bowHeight: [],
        bowAngle: [],
        handPosture: [],
        elbowPosture: [],
        all: []
      });
    } finally {
      setIsLoadingImages(false);
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

    console.log('=== SESSION END EVENT ===');
    console.log('Event userId:', eventUserId);
    console.log('Event timestamp:', timestamp);
    console.log('Event timestamp type:', typeof timestamp);

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

  // Detail modal helpers
  const openDetail = (sectionKey: string) => {
    console.log('=== OPENING DETAIL ===');
    console.log('Section:', sectionKey);
    setSelectedDetailSection(sectionKey);
    
    // load image
    if (summaryData) {
      console.log('Loading images for current summary data');
      loadSessionImages(summaryData);
    } else {
      console.warn('No summaryData available');
    }
  };

  const closeDetail = () => {
    setSelectedDetailSection(null);
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

  // Helper function to normalize percentages
  function normalizePercentages(values: number[]): number[] {
    const total = values.reduce((sum, v) => sum + (v || 0), 0);
    if (total <= 0) {
      return values.map(() => 0);
    }
    const raw = values.map(v => ((v || 0) / total) * 100);
    const rounded = raw.map(v => Math.round(v));
    const diff = 100 - rounded.reduce((sum, v) => sum + v, 0);

    if (diff !== 0) {
      let idxMax = 0;
      for (let i = 1; i < rounded.length; i++) {
        if (rounded[i] > rounded[idxMax]) idxMax = i;
      }
      rounded[idxMax] += diff;
    }

    return rounded;
  }

  // Render summary content with normalized percentages and colors (UPDATED)
  const renderSummaryContent = (data: SummaryData | null) => {
    if (!data) {
      return <Text>No data available</Text>;
    }

    // Normalized metric values for Bow Height
    const heightTopRaw = data.heightBreakdown?.Top ?? 0;
    const heightMiddleRaw = data.heightBreakdown?.Middle ?? 0;
    const heightBottomRaw = data.heightBreakdown?.Bottom ?? 0;
    const [heightTopPct, heightMiddlePct, heightBottomPct] = normalizePercentages([
      heightTopRaw,
      heightMiddleRaw,
      heightBottomRaw,
    ]);

    // Normalized metric values for Bow Angle
    const angleCorrectRaw = data.angleBreakdown?.Correct ?? 0;
    const angleWrongRaw = data.angleBreakdown?.Wrong ?? 0;
    const [angleCorrectPct, angleWrongPct] = normalizePercentages([
      angleCorrectRaw,
      angleWrongRaw,
    ]);

    // Normalized metric values for Hand Posture
    const handCorrectRaw = data.handPostureBreakdown?.Correct ?? 0;
    const handSupinationRaw = data.handPostureBreakdown?.Supination ?? 0;
    const handTooMuchRaw = data.handPostureBreakdown?.['Too much pronation'] ?? 0;
    const [handCorrectPct, handSupinationPct, handTooMuchPct] = normalizePercentages([
      handCorrectRaw,
      handSupinationRaw,
      handTooMuchRaw,
    ]);

    // Normalized metric values for Elbow Posture
    const elbowCorrectRaw = data.elbowPostureBreakdown?.Correct ?? 0;
    const elbowLowRaw = data.elbowPostureBreakdown?.['Low elbow'] ?? 0;
    const elbowHighRaw = data.elbowPostureBreakdown?.['Elbow too high'] ?? 0;
    const [elbowCorrectPct, elbowLowPct, elbowHighPct] = normalizePercentages([
      elbowCorrectRaw,
      elbowLowRaw,
      elbowHighRaw,
    ]);

    // Format timestamp
    let formattedTimestamp = "";
    if (data.timestamp) {
      const dt = new Date(data.timestamp.replace(" ", "T"));
      const weekday = dt.toLocaleString([], { weekday: "short" });
      const date = dt.toLocaleDateString("en-CA");
      const time = dt.toLocaleString([], { hour: "numeric", minute: "2-digit" });
      formattedTimestamp = `${weekday}, ${date}, ${time}`;
    }

    return (
      <>
        <View style={styles.section}>
          <View style={styles.sectionHeaderRow}>
            <Text style={styles.sectionTitle}>Bow Height</Text>
            <TouchableOpacity
              style={styles.viewMoreButton}
              onPress={() => openDetail('bowHeight')}
            >
              <Text style={styles.viewMoreButtonText}>View more</Text>
            </TouchableOpacity>
          </View>
          
          <View style={styles.metricRow}>
            <View style={styles.metricDotWarning} />
            <Text style={styles.metricLabel}>Top</Text>
            <Text style={styles.metricPercent}>{heightTopPct.toFixed(0)}%</Text>
          </View>
          <View style={styles.metricRow}>
            <View style={styles.metricDotGood} />
            <Text style={styles.metricLabel}>Middle (ideal)</Text>
            <Text style={styles.metricPercent}>{heightMiddlePct.toFixed(0)}%</Text>
          </View>
          <View style={styles.metricRow}>
            <View style={styles.metricDotWarning} />
            <Text style={styles.metricLabel}>Bottom</Text>
            <Text style={styles.metricPercent}>{heightBottomPct.toFixed(0)}%</Text>
          </View>
        </View>

        <View style={styles.section}>
          <View style={styles.sectionHeaderRow}>
            <Text style={styles.sectionTitle}>Bow Angle</Text>
            <TouchableOpacity
              style={styles.viewMoreButton}
              onPress={() => openDetail('bowAngle')}
            >
              <Text style={styles.viewMoreButtonText}>View more</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.metricRow}>
            <View style={styles.metricDotGood} />
            <Text style={styles.metricLabel}>Parallel with bridge</Text>
            <Text style={styles.metricPercent}>{angleCorrectPct.toFixed(0)}%</Text>
          </View>
          <View style={styles.metricRow}>
            <View style={styles.metricDotWarning} />
            <Text style={styles.metricLabel}>Tilted</Text>
            <Text style={styles.metricPercent}>{angleWrongPct.toFixed(0)}%</Text>
          </View>
        </View>

        <View style={styles.section}>
          <View style={styles.sectionHeaderRow}>
            <Text style={styles.sectionTitle}>Hand Posture</Text>
            <TouchableOpacity
              style={styles.viewMoreButton}
              onPress={() => openDetail('handPosture')}
            >
              <Text style={styles.viewMoreButtonText}>View more</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.metricRow}>
            <View style={styles.metricDotGood} />
            <Text style={styles.metricLabel}>Natural pronation</Text>
            <Text style={styles.metricPercent}>{handCorrectPct.toFixed(0)}%</Text>
          </View>
          <View style={styles.metricRow}>
            <View style={styles.metricDotWarning} />
            <Text style={styles.metricLabel}>Supination</Text>
            <Text style={styles.metricPercent}>{handSupinationPct.toFixed(0)}%</Text>
          </View>
          <View style={styles.metricRow}>
            <View style={styles.metricDotWarning} />
            <Text style={styles.metricLabel}>Too much pronation</Text>
            <Text style={styles.metricPercent}>{handTooMuchPct.toFixed(0)}%</Text>
          </View>
        </View>

        <View style={styles.section}>
          <View style={styles.sectionHeaderRow}>
            <Text style={styles.sectionTitle}>Elbow Posture</Text>
            <TouchableOpacity
              style={styles.viewMoreButton}
              onPress={() => openDetail('elbowPosture')}
            >
              <Text style={styles.viewMoreButtonText}>View more</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.metricRow}>
            <View style={styles.metricDotGood} />
            <Text style={styles.metricLabel}>Natural</Text>
            <Text style={styles.metricPercent}>{elbowCorrectPct.toFixed(0)}%</Text>
          </View>
          <View style={styles.metricRow}>
            <View style={styles.metricDotWarning} />
            <Text style={styles.metricLabel}>Elbow too low</Text>
            <Text style={styles.metricPercent}>{elbowLowPct.toFixed(0)}%</Text>
          </View>
          <View style={styles.metricRow}>
            <View style={styles.metricDotWarning} />
            <Text style={styles.metricLabel}>Elbow too high</Text>
            <Text style={styles.metricPercent}>{elbowHighPct.toFixed(0)}%</Text>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.timestamp}>
            <Text style={styles.subTitle}>Total Playing Time: </Text>
            {data?.sessionDuration || "0s"}
          </Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.timestamp}>Completed On: {formattedTimestamp}</Text>
        </View>
      </>
    );
  };

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
                      const displayNumber = historySessions.length - actualIndex;
                      return (
                        <TouchableOpacity
                          key={actualIndex}
                          style={styles.historyItem}
                          onPress={() => {
                            console.log('=== HISTORY ITEM CLICKED ===');
                            console.log('Actual index:', actualIndex);
                            console.log('Session:', JSON.stringify(session, null, 2));
                            setSelectedHistoryIndex(actualIndex);
                            loadSessionImages(historySessions[actualIndex]);
                          }}
                        >
                          <Text style={styles.historyItemTitle}>
                            Session {displayNumber}
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
                      Session {historySessions.length - selectedHistoryIndex}
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

      {/* Exit Confirmation Modal */}
      <Modal visible={exitConfirmVisible} animationType="fade" transparent={true}>
        <View style={styles.modalContainer}>
          <View style={styles.exitModalContent}>
            <Text style={styles.exitModalTitle}>Exit without saving the session?</Text>

            <View style={styles.exitModalButtons}>
              <TouchableOpacity
                style={styles.exitModalYesButton}
                onPress={() => {
                  // stop detection safely
                  setIsDetectionEnabled(false);
                  setShowSetupOverlay(false);

                  // close warning modal
                  setExitConfirmVisible(false);

                  // exit camera screen
                  onClose();
                }}
                activeOpacity={0.8}
              >
                <Text style={styles.exitModalYesText}>Yes</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.exitModalNoButton}
                onPress={() => setExitConfirmVisible(false)}
                activeOpacity={0.8}
              >
                <Text style={styles.exitModalNoText}>No</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Learn Postures Modal */}
      <Modal visible={learnVisible} animationType="slide" transparent={true}>
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <ScrollView>
              <Text style={styles.title}>Learn Postures</Text>

              <Text style={styles.detailText}>
                Add posture education content here (images / tips / examples).
              </Text>

              <Button title="Close" onPress={() => setLearnVisible(false)} />
            </ScrollView>
          </View>
        </View>
      </Modal>

      {/* Detail Modal for "View more" */}
      <Modal
        visible={!!selectedDetailSection}
        animationType="fade"
        transparent={true}
      >
        <View style={styles.modalContainer}>
          <View style={styles.detailModalContent}>
            <ScrollView>
              <Text style={styles.title}>
                {selectedDetailSection === 'bowHeight' && 'Bow Height Details'}
                {selectedDetailSection === 'bowAngle' && 'Bow Angle Details'}
                {selectedDetailSection === 'handPosture' && 'Hand Posture Details'}
                {selectedDetailSection === 'elbowPosture' && 'Elbow Posture Details'}
              </Text>

              {/* Images section */}
              {isLoadingImages ? (
                <View style={styles.detailImagePlaceholder}>
                  <Text style={styles.detailImageText}>Loading images...</Text>
                </View>
              ) : getImagesForSection(selectedDetailSection || '').length > 0 ? (
                <>
                  <Text style={styles.imageCountText}>
                    {getImagesForSection(selectedDetailSection || '').length} images found
                  </Text>
                  <ScrollView horizontal showsHorizontalScrollIndicator={true}>
                    {getImagesForSection(selectedDetailSection || '').map((imagePath, index) => {
                      console.log(`Rendering image ${index}:`, imagePath);
                      return (
                        <View key={index} style={styles.imageContainer}>
                          <Image
                            source={{ uri: `file://${imagePath}` }}
                            style={styles.detailImage}
                            resizeMode="contain"
                            onError={(error) => {
                              console.error(`Image ${index} failed to load:`, error.nativeEvent.error);
                            }}
                            onLoad={() => {
                              console.log(`Image ${index} loaded successfully`);
                            }}
                          />
                          <Text style={styles.imageLabel}>
                            {imagePath.split('/').pop()?.split('.')[0].replace(/_/g, ' ')}
                          </Text>
                        </View>
                      );
                    })}
                  </ScrollView>
                </>
              ) : (
                <View style={styles.detailImagePlaceholder}>
                  <Text style={styles.detailImageText}>
                    No images available for this section
                  </Text>
                  <Text style={styles.detailImageSubtext}>
                    Total images loaded: {sessionImages.all.length}
                  </Text>
                </View>
              )}

              <Text style={styles.detailText}>
                This panel shows screenshots from your session categorized by posture type.
                Review your technique across different moments during practice.
              </Text>

              <Button title="Close" onPress={closeDetail} />
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
        onCalibrated={handleCalibrated}
        skipCalibration={!showSetupOverlay}
        maxBowAngle={maxAngle}
      />

      {/* Top-left menu + dropdown */}
      <View style={styles.topLeftMenuArea}>
        <TouchableOpacity
          style={styles.menuFab}
          onPress={() => setToolbarExpanded(v => !v)}
          activeOpacity={0.8}
        >
          <Text style={styles.menuFabIcon}>{toolbarExpanded ? '✕' : '⠿'}</Text>
        </TouchableOpacity>

        {toolbarExpanded && (
          <View style={styles.menuPanel}>
            <TouchableOpacity
              style={styles.menuItem}
              onPress={() => {
                toggleCamera();
                setToolbarExpanded(false);
              }}
              activeOpacity={0.8}
            >
              <Text style={styles.menuItemIcon}>📷</Text>
              <Text style={styles.menuItemText}>Flip camera</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.menuItem}
              onPress={() => setIsMirrored(v => !v)}
              activeOpacity={0.8}
            >
              <Text style={styles.menuItemIcon}>🪞</Text>
              <Text style={styles.menuItemText}>
                Mirror the view {isMirrored ? '(On)' : '(Off)'}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.menuItem}
              onPress={() => {
                openSettings();
                setToolbarExpanded(false);
              }}
              activeOpacity={0.8}
            >
              <Text style={styles.menuItemIcon}>🎚️</Text>
              <Text style={styles.menuItemText}>Threshold adjust</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.menuItem}
              onPress={() => {
                loadSessionHistory();
                setToolbarExpanded(false);
              }}
              disabled={isLoadingHistory}
              activeOpacity={0.8}
            >
              <Text style={styles.menuItemIcon}>📈</Text>
              <Text style={styles.menuItemText}>
                {isLoadingHistory ? 'Loading…' : 'Session history'}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.menuItem}
              onPress={() => {
                setLearnVisible(true);
                setToolbarExpanded(false);
              }}
              activeOpacity={0.8}
            >
              <Text style={styles.menuItemIcon}>📖</Text>
              <Text style={styles.menuItemText}>Learn postures</Text>
            </TouchableOpacity>
          </View>
        )}
      </View>
      
      <TouchableOpacity
        style={styles.closeButton}
        onPress={() => {
          if (isDetectionEnabled) setExitConfirmVisible(true);
          else onClose();
        }}
        activeOpacity={0.7}
      >
        <Text style={styles.closeButtonText}>✕</Text>
      </TouchableOpacity>
      
      <TouchableOpacity
        style={styles.detectionButton}
        onPress={() => {
          if (!isDetectionEnabled) {
            setShowSetupOverlay(true);
          }
          setIsDetectionEnabled(!isDetectionEnabled);
          console.log("DetectionEnabled: ", !isDetectionEnabled);
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
            <Bullet>Point your cello towards the camera</Bullet>

            <TouchableOpacity style={styles.readyBtn} onPress={skipCalibration} activeOpacity={0.9}>
              <Text style={styles.readyText}>Skip</Text>
            </TouchableOpacity>
          </View>
        </>
      )}
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

export default CameraComponent;