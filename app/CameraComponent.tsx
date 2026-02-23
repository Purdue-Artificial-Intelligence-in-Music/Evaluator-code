import React, { useState, useEffect } from 'react';
import { View, TouchableOpacity, Modal, Text, ScrollView, Button, TextInput, Alert, Image, Dimensions } from 'react-native';
import { requireNativeViewManager, requireNativeModule } from 'expo-modules-core';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { styles } from '../styles/CameraComponent.styles';
import { ICONS } from '../styles/CameraComponent.styles';
import LearnPosture from './LearnPosture'; // adjust path as needed

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
  durationSeconds?: number;
  durationFormatted?: string;
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
  initialSetupOpen
}) => {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isDetectionEnabled, setIsDetectionEnabled] = useState(false);
  const [lensType, setLensType] = useState('front'); // use front or back camera
  const [userId, setUserId] = useState('default_user');
  // const [showSetupOverlay, setShowSetupOverlay] = useState(false);
  const [showSetupOverlay, setShowSetupOverlay] = useState(initialSetupOpen);
  const [showCountdown, setShowCountdown] = useState(false);
  const [countdownVal, setCountdownVal] = useState(3)
  const [countdownLength] = useState(3)
  const [isStartDetectionVisible, setStartDetectionVisible] = useState(true)

  const [summaryVisible, setSummaryVisible] = useState(false);
  const [summaryData, setSummaryData] = useState<SummaryData | null>(null);
  const [maxAngle, setMaxAngle] = useState(18);

  // Settings modal state
  const [settingsVisible, setSettingsVisible] = useState(false);
  const [tempMaxAngle, setTempMaxAngle] = useState(18);

  // History modal state
  const [historyVisible, setHistoryVisible] = useState(!!initialHistoryOpen);
  const [historySessions, setHistorySessions] = useState<SummaryData[]>([]);
  const [selectedHistoryIndex, setSelectedHistoryIndex] = useState<number | null>(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [currentPage, setCurrentPage] = useState(0); // current page on "Session Summary History"
  const historyMode = !!initialHistoryOpen || historyVisible;

  // Toolbar state
  const [toolbarExpanded, setToolbarExpanded] = useState(false);

  // Mirror toggle (pass to native view if supported)
  const [isMirrored, setIsMirrored] = useState(false);

  // Learn Postures modal
  const [learnVisible, setLearnVisible] = useState(false);
  const [learnActiveTab, setLearnActiveTab] = useState('overall');

  // Exit confirmation modal
  const [exitConfirmVisible, setExitConfirmVisible] = useState(false);

  // Detail modal state
  const [selectedDetailSection, setSelectedDetailSection] = useState<string | null>(null);

  // Timer
  const [elapsedTime, setElapsedTime] = useState<string>('00:00');

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
      setIsCameraActive(false);
      setIsDetectionEnabled(false);
      setHistoryVisible(true);
    }
  }, [initialHistoryOpen]);

  useEffect(() => {
     if (initialHistoryOpen) {
      setIsCameraActive(false);
      return;
    }

    const timer = setTimeout(() => {
      setIsCameraActive(true);
    }, startDelay || 100);

    return () => clearTimeout(timer);
  }, [startDelay, initialHistoryOpen]);

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

  // Timer
  useEffect(() => {
    if (!isDetectionEnabled || !sessionStartTime) {
      setElapsedTime('00:00');
      return;
    }

    const interval = setInterval(() => {
      const diffMs = new Date().getTime() - sessionStartTime.getTime();
      const totalSeconds = Math.floor(diffMs / 1000);
      const minutes = Math.floor(totalSeconds / 60);
      const seconds = totalSeconds % 60;
      setElapsedTime(
        `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
      );
    }, 1000);

    return () => clearInterval(interval);
  }, [isDetectionEnabled, sessionStartTime]);

  const toggleCamera = () => {
    setLensType(prev => {
      if (prev === 'front') setIsMirrored(false); // reset mirroring option after switching to back cam
      return prev === 'back' ? 'front' : 'back';
    });
  };

  const handleMirrorToggle = () => {
    if (lensType !== 'front') {
      Alert.alert('', 'Mirroring option only available for front camera');
      return;
    }
    setIsMirrored(v => !v);
  };

  // Commented out Calibration code as we are using a timer, can be deleted.
  /*const handleCalibrated = async (event: any) => {
    console.log("Calibration", "Calibration successful");
    setShowSetupOverlay(false);
    setSessionStartTime(new Date());
  };

  const skipCalibration = () => {
    console.log("Calibration", "Skipping Calibration");
    setShowSetupOverlay(false);
    setSessionStartTime(new Date());
  };*/

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
      timestamp,
      durationSeconds,
      durationFormatted
    } = event.nativeEvent;

    console.log('=== SESSION END EVENT ===');
    console.log('Event userId:', eventUserId);
    console.log('Event timestamp:', timestamp);
    console.log('Event timestamp type:', typeof timestamp);
    console.log('Event durationFormatted:', durationFormatted);
    console.log('Event durationSeconds:', durationSeconds);

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
      durationSeconds: durationSeconds || 0,
      durationFormatted: durationFormatted || finalDuration,
    };
    //navigation.navigate('SessionSummary', { summaryData: newSummaryData });
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

    // If CameraComponent was opened ONLY to show history from HomeScreen,
    // close the whole CameraComponent overlay so HomeScreen is clickable again.
    if (initialHistoryOpen) {
      onClose();
    }
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
            <Text style={styles.sectionTitle}>Bow Contact Point</Text>
            <TouchableOpacity
              style={styles.viewMoreButton}
              onPress={() => openDetail('bowHeight')}
            >
              <Text style={styles.viewMoreButtonText}>View more</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.metricRow}>
            <Image source={ICONS.tick_square} style={styles.metricDotGood} />
            <Text style={styles.metricLabel}>Good</Text>
            <Text style={styles.metricPercent}>{heightMiddlePct.toFixed(0)}%</Text>
          </View>
          <View style={styles.metricRow}>
            <Image source={ICONS.wrong} style={styles.metricDotWarning} />
            <Text style={styles.metricLabel}>Too close to fingerboard</Text>
            <Text style={styles.wrongMetricPercent}>{heightTopPct.toFixed(0)}%</Text>
          </View>
          <View style={styles.metricRow}>
            <Image source={ICONS.wrong} style={styles.metricDotWarning} />
            <Text style={styles.metricLabel}>Too close to the bridge</Text>
            <Text style={styles.wrongMetricPercent}>{heightBottomPct.toFixed(0)}%</Text>
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
            <Image source={ICONS.tick_square} style={styles.metricDotGood} />
            <Text style={styles.metricLabel}>Straight bow</Text>
            <Text style={styles.metricPercent}>{angleCorrectPct.toFixed(0)}%</Text>
          </View>
          <View style={styles.metricRow}>
            <Image source={ICONS.wrong} style={styles.metricDotWarning} />
            <Text style={styles.metricLabel}>Crooked bow</Text>
            <Text style={styles.wrongMetricPercent}>{angleWrongPct.toFixed(0)}%</Text>
          </View>
        </View>

        <View style={styles.section}>
          <View style={styles.sectionHeaderRow}>
            <Text style={styles.sectionTitle}>Bow Hand Position</Text>
            <TouchableOpacity
              style={styles.viewMoreButton}
              onPress={() => openDetail('handPosture')}
            >
              <Text style={styles.viewMoreButtonText}>View more</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.metricRow}>
            <Image source={ICONS.tick_square} style={styles.metricDotGood} />
            <Text style={styles.metricLabel}>Optimal pronation</Text>
            <Text style={styles.metricPercent}>{handCorrectPct.toFixed(0)}%</Text>
          </View>
          <View style={styles.metricRow}>
            <Image source={ICONS.wrong} style={styles.metricDotWarning} />
            <Text style={styles.metricLabel}>Supination</Text>
            <Text style={styles.wrongMetricPercent}>{handSupinationPct.toFixed(0)}%</Text>
          </View>
          <View style={styles.metricRow}>
            <Image source={ICONS.wrong} style={styles.metricDotWarning} />
            <Text style={styles.metricLabel}>Over pronate</Text>
            <Text style={styles.wrongMetricPercent}>{handTooMuchPct.toFixed(0)}%</Text>
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
            <Image source={ICONS.tick_square} style={styles.metricDotGood} />
            <Text style={styles.metricLabel}>Good</Text>
            <Text style={styles.metricPercent}>{elbowCorrectPct.toFixed(0)}%</Text>
          </View>
          <View style={styles.metricRow}>
            <Image source={ICONS.wrong} style={styles.metricDotWarning} />
            <Text style={styles.metricLabel}>Too low</Text>
            <Text style={styles.wrongMetricPercent}>{elbowLowPct.toFixed(0)}%</Text>
          </View>
          <View style={styles.metricRow}>
            <Image source={ICONS.wrong} style={styles.metricDotWarning} />
            <Text style={styles.metricLabel}>Too high</Text>
            <Text style={styles.wrongMetricPercent}>{elbowHighPct.toFixed(0)}%</Text>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.timestamp}>
            <Text style={styles.subTitle}>Total Playing Time: </Text>
            {data?.durationFormatted || data?.sessionDuration || "0s"}
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
       <LearnPosture
         visible={learnVisible}
         onClose={() => setLearnVisible(false)}
       />


      {/* Detail Modal for "View more" */}
      <Modal
        visible={!!selectedDetailSection}
        animationType="fade"
        transparent={true}
      >
          <View style={styles.modalContainer}>
            <View style={styles.detailModalContent}>
              <ScrollView>

               {selectedDetailSection === 'bowHeight' && (() => {
                 const src = require('../assets/postures/2.Bow Contact Point examples.jpg');
                 const resolved = Image.resolveAssetSource(src);
                 const modalWidth = Dimensions.get('window').width - 80;
                 const imgHeight = resolved?.width ? (modalWidth * resolved.height) / resolved.width : 200;
                 return (
                   <>
                     <Text style={styles.title}>Bow Contact Point</Text>
                     <Text style={styles.detailText}>
                       The bow should be placed between the fingerboard and the bridge.
                       Especially for beginners, it is important to check that the bow is not too close to the fingerboard (too high) or too close to the bridge (too low).
                       Advanced players may intentionally place the bow higher or lower to change tone for musical purposes.
                     </Text>
                     <Image
                       source={src}
                       style={[styles.detailImage, { height: imgHeight }]}
                       resizeMode="contain"
                     />
                   </>
                 );
               })()}

               {selectedDetailSection === 'bowAngle' && (() => {
                 const src = require('../assets/postures/3.bow angle examples.jpg');
                 const resolved = Image.resolveAssetSource(src);
                 const modalWidth = Dimensions.get('window').width - 80;
                 const imgHeight = resolved?.width ? (modalWidth * resolved.height) / resolved.width : 200;
                 return (
                   <>
                     <Text style={styles.title}>Bow Angle</Text>
                     <Text style={styles.detailText}>
                       The bow should remain perpendicular to the strings and parallel to the bridge.
                       Each string (A, D, G, and C) has its own correct bow angle, and this angle must adjust as you move from one string to another.
                     </Text>
                     <Image
                       source={src}
                       style={[styles.detailImage, { height: imgHeight }]}
                       resizeMode="contain"
                     />
                   </>
                 );
               })()}

               {selectedDetailSection === 'handPosture' && (() => {
                 const src = require('../assets/postures/4. Bow han example.jpg');
                 const resolved = Image.resolveAssetSource(src);
                 const modalWidth = Dimensions.get('window').width - 80;
                 const imgHeight = resolved?.width ? (modalWidth * resolved.height) / resolved.width : 200;
                 return (
                   <>
                     <Text style={styles.title}>Bow Hand Position</Text>
                     <Text style={styles.detailText}>
                       The standard bow hand posture may vary slightly depending on hand size, but the bow should be held with a slight tilt toward the left hand side. This tilt of the hand wrist and elbow is called pronation. Especially at the tip of the bow, both the hand and wrist should remain pronated.
                       If the hand and wrist tilt to the right, this is called supination, which leads to poor tone and reduced control.
                     </Text>
                     <Image
                       source={src}
                       style={[styles.detailImage, { height: imgHeight }]}
                       resizeMode="contain"
                     />
                   </>
                 );
               })()}

               {selectedDetailSection === 'elbowPosture' && (() => {
                 const src = require('../assets/postures/5. elbow example.jpg');
                 const resolved = Image.resolveAssetSource(src);
                 const modalWidth = Dimensions.get('window').width - 80;
                 const imgHeight = resolved?.width ? (modalWidth * resolved.height) / resolved.width : 200;
                 return (
                   <>
                     <Text style={styles.title}>Elbow Posture</Text>
                     <Text style={styles.detailText}>
                       At the frog, the elbow should be comfortably lowered and positioned close to the cello body.
                     </Text>
                     <Image
                       source={src}
                       style={[styles.detailImage, { height: imgHeight }]}
                       resizeMode="contain"
                     />
                   </>
                 );
               })()}

                <Button title="Close" onPress={closeDetail} />
              </ScrollView>
            </View>
          </View>
        </Modal>

      {!initialHistoryOpen && (
        <CameraxView
          style={styles.camera}
          userId={userId}
          cameraActive={isCameraActive}
          detectionEnabled={isDetectionEnabled}
          lensType={lensType}
          onSessionEnd={handleSessionEnd}
          /*onCalibrated={handleCalibrated} // Caulibration code, Can be deleted.
          skipCalibration={!showSetupOverlay}*/
          maxBowAngle={maxAngle}
          flip={isMirrored}
        />
      )}
      {!initialHistoryOpen && (
        <>
        {/* Top-left menu + dropdown */}
        <View style={styles.topLeftMenuArea}>
          <TouchableOpacity
            style={styles.menuFab}
            onPress={() => setToolbarExpanded(v => !v)}
            activeOpacity={0.8}
          >
            <Image
              source={toolbarExpanded ? ICONS.exit : ICONS.tools}
              style={styles.menuFabIconImg}
            />
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
                <Image source={ICONS.flip_camera} style={styles.menuItemIconImg} />
                <Text style={styles.menuItemText}>Flip camera</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[
                  styles.menuItem,
                  lensType !== 'front' && { opacity: 0.4 }  // Display grey text to indicate "not available"
                ]}
                onPress={handleMirrorToggle}
                activeOpacity={0.8}
              >
                <Image source={ICONS.mirror} style={styles.menuItemIconImg} />
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
                <Image source={ICONS.adjust_threshold} style={styles.menuItemIconImg} />
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
                <Image source={ICONS.session_summary} style={styles.menuItemIconImg} />
                <Text style={styles.menuItemText}>
                  {isLoadingHistory ? 'Loading…' : 'Session history'}
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.menuItem}
                onPress={() => {
                  setLearnActiveTab('overall');
                  setLearnVisible(true);
                  setToolbarExpanded(false);
                }}
                activeOpacity={0.8}
              >
                <Image source={ICONS.instructions} style={styles.menuItemIconImg} />
                <Text style={styles.menuItemText}>Learn postures</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>
        </>
      )}

      {!historyMode && (
        <TouchableOpacity
          style={styles.closeButton}
          onPress={() => {
            if (isDetectionEnabled) setExitConfirmVisible(true);
            else onClose();
          }}
          activeOpacity={0.7}
        >
          <Image source={ICONS.exit} style={styles.closeButtonIconImg} />
        </TouchableOpacity>
      )}
      
      {/* Show start/stop button when setup overlay is off. Setup overlay has its own start button */}
      {!initialHistoryOpen && !showSetupOverlay && (
        <>
        {isDetectionEnabled && (
          <View style={styles.timerDisplay}>
            <Text style={styles.timerText}>{elapsedTime}</Text>
          </View>
        )}
        { isStartDetectionVisible && <TouchableOpacity
          style={styles.detectionButton}
          onPress={() => {
            setStartDetectionVisible(false);
            if (!isDetectionEnabled) {
              // setShowSetupOverlay(true);
              setShowCountdown(true);
              let count = countdownLength;
              setCountdownVal(countdownLength);

              let interval = setInterval(() => {
                  count--;
                  setCountdownVal(count)
                  if (count <= 0) {
                    clearInterval(interval);
                    setIsDetectionEnabled(!isDetectionEnabled);
                    setSessionStartTime(new Date());
                    setStartDetectionVisible(true);
                    setShowCountdown(false);
                    console.log("DetectionEnabled: ", !isDetectionEnabled);
                  }
              }, 1000);
            } else {
              setIsDetectionEnabled(!isDetectionEnabled);
              console.log("DetectionEnabled: ", !isDetectionEnabled);
            }
          }}
        >
          <Text style={styles.buttonText}>
            {isDetectionEnabled ? 'Stop Detection' : 'Start Detection'}
          </Text>
        </TouchableOpacity>}
        </>
      )}

      {!historyMode && showSetupOverlay && (
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
            <Bullet>Place phone upright (portrait), ~1-2 ft (30-60 cm) away</Bullet>
            <Bullet>Center yourself and the cello inside the outline</Bullet>
            <Bullet>Keep the bridge near the dotted line</Bullet>
            <Bullet>Point your cello towards the camera</Bullet>

            {/* Start detection button inside setup overlay section */}
            <TouchableOpacity
              style={[styles.readyBtn, { position: 'relative', bottom: 0, marginTop: 16 }]}
              onPress={() => {
                setShowSetupOverlay(false);
                setShowCountdown(true);
                let count = countdownLength;
                setCountdownVal(countdownLength);
                let interval = setInterval(() => {
                  count--;
                  setCountdownVal(count);
                  if (count <= 0) {
                    clearInterval(interval);
                    setIsDetectionEnabled(true);
                    setSessionStartTime(new Date());
                    setShowCountdown(false);
                  }
                }, 1000);
              }}
              activeOpacity={0.8}
            >
              <Text style={styles.buttonText}>Start Detection</Text>
            </TouchableOpacity>
          </View>
        </>
      )}
      {!historyMode && showCountdown && (
        <>
        {/* dark overlay */}
        <View pointerEvents="none" style={styles.vignette} />

        {/* cello silhouette */}
        <View pointerEvents="none" style={styles.silhouetteWrap}>
          <View style={styles.celloBody} />
          <View style={styles.bridgeGuide} />
          <View style={styles.endpinGuide} />
        </View>

        {/* Countdown Circle */}
        <View style={styles.countdownCircle}>
            <Text style={styles.countdownText}>{countdownVal.toString()}</Text>
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