import { StyleSheet, Dimensions, Platform } from 'react-native';

const { width: W, height: H } = Dimensions.get('window');
const BODY_W = W - 32;
const BODY_H = Math.min(H * 0.78, (W - 32) * 1.9);
const BODY_TOP = H * 0.08;

const COUNTDOWN_WIDTH = W / 5;

export const ICONS = {
    tools: require('../assets/icons-2.5/tools.png'),
  exit: require('../assets/icons-2.5/exit.png'),
  flip_camera: require('../assets/icons-2.5/flip camera.png'),
  mirror: require('../assets/icons-2.5/mirror the view.png'),
  adjust_threshold: require('../assets/icons-2.5/adjust threshold.png'),
  session_summary: require('../assets/icons-2.5/session summary.png'),
  instructions: require('../assets/icons-2.5/instructions.png'),

  // dots in summary
  tick_square: require('../assets/icons-2.5/Tick_Square.png'),
  wrong: require('../assets/icons-2.5/wrong.png'),
};

export const styles = StyleSheet.create({
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
  settingsButton: {
    position: 'absolute',
    top: 150,
    right: 20,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  settingsButtonText: {
    fontSize: 22,
  },
  // History Button
  historyButton: {
    position: 'absolute',
    top: 200,
    right: 20,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  historyButtonText: {
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
  settingsModalContent: {
    width: '85%',
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 24
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
    marginBottom: 0,  // 从 10 改成 0，因为现在间距由 sectionHeaderRow 控制
    flex: 1,  // 添加这行，让标题占据剩余空间
  },
  subTitle: {
    fontSize: 15,
    fontWeight: "bold",
  },
  timestamp: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center'
  },

  // Settings Modal Styles
  settingsSection: {
    marginBottom: 24
  },
  settingsLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8
  },
  settingsInput: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    backgroundColor: '#f9f9f9'
  },
  settingsHint: {
    fontSize: 12,
    color: '#666',
    marginTop: 6
  },
  settingsFooter: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: 12
  },
  cancelButton: {
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
    backgroundColor: '#f0f0f0'
  },
  saveButton: {
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
    backgroundColor: '#4a4a4a'
  },
  cancelButtonText: {
    color: '#333',
    fontSize: 16,
    fontWeight: '600'
  },
  saveButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600'
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
  countdownCircle: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: [
      { translateX: -COUNTDOWN_WIDTH / 2 },
      { translateY: -COUNTDOWN_WIDTH / 2 }
    ],
    width: COUNTDOWN_WIDTH,
    height: COUNTDOWN_WIDTH,
    borderRadius: COUNTDOWN_WIDTH / 2,
    backgroundColor: 'rgba(255,255,255,1)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  countdownText: {
    color: 'black',
    fontWeight: '700',
    fontSize: COUNTDOWN_WIDTH / 2,
    letterSpacing: 0.2,
    textAlign: 'center',
    textAlignVertical: 'center',
    includeFontPadding: false,
  },
  cardTitle: {
    color: 'white',
    fontWeight: '700',
    fontSize: 20,
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
    fontSize: 16,
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
    fontSize: 15,
    letterSpacing: 0.3,
  },

  // History modal styles
  loadingText: {
    textAlign: 'center',
    fontSize: 16,
    color: '#007AFF',
    marginVertical: 20,
  },
  noHistoryText: {
    textAlign: 'center',
    fontSize: 16,
    color: '#666',
    marginVertical: 20,
  },
  historySubtitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
    color: '#333',
  },
  historyItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    marginBottom: 10,
  },
  historyItemTitle: {
    flex: 1,
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  historyItemDate: {
    fontSize: 14,
    color: '#666',
    marginRight: 8,
  },
  historyItemArrow: {
    fontSize: 20,
    color: '#007AFF',
  },
  backButton: {
    padding: 12,
    marginBottom: 16,
  },
  backButtonText: {
    color: '#007AFF',
    fontSize: 16,
    fontWeight: '600',
  },
  detailTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 16,
    textAlign: 'center',
    color: '#333',
  },

  paginationContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 20,
    marginBottom: 10,
    paddingHorizontal: 20,
  },
  paginationArrowButton: {
    padding: 8,
    minWidth: 40,
    alignItems: 'center',
  },
  paginationArrow: {
    fontSize: 36,
    fontWeight: '300',
    color: '#007AFF',
    lineHeight: 36,
  },
  paginationArrowDisabled: {
    color: '#cccccc',
  },
  paginationText: {
    fontSize: 14,
    color: '#666',
    fontWeight: '500',
  },
  metricRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
  metricDotGood: {
    width: 12,
    height: 12,
    marginRight: 8,
    resizeMode: 'contain',
  },
  metricDotWarning: {
    width: 12,
    height: 12,
    marginRight: 8,
    resizeMode: 'contain',
  },
  metricLabel: {
    flex: 1,
    fontSize: 14,
    color: '#222',
  },
  metricPercent: {
    fontSize: 14,
    fontWeight: '600',
  },
  wrongMetricPercent: {
    fontSize: 14,
    fontWeight: '600',
    color: '#D3D3D3'
  },
  sectionHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 6,
    marginBottom: 8,
  },
  viewMoreButton: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: '#ddd',
    backgroundColor: '#fafafa',
  },
  viewMoreButtonText: {
    fontSize: 12,
    fontWeight: '500',
    color: '#333',
  },
  historyItemSubtitle: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  detailModalContent: {
    width: '90%',
    maxHeight: '80%',
    backgroundColor: 'white',
    borderRadius: 10,
    padding: 20,
  },
  detailImagePlaceholder: {
    height: 140,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#ddd',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
    backgroundColor: '#fafafa',
  },
  detailImageText: {
    fontSize: 14,
    color: '#555',
  },
  imageContainer: {
  marginRight: 10,
  alignItems: 'center',
},
  detailImage: {
    width: 250,
    height: 350,
    borderRadius: 8,
    backgroundColor: '#f0f0f0',
  },
  imageLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 8,
    textAlign: 'center',
    maxWidth: 250,
  },
  imageCountText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 12,
    textAlign: 'center',
  },
  detailImageSubtext: {
    fontSize: 12,
    color: '#999',
    marginTop: 8,
  },
  detailText: {
    fontSize: 14,
    lineHeight: 20,
    color: '#333',
    marginTop: 16,
    marginBottom: 16,
  },

  // ===== Collapsible Toolbar (Top-Left) =====
  topLeftMenuArea: {
    position: 'absolute',
    top: 50,      // matches your closeButton top
    left: 20,
    zIndex: 3000,
    alignItems: 'flex-start',
  },

  menuFab: {
    width: 40,   // match your other round buttons
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },

  menuFabIcon: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },

  menuPanel: {
    marginTop: 10,
    width: 210,
    borderRadius: 12,
    backgroundColor: 'rgba(0,0,0,0.7)',
    overflow: 'hidden',
  },

  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 12,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.12)',
  },

  menuItemIcon: {
    width: 22,
    fontSize: 16,
    color: 'white',
    textAlign: 'center',
    marginRight: 10,
  },

  menuItemText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600',
  },

  // ===== Exit Confirmation Modal =====
  exitModalContent: {
    width: '85%',
    backgroundColor: 'white',
    borderRadius: 12,
    paddingVertical: 22,
    paddingHorizontal: 18,
    alignItems: 'center',
  },
  exitModalTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 18,
    textAlign: 'center',
  },
  exitModalButtons: {
    flexDirection: 'row',
    gap: 14,
  },
  exitModalYesButton: {
    paddingVertical: 10,
    paddingHorizontal: 28,
    borderRadius: 10,
    backgroundColor: '#e6e6e6',
  },
  exitModalNoButton: {
    paddingVertical: 10,
    paddingHorizontal: 28,
    borderRadius: 10,
    backgroundColor: '#4a4a4a',
  },
  exitModalYesText: {
    fontSize: 15,
    fontWeight: '700',
    color: '#333',
  },
  exitModalNoText: {
    fontSize: 15,
    fontWeight: '700',
    color: 'white',
  },
  menuFabIconImg: {
  width: 22,
  height: 22,
  resizeMode: 'contain',
  },

  menuItemIconImg: {
    width: 18,
    height: 18,
    resizeMode: 'contain',
    marginRight: 10,
  },

  closeButtonIconImg: {
    width: 22,
    height: 22,
    resizeMode: 'contain',
  },
});
