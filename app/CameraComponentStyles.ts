import { StyleSheet, Dimensions, Platform } from 'react-native';

const { width: W, height: H } = Dimensions.get('window');
const BODY_W = W - 32;
const BODY_H = Math.min(H * 0.78, (W - 32) * 1.9);
const BODY_TOP = H * 0.08;

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
    marginBottom: 10
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
});