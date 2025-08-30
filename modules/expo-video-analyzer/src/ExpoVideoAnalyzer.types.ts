import type { StyleProp, ViewStyle } from 'react-native';

export type OnLoadEventPayload = {
  url: string;
};

export type ChangeEventPayload = {
  value: string;
};

export type ExpoVideoAnalyzerViewProps = {
  url: string;
  onLoad: (event: { nativeEvent: OnLoadEventPayload }) => void;
  style?: StyleProp<ViewStyle>;
};

export type VideoAnalysisResult = {
  status: 'analysis_complete' | 'analysis_failed' | 'processing';
  videoUri: string;
  result: string;
  timestamp: number;
};

export type ExpoVideoAnalyzerModuleEvents = {
  // 为未来的进度事件预留
  onAnalysisProgress: (progress: { percentage: number }) => void;
  onAnalysisComplete: (result: VideoAnalysisResult) => void;
};
