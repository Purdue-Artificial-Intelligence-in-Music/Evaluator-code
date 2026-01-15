import type { StyleProp, ViewStyle } from 'react-native';

export type OnLoadEventPayload = {
  url: string;
};

export type CameraxModuleEvents = {
  onChange: (params: ChangeEventPayload) => void;
};

export type ChangeEventPayload = {
  value: string;
};

export type SessionSummaryPayload = {
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
};

export type CameraxViewProps = {
  userId?: string;
  cameraActive?: boolean;
  detectionEnabled?: boolean;
  lensType?: string;
  maxBowAngle?: number;
  // Force delegate for testing: "npu" | "gpu" | "cpu" | null (auto)
  forcedDelegate?: string | null;
  onDetectionResult?: (event: { nativeEvent: any }) => void;
  onNoDetection?: (event: { nativeEvent: any }) => void;
  onSessionEnd?: (event: { nativeEvent: SessionSummaryPayload }) => void;
  style?: StyleProp<ViewStyle>;
};
