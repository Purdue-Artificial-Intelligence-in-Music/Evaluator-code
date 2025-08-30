import { NativeModule, requireNativeModule } from 'expo';
import { ExpoVideoAnalyzerModuleEvents, VideoAnalysisResult } from './ExpoVideoAnalyzer.types';

declare class ExpoVideoAnalyzerModule extends NativeModule<ExpoVideoAnalyzerModuleEvents> {
  getStatus: () => string;
  analyzeVideo: (videoUri: string) => Promise<VideoAnalysisResult>;
}

// This call loads the native module object from the JSI
export default requireNativeModule<ExpoVideoAnalyzerModule>('ExpoVideoAnalyzer');
