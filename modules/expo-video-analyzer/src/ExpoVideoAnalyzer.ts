import ExpoVideoAnalyzerModule from './ExpoVideoAnalyzerModule';
import { VideoAnalysisResult } from './ExpoVideoAnalyzer.types';

export function getStatus(): string {
  return ExpoVideoAnalyzerModule.getStatus();
}

export async function analyzeVideo(videoUri: string): Promise<any> {
  return await ExpoVideoAnalyzerModule.analyzeVideo(videoUri);
}

// Placeholder for future function
export async function getAnalysisHistory(): Promise<VideoAnalysisResult[]> {
  // TODO
  return [];
}

// Placeholder for future function
export async function clearAnalysisHistory(): Promise<boolean> {
  // TODO
  return true;
}