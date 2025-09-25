import ExpoVideoAnalyzerModule from './ExpoVideoAnalyzerModule';
import { VideoAnalysisResult } from './ExpoVideoAnalyzer.types';

export function getStatus(): string {
  return ExpoVideoAnalyzerModule.getStatus();
}

export async function openVideo(videoUri: string): Promise<any> {
  return await ExpoVideoAnalyzerModule.openVideo(videoUri);
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

export async function initialize(): Promise<any> {
  return await ExpoVideoAnalyzerModule.initialize();
}

export async function processFrame(uri: string): Promise<any> {
  return await ExpoVideoAnalyzerModule.processFrame(uri);
}

export async function processVideoComplete(videoUri: String): Promise<any> {
  return await ExpoVideoAnalyzerModule.processVideoComplete(videoUri);
}

export async function testFFmpeg(): Promise<any> {
  return await ExpoVideoAnalyzerModule.testFFmpeg();
}

export async function checkFileExists(filePath: String): Promise<any> {
  return await ExpoVideoAnalyzerModule.checkFileExists(filePath);
}

export async function cancelProcessing() {
  return await ExpoVideoAnalyzerModule.cancelProcessing();
}
