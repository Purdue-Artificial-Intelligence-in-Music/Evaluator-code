// Reexport the native module. On web, it will be resolved to ExpoVideoAnalyzerModule.web.ts
// and on native platforms to ExpoVideoAnalyzerModule.ts
export { default } from './src/ExpoVideoAnalyzerModule';

export * from  './src/ExpoVideoAnalyzer.types';
