// Reexport the native module. On web, it will be resolved to CameraxModule.web.ts
// and on native platforms to CameraxModule.ts
export { default } from './src/CameraxModule';
export { default as CameraxView } from './src/CameraxView';
export * from  './src/Camerax.types';
