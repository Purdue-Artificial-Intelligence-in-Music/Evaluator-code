// Reexport the native module. On web, it will be resolved to CameraxModule.web.ts
// and on native platforms to CameraxModule.ts

// export { default } from './src/CameraxModule';
// export { default as CameraxView } from './src/CameraxView';
// export * from  './src/Camerax.types';

import { NativeModulesProxy, requireNativeViewManager } from 'expo-modules-core';

export const Camerax = NativeModulesProxy.Camerax;
export const CameraxView = requireNativeViewManager('Camerax');

export * from './src/Camerax.types';