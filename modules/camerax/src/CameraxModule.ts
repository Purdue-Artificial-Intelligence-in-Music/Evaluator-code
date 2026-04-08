import { NativeModule, requireNativeModule } from 'expo';

import { CameraxModuleEvents } from './Camerax.types';

declare class CameraxModule extends NativeModule<CameraxModuleEvents> {
  PI: number;
  hello(): string;
  setValueAsync(value: string): Promise<void>;
  getRecentSessions(userId: string, count: number): Promise<any[]>;
  getSessionImages(userId: string, timestamp: string): Promise<string[]>;
}

// This call loads the native module object from the JSI.
export default requireNativeModule<CameraxModule>('Camerax');
