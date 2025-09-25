import { requireNativeView } from 'expo';
import * as React from 'react';

import { CameraxViewProps } from './Camerax.types';

const NativeView: React.ComponentType<CameraxViewProps> =
  requireNativeView('Camerax');

export default function CameraxView(props: CameraxViewProps) {
  return <NativeView {...props} />;
}
