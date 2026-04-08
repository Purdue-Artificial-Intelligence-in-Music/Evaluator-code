// app/_layout.tsx
import React, { useEffect, useState } from 'react';
import { Stack, useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Sentry from '@sentry/react-native';

Sentry.init({
  dsn: 'https://180db5f2b94a7251bb35d5a42fe82aa0@o4511027696959488.ingest.us.sentry.io/4511146481942528',

  // Adds more context data to events (IP address, cookies, user, etc.)
  // For more information, visit: https://docs.sentry.io/platforms/react-native/data-management/data-collected/
  sendDefaultPii: true,

  // Enable Logs
  enableLogs: true,

  // Configure Session Replay
  replaysSessionSampleRate: 0.1,
  replaysOnErrorSampleRate: 1,
  integrations: [Sentry.mobileReplayIntegration(), Sentry.feedbackIntegration()],

  // uncomment the line below to enable Spotlight (https://spotlightjs.com)
  // spotlight: __DEV__,
});

export default Sentry.wrap(function RootLayout() {
  const router = useRouter();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // mark as mounted
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return; // wait for mounting

    // check if user has logged in
    const checkAuth = async () => {
      Sentry.addBreadcrumb({
        category: 'auth',
        message: 'Checking stored user authentication',
        level: 'info',
      });

      try {
        const email = await AsyncStorage.getItem('userEmail');
        console.log('Stored email:', email);

        if (email) {
          Sentry.setUser({ id: email, email });
        } else {
          Sentry.setUser(null);
          router.replace('/login');
        }

      } catch (error) {
        Sentry.captureException(error, {
          tags: { area: 'auth', action: 'check_auth_layout' },
        });
        console.error('Error checking auth:', error);
      }
    };

    checkAuth();
  }, [mounted]);

  return (
    <Stack>
      <Stack.Screen 
        name="index" 
        options={{ title: 'Evaluator' }} 
      />
      <Stack.Screen 
        name="login" 
        options={{ title: 'Login' }} 
      />
    </Stack>
  );
});