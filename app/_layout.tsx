// app/_layout.tsx
import React, { useEffect, useState } from 'react';
import { Stack, useRouter } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Sentry from '@sentry/react-native';

Sentry.init({
  dsn: 'https://c6215d1cf1960b60f35cf6870e78eb0e@o4511027696959488.ingest.us.sentry.io/4511027701481472',
  debug: true,
});

function RootLayoutInner() {
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
      try {
        const email = await AsyncStorage.getItem('userEmail');
        console.log('Stored email:', email);
        
        if (!email) {
          console.log('No email found, redirecting to login');
          router.replace('/login');
        }
      } catch (error) {
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
}

export default Sentry.wrap(RootLayoutInner);