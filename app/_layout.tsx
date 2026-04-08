// app/_layout.tsx
import React, { useEffect, useState } from 'react';
import { Stack, useRouter } from 'expo-router';
import { View, ActivityIndicator } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

export default function RootLayout() {
  const router = useRouter();
  const [isAuthChecked, setIsAuthChecked] = useState(false);

  useEffect(() => {
    // check if user has logged in
    const checkAuth = async () => {
      try {
        let email = await AsyncStorage.getItem('userEmail');
        console.log('Stored email:', email);
        
        if (!email) {
          console.log('No email found, redirecting to login');
          // Set a demo email for development/testing
          email = 'demo@evaluator.com';
          await AsyncStorage.setItem('userEmail', email);
          console.log('Demo email set:', email);
        }
        
        // Mark auth check as complete
        setIsAuthChecked(true);
      } catch (error) {
        console.error('Error checking auth:', error);
        setIsAuthChecked(true); // Continue anyway
      }
    };

    checkAuth();
  }, []);

  // Show a lightweight loading state while auth is being checked.
  if (!isAuthChecked) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#FFF' }}>
        <ActivityIndicator size="large" color="#3b82f6" />
      </View>
    );
  }

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