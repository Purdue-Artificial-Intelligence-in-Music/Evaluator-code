// components/LogoutButton.tsx
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useRouter } from 'expo-router';
import { Pressable, Text } from 'react-native';
import React from 'react';

export default function LogoutButton() {
  const router = useRouter();

  async function onLogout() {
    await AsyncStorage.removeItem('userEmail'); // ⬅️ clear the email
    router.replace('/login');                   // ⬅️ go to login
  }

  return (
    <Pressable onPress={onLogout} style={{ padding: 12 }}>
      <Text style={{ color: '#ef4444', fontWeight: '600' }}>Sign out</Text>
    </Pressable>
  );
}