import React, { useState } from 'react';
import { View, Text, Pressable, TextInput } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useRouter } from 'expo-router';

export default function Login() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [touched, setTouched] = useState(false);

  const isValidEmail = (v: string) =>
    /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v.trim());

  async function onContinue() {
    if (!isValidEmail(email)) return;
    await AsyncStorage.setItem('userEmail', email.trim());
    router.replace('/'); // go to index.tsx
  }

  const showError = touched && email.length > 0 && !isValidEmail(email);

  return (
    <View style={{ flex: 1, justifyContent: 'center', padding: 24, backgroundColor: '#0b1020' }}>
      <Text style={{ color: 'white', fontSize: 28, fontWeight: '700', marginBottom: 16 }}>
        Welcome
      </Text>
      <Text style={{ color: '#98a2b3', fontSize: 16, marginBottom: 16 }}>
        Enter your email to continue. No password needed.
      </Text>

      <TextInput
        value={email}
        onChangeText={setEmail}
        onBlur={() => setTouched(true)}
        placeholder="you@example.com"
        placeholderTextColor="#94a3b8"
        autoCapitalize="none"
        autoComplete="email"
        keyboardType="email-address"
        textContentType="emailAddress"
        style={{
          backgroundColor: '#111827',
          color: 'white',
          borderColor: showError ? '#ef4444' : '#1f2937',
          borderWidth: 1,
          borderRadius: 12,
          paddingHorizontal: 14,
          paddingVertical: 12,
          marginBottom: showError ? 6 : 16,
        }}
      />

      {showError && (
        <Text style={{ color: '#ef4444', marginBottom: 10 }}>
          Please enter a valid email.
        </Text>
      )}

      <Pressable
        onPress={onContinue}
        disabled={!isValidEmail(email)}
        style={({ pressed }) => ({
          opacity: !isValidEmail(email) ? 0.5 : pressed ? 0.9 : 1,
          backgroundColor: '#3b82f6',
          paddingVertical: 14,
          borderRadius: 12,
          alignItems: 'center',
        })}
      >
        <Text style={{ color: 'white', fontSize: 16, fontWeight: '600' }}>Continue</Text>
      </Pressable>
    </View>
  );
}