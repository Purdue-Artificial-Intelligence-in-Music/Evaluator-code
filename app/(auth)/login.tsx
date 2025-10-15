import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, ActivityIndicator, Alert } from 'react-native';
import { router } from 'expo-router';
import { login } from '../../src/services/auth';

export default function LoginScreen() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPassword, setShowPassword] = useState(false);

  async function onLoginPress() {
    setError(null);
    setIsLoading(true);
    try {
      await login(email, password);     // fake login for now
      Alert.alert('Success', 'Logged in!');
      router.replace('/home');              // return to your app’s existing home route
    } catch (e: any) {
      setError(e?.message || 'Login failed');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Welcome back</Text>
      <Text style={styles.subtitle}>Sign in to continue</Text>

      <TextInput
        value={email}
        onChangeText={setEmail}
        placeholder="Email"
        placeholderTextColor="#9CA3AF"
        autoCapitalize="none"
        keyboardType="email-address"
        style={styles.input}
      />

      <View style={{ width: '100%' }}>
        <TextInput
          value={password}
          onChangeText={setPassword}
          placeholder="Password"
          placeholderTextColor="#9CA3AF"
          secureTextEntry={!showPassword}
          style={styles.input}
        />
        <TouchableOpacity style={styles.toggle} onPress={() => setShowPassword(s => !s)}>
          <Text style={{ color: '#60A5FA' }}>{showPassword ? 'Hide' : 'Show'}</Text>
        </TouchableOpacity>
      </View>

      {error ? <Text style={styles.error}>{error}</Text> : null}

      <TouchableOpacity style={styles.primaryBtn} onPress={onLoginPress} disabled={isLoading}>
        {isLoading ? <ActivityIndicator /> : <Text style={styles.primaryText}>Log In</Text>}
      </TouchableOpacity>

      <TouchableOpacity style={styles.back} onPress={() => router.back()}>
        <Text style={styles.backText}>← Back</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0B0B0F', padding: 24, alignItems: 'center', justifyContent: 'center' },
  title: { color: 'white', fontSize: 28, fontWeight: '700', marginBottom: 6 },
  subtitle: { color: '#9CA3AF', marginBottom: 24 },
  input: {
    width: '100%', backgroundColor: '#111827', color: 'white',
    padding: 14, borderRadius: 12, borderWidth: 1, borderColor: '#374151', marginBottom: 12,
  },
  toggle: { position: 'absolute', right: 12, top: 16 },
  error: { color: '#F87171', marginBottom: 8, alignSelf: 'flex-start' },
  primaryBtn: { backgroundColor: '#3B82F6', paddingVertical: 14, borderRadius: 16, width: '100%', marginTop: 4 },
  primaryText: { color: 'white', textAlign: 'center', fontSize: 16, fontWeight: '600' },
  back: { position: 'absolute', top: 60, left: 20 },
  backText: { color: '#93C5FD', fontSize: 16 },
});