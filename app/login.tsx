import React, { useState } from 'react';
import { View, Text, Pressable, TextInput, Modal } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useRouter } from 'expo-router';

export default function Login() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [touched, setTouched] = useState(false);
  const [showModal, setShowModal] = useState(false);

  const isValidEmail = (v: string) =>
    /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v.trim());

  async function onContinue() {
    if (!isValidEmail(email)) return;
    await AsyncStorage.setItem('userEmail', email.trim());
    console.log("received user email: ", email, " and stored");
    setShowModal(true); // 显示弹窗而不是直接跳转
  }

  function onConfirm() {
    setShowModal(false);
    router.replace('/'); // 确认后跳转到主页
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

      {/* 确认弹窗 */}
      <Modal
        visible={showModal}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setShowModal(false)}
      >
        <View style={{
          flex: 1,
          justifyContent: 'center',
          alignItems: 'center',
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
        }}>
          <View style={{
            backgroundColor: '#1f2937',
            borderRadius: 16,
            padding: 24,
            width: '85%',
            maxWidth: 400,
          }}>
            <Text style={{
              color: 'white',
              fontSize: 18,
              fontWeight: '600',
              marginBottom: 16,
              textAlign: 'center',
            }}>
              Important Notice
            </Text>
            
            <Text style={{
              color: '#d1d5db',
              fontSize: 15,
              lineHeight: 22,
              marginBottom: 24,
              textAlign: 'center',
            }}>
              This app is a supplementary tool for checking overall posture. Posture standards may vary by school, so always follow your instructor's advice first.
            </Text>

            <Pressable
              onPress={onConfirm}
              style={({ pressed }) => ({
                backgroundColor: '#3b82f6',
                paddingVertical: 14,
                borderRadius: 12,
                alignItems: 'center',
                opacity: pressed ? 0.9 : 1,
              })}
            >
              <Text style={{ color: 'white', fontSize: 16, fontWeight: '600' }}>
                Confirm
              </Text>
            </Pressable>
          </View>
        </View>
      </Modal>
    </View>
  );
}