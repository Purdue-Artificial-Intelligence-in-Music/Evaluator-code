import * as SecureStore from 'expo-secure-store';

const KEY = 'refreshToken';

// Fake login: accepts any email with "@" and password length >= 6
export async function login(email: string, password: string) {
  await wait(600); // simulate network
  if (!email.includes('@')) throw new Error('Please enter a valid email');
  if (password.length < 6) throw new Error('Password must be at least 6 characters');

  const fakeRefreshToken = 'rt_' + Date.now();
  await SecureStore.setItemAsync(KEY, fakeRefreshToken, { keychainService: 'myapp.auth' });
  return { ok: true };
}

export async function hasSession() {
  const rt = await SecureStore.getItemAsync(KEY, { keychainService: 'myapp.auth' });
  return !!rt;
}

export async function clearSession() {
  await SecureStore.deleteItemAsync(KEY, { keychainService: 'myapp.auth' });
}

function wait(ms: number) {
  return new Promise(res => setTimeout(res, ms));
}