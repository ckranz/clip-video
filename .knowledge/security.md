# Web Security Guidelines

Defense-in-depth security patterns for web development.

---

## Core Principles

```
DEFENSE IN DEPTH: Layer multiple protections so if one fails, others protect.
FAIL SECURE: When checks fail, default to denying access.
MINIMIZE INFORMATION LEAKAGE: Error messages should not reveal system internals.
PRINCIPLE OF LEAST PRIVILEGE: Grant only minimum permissions necessary.
```

---

## Authentication & Authorization

### Token Storage

```javascript
// NEVER store tokens in localStorage (accessible to XSS)
localStorage.setItem('authToken', token) // DANGEROUS

// Let Firebase SDK manage token persistence
import { getAuth, getIdTokenResult } from 'firebase/auth'
const auth = getAuth()
// Firebase handles token storage securely
```

### Authorization Decisions

```javascript
// NEVER use client-side state for auth decisions
if (userStore.isAdmin) { /* DANGEROUS */ }

// ALWAYS verify token claims from auth provider
const tokenResult = await getIdTokenResult(auth.currentUser)
if (tokenResult.claims.role === 'admin') { /* SECURE */ }
```

### Session Management

Clear all stores on BOTH login and logout:

```javascript
async authenticate(credentials) {
  // Clear stores BEFORE authenticating (prevents data leakage)
  this.clearAllStores()
  await signInWithEmailAndPassword(auth, credentials.email, credentials.password)
}

clearAllStores() {
  // Reset in-memory state
  userStore.$reset()
  // Also clear localStorage/sessionStorage
  localStorage.removeItem('user')
  sessionStorage.removeItem('user')
}
```

---

## Input Validation

### Form Validation

- Implement validation on both client AND server
- Password minimum: 12+ characters
- Consider HIBP breach checking for passwords
- Use regex for name validation (Unicode-aware)

### HIBP Password Checking

```javascript
async function checkPassword(password) {
  const hash = await sha1(password)
  const prefix = hash.slice(0, 5)
  const suffix = hash.slice(5)

  // K-anonymity: only send first 5 chars
  const response = await fetch(`https://api.pwnedpasswords.com/range/${prefix}`)
  // Parse response to check if suffix matches
}
```

---

## XSS Prevention

### DOMPurify Configuration

```javascript
import DOMPurify from 'dompurify'

export function sanitizeHtml(dirty) {
  return DOMPurify.sanitize(dirty, {
    ALLOWED_TAGS: ['a', 'b', 'i', 'em', 'strong', 'p', 'br', 'ul', 'ol', 'li', 'h1', 'h2', 'h3'],
    ALLOWED_ATTR: ['href', 'title', 'class'],
    FORBID_TAGS: ['script', 'style', 'iframe', 'object', 'embed', 'form'],
    SANITIZE_DOM: true
  })
}
```

### Vue Usage

```vue
<!-- NEVER use unsanitized content -->
<div v-html="userContent"></div>

<!-- ALWAYS sanitize -->
<div v-html="sanitizeHtml(userContent)"></div>
```

---

## Security Headers

```json
{
  "X-Frame-Options": "DENY",
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "Content-Security-Policy": "default-src 'self'; script-src 'self'; frame-ancestors 'none'",
  "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
}
```

---

## API Security

### Hide Function Existence

```javascript
// BAD: Reveals function exists
if (!request.auth) {
  throw new HttpsError('unauthenticated', 'Authentication required')
}

// GOOD: Looks like function doesn't exist
if (!request.auth) {
  throw new HttpsError('not-found', 'Not found')
}
```

### Anti-Enumeration

```javascript
// BAD: Reveals if user exists
if (!userFound) throw new HttpsError('not-found', 'User not found')

// GOOD: Generic error
if (!userFound || !isAdmin) {
  await randomDelay() // Prevent timing attacks
  throw new HttpsError('invalid-argument', 'Operation failed')
}
```

---

## Firestore Rules Pattern

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    function isAuthenticated() { return request.auth != null }
    function isOwner(userId) { return request.auth.uid == userId }

    match /users/{userId} {
      allow read: if isOwner(userId)
      allow create: if false  // Only Cloud Functions
      allow update: if isOwner(userId) &&
        request.resource.data.diff(resource.data).affectedKeys().hasOnly(['name'])
      allow delete: if false
    }

    // Default deny
    match /{document=**} {
      allow read, write: if false
    }
  }
}
```

---

## Build-Time Security

### Dangerous Packages (Block in Client)

- firebase-admin (server-side SDK)
- @google-cloud/firestore
- jsonwebtoken (JWT signing)
- bcrypt (password hashing)
- dotenv (env loading)

### Pinia Store Security

```javascript
// NEVER persist sensitive data
persist: true  // DANGEROUS

// Only persist non-sensitive UI preferences
persist: {
  paths: ['isDarkMode', 'sidebarCollapsed']  // SAFE
}
```

---

## Quick Reference

### Patterns to AVOID

```javascript
localStorage.setItem('token', token)
persist: true  // in Pinia
v-html="unsanitizedContent"
throw new HttpsError('not-found', 'User not found')
import { admin } from 'firebase-admin'  // in client
process.env.API_SECRET  // in client
```

### Patterns to USE

```javascript
const tokenResult = await getIdTokenResult(auth.currentUser)
persist: { paths: ['nonSensitiveData'] }
v-html="sanitizeHtml(content)"
throw new HttpsError('not-found', 'Not found')
import { getFirestore } from 'firebase/firestore'
import.meta.env.VITE_PUBLIC_KEY
```

---

## Security Testing Checklist

- [ ] All `v-html` uses have sanitization
- [ ] No sensitive data in localStorage/Pinia persistence
- [ ] All API endpoints validate authentication
- [ ] Firestore rules deny by default
- [ ] No hardcoded secrets or API keys
- [ ] CSP headers properly configured
- [ ] Error messages don't leak system info
- [ ] Input validation on all user inputs
