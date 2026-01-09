# Testing Guidelines

Practical guidance for writing effective tests. Focus on what to test, not how to use frameworks.

---

## Unit Test Patterns

### What to Unit Test

Unit tests are for **isolated logic** with clear inputs and outputs:

- Pure functions (same input always gives same output)
- Calculations and algorithms
- Data transformations and mappings
- Validation logic
- State transitions

### Good Unit Test Characteristics

```
ISOLATED: No database, no API calls, no file system
FAST: Milliseconds, not seconds
DETERMINISTIC: Same result every time
FOCUSED: One logical assertion per test
```

### Example Patterns

**Testing calculations:**
```javascript
// Testing a confidence score calculation
describe('calculateInterestScore', () => {
  it('increases score based on view duration', () => {
    const result = calculateInterestScore({ viewDurationMs: 60000 })
    expect(result.score).toBeGreaterThan(0)
  })

  it('caps score at maximum', () => {
    const result = calculateInterestScore({ viewDurationMs: 999999999 })
    expect(result.score).toBeLessThanOrEqual(100)
  })

  it('returns zero for no interaction', () => {
    const result = calculateInterestScore({ viewDurationMs: 0 })
    expect(result.score).toBe(0)
  })
})
```

**Testing transformations:**
```javascript
// Testing data mapping
describe('mapUserPreferences', () => {
  it('converts sensitivity slider to weight', () => {
    expect(mapUserPreferences({ sensitivity: 0 })).toEqual({ weight: 0.1 })
    expect(mapUserPreferences({ sensitivity: 50 })).toEqual({ weight: 0.5 })
    expect(mapUserPreferences({ sensitivity: 100 })).toEqual({ weight: 1.0 })
  })
})
```

---

## Integration Test Patterns

### What to Integration Test

Integration tests verify **components working together**:

- Database operations (CRUD)
- API endpoint behavior
- Service-to-service communication
- Event flows and webhooks
- Authentication flows

### Good Integration Test Characteristics

```
REALISTIC: Uses real (or realistic mock) dependencies
ISOLATED: Own test database/environment
CLEANED UP: Resets state between tests
SLOWER: Seconds acceptable, minutes not
```

### Example Patterns

**Testing API endpoints:**
```javascript
describe('POST /api/interests', () => {
  beforeEach(async () => {
    await testDb.clear('interests')
  })

  it('creates interest record for authenticated user', async () => {
    const response = await request(app)
      .post('/api/interests')
      .set('Authorization', `Bearer ${testToken}`)
      .send({ productId: 'prod_123', duration: 5000 })

    expect(response.status).toBe(201)
    expect(response.body.id).toBeDefined()

    // Verify in database
    const record = await testDb.findOne('interests', response.body.id)
    expect(record.productId).toBe('prod_123')
  })

  it('rejects unauthenticated requests', async () => {
    const response = await request(app)
      .post('/api/interests')
      .send({ productId: 'prod_123' })

    expect(response.status).toBe(401)
  })
})
```

**Testing database operations:**
```javascript
describe('InterestRepository', () => {
  it('decays old interests correctly', async () => {
    // Setup: create interest from 3 weeks ago
    await repo.create({
      userId: 'user_1',
      productId: 'prod_1',
      score: 100,
      createdAt: threeWeeksAgo()
    })

    // Act: run decay
    await repo.applyDecay({ decayStartDays: 14 })

    // Assert: score reduced
    const updated = await repo.findByUser('user_1')
    expect(updated[0].score).toBeLessThan(100)
  })
})
```

---

## Testability Signals

### Signs Something is Easily Unit-Testable

Look for these in requirements:

| Signal | Example | Test Type |
|--------|---------|-----------|
| "Calculate" | "Calculate interest score" | Unit |
| "If X then Y" | "If viewed > 30s, increase score" | Unit |
| "Convert/transform" | "Convert slider to weight" | Unit |
| "Validate" | "Validate email format" | Unit |
| "Must be/return" | "Must return sorted list" | Unit |
| Numbers/thresholds | "Decay after 14 days" | Unit |

### Signs Something Needs Integration Testing

| Signal | Example | Test Type |
|--------|---------|-----------|
| "User sees" | "User sees notification" | Integration/E2E |
| "Saves/persists" | "Saves interest to database" | Integration |
| "Sends/receives" | "Sends notification email" | Integration |
| "Across sessions" | "Preferences persist across sessions" | Integration |
| "Real-time" | "Updates in real-time" | Integration |
| External service | "Fetches from price API" | Integration |

### Signs Something is Hard to Test

| Signal | Approach |
|--------|----------|
| Subjective ("smart", "relevant") | Define measurable proxy |
| Time-dependent | Use injectable clock |
| Random/probabilistic | Test distribution, not exact values |
| UI-dependent | Separate logic from presentation |
| Third-party black box | Mock at boundary |

---

## Assertion Patterns

### Strong Assertions

```javascript
// GOOD: Specific, verifiable
expect(score).toBe(75)
expect(results).toHaveLength(3)
expect(user.role).toBe('admin')
expect(error.code).toBe('INVALID_INPUT')
```

### Weak Assertions (Avoid)

```javascript
// BAD: Too vague
expect(score).toBeTruthy()
expect(results.length).toBeGreaterThan(0)
expect(user).toBeDefined()
expect(error).not.toBeNull()
```

### Testing Boundaries

Always test:
- Zero/empty inputs
- Maximum values
- Just below and just above thresholds
- Invalid types (null, undefined, wrong type)

```javascript
describe('decay threshold boundary', () => {
  it('does not decay at exactly 14 days', () => {
    const interest = createInterest({ daysOld: 14 })
    expect(shouldDecay(interest)).toBe(false)
  })

  it('decays at 15 days', () => {
    const interest = createInterest({ daysOld: 15 })
    expect(shouldDecay(interest)).toBe(true)
  })
})
```

---

## Test Data Guidelines

### Factories Over Fixtures

```javascript
// GOOD: Factory with defaults
const createUser = (overrides = {}) => ({
  id: `user_${Date.now()}`,
  email: 'test@example.com',
  role: 'user',
  ...overrides
})

// Usage
const admin = createUser({ role: 'admin' })
const specificUser = createUser({ id: 'user_123', email: 'specific@test.com' })
```

### Time Handling

```javascript
// BAD: Hardcoded dates break over time
const testDate = new Date('2024-01-15')

// GOOD: Relative dates
const testDate = new Date(Date.now() - 14 * 24 * 60 * 60 * 1000) // 14 days ago

// BETTER: Injectable clock
const createInterestService = (clock = Date) => ({
  isExpired: (interest) => {
    const age = clock.now() - interest.createdAt
    return age > EXPIRY_MS
  }
})

// In tests
const mockClock = { now: () => new Date('2024-06-15').getTime() }
const service = createInterestService(mockClock)
```

### Seasonal/Calendar Edge Cases

Test these explicitly:
- Year boundaries (Dec 31 → Jan 1)
- Month boundaries (Feb 28/29)
- Timezone transitions (DST)
- Holiday periods (if business logic depends on them)

```javascript
describe('seasonal decay', () => {
  it('holds November interests until after Christmas', () => {
    const novemberInterest = createInterest({
      createdAt: new Date('2024-11-15'),
      evaluatedAt: new Date('2024-12-20')
    })
    expect(shouldDecay(novemberInterest)).toBe(false)
  })

  it('allows decay after December 26', () => {
    const novemberInterest = createInterest({
      createdAt: new Date('2024-11-15'),
      evaluatedAt: new Date('2024-12-27')
    })
    expect(shouldDecay(novemberInterest)).toBe(true)
  })
})
```

---

## Quick Reference

### When to Write Tests

| Situation | Action |
|-----------|--------|
| New calculation/algorithm | Unit test immediately |
| New API endpoint | Integration test immediately |
| Bug fix | Write failing test first, then fix |
| Refactoring | Ensure tests exist before refactoring |
| "It works on my machine" | Add integration test |

### Test Naming Convention

```
describe('[Unit/Component being tested]', () => {
  it('[action] when [condition]', () => {
    // Given: setup
    // When: action
    // Then: assertion
  })
})
```

### Minimum Coverage Targets

| Code Type | Target |
|-----------|--------|
| Business logic | 90%+ |
| Utilities | 80%+ |
| API handlers | 70%+ |
| UI components | 60%+ (logic only) |
