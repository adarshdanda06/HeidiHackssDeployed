# Heidi Health API Integration Guide

## ✅ Setup Status: READY

All Heidi API configuration is complete!

## 🔑 Configuration

Your API key is configured in `.env.local`:
```env
HEIDI_API_KEY=HIztzs28cXhQ3m4rMKYylG77i0bC283U
HEIDI_API_BASE_URL=https://api.heidi.health (optional, defaults to this)
```

## 📡 Available Endpoints

### 1. Test Connection
**GET** `/api/heidi/test`

Tests if the Heidi API key is configured and working.

**Response:**
```json
{
  "success": true,
  "message": "Heidi API connection successful",
  "apiConfigured": true,
  "baseUrl": "https://api.heidi.health",
  "apiKeyPreview": "HIztzs28..."
}
```

### 2. Fetch Transcript
**POST** `/api/heidi/transcript`

Fetches a consultation transcript by visit ID.

**Request Body:**
```json
{
  "visitId": "visit-123"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "visitId": "visit-123",
    "transcript": "Patient presents with...",
    "patientName": "John Doe",
    "visitDate": "2024-11-22T10:30:00Z",
    "duration": 900,
    "metadata": {}
  },
  "fetchedAt": "2024-11-22T11:00:00Z"
}
```

## 🔧 Usage in Code

```typescript
// Fetch a transcript
const response = await fetch('/api/heidi/transcript', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ visitId: 'visit-123' }),
});

const { data } = await response.json();
console.log(data.transcript);
```

## 🎯 Integration Points

### Current Workflow:
1. **User enters Visit ID** in the UI
2. **Frontend calls** `/api/heidi/transcript`
3. **Backend fetches** transcript from Heidi
4. **Returns transcript** to frontend
5. **Frontend sends transcript** to `/api/analyze` (Claude)
6. **Claude extracts** clinical actions
7. **UI displays** action cards for review

### File Structure:
```
app/api/heidi/
├── test/route.ts           # Connection test endpoint
└── transcript/route.ts     # Fetch transcript endpoint

lib/types.ts                # HeidiTranscript types
```

## 🧪 Testing

### 1. Test API Configuration
```bash
curl http://localhost:3000/api/heidi/test
```

### 2. Test Transcript Fetch (with real visit ID)
```bash
curl -X POST http://localhost:3000/api/heidi/transcript \
  -H "Content-Type: application/json" \
  -d '{"visitId": "your-visit-id"}'
```

### 3. Test from Browser
Open: http://localhost:3000/api/heidi/test

## ⚠️ API Endpoint Discovery

Since we don't have Heidi's official documentation, the endpoint URLs are educated guesses based on common REST API patterns:

**Current assumptions:**
- Base URL: `https://api.heidi.health`
- Transcript endpoint: `/v1/visits/{visitId}/transcript`
- Auth: Bearer token via `Authorization` header

**If these don't work, try:**
- `/api/visits/{id}/transcript`
- `/transcripts/{id}`
- `/v1/transcripts/{id}`
- Different HTTP methods (POST instead of GET)

## 🔄 Next Steps

1. ✅ Heidi API client setup (DONE)
2. ✅ Test endpoint created (DONE)
3. ✅ Transcript fetching endpoint (DONE)
4. ⏳ Test with real visit ID
5. ⏳ Adjust endpoint URLs if needed
6. ⏳ Integrate with Claude analysis pipeline

## 🤝 Working with Your Team

- **Your friend** is handling Medplum (FHIR resource creation)
- **You** are handling Heidi API (transcript ingestion)
- **Next**: Someone needs to handle Claude analysis (`/api/analyze`)

## 📝 Error Handling

The API handles these error cases:
- Missing API key
- Invalid visit ID
- Network errors
- Heidi API failures (4xx, 5xx responses)
- Timeout (5 second limit on health checks)

All errors return structured JSON with helpful messages.

## 🎨 Frontend Integration

Example React component usage:
```typescript
const fetchTranscript = async (visitId: string) => {
  try {
    const response = await fetch('/api/heidi/transcript', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ visitId }),
    });
    
    if (!response.ok) throw new Error('Failed to fetch');
    
    const { data } = await response.json();
    return data.transcript;
  } catch (error) {
    console.error('Error fetching transcript:', error);
    throw error;
  }
};
```
