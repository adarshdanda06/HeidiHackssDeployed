# Claude Analysis API - Implementation Guide

## Overview
The `/api/analyze` endpoint uses Claude 3.5 Sonnet to extract structured clinical actions from consultation transcripts and convert them into FHIR R4-compliant resources.

## Endpoint Details

### `POST /api/analyze`

**Request Body:**
```json
{
  "transcript": "string (required)",
  "patientContext": "string (optional)"
}
```

**Response:**
```json
{
  "actions": [
    {
      "type": "medication" | "lab" | "imaging" | "referral" | "followup",
      "description": "Human-readable description for UI",
      "resource": {
        // FHIR R4 Resource (MedicationRequest or ServiceRequest)
      }
    }
  ]
}
```

## Action Types

| Type | FHIR Resource | Description |
|------|--------------|-------------|
| `medication` | `MedicationRequest` | Prescriptions and medication orders |
| `lab` | `ServiceRequest` | Laboratory test orders |
| `imaging` | `ServiceRequest` | Imaging study orders (X-ray, CT, MRI) |
| `referral` | `ServiceRequest` | Specialist referral requests |
| `followup` | `ServiceRequest` | Follow-up appointment scheduling |

## FHIR Resource Structure

### MedicationRequest (Medications)
```json
{
  "resourceType": "MedicationRequest",
  "status": "draft",
  "intent": "order",
  "medicationCodeableConcept": {
    "coding": [{
      "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
      "code": "308182",
      "display": "Amoxicillin 500 MG"
    }],
    "text": "Amoxicillin 500mg"
  },
  "dosageInstruction": [{
    "text": "500mg three times daily for 7 days",
    "timing": {
      "repeat": {
        "frequency": 3,
        "period": 1,
        "periodUnit": "d"
      }
    }
  }]
}
```

### ServiceRequest (Labs, Imaging, Referrals)
```json
{
  "resourceType": "ServiceRequest",
  "status": "draft",
  "intent": "order",
  "category": [{
    "coding": [{
      "system": "http://snomed.info/sct",
      "code": "108252007",
      "display": "Laboratory procedure"
    }]
  }],
  "code": {
    "coding": [{
      "system": "http://loinc.org",
      "code": "58410-2",
      "display": "Complete blood count (hemogram) panel"
    }],
    "text": "CBC"
  }
}
```

## Testing

### Interactive Test UI
Open in browser: `http://localhost:3000/analyze-tester.html`

The test page includes:
- 3 pre-loaded example transcripts
- Real-time Claude API testing
- Visual action card display
- Raw JSON response viewer

### Manual Testing with cURL
```bash
curl -X POST http://localhost:3000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Patient presents with cough and fever. Prescribe Amoxicillin 500mg TID for 7 days. Order chest X-ray and CBC."
  }'
```

## Implementation Details

### Claude Configuration
- **Model:** `claude-3-5-sonnet-20241022`
- **Max Tokens:** 4096
- **Temperature:** 0.3 (low for consistent structured output)
- **System Prompt:** Specialized clinical FHIR extraction prompt

### Error Handling
- Validates transcript presence and type
- Checks for ANTHROPIC_API_KEY configuration
- Handles Claude API errors with detailed messages
- Robust JSON parsing (strips markdown, extracts JSON from text)
- Validates action structure before returning

### JSON Parsing Strategy
1. Try direct `JSON.parse()` first
2. If fails, remove markdown code blocks (```json```)
3. Extract JSON object using regex pattern matching
4. Validate structure and filter invalid actions

## Environment Variables

Required in `.env.local`:
```bash
ANTHROPIC_API_KEY=sk-ant-api03-xxx...
```

## Integration with Frontend

### Example Usage in React Component
```typescript
async function analyzeTranscript(transcript: string) {
  const response = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ transcript })
  });
  
  const data = await response.json();
  return data.actions; // Array of clinical actions
}
```

### Displaying Actions
```typescript
{data.actions.map((action, idx) => (
  <ActionCard
    key={idx}
    type={action.type}
    description={action.description}
    resource={action.resource}
  />
))}
```

## Next Steps

1. **Execute Endpoint:** Create `/api/execute` to write approved FHIR resources to Medplum
2. **Frontend Integration:** Wire TranscriptInputPanel → analyze → ActionCard display
3. **Validation:** Add clinical validation rules (drug interactions, contraindications)
4. **Patient Context:** Enhance with patient demographics and medical history

## Files Created

- `/app/api/analyze/route.ts` - Main API endpoint
- `/public/analyze-tester.html` - Interactive testing UI
- `CLAUDE_ANALYSIS_API.md` - This documentation

## Coding Standards Used

- ✅ TypeScript with full type safety
- ✅ FHIR R4 resource types from `@medplum/fhirtypes`
- ✅ Proper error handling and logging
- ✅ Status "draft" for all resources (require approval)
- ✅ Standard medical coding systems (RxNorm, LOINC, SNOMED CT)
- ✅ Comprehensive console logging for debugging
