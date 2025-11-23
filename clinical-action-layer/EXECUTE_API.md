# Execute API Documentation

## Overview
The `/api/execute` endpoint creates FHIR resources in Medplum by taking draft clinical actions (from `/api/analyze`) and executing them with a patient reference.

## Endpoint Details
- **URL:** `/api/execute`
- **Method:** `POST`
- **Content-Type:** `application/json`

## Request Body

```typescript
{
  action: {
    type: 'medication' | 'lab' | 'imaging' | 'referral' | 'followup';
    description: string;
    resource: MedicationRequest | ServiceRequest;
  };
  patientId: string; // Medplum Patient resource ID
}
```

### Fields

#### `action` (required)
The clinical action object returned from `/api/analyze`. Contains:
- `type`: The type of clinical action
- `description`: Human-readable description
- `resource`: The draft FHIR resource (MedicationRequest or ServiceRequest)

#### `patientId` (required)
The Medplum Patient resource ID (UUID format) to link the order to.

## Response

### Success Response (201 Created)

```json
{
  "success": true,
  "resourceId": "123e4567-e89b-12d3-a456-426614174000",
  "resourceType": "MedicationRequest",
  "message": "Successfully created MedicationRequest for patient abc-123"
}
```

### Error Responses

#### 400 Bad Request
```json
{
  "error": "Missing required field: patientId"
}
```

#### 404 Not Found
```json
{
  "error": "Patient not found: abc-123",
  "details": "The specified patient ID does not exist in Medplum"
}
```

#### 503 Service Unavailable
```json
{
  "error": "Failed to authenticate with Medplum"
}
```

#### 500 Internal Server Error
```json
{
  "error": "Failed to create resource in Medplum",
  "details": "..."
}
```

## What the Endpoint Does

1. **Validates Input**: Checks that both `action` and `patientId` are provided
2. **Authenticates**: Gets authenticated Medplum client using OAuth2
3. **Attaches Patient Reference**: Adds `subject: { reference: "Patient/{patientId}" }` to the FHIR resource
4. **Sets Resource Status**: Ensures `status: "draft"` and `intent: "order"`
5. **Creates in Medplum**: Calls `medplum.createResource()` to persist the resource
6. **Returns Result**: Returns the created resource ID and type

## Example Usage

### Example 1: Execute Medication Order

```bash
curl -X POST http://localhost:3000/api/execute \
  -H "Content-Type: application/json" \
  -d '{
    "patientId": "abc-123-def-456",
    "action": {
      "type": "medication",
      "description": "Prescribe Amoxicillin 500mg three times daily for 7 days",
      "resource": {
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
          },
          "doseAndRate": [{
            "doseQuantity": {
              "value": 500,
              "unit": "mg"
            }
          }]
        }]
      }
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "resourceId": "med-123-456-789",
  "resourceType": "MedicationRequest",
  "message": "Successfully created MedicationRequest for patient abc-123-def-456"
}
```

### Example 2: Execute Lab Test Order

```bash
curl -X POST http://localhost:3000/api/execute \
  -H "Content-Type: application/json" \
  -d '{
    "patientId": "abc-123-def-456",
    "action": {
      "type": "lab",
      "description": "Order Complete Blood Count (CBC)",
      "resource": {
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
        },
        "priority": "routine"
      }
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "resourceId": "srv-789-012-345",
  "resourceType": "ServiceRequest",
  "message": "Successfully created ServiceRequest for patient abc-123-def-456"
}
```

## Integration with Frontend

### Workflow
1. User views transcript in UI
2. Frontend calls `/api/analyze` to extract actions
3. UI displays action cards for review
4. User clicks "Approve" on an action
5. Frontend calls `/api/execute` with the action and selected patient ID
6. FHIR resource is created in Medplum
7. UI shows confirmation with resource ID

### Frontend Example (React)

```typescript
async function approveAction(action: ClinicalAction, patientId: string) {
  try {
    const response = await fetch('/api/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, patientId })
    });

    const result = await response.json();

    if (response.ok) {
      console.log(`Created ${result.resourceType} with ID: ${result.resourceId}`);
      // Update UI to show success
    } else {
      console.error('Execute failed:', result.error);
      // Show error to user
    }
  } catch (error) {
    console.error('Request failed:', error);
  }
}
```

## Testing

### Interactive Test UI
Open `http://localhost:3000/test-execute.html` in your browser for an interactive testing interface with:
- Pre-loaded example actions (medication, lab, imaging)
- Patient ID selector (fetches real patients from Medplum)
- JSON editor for custom actions
- Real-time execution and response display

### Manual Testing with curl

1. Get a real patient ID:
```bash
curl http://localhost:3000/api/medplum/patients
```

2. Execute an action:
```bash
curl -X POST http://localhost:3000/api/execute \
  -H "Content-Type: application/json" \
  -d '{ "patientId": "...", "action": { ... } }'
```

## FHIR Resource Details

### Supported Resource Types
- **MedicationRequest**: For medication orders/prescriptions
- **ServiceRequest**: For labs, imaging, referrals, follow-ups

### Resource Modifications
The endpoint automatically:
- Adds `subject.reference` pointing to the patient
- Sets `status` to `"draft"` (requires approval workflow)
- Sets `intent` to `"order"` (standard for active orders)

### Patient Reference Format
```json
{
  "subject": {
    "reference": "Patient/abc-123-def-456"
  }
}
```

## Error Handling

The endpoint provides detailed error messages for common issues:
- **Missing fields**: Clear validation errors
- **Invalid patient ID**: 404 with helpful message
- **Medplum authentication failure**: 503 with service unavailable
- **Resource creation failure**: 500 with Medplum error details

## Security Considerations

- Endpoint requires valid Medplum authentication (OAuth2 client credentials)
- Patient ID validation prevents creating orders for non-existent patients
- Draft status ensures resources require explicit approval before execution
- All operations are logged for audit trails

## Architecture Notes

This endpoint is part of the "Clinical Action Layer" architecture:
- **Analyze** (`/api/analyze`): AI extracts clinical intents → draft FHIR resources
- **Review** (Frontend UI): Doctor reviews and approves actions
- **Execute** (`/api/execute`): Creates approved resources in Medplum
- **EMR Integration** (Medplum): FHIR resources stored in production system

## Next Steps

After executing resources:
1. Resources appear in Medplum console with `status: "draft"`
2. Doctor can review in EMR
3. Doctor can update status to `"active"` to finalize
4. Resources integrate with broader care workflows (e-prescribing, lab orders, etc.)

## Related Endpoints

- **`GET /api/medplum/patients`**: Fetch available patients
- **`POST /api/analyze`**: Extract clinical actions from transcripts
- **`GET /api/heidi/transcript/{sessionId}`**: Fetch consultation transcripts

## Logs

The endpoint logs key operations:
```
[Execute] Processing MedicationRequest for patient abc-123
[Execute] Action type: medication, Description: Prescribe Amoxicillin...
[Execute] Creating resource in Medplum: {...}
[Execute] ✅ Successfully created MedicationRequest with ID: med-123
```

Monitor server logs for troubleshooting and audit trails.
