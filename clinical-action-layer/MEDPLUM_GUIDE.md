# Medplum SDK - How It Works

## ✅ Setup Status: READY TO USE

All Medplum configuration is complete! Your credentials are working.

## 🔑 Authentication Flow

```typescript
// 1. Create client instance
const medplum = new MedplumClient({
  baseUrl: 'https://api.medplum.com/',
  clientId: 'your-client-id',
  clientSecret: 'your-client-secret'
});

// 2. Authenticate (OAuth2 Client Credentials)
await medplum.startClientLogin(clientId, clientSecret);

// 3. Now you can make FHIR requests!
```

## 📋 Common Operations

### **Read a Resource**
```typescript
const patient = await medplum.readResource('Patient', patientId);
```

### **Search for Resources**
```typescript
const patients = await medplum.searchResources('Patient', {
  name: 'John',
  _count: 10
});
```

### **Create a Resource (What we'll use most)**
```typescript
const medicationRequest = await medplum.createResource({
  resourceType: 'MedicationRequest',
  status: 'draft',
  intent: 'order',
  subject: { reference: `Patient/${patientId}` },
  medicationCodeableConcept: {
    text: 'Amoxicillin 500mg',
  },
  // ... more FHIR fields
});
```

### **Update a Resource**
```typescript
const updated = await medplum.updateResource({
  ...existingResource,
  status: 'active' // Change field
});
```

## 🏥 FHIR Resources We'll Use

### **1. MedicationRequest** (Prescriptions)
- Used for ordering medications
- Fields: medication, dosage, frequency, duration, route
- Status: draft → active → completed

### **2. ServiceRequest** (Labs & Imaging)
- Used for ordering tests and procedures
- Category: lab-procedure vs imaging
- Fields: code (LOINC), priority, indication

### **3. Appointment** (Follow-ups)
- Used for scheduling future visits
- Fields: reason, timeframe, participant

## 🔍 Our Implementation

Check these files:
- `lib/medplum-client.ts` - Singleton client with auto-auth
- `app/api/test-medplum/route.ts` - Connection test endpoint
- `app/api/execute/route.ts` - Creates FHIR resources (to be implemented)

## 🧪 Testing Medplum

**Test the connection:**
Visit: http://localhost:3000/api/test-medplum

Should return:
```json
{
  "success": true,
  "message": "Medplum connection successful!",
  "profile": { "id": "...", "resourceType": "..." }
}
```

## 💡 Key Concepts

### **Singleton Pattern**
We use one shared `MedplumClient` instance for all requests to avoid re-authenticating every time.

### **Server-Side Only**
Medplum operations happen in API routes (server-side) because:
1. Protects your `CLIENT_SECRET`
2. Prevents CORS issues
3. Better security for FHIR operations

### **Draft → Active**
All resources start as `status: 'draft'` so doctors can review before they become official orders.

## 📚 Resources

- [Medplum Docs](https://www.medplum.com/docs)
- [FHIR R4 Spec](https://hl7.org/fhir/R4/)
- [LOINC Codes](https://loinc.org/)
- [RxNorm Codes](https://www.nlm.nih.gov/research/umls/rxnorm/)

## 🚦 Next Steps

1. ✅ Medplum client setup (DONE)
2. ✅ Test connection (DONE)
3. ⏳ Implement API routes (ingest, analyze, execute)
4. ⏳ Build UI components
5. ⏳ Test end-to-end workflow
