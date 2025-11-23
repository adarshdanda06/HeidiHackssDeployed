import { NextRequest, NextResponse } from 'next/server';
import Anthropic from '@anthropic-ai/sdk';
import { MedicationRequest, ServiceRequest } from '@medplum/fhirtypes';
import { fillMissingFields } from '@/lib/smart-dummy-data';
import { getMedplumClient } from '@/lib/medplum-client';

// Initialize Anthropic client
const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

interface AnalyzeRequest {
  transcript: string;
  patientContext?: string;
  patient?: any; // Patient data from session context
}

interface ClinicalActionResponse {
  type: 'medication' | 'lab' | 'imaging' | 'referral' | 'followup' | 'scheduling';
  description: string;
  questionnaireId?: string; // Optional: ID of the Medplum Questionnaire to use
  questionnaireName?: string; // Optional: Human-readable name of the questionnaire
  resource: MedicationRequest | ServiceRequest | any;
  reason?: string;
  when?: string;
  subject?: string;
  body?: string;
}

interface AnalyzeResponse {
  actions: ClinicalActionResponse[];
}

const SYSTEM_PROMPT = `You are an expert Clinical AI Assistant specializing in FHIR R4.

Your task is to analyze clinical consultation transcripts and extract actionable clinical intents that can be converted into FHIR R4 resources.

CRITICAL RULES:
1. Output strictly raw JSON. No markdown formatting, no code blocks, no explanations.
2. Extract clinical actions into an array called "actions"
3. For each action, generate a Draft FHIR Resource:
   - MedicationRequest for medications/prescriptions
   - ServiceRequest for labs, imaging, referrals
   - QuestionnaireResponse for mental health screenings (PHQ-4)
   - Appointment for follow-ups and scheduling
4. Include a human-readable "description" for UI display
5. Categorize each action with a "type": "medication", "lab", "imaging", "referral", "followup", "questionnaire_response", or "scheduling"
6. **QUESTIONNAIRE MATCHING RULES - STRICT**:
   - When the user message includes AVAILABLE QUESTIONNAIRES, you MUST ONLY select from that exact list
   - NEVER create an action type unless a matching questionnaire exists in the provided list
   - If a clinical action doesn't have a matching questionnaire, DO NOT include that action
   - Match action type to questionnaire type: medication→medication, lab→lab, imaging→imaging, referral→referral, followup→followup
   - Include "questionnaireId" and "questionnaireName" fields for EVERY action that has a matching questionnaire
   - If no questionnaire matches the clinical intent, skip that action entirely UNLESS it's a scheduling action

TYPE DEFINITIONS:
- "medication": Prescriptions and drug orders
- "lab": Laboratory tests
- "imaging": Radiology/Imaging studies
- "referral": Referrals to other specialists
- "followup": Clinical follow-up appointments within the EMR (e.g. "See patient in 2 weeks")
- "scheduling": Email communications to the patient regarding next steps, follow-ups, or summary of instructions.
   - For "scheduling" actions, YOU MUST INCLUDE:
     - "reason": Internal reason for the scheduling (brief).
     - "when": Suggested time (brief).
     - "subject": A user-friendly email subject line.
     - "body": A warm, user-friendly email body that reads through the context in the transcript, summarizes after-meeting instructions, and includes follow-up meeting time if necessary. Do not include any other context or meta-text.

OUTPUT FORMAT (raw JSON only):
{
  "actions": [
    {
      "type": "medication",
      "description": "Prescribe Amoxicillin 500mg three times daily for 7 days",
      "questionnaireId": "abc-123-def",
      "questionnaireName": "MedicationPrescriptionOrderForm",
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
    },
    {
      "type": "lab",
      "description": "Order Complete Blood Count (CBC)",
      "questionnaireId": "xyz-789-ghi",
      "questionnaireName": "BloodTestOrderForm",
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
    },
    {
      "type": "scheduling",
      "description": "Schedule a meeting to review surgical options",
      "reason": "Patient requested detailed discussion about surgical risks and benefits",
      "when": "Next Tuesday afternoon",
      "subject": "Follow-up: Surgical Options Discussion",
      "body": "Hi [Patient Name],\n\nIt was good to speak with you today. As discussed, we should schedule a time next Tuesday afternoon to go over the surgical options in more detail. Please let us know what time works best for you.\n\nBest regards,\nDr. Smith",
      "resource": {
        "resourceType": "Appointment",
        "status": "proposed",
        "description": "Meeting to review surgical options",
        "start": "2023-11-28T14:00:00Z" 
      }
    }
  ]
}

IMPORTANT FHIR GUIDELINES:
- MedicationRequest MUST have: resourceType, status ("draft"), intent ("order"), medicationCodeableConcept
- ServiceRequest MUST have: resourceType, status ("draft"), intent ("order"), code
- **CRITICAL FOR IMAGING/LABS**: ServiceRequest MUST include:
  * code: { text: "name of test/scan" }
  * bodySite: [{ text: "anatomical location" }] (e.g., "chest", "abdomen", "pelvis")
  * priority: "routine" | "urgent" | "asap" | "stat"
  * reasonCode: [{ text: "clinical indication/reason for ordering" }]
  * category: [{ text: "type of service" }] (e.g., "Laboratory", "Radiology")
- Use standard coding systems: RxNorm for medications, LOINC for labs, SNOMED CT for procedures
- Set status to "draft" (not "active") since these need doctor approval
- Include display text for all codes for human readability
- Include as much detail as possible from the transcript in the FHIR resource fields
- If you cannot determine specific codes, use text-only descriptions

**EXAMPLE IMAGING ORDER WITH QUESTIONNAIRE:**
{
  "type": "imaging",
  "description": "Order CT scan of chest to evaluate pneumonia",
  "questionnaireId": "xyz-123",
  "questionnaireName": "CT Scan Request Form",
  "resource": {
    "resourceType": "QuestionnaireResponse",
    "status": "in-progress",
    "questionnaire": "Questionnaire/xyz-123",
    "item": [
      {
        "linkId": "patientName",
        "answer": [{"valueString": "John Smith"}]
      },
      {
        "linkId": "examType",
        "answer": [{"valueString": "CT Scan"}]
      },
      {
        "linkId": "bodyRegion",
        "answer": [{"valueString": "chest"}]
      },
      {
        "linkId": "clinicalIndication",
        "answer": [{"valueString": "Evaluate pneumonia, patient has persistent cough and fever"}]
      },
      {
        "linkId": "priority",
        "answer": [{"valueCoding": {"code": "routine", "display": "Routine"}}]
      }
    ]
  }
}

**IMPORTANT:** The "item" array should contain entries for EACH linkId from the questionnaire definition, with answers extracted from the transcript.

MENTAL HEALTH SCREENING LOGIC (PHQ-4):
IF the patient mentions symptoms of Anxiety (nervous, edge, worry) OR Depression (down, lost interest):
- Create a "QuestionnaireResponse" resource.
- Set "questionnaire" to "Questionnaire/bb1ece1d-7116-49d9-a082-86262208b517" (The ID of the PHQ-4 form).
- Extract answers based on the transcript using these EXACT linkIds:
  - Anxiety Q1 (Nervous/Edge) -> linkId: "/69725-0"
  - Anxiety Q2 (Worrying)     -> linkId: "/68509-9"
  - Depression Q3 (Interest)  -> linkId: "/44250-9"
  - Depression Q4 (Hopeless)  -> linkId: "/44255-8"
- For the answer values, you MUST use valueCoding from this list:
  - "Not at all" (Code: LA6568-5)
  - "Several days" (Code: LA6569-3)
  - "More than half the days" (Code: LA6570-1)
  - "Nearly every day" (Code: LA6571-9)

EXAMPLE QUESTIONNAIRE OUTPUT:
{
  "type": "questionnaire_response",
  "description": "Auto-filled PHQ-4 Mental Health Screen",
  "resource": {
    "resourceType": "QuestionnaireResponse",
    "status": "completed",
    "questionnaire": "Questionnaire/bb1ece1d-7116-49d9-a082-86262208b517",
    "item": [
      {
        "linkId": "/69725-0",
        "answer": [{ "valueCoding": { "code": "LA6569-3", "display": "Several days" } }]
      }
      // ... other items
    ]
  }
}

Extract ONLY clinically actionable items:
- Medications/prescriptions
- Lab tests
- Imaging studies
- Specialist referrals
- Follow-up appointments
- Scheduling/Administrative meetings

Do NOT extract:
- General advice or counseling
- Lifestyle recommendations without specific follow-up
- Past medical history
- Physical exam findings (unless they require action)`;

export async function POST(request: NextRequest) {
  try {
    const body: AnalyzeRequest = await request.json();
    const { transcript, patientContext, patient } = body;

    if (!transcript || typeof transcript !== 'string') {
      return NextResponse.json(
        { error: 'Missing or invalid transcript in request body' },
        { status: 400 }
      );
    }

    if (!process.env.ANTHROPIC_API_KEY) {
      return NextResponse.json(
        { error: 'ANTHROPIC_API_KEY not configured' },
        { status: 500 }
      );
    }

    // Fetch available questionnaires from Medplum first
    let availableQuestionnaires: any[] = [];
    try {
      const questionnairesResponse = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/api/questionnaires`);
      if (questionnairesResponse.ok) {
        const data = await questionnairesResponse.json();
        availableQuestionnaires = data.questionnaires || [];
        console.log(`[Analyze] Found ${availableQuestionnaires.length} available questionnaires in Medplum`);
        
        // Fetch full questionnaire definitions for the relevant ones
        const questionnaireDetails = await Promise.all(
          availableQuestionnaires.map(async (q) => {
            try {
              const detailResponse = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/api/questionnaire/${q.id}`);
              if (detailResponse.ok) {
                const detail = await detailResponse.json();
                return {
                  id: q.id,
                  name: q.name,
                  type: q.type,
                  title: q.title,
                  description: q.description,
                  items: detail.questionnaire?.item || []
                };
              }
            } catch (err) {
              console.warn(`[Analyze] Failed to fetch questionnaire ${q.id}:`, err);
            }
            return null;
          })
        );
        
        availableQuestionnaires = questionnaireDetails.filter(q => q !== null);
        console.log(`[Analyze] Loaded ${availableQuestionnaires.length} full questionnaire definitions`);
      }
    } catch (err) {
      console.warn('[Analyze] Could not fetch questionnaires, proceeding with generic forms', err);
    }

    // Build questionnaire context for the AI
    const questionnaireContext = availableQuestionnaires.length > 0
      ? `\n\nAVAILABLE QUESTIONNAIRES IN MEDPLUM:\n${availableQuestionnaires.map(q => {
          const itemsSummary = q.items.map((item: any) => 
            `  - linkId: "${item.linkId}", text: "${item.text}", type: ${item.type}${item.required ? ' (REQUIRED)' : ''}${
              item.answerOption ? `, options: [${item.answerOption.map((opt: any) => 
                opt.valueCoding?.display || opt.valueString
              ).join(', ')}]` : ''
            }`
          ).join('\n');
          
          return `\n**Questionnaire: ${q.name}** (ID: ${q.id}, Type: ${q.type})\nTitle: ${q.title || 'N/A'}\nDescription: ${q.description || 'N/A'}\nFields:\n${itemsSummary}\n`;
        }).join('\n')}\n\n**CRITICAL INSTRUCTIONS FOR USING QUESTIONNAIRES:**
- You MUST generate a QuestionnaireResponse resource that fills out ALL the fields (linkIds) listed above
- For each action, include "questionnaireId" and "questionnaireName" 
- The "resource" MUST be a QuestionnaireResponse with "item" array containing answers for EVERY single linkId from the questionnaire definition
- For NESTED GROUPS (type: group), you must create items with nested "item" arrays, not "answer" arrays
- For QUESTION FIELDS (non-group types), create answer arrays with the appropriate value type
- Extract values from the transcript and match them to the appropriate linkIds
- For patient demographic fields, use the EXACT patient data provided in the prompt
- For fields not mentioned in the transcript, use intelligent defaults based on the field name and clinical context
- For choice fields with answerOption, you MUST use one of the provided options exactly as specified (use valueCoding with code and display)
- For boolean fields, use valueBoolean (not valueString)
- If a clinical action doesn't match any questionnaire type, skip that action entirely

**NESTED GROUP STRUCTURE EXAMPLE:**
If a questionnaire has groups with nested items, the item array should contain BOTH group containers AND their nested fields.
For a group linkId "exam-details" containing "examType" and "bodyRegion", format like:
{
  "linkId": "exam-details",
  "item": [
    { "linkId": "examType", "answer": [{"valueString": "CT Scan"}] },
    { "linkId": "bodyRegion", "answer": [{"valueString": "Chest"}] }
  ]
}

**YOU MUST INCLUDE AN ITEM (or nested items for groups) FOR EVERY SINGLE LINKID IN THE QUESTIONNAIRE!**`
      : '';

    // Build user message
    let userMessage = `Analyze this clinical consultation transcript and extract actionable clinical intents:\n\n${transcript}`;
    
    // Add patient data for better autofill
    if (patient) {
      userMessage += `\n\n**PATIENT INFORMATION (Use this to fill patient fields):**
- Patient Name: ${patient.name || 'Unknown'}
- Date of Birth: ${patient.dob || 'Unknown'}
- Age: ${patient.age || 'Unknown'}
- Gender: ${patient.gender || 'Unknown'}
- MRN: ${patient.mrn || 'Unknown'}
- Phone: ${patient.phone || 'Unknown'}
- Email: ${patient.email || 'Unknown'}
- Address: ${patient.address || 'Unknown'}

**CRITICAL: When generating QuestionnaireResponse items, USE THE EXACT VALUES ABOVE for patient demographic fields!**`;
    } else if (patientContext) {
      userMessage += `\n\nPatient Context:\n${patientContext}`;
    }

    userMessage += questionnaireContext;

    console.log('[Analyze] ====== SENDING TO CLAUDE ======');
    console.log('[Analyze] User message length:', userMessage.length);
    console.log('[Analyze] Questionnaire context preview:', questionnaireContext.substring(0, 500));

    // Call Claude
    console.log('[Analyze] Calling Claude API...');
    const message = await anthropic.messages.create({
      model: 'claude-sonnet-4-5-20250929',
      max_tokens: 8192,
      temperature: 0.3, // Lower temperature for more consistent structured output
      system: SYSTEM_PROMPT,
      messages: [
        {
          role: 'user',
          content: userMessage,
        },
      ],
    });

    // Extract text response
    const textContent = message.content.find((block) => block.type === 'text');
    if (!textContent || textContent.type !== 'text') {
      throw new Error('No text response from Claude');
    }

    const responseText = textContent.text;
    console.log('[Analyze] ====== CLAUDE RESPONSE ======');
    console.log('[Analyze] Full response:', responseText);
    console.log('[Analyze] ====== END RESPONSE ======');

    // Parse JSON from response (handle cases where Claude adds markdown or extra text)
    let parsedResponse: AnalyzeResponse;
    try {
      // Try direct parse first
      parsedResponse = JSON.parse(responseText);
    } catch (parseError) {
      // If direct parse fails, try to extract JSON from markdown code blocks or surrounding text
      console.log('[Analyze] Direct JSON parse failed, attempting extraction...');
      
      // Remove markdown code blocks
      let cleanedText = responseText.replace(/```json\n?/g, '').replace(/```\n?/g, '');
      
      // Try to find JSON object boundaries
      const jsonMatch = cleanedText.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        cleanedText = jsonMatch[0];
      }
      
      try {
        parsedResponse = JSON.parse(cleanedText);
      } catch (secondError) {
        console.error('[Analyze] Failed to parse Claude response:', responseText);
        throw new Error('Failed to parse JSON from Claude response');
      }
    }

    // Validate response structure
    if (!parsedResponse.actions || !Array.isArray(parsedResponse.actions)) {
      console.error('[Analyze] Invalid response structure:', parsedResponse);
      throw new Error('Invalid response structure: missing actions array');
    }

    // Validate each action and fill missing fields with smart dummy data
    const validatedActions = [];
    
    for (const action of parsedResponse.actions) {
      // Basic validation
      if (!action.type || !action.description || !action.resource) {
        console.warn('[Analyze] Skipping invalid action:', action);
        continue;
      }
      if (!['medication', 'lab', 'imaging', 'referral', 'followup', 'questionnaire_response', 'scheduling'].includes(action.type)) {
        console.warn('[Analyze] Skipping action with invalid type:', action.type);
        continue;
      }
      
      // If this action has a QuestionnaireResponse, fill missing fields with smart dummy data
      if (action.resource.resourceType === 'QuestionnaireResponse' && action.questionnaireId) {
        try {
          console.log(`[Analyze] Filling missing fields for questionnaire: ${action.questionnaireName}`);
          
          // Fetch the full questionnaire to know all required fields
          const medplum = await getMedplumClient();
          const questionnaire = await medplum.readResource('Questionnaire', action.questionnaireId);
          
          // Extract clinical context from description
          const clinicalContext = action.description;
          const urgency = action.description.toLowerCase().includes('stat') || 
                         action.description.toLowerCase().includes('immediate') ? 'stat' :
                         action.description.toLowerCase().includes('urgent') ? 'urgent' : 'routine';
          
          // Fill missing fields with smart dummy data
          const filledResource = fillMissingFields(
            action.resource,
            questionnaire,
            {
              patientName: patient?.name,
              patientAge: patient?.age,
              clinicalContext,
              urgency
            }
          );
          
          console.log(`[Analyze] Fields filled for ${action.questionnaireName}:`, {
            originalItemCount: action.resource.item?.length || 0,
            filledItemCount: filledResource.item?.length || 0
          });
          
          action.resource = filledResource;
        } catch (fillError) {
          console.error(`[Analyze] Error filling fields for ${action.questionnaireName}:`, fillError);
          // Continue with original resource if filling fails
        }
      }
      
      validatedActions.push(action);
    }

    console.log(`[Analyze] Successfully processed ${validatedActions.length} actions`);

    return NextResponse.json({
      actions: validatedActions,
    });

  } catch (error) {
    console.error('[Analyze] Error:', error);
    
    if (error instanceof Anthropic.APIError) {
      return NextResponse.json(
        { error: `Claude API error: ${error.message}` },
        { status: error.status || 500 }
      );
    }

    return NextResponse.json(
      { 
        error: error instanceof Error ? error.message : 'Failed to analyze transcript',
        details: error instanceof Error ? error.stack : undefined
      },
      { status: 500 }
    );
  }
}
