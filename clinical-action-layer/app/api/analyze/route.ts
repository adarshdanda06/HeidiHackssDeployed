import { NextRequest, NextResponse } from 'next/server';
import Anthropic from '@anthropic-ai/sdk';
import { MedicationRequest, ServiceRequest } from '@medplum/fhirtypes';

// Initialize Anthropic client
const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

interface AnalyzeRequest {
  transcript: string;
  patientContext?: string;
}

interface ClinicalActionResponse {
  type: 'medication' | 'lab' | 'imaging' | 'referral' | 'followup' | 'scheduling';
  description: string;
  reason?: string;
  when?: string;
  subject?: string;
  body?: string;
  resource: MedicationRequest | ServiceRequest | any;
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
4. Include a human-readable "description" for UI display
Categorize each action with a "type": "medication", "lab", "imaging", "referral", "followup", "scheduling", or "questionnaire_response"

For SCHEDULING actions:
- MUST include: reason, when, subject, body
- The body MUST include the practitioner's name and practitioner address (if available in patient context)
- Format the email in a warm, professional tone
- Summarize key discussion points and next steps from the consultation

OUTPUT FORMAT (raw JSON only):
{
  "actions": [
    {
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
    },
    {
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
  ]
}

IMPORTANT FHIR GUIDELINES:
- MedicationRequest MUST have: resourceType, status ("draft"), intent ("order"), medicationCodeableConcept
- ServiceRequest MUST have: resourceType, status ("draft"), intent ("order"), code
- Use standard coding systems: RxNorm for medications, LOINC for labs, SNOMED CT for procedures
- Set status to "draft" (not "active") since these need doctor approval
- Include display text for all codes for human readability
- If you cannot determine specific codes, use text-only descriptions

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

Do NOT extract:
- General advice or counseling
- Lifestyle recommendations without specific follow-up
- Past medical history
- Physical exam findings (unless they require action)`;

export async function POST(request: NextRequest) {
  try {
    const body: AnalyzeRequest = await request.json();
    const { transcript, patientContext } = body;

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

    // Build user message
    let userMessage = `Analyze this clinical consultation transcript and extract actionable clinical intents:\n\n${transcript}`;
    
    if (patientContext) {
      userMessage += `\n\nPatient Context:\n${patientContext}`;
    }

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
    console.log('[Analyze] Claude response received:', responseText.substring(0, 200) + '...');

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

    // Validate each action
    const validatedActions = parsedResponse.actions.filter((action) => {
      // Validate type
      if (!action.type || !['medication', 'lab', 'imaging', 'referral', 'followup', 'scheduling', 'questionnaire_response'].includes(action.type)) {
        console.warn('[Analyze] Skipping action with invalid type:', action.type);
        return false;
      }

      // Validate description
      if (!action.description) {
        console.warn('[Analyze] Skipping action without description:', action);
        return false;
      }

      // Scheduling actions need: reason, when, subject, body (no resource needed)
      if (action.type === 'scheduling') {
        if (!action.reason || !action.when || !action.subject || !action.body) {
          console.warn('[Analyze] Skipping invalid scheduling action - missing required fields:', action);
          return false;
        }
        return true;
      }

      // Other action types need a resource (handle both 'resource' and 'fhirResource' field names)
      const hasResource = action.resource || (action as any).fhirResource;
      if (!hasResource) {
        console.warn('[Analyze] Skipping action without resource:', action);
        return false;
      }

      // Normalize fhirResource to resource for consistency
      if ((action as any).fhirResource && !action.resource) {
        action.resource = (action as any).fhirResource;
        delete (action as any).fhirResource;
      }

      return true;
    });

    console.log(`[Analyze] Successfully extracted ${validatedActions.length} actions`);

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
