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

const SYSTEM_PROMPT = `You are a clinical workflow assistant. Your job is to read a transcript and extract actionable items for the care team. You will output a JSON array of actions. Each action must have a type, reason, and when. For scheduling actions, include a subject and body for the email. The body should be a draft email that the care team can send to the provider or patient. If the patient context is provided, use it to inform the actions. For scheduling actions, you MUST include the practitioner's name and practitioner address in the email body if they are available in the patient context. If you are unsure, say so in the reason field. Do not hallucinate information. Do not include any patient identifiers in the output except for the patientId field.

QUESTIONNAIRE MATCHING RULES - STRICT:
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

IMPORTANT FHIR GUIDELINES:
- MedicationRequest MUST have: resourceType, status ("draft"), intent ("order"), medicationCodeableConcept
- ServiceRequest MUST have: resourceType, status ("draft"), intent ("order"), code
- Use standard coding systems: RxNorm for medications, LOINC for labs, SNOMED CT for procedures
- Set status to "draft" (not "active") since these need doctor approval
- Include display text for all codes for human readability
- If you cannot determine specific codes, use text-only descriptions

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
- Physical exam findings (unless they require action)

OUTPUT FORMAT: Return raw JSON only with an "actions" array.`;

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

    // Fetch available questionnaires from Medplum first
    let availableQuestionnaires: any[] = [];
    try {
      const questionnairesResponse = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/api/questionnaires`);
      if (questionnairesResponse.ok) {
        const data = await questionnairesResponse.json();
        availableQuestionnaires = data.questionnaires || [];
        console.log(`[Analyze] Found ${availableQuestionnaires.length} available questionnaires in Medplum`);
      }
    } catch (err) {
      console.warn('[Analyze] Could not fetch questionnaires, proceeding with generic forms', err);
    }

    // Build questionnaire context for the AI
    const questionnaireContext = availableQuestionnaires.length > 0
      ? `\n\nAVAILABLE QUESTIONNAIRES IN MEDPLUM:\n${availableQuestionnaires.map(q => 
          `- ${q.name} (ID: ${q.id}, Type: ${q.type}): ${q.description || q.title || 'No description'}`
        ).join('\n')}\n\nCRITICAL: You can ONLY create actions for types that have questionnaires above. If a clinical action doesn't match any questionnaire type, DO NOT include it in your response. For example, if there's no referral questionnaire, skip referral actions. If there's no followup questionnaire, skip followup actions.`
      : '';

    // Build user message
    let userMessage = `Analyze this clinical consultation transcript and extract actionable clinical intents:\n\n${transcript}`;
    
    if (patientContext) {
      userMessage += `\n\nPatient Context:\n${patientContext}`;
    }

    userMessage += questionnaireContext;

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
      if (!action.type || !action.description || !action.resource) {
        console.warn('[Analyze] Skipping invalid action:', action);
        return false;
      }
      if (!['medication', 'lab', 'imaging', 'referral', 'followup', 'scheduling'].includes(action.type)) {
        console.warn('[Analyze] Skipping action with invalid type:', action.type);
        return false;
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
        { error: `Claude API error: ${String((error as Error).message)}` },
        { status: (error as any).status || 500 }
      );
    }
    return NextResponse.json(
      { 
        error: error instanceof Error ? (error as Error).message : 'Failed to analyze transcript',
        details: error instanceof Error ? (error as Error).stack : undefined
      },
      { status: 500 }
    );
  }
}
