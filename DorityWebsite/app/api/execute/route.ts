import { NextRequest, NextResponse } from 'next/server';
import { MedicationRequest, ServiceRequest, Appointment, QuestionnaireResponse, Resource } from '@medplum/fhirtypes';
import { getMedplumClient } from '@/lib/medplum-client';
import sgMail from '@sendgrid/mail';

/**
 * Request body structure for execute endpoint
 */
interface ExecuteRequest {
  action: {
    type: 'medication' | 'lab' | 'imaging' | 'referral' | 'followup' | 'scheduling' | 'questionnaire_response';
    description: string;
    resource: MedicationRequest | ServiceRequest | Appointment | QuestionnaireResponse;
    email?: string; // For scheduling
    subject?: string; // For scheduling
    body?: string; // For scheduling
  };
  patientId: string; // Medplum Patient resource ID
}

// Hardcoded default email as requested
const TARGET_EMAIL = "adarsh.danda1@gmail.com";

/**
 * Response structure for successful execution
 */
interface ExecuteResponse {
  success: true;
  resourceId: string;
  resourceType: string;
  message: string;
}

/**
 * Execute endpoint - Creates FHIR resources in Medplum
 * 
 * This endpoint takes a draft clinical action (from /api/analyze) and:
 * 1. Attaches the patient reference
 * 2. Writes the FHIR resource to Medplum
 * 3. Returns the created resource ID
 * 
 * POST /api/execute
 * Body: { action: ClinicalAction, patientId: string }
 */
export async function POST(request: NextRequest) {
  try {
    // Parse request body
    const body: ExecuteRequest = await request.json();
    const { action, patientId } = body;

    // Validate required fields
    if (!action) {
      return NextResponse.json(
        { error: 'Missing required field: action' },
        { status: 400 }
      );
    }

    if (!patientId || typeof patientId !== 'string') {
      return NextResponse.json(
        { error: 'Missing or invalid required field: patientId' },
        { status: 400 }
      );
    }

    if (!action.resource) {
      return NextResponse.json(
        { error: 'Missing resource in action' },
        { status: 400 }
      );
    }

    // Validate resource type
    const resourceType = action.resource.resourceType;
    
    // Handle QuestionnaireResponse - convert to ServiceRequest based on action type
    if (resourceType === 'QuestionnaireResponse') {
      console.log('[Execute] Converting QuestionnaireResponse to ServiceRequest/MedicationRequest');
      
      // Extract data from questionnaire response
      const extractValue = (linkId: string): any => {
        const findInItems = (items: any[]): any => {
          for (const item of items) {
            if (item.linkId === linkId && item.answer && item.answer.length > 0) {
              const answer = item.answer[0];
              return answer.valueString || answer.valueBoolean || answer.valueInteger || 
                     answer.valueDecimal || answer.valueDate || answer.valueCoding?.display;
            }
            if (item.item && Array.isArray(item.item)) {
              const found = findInItems(item.item);
              if (found !== undefined) return found;
            }
          }
          return undefined;
        };
        return findInItems(action.resource.item || []);
      };
      
      // Convert to ServiceRequest for imaging/lab/referral actions
      if (action.type === 'imaging' || action.type === 'lab' || action.type === 'referral') {
        action.resource = {
          resourceType: 'ServiceRequest',
          status: 'draft',
          intent: 'order',
          category: [{
            text: action.type === 'imaging' ? 'Radiology' : 
                  action.type === 'lab' ? 'Laboratory' : 'Referral'
          }],
          code: {
            text: extractValue('examType') || extractValue('testType') || extractValue('scanType') || action.description
          },
          bodySite: extractValue('bodyRegion') ? [{
            text: extractValue('bodyRegion')
          }] : undefined,
          priority: (extractValue('priority') || 'routine').toLowerCase(),
          reasonCode: extractValue('clinicalIndication') ? [{
            text: extractValue('clinicalIndication')
          }] : undefined,
          note: extractValue('notes') || extractValue('additionalInfo') ? [{
            text: extractValue('notes') || extractValue('additionalInfo')
          }] : undefined,
          subject: {
            reference: `Patient/${patientId}`
          }
        } as ServiceRequest;
      }
      // Convert to MedicationRequest for medication actions
      else if (action.type === 'medication') {
        action.resource = {
          resourceType: 'MedicationRequest',
          status: 'draft',
          intent: 'order',
          medicationCodeableConcept: {
            text: extractValue('medication') || action.description
          },
          subject: {
            reference: `Patient/${patientId}`
          },
          dosageInstruction: [{
            text: `${extractValue('dose') || ''} ${extractValue('route') || 'oral'} ${extractValue('frequency') || ''}`.trim()
          }],
          dispenseRequest: extractValue('quantity') ? {
            quantity: {
              value: parseInt(extractValue('quantity')) || 30
            }
          } : undefined
        } as MedicationRequest;
      }
    }
    
    // Re-validate after potential conversion
    const finalResourceType = action.resource.resourceType;
    if (finalResourceType !== 'MedicationRequest' && finalResourceType !== 'ServiceRequest' && finalResourceType !== 'Appointment' && finalResourceType !== 'QuestionnaireResponse') {
      return NextResponse.json(
        { 
          error: `Invalid resource type: ${finalResourceType}. Expected MedicationRequest, ServiceRequest, Appointment, or QuestionnaireResponse, Failed to create resource in Medplum` 
        },
        { status: 400 }
      );
    }

    console.log(`[Execute] Processing ${finalResourceType} for patient ${patientId}`);
    console.log(`[Execute] Action type: ${action.type}, Description: ${action.description}`);

    // Handle Email Sending for Scheduling Actions
    // Disabled: Consolidating into single aftercare email
    /*
    if (action.type === 'scheduling') {
      console.log('📧 Processing scheduling action email...');
      
      if (process.env.SENDGRID_API_KEY) {
        try {
          sgMail.setApiKey(process.env.SENDGRID_API_KEY);
          
          const recipient = action.email || TARGET_EMAIL;

          const msg = {
            to: recipient,
            from: process.env.SENDGRID_FROM_EMAIL || 'adarsh.danda1@gmail.com', // Must be verified in SendGrid
            subject: action.subject || `New Meeting Request: ${action.description}`,
            text: action.body || `A new follow-up meeting has been suggested.\n\nDetails: ${action.description}\n\nPatient ID: ${patientId}`,
            html: `
              <h2>New Meeting Request</h2>
              <p><strong>Details:</strong> ${action.description}</p>
              <p><strong>Message:</strong> ${action.body || action.description}</p>
              <hr/>
              <p><em>Generated by Clinical Action Layer</em></p>
            `,
          };

          await sgMail.send(msg);
          console.log('✅ Email sent successfully to', recipient);
        } catch (emailError: any) {
          console.error('❌ Failed to send email:', emailError);
          if (emailError.response) {
            console.error(emailError.response.body);
          }
          // We continue to create the FHIR resource even if email fails
        }
      } else {
        console.warn('⚠️ SENDGRID_API_KEY not found. Email skipped.');
      }
    }
    */

    // Get authenticated Medplum client
    let medplum;
    try {
      medplum = await getMedplumClient();
    } catch (authError) {
      console.error('[Execute] Medplum authentication failed:', authError);
      return NextResponse.json(
        { error: 'Failed to authenticate with Medplum' },
        { status: 503 }
      );
    }

    // Clone the resource to avoid mutating the original
    const resourceToCreate: any = { ...action.resource };

    // Attach patient reference to the resource
    if (finalResourceType === 'Appointment') {
        // For Appointment, the patient is a participant
        resourceToCreate.participant = [
            {
                actor: {
                    reference: `Patient/${patientId}`,
                    display: undefined
                },
                status: 'needs-action'
            },
            ...(resourceToCreate.participant || [])
        ];
        
        // CRITICAL: Medplum requires both start and end, or neither
        // If start exists but no end, add a default 1-hour end time
        if (resourceToCreate.start && !resourceToCreate.end) {
            const startDate = new Date(resourceToCreate.start);
            const endDate = new Date(startDate.getTime() + 60 * 60 * 1000); // Add 1 hour
            resourceToCreate.end = endDate.toISOString();
            console.log('[Execute] Added end time to Appointment (1 hour after start)');
        }
        // If neither start nor end, remove both to satisfy constraint
        if (!resourceToCreate.start && !resourceToCreate.end) {
            delete resourceToCreate.start;
            delete resourceToCreate.end;
            console.log('[Execute] Removed start/end from Appointment (neither specified)');
        }
    } else if (finalResourceType === 'QuestionnaireResponse') {
        // For QuestionnaireResponse, patient is subject and source
        resourceToCreate.subject = {
            reference: `Patient/${patientId}`,
            display: undefined, 
        };
        resourceToCreate.source = {
            reference: `Patient/${patientId}`,
            display: undefined,
        };
    } else {
        // For MedicationRequest/ServiceRequest, patient is subject
        resourceToCreate.subject = {
            reference: `Patient/${patientId}`,
            display: undefined, 
        };
    }

    // Ensure status is "draft" or "proposed"
    if (finalResourceType === 'Appointment') {
        resourceToCreate.status = 'proposed';
    } else if (finalResourceType === 'QuestionnaireResponse') {
        // QuestionnaireResponse usually 'completed' or 'in-progress'
        // If not set, default to 'completed' since these are historical/screening
        if (!resourceToCreate.status) {
            resourceToCreate.status = 'completed';
        }
    } else {
        resourceToCreate.status = 'draft';
        resourceToCreate.intent = 'order';
    }

    console.log('[Execute] Creating resource in Medplum:', {
      resourceType: finalResourceType,
      patientId,
      status: resourceToCreate.status,
    });
    console.log('[Execute] Full resource to create:', JSON.stringify(resourceToCreate, null, 2));

    // Create resource in Medplum
    let createdResource: Resource;
    try {
      createdResource = await medplum.createResource(resourceToCreate as any);
      console.log(`[Execute] ✅ Successfully created ${finalResourceType} with ID: ${createdResource.id}`);
    } catch (createError: any) {
      console.error('[Execute] Medplum createResource failed:', createError);
      console.error('[Execute] Error details:', {
        message: createError.message,
        outcome: createError.outcome,
        issues: createError.outcome?.issue
      });
      
      if (createError.message?.includes('Patient')) {
        return NextResponse.json(
          { 
            error: `Patient not found: ${patientId}`,
            details: 'The specified patient ID does not exist in Medplum' 
          },
          { status: 404 }
        );
      }

      return NextResponse.json(
        { 
          error: 'Failed to create resource in Medplum',
          details: createError.message || 'Unknown error',
          outcome: createError.outcome
        },
        { status: 500 }
      );
    }

    // Return success response
    const response: ExecuteResponse = {
      success: true,
      resourceId: createdResource.id!,
      resourceType: createdResource.resourceType!,
      message: `Successfully created ${createdResource.resourceType} for patient ${patientId}`,
    };

    return NextResponse.json(response, { status: 201 });

  } catch (error) {
    console.error('[Execute] Unexpected error:', error);
    
    return NextResponse.json(
      { 
        error: error instanceof Error ? error.message : 'Internal server error',
        details: error instanceof Error ? error.stack : undefined
      },
      { status: 500 }
    );
  }
}
