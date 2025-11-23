import { NextRequest, NextResponse } from 'next/server';
import { MedicationRequest, ServiceRequest, Resource } from '@medplum/fhirtypes';
import { getMedplumClient } from '@/lib/medplum-client';

/**
 * Request body structure for execute endpoint
 */
interface ExecuteRequest {
  action: {
    type: 'medication' | 'lab' | 'imaging' | 'referral' | 'followup';
    description: string;
    resource: MedicationRequest | ServiceRequest;
  };
  patientId: string; // Medplum Patient resource ID
}

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
    if (resourceType !== 'MedicationRequest' && resourceType !== 'ServiceRequest') {
      return NextResponse.json(
        { 
          error: `Invalid resource type: ${resourceType}. Expected MedicationRequest or ServiceRequest` 
        },
        { status: 400 }
      );
    }

    console.log(`[Execute] Processing ${resourceType} for patient ${patientId}`);
    console.log(`[Execute] Action type: ${action.type}, Description: ${action.description}`);

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
    const resourceToCreate = { ...action.resource };

    // Attach patient reference to the resource
    // This links the order/request to the specific patient
    resourceToCreate.subject = {
      reference: `Patient/${patientId}`,
      display: undefined, // Will be populated by Medplum if patient exists
    };

    // Ensure status is "draft" for doctor review workflow
    resourceToCreate.status = 'draft';

    // Ensure intent is "order" (standard for prescriptions/orders)
    resourceToCreate.intent = 'order';

    console.log('[Execute] Creating resource in Medplum:', {
      resourceType,
      patientId,
      hasSubject: !!resourceToCreate.subject,
      status: resourceToCreate.status,
      intent: resourceToCreate.intent,
    });

    // Create resource in Medplum
    let createdResource: Resource;
    try {
      createdResource = await medplum.createResource(resourceToCreate as any);
      console.log(`[Execute] ✅ Successfully created ${resourceType} with ID: ${createdResource.id}`);
    } catch (createError: any) {
      console.error('[Execute] Medplum createResource failed:', createError);
      
      // Provide helpful error messages for common issues
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
          details: createError.message || 'Unknown error'
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
