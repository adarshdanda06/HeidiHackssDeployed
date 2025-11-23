import { NextRequest, NextResponse } from 'next/server';
import { Questionnaire } from '@medplum/fhirtypes';
import { getMedplumClient } from '@/lib/medplum-client';

/**
 * GET /api/questionnaire/[id]
 * Fetch a specific Questionnaire from Medplum by ID
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id: questionnaireId } = await params;

    if (!questionnaireId) {
      return NextResponse.json(
        { error: 'Questionnaire ID is required' },
        { status: 400 }
      );
    }

    console.log(`[Questionnaire] Fetching questionnaire: ${questionnaireId}`);

    const medplum = await getMedplumClient();
    
    // Fetch the specific questionnaire
    const questionnaire = await medplum.readResource('Questionnaire', questionnaireId) as Questionnaire;

    if (!questionnaire) {
      return NextResponse.json(
        { error: 'Questionnaire not found' },
        { status: 404 }
      );
    }

    console.log(`[Questionnaire] Found: ${questionnaire.title || questionnaire.name}`);

    return NextResponse.json({ questionnaire });

  } catch (error: any) {
    console.error('[Questionnaire] Error:', error);
    
    if (error.message?.includes('not found') || error.status === 404) {
      return NextResponse.json(
        { error: 'Questionnaire not found' },
        { status: 404 }
      );
    }

    return NextResponse.json(
      { 
        error: 'Failed to fetch questionnaire',
        details: error.message 
      },
      { status: 500 }
    );
  }
}
