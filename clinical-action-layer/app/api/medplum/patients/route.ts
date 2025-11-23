import { NextRequest, NextResponse } from 'next/server';
import { getMedplumClient } from '@/lib/medplum-client';
import { Patient } from '@medplum/fhirtypes';

interface SimplifiedPatient {
  patientId: string;
  patientFirstName: string | undefined;
  patientLastName: string | undefined;
}

export async function GET(request: NextRequest) {
  try {
    console.log('[Patients] Fetching patients from Medplum...');
    
    const medplum = await getMedplumClient();
    
    // Fetch up to 100 patients from Medplum FHIR API
    const patients = await medplum.searchResources('Patient', {
      _count: '100',
    });

    console.log(`[Patients] Found ${patients.length} patients`);

    // Transform to simplified format (matching your friend's structure)
    const simplifiedPatients: SimplifiedPatient[] = patients.map((patient: Patient) => ({
      patientId: patient.id || '',
      patientFirstName: patient.name?.[0]?.given?.[0],
      patientLastName: patient.name?.[0]?.family,
    }));

    return NextResponse.json(simplifiedPatients);

  } catch (error) {
    console.error('[Patients] Error fetching patients:', error);
    
    return NextResponse.json(
      { 
        error: error instanceof Error ? error.message : 'Failed to fetch patients',
        details: error instanceof Error ? error.stack : undefined
      },
      { status: 500 }
    );
  }
}
