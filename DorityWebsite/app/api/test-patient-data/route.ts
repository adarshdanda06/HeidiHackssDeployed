import { NextRequest, NextResponse } from 'next/server';
import { getMedplumClient } from '@/lib/medplum-client';
import { extractPatientData } from '@/lib/patient-utils';

/**
 * Test endpoint to inspect all patient data available from Medplum
 * Usage: GET /api/test-patient-data?patientId=<patient-id>
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const patientId = searchParams.get('patientId');

    if (!patientId) {
      // Get first available patient for testing
      const medplum = await getMedplumClient();
      const patients = await medplum.searchResources('Patient', { _count: '1' });
      
      if (patients.length === 0) {
        return NextResponse.json({ error: 'No patients found in Medplum' }, { status: 404 });
      }

      return NextResponse.json({
        message: 'No patientId provided. Use ?patientId=<id> to test a specific patient.',
        availablePatient: {
          id: patients[0].id,
          name: patients[0].name?.[0]?.text || 
                `${patients[0].name?.[0]?.given?.[0] || ''} ${patients[0].name?.[0]?.family || ''}`.trim(),
          testUrl: `/api/test-patient-data?patientId=${patients[0].id}`
        }
      });
    }

    // Fetch the specific patient
    const medplum = await getMedplumClient();
    const patient = await medplum.readResource('Patient', patientId);

    // Extract comprehensive data
    const extractedData = extractPatientData(patient);

    // Return both raw and extracted data
    return NextResponse.json({
      patientId,
      rawFHIR: patient,
      extractedData,
      availableFields: {
        core: {
          id: !!patient.id,
          active: patient.active !== undefined,
          resourceType: patient.resourceType
        },
        names: {
          count: patient.name?.length || 0,
          hasFamily: !!patient.name?.[0]?.family,
          hasGiven: !!patient.name?.[0]?.given?.length,
          hasPrefix: !!patient.name?.[0]?.prefix?.length,
          hasSuffix: !!patient.name?.[0]?.suffix?.length,
          hasNickname: patient.name?.some(n => n.use === 'nickname')
        },
        demographics: {
          hasBirthDate: !!patient.birthDate,
          hasGender: !!patient.gender,
          hasMaritalStatus: !!patient.maritalStatus,
          hasDeceasedInfo: patient.deceasedBoolean !== undefined || !!patient.deceasedDateTime
        },
        identifiers: {
          count: patient.identifier?.length || 0,
          hasMRN: patient.identifier?.some(id => id.type?.coding?.[0]?.code === 'MR'),
          types: patient.identifier?.map(id => id.type?.coding?.[0]?.display || id.type?.text || 'Unknown')
        },
        contact: {
          telecomCount: patient.telecom?.length || 0,
          hasPhone: patient.telecom?.some(t => t.system === 'phone'),
          hasEmail: patient.telecom?.some(t => t.system === 'email'),
          hasMobile: patient.telecom?.some(t => t.system === 'phone' && t.use === 'mobile')
        },
        address: {
          count: patient.address?.length || 0,
          hasStreet: !!patient.address?.[0]?.line?.length,
          hasCity: !!patient.address?.[0]?.city,
          hasState: !!patient.address?.[0]?.state,
          hasPostalCode: !!patient.address?.[0]?.postalCode
        },
        emergencyContacts: {
          count: patient.contact?.length || 0,
          details: patient.contact?.map(c => ({
            hasName: !!c.name,
            hasPhone: c.telecom?.some(t => t.system === 'phone'),
            hasRelationship: !!c.relationship?.length
          }))
        },
        careTeam: {
          hasGeneralPractitioner: !!patient.generalPractitioner?.length,
          hasManagingOrg: !!patient.managingOrganization
        },
        communication: {
          count: patient.communication?.length || 0,
          languages: patient.communication?.map(c => c.language?.text || c.language?.coding?.[0]?.display)
        },
        extensions: {
          count: patient.extension?.length || 0
        }
      }
    }, { 
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('[Test Patient Data] Error:', error);
    return NextResponse.json(
      {
        error: 'Failed to fetch patient data',
        details: error instanceof Error ? error.message : 'Unknown error',
        stack: error instanceof Error ? error.stack : undefined
      },
      { status: 500 }
    );
  }
}
