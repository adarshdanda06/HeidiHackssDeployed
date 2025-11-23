import { NextRequest, NextResponse } from 'next/server';
import { getMedplumClient } from '@/lib/medplum-client';
import { Patient, Organization, Address } from '@medplum/fhirtypes';

interface SimplifiedPatient {
  patientId: string;
  patientFirstName: string | undefined;
  patientLastName: string | undefined;
  generalPractitioner: string | undefined;
  patientAddress: string;
  preferredPharmacy: string;
  organizationAddress?: string;
}

const formatAddress = (address?: Address) => {
  if (!address) {
    return '';
  }

  return [
    address.line?.[0],
    address.city,
    address.state,
    address.postalCode,
    address.country,
  ]
    .filter(Boolean)
    .join(', ');
};

export async function GET(request: NextRequest) {
  try {
    console.log('[Patients] Fetching patients from Medplum...');

    const medplum = await getMedplumClient();

    const patients = await medplum.searchResources('Patient', {
      _count: '100',
    });

    console.log(`[Patients] Found ${patients.length} patients`);

    const organizations = await medplum.searchResources('Organization', {
      _count: '100',
    });

    const orgAddressMap = organizations.reduce<Record<string, string>>((acc, org: Organization) => {
      if (!org.id) {
        return acc;
      }
      const address = formatAddress(org.address?.[0]);
      if (address) {
        acc[`Organization/${org.id}`] = address;
      }
      return acc;
    }, {});

    const orgNameMap = organizations.reduce<Record<string, string>>((acc, org: Organization) => {
      if (org.id && org.name) {
        acc[`Organization/${org.id}`] = org.name;
      }
      return acc;
    }, {});

    const simplifiedPatients: SimplifiedPatient[] = patients.map((patient: Patient) => {
      const gpReference = patient.generalPractitioner?.[0];
      const referenceId = gpReference?.reference;
      const generalPractitioner =
        gpReference?.display || (referenceId ? orgNameMap[referenceId] : undefined);

      const organizationAddress =
        (referenceId && orgAddressMap[referenceId]) ||
        (generalPractitioner ? orgAddressMap[generalPractitioner] : undefined) ||
        '';

      return {
        patientId: patient.id || '',
        patientFirstName: patient.name?.[0]?.given?.[0],
        patientLastName: patient.name?.[0]?.family,
        generalPractitioner,
        patientAddress: formatAddress(patient.address?.[0]),
        preferredPharmacy: generalPractitioner || '',
        organizationAddress,
      };
    });

    return NextResponse.json({
      patients: simplifiedPatients,
    });
  } catch (error) {
    console.error('[Patients] Error fetching patients:', error);

    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Failed to fetch patients',
        details: error instanceof Error ? error.stack : undefined,
      },
      { status: 500 }
    );
  }
}
