import { NextRequest, NextResponse } from "next/server";
import { getMedplumClient } from "@/lib/medplum-client";
import { Patient, Address } from "@medplum/fhirtypes";
import { extractPatientData } from "@/lib/patient-utils";

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

const findPreferredPharmacy = (patient: Patient) => {
  const matchesPharmacy = (text?: string) =>
    !!text && text.toLowerCase().includes('pharm');

  const contact = patient.contact?.find((c) => {
    const relationshipMatches = c.relationship?.some((relationship) =>
      relationship.coding?.some(
        (coding) =>
          coding.code?.toLowerCase().includes('pharm') ||
          coding.display?.toLowerCase().includes('pharm')
      )
    );

    return (
      relationshipMatches ||
      matchesPharmacy(c.name?.text) ||
      matchesPharmacy(c.organization?.display)
    );
  });

  if (contact) {
    if (contact.organization?.display) {
      return contact.organization.display;
    }

    if (contact.name) {
      const nameParts = [
        contact.name.text,
        contact.name.given?.join(' '),
        contact.name.family,
      ]
        .filter(Boolean)
        .join(' ')
        .trim();

      if (nameParts) {
        return nameParts;
      }
    }
  }

  return patient.generalPractitioner?.[0]?.display || '';
};

interface PatientSelection {
  patientAddress?: string;
  preferredPharmacy?: string;
  generalPractitioner?: string;
  organizationAddress?: string;
}

export async function POST(request: NextRequest) {
  try {
    const { patientId, patientSelection } = await request.json();

    if (!patientId) {
      return NextResponse.json(
        { error: "Patient ID is required" },
        { status: 400 }
      );
    }

    // Fetch real patient from Medplum with ALL fields
    const medplum = await getMedplumClient();
    const patient = await medplum.readResource("Patient", patientId);

    // Extract comprehensive patient data using utility
    const patientData = extractPatientData(patient);
    
    // Extract pharmacy and GP from selection or FHIR data
    const patientAddress = patientData.address?.full || patientSelection?.patientAddress || "Not provided";
    const preferredPharmacy = findPreferredPharmacy(patient) || patientSelection?.preferredPharmacy || "Not specified";
    const generalPractitioner = patientSelection?.generalPractitioner || patient.generalPractitioner?.[0]?.display || "Not specified";
    const organizationAddress = patientSelection?.organizationAddress || "";
    
    console.log('[Session Start] Extracted patient data:', {
      id: patientData.id,
      name: patientData.fullName,
      hasPhone: !!patientData.primaryPhone,
      hasEmail: !!patientData.email,
      hasAddress: !!patientData.address,
      emergencyContacts: patientData.emergencyContacts.length
    });

    // Build patient summary for session context
    const patientSummary = {
      id: patientData.id,
      name: patientData.fullName,
      mrn: patientData.mrn,
      dob: patientData.dateOfBirth,
      keyProblems: "Loading from medical history...",
      currentMeds: "Loading from medication list...",
      allergies: [] as string[],
      preferredPharmacy,
      address: patientAddress,
      generalPractitioner,
      organizationAddress,
      insurance: "Not specified",
      // Include additional fields for questionnaire autofill
      gender: patientData.gender,
      age: patientData.age,
      phone: patientData.primaryPhone || patientData.mobilePhone,
      email: patientData.email,
      emergencyContactName: patientData.emergencyContacts[0]?.name,
      emergencyContactPhone: patientData.emergencyContacts[0]?.phone,
    };

    // Build detailed history summary
    const historySummary = [
      `Patient: ${patientData.fullName}`,
      patientData.age ? `Age: ${patientData.age} years` : '',
      `Date of Birth: ${patientData.dateOfBirth}`,
      `MRN: ${patientData.mrn}`,
      patientData.gender ? `Gender: ${patientData.gender}` : '',
      patientData.maritalStatus ? `Marital Status: ${patientData.maritalStatus}` : '',
      '',
      patientData.primaryPhone ? `Phone: ${patientData.primaryPhone}` : '',
      patientData.email ? `Email: ${patientData.email}` : '',
      patientData.address ? `Address: ${patientData.address.full}` : '',
      '',
      patientData.emergencyContacts.length > 0 ? `Emergency Contact: ${patientData.emergencyContacts[0].name}${patientData.emergencyContacts[0].phone ? ` (${patientData.emergencyContacts[0].phone})` : ''}` : '',
      '',
      'Note: Full medical history available in EMR'
    ].filter(Boolean).join('\n');

    return NextResponse.json({
      sessionId: `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      patient: patientSummary,
      patientData: patientData, // Full patient data for questionnaire autofill
      historySummary,
    });
  } catch (error) {
    console.error("[Session Start] Error:", error);
    return NextResponse.json(
      { 
        error: "Failed to start session",
        details: error instanceof Error ? error.message : "Unknown error"
      },
      { status: 500 }
    );
  }
}
