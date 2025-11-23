/**
 * Utility functions for extracting and formatting FHIR Patient data from Medplum
 * Based on FHIR R4 Patient Resource specification
 */

import { Patient } from '@medplum/fhirtypes';

export interface EnhancedPatientData {
  // Core Identity
  id: string;
  mrn: string;
  
  // Name Information
  fullName: string;
  firstName: string;
  middleName?: string;
  lastName: string;
  prefix?: string; // Dr., Mr., Mrs., etc.
  suffix?: string; // Jr., Sr., III, etc.
  nickname?: string;
  
  // Demographics
  dateOfBirth: string;
  age?: number;
  gender?: 'male' | 'female' | 'other' | 'unknown';
  maritalStatus?: string;
  
  // Contact Information
  primaryPhone?: string;
  mobilePhone?: string;
  homePhone?: string;
  workPhone?: string;
  email?: string;
  
  // Address Information
  address?: {
    street: string;
    city: string;
    state: string;
    postalCode: string;
    country?: string;
    full: string; // Full formatted address
  };
  
  // Emergency Contacts
  emergencyContacts: Array<{
    name: string;
    relationship?: string;
    phone?: string;
    email?: string;
  }>;
  
  // Medical Information
  preferredLanguage?: string;
  communicationPreferences?: string[];
  
  // Care Team
  primaryCareProvider?: string;
  managingOrganization?: string;
  
  // Status
  active: boolean;
  deceased: boolean;
  
  // Additional Identifiers
  otherIdentifiers: Array<{
    type: string;
    value: string;
  }>;
}

/**
 * Calculate age from birthdate
 */
function calculateAge(birthDate: string): number | undefined {
  if (!birthDate) return undefined;
  
  const today = new Date();
  const birth = new Date(birthDate);
  let age = today.getFullYear() - birth.getFullYear();
  const monthDiff = today.getMonth() - birth.getMonth();
  
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
    age--;
  }
  
  return age;
}

/**
 * Extract comprehensive patient data from FHIR Patient resource
 */
export function extractPatientData(patient: Patient): EnhancedPatientData {
  // Extract name components
  const primaryName = patient.name?.[0] || {};
  const firstName = primaryName.given?.[0] || '';
  const middleName = primaryName.given?.[1];
  const lastName = primaryName.family || '';
  const fullName = primaryName.text || `${firstName} ${middleName || ''} ${lastName}`.trim();
  const prefix = primaryName.prefix?.[0];
  const suffix = primaryName.suffix?.[0];
  
  // Find nickname if exists
  const nicknameName = patient.name?.find(n => n.use === 'nickname');
  const nickname = nicknameName?.given?.[0];
  
  // Extract MRN (Medical Record Number)
  const mrn = patient.identifier?.find(
    id => id.type?.coding?.[0]?.code === 'MR' || id.type?.text?.toLowerCase().includes('medical record')
  )?.value || 'N/A';
  
  // Extract other identifiers
  const otherIdentifiers = (patient.identifier || [])
    .filter(id => id.type?.coding?.[0]?.code !== 'MR')
    .map(id => ({
      type: id.type?.coding?.[0]?.display || id.type?.text || 'Unknown',
      value: id.value || ''
    }));
  
  // Extract phone numbers by type
  const phones = patient.telecom?.filter(t => t.system === 'phone') || [];
  const primaryPhone = phones.find(p => !p.use || p.use === 'home')?.value;
  const mobilePhone = phones.find(p => p.use === 'mobile')?.value;
  const homePhone = phones.find(p => p.use === 'home')?.value;
  const workPhone = phones.find(p => p.use === 'work')?.value;
  
  // Extract email
  const email = patient.telecom?.find(t => t.system === 'email')?.value;
  
  // Extract primary address
  const primaryAddress = patient.address?.[0];
  let address;
  if (primaryAddress) {
    const street = primaryAddress.line?.join(', ') || '';
    const city = primaryAddress.city || '';
    const state = primaryAddress.state || '';
    const postalCode = primaryAddress.postalCode || '';
    const country = primaryAddress.country;
    const full = primaryAddress.text || `${street}, ${city}, ${state} ${postalCode}`.trim();
    
    address = { street, city, state, postalCode, country, full };
  }
  
  // Extract emergency contacts
  const emergencyContacts = (patient.contact || []).map(contact => ({
    name: contact.name?.text || `${contact.name?.given?.[0] || ''} ${contact.name?.family || ''}`.trim(),
    relationship: contact.relationship?.[0]?.text || contact.relationship?.[0]?.coding?.[0]?.display,
    phone: contact.telecom?.find(t => t.system === 'phone')?.value,
    email: contact.telecom?.find(t => t.system === 'email')?.value
  }));
  
  // Extract preferred language
  const preferredCommunication = patient.communication?.find(c => c.preferred);
  const preferredLanguage = preferredCommunication?.language?.text || 
    preferredCommunication?.language?.coding?.[0]?.display;
  
  // Extract all communication languages
  const communicationPreferences = patient.communication?.map(
    c => c.language?.text || c.language?.coding?.[0]?.display || ''
  ).filter(Boolean);
  
  // Extract marital status
  const maritalStatus = patient.maritalStatus?.text || 
    patient.maritalStatus?.coding?.[0]?.display;
  
  // Extract care providers
  const primaryCareProvider = patient.generalPractitioner?.[0]?.display;
  const managingOrganization = patient.managingOrganization?.display;
  
  // Calculate age
  const age = calculateAge(patient.birthDate || '');
  
  return {
    id: patient.id || '',
    mrn,
    fullName,
    firstName,
    middleName,
    lastName,
    prefix,
    suffix,
    nickname,
    dateOfBirth: patient.birthDate || 'Unknown',
    age,
    gender: patient.gender,
    maritalStatus,
    primaryPhone,
    mobilePhone,
    homePhone,
    workPhone,
    email,
    address,
    emergencyContacts,
    preferredLanguage,
    communicationPreferences,
    primaryCareProvider,
    managingOrganization,
    active: patient.active !== false,
    deceased: patient.deceasedBoolean === true || !!patient.deceasedDateTime,
    otherIdentifiers
  };
}

/**
 * Format patient data for display in questionnaires
 */
export function formatPatientForQuestionnaire(patientData: EnhancedPatientData) {
  return {
    patient_name: patientData.fullName,
    patient_first_name: patientData.firstName,
    patient_last_name: patientData.lastName,
    date_of_birth: patientData.dateOfBirth,
    age: patientData.age?.toString(),
    gender: patientData.gender,
    mrn: patientData.mrn,
    phone: patientData.primaryPhone || patientData.mobilePhone,
    email: patientData.email,
    address: patientData.address?.full,
    emergency_contact_name: patientData.emergencyContacts[0]?.name,
    emergency_contact_phone: patientData.emergencyContacts[0]?.phone,
  };
}
