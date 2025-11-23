/**
 * Maps FHIR resources to Questionnaire responses for autofilling forms
 */

import { ServiceRequest, MedicationRequest } from '@medplum/fhirtypes';

/**
 * Extract values from a FHIR resource and map to questionnaire linkIds
 * This uses intelligent field matching based on common questionnaire patterns
 */
export function mapFHIRToQuestionnaireResponses(
  resource: any,
  questionnaire: any,
  patientData?: any
): Record<string, any> {
  console.log('[Mapper] Starting mapping...');
  console.log('[Mapper] Resource:', resource);
  console.log('[Mapper] Patient data:', patientData);
  console.log('[Mapper] Questionnaire items:', questionnaire?.item);
  
  const responses: Record<string, any> = {};

  if (!questionnaire?.item || !resource) {
    console.log('[Mapper] Missing questionnaire items or resource');
    return responses;
  }

  // Build a mapping of question text/linkId patterns to values from the FHIR resource
  const fieldMappings = extractFieldMappings(resource, patientData);
  console.log('[Mapper] Field mappings extracted:', fieldMappings);

  // Iterate through questionnaire items and try to match them
  for (const item of questionnaire.item) {
    if (!item.linkId) continue;

    const linkId = item.linkId;
    const questionText = (item.text || '').toLowerCase();
    console.log(`[Mapper] Processing item - linkId: ${linkId}, text: "${questionText}"`);

    // Try to find a matching value from the FHIR resource
    const value = findMatchingValue(linkId, questionText, fieldMappings, item);
    
    if (value !== null && value !== undefined) {
      console.log(`[Mapper] Matched value for ${linkId}:`, value);
      responses[linkId] = value;
    } else {
      console.log(`[Mapper] No match found for ${linkId}`);
    }
  }

  console.log('[Mapper] Final responses:', responses);
  return responses;
}

/**
 * Extract common fields from FHIR resource into a searchable structure
 */
function extractFieldMappings(resource: any, patientData?: any): Record<string, any> {
  console.log('[Mapper] Extracting field mappings from resource type:', resource.resourceType);
  const mappings: Record<string, any> = {};

  // Patient demographic data
  if (patientData) {
    if (patientData.name) mappings['patient_name'] = patientData.name;
    if (patientData.dob) mappings['date_of_birth'] = patientData.dob;
    if (patientData.mrn) mappings['mrn'] = patientData.mrn;
    if (patientData.id) mappings['patient_id'] = patientData.id;
  }

  // ServiceRequest fields (lab, imaging)
  if (resource.resourceType === 'ServiceRequest') {
    const sr = resource as ServiceRequest;
    
    // Test/Study name
    if (sr.code?.text) mappings['test_name'] = sr.code.text;
    if (sr.code?.text) mappings['study_name'] = sr.code.text;
    if (sr.code?.text) mappings['exam_type'] = sr.code.text;
    
    // Priority
    if (sr.priority) mappings['priority'] = sr.priority;
    
    // Body site
    if (sr.bodySite?.[0]?.text) mappings['body_site'] = sr.bodySite[0].text;
    if (sr.bodySite?.[0]?.text) mappings['region'] = sr.bodySite[0].text;
    if (sr.bodySite?.[0]?.text) mappings['area'] = sr.bodySite[0].text;
    
    // Clinical indication
    if (sr.reasonCode?.[0]?.text) mappings['indication'] = sr.reasonCode[0].text;
    if (sr.reasonCode?.[0]?.text) mappings['reason'] = sr.reasonCode[0].text;
    if (sr.reasonCode?.[0]?.text) mappings['clinical_indication'] = sr.reasonCode[0].text;
    
    // Category
    if (sr.category?.[0]?.text) mappings['category'] = sr.category[0].text;
    
    // Patient instructions
    if (sr.patientInstruction) mappings['instructions'] = sr.patientInstruction;
    
    // Notes
    if (sr.note?.[0]?.text) mappings['notes'] = sr.note[0].text;
    if (sr.note?.[0]?.text) mappings['comments'] = sr.note[0].text;
  }

  // MedicationRequest fields
  if (resource.resourceType === 'MedicationRequest') {
    const mr = resource as MedicationRequest;
    
    // Medication name
    if (mr.medicationCodeableConcept?.text) {
      mappings['medication'] = mr.medicationCodeableConcept.text;
      mappings['medication_name'] = mr.medicationCodeableConcept.text;
    }
    
    // Dosage
    if (mr.dosageInstruction?.[0]?.text) {
      mappings['dosage'] = mr.dosageInstruction[0].text;
      mappings['dose'] = mr.dosageInstruction[0].text;
      mappings['instructions'] = mr.dosageInstruction[0].text;
    }
    
    // Route
    if (mr.dosageInstruction?.[0]?.route?.text) {
      mappings['route'] = mr.dosageInstruction[0].route.text;
    }
    
    // Frequency
    if (mr.dosageInstruction?.[0]?.timing?.repeat?.frequency) {
      mappings['frequency'] = mr.dosageInstruction[0].timing.repeat.frequency;
    }
    
    // Clinical indication
    if (mr.reasonCode?.[0]?.text) {
      mappings['indication'] = mr.reasonCode[0].text;
      mappings['reason'] = mr.reasonCode[0].text;
    }
    
    // Pharmacy
    if (mr.dispenseRequest?.performer?.display) {
      mappings['pharmacy'] = mr.dispenseRequest.performer.display;
    }
  }

  return mappings;
}

/**
 * Find a matching value for a questionnaire item from the extracted mappings
 */
function findMatchingValue(
  linkId: string,
  questionText: string,
  mappings: Record<string, any>,
  item: any
): any {
  // Direct linkId match (e.g., linkId might be "priority" and we have mappings['priority'])
  if (mappings[linkId]) {
    return formatValueForType(mappings[linkId], item.type);
  }

  // Match based on question text keywords
  const keywords = [
    { patterns: ['patient name', 'name of patient', 'full name'], keys: ['patient_name'] },
    { patterns: ['date of birth', 'dob', 'birth date', 'birthdate'], keys: ['date_of_birth'] },
    { patterns: ['medical record', 'mrn', 'patient id', 'record number'], keys: ['mrn', 'patient_id'] },
    { patterns: ['test name', 'exam type', 'study', 'procedure', 'imaging', 'scan type'], keys: ['test_name', 'study_name', 'exam_type'] },
    { patterns: ['priority', 'urgency'], keys: ['priority'] },
    { patterns: ['body site', 'region', 'area', 'location', 'anatomical'], keys: ['body_site', 'region', 'area'] },
    { patterns: ['indication', 'reason', 'clinical reason', 'why', 'purpose'], keys: ['indication', 'reason', 'clinical_indication'] },
    { patterns: ['medication', 'drug', 'prescription'], keys: ['medication', 'medication_name'] },
    { patterns: ['dosage', 'dose', 'amount'], keys: ['dosage', 'dose'] },
    { patterns: ['route', 'method of administration'], keys: ['route'] },
    { patterns: ['frequency', 'how often'], keys: ['frequency'] },
    { patterns: ['instructions', 'directions'], keys: ['instructions'] },
    { patterns: ['notes', 'comments', 'additional'], keys: ['notes', 'comments'] },
    { patterns: ['pharmacy'], keys: ['pharmacy'] },
    { patterns: ['contrast', 'dye'], keys: ['contrast'] },
  ];

  for (const { patterns, keys } of keywords) {
    for (const pattern of patterns) {
      if (questionText.includes(pattern)) {
        for (const key of keys) {
          if (mappings[key]) {
            return formatValueForType(mappings[key], item.type);
          }
        }
      }
    }
  }

  // If it's a choice/select field, try to match against answer options
  if (item.type === 'choice' && item.answerOption) {
    // Check if any mapping value matches an answer option
    for (const value of Object.values(mappings)) {
      const matchedOption = item.answerOption.find((opt: any) => {
        const optionDisplay = opt.valueCoding?.display || opt.valueString || '';
        const optionCode = opt.valueCoding?.code || '';
        return (
          optionDisplay.toLowerCase() === String(value).toLowerCase() ||
          optionCode.toLowerCase() === String(value).toLowerCase()
        );
      });
      
      if (matchedOption) {
        return matchedOption.valueCoding?.code || matchedOption.valueString;
      }
    }
  }

  return null;
}

/**
 * Format a value to match the expected type for the questionnaire item
 */
function formatValueForType(value: any, itemType: string): any {
  switch (itemType) {
    case 'boolean':
      if (typeof value === 'boolean') return value;
      if (typeof value === 'string') {
        const lower = value.toLowerCase();
        if (lower === 'yes' || lower === 'true') return true;
        if (lower === 'no' || lower === 'false') return false;
      }
      return null;
      
    case 'integer':
      return typeof value === 'number' ? Math.floor(value) : parseInt(value, 10);
      
    case 'decimal':
      return typeof value === 'number' ? value : parseFloat(value);
      
    case 'date':
      if (value instanceof Date) return value.toISOString().split('T')[0];
      return value;
      
    case 'string':
    case 'text':
    case 'choice':
    default:
      return String(value);
  }
}
