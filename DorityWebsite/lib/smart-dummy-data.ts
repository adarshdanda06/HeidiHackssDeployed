/**
 * Smart dummy data generator for filling missing questionnaire fields
 * Uses context-aware realistic values based on field type and clinical context
 * Prefers real patient data from Medplum when available
 */

import { QuestionnaireItem } from '@medplum/fhirtypes';

interface DummyDataContext {
  // Real patient data from Medplum (preferred)
  patientName?: string;
  patientAge?: number;
  patientDob?: string;
  patientGender?: string;
  patientMrn?: string;
  patientPhone?: string;
  patientEmail?: string;
  patientAddress?: string;
  emergencyContactName?: string;
  emergencyContactPhone?: string;
  preferredPharmacy?: string;
  insurance?: string;
  // Clinical context
  clinicalContext?: string; // e.g., "subarachnoid hemorrhage", "chest pain"
  urgency?: 'stat' | 'urgent' | 'routine';
}

/**
 * Generate smart dummy data for a questionnaire field based on its linkId, type, and context
 * ALWAYS prefers real patient data from Medplum when available
 */
export function generateSmartDummyValue(
  item: QuestionnaireItem,
  context: DummyDataContext = {}
): any {
  const linkId = item.linkId?.toLowerCase() || '';
  const type = item.type;

  // Patient demographic fields - USE REAL MEDPLUM DATA
  if (linkId.includes('patient') && linkId.includes('name')) {
    return context.patientName || 'John Doe';
  }
  
  if (linkId.includes('dob') || linkId.includes('birth')) {
    if (context.patientDob) return context.patientDob;
    const age = context.patientAge || 45;
    const year = new Date().getFullYear() - age;
    return `${year}-06-15`;
  }
  
  if (linkId.includes('age')) {
    return context.patientAge?.toString() || '45';
  }
  
  if (linkId.includes('gender') || linkId.includes('sex')) {
    return context.patientGender || 'Unknown';
  }
  
  if (linkId.includes('mrn') || linkId.includes('medical') && linkId.includes('record')) {
    return context.patientMrn || `MRN${Math.random().toString().slice(2, 9)}`;
  }

  // Contact information - USE REAL MEDPLUM DATA
  if (linkId.includes('phone') || linkId.includes('telephone')) {
    if (linkId.includes('emergency')) {
      return context.emergencyContactPhone || context.patientPhone || '555-0123';
    }
    return context.patientPhone || '555-0123';
  }
  
  if (linkId.includes('email')) {
    return context.patientEmail || 'patient@example.com';
  }
  
  if (linkId.includes('address')) {
    return context.patientAddress || '123 Main Street, Anytown, ST 12345';
  }
  
  if (linkId.includes('emergency') && linkId.includes('contact')) {
    if (linkId.includes('name')) {
      return context.emergencyContactName || 'Emergency Contact';
    }
    if (linkId.includes('phone')) {
      return context.emergencyContactPhone || '555-0124';
    }
  }

  // Clinical fields - Imaging/Procedures
  if (linkId.includes('examtype') || linkId.includes('exam') && linkId.includes('type')) {
    if (context.clinicalContext?.toLowerCase().includes('head') || 
        context.clinicalContext?.toLowerCase().includes('brain')) {
      return 'CT Brain (non-contrast)';
    }
    return 'Standard examination';
  }
  
  if (linkId.includes('bodyregion') || linkId.includes('body') && linkId.includes('region')) {
    if (context.clinicalContext?.toLowerCase().includes('brain') || 
        context.clinicalContext?.toLowerCase().includes('head')) {
      return 'Brain/Head';
    }
    if (context.clinicalContext?.toLowerCase().includes('chest')) {
      return 'Chest';
    }
    return 'See clinical indication';
  }
  
  if (linkId.includes('contrast')) {
    if (type === 'boolean') return false;
    return 'No contrast';
  }
  
  if (linkId.includes('priority')) {
    if (context.urgency === 'stat') return 'STAT';
    if (context.urgency === 'urgent') return 'Urgent';
    return 'Routine';
  }
  
  if (linkId.includes('indication') || linkId.includes('reason') || linkId.includes('diagnosis')) {
    return context.clinicalContext || 'Clinical evaluation required';
  }

  // Safety/Contraindications
  if (linkId.includes('contraindication')) {
    return 'None known';
  }
  
  if (linkId.includes('allerg')) {
    return 'No known allergies';
  }
  
  if (linkId.includes('pregnancy') || linkId.includes('pregnant')) {
    if (type === 'boolean') return false;
    return 'Not pregnant / Not applicable';
  }
  
  if (linkId.includes('implant') || linkId.includes('pacemaker') || 
      linkId.includes('metal') || linkId.includes('claustrophobia')) {
    if (type === 'boolean') return false;
    return 'No';
  }

  // Medication fields
  if (linkId.includes('medication') && linkId.includes('name')) {
    return 'As prescribed';
  }
  
  if (linkId.includes('dosage') || linkId.includes('dose')) {
    return 'Standard dose per protocol';
  }
  
  if (linkId.includes('route')) {
    return 'Oral';
  }
  
  if (linkId.includes('frequency')) {
    return 'As directed';
  }
  
  if (linkId.includes('duration')) {
    return '7 days';
  }
  
  if (linkId.includes('quantity')) {
    if (linkId.includes('refill')) return '0';
    return '30';
  }
  
  if (linkId.includes('refill')) {
    if (type === 'integer' || type === 'decimal') return 0;
    return 'No refills';
  }

  // Provider fields - Generate realistic names
  if (linkId.includes('provider') || linkId.includes('physician') || 
      linkId.includes('prescriber') || linkId.includes('ordering')) {
    // Generate realistic provider names
    const providerNames = [
      'Dr. Sarah Chen, MD',
      'Dr. Michael Johnson, MD',
      'Dr. Emily Rodriguez, MD',
      'Dr. David Kim, MD',
      'Dr. Jennifer Williams, MD',
      'Dr. Robert Thompson, MD'
    ];
    return providerNames[Math.floor(Math.random() * providerNames.length)];
  }
  
  if (linkId.includes('npi')) {
    // Generate realistic NPI number (10 digits)
    return `12345${Math.random().toString().slice(2, 7)}`;
  }
  
  if (linkId.includes('dea')) {
    // Generate realistic DEA number format (2 letters + 7 digits)
    return `AB${Math.random().toString().slice(2, 9)}`;
  }
  
  if (linkId.includes('signature') || linkId.includes('sign')) {
    return 'Electronically signed';
  }

  // Pharmacy fields - USE REAL MEDPLUM DATA
  if (linkId.includes('pharmacy')) {
    if (linkId.includes('preference') || linkId.includes('name')) {
      return context.preferredPharmacy || 'Hospital Pharmacy';
    }
    return context.preferredPharmacy || 'Patient preferred pharmacy';
  }

  // Scheduling/Logistics
  if (linkId.includes('scheduling') || linkId.includes('appointment')) {
    if (context.urgency === 'stat') {
      return 'Immediate - STAT order';
    }
    return 'Schedule at earliest availability';
  }
  
  if (linkId.includes('transport') || linkId.includes('sedation')) {
    if (type === 'boolean') return false;
    return 'Not required';
  }
  
  if (linkId.includes('fasting')) {
    if (type === 'boolean') return false;
    return 'Not required';
  }

  // Instructions/Notes
  if (linkId.includes('instruction') || linkId.includes('note') || linkId.includes('comment')) {
    return 'See clinical indication for details';
  }
  
  if (linkId.includes('special')) {
    return 'None';
  }

  // Insurance/Billing - USE REAL MEDPLUM DATA
  if (linkId.includes('insurance')) {
    return context.insurance || 'Patient insurance on file';
  }

  // Generic fallbacks based on type
  switch (type) {
    case 'boolean':
      return false;
    
    case 'integer':
      return 0;
    
    case 'decimal':
      return 0.0;
    
    case 'date':
      // Use current date for order dates, realistic dates for appointments
      if (linkId.includes('order') || linkId.includes('request')) {
        return new Date().toISOString().split('T')[0];
      }
      if (linkId.includes('appointment') || linkId.includes('schedule')) {
        // Future date (7 days from now)
        const futureDate = new Date();
        futureDate.setDate(futureDate.getDate() + 7);
        return futureDate.toISOString().split('T')[0];
      }
      return new Date().toISOString().split('T')[0];
    
    case 'dateTime':
      if (linkId.includes('appointment') || linkId.includes('schedule')) {
        // Future datetime (7 days from now at 2 PM)
        const futureDate = new Date();
        futureDate.setDate(futureDate.getDate() + 7);
        futureDate.setHours(14, 0, 0, 0);
        return futureDate.toISOString();
      }
      return new Date().toISOString();
    
    case 'time':
      // Realistic medical appointment times
      const times = ['09:00', '10:00', '11:00', '14:00', '15:00', '16:00'];
      return times[Math.floor(Math.random() * times.length)];
    
    case 'choice':
      // CRITICAL: Return the actual code/value that the form expects
      // The form uses: option.valueCoding?.code || option.valueString
      if (item.answerOption && item.answerOption.length > 0) {
        const firstOption = item.answerOption[0];
        
        // Return the CODE (for valueCoding) or STRING (for valueString)
        // This is what the select dropdown uses as the option value
        if (firstOption.valueCoding?.code) {
          return firstOption.valueCoding.code;
        }
        if (firstOption.valueCoding?.display) {
          return firstOption.valueCoding.display;
        }
        if (firstOption.valueString) {
          return firstOption.valueString;
        }
      }
      return ''; // Empty string so form shows "Select an option..."
    
    case 'string':
    case 'text':
    default:
      // Check if it's a required field to give it more attention
      if (item.required) {
        return 'To be completed';
      }
      return '';
  }
}

/**
 * Fill all missing fields in a QuestionnaireResponse with smart dummy data
 */
export function fillMissingFields(
  questionnaireResponse: any,
  questionnaire: any,
  context: DummyDataContext = {}
): any {
  const filledResponse = JSON.parse(JSON.stringify(questionnaireResponse));
  
  // Helper to check if an item already has an answer
  const hasAnswer = (itemId: string, items: any[]): boolean => {
    for (const item of items) {
      if (item.linkId === itemId && item.answer && item.answer.length > 0) {
        return true;
      }
      if (item.item && hasAnswer(itemId, item.item)) {
        return true;
      }
    }
    return false;
  };
  
  // Helper to add missing items recursively
  const fillItems = (questionnaireItems: any[], responseItems: any[], parentGroup?: any) => {
    for (const qItem of questionnaireItems) {
      // Skip display items
      if (qItem.type === 'display') continue;
      
      // Check if this item already has an answer
      const alreadyAnswered = hasAnswer(qItem.linkId, responseItems);
      
      if (!alreadyAnswered && qItem.type !== 'group') {
        // Generate dummy value for this field
        const dummyValue = generateSmartDummyValue(qItem, context);
        
        // Create answer object based on type
        let answer;
        if (qItem.type === 'boolean') {
          answer = [{ valueBoolean: dummyValue }];
        } else if (qItem.type === 'integer') {
          answer = [{ valueInteger: dummyValue }];
        } else if (qItem.type === 'decimal') {
          answer = [{ valueDecimal: dummyValue }];
        } else if (qItem.type === 'date') {
          answer = [{ valueDate: dummyValue }];
        } else if (qItem.type === 'dateTime') {
          answer = [{ valueDateTime: dummyValue }];
        } else if (qItem.type === 'choice' && qItem.answerOption && qItem.answerOption.length > 0) {
          // CRITICAL: For choice fields, create proper answer structure
          const firstOption = qItem.answerOption[0];
          
          if (firstOption.valueCoding) {
            // Use valueCoding if the option has it
            answer = [{ valueCoding: {
              system: firstOption.valueCoding.system,
              code: firstOption.valueCoding.code,
              display: firstOption.valueCoding.display
            }}];
          } else if (firstOption.valueString) {
            // Use valueString if the option has it
            answer = [{ valueString: firstOption.valueString }];
          } else {
            // Fallback to valueString with dummy value
            answer = [{ valueString: dummyValue }];
          }
        } else {
          answer = [{ valueString: dummyValue }];
        }
        
        // Add to response
        const newItem = {
          linkId: qItem.linkId,
          answer
        };
        
        // Find or create the parent group
        if (parentGroup) {
          let parentInResponse = responseItems.find((i: any) => i.linkId === parentGroup.linkId);
          if (!parentInResponse) {
            parentInResponse = {
              linkId: parentGroup.linkId,
              item: []
            };
            responseItems.push(parentInResponse);
          }
          if (!parentInResponse.item) {
            parentInResponse.item = [];
          }
          parentInResponse.item.push(newItem);
        } else {
          responseItems.push(newItem);
        }
      }
      
      // Recursively fill nested items (groups)
      if (qItem.type === 'group' && qItem.item) {
        let groupInResponse = responseItems.find((i: any) => i.linkId === qItem.linkId);
        if (!groupInResponse) {
          groupInResponse = {
            linkId: qItem.linkId,
            item: []
          };
          responseItems.push(groupInResponse);
        }
        if (!groupInResponse.item) {
          groupInResponse.item = [];
        }
        fillItems(qItem.item, groupInResponse.item, qItem);
      }
    }
  };
  
  if (questionnaire.item && filledResponse.item) {
    fillItems(questionnaire.item, filledResponse.item);
  }
  
  return filledResponse;
}
