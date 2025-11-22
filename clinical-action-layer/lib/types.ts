export type ActionType = 'medication' | 'lab' | 'imaging' | 'referral' | 'followup';

export type ActionStatus = 'pending' | 'approved' | 'rejected';

export interface ClinicalAction {
  id: string;
  type: ActionType;
  title: string;
  description: string;
  status: ActionStatus;
  data: MedicationActionData | LabActionData | ImagingActionData | ReferralActionData | FollowUpActionData;
  warnings?: string[];
  errors?: string[];
}

export interface MedicationActionData {
  medicationName: string;
  dosage: string;
  frequency: string;
  duration: string;
  route: string;
  rxNormCode?: string;
  indication?: string;
}

export interface LabActionData {
  testName: string;
  loincCode?: string;
  priority: 'routine' | 'urgent' | 'stat';
  clinicalNote?: string;
}

export interface ImagingActionData {
  procedureName: string;
  bodyPart: string;
  priority: 'routine' | 'urgent' | 'stat';
  clinicalIndication: string;
  loincCode?: string;
}

export interface ReferralActionData {
  specialty: string;
  reason: string;
  priority: 'routine' | 'urgent';
}

export interface FollowUpActionData {
  timeframe: string;
  reason: string;
  appointmentType: string;
}

// Heidi API Types
export interface HeidiTranscript {
  visitId: string;
  transcript: string;
  patientName?: string;
  visitDate?: string;
  duration?: number;
  metadata?: Record<string, any>;
}

export interface HeidiApiResponse {
  success: boolean;
  data?: HeidiTranscript;
  error?: string;
  fetchedAt?: string;
}
