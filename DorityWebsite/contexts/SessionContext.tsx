"use client";

import { createContext, useContext, useState, useCallback, ReactNode } from "react";

// TypeScript interfaces
export interface PatientSelection {
  patientAddress?: string;
  preferredPharmacy?: string;
  generalPractitioner?: string;
  organizationAddress?: string;
  heidiSessionId?: string;
}

export interface PatientSummary {
  id: string;
  name: string;
  mrn: string;
  dob: string;
  keyProblems: string;
  currentMeds: string;
  allergies: string[];
  preferredPharmacy: string;
  insurance: string;
  address: string;
  generalPractitioner: string;
  organizationAddress: string;
  heidiSessionId?: string;
  // Additional fields for questionnaire autofill
  gender?: string;
  age?: number;
  phone?: string;
  email?: string;
  emergencyContactName?: string;
  emergencyContactPhone?: string;
}

export interface SuggestedAction {
  id: string;
  type: "medication" | "imaging" | "lab" | "referral" | "followup" | "aftercare" | "scheduling";
  status: "pending" | "approved" | "rejected";
  title: string;
  categoryLabel: string;
  details: string;
  doseInfo?: string;
  pharmacy?: string;
  safetyFlag?: "high" | "medium" | "low" | null;
  safetyMessage?: string;
  rationale: string;
  questionnaireId?: string; // Medplum Questionnaire ID
  questionnaireName?: string; // Human-readable questionnaire name
  email?: string; // For scheduling actions
  when?: string; // For scheduling actions
  reason?: string; // For scheduling actions
  subject?: string; // For scheduling actions
  body?: string; // For scheduling actions
  fhirPreview: {
    resourceType: string;
    status: string;
    [key: string]: unknown;
  };
}

interface SessionState {
  currentStep: 1 | 2 | 3 | 4;
  sessionId: string | null;
  patient: PatientSummary | null;
  historySummary: string;
  transcript: string;
  suggestedActions: SuggestedAction[];
  approvedActions: SuggestedAction[];
  aftercareSummary: string;
  aftercareSubject: string;
  isLoading: {
    startSession: boolean;
    analyze: boolean;
    apply: boolean;
    aftercare: boolean;
  };
  error: string | null;
}

interface SessionContextType extends SessionState {
  setCurrentStep: (step: 1 | 2 | 3 | 4) => void;
  setTranscript: (transcript: string) => void;
  setAftercareSummary: (summary: string) => void;
  setAftercareSubject: (subject: string) => void;
  startSession: (patientId: string, patientSelection?: PatientSelection) => Promise<void>;
  analyzeTranscript: () => Promise<void>;
  updateActionStatus: (actionId: string, status: "pending" | "approved" | "rejected") => void;
  updateAction: (actionId: string, updates: Partial<SuggestedAction>) => void;
  applyApprovedActions: () => Promise<void>;
  generateAftercare: () => Promise<void>;
  sendAftercare: (email: string) => Promise<void>;
  clearError: () => void;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);

const initialState: SessionState = {
  currentStep: 1,
  sessionId: null,
  patient: null,
  historySummary: "",
  transcript: "",
  suggestedActions: [],
  approvedActions: [],
  aftercareSummary: "",
  aftercareSubject: "",
  isLoading: {
    startSession: false,
    analyze: false,
    apply: false,
    aftercare: false,
  },
  error: null,
};

export function SessionProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<SessionState>(initialState);

  const setCurrentStep = useCallback((step: 1 | 2 | 3 | 4) => {
    setState((prev) => ({ ...prev, currentStep: step }));
  }, []);

  const setTranscript = useCallback((transcript: string) => {
    setState((prev) => ({ ...prev, transcript }));
  }, []);

  const setAftercareSummary = useCallback((summary: string) => {
    setState((prev) => ({ ...prev, aftercareSummary: summary }));
  }, []);

  const setAftercareSubject = useCallback((subject: string) => {
    setState((prev) => ({ ...prev, aftercareSubject: subject }));
  }, []);

  const clearError = useCallback(() => {
    setState((prev) => ({ ...prev, error: null }));
  }, []);

  const startSession = useCallback(async (patientId: string, patientSelection?: PatientSelection) => {
    setState((prev) => ({
      ...prev,
      isLoading: { ...prev.isLoading, startSession: true },
      error: null,
    }));

    try {
      const response = await fetch("/api/session/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patientId, patientSelection }),
      });

      if (!response.ok) {
        throw new Error(`Failed to start session: ${response.statusText}`);
      }

      const data = await response.json();

      setState((prev) => ({
        ...prev,
        sessionId: data.sessionId,
        patient: data.patient,
        historySummary: data.historySummary || "",
        currentStep: 2,
        isLoading: { ...prev.isLoading, startSession: false },
      }));
    } catch (error) {
      setState((prev) => ({
        ...prev,
        error: error instanceof Error ? error.message : "Failed to start session",
        isLoading: { ...prev.isLoading, startSession: false },
      }));
    }
  }, []);

  const analyzeTranscript = useCallback(async () => {
    if (!state.sessionId) {
      setState((prev) => ({ ...prev, error: "No active session" }));
      return;
    }

    setState((prev) => ({
      ...prev,
      isLoading: { ...prev.isLoading, analyze: true },
      error: null,
    }));

    try {
      // Call real Claude API endpoint
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript: state.transcript,
          patientContext: state.patient ? `Patient: ${state.patient.name}` : undefined,
          patient: state.patient, // Include full patient data for autofill
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to analyze transcript: ${response.statusText}`);
      }

      const data = await response.json();

      // Transform backend actions to frontend format
      const transformedActions: SuggestedAction[] = data.actions.map((action: any, index: number) => ({
        id: `action-${Date.now()}-${index}`,
        type: action.type,
        status: "pending" as const,
        title: action.description,
        categoryLabel: action.type.charAt(0).toUpperCase() + action.type.slice(1),
        details: action.description,
        rationale: `Extracted from transcript by AI`,
        questionnaireId: action.questionnaireId,
        questionnaireName: action.questionnaireName,
        fhirPreview: action.resource,
        email: action.email,
        when: action.when,
        reason: action.reason,
        subject: action.subject,
        body: action.body,
      }));

      setState((prev) => ({
        ...prev,
        suggestedActions: transformedActions,
        currentStep: 3,
        isLoading: { ...prev.isLoading, analyze: false },
      }));
    } catch (error) {
      setState((prev) => ({
        ...prev,
        error: error instanceof Error ? error.message : "Failed to analyze transcript",
        isLoading: { ...prev.isLoading, analyze: false },
      }));
    }
  }, [state.sessionId, state.transcript, state.patient]);

  const updateActionStatus = useCallback(
    (actionId: string, status: "pending" | "approved" | "rejected") => {
      setState((prev) => {
        const updatedActions = prev.suggestedActions.map((action) =>
          action.id === actionId ? { ...action, status } : action
        );

        const approvedActions = updatedActions.filter((a) => a.status === "approved");

        return {
          ...prev,
          suggestedActions: updatedActions,
          approvedActions,
        };
      });
    },
    []
  );

  const updateAction = useCallback(
    (actionId: string, updates: Partial<SuggestedAction>) => {
      setState((prev) => {
        const updatedActions = prev.suggestedActions.map((action) =>
          action.id === actionId ? { ...action, ...updates } : action
        );

        const approvedActions = updatedActions.filter((a) => a.status === "approved");

        return {
          ...prev,
          suggestedActions: updatedActions,
          approvedActions,
        };
      });
    },
    []
  );

  const applyApprovedActions = useCallback(async () => {
    if (!state.sessionId) {
      setState((prev) => ({ ...prev, error: "No active session" }));
      return;
    }

    if (state.approvedActions.length === 0) {
      setState((prev) => ({ ...prev, error: "No approved actions to apply" }));
      return;
    }

    if (!state.patient?.id) {
      setState((prev) => ({ ...prev, error: "No patient selected" }));
      return;
    }

    setState((prev) => ({
      ...prev,
      isLoading: { ...prev.isLoading, apply: true },
      error: null,
    }));

    try {
      // Execute each approved action to Medplum
      const results = [];
      const errors = [];

      for (const action of state.approvedActions) {
        try {
          const response = await fetch("/api/execute", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              action: {
                type: action.type,
                description: action.title,
                resource: action.fhirPreview,
              },
              patientId: state.patient.id,
            }),
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Failed to execute ${action.type}`);
          }

          const result = await response.json();
          results.push({ action, result });
        } catch (err) {
          errors.push({ action, error: err instanceof Error ? err.message : 'Unknown error' });
        }
      }

      if (errors.length > 0) {
        throw new Error(`Failed to apply ${errors.length} action(s): ${errors.map(e => e.error).join(', ')}`);
      }

      setState((prev) => ({
        ...prev,
        currentStep: 4,
        isLoading: { ...prev.isLoading, apply: false },
      }));
    } catch (error) {
      setState((prev) => ({
        ...prev,
        error: error instanceof Error ? error.message : "Failed to apply actions",
        isLoading: { ...prev.isLoading, apply: false },
      }));
    }
  }, [state.sessionId, state.approvedActions, state.patient]);

  const generateAftercare = useCallback(async () => {
    if (!state.sessionId) {
      setState((prev) => ({ ...prev, error: "No active session" }));
      return;
    }

    setState((prev) => ({
      ...prev,
      isLoading: { ...prev.isLoading, aftercare: true },
      error: null,
    }));

    try {
      const response = await fetch("/api/session/aftercare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId: state.sessionId,
          approvedActions: state.approvedActions,
          historySummary: state.historySummary,
          transcript: state.transcript,
          patient: state.patient,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to generate aftercare: ${response.statusText}`);
      }

      const data = await response.json();

      setState((prev) => ({
        ...prev,
        aftercareSummary: data.body || data.aftercareSummary || "",
        aftercareSubject: data.subject || "Visit Summary",
        isLoading: { ...prev.isLoading, aftercare: false },
      }));
    } catch (error) {
      setState((prev) => ({
        ...prev,
        error: error instanceof Error ? error.message : "Failed to generate aftercare",
        isLoading: { ...prev.isLoading, aftercare: false },
      }));
    }
  }, [state.sessionId, state.approvedActions, state.historySummary, state.transcript]);

  const sendAftercare = useCallback(
    async (email: string) => {
      if (!state.sessionId) {
        setState((prev) => ({ ...prev, error: "No active session" }));
        return;
      }

      setState((prev) => ({
        ...prev,
        isLoading: { ...prev.isLoading, aftercare: true },
        error: null,
      }));

      try {
        const response = await fetch("/api/session/send-aftercare", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sessionId: state.sessionId,
            email,
            summary: state.aftercareSummary,
            subject: state.aftercareSubject,
          }),
        });

        if (!response.ok) {
          throw new Error(`Failed to send aftercare email: ${response.statusText}`);
        }

        setState((prev) => ({
          ...prev,
          isLoading: { ...prev.isLoading, aftercare: false },
        }));
      } catch (error) {
        setState((prev) => ({
          ...prev,
          error: error instanceof Error ? error.message : "Failed to send aftercare email",
          isLoading: { ...prev.isLoading, aftercare: false },
        }));
      }
    },
    [state.sessionId, state.aftercareSummary]
  );

  const value: SessionContextType = {
    ...state,
    setCurrentStep,
    setTranscript,
    setAftercareSummary,
    setAftercareSubject,
    startSession,
    analyzeTranscript,
    updateActionStatus,
    updateAction, // Ensure this is exported in the value object
    applyApprovedActions,
    generateAftercare,
    sendAftercare,
    clearError,
  };

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession() {
  const context = useContext(SessionContext);
  if (!context) {
    throw new Error("useSession must be used within a SessionProvider");
  }
  return context;
}
