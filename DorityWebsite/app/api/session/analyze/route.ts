import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";

export async function POST(request: NextRequest) {
  try {
    const { sessionId, transcript } = await request.json();

    // Simulate AI processing delay
    await new Promise((resolve) => setTimeout(resolve, 2500));

    const suggestedActions = [];
    const transcriptLower = transcript.toLowerCase();

    // Anthropic Analysis for Scheduling
    if (process.env.ANTHROPIC_API_KEY) {
      try {
        const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
        const msg = await anthropic.messages.create({
          model: "claude-3-5-sonnet-20241022",
          max_tokens: 1024,
          messages: [
            { 
              role: "user", 
              content: `Analyze this medical transcript and determine if a follow-up meeting, appointment, or consultation needs to be scheduled. 
              If yes, extract the specific reason ("why") and the suggested time/date ("when").
              
              Return ONLY a valid JSON object with the following keys:
              - "needsMeeting": boolean
              - "reason": string (concise explanation)
              - "when": string (suggested timeframe)
              
              Do not include any markdown formatting or extra text, just the JSON object.
              
              Transcript:
              ${transcript}` 
            }
          ]
        });

        const content = msg.content[0].type === 'text' ? msg.content[0].text : '';
        console.log("Claude Analysis:", content);
        
        const analysis = JSON.parse(content.trim());

        if (analysis.needsMeeting) {
          suggestedActions.push({
            id: `action-${Date.now()}-schedule`,
            type: "scheduling",
            status: "pending",
            title: "Schedule Follow-up Meeting",
            categoryLabel: "Scheduling",
            details: `Schedule a meeting. Reason: ${analysis.reason}`,
            when: analysis.when,
            reason: analysis.reason,
            safetyFlag: "medium",
            rationale: analysis.reason,
            fhirPreview: {
              resourceType: "Appointment",
              status: "proposed",
              description: analysis.reason,
              start: analysis.when 
            },
          });
        }
      } catch (err) {
        console.error("Anthropic analysis failed:", err);
      }
    }

    // Medication suggestions
    if (transcriptLower.includes("metformin") || transcriptLower.includes("diabetes") || transcriptLower.includes("blood sugar")) {
      suggestedActions.push({
        id: `action-${Date.now()}-1`,
        type: "medication",
        status: "pending",
        title: "Metformin 500mg",
        categoryLabel: "Medications",
        details: "Take 1 tablet twice daily with meals for blood glucose control",
        doseInfo: "500mg PO BID",
        pharmacy: "CVS Pharmacy - Main Street",
        safetyFlag: "low",
        safetyMessage: "No significant drug interactions detected",
        rationale: "Patient reports elevated fasting glucose readings; current dose suboptimal",
        fhirPreview: {
          resourceType: "MedicationRequest",
          status: "draft",
          intent: "order",
          medicationCodeableConcept: { text: "Metformin 500mg tablet" },
        },
      });
    }

    if (transcriptLower.includes("lisinopril") || transcriptLower.includes("blood pressure") || transcriptLower.includes("hypertension")) {
      suggestedActions.push({
        id: `action-${Date.now()}-2`,
        type: "medication",
        status: "pending",
        title: "Lisinopril 10mg",
        categoryLabel: "Medications",
        details: "Take 1 tablet once daily in the morning for blood pressure control",
        doseInfo: "10mg PO QD",
        pharmacy: "CVS Pharmacy - Main Street",
        safetyFlag: "medium",
        safetyMessage: "Monitor potassium levels; patient has CKD Stage 3",
        rationale: "BP readings averaging 145/90 mmHg over past 2 weeks; target <130/80",
        fhirPreview: {
          resourceType: "MedicationRequest",
          status: "draft",
          intent: "order",
          medicationCodeableConcept: { text: "Lisinopril 10mg tablet" },
        },
      });
    }

    // Lab/Imaging suggestions
    if (transcriptLower.includes("cbc") || transcriptLower.includes("blood work") || transcriptLower.includes("lab") || transcriptLower.includes("test")) {
      suggestedActions.push({
        id: `action-${Date.now()}-3`,
        type: "lab",
        status: "pending",
        title: "Complete Blood Count (CBC)",
        categoryLabel: "Labs & Imaging",
        details: "Routine monitoring panel with differential",
        safetyFlag: null,
        rationale: "Baseline workup for new symptoms; last CBC 8 months ago",
        fhirPreview: {
          resourceType: "ServiceRequest",
          status: "draft",
          intent: "order",
          code: { text: "CBC with differential" },
        },
      });
    }

    if (transcriptLower.includes("a1c") || transcriptLower.includes("hemoglobin") || transcriptLower.includes("diabetes")) {
      suggestedActions.push({
        id: `action-${Date.now()}-4`,
        type: "lab",
        status: "pending",
        title: "Hemoglobin A1C",
        categoryLabel: "Labs & Imaging",
        details: "3-month average blood glucose monitoring",
        safetyFlag: null,
        rationale: "Diabetes management requires A1C check every 3 months per guidelines",
        fhirPreview: {
          resourceType: "ServiceRequest",
          status: "draft",
          intent: "order",
          code: { text: "Hemoglobin A1C" },
        },
      });
    }

    if (transcriptLower.includes("x-ray") || transcriptLower.includes("xray") || transcriptLower.includes("chest")) {
      suggestedActions.push({
        id: `action-${Date.now()}-5`,
        type: "imaging",
        status: "pending",
        title: "Chest X-Ray (2 views)",
        categoryLabel: "Labs & Imaging",
        details: "PA and lateral views to evaluate respiratory symptoms",
        safetyFlag: null,
        rationale: "Patient reports persistent cough x 3 weeks; rule out pneumonia",
        fhirPreview: {
          resourceType: "ServiceRequest",
          status: "draft",
          intent: "order",
          code: { text: "Chest X-Ray, 2 views" },
        },
      });
    }

    // Referral suggestions
    if (transcriptLower.includes("cardio") || transcriptLower.includes("heart") || transcriptLower.includes("specialist")) {
      suggestedActions.push({
        id: `action-${Date.now()}-6`,
        type: "referral",
        status: "pending",
        title: "Cardiology Referral",
        categoryLabel: "Referrals & Follow-up",
        details: "Refer to cardiologist for evaluation of chest pain and abnormal EKG findings",
        safetyFlag: "high",
        safetyMessage: "Urgent referral recommended within 1 week",
        rationale: "New onset chest pain with EKG changes; requires specialist evaluation",
        fhirPreview: {
          resourceType: "ServiceRequest",
          status: "draft",
          intent: "order",
          code: { text: "Cardiology consultation" },
        },
      });
    }

    // Follow-up
    if (transcriptLower.includes("follow") || transcriptLower.includes("return") || suggestedActions.length > 0) {
      suggestedActions.push({
        id: `action-${Date.now()}-7`,
        type: "followup",
        status: "pending",
        title: "Follow-up Appointment in 4 weeks",
        categoryLabel: "Referrals & Follow-up",
        details: "Schedule return visit to review lab results and medication response",
        safetyFlag: null,
        rationale: "Standard follow-up for new medication regimen and pending lab work",
        fhirPreview: {
          resourceType: "Appointment",
          status: "proposed",
          serviceType: [{ text: "Follow-up visit" }],
        },
      });
    }

    // Default if nothing matched
    if (suggestedActions.length === 0) {
      suggestedActions.push({
        id: `action-${Date.now()}-1`,
        type: "followup",
        status: "pending",
        title: "Routine Follow-up Visit",
        categoryLabel: "Referrals & Follow-up",
        details: "Schedule appointment to discuss ongoing care and health maintenance",
        safetyFlag: null,
        rationale: "Continue monitoring of chronic conditions",
        fhirPreview: {
          resourceType: "Appointment",
          status: "proposed",
          serviceType: [{ text: "Office visit" }],
        },
      });
    }

    return NextResponse.json({
      suggestedActions,
    });
  } catch (error) {
    return NextResponse.json(
      { error: "Failed to analyze transcript" },
      { status: 500 }
    );
  }
}
