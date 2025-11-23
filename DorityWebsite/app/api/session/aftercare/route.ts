import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";

export async function POST(request: NextRequest) {
  try {
    const { sessionId, approvedActions, historySummary, transcript, patient } = await request.json();

    // If Anthropic key is not available, fall back to basic generation (or empty)
    if (!process.env.ANTHROPIC_API_KEY) {
      return NextResponse.json(
        { error: "ANTHROPIC_API_KEY not configured" },
        { status: 500 }
      );
    }

    const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

    // Format actions for the prompt
    const medications = approvedActions.filter((a: any) => a.type === "medication");
    const labs = approvedActions.filter((a: any) => a.type === "lab");
    const imaging = approvedActions.filter((a: any) => a.type === "imaging");
    const referrals = approvedActions.filter((a: any) => a.type === "referral");
    const followups = approvedActions.filter((a: any) => a.type === "followup");
    const scheduling = approvedActions.filter((a: any) => a.type === "scheduling");

    const actionsContext = JSON.stringify({
      medications,
      labs,
      imaging,
      referrals,
      followups,
      scheduling
    }, null, 2);

    const organizationName = patient?.generalPractitioner || "your organization/practitioner";
    const organizationAddress = patient?.organizationAddress || patient?.address || "the address on file";

    const systemPrompt = `You are a helpful medical assistant drafting a patient after-visit summary email.
Your goal is to create a SINGLE, WARM, and USER-FRIENDLY email that consolidates all information:
1. A friendly opening acknowledging the visit.
2. A summary of what was discussed (using the transcript for context).
3. Clear instructions for next steps (medications, labs, referrals).
4. IMPORTANT: If there are any "scheduling" actions (like follow-up meetings), integrate them naturally into the email. 
   - Do NOT use robotic headers like "New Meeting Request" or "Reason: ...". 
   - Instead, say something like "We'd like to schedule a follow-up to discuss X, ideally next Tuesday."
   - Use the "reason" and "when" fields from the scheduling action to write this naturally.
5. A friendly closing.
6. Mention the patient's organization/practitioner and include the location where they can be reached.
7. End the email by reminding the patient that ${organizationName} is located at ${organizationAddress}.

Output purely a JSON object with two keys:
- "subject": A user-friendly email subject line.
- "body": The full email body text.

Do not use markdown for the JSON. Just raw JSON.`;

    const patientContext = patient
      ? `Patient: ${patient.name}
Address: ${patient.address || "Not provided"}
Preferred Pharmacy: ${patient.preferredPharmacy || "Not specified"}
Organization/Practitioner: ${organizationName}
Organization Address: ${organizationAddress}`
      : "Patient information not available.";

    const reminderInstruction = patient
      ? `Please finish the email by reminding the patient that ${organizationName} is located at ${organizationAddress}.`
      : "Please finish the email by reminding the patient of their organization/practitioner and the address on file.";

    const userMessage = `Patient Details:
${patientContext}

Here is the transcript of the visit:
${transcript ? transcript : "No transcript available."}

Here are the approved actions to include:
${actionsContext}

Please draft the email.

${reminderInstruction}`;

    const msg = await anthropic.messages.create({
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 2000,
      system: systemPrompt,
      messages: [
        { role: "user", content: userMessage }
      ]
    });

    const content = msg.content[0].type === 'text' ? msg.content[0].text : '';
    let generatedEmail;
    
    try {
      generatedEmail = JSON.parse(content.trim());
    } catch (e) {
      // Fallback if JSON parse fails
      console.error("Failed to parse Claude response:", content);
      generatedEmail = {
        subject: "Your Visit Summary",
        body: content // Treat the whole content as body if not JSON
      };
    }

    // Combine subject and body for the frontend editor
    // The frontend likely expects a single string for the "summary" field.
    // We will format it so the user can see the subject.
    const reminderNote = `P.S. ${organizationName} is located at ${organizationAddress}.`;
    const bodyWithReminder = `${generatedEmail.body}\n\n${reminderNote}`;

    const combinedSummary = `Subject: ${generatedEmail.subject}\n\n${bodyWithReminder}`;
    console.log('Combined Summary:', combinedSummary);
    return NextResponse.json({
      aftercareSummary: combinedSummary,
    });

  } catch (error) {
    console.error("Error generating aftercare:", error);
    return NextResponse.json(
      { error: "Failed to generate aftercare summary" },
      { status: 500 }
    );
  }
}
