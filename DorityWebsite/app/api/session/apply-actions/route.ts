import { NextRequest, NextResponse } from "next/server";
import sgMail from '@sendgrid/mail';

// Hardcoded email as requested
const TARGET_EMAIL = "adarsh.danda1@gmail.com"; // TODO: Update this to your verified sender/recipient

export async function POST(request: NextRequest) {
  try {
    const { sessionId, actions } = await request.json();

    // Simulate EMR write delay
    await new Promise((resolve) => setTimeout(resolve, 1500));

    // Process actions (including Email sending)
    for (const action of actions) {
      // Email sending moved to Aftercare summary for consolidated communication
      /* 
      if (action.type === 'scheduling') {
        console.log('📧 Processing scheduling action...');
        
        if (process.env.SENDGRID_API_KEY) {
          // ... existing email logic ...
        }
      } 
      */
    }

    // Mark all actions as successfully applied
    const appliedActions = actions.map((action: any) => ({
      ...action,
      status: "approved",
      appliedAt: new Date().toISOString(),
      appliedToEMR: true,
    }));

    return NextResponse.json({
      success: true,
      appliedActions,
      message: `Successfully applied ${appliedActions.length} action(s) to Medplum EMR`,
    });
  } catch (error) {
    return NextResponse.json(
      { error: "Failed to apply actions to EMR" },
      { status: 500 }
    );
  }
}
