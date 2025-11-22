import { NextRequest, NextResponse } from 'next/server';
import { getOrRefreshJWT } from '@/lib/heidi-auth';

/**
 * Create a document for a session
 * This triggers Heidi to generate/transcribe the session
 * POST /api/heidi/create-document
 * Body: { sessionId: string }
 */
export async function POST(request: NextRequest) {
  try {
    const { sessionId } = await request.json();

    if (!sessionId) {
      return NextResponse.json(
        { error: 'sessionId is required' },
        { status: 400 }
      );
    }

    // Get JWT token
    const email = 'test@heidihealth.com';
    const userId = 'hackathon-user-123';
    const jwtToken = await getOrRefreshJWT(email, userId);

    const baseUrl = process.env.HEIDI_API_BASE_URL || 'https://registrar.api.heidihealth.com/api/v2/ml-scribe/open-api';
    const url = `${baseUrl}/sessions/${sessionId}/documents`;

    console.log(`📝 Creating document for session: ${sessionId}`);

    // According to the API docs, generation_method is REQUIRED
    // For hackathon, let's try without a template first (may need to get template IDs)
    const requestBody = {
      document_tab_type: 'DOCUMENT',
      brain: 'LEFT',
    };

    console.log('Request body:', JSON.stringify(requestBody));

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${jwtToken}`,
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(30000),
    });

    console.log(`📡 Response status: ${response.status}`);

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`❌ Create document error:`, errorText);
      
      return NextResponse.json(
        { 
          error: `Heidi API error: ${response.statusText}`,
          details: errorText,
          status: response.status,
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    console.log('✅ Document created successfully');

    return NextResponse.json({
      success: true,
      data,
      message: 'Document created! You can now fetch the transcript.',
    });

  } catch (error: any) {
    console.error('💥 Error creating document:', error);
    return NextResponse.json(
      { 
        error: 'Failed to create document',
        message: error.message,
      },
      { status: 500 }
    );
  }
}
