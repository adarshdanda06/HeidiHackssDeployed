import { NextRequest, NextResponse } from 'next/server';
import { getOrRefreshJWT } from '@/lib/heidi-auth';

export interface HeidiTranscript {
  visitId: string;
  transcript: string;
  patientName?: string;
  visitDate?: string;
  duration?: number;
  metadata?: Record<string, any>;
}

/**
 * Fetch transcript from Heidi Health API
 * POST /api/heidi/transcript
 * Body: { visitId: string }
 */
export async function POST(request: NextRequest) {
  try {
    const { visitId, sessionId } = await request.json();
    const id = sessionId || visitId;

    if (!id) {
      return NextResponse.json(
        { error: 'sessionId or visitId is required' },
        { status: 400 }
      );
    }

    const apiKey = process.env.HEIDI_API_KEY;
    if (!apiKey) {
      return NextResponse.json(
        { error: 'HEIDI_API_KEY not configured' },
        { status: 500 }
      );
    }

    // Get JWT token for authentication
    // In production, email and userId should come from your auth system
    const email = 'test@heidihealth.com';
    const userId = 'hackathon-user-123';
    
    console.log('🔐 Authenticating with Heidi...');
    const jwtToken = await getOrRefreshJWT(email, userId);

    // Heidi API endpoint - Official base URL from documentation
    // Base: https://registrar.api.heidihealth.com/api/v2/ml-scribe/open-api/
    // To get transcript: GET /sessions/{session_id}/transcript
    const heidiBaseUrl = process.env.HEIDI_API_BASE_URL || 'https://registrar.api.heidihealth.com/api/v2/ml-scribe/open-api';
    
    const url = `${heidiBaseUrl}/sessions/${id}/transcript`;

    console.log(`📞 Fetching transcript for session: ${id}`);
    console.log(`🌐 URL: ${url}`);

    console.log(`📞 Fetching documents for session: ${id}`);
    console.log(`🔑 API Key: ${apiKey.substring(0, 10)}...`);
    console.log(`🌐 URL: ${url}`);

    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${jwtToken}`,
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        signal: AbortSignal.timeout(10000),
      });

      console.log(`📡 Response status: ${response.status}`);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`❌ Heidi API error (${response.status}):`, errorText);
        
        return NextResponse.json(
          { 
            error: `Heidi API error: ${response.statusText}`,
            details: errorText,
            status: response.status,
            endpoint: url,
          },
          { status: response.status }
        );
      }

      const data = await response.json();
      console.log('✅ Transcript fetched successfully');

      // The API returns the transcript directly
      const transcriptText = data.transcript || data.text || data.content || JSON.stringify(data);

      const transcript: HeidiTranscript = {
        visitId: id,
        transcript: transcriptText,
        patientName: data.patient_name || data.patientName,
        visitDate: data.visit_date || data.visitDate || data.timestamp,
        duration: data.duration,
        metadata: data,
      };

      return NextResponse.json({
        success: true,
        data: transcript,
        fetchedAt: new Date().toISOString(),
        endpoint: url,
      });

    } catch (error: any) {
      console.error('💥 Error fetching from Heidi:', error);
      return NextResponse.json(
        { 
          error: 'Failed to fetch transcript from Heidi API',
          message: error.message,
          details: error.toString(),
          endpoint: url,
        },
        { status: 500 }
      );
    }

  } catch (error: any) {
    console.error('💥 Error in Heidi transcript fetch:', error);
    return NextResponse.json(
      { 
        error: 'Failed to fetch transcript from Heidi API',
        message: error.message,
        details: error.toString() 
      },
      { status: 500 }
    );
  }
}

/**
 * Test Heidi API connection
 * GET /api/heidi/transcript
 */
export async function GET(request: NextRequest) {
  const apiKey = process.env.HEIDI_API_KEY;
  
  return NextResponse.json({
    status: 'Heidi API endpoint ready',
    configured: !!apiKey,
    apiKeyPresent: !!apiKey,
    apiKeyPreview: apiKey ? `${apiKey.substring(0, 8)}...` : 'Not set',
    baseUrl: process.env.HEIDI_API_BASE_URL || 'https://api.heidi.health (default)',
    usage: 'POST with { sessionId: "337851254565527952685384877024185083869" }',
    exampleSessionIds: [
      '337851254565527952685384877024185083869',
      '750333248699968106772992654159342594770',
      '209429578973190336673242710141917128963',
    ],
  });
}
