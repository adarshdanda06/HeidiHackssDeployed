import { NextRequest, NextResponse } from 'next/server';

/**
 * Get available Heidi sessions
 * GET /api/heidi/sessions
 */
export async function GET(request: NextRequest) {
  try {
    const apiKey = process.env.HEIDI_API_KEY;
    
    if (!apiKey) {
      return NextResponse.json({
        success: false,
        error: 'HEIDI_API_KEY not configured',
      }, { status: 500 });
    }

    const heidiBaseUrl = process.env.HEIDI_API_BASE_URL || 'https://api.heidi.health';
    
    // Try to fetch available sessions
    // Common endpoints: /sessions, /v1/sessions, /api/sessions
    const possibleEndpoints = [
      `${heidiBaseUrl}/sessions`,
      `${heidiBaseUrl}/v1/sessions`,
      `${heidiBaseUrl}/api/sessions`,
    ];

    for (const url of possibleEndpoints) {
      try {
        console.log(`🔍 Trying: ${url}`);
        
        const response = await fetch(url, {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          signal: AbortSignal.timeout(5000),
        });

        if (response.ok) {
          const data = await response.json();
          console.log('✅ Sessions endpoint found:', url);
          
          return NextResponse.json({
            success: true,
            endpoint: url,
            sessions: data.sessions || data.data || data,
            count: Array.isArray(data.sessions) ? data.sessions.length : 
                   Array.isArray(data.data) ? data.data.length :
                   Array.isArray(data) ? data.length : 0,
          });
        }
      } catch (error) {
        console.log(`❌ Failed: ${url}`);
        continue;
      }
    }

    // If no endpoint works, return the hardcoded session IDs from the hackathon
    return NextResponse.json({
      success: true,
      message: 'Using hackathon-provided session IDs',
      sessions: [
        { id: '337851254565527952685384877024185083869', label: 'Session 1' },
        { id: '750333248699968106772992654159342594770', label: 'Session 2' },
        { id: '209429578973190336673242710141917128963', label: 'Session 3' },
        { id: '316272209747326581157737075663692625433', label: 'Session 4' },
        { id: '483297810582323025582777956703408080822', label: 'Session 5' },
        { id: '189878368687884891206528465309407076433', label: 'Session 6' },
        { id: '179340005192510878551324680590964837821', label: 'Session 7' },
        { id: '535873167906829714469358805153241006567', label: 'Session 8' },
      ],
      count: 8,
      note: 'These are the session IDs provided by the hackathon organizers',
    });

  } catch (error: any) {
    console.error('Error fetching sessions:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to fetch sessions',
      message: error.message,
    }, { status: 500 });
  }
}
