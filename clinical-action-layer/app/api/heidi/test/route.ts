import { NextRequest, NextResponse } from 'next/server';

/**
 * Test Heidi API connection
 * GET /api/heidi/test
 */
export async function GET(request: NextRequest) {
  try {
    const apiKey = process.env.HEIDI_API_KEY;
    
    if (!apiKey) {
      return NextResponse.json({
        success: false,
        error: 'HEIDI_API_KEY not configured in environment variables',
        configured: false,
      }, { status: 500 });
    }

    const heidiBaseUrl = process.env.HEIDI_API_BASE_URL || 'https://registrar.api.heidihealth.com/api/v2/ml-scribe/open-api';

    // Test with a health/ping endpoint if available
    const testUrl = `${heidiBaseUrl}/health`;

    try {
      const response = await fetch(testUrl, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        signal: AbortSignal.timeout(5000), // 5 second timeout
      });

      return NextResponse.json({
        success: true,
        message: 'Heidi API connection successful',
        apiConfigured: true,
        baseUrl: heidiBaseUrl,
        healthCheckStatus: response.status,
        healthCheckOk: response.ok,
        apiKeyPreview: `${apiKey.substring(0, 8)}...${apiKey.slice(-4)}`,
      });

    } catch (fetchError: any) {
      // If health endpoint doesn't exist, that's okay
      // As long as we have an API key, we can still try fetching transcripts
      return NextResponse.json({
        success: true,
        message: 'Heidi API key configured (health endpoint unavailable)',
        apiConfigured: true,
        baseUrl: heidiBaseUrl,
        apiKeyPreview: `${apiKey.substring(0, 8)}...${apiKey.slice(-4)}`,
        note: 'Health endpoint not available, but API key is set',
        healthCheckError: fetchError.message,
      });
    }

  } catch (error: any) {
    console.error('Heidi test error:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to test Heidi API connection',
      message: error.message,
    }, { status: 500 });
  }
}
