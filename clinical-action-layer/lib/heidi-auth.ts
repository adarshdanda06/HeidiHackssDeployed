import { NextRequest, NextResponse } from 'next/server';

interface JWTResponse {
  token: string;
  expiration_time: string;
}

/**
 * Get JWT token from Heidi API
 * This token is required for all subsequent API calls
 */
export async function getHeidiJWT(email: string, userId: string): Promise<string> {
  const apiKey = process.env.HEIDI_API_KEY;
  
  if (!apiKey) {
    throw new Error('HEIDI_API_KEY not configured');
  }

  const baseUrl = process.env.HEIDI_API_BASE_URL || 'https://registrar.api.heidihealth.com/api/v2/ml-scribe/open-api';
  
  // Authenticate to get JWT
  const url = `${baseUrl}/jwt?email=${encodeURIComponent(email)}&third_party_internal_id=${encodeURIComponent(userId)}`;
  
  console.log('🔑 Getting JWT from Heidi...');
  
  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Heidi-Api-Key': apiKey,
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error('❌ JWT auth failed:', errorText);
    throw new Error(`Failed to get JWT: ${response.status} ${response.statusText}`);
  }

  const data: JWTResponse = await response.json();
  console.log('✅ JWT obtained successfully');
  
  return data.token;
}

// Cache JWT tokens by user (in production, use Redis or similar)
const jwtCache = new Map<string, { token: string; expiration: Date }>();

/**
 * Get cached JWT or fetch new one
 */
export async function getOrRefreshJWT(email: string, userId: string): Promise<string> {
  const cacheKey = `${email}:${userId}`;
  const cached = jwtCache.get(cacheKey);
  
  // Check if cached token is still valid (with 5 min buffer)
  if (cached && cached.expiration > new Date(Date.now() + 5 * 60 * 1000)) {
    console.log('♻️ Using cached JWT');
    return cached.token;
  }
  
  // Fetch new token
  const token = await getHeidiJWT(email, userId);
  
  // Cache it (tokens typically expire in 24 hours)
  jwtCache.set(cacheKey, {
    token,
    expiration: new Date(Date.now() + 23 * 60 * 60 * 1000), // 23 hours
  });
  
  return token;
}

/**
 * Test JWT authentication
 * GET /api/heidi/auth
 */
export async function GET(request: NextRequest) {
  try {
    // Use a test email and user ID (in production, these would come from your auth system)
    const email = 'test@heidihealth.com';
    const userId = 'test-user-123';
    
    const token = await getHeidiJWT(email, userId);
    
    return NextResponse.json({
      success: true,
      message: 'JWT authentication successful',
      tokenPreview: `${token.substring(0, 20)}...`,
      tokenLength: token.length,
    });
    
  } catch (error: any) {
    console.error('Auth error:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to authenticate with Heidi API',
      message: error.message,
    }, { status: 500 });
  }
}
