import { NextRequest, NextResponse } from 'next/server';
import { getHeidiJWT } from '@/lib/heidi-auth';

/**
 * Test JWT authentication
 * GET /api/heidi/auth
 */
export async function GET(request: NextRequest) {
  try {
    // Use a test email and user ID (in production, these would come from your auth system)
    const email = 'test@heidihealth.com';
    const userId = 'hackathon-user-123';
    
    console.log('🔑 Testing Heidi JWT authentication...');
    const token = await getHeidiJWT(email, userId);
    
    return NextResponse.json({
      success: true,
      message: 'JWT authentication successful!',
      tokenPreview: `${token.substring(0, 30)}...`,
      tokenLength: token.length,
      email,
      userId,
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
