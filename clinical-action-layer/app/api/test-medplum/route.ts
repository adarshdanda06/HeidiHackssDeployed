import { NextRequest, NextResponse } from 'next/server';
import { getMedplumClient } from '@/lib/medplum-client';

export async function GET(request: NextRequest) {
  try {
    // Test Medplum connection
    const medplum = await getMedplumClient();
    
    // Get your profile to verify authentication
    const profile = await medplum.getProfile();
    
    // Try to search for a patient (just to test read access)
    const patients = await medplum.searchResources('Patient', { _count: 1 });
    
    return NextResponse.json({
      success: true,
      message: 'Medplum connection successful!',
      profile: {
        id: profile.id,
        resourceType: profile.resourceType,
        display: profile.display,
      },
      patientCount: patients.length,
      baseUrl: process.env.MEDPLUM_BASE_URL,
    });
    
  } catch (error: any) {
    console.error('Medplum test error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to connect to Medplum',
        details: error.toString(),
      },
      { status: 500 }
    );
  }
}
