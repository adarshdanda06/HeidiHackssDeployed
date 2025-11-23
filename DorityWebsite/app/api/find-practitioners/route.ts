import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const city = searchParams.get('city');
  const state = searchParams.get('state');
  const postal_code = searchParams.get('postal_code');
  const taxonomy_description = searchParams.get('specialty');
  const limit = searchParams.get('limit') || '10';

  if (!city && !state && !postal_code) {
    return NextResponse.json({ error: 'At least one location parameter (city, state, postal_code) is required' }, { status: 400 });
  }

  try {
    const params = new URLSearchParams({
      version: '2.1',
      limit: limit,
      address_purpose: 'LOCATION', // Practice location
    });

    if (city) params.append('city', city);
    if (state) params.append('state', state);
    if (postal_code) params.append('postal_code', postal_code);
    if (taxonomy_description) params.append('taxonomy_description', taxonomy_description);

    const response = await fetch(`https://npiregistry.cms.hhs.gov/api/?${params.toString()}`);
    
    if (!response.ok) {
      throw new Error(`NPI Registry API failed with status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching from NPI Registry:', error);
    return NextResponse.json({ error: 'Failed to fetch practitioners' }, { status: 500 });
  }
}

