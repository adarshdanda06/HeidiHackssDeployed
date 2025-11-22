# Clinical Action Layer - Project Context

## Project Overview
We are building a "Sidecar" Web App (Next.js) that automates clinical data entry. It sits next to an EMR (Medplum) and uses AI to listen to patient consultations and pre-fill orders for the doctor to approve.

## Tech Stack
- **Frontend:** Next.js 14+ (App Router), React, Tailwind CSS, Lucide React (Icons).
- **Backend:** Medplum (Headless FHIR Server).
- **AI:** Claude 3.5 Sonnet (via Anthropic SDK).
- **Inputs:** Heidi API (Transcripts).

## Core Architecture
1. **SMART on FHIR:** The app is designed to be interoperable. It talks to Medplum via the FHIR REST API.
2. **Data Standards:** All data written to the DB must be valid FHIR R4 resources (MedicationRequest, ServiceRequest).
3. **No Direct DB Access:** We use the `@medplum/core` SDK for all data fetching and writing.

## Key Workflows
- **Ingest:** Fetch transcript from Heidi API.
- **Process:** Claude extracts clinical intents -> maps to FHIR resources.
- **Review:** UI displays "Action Cards" (Draft Orders).
- **Execute:** Doctor clicks "Approve" -> SDK writes to Medplum.

## Coding Rules
- Use TypeScript for everything.
- Use functional React components.
- Do not use "any" types; use proper FHIR types (e.g., `MedicationRequest`, `Patient`) from `@medplum/fhirtypes`.
- Handle loading states and errors gracefully in the UI.
