"use client";

import { useState, useEffect } from "react";
import { Questionnaire, QuestionnaireItem } from "@medplum/fhirtypes";
import { Loader2, MapPin } from "lucide-react";
import { mapFHIRToQuestionnaireResponses } from "@/lib/questionnaireMapper";

interface QuestionnaireFormProps {
  questionnaireId: string;
  isEditable: boolean;
  onResponseChange?: (responses: Record<string, any>) => void;
  initialResponses?: Record<string, any>;
  fhirResource?: any; // FHIR resource to extract values from for autofilling
  patientData?: any; // Patient demographic data for autofilling patient fields
}

interface QuestionnaireItemRendererProps {
  item: QuestionnaireItem;
  isEditable: boolean;
  value: any;
  onChange: (linkId: string, value: any) => void;
  depth?: number;
  patientData?: any;
}

function QuestionnaireItemRenderer({
  item,
  isEditable,
  value,
  onChange,
  depth = 0,
  patientData
}: QuestionnaireItemRendererProps) {
  const isRequired = item.required === true;
  const linkId = item.linkId || '';
  
  // Get the actual value for this item from the flat responses object
  const itemValue = value?.[linkId];
  
  // Provider Search State
  const [practitioners, setPractitioners] = useState<any[]>([]);
  const [isLoadingPractitioners, setIsLoadingPractitioners] = useState(false);
  const [showPractitionerSearch, setShowPractitionerSearch] = useState(false);

  // Identify provider fields
  const isProviderField = (
    (item.text?.toLowerCase().includes('doctor') || 
     // Check for 'clinic' but exclude 'clinical'
     (item.text?.toLowerCase().includes('clinic') && !item.text?.toLowerCase().includes('clinical')) ||
     item.linkId?.toLowerCase().includes('provider') ||
     item.linkId?.toLowerCase().includes('performer')) && 
    (item.type === 'string' || item.type === 'text' || item.type === 'reference') &&
    // Explicitly exclude reason/indication fields just in case
    !item.text?.toLowerCase().includes('reason') &&
    !item.linkId?.toLowerCase().includes('reason')
  );

  const searchPractitioners = async () => {
    setIsLoadingPractitioners(true);
    setShowPractitionerSearch(true);
    try {
      // Try to parse city/state from patient address
      let city = '';
      let state = '';
      
      if (patientData?.address) {
        const parts = patientData.address.split(',');
        if (parts.length >= 2) {
          const stateZip = parts[parts.length - 1].trim().split(' ');
          if (stateZip.length >= 1) state = stateZip[0];
          city = parts[parts.length - 2].trim();
        }
      }

      // Try to find specialty from other form fields
      let specialty = '';
      if (value) {
        // Look for keys containing 'specialty'
        const specialtyKey = Object.keys(value).find(k => k.toLowerCase().includes('specialty'));
        if (specialtyKey && value[specialtyKey]) {
          specialty = value[specialtyKey];
        }
      }

      const params = new URLSearchParams();
      if (city) params.append('city', city);
      if (state) params.append('state', state);
      if (specialty) params.append('specialty', specialty);
      params.append('limit', '5');

      const response = await fetch(`/api/find-practitioners?${params.toString()}`);
      const data = await response.json();
      
      if (data.results) {
        setPractitioners(data.results);
      } else {
        setPractitioners([]);
      }
    } catch (error) {
      console.error("Failed to search practitioners", error);
    } finally {
      setIsLoadingPractitioners(false);
    }
  };
  
  // Debug log to see what values we're rendering - SPECIAL CHECK FOR MEDICATION
  if (linkId === 'medication-name' || linkId === 'medication' || item.text?.toLowerCase().includes('medication')) {
    console.log(`🔍🔍🔍 MEDICATION FIELD DEBUG:`);
    console.log(`  linkId: "${linkId}"`);
    console.log(`  item.text: "${item.text}"`);
    console.log(`  itemValue:`, itemValue);
    console.log(`  value object has medication-name?:`, value?.['medication-name']);
    console.log(`  value object has medication?:`, value?.['medication']);
    console.log(`  All available linkIds:`, Object.keys(value || {}).slice(0, 10));
  }
  
  // Debug log to see what values we're rendering
  if (isRequired && (!itemValue || itemValue === '')) {
    console.log(`[QuestionnaireItemRenderer] 🔴 Required field EMPTY: ${linkId} (${item.text})`);
  } else if (itemValue) {
    console.log(`[QuestionnaireItemRenderer] ✅ Field has value: ${linkId} =`, itemValue);
  }

  // Render based on item type
  const renderInput = () => {
    // Check if field is empty and required for highlighting
    const isEmpty = !itemValue || itemValue === '' || itemValue === 'Select an option...' || itemValue === 'Select an option';
    // HIGHLIGHT ALL EMPTY FIELDS, not just required ones
    const needsAttention = isEmpty && isEditable;
    
    // Debug logging for dropdowns
    if (item.type === 'choice') {
      console.log(`[Dropdown] "${item.text}" (${linkId}):`, {
        itemValue,
        isEmpty,
        isEditable,
        needsAttention
      });
    }

    // Custom Provider Search UI
    if (isProviderField && isEditable) {
      return (
        <div className="space-y-2">
          <div className="flex gap-2">
            <input
              type="text"
              value={itemValue || ''}
              onChange={(e) => onChange(linkId, e.target.value)}
              disabled={!isEditable}
              className={`flex-1 text-xs text-zinc-600 border rounded px-2 py-1.5 focus:outline-none focus:ring-2 disabled:bg-zinc-50 disabled:text-zinc-500 ${
                needsAttention 
                  ? 'border-amber-300 bg-amber-50/30 focus:ring-amber-500/20 focus:border-amber-500/30' 
                  : 'border-zinc-200 focus:ring-[#7C2D3E]/20 focus:border-[#7C2D3E]/30'
              }`}
              placeholder={needsAttention ? 'Required field - please complete' : (item.text || "Enter provider name")}
            />
            <button
              onClick={searchPractitioners}
              disabled={isLoadingPractitioners}
              type="button" // Prevent form submission
              className="px-3 py-1.5 text-xs font-medium text-white bg-[#7C2D3E] hover:bg-[#5A1F2D] rounded-lg transition-all flex items-center gap-1.5 disabled:opacity-50 whitespace-nowrap"
            >
              {isLoadingPractitioners ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <MapPin className="w-3.5 h-3.5" />}
              Find Local
            </button>
          </div>
          
          {/* Search Results */}
          {showPractitionerSearch && practitioners.length > 0 && (
            <div className="border border-zinc-200 rounded-lg overflow-hidden max-h-40 overflow-y-auto bg-zinc-50/50">
              {practitioners.map((p, i) => (
                <div 
                  key={i}
                  onClick={() => {
                    const name = `${p.basic.name_prefix || ''} ${p.basic.first_name} ${p.basic.last_name} ${p.basic.credential || ''}`.trim();
                    const address = p.addresses[0] ? `${p.addresses[0].address_1}, ${p.addresses[0].city}, ${p.addresses[0].state}` : '';
                    onChange(linkId, `${name} - ${address} (NPI: ${p.number})`);
                    setShowPractitionerSearch(false);
                  }}
                  className="p-2.5 hover:bg-white hover:shadow-sm cursor-pointer border-b border-zinc-100 last:border-0 text-xs transition-all"
                >
                  <div className="font-medium text-zinc-900">
                    {p.basic.name_prefix} {p.basic.first_name} {p.basic.last_name} {p.basic.credential}
                  </div>
                  <div className="text-zinc-500 truncate mt-0.5">
                    {p.taxonomies?.[0]?.desc} • {p.addresses?.[0]?.city}, {p.addresses?.[0]?.state}
                  </div>
                </div>
              ))}
            </div>
          )}
          {showPractitionerSearch && practitioners.length === 0 && !isLoadingPractitioners && (
             <div className="text-xs text-zinc-500 italic p-1">
               No local practitioners found. Try manually entering the name.
             </div>
          )}
        </div>
      );
    }
    
    switch (item.type) {
      case 'string':
      case 'text':
        if (item.type === 'text') {
          return (
            <textarea
              value={itemValue || ''}
              onChange={(e) => onChange(linkId, e.target.value)}
              disabled={!isEditable}
              className={`w-full text-xs text-zinc-600 border rounded px-2 py-1.5 focus:outline-none focus:ring-2 resize-none disabled:bg-zinc-50 disabled:text-zinc-500 ${
                needsAttention 
                  ? 'border-amber-300 bg-amber-50/30 focus:ring-amber-500/20 focus:border-amber-500/30' 
                  : 'border-zinc-200 focus:ring-[#7C2D3E]/20 focus:border-[#7C2D3E]/30'
              }`}
              rows={3}
              placeholder={needsAttention ? 'Required field - please complete' : item.text}
            />
          );
        }
        return (
          <input
            type="text"
            value={itemValue || ''}
            onChange={(e) => onChange(linkId, e.target.value)}
            disabled={!isEditable}
            className={`w-full text-xs text-zinc-600 border rounded px-2 py-1.5 focus:outline-none focus:ring-2 disabled:bg-zinc-50 disabled:text-zinc-500 ${
              needsAttention 
                ? 'border-amber-300 bg-amber-50/30 focus:ring-amber-500/20 focus:border-amber-500/30' 
                : 'border-zinc-200 focus:ring-[#7C2D3E]/20 focus:border-[#7C2D3E]/30'
            }`}
            placeholder={needsAttention ? 'Required field - please complete' : item.text}
          />
        );

      case 'boolean':
        return (
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name={linkId}
                checked={itemValue === true}
                onChange={() => onChange(linkId, true)}
                disabled={!isEditable}
                className="text-[#7C2D3E] focus:ring-[#7C2D3E]/20"
              />
              <span className="text-xs text-zinc-700">Yes</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name={linkId}
                checked={itemValue === false}
                onChange={() => onChange(linkId, false)}
                disabled={!isEditable}
                className="text-[#7C2D3E] focus:ring-[#7C2D3E]/20"
              />
              <span className="text-xs text-zinc-700">No</span>
            </label>
          </div>
        );

      case 'choice':
        return (
          <select
            value={itemValue || ''}
            onChange={(e) => {
              // For dropdowns, we need to handle valueCoding properly
              const selectedOption = item.answerOption?.find(
                opt => (opt.valueCoding?.code || opt.valueString) === e.target.value
              );
              
              if (selectedOption) {
                // If the option has valueCoding, pass the full coding object
                if (selectedOption.valueCoding) {
                  onChange(linkId, selectedOption.valueCoding.code);
                } else {
                  onChange(linkId, e.target.value);
                }
              } else {
                onChange(linkId, e.target.value);
              }
            }}
            disabled={!isEditable}
            className={`w-full text-xs text-zinc-600 border rounded px-2 py-1.5 focus:outline-none focus:ring-2 disabled:bg-zinc-50 disabled:text-zinc-500 ${
              needsAttention 
                ? 'border-amber-300 bg-amber-50/30 focus:ring-amber-500/20 focus:border-amber-500/30' 
                : 'border-zinc-200 focus:ring-[#7C2D3E]/20 focus:border-[#7C2D3E]/30'
            }`}
          >
            <option value="">{needsAttention ? 'Required - Select an option...' : 'Select an option...'}</option>
            {item.answerOption?.map((option, idx) => (
              <option 
                key={idx} 
                value={option.valueCoding?.code || option.valueString || idx}
              >
                {option.valueCoding?.display || option.valueString}
              </option>
            ))}
          </select>
        );

      case 'date':
        return (
          <input
            type="date"
            value={itemValue || ''}
            onChange={(e) => onChange(linkId, e.target.value)}
            disabled={!isEditable}
            className="w-full text-xs text-zinc-600 border border-zinc-200 rounded px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-[#7C2D3E]/20 focus:border-[#7C2D3E]/30 disabled:bg-zinc-50 disabled:text-zinc-500"
          />
        );

      case 'integer':
      case 'decimal':
        return (
          <input
            type="number"
            step={item.type === 'decimal' ? '0.01' : '1'}
            value={itemValue || ''}
            onChange={(e) => onChange(linkId, item.type === 'decimal' ? parseFloat(e.target.value) : parseInt(e.target.value))}
            disabled={!isEditable}
            className="w-full text-xs text-zinc-600 border border-zinc-200 rounded px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-[#7C2D3E]/20 focus:border-[#7C2D3E]/30 disabled:bg-zinc-50 disabled:text-zinc-500"
            placeholder={item.text}
          />
        );

      case 'display':
        return (
          <p className="text-xs text-zinc-600 italic">{item.text}</p>
        );

      case 'group':
        // Groups are handled separately
        return null;

      default:
        return (
          <input
            type="text"
            value={itemValue || ''}
            onChange={(e) => onChange(linkId, e.target.value)}
            disabled={!isEditable}
            className="w-full text-xs text-zinc-600 border border-zinc-200 rounded px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-[#7C2D3E]/20 focus:border-[#7C2D3E]/30 disabled:bg-zinc-50 disabled:text-zinc-500"
            placeholder={item.text}
          />
        );
    }
  };

  // Handle group items differently
  if (item.type === 'group') {
    return (
      <div className={`space-y-3 ${depth > 0 ? 'ml-4 pl-4 border-l-2 border-zinc-200' : ''}`}>
        {item.text && (
          <h5 className="text-xs font-semibold text-zinc-800 mb-2">{item.text}</h5>
        )}
        {item.item?.map((subItem, idx) => (
          <QuestionnaireItemRenderer
            key={idx}
            item={subItem}
            isEditable={isEditable}
            value={value}
            onChange={onChange}
            depth={depth + 1}
            patientData={patientData}
          />
        ))}
      </div>
    );
  }

  // Handle display items
  if (item.type === 'display') {
    return (
      <div className="p-2.5 bg-blue-50/30 border border-blue-200/50 rounded-lg">
        <p className="text-xs text-blue-900">{item.text}</p>
      </div>
    );
  }

  // Regular question item
  return (
    <div className="p-2.5 bg-white border border-zinc-200/50 rounded-lg">
      <label className="block text-xs font-semibold text-zinc-700 mb-1.5">
        {item.text}
        {isRequired && <span className="text-red-500 ml-1">*</span>}
      </label>
      {renderInput()}
    </div>
  );
}

export default function QuestionnaireForm({
  questionnaireId,
  isEditable,
  onResponseChange,
  initialResponses = {},
  fhirResource,
  patientData
}: QuestionnaireFormProps) {
  const [questionnaire, setQuestionnaire] = useState<Questionnaire | null>(null);
  const [responses, setResponses] = useState<Record<string, any>>(initialResponses);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hasAutofilledOnce, setHasAutofilledOnce] = useState(false);

  // Fetch questionnaire definition
  useEffect(() => {
    async function fetchQuestionnaire() {
      try {
        setLoading(true);
        setError(null);
        
        console.log('[QuestionnaireForm] Fetching questionnaire:', questionnaireId);
        
        const response = await fetch(`/api/questionnaire/${questionnaireId}`);
        
        if (!response.ok) {
          throw new Error('Failed to fetch questionnaire');
        }

        const data = await response.json();
        console.log('[QuestionnaireForm] Questionnaire data:', data.questionnaire);
        setQuestionnaire(data.questionnaire);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching questionnaire:', err);
        setError('Failed to load questionnaire form');
        setLoading(false);
      }
    }

    if (questionnaireId) {
      fetchQuestionnaire();
    }
  }, [questionnaireId]);

  // Autofill from fhirResource once when questionnaire is loaded
  useEffect(() => {
    if (!questionnaire || hasAutofilledOnce) return;
    
    console.log('[QuestionnaireForm] Starting autofill...');
    console.log('[QuestionnaireForm] FHIR resource:', fhirResource);
    
    // If fhirResource is a QuestionnaireResponse, extract answers directly
    if (fhirResource && fhirResource.resourceType === 'QuestionnaireResponse') {
      console.log('[QuestionnaireForm] ===== EXTRACTION START =====');
      console.log('[QuestionnaireForm] Full fhirResource:', JSON.stringify(fhirResource, null, 2));
      console.log('[QuestionnaireForm] fhirResource.item count:', fhirResource.item?.length || 0);
      console.log('[QuestionnaireForm] First 3 items:', fhirResource.item?.slice(0, 3));
      const extractedResponses: Record<string, any> = {};
      
      // Recursive function to extract answers from nested items
      const extractAnswers = (items: any[], depth = 0) => {
        const indent = '  '.repeat(depth);
        console.log(`${indent}[QuestionnaireForm] Extracting from ${items.length} items at depth ${depth}`);
        
        for (const item of items) {
          console.log(`${indent}[QuestionnaireForm] Processing item:`, item.linkId, 'type:', item.type);
          
          // If this item has a direct answer, extract it
          if (item.linkId && item.answer && item.answer.length > 0) {
            const answer = item.answer[0];
            console.log(`${indent}  -> Found answer:`, answer);
            
            // Extract the actual value from the answer
            // CRITICAL: For valueCoding, use CODE not display (form expects code as value)
            let value;
            if (answer.valueCoding) {
              // For choice fields with valueCoding, use the CODE (what the select expects)
              value = answer.valueCoding.code || answer.valueCoding.display;
            } else {
              value = answer.valueString || 
                      answer.valueBoolean || 
                      answer.valueInteger || 
                      answer.valueDecimal ||
                      answer.valueDate ||
                      answer.valueReference?.display;
            }
            
            if (value !== undefined && value !== null) {
              extractedResponses[item.linkId] = value;
              console.log(`${indent}  ✅ Extracted ${item.linkId}:`, value);
            } else {
              console.log(`${indent}  ❌ No extractable value from answer:`, answer);
            }
          } else {
            console.log(`${indent}  -> No answer array for ${item.linkId}`);
          }
          
          // If this item has nested items (group), recursively extract from them
          if (item.item && Array.isArray(item.item)) {
            console.log(`${indent}  -> Recursing into ${item.item.length} nested items`);
            extractAnswers(item.item, depth + 1);
          }
        }
      };
      
      if (fhirResource.item && Array.isArray(fhirResource.item)) {
        extractAnswers(fhirResource.item);
      }
      
      console.log('[QuestionnaireForm] ===== EXTRACTION COMPLETE =====');
      console.log('[QuestionnaireForm] Final extracted responses:', extractedResponses);
      console.log('[QuestionnaireForm] Total fields extracted:', Object.keys(extractedResponses).length);
      console.log('[QuestionnaireForm] LinkIds extracted:', Object.keys(extractedResponses));
      console.log('[QuestionnaireForm] Setting responses state...');
      
      // WORKAROUND: Map common field mismatches
      const fieldMappings: Record<string, string> = {
        'medication-name': 'medication',
        'medication-strength': 'strength',
        'medication-form': 'form',
      };
      
      for (const [fromField, toField] of Object.entries(fieldMappings)) {
        if (extractedResponses[fromField] && !extractedResponses[toField]) {
          console.log(`[QuestionnaireForm] 🔧 Mapping ${fromField} → ${toField}: "${extractedResponses[fromField]}"`);
          extractedResponses[toField] = extractedResponses[fromField];
        }
      }
      
      setResponses(extractedResponses);
      onResponseChange?.(extractedResponses);
      setHasAutofilledOnce(true);
    }
    // Fallback: try to autofill from FHIR resource if provided
    else if (fhirResource && questionnaire) {
      console.log('[QuestionnaireForm] Starting fallback autofill...');
      console.log('[QuestionnaireForm] Patient data:', patientData);
      const autofilledResponses = mapFHIRToQuestionnaireResponses(
        fhirResource,
        questionnaire,
        patientData
      );
      console.log('[QuestionnaireForm] Autofilled responses:', autofilledResponses);
      setResponses(autofilledResponses);
      onResponseChange?.(autofilledResponses);
      setHasAutofilledOnce(true);
    } else {
      console.log('[QuestionnaireForm] No autofill - fhirResource:', !!fhirResource, 'questionnaire:', !!questionnaire);
      setHasAutofilledOnce(true);
    }
  }, [questionnaire, fhirResource, patientData, onResponseChange, hasAutofilledOnce]);

  const handleResponseChange = (linkId: string, value: any) => {
    const newResponses = { ...responses, [linkId]: value };
    setResponses(newResponses);
    onResponseChange?.(newResponses);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="w-5 h-5 text-zinc-400 animate-spin" />
        <span className="ml-2 text-xs text-zinc-500">Loading form...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
        <p className="text-xs text-red-700">{error}</p>
      </div>
    );
  }

  if (!questionnaire || !questionnaire.item) {
    return (
      <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg">
        <p className="text-xs text-amber-700">No questions available for this form.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {questionnaire.title && (
        <h4 className="text-sm font-semibold text-zinc-900">
          {questionnaire.title}
        </h4>
      )}
      {questionnaire.description && (
        <p className="text-xs text-zinc-600 italic">{questionnaire.description}</p>
      )}
      
      <div className="space-y-3">
        {questionnaire.item.map((item, idx) => (
          <QuestionnaireItemRenderer
            key={idx}
            item={item}
            isEditable={isEditable}
            value={responses}
            onChange={handleResponseChange}
            patientData={patientData}
          />
        ))}
      </div>
    </div>
  );
}
