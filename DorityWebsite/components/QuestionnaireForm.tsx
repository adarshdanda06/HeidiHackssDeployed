"use client";

import { useState, useEffect } from "react";
import { Questionnaire, QuestionnaireItem } from "@medplum/fhirtypes";
import { Loader2 } from "lucide-react";
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
}

function QuestionnaireItemRenderer({
  item,
  isEditable,
  value,
  onChange,
  depth = 0
}: QuestionnaireItemRendererProps) {
  const isRequired = item.required === true;
  const linkId = item.linkId || '';
  
  // Get the actual value for this item from the flat responses object
  const itemValue = value?.[linkId];
  
  // Debug log for nested items
  if (depth > 0 && itemValue !== undefined) {
    console.log(`[QuestionnaireItemRenderer] Rendering nested item ${linkId} with value:`, itemValue);
  }

  // Render based on item type
  const renderInput = () => {
    switch (item.type) {
      case 'string':
      case 'text':
        if (item.type === 'text') {
          return (
            <textarea
              value={itemValue || ''}
              onChange={(e) => onChange(linkId, e.target.value)}
              disabled={!isEditable}
              className="w-full text-xs text-zinc-600 border border-zinc-200 rounded px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-[#7C2D3E]/20 focus:border-[#7C2D3E]/30 resize-none disabled:bg-zinc-50 disabled:text-zinc-500"
              rows={3}
              placeholder={item.text}
            />
          );
        }
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
            onChange={(e) => onChange(linkId, e.target.value)}
            disabled={!isEditable}
            className="w-full text-xs text-zinc-600 border border-zinc-200 rounded px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-[#7C2D3E]/20 focus:border-[#7C2D3E]/30 disabled:bg-zinc-50 disabled:text-zinc-500"
          >
            <option value="">Select an option...</option>
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

  useEffect(() => {
    async function fetchQuestionnaire() {
      try {
        setLoading(true);
        setError(null);
        
        console.log('[QuestionnaireForm] Fetching questionnaire:', questionnaireId);
        console.log('[QuestionnaireForm] FHIR resource:', fhirResource);
        
        const response = await fetch(`/api/questionnaire/${questionnaireId}`);
        
        if (!response.ok) {
          throw new Error('Failed to fetch questionnaire');
        }

        const data = await response.json();
        console.log('[QuestionnaireForm] Questionnaire data:', data.questionnaire);
        setQuestionnaire(data.questionnaire);
        
        // If fhirResource is a QuestionnaireResponse, extract answers directly
        if (fhirResource && fhirResource.resourceType === 'QuestionnaireResponse') {
          console.log('[QuestionnaireForm] ===== EXTRACTION START =====');
          console.log('[QuestionnaireForm] Full fhirResource:', JSON.stringify(fhirResource, null, 2));
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
          setResponses(extractedResponses);
          onResponseChange?.(extractedResponses);
        }
        // Fallback: try to autofill from FHIR resource if provided
        else if (fhirResource && data.questionnaire) {
          console.log('[QuestionnaireForm] Starting fallback autofill...');
          console.log('[QuestionnaireForm] Patient data:', patientData);
          const autofilledResponses = mapFHIRToQuestionnaireResponses(
            fhirResource,
            data.questionnaire,
            patientData
          );
          console.log('[QuestionnaireForm] Autofilled responses:', autofilledResponses);
          setResponses(autofilledResponses);
          onResponseChange?.(autofilledResponses);
        } else {
          console.log('[QuestionnaireForm] No autofill - fhirResource:', !!fhirResource, 'questionnaire:', !!data.questionnaire);
        }
      } catch (err) {
        console.error('Error fetching questionnaire:', err);
        setError('Failed to load questionnaire form');
      } finally {
        setLoading(false);
      }
    }

    if (questionnaireId) {
      fetchQuestionnaire();
    }
  }, [questionnaireId, fhirResource]);

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
          />
        ))}
      </div>
    </div>
  );
}
