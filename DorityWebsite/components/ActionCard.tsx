"use client";

import { useState } from "react";
import { Pill, Stethoscope, Image, FlaskConical, UserPlus, Calendar, FileText, AlertTriangle, CheckCircle2, Check, X, FileEdit, XCircle, Mail, Send } from "lucide-react";
import { useSession, type SuggestedAction } from "@/contexts/SessionContext";
import QuestionnaireForm from "./QuestionnaireForm";

interface ActionCardProps {
  action: SuggestedAction;
}

// Calculate completion percentage based on available fields
function calculateCompletionPercentage(action: SuggestedAction): number {
  // If scheduling, always 100% if essential fields exist
  if (action.type === 'scheduling') return 100;

  const fields = [
    action.title,
    action.details,
    action.rationale,
    action.doseInfo,
    action.pharmacy,
    action.safetyFlag,
  ];
  const filledFields = fields.filter(f => f && f !== "").length;
  return Math.round((filledFields / fields.length) * 100);
}

// Form field component for editable fields
function FormField({ 
  label, 
  value, 
  isItalic = false, 
  isAlert = false,
  isEditable = false,
  onChange,
  fieldType = 'text'
}: { 
  label: string; 
  value: string; 
  isItalic?: boolean;
  isAlert?: boolean;
  isEditable?: boolean;
  onChange?: (value: string) => void;
  fieldType?: 'text' | 'textarea' | 'select';
}) {
  if (isEditable) {
    return (
      <div className={`p-2.5 bg-white border border-zinc-200/50 rounded-lg ${isAlert ? 'border-amber-300 bg-amber-50/30' : ''}`}>
        <label className="block text-xs font-semibold text-zinc-700 mb-1.5">{label}</label>
        {fieldType === 'textarea' ? (
          <textarea
            value={value}
            onChange={(e) => onChange?.(e.target.value)}
            className="w-full text-xs text-zinc-600 border border-zinc-200 rounded px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-[#7C2D3E]/20 focus:border-[#7C2D3E]/30 resize-none"
            rows={2}
          />
        ) : (
          <input
            type="text"
            value={value}
            onChange={(e) => onChange?.(e.target.value)}
            className="w-full text-xs text-zinc-600 border border-zinc-200 rounded px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-[#7C2D3E]/20 focus:border-[#7C2D3E]/30"
          />
        )}
      </div>
    );
  }

  return (
    <div className={`p-2.5 bg-white border border-zinc-200/50 rounded-lg ${isAlert ? 'border-amber-300 bg-amber-50/30' : ''}`}>
      <label className="block text-xs font-semibold text-zinc-700 mb-1">{label}</label>
      <p className={`text-xs text-zinc-600 ${isItalic ? 'italic' : ''} ${isAlert ? 'text-amber-900 font-medium' : ''}`}>
        {value}
      </p>
    </div>
  );
}

const TYPE_ICONS = {
  medication: Pill,
  imaging: Image,
  lab: FlaskConical,
  referral: UserPlus,
  followup: Calendar,
  aftercare: FileText,
  scheduling: Mail,
};

const TYPE_COLORS = {
  medication: "bg-blue-100 text-blue-600",
  imaging: "bg-purple-100 text-purple-600",
  lab: "bg-green-100 text-green-600",
  referral: "bg-orange-100 text-orange-600",
  followup: "bg-cyan-100 text-cyan-600",
  aftercare: "bg-pink-100 text-pink-600",
  scheduling: "bg-indigo-100 text-indigo-600",
};

const SAFETY_COLORS = {
  high: "bg-red-100 text-red-700 border-red-200/50",
  medium: "bg-amber-100 text-amber-700 border-amber-200/50",
  low: "bg-emerald-100 text-emerald-700 border-emerald-200/50",
};

export default function ActionCard({ action }: ActionCardProps) {
  const { updateActionStatus, updateAction } = useSession();
  const [showForm, setShowForm] = useState(false);
  const [showFormModal, setShowFormModal] = useState(false);
  const [isEditingForm, setIsEditingForm] = useState(false);
  
  // Initialize form data, including default email for scheduling
  const [formData, setFormData] = useState<any>(() => {
    if (action.type === 'scheduling') {
      return {
        ...action.fhirPreview,
        email: action.email || 'adarsh.danda1@gmail.com',
        subject: action.title ? `New Meeting Request: ${action.title}` : 'New Meeting Request',
        body: `A new follow-up meeting has been suggested.\n\nReason: ${action.reason || action.rationale}\nSuggested Time: ${action.when || 'As soon as possible'}`
      };
    }
    return action.fhirPreview;
  });

  const handleApprove = () => {
    // If scheduling, save the current formData (with edited email) before approving
    if (action.type === 'scheduling') {
      updateAction(action.id, {
        email: formData.email,
        title: formData.subject?.replace('New Meeting Request: ', '') || action.title,
        // Add other fields if needed for the backend
        // backend 'apply-actions' will use action.email, action.subject, action.body if present
        ...formData
      });
    }
    updateActionStatus(action.id, "approved");
  };

  const handleReject = () => {
    updateActionStatus(action.id, "rejected");
  };

  const handleSaveForm = () => {
    updateAction(action.id, { 
        fhirPreview: formData, // Keep fhirPreview updated
        // For scheduling, we also update top-level props if changed
        ...(action.type === 'scheduling' ? {
            email: formData.email,
            // potentially other fields
        } : {})
    });
    setIsEditingForm(false);
    setShowFormModal(false);
  };

  const handleOpenFormModal = () => {
    setShowFormModal(true);
    setIsEditingForm(true);
  };

  const Icon = TYPE_ICONS[action.type] || FileText;
  const isApproved = action.status === "approved";
  const isRejected = action.status === "rejected";
  const completionPercentage = calculateCompletionPercentage(action);

  if (isRejected) {
    return (
      <div className="bg-zinc-50 border border-zinc-200/70 rounded-2xl shadow-sm opacity-60">
        <div className="p-4 flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 bg-zinc-200 text-zinc-400">
            <Icon className="w-5 h-5" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="text-sm font-semibold text-zinc-500 line-through truncate">
                {action.title}
              </h3>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-zinc-200 text-zinc-600 text-xs font-semibold rounded-full">
                <X className="w-3 h-3" />
                Rejected
              </span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
    <div
      className={`group bg-white border border-zinc-200/70 rounded-2xl shadow-sm transition-all hover:shadow-md ${
        isApproved ? "bg-gradient-to-r from-emerald-50/30 to-white border-emerald-300" : "hover:-translate-y-0.5"
      }`}
    >
      <div className="p-4 flex items-start gap-4">
        {/* Left: Icon */}
        <div
          className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 transition-colors ${
            isApproved ? "bg-emerald-100 text-emerald-600" : TYPE_COLORS[action.type]
          }`}
        >
          <Icon className="w-5 h-5" />
        </div>

        {/* Middle: Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between mb-2">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <h3 className="text-sm font-semibold text-zinc-900">
                  {action.title}
                </h3>
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-zinc-100 text-zinc-700 border border-zinc-200/50 capitalize">
                  {action.type}
                </span>
                {isApproved && (
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-emerald-100 text-emerald-700 text-xs font-semibold rounded-full border border-emerald-200/50">
                    <Check className="w-3 h-3" />
                    {action.type === 'scheduling' ? 'Sent' : 'Approved'}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-3 mb-1">
                <p className="text-xs text-zinc-600">
                  {action.details}
                </p>
              </div>
              {/* Questionnaire Form Indicator */}
              {action.questionnaireName && (
                <div className="flex items-center gap-1.5 mt-1.5 text-xs">
                  <FileText className="w-3 h-3 text-zinc-400" />
                  <span className="text-zinc-500">
                    Using form: <span className="font-medium text-zinc-700">{action.questionnaireName}</span>
                  </span>
                </div>
              )}
              {/* Completion Progress Bar */}
              <div className="flex items-center gap-2 mt-2">
                <div className="flex-1 bg-zinc-100 rounded-full h-1.5 overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-300 ${
                      completionPercentage === 100 ? 'bg-emerald-500' : 
                      completionPercentage >= 75 ? 'bg-blue-500' : 
                      completionPercentage >= 50 ? 'bg-amber-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${completionPercentage}%` }}
                  />
                </div>
                <span className="text-xs font-medium text-zinc-600 tabular-nums">
                  {completionPercentage}%
                </span>
              </div>
              {action.doseInfo && (
                <p className="text-xs text-zinc-500 mt-2">
                  <span className="font-semibold">Dose:</span> {action.doseInfo}
                </p>
              )}
              {action.pharmacy && (
                <p className="text-xs text-zinc-500 mt-1">
                  <span className="font-semibold">Pharmacy:</span> {action.pharmacy}
                </p>
              )}
            </div>
          </div>

          {/* Rationale */}
          {action.rationale && (
            <div className="text-xs text-zinc-500 italic mt-2 pl-3 border-l-2 border-zinc-200">
              "{action.rationale}"
            </div>
          )}

          {/* Safety Flag */}
          {action.safetyFlag && action.safetyMessage && (
            <div className={`mt-3 px-3 py-2 rounded-lg text-xs flex items-start gap-2 ${SAFETY_COLORS[action.safetyFlag]}`}>
              <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
              <span className="font-medium">{action.safetyMessage}</span>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex items-center gap-2 mt-4">
            {/* Special View/Edit buttons for Scheduling */}
            {action.type === 'scheduling' ? (
                <>
                    <button
                    onClick={() => setShowForm(!showForm)}
                    className="px-3 py-1.5 text-xs font-medium text-zinc-700 hover:text-zinc-900 hover:bg-zinc-100/80 rounded-lg transition-all border border-zinc-200/70 flex items-center gap-1.5"
                    >
                    <Mail className="w-3.5 h-3.5" />
                    {showForm ? "Hide" : "View"} Email
                    </button>
                    <button
                    onClick={handleOpenFormModal}
                    className="px-3 py-1.5 text-xs font-medium text-[#7C2D3E] hover:text-white hover:bg-[#7C2D3E] rounded-lg transition-all border border-[#7C2D3E] flex items-center gap-1.5"
                    >
                    <FileEdit className="w-3.5 h-3.5" />
                    Edit Email
                    </button>
                </>
            ) : (
                <>
                    <button
                    onClick={() => setShowForm(!showForm)}
                    className="px-3 py-1.5 text-xs font-medium text-zinc-700 hover:text-zinc-900 hover:bg-zinc-100/80 rounded-lg transition-all border border-zinc-200/70 flex items-center gap-1.5"
                    >
                    <FileEdit className="w-3.5 h-3.5" />
                    {showForm ? "Hide" : "View"} Form
                    </button>
                    <button
                    onClick={handleOpenFormModal}
                    className="px-3 py-1.5 text-xs font-medium text-[#7C2D3E] hover:text-white hover:bg-[#7C2D3E] rounded-lg transition-all border border-[#7C2D3E] flex items-center gap-1.5"
                    >
                    <FileEdit className="w-3.5 h-3.5" />
                    Edit Form
                    </button>
                </>
            )}
            
            {!isApproved && (
              <>
                <button
                  onClick={handleReject}
                  className="px-3 py-1.5 text-xs font-medium text-zinc-700 hover:text-zinc-900 hover:bg-zinc-100/80 rounded-lg transition-all border border-zinc-200/70"
                >
                  Reject
                </button>
                {/* Send button for Scheduling, Approve & Sign for others */}
                {action.type === 'scheduling' ? (
                    <button
                        onClick={handleApprove}
                        className="px-4 py-1.5 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 rounded-full shadow-sm transition-all flex items-center gap-1.5"
                    >
                        <Send className="w-3.5 h-3.5" />
                        Send
                    </button>
                ) : (
                    <button
                        onClick={handleApprove}
                        className="px-4 py-1.5 text-xs font-semibold text-white bg-[#7C2D3E] hover:bg-[#5A1F2D] rounded-full shadow-sm transition-all flex items-center gap-1.5"
                    >
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        Approve & Sign
                    </button>
                )}
              </>
            )}
          </div>

          {/* Form Details (Expandable) - Read Only Preview */}
          {showForm && (
            <div className="mt-4 p-4 bg-zinc-50/50 border border-zinc-200/70 rounded-xl space-y-4">
              {/* If questionnaire exists, use dynamic form */}
              {action.questionnaireId ? (
                <QuestionnaireForm 
                  questionnaireId={action.questionnaireId}
                  isEditable={false}
                  initialResponses={{}}
                />
              ) : (
                <>
                  <h4 className="text-sm font-semibold text-zinc-900 mb-3 flex items-center gap-2">
                    {action.type === 'scheduling' ? (
                        <>
                            <Mail className="w-4 h-4" />
                            Email Draft
                        </>
                    ) : (
                        <>
                            <FileText className="w-4 h-4" />
                            {action.type === 'medication' && 'Medication Order Form'}
                            {action.type === 'imaging' && 'Imaging Request Form'}
                            {action.type === 'lab' && 'Laboratory Order Form'}
                            {action.type === 'referral' && 'Referral Form'}
                            {action.type === 'followup' && 'Follow-up Appointment Form'}
                            {action.type === 'aftercare' && 'After-care Instructions Form'}
                        </>
                    )}
                  </h4>

                  {/* Scheduling-specific fields */}
                  {action.type === 'scheduling' && (
                    <div className="space-y-3">
                      <FormField label="To" value={formData.email || 'adarsh.danda1@gmail.com'} />
                      <FormField label="Subject" value={formData.subject || `New Meeting Request: ${action.title}`} />
                      <FormField label="Message" value={formData.body} isItalic fieldType="textarea" />
                    </div>
                  )}

                  {/* Medication-specific fields */}
                  {action.type === 'medication' && (
                <div className="space-y-3">
                  <FormField 
                    label="Medication Name *" 
                    value={formData.medicationCodeableConcept?.text || action.title} 
                    isEditable={false}
                  />
                  <FormField 
                    label="RxNorm Code" 
                    value={formData.medicationCodeableConcept?.coding?.[0]?.code || 'Not specified'} 
                    isEditable={false}
                  />
                  <FormField 
                    label="Dosage & Instructions *" 
                    value={formData.dosageInstruction?.[0]?.text || action.doseInfo || 'Not specified'} 
                    isEditable={false}
                  />
                  <FormField 
                    label="Route of Administration" 
                    value={formData.dosageInstruction?.[0]?.route?.text || 'Oral'} 
                    isEditable={false}
                  />
                  <FormField 
                    label="Dose Quantity" 
                    value={formData.dosageInstruction?.[0]?.doseAndRate?.[0]?.doseQuantity?.value 
                      ? `${formData.dosageInstruction[0].doseAndRate[0].doseQuantity.value} ${formData.dosageInstruction[0].doseAndRate[0].doseQuantity.unit || 'mg'}`
                      : 'See instructions'} 
                    isEditable={false}
                  />
                  <FormField 
                    label="Frequency" 
                    value={formData.dosageInstruction?.[0]?.timing?.repeat?.frequency 
                      ? `${formData.dosageInstruction[0].timing.repeat.frequency}x per ${formData.dosageInstruction[0].timing.repeat.periodUnit || 'day'}`
                      : 'As directed'} 
                    isEditable={false}
                  />
                  <FormField 
                    label="Dispense Quantity" 
                    value={formData.dispenseRequest?.quantity?.value || '30-day supply'} 
                    isEditable={false}
                  />
                  <FormField 
                    label="Number of Refills" 
                    value={formData.dispenseRequest?.numberOfRepeatsAllowed?.toString() || '3'} 
                    isEditable={false}
                  />
                  <FormField 
                    label="Preferred Pharmacy" 
                    value={action.pharmacy || 'Patient\'s preferred pharmacy'} 
                    isEditable={false}
                  />
                  <FormField 
                    label="Substitution Permitted" 
                    value={formData.substitution?.allowedBoolean !== false ? 'Yes (generic permitted)' : 'No (dispense as written)'} 
                    isEditable={isEditingForm}
                  />
                  <FormField 
                    label="Clinical Indication" 
                    value={formData.reasonCode?.[0]?.text || action.rationale} 
                    isItalic 
                    isEditable={isEditingForm}
                    fieldType="textarea"
                  />
                  {action.safetyFlag && (
                    <FormField 
                      label="Safety Alert" 
                      value={action.safetyMessage || ''} 
                      isAlert 
                    />
                  )}
                </div>
              )}

              {/* Imaging-specific fields */}
              {action.type === 'imaging' && (
                <div className="space-y-3">
                  <FormField 
                    label="Imaging Study *" 
                    value={formData.code?.text || action.title} 
                    isEditable={false}
                  />
                  <FormField 
                    label="LOINC Code" 
                    value={formData.code?.coding?.[0]?.code || 'Not specified'} 
                    isEditable={false}
                  />
                  <FormField 
                    label="Body Site / Region *" 
                    value={formData.bodySite?.[0]?.text || action.details} 
                    isEditable={false}
                  />
                  <FormField 
                    label="Priority *" 
                    value={formData.priority || (action.title.includes('STAT') ? 'stat' : 'routine')} 
                    isEditable={false}
                  />
                  <FormField 
                    label="Clinical Indication *" 
                    value={formData.reasonCode?.[0]?.text || action.rationale} 
                    isItalic 
                    isEditable={false}
                  />
                  <FormField 
                    label="Relevant Clinical History" 
                    value={formData.supportingInfo || 'See patient chart for complete history'} 
                    isEditable={false}
                  />
                  <FormField 
                    label="Special Instructions" 
                    value={formData.patientInstruction || 'Patient should arrive 30 minutes early'} 
                    isEditable={false}
                  />
                  {action.safetyFlag && (
                    <FormField 
                      label="Safety Considerations" 
                      value={action.safetyMessage || ''} 
                      isAlert 
                    />
                  )}
                </div>
              )}

              {/* Lab-specific fields */}
              {action.type === 'lab' && (
                <div className="space-y-3">
                  <FormField label="Test Name" value={action.title} />
                  <FormField label="Clinical Indication" value={action.details} />
                  <FormField label="Priority" value="Routine" />
                  <FormField label="Specimen Type" value="Blood/Serum" />
                  <FormField label="Collection Method" value="Venipuncture" />
                  <FormField label="Fasting Required" value="No (unless otherwise noted)" />
                  <FormField label="Special Handling" value="Standard processing" />
                  <FormField label="Expected Results Timeline" value="3-5 business days" />
                  <FormField label="Clinical Rationale" value={action.rationale} isItalic />
                  <FormField label="Notify Provider When" value="Results available" />
                </div>
              )}

              {/* Referral-specific fields */}
              {action.type === 'referral' && (
                <div className="space-y-3">
                  <FormField label="Specialty / Service" value={action.title} />
                  <FormField label="Reason for Referral" value={action.details} />
                  <FormField label="Urgency" value="Routine (within 2-4 weeks)" />
                  <FormField label="Specific Provider Requested" value="No preference" />
                  <FormField label="Clinical Summary" value={action.rationale} isItalic />
                  <FormField label="Relevant Diagnoses" value="See patient problem list" />
                  <FormField label="Current Medications" value="See current medication list" />
                  <FormField label="Tests to Send" value="Include recent lab results and imaging" />
                  <FormField label="Expected Consultation Type" value="Initial consultation" />
                  <FormField label="Insurance Authorization" value="May be required - check with patient's plan" />
                </div>
              )}

              {/* Follow-up-specific fields */}
              {action.type === 'followup' && (
                <div className="space-y-3">
                  <FormField label="Appointment Type" value={action.title} />
                  <FormField label="Reason for Visit" value={action.details} />
                  <FormField label="Timeframe" value={action.title.includes('1 week') ? '1 week' : action.title.includes('2 weeks') ? '2 weeks' : '1 month'} />
                  <FormField label="Visit Duration" value="30 minutes" />
                  <FormField label="Provider" value="Same provider" />
                  <FormField label="Clinical Notes" value={action.rationale} isItalic />
                  <FormField label="Items to Review" value="Symptom progression, treatment response" />
                  <FormField label="Tests Before Visit" value="None required (unless clinically indicated)" />
                </div>
              )}

              {/* After-care-specific fields */}
              {action.type === 'aftercare' && (
                <div className="space-y-3">
                  <FormField label="Instruction Type" value={action.title} />
                  <FormField label="Summary" value={action.details} />
                  <FormField label="Clinical Context" value={action.rationale} isItalic />
                  <FormField label="Delivery Method" value="Email and printed handout" />
                  <FormField label="Language" value="English (translate as needed)" />
                  <FormField label="Follow-up Contact" value="Patient may call with questions" />
                </div>
              )}

              {/* FHIR Resource Info */}
              {!action.questionnaireId && (
                <div className="mt-4 pt-4 border-t border-zinc-200/70">
                  <h5 className="text-xs font-semibold text-zinc-700 mb-2">FHIR Resource Details</h5>
                  <div className="space-y-1.5 text-xs">
                    <div className="flex items-start gap-2">
                      <span className="font-semibold text-zinc-600 min-w-[120px]">Resource Type:</span>
                      <span className="text-zinc-500">{action.fhirPreview.resourceType}</span>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="font-semibold text-zinc-600 min-w-[120px]">Status:</span>
                      <span className="text-zinc-500">{action.fhirPreview.status}</span>
                    </div>
                  </div>
                </div>
              )}
              </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>

      {/* Form Edit Modal - Render outside card */}
      {showFormModal && (
        <div 
          className="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm"
          style={{ pointerEvents: 'auto' }}
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setShowFormModal(false);
              setIsEditingForm(false);
              // Reset form data if cancelling? Or keep edits? 
              // Usually cancel discards edits, but here we just close.
              // Let's keep it simple and just close.
            }
          }}
        >
          <div 
            className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[85vh] overflow-hidden flex flex-col relative"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="px-6 py-4 border-b border-zinc-200 flex items-center justify-between bg-gradient-to-r from-[#F7F1EC] to-white">
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${TYPE_COLORS[action.type]}`}>
                  <Icon className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-zinc-900">
                    {action.type === 'medication' && 'Edit Medication Order'}
                    {action.type === 'imaging' && 'Edit Imaging Request'}
                    {action.type === 'lab' && 'Edit Laboratory Order'}
                    {action.type === 'referral' && 'Edit Referral'}
                    {action.type === 'followup' && 'Edit Follow-up Appointment'}
                    {action.type === 'aftercare' && 'Edit After-care Instructions'}
                    {action.type === 'scheduling' && 'Edit Email Draft'}
                  </h3>
                  <p className="text-xs text-zinc-600 mt-0.5">{action.title}</p>
                </div>
              </div>
              <button
                onClick={() => {
                  setShowFormModal(false);
                  setIsEditingForm(false);
                  // setFormData(action.fhirPreview); // reset to original
                }}
                className="p-2 hover:bg-zinc-100 rounded-lg transition-colors"
              >
                <XCircle className="w-5 h-5 text-zinc-500" />
              </button>
            </div>

            {/* Modal Body - Scrollable Form */}
            <div className="flex-1 overflow-y-auto p-6">
              <div className="space-y-4">
                {/* Use dynamic questionnaire if available */}
                {action.questionnaireId ? (
                  <QuestionnaireForm 
                    questionnaireId={action.questionnaireId}
                    isEditable={true}
                    initialResponses={{}}
                    onResponseChange={(responses) => {
                      console.log('Questionnaire responses:', responses);
                      // TODO: Save responses to action
                    }}
                  />
                ) : (
                  <>
                    {/* Scheduling Form */}
                    {action.type === 'scheduling' && (
                        <div className="space-y-3">
                            <FormField 
                                label="To *" 
                                value={formData.email} 
                                isEditable={true}
                                onChange={(val) => setFormData({...formData, email: val})}
                            />
                            <FormField 
                                label="Subject *" 
                                value={formData.subject} 
                                isEditable={true}
                                onChange={(val) => setFormData({...formData, subject: val})}
                            />
                            <FormField 
                                label="Message *" 
                                value={formData.body} 
                                isEditable={true}
                                fieldType="textarea"
                                onChange={(val) => setFormData({...formData, body: val})}
                            />
                        </div>
                    )}

                    {/* Medication Form */}
                    {action.type === 'medication' && (
                  <div className="space-y-3">
                    <FormField 
                      label="Medication Name *" 
                      value={formData.medicationCodeableConcept?.text || action.title} 
                      isEditable={true}
                      onChange={(val) => setFormData({...formData, medicationCodeableConcept: {...formData.medicationCodeableConcept, text: val}})}
                    />
                    <FormField 
                      label="RxNorm Code" 
                      value={formData.medicationCodeableConcept?.coding?.[0]?.code || 'Not specified'} 
                      isEditable={true}
                      onChange={(val) => setFormData({
                        ...formData, 
                        medicationCodeableConcept: {
                          ...formData.medicationCodeableConcept,
                          coding: [{...formData.medicationCodeableConcept?.coding?.[0], code: val}]
                        }
                      })}
                    />
                    <FormField 
                      label="Dosage & Instructions *" 
                      value={formData.dosageInstruction?.[0]?.text || action.doseInfo || 'Not specified'} 
                      isEditable={true}
                      fieldType="textarea"
                      onChange={(val) => setFormData({
                        ...formData,
                        dosageInstruction: [{...formData.dosageInstruction?.[0], text: val}]
                      })}
                    />
                    <FormField 
                      label="Route of Administration" 
                      value={formData.dosageInstruction?.[0]?.route?.text || 'Oral'} 
                      isEditable={true}
                    />
                    <FormField 
                      label="Dose Quantity" 
                      value={formData.dosageInstruction?.[0]?.doseAndRate?.[0]?.doseQuantity?.value 
                        ? `${formData.dosageInstruction[0].doseAndRate[0].doseQuantity.value} ${formData.dosageInstruction[0].doseAndRate[0].doseQuantity.unit || 'mg'}`
                        : 'See instructions'} 
                      isEditable={true}
                    />
                    <FormField 
                      label="Dispense Quantity" 
                      value={formData.dispenseRequest?.quantity?.value || '30-day supply'} 
                      isEditable={true}
                    />
                    <FormField 
                      label="Number of Refills" 
                      value={formData.dispenseRequest?.numberOfRepeatsAllowed?.toString() || '3'} 
                      isEditable={true}
                    />
                    <FormField 
                      label="Preferred Pharmacy" 
                      value={action.pharmacy || 'Patient\'s preferred pharmacy'} 
                      isEditable={true}
                    />
                    <FormField 
                      label="Clinical Indication" 
                      value={formData.reasonCode?.[0]?.text || action.rationale} 
                      isItalic 
                      isEditable={true}
                      fieldType="textarea"
                    />
                    {action.safetyFlag && (
                      <FormField 
                        label="Safety Alert" 
                        value={action.safetyMessage || ''} 
                        isAlert 
                      />
                    )}
                  </div>
                )}

                {/* Imaging Form */}
                {action.type === 'imaging' && (
                  <div className="space-y-3">
                    <FormField 
                      label="Imaging Study *" 
                      value={formData.code?.text || action.title} 
                      isEditable={true}
                      onChange={(val) => setFormData({...formData, code: {...formData.code, text: val}})}
                    />
                    <FormField 
                      label="LOINC Code" 
                      value={formData.code?.coding?.[0]?.code || 'Not specified'} 
                      isEditable={true}
                    />
                    <FormField 
                      label="Body Site / Region *" 
                      value={formData.bodySite?.[0]?.text || action.details} 
                      isEditable={true}
                      onChange={(val) => setFormData({...formData, bodySite: [{text: val}]})}
                    />
                    <FormField 
                      label="Priority *" 
                      value={formData.priority || (action.title.includes('STAT') ? 'stat' : 'routine')} 
                      isEditable={true}
                    />
                    <FormField 
                      label="Clinical Indication *" 
                      value={formData.reasonCode?.[0]?.text || action.rationale} 
                      isItalic 
                      isEditable={true}
                      fieldType="textarea"
                    />
                    <FormField 
                      label="Relevant Clinical History" 
                      value={formData.supportingInfo || 'See patient chart for complete history'} 
                      isEditable={true}
                      fieldType="textarea"
                    />
                    <FormField 
                      label="Special Instructions" 
                      value={formData.patientInstruction || 'Patient should arrive 30 minutes early'} 
                      isEditable={true}
                      fieldType="textarea"
                    />
                    {action.safetyFlag && (
                      <FormField 
                        label="Safety Considerations" 
                        value={action.safetyMessage || ''} 
                        isAlert 
                      />
                    )}
                  </div>
                )}

                {/* Lab Form */}
                {action.type === 'lab' && (
                  <div className="space-y-3">
                    <FormField label="Test Name" value={action.title} isEditable={true} />
                    <FormField label="Clinical Indication" value={action.details} isEditable={true} fieldType="textarea" />
                    <FormField label="Priority" value="Routine" isEditable={true} />
                    <FormField label="Specimen Type" value="Blood/Serum" isEditable={true} />
                    <FormField label="Clinical Rationale" value={action.rationale} isItalic isEditable={true} fieldType="textarea" />
                  </div>
                )}

                {/* Referral Form */}
                {action.type === 'referral' && (
                  <div className="space-y-3">
                    <FormField label="Specialty / Service" value={action.title} isEditable={true} />
                    <FormField label="Reason for Referral" value={action.details} isEditable={true} fieldType="textarea" />
                    <FormField label="Urgency" value="Routine (within 2-4 weeks)" isEditable={true} />
                    <FormField label="Clinical Summary" value={action.rationale} isItalic isEditable={true} fieldType="textarea" />
                  </div>
                )}

                {/* Follow-up Form */}
                {action.type === 'followup' && (
                  <div className="space-y-3">
                    <FormField label="Appointment Type" value={action.title} isEditable={true} />
                    <FormField label="Reason for Visit" value={action.details} isEditable={true} fieldType="textarea" />
                    <FormField label="Timeframe" value={action.title.includes('1 week') ? '1 week' : action.title.includes('2 weeks') ? '2 weeks' : '1 month'} isEditable={true} />
                    <FormField label="Clinical Notes" value={action.rationale} isItalic isEditable={true} fieldType="textarea" />
                  </div>
                )}
                </>
                )}
              </div>
            </div>

            {/* Modal Footer */}
            <div className="px-6 py-4 border-t border-zinc-200 flex items-center justify-between bg-zinc-50">
              <p className="text-xs text-zinc-500">Fields marked with * are required</p>
              <div className="flex items-center gap-2">
                <button 
                  onClick={() => {
                    setShowFormModal(false);
                    setIsEditingForm(false);
                    // setFormData(action.fhirPreview);
                  }}
                  className="px-4 py-2 text-xs font-medium text-zinc-700 hover:bg-zinc-200 rounded-lg transition-all border border-zinc-300"
                >
                  Cancel
                </button>
                <button 
                  onClick={handleSaveForm}
                  className="px-6 py-2 text-xs font-semibold text-white bg-[#7C2D3E] hover:bg-[#5A1F2D] rounded-lg transition-all shadow-sm flex items-center gap-1.5"
                >
                  <Check className="w-4 h-4" />
                  Save Changes
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
