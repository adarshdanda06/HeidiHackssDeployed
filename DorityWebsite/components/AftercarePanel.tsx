"use client";

import { useState } from "react";
import { Mail, Eye, Send, Loader2, CheckCircle, FileText, Sparkles } from "lucide-react";
import { useSession } from "@/contexts/SessionContext";

export default function AftercarePanel() {
  const {
    patient,
    approvedActions,
    aftercareSummary,
    setAftercareSummary,
    aftercareSubject,
    setAftercareSubject,
    generateAftercare,
    sendAftercare,
    isLoading,
    error,
    clearError,
  } = useSession();

  const [email, setEmail] = useState(patient?.insurance || ""); // Using insurance field as email placeholder
  const [showPreview, setShowPreview] = useState(false);
  const [sendSuccess, setSendSuccess] = useState(false);

  const handleGenerate = async () => {
    clearError();
    await generateAftercare();
  };

  const handleSendEmail = async () => {
    if (!email.trim()) {
      alert("Please enter an email address");
      return;
    }

    clearError();
    setSendSuccess(false);
    await sendAftercare(email);
    setSendSuccess(true);

    // Clear success message after 5 seconds
    setTimeout(() => setSendSuccess(false), 5000);
  };

  const hasAftercare = aftercareSummary.trim().length > 0;

  return (
    <div className="bg-white border border-zinc-200/70 rounded-2xl shadow-sm p-6">
      <div className="flex items-start justify-between mb-6">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <h2 className="text-lg font-semibold text-zinc-900">After-Care Summary</h2>
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-700 border border-blue-200/50">
              <FileText className="w-3 h-3 mr-1" />
              Note
            </span>
          </div>
          <p className="text-sm text-zinc-600">
            Generate and send a patient-friendly summary of approved actions
          </p>
        </div>
      </div>

      {/* Generate Button (if no summary yet) */}
      {!hasAftercare && (
        <div className="mb-6 text-center py-8 bg-zinc-50 rounded-xl border border-zinc-200/70">
          <FileText className="w-12 h-12 mx-auto mb-3 text-zinc-300" />
          <p className="text-sm font-medium text-zinc-600 mb-4">
            No after-care summary generated yet
          </p>
          <button
            onClick={handleGenerate}
            disabled={isLoading.aftercare || approvedActions.length === 0}
            className="px-5 py-2.5 text-sm font-semibold text-white bg-[#7C2D3E] hover:bg-[#5A1F2D] rounded-full shadow-sm transition-all disabled:bg-zinc-300 disabled:cursor-not-allowed inline-flex items-center gap-2"
          >
            {isLoading.aftercare ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Generating…
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                Generate from {approvedActions.length} Approved Action{approvedActions.length !== 1 ? "s" : ""}
              </>
            )}
          </button>
          {approvedActions.length === 0 && (
            <p className="text-xs text-zinc-500 mt-3">
              Approve at least one action to generate after-care summary
            </p>
          )}
        </div>
      )}

      {/* After-care content (if generated) */}
      {hasAftercare && (
        <>
          {/* Email Address */}
          <div className="mb-4">
            <label className="block text-sm font-semibold text-zinc-700 mb-2">
              Patient Email Address
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="adarsh.danda1@gmail.com"
              className="w-full px-3.5 py-2.5 bg-white border border-zinc-200/70 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#7C2D3E]/20 focus:border-[#7C2D3E]/30 text-sm transition-all placeholder:text-zinc-400"
            />
          </div>

          {/* Subject Line */}
          <div className="mb-4">
            <label className="block text-sm font-semibold text-zinc-700 mb-2">
              Subject Line
            </label>
            <input
              type="text"
              value={aftercareSubject}
              onChange={(e) => setAftercareSubject(e.target.value)}
              className="w-full px-3.5 py-2.5 bg-white border border-zinc-200/70 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#7C2D3E]/20 focus:border-[#7C2D3E]/30 text-sm transition-all placeholder:text-zinc-400"
            />
          </div>

          {/* Summary Text */}
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-semibold text-zinc-700">
                Summary Message
              </label>
              <button
                onClick={handleGenerate}
                disabled={isLoading.aftercare}
                className="text-xs font-medium text-[#7C2D3E] hover:text-[#5A1F2D] flex items-center gap-1"
              >
                <Sparkles className="w-3 h-3" />
                Regenerate
              </button>
            </div>
            <textarea
              value={aftercareSummary}
              onChange={(e) => setAftercareSummary(e.target.value)}
              rows={12}
              className="w-full px-4 py-3 bg-white border border-zinc-200/70 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#7C2D3E]/20 focus:border-[#7C2D3E]/30 text-sm font-mono resize-none shadow-inner transition-all"
            />
            <p className="text-xs text-zinc-500 mt-2">
              Edit the message above before sending
            </p>
          </div>

          {/* Buttons */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowPreview(!showPreview)}
              className="px-4 py-2 text-sm font-medium text-zinc-700 hover:text-zinc-900 hover:bg-zinc-100/80 rounded-lg transition-all flex items-center gap-2"
            >
              <Eye className="w-4 h-4" />
              {showPreview ? "Hide Preview" : "Preview Email"}
            </button>

            <button
              onClick={handleSendEmail}
              disabled={isLoading.aftercare || sendSuccess || !email.trim()}
              className="px-5 py-2 text-sm font-semibold text-white bg-[#7C2D3E] hover:bg-[#5A1F2D] rounded-full shadow-sm transition-all disabled:bg-zinc-300 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {isLoading.aftercare ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Sending…
                </>
              ) : sendSuccess ? (
                <>
                  <CheckCircle className="w-4 h-4" />
                  Sent!
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  Send After-Care Email
                </>
              )}
            </button>
          </div>

          {/* Success Message */}
          {sendSuccess && (
            <div className="mt-4 bg-emerald-50/50 border border-emerald-200/50 rounded-xl p-4 flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-semibold text-emerald-900">
                  After-care email sent successfully!
                </p>
                <p className="text-sm text-emerald-700 mt-1">
                  Email sent to <strong>{email}</strong>
                </p>
              </div>
            </div>
          )}

          {/* Email Preview */}
          {showPreview && (
            <div className="mt-6 border border-zinc-200/70 rounded-xl overflow-hidden">
              <div className="bg-zinc-50 px-4 py-2.5 border-b border-zinc-200/70">
                <p className="text-xs font-semibold text-zinc-700">Email Preview</p>
              </div>
              <div className="bg-white p-6">
                <div className="mb-4 text-sm text-zinc-600 space-y-1">
                  <p>
                    <strong>To:</strong> {email || "(no email provided)"}
                  </p>
                  <p>
                    <strong>From:</strong> noreply@clinicalactionlayer.com
                  </p>
                  <p>
                    <strong>Subject:</strong> {aftercareSubject || "Your Visit Summary"}
                  </p>
                </div>
                <div className="border-t border-zinc-200/70 pt-4">
                  <div className="whitespace-pre-wrap text-sm text-zinc-800 font-sans leading-relaxed">
                    {aftercareSummary}
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* Error Message */}
      {error && (
        <div className="mt-4 bg-red-50 border border-red-200/50 rounded-xl p-4 text-sm text-red-900">
          {error}
        </div>
      )}
    </div>
  );
}
