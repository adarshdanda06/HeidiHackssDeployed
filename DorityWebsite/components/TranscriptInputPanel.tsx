"use client";

import { FileText, Loader2, Sparkles, CheckCircle2 } from "lucide-react";
import { useSession } from "@/contexts/SessionContext";
import HeidiTranscriptFetcher from "./HeidiTranscriptFetcher";

export default function TranscriptInputPanel() {
  const {
    patient,
    transcript,
    setTranscript,
    analyzeTranscript,
    isLoading,
    error,
    clearError,
  } = useSession();

  const handleGenerateActions = async () => {
    clearError();
    await analyzeTranscript();
  };

  const handleTranscriptFetched = (fetchedTranscript: string, metadata?: any) => {
    setTranscript(fetchedTranscript);
    
    // Optional: Show a success message or metadata
    if (metadata) {
      console.log('Transcript metadata:', metadata);
    }
  };

  const isDisabled = !patient;
  const charCount = transcript.length;
  const hasContent = transcript.trim().length > 20; // Minimum threshold

  return (
    <div className="bg-white border border-zinc-200/70 rounded-2xl shadow-sm p-6 flex flex-col min-h-[500px]">
      {/* Header */}
      <div className="mb-4">
        <h2 className="text-sm font-semibold text-zinc-900 mb-1">Session Transcript</h2>
        <p className="text-xs text-zinc-600">
          Fetch from Heidi API or paste/type the consultation transcript
        </p>
      </div>

      {isDisabled && (
        <div className="bg-amber-50/50 border border-amber-200/50 rounded-xl p-3.5 mb-4 flex items-start gap-2.5">
          <FileText className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
          <div>
            <p className="text-sm font-medium text-amber-900">Select a patient to begin</p>
            <p className="text-xs text-amber-700 mt-0.5">
              Search and select a patient on the left first
            </p>
          </div>
        </div>
      )}

      {/* Heidi API Fetcher */}
      {!isDisabled && (
        <div className="mb-4">
          <HeidiTranscriptFetcher 
            onTranscriptFetched={handleTranscriptFetched}
            disabled={isDisabled || isLoading.analyze}
          />
        </div>
      )}

      {/* Divider */}
      {!isDisabled && (
        <div className="relative mb-4">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-zinc-200"></div>
          </div>
          <div className="relative flex justify-center text-xs uppercase">
            <span className="bg-white px-2 text-zinc-500">Or manually enter</span>
          </div>
        </div>
      )}

      <div className="flex-1 flex flex-col">
        <textarea
          disabled={isDisabled}
          value={transcript}
          onChange={(e) => setTranscript(e.target.value)}
          placeholder={
            isDisabled
              ? "Select a patient first…"
              : "Patient presents with complaints of increased thirst and frequent urination..."
          }
          className="flex-1 w-full p-4 text-sm bg-white border border-zinc-200/70 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#7C2D3E]/20 focus:border-[#7C2D3E]/30 resize-none disabled:bg-zinc-50 disabled:text-zinc-400 disabled:cursor-not-allowed shadow-inner transition-all placeholder:text-zinc-400"
        />

        <div className="flex items-center justify-between mt-3">
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">
              {charCount} character{charCount !== 1 ? "s" : ""}
            </span>
            {hasContent && !isDisabled && !isLoading.analyze && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200/50">
                <CheckCircle2 className="w-3 h-3" />
                AI ready
              </span>
            )}
          </div>

          <button
            onClick={handleGenerateActions}
            disabled={isDisabled || isLoading.analyze || !hasContent}
            className="px-4 py-2.5 text-sm font-semibold text-white bg-[#7C2D3E] hover:bg-[#5A1F2D] rounded-full shadow-sm transition-all disabled:bg-zinc-300 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isLoading.analyze ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analyzing…
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                Generate Actions
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="mt-3 bg-red-50 border border-red-200/50 rounded-xl p-3 text-sm text-red-900">
            {error}
          </div>
        )}

        {isLoading.analyze && (
          <div className="mt-3 bg-blue-50/50 border border-blue-200/50 rounded-xl p-3 flex items-center gap-2.5">
            <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
            <p className="text-sm text-blue-900">
              Analyzing transcript and generating draft orders…
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
