"use client";

import { useState } from "react";
import { Loader2, Inbox, CheckCircle } from "lucide-react";
import ActionCard from "./ActionCard";
import { useSession } from "@/contexts/SessionContext";

export default function ActionList() {
  const { suggestedActions, approvedActions, isLoading, applyApprovedActions, error, clearError } = useSession();
  const [applySuccess, setApplySuccess] = useState(false);

  const handleApplyActions = async () => {
    clearError();
    setApplySuccess(false);
    await applyApprovedActions();
    setApplySuccess(true);
    
    // Clear success message after 3 seconds
    setTimeout(() => setApplySuccess(false), 3000);
  };

  // Loading state
  if (isLoading.analyze) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-zinc-500">
        <Loader2 className="w-8 h-8 animate-spin mb-3" />
        <p className="text-sm font-medium">Generating draft orders from transcript…</p>
      </div>
    );
  }

  // Empty state
  if (suggestedActions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-zinc-400">
        <Inbox className="w-12 h-12 mb-3" />
        <p className="text-sm font-medium">No draft actions yet</p>
        <p className="text-xs mt-1">Generate actions from the transcript to begin</p>
      </div>
    );
  }

  // Group actions by category
  const groupedActions = suggestedActions
    .filter(action => action.type !== 'scheduling') // Hide scheduling actions as they are handled in Aftercare
    .reduce((acc, action) => {
    const category = action.categoryLabel || "Other";
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(action);
    return acc;
  }, {} as Record<string, typeof suggestedActions>);

  const canApply = approvedActions.length > 0 && !isLoading.apply;

  return (
    <div className="space-y-6">
      {/* Grouped Actions */}
      {Object.entries(groupedActions).map(([category, actions]) => (
        <div key={category}>
          <h3 className="text-sm font-semibold text-zinc-700 mb-3 px-1">
            {category}
          </h3>
          <div className="space-y-3">
            {actions.map((action) => (
              <ActionCard key={action.id} action={action} />
            ))}
          </div>
        </div>
      ))}

      {/* Apply Actions Button */}
      <div className="pt-6 border-t border-zinc-200/70">
        <button
          onClick={handleApplyActions}
          disabled={!canApply}
          className="w-full px-6 py-3.5 text-sm font-semibold text-white bg-[#7C2D3E] hover:bg-[#5A1F2D] rounded-full shadow-sm transition-all disabled:bg-zinc-300 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isLoading.apply ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Applying to EMR…
            </>
          ) : (
            <>
              <CheckCircle className="w-5 h-5" />
              Apply {approvedActions.length} Approved Action{approvedActions.length !== 1 ? "s" : ""} to EMR
            </>
          )}
        </button>

        {!canApply && approvedActions.length === 0 && (
          <p className="text-xs text-zinc-500 text-center mt-2">
            Approve at least one action to continue
          </p>
        )}
      </div>

      {/* Success Message */}
      {applySuccess && !error && (
        <div className="bg-emerald-50/50 border border-emerald-200/50 rounded-xl p-4 flex items-start gap-3">
          <CheckCircle className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-emerald-900">
              Actions applied successfully!
            </p>
            <p className="text-sm text-emerald-700 mt-1">
              {approvedActions.length} action{approvedActions.length !== 1 ? "s" : ""} sent to Medplum EMR
            </p>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200/50 rounded-xl p-4 text-sm text-red-900">
          {error}
        </div>
      )}
    </div>
  );
}
