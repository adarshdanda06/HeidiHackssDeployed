"use client";

import { useState } from "react";
import { Download, Loader2, AlertCircle, Check, ExternalLink } from "lucide-react";

interface HeidiSession {
  id: string;
  label: string;
}

interface HeidiTranscriptFetcherProps {
  onTranscriptFetched: (transcript: string, metadata?: any) => void;
  disabled?: boolean;
}

export default function HeidiTranscriptFetcher({ onTranscriptFetched, disabled }: HeidiTranscriptFetcherProps) {
  const [sessions, setSessions] = useState<HeidiSession[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string>("");
  const [isLoadingSessions, setIsLoadingSessions] = useState(false);
  const [isLoadingTranscript, setIsLoadingTranscript] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const loadSessions = async () => {
    setIsLoadingSessions(true);
    setError(null);
    
    try {
      const response = await fetch('/api/heidi/sessions');
      
      if (!response.ok) {
        throw new Error('Failed to fetch Heidi sessions');
      }

      const data = await response.json();
      
      if (data.success && data.sessions) {
        setSessions(data.sessions);
        if (data.sessions.length > 0) {
          setSelectedSessionId(data.sessions[0].id);
        }
      } else {
        throw new Error(data.error || 'No sessions available');
      }
      
    } catch (err) {
      console.error('Error loading sessions:', err);
      setError(err instanceof Error ? err.message : 'Failed to load sessions');
    } finally {
      setIsLoadingSessions(false);
    }
  };

  const fetchTranscript = async () => {
    if (!selectedSessionId) {
      setError('Please select a session');
      return;
    }

    setIsLoadingTranscript(true);
    setError(null);
    setSuccess(false);
    
    try {
      const response = await fetch('/api/heidi/transcript', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sessionId: selectedSessionId }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to fetch transcript');
      }

      const data = await response.json();
      
      // API returns data.data.transcript (the endpoint wraps in { success, data })
      if (data.success && data.data && data.data.transcript) {
        onTranscriptFetched(data.data.transcript, data.data);
        setSuccess(true);
        
        // Clear success message after 3 seconds
        setTimeout(() => setSuccess(false), 3000);
      } else {
        console.error('Unexpected response format:', data);
        throw new Error('No transcript data received');
      }
      
    } catch (err) {
      console.error('Error fetching transcript:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch transcript');
    } finally {
      setIsLoadingTranscript(false);
    }
  };

  // Auto-load sessions on mount
  useState(() => {
    if (!disabled) {
      loadSessions();
    }
  });

  return (
    <div className="bg-gradient-to-br from-purple-50 to-pink-50 border border-purple-200/50 rounded-xl p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ExternalLink className="w-4 h-4 text-purple-600" />
          <h3 className="text-sm font-semibold text-purple-900">Fetch from Heidi API</h3>
        </div>
        {success && (
          <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200/50">
            <Check className="w-3 h-3" />
            Loaded
          </span>
        )}
      </div>

      <p className="text-xs text-purple-700">
        Load a consultation transcript from Heidi Health's API
      </p>

      {sessions.length === 0 && !isLoadingSessions && (
        <button
          onClick={loadSessions}
          disabled={disabled || isLoadingSessions}
          className="w-full px-3 py-2 text-sm font-medium text-purple-700 bg-white hover:bg-purple-50 border border-purple-200 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isLoadingSessions ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading sessions...
            </>
          ) : (
            <>
              <Download className="w-4 h-4" />
              Load Available Sessions
            </>
          )}
        </button>
      )}

      {sessions.length > 0 && (
        <div className="space-y-2">
          <label className="block text-xs font-medium text-purple-900">
            Select Session
          </label>
          <select
            value={selectedSessionId}
            onChange={(e) => setSelectedSessionId(e.target.value)}
            disabled={disabled || isLoadingTranscript}
            className="w-full px-3 py-2 text-sm bg-white border border-purple-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500/20 focus:border-purple-500/30 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {sessions.map((session) => (
              <option key={session.id} value={session.id}>
                {session.label} ({session.id.substring(0, 20)}...)
              </option>
            ))}
          </select>

          <button
            onClick={fetchTranscript}
            disabled={disabled || isLoadingTranscript || !selectedSessionId}
            className="w-full px-3 py-2.5 text-sm font-semibold text-white bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-sm"
          >
            {isLoadingTranscript ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Fetching...
              </>
            ) : (
              <>
                <Download className="w-4 h-4" />
                Fetch Transcript
              </>
            )}
          </button>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200/50 rounded-lg p-3 flex items-start gap-2">
          <AlertCircle className="w-4 h-4 text-red-600 mt-0.5 flex-shrink-0" />
          <div>
            <p className="text-xs font-medium text-red-900">Error</p>
            <p className="text-xs text-red-700 mt-0.5">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
}
