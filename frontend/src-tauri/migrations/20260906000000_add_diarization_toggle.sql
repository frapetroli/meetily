-- Add diarization_enabled toggle to transcript_settings
-- Global opt-in toggle for speaker diarization, default OFF (see docs/adr/0010)
ALTER TABLE transcript_settings ADD COLUMN diarization_enabled INTEGER NOT NULL DEFAULT 0;
