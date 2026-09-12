-- Add diarization_max_speakers setting to transcript_settings
-- Upper bound on the number of speakers the spectral clustering method (NME-SC) may
-- estimate. Default 20 mirrors clustering::DEFAULT_MAX_SPEAKERS (see docs/adr/0024).
ALTER TABLE transcript_settings ADD COLUMN diarization_max_speakers INTEGER NOT NULL DEFAULT 20;
