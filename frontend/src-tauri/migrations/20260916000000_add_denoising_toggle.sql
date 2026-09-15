-- Add denoising_enabled toggle to transcript_settings
-- Global opt-in toggle for audio denoising, default OFF (see docs/adr/0027)
-- Independent of diarization_enabled: denoising also benefits plain ASR.
ALTER TABLE transcript_settings ADD COLUMN denoising_enabled INTEGER NOT NULL DEFAULT 0;
