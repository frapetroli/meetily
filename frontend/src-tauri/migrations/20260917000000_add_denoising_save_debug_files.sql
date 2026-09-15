-- Add denoising_save_debug_files toggle to transcript_settings
-- Opt-in (default OFF) sub-setting of denoising_enabled: whether to also save a debug
-- copy of the ASR-bound/diarization-bound signal after denoising, alongside the raw
-- recording. Uncompressed WAV files add real disk usage, so this stays opt-in on top
-- of denoising_enabled rather than automatic (see docs/adr/0027 in the docs workspace).
ALTER TABLE transcript_settings ADD COLUMN denoising_save_debug_files INTEGER NOT NULL DEFAULT 0;
