-- Add custom_vocabularies table + active_vocabulary_id setting (settings-only vocabulary
-- bias feature, see docs/adr/0029): named lists of expected terms (names, acronyms,
-- technical jargon) the user can create/select to bias Whisper's decoding via
-- initial_prompt, complementing the cross-chunk continuity of ADR-0028.
CREATE TABLE IF NOT EXISTS custom_vocabularies (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    terms TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- NULL = "Nessuno" (feature off, default). No FOREIGN KEY constraint on purpose: deleting
-- the active vocabulary is handled defensively at read time (falls back to no bias)
-- rather than relying on SQLite foreign-key enforcement, which isn't guaranteed to be
-- enabled for every connection in this app.
ALTER TABLE transcript_settings ADD COLUMN active_vocabulary_id TEXT;
