use crate::database::models::{Setting, TranscriptSetting};
use crate::summary::CustomOpenAIConfig;
use sqlx::SqlitePool;

#[derive(serde::Deserialize, Debug)]
pub struct SaveModelConfigRequest {
    pub provider: String,
    pub model: String,
    #[serde(rename = "whisperModel")]
    pub whisper_model: String,
    #[serde(rename = "apiKey")]
    pub api_key: Option<String>,
    #[serde(rename = "ollamaEndpoint")]
    pub ollama_endpoint: Option<String>,
}

#[derive(serde::Deserialize, Debug)]
pub struct SaveTranscriptConfigRequest {
    pub provider: String,
    pub model: String,
    #[serde(rename = "apiKey")]
    pub api_key: Option<String>,
}

pub struct SettingsRepository;

// Transcript providers: localWhisper, deepgram, elevenLabs, groq, openai
// Summary providers: openai, claude, ollama, groq, added openrouter
// NOTE: Handle data exclusion in the higher layer as this is database abstraction layer(using SELECT *)

impl SettingsRepository {
    pub async fn get_model_config(
        pool: &SqlitePool,
    ) -> std::result::Result<Option<Setting>, sqlx::Error> {
        let setting = sqlx::query_as::<_, Setting>("SELECT * FROM settings LIMIT 1")
            .fetch_optional(pool)
            .await?;
        Ok(setting)
    }

    pub async fn save_model_config(
        pool: &SqlitePool,
        provider: &str,
        model: &str,
        whisper_model: &str,
        ollama_endpoint: Option<&str>,
    ) -> std::result::Result<(), sqlx::Error> {
        // Using id '1' for backward compatibility
        sqlx::query(
            r#"
            INSERT INTO settings (id, provider, model, whisperModel, ollamaEndpoint)
            VALUES ('1', $1, $2, $3, $4)
            ON CONFLICT(id) DO UPDATE SET
                provider = excluded.provider,
                model = excluded.model,
                whisperModel = excluded.whisperModel,
                ollamaEndpoint = excluded.ollamaEndpoint
            "#,
        )
        .bind(provider)
        .bind(model)
        .bind(whisper_model)
        .bind(ollama_endpoint)
        .execute(pool)
        .await?;

        Ok(())
    }

    pub async fn save_api_key(
        pool: &SqlitePool,
        provider: &str,
        api_key: &str,
    ) -> std::result::Result<(), sqlx::Error> {
        // Custom OpenAI uses JSON config (customOpenAIConfig) instead of a separate API key column
        if provider == "custom-openai" {
            return Err(sqlx::Error::Protocol(
                "custom-openai provider should use save_custom_openai_config() instead of save_api_key()".into(),
            ));
        }

        let api_key_column = match provider {
            "openai" => "openaiApiKey",
            "claude" => "anthropicApiKey",
            "ollama" => "ollamaApiKey",
            "groq" => "groqApiKey",
            "openrouter" => "openRouterApiKey",
            "builtin-ai" => return Ok(()), // No API key needed
            _ => {
                return Err(sqlx::Error::Protocol(
                    format!("Invalid provider: {}", provider).into(),
                ))
            }
        };

        let query = format!(
            r#"
            INSERT INTO settings (id, provider, model, whisperModel, "{}")
            VALUES ('1', 'openai', 'gpt-4o-2024-11-20', 'large-v3', $1)
            ON CONFLICT(id) DO UPDATE SET
                "{}" = $1
            "#,
            api_key_column, api_key_column
        );
        sqlx::query(&query).bind(api_key).execute(pool).await?;

        Ok(())
    }

    pub async fn get_api_key(
        pool: &SqlitePool,
        provider: &str,
    ) -> std::result::Result<Option<String>, sqlx::Error> {
        // Custom OpenAI uses JSON config - extract API key from there
        if provider == "custom-openai" {
            let config = Self::get_custom_openai_config(pool).await?;
            return Ok(config.and_then(|c| c.api_key));
        }

        let api_key_column = match provider {
            "openai" => "openaiApiKey",
            "ollama" => "ollamaApiKey",
            "groq" => "groqApiKey",
            "claude" => "anthropicApiKey",
            "openrouter" => "openRouterApiKey",
            "builtin-ai" => return Ok(None), // No API key needed
            _ => {
                return Err(sqlx::Error::Protocol(
                    format!("Invalid provider: {}", provider).into(),
                ))
            }
        };

        let query = format!(
            "SELECT {} FROM settings WHERE id = '1' LIMIT 1",
            api_key_column
        );
        let api_key = sqlx::query_scalar(&query).fetch_optional(pool).await?;
        Ok(api_key)
    }

    pub async fn get_transcript_config(
        pool: &SqlitePool,
    ) -> std::result::Result<Option<TranscriptSetting>, sqlx::Error> {
        let setting =
            sqlx::query_as::<_, TranscriptSetting>("SELECT * FROM transcript_settings LIMIT 1")
                .fetch_optional(pool)
                .await?;
        Ok(setting)

    }

    pub async fn save_transcript_config(
        pool: &SqlitePool,
        provider: &str,
        model: &str,
    ) -> std::result::Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            INSERT INTO transcript_settings (id, provider, model)
            VALUES ('1', $1, $2)
            ON CONFLICT(id) DO UPDATE SET
                provider = excluded.provider,
                model = excluded.model
            "#,
        )
        .bind(provider)
        .bind(model)
        .execute(pool)
        .await?;

        Ok(())
    }

    /// Global opt-in toggle (ADR-0010), default `false`. Deliberately not part of
    /// `TranscriptSetting`/`get_transcript_config()` (which map `SELECT *` and are
    /// consumed widely, including by the frontend's `TranscriptConfig` type) -- a
    /// dedicated targeted query keeps this addition low-blast-radius.
    pub async fn get_diarization_enabled(pool: &SqlitePool) -> std::result::Result<bool, sqlx::Error> {
        let enabled: Option<bool> = sqlx::query_scalar(
            "SELECT diarization_enabled FROM transcript_settings WHERE id = '1' LIMIT 1",
        )
        .fetch_optional(pool)
        .await?;
        Ok(enabled.unwrap_or(false))
    }

    pub async fn save_diarization_enabled(
        pool: &SqlitePool,
        enabled: bool,
    ) -> std::result::Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            INSERT INTO transcript_settings (id, provider, model, diarization_enabled)
            VALUES ('1', 'parakeet', $1, $2)
            ON CONFLICT(id) DO UPDATE SET
                diarization_enabled = excluded.diarization_enabled
            "#,
        )
        .bind(crate::config::DEFAULT_PARAKEET_MODEL)
        .bind(enabled)
        .execute(pool)
        .await?;

        Ok(())
    }

    /// Upper bound on the number of speakers the spectral clustering method (NME-SC) may
    /// estimate for a call (see `docs/adr/0024-...md`), default
    /// `clustering::DEFAULT_MAX_SPEAKERS` (20). Same dedicated-query pattern as
    /// `get_diarization_enabled` above, for the same reason (kept out of
    /// `TranscriptSetting`/`get_transcript_config()`).
    pub async fn get_diarization_max_speakers(pool: &SqlitePool) -> std::result::Result<usize, sqlx::Error> {
        let max_speakers: Option<i64> = sqlx::query_scalar(
            "SELECT diarization_max_speakers FROM transcript_settings WHERE id = '1' LIMIT 1",
        )
        .fetch_optional(pool)
        .await?;
        Ok(max_speakers
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v > 0)
            .unwrap_or(crate::diarization::clustering::DEFAULT_MAX_SPEAKERS))
    }

    pub async fn save_diarization_max_speakers(
        pool: &SqlitePool,
        max_speakers: usize,
    ) -> std::result::Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            INSERT INTO transcript_settings (id, provider, model, diarization_max_speakers)
            VALUES ('1', 'parakeet', $1, $2)
            ON CONFLICT(id) DO UPDATE SET
                diarization_max_speakers = excluded.diarization_max_speakers
            "#,
        )
        .bind(crate::config::DEFAULT_PARAKEET_MODEL)
        .bind(max_speakers as i64)
        .execute(pool)
        .await?;

        Ok(())
    }

    /// Global opt-in toggle for audio denoising (ADR-0027), default `false`.
    /// Independent of `diarization_enabled` -- denoising also benefits plain ASR, so it
    /// is never gated on the diarization toggle. Same dedicated-query pattern as
    /// `get_diarization_enabled` above (kept out of `TranscriptSetting`/
    /// `get_transcript_config()`), for the same low-blast-radius reason.
    pub async fn get_denoising_enabled(pool: &SqlitePool) -> std::result::Result<bool, sqlx::Error> {
        let enabled: Option<bool> = sqlx::query_scalar(
            "SELECT denoising_enabled FROM transcript_settings WHERE id = '1' LIMIT 1",
        )
        .fetch_optional(pool)
        .await?;
        Ok(enabled.unwrap_or(false))
    }

    pub async fn save_denoising_enabled(
        pool: &SqlitePool,
        enabled: bool,
    ) -> std::result::Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            INSERT INTO transcript_settings (id, provider, model, denoising_enabled)
            VALUES ('1', 'parakeet', $1, $2)
            ON CONFLICT(id) DO UPDATE SET
                denoising_enabled = excluded.denoising_enabled
            "#,
        )
        .bind(crate::config::DEFAULT_PARAKEET_MODEL)
        .bind(enabled)
        .execute(pool)
        .await?;

        Ok(())
    }

    /// Whether to also save, alongside the raw recording, a debug copy of the exact
    /// signal handed to ASR and to diarization after denoising (`audio_denoised_asr.wav`/
    /// `audio_denoised_diarization.wav`). Default `false`: these are uncompressed WAV
    /// files (32-bit float PCM) that add real disk usage -- roughly 1.4GB/hour for a
    /// live recording (48kHz, two files), ~460MB/hour for batch (16kHz) -- so this is
    /// opt-in on top of `denoising_enabled`, not automatic. Meaningless when
    /// `denoising_enabled` is off (there would be nothing to save). Same dedicated-query
    /// pattern as the settings above.
    pub async fn get_denoising_save_debug_files(pool: &SqlitePool) -> std::result::Result<bool, sqlx::Error> {
        let enabled: Option<bool> = sqlx::query_scalar(
            "SELECT denoising_save_debug_files FROM transcript_settings WHERE id = '1' LIMIT 1",
        )
        .fetch_optional(pool)
        .await?;
        Ok(enabled.unwrap_or(false))
    }

    pub async fn save_denoising_save_debug_files(
        pool: &SqlitePool,
        enabled: bool,
    ) -> std::result::Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            INSERT INTO transcript_settings (id, provider, model, denoising_save_debug_files)
            VALUES ('1', 'parakeet', $1, $2)
            ON CONFLICT(id) DO UPDATE SET
                denoising_save_debug_files = excluded.denoising_save_debug_files
            "#,
        )
        .bind(crate::config::DEFAULT_PARAKEET_MODEL)
        .bind(enabled)
        .execute(pool)
        .await?;

        Ok(())
    }

    /// Id of the `custom_vocabularies` row currently active for Whisper's `initial_prompt`
    /// vocabulary bias (ADR-0029), or `None` ("Nessuno", default -- feature off). Same
    /// dedicated-query pattern as `get_diarization_enabled` above, for the same reason.
    /// Double `Option` because the column itself is nullable (unlike the other settings
    /// here): the outer `Option` is "row exists at all", the inner is "column is NULL".
    pub async fn get_active_vocabulary_id(
        pool: &SqlitePool,
    ) -> std::result::Result<Option<String>, sqlx::Error> {
        let id: Option<Option<String>> = sqlx::query_scalar(
            "SELECT active_vocabulary_id FROM transcript_settings WHERE id = '1' LIMIT 1",
        )
        .fetch_optional(pool)
        .await?;
        Ok(id.flatten())
    }

    pub async fn save_active_vocabulary_id(
        pool: &SqlitePool,
        vocabulary_id: Option<&str>,
    ) -> std::result::Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            INSERT INTO transcript_settings (id, provider, model, active_vocabulary_id)
            VALUES ('1', 'parakeet', $1, $2)
            ON CONFLICT(id) DO UPDATE SET
                active_vocabulary_id = excluded.active_vocabulary_id
            "#,
        )
        .bind(crate::config::DEFAULT_PARAKEET_MODEL)
        .bind(vocabulary_id)
        .execute(pool)
        .await?;

        Ok(())
    }

    pub async fn save_transcript_api_key(
        pool: &SqlitePool,
        provider: &str,
        api_key: &str,
    ) -> std::result::Result<(), sqlx::Error> {
        let api_key_column = match provider {
            "localWhisper" => "whisperApiKey",
            "parakeet" => return Ok(()), // Parakeet doesn't need an API key, return early
            "deepgram" => "deepgramApiKey",
            "elevenLabs" => "elevenLabsApiKey",
            "groq" => "groqApiKey",
            "openai" => "openaiApiKey",
            _ => {
                return Err(sqlx::Error::Protocol(
                    format!("Invalid provider: {}", provider).into(),
                ))
            }
        };

        let query = format!(
            r#"
            INSERT INTO transcript_settings (id, provider, model, "{}")
            VALUES ('1', 'parakeet', '{}', $1)
            ON CONFLICT(id) DO UPDATE SET
                "{}" = $1
            "#,
            api_key_column, crate::config::DEFAULT_PARAKEET_MODEL, api_key_column
        );
        sqlx::query(&query).bind(api_key).execute(pool).await?;

        Ok(())
    }

    pub async fn get_transcript_api_key(
        pool: &SqlitePool,
        provider: &str,
    ) -> std::result::Result<Option<String>, sqlx::Error> {
        let api_key_column = match provider {
            "localWhisper" => "whisperApiKey",
            "parakeet" => return Ok(None), // Parakeet doesn't need an API key
            "deepgram" => "deepgramApiKey",
            "elevenLabs" => "elevenLabsApiKey",
            "groq" => "groqApiKey",
            "openai" => "openaiApiKey",
            _ => {
                return Err(sqlx::Error::Protocol(
                    format!("Invalid provider: {}", provider).into(),
                ))
            }
        };

        let query = format!(
            "SELECT {} FROM transcript_settings WHERE id = '1' LIMIT 1",
            api_key_column
        );
        let api_key = sqlx::query_scalar(&query).fetch_optional(pool).await?;
        Ok(api_key)
    }

    pub async fn delete_api_key(
        pool: &SqlitePool,
        provider: &str,
    ) -> std::result::Result<(), sqlx::Error> {
        // Custom OpenAI uses JSON config - clear the entire config
        if provider == "custom-openai" {
            sqlx::query("UPDATE settings SET customOpenAIConfig = NULL WHERE id = '1'")
                .execute(pool)
                .await?;
            return Ok(());
        }

        let api_key_column = match provider {
            "openai" => "openaiApiKey",
            "ollama" => "ollamaApiKey",
            "groq" => "groqApiKey",
            "claude" => "anthropicApiKey",
            "openrouter" => "openRouterApiKey",
            "builtin-ai" => return Ok(()), // No API key needed
            _ => {
                return Err(sqlx::Error::Protocol(
                    format!("Invalid provider: {}", provider).into(),
                ))
            }
        };

        let query = format!(
            "UPDATE settings SET {} = NULL WHERE id = '1'",
            api_key_column
        );
        sqlx::query(&query).execute(pool).await?;

        Ok(())
    }

    // ===== CUSTOM OPENAI CONFIG METHODS =====

    /// Gets the custom OpenAI configuration from JSON
    ///
    /// # Returns
    /// * `Ok(Some(CustomOpenAIConfig))` - Config exists and is valid JSON
    /// * `Ok(None)` - No config stored
    /// * `Err(sqlx::Error)` - Database error
    pub async fn get_custom_openai_config(
        pool: &SqlitePool,
    ) -> std::result::Result<Option<CustomOpenAIConfig>, sqlx::Error> {
        use sqlx::Row;

        let row = sqlx::query(
            r#"
            SELECT customOpenAIConfig
            FROM settings
            WHERE id = '1'
            LIMIT 1
            "#
        )
        .fetch_optional(pool)
        .await?;

        match row {
            Some(record) => {
                let config_json: Option<String> = record.get("customOpenAIConfig");

                if let Some(json) = config_json {
                    // Parse JSON into CustomOpenAIConfig
                    let config: CustomOpenAIConfig = serde_json::from_str(&json)
                        .map_err(|e| sqlx::Error::Protocol(
                            format!("Invalid JSON in customOpenAIConfig: {}", e).into()
                        ))?;

                    Ok(Some(config))
                } else {
                    Ok(None)
                }
            }
            None => Ok(None),
        }
    }

    /// Saves the custom OpenAI configuration as JSON
    ///
    /// # Arguments
    /// * `pool` - Database connection pool
    /// * `config` - CustomOpenAIConfig to save (includes endpoint, apiKey, model, maxTokens, temperature, topP)
    ///
    /// # Returns
    /// * `Ok(())` - Config saved successfully
    /// * `Err(sqlx::Error)` - Database or JSON serialization error
    pub async fn save_custom_openai_config(
        pool: &SqlitePool,
        config: &CustomOpenAIConfig,
    ) -> std::result::Result<(), sqlx::Error> {
        // Serialize config to JSON
        let config_json = serde_json::to_string(config)
            .map_err(|e| sqlx::Error::Protocol(
                format!("Failed to serialize config to JSON: {}", e).into()
            ))?;

        // Upsert into settings table
        sqlx::query(
            r#"
            INSERT INTO settings (id, provider, model, whisperModel, customOpenAIConfig)
            VALUES ('1', 'custom-openai', $1, 'large-v3', $2)
            ON CONFLICT(id) DO UPDATE SET
                customOpenAIConfig = excluded.customOpenAIConfig
            "#,
        )
        .bind(&config.model)
        .bind(config_json)
        .execute(pool)
        .await?;

        Ok(())
    }
}
