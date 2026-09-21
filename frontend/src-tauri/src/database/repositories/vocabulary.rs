// database/repositories/vocabulary.rs
//
// CRUD for named custom vocabularies (ADR-0029) -- domain-term lists the user creates and
// selects (settings-only, no per-recording picker) to bias Whisper's `initial_prompt`.

use crate::database::models::CustomVocabulary;
use chrono::Utc;
use log::warn;
use sqlx::SqlitePool;
use uuid::Uuid;

pub struct VocabularyRepository;

impl VocabularyRepository {
    pub async fn list(pool: &SqlitePool) -> Result<Vec<CustomVocabulary>, sqlx::Error> {
        sqlx::query_as::<_, CustomVocabulary>(
            "SELECT * FROM custom_vocabularies ORDER BY name COLLATE NOCASE ASC",
        )
        .fetch_all(pool)
        .await
    }

    pub async fn get(
        pool: &SqlitePool,
        id: &str,
    ) -> Result<Option<CustomVocabulary>, sqlx::Error> {
        sqlx::query_as::<_, CustomVocabulary>("SELECT * FROM custom_vocabularies WHERE id = $1")
            .bind(id)
            .fetch_optional(pool)
            .await
    }

    pub async fn create(
        pool: &SqlitePool,
        name: &str,
        terms: &str,
    ) -> Result<CustomVocabulary, sqlx::Error> {
        let id = format!("vocab-{}", Uuid::new_v4());
        let now = Utc::now().to_rfc3339();

        sqlx::query(
            "INSERT INTO custom_vocabularies (id, name, terms, created_at, updated_at) VALUES ($1, $2, $3, $4, $5)",
        )
        .bind(&id)
        .bind(name)
        .bind(terms)
        .bind(&now)
        .bind(&now)
        .execute(pool)
        .await?;

        Ok(CustomVocabulary {
            id,
            name: name.to_string(),
            terms: terms.to_string(),
            created_at: now.clone(),
            updated_at: now,
        })
    }

    /// Returns `Ok(false)` (not an error) if `id` doesn't exist -- callers decide whether
    /// that's an error worth surfacing (e.g. the vocabulary was deleted concurrently).
    pub async fn update(
        pool: &SqlitePool,
        id: &str,
        name: &str,
        terms: &str,
    ) -> Result<bool, sqlx::Error> {
        let now = Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE custom_vocabularies SET name = $1, terms = $2, updated_at = $3 WHERE id = $4",
        )
        .bind(name)
        .bind(terms)
        .bind(now)
        .bind(id)
        .execute(pool)
        .await?;
        Ok(result.rows_affected() > 0)
    }

    /// Returns `Ok(false)` (not an error) if `id` doesn't exist. Does not touch
    /// `transcript_settings.active_vocabulary_id` -- if the deleted vocabulary was the
    /// active one, `resolve_active_terms` below falls back to no bias defensively rather
    /// than requiring the caller to clear the setting first.
    pub async fn delete(pool: &SqlitePool, id: &str) -> Result<bool, sqlx::Error> {
        let result = sqlx::query("DELETE FROM custom_vocabularies WHERE id = $1")
            .bind(id)
            .execute(pool)
            .await?;
        Ok(result.rows_affected() > 0)
    }

    /// Resolves the currently active vocabulary (`transcript_settings.active_vocabulary_id`)
    /// to its terms string. Callers should call this **once** per recording session/batch
    /// job and reuse the result across all chunks/segments, not re-query per chunk (same
    /// pattern as `diarization::model::resolve_paths_if_enabled`).
    ///
    /// Returns `None` -- never an error -- when: no vocabulary is selected; the selected
    /// one was deleted after being set active (defensive fallback, logged as a warning);
    /// or the settings read itself fails. A vocabulary-bias feature isn't worth blocking
    /// or failing transcription over.
    pub async fn resolve_active_terms<R: tauri::Runtime>(
        app: &tauri::AppHandle<R>,
    ) -> Option<String> {
        use tauri::Manager;

        // NOTE: `app_for_state` must be a named binding, not inline `app.clone().state()` --
        // otherwise the temporary AppHandle from `.clone()` would be dropped at the end of
        // this statement, before `state`'s borrow of it is used on the next line (same
        // caveat documented in `diarization::model::resolve_paths_if_enabled`).
        let app_for_state = app.clone();
        let state: tauri::State<'_, crate::state::AppState> = app_for_state.state();
        let pool = state.db_manager.pool().clone();

        let vocabulary_id = match crate::database::repositories::setting::SettingsRepository::get_active_vocabulary_id(&pool).await {
            Ok(Some(id)) => id,
            Ok(None) => return None,
            Err(e) => {
                warn!("Failed to read active_vocabulary_id setting: {}", e);
                return None;
            }
        };

        match Self::get(&pool, &vocabulary_id).await {
            Ok(Some(vocabulary)) => Some(vocabulary.terms),
            Ok(None) => {
                warn!(
                    "active_vocabulary_id '{}' no longer exists (vocabulary deleted?) -- falling back to no vocabulary bias",
                    vocabulary_id
                );
                None
            }
            Err(e) => {
                warn!("Failed to load vocabulary '{}': {}", vocabulary_id, e);
                None
            }
        }
    }
}
