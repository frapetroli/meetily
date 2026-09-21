import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { Progress } from './ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Button } from './ui/button';
import { Label } from './ui/label';
import { Switch } from './ui/switch';
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog';
import { Eye, EyeOff, Lock, Unlock } from 'lucide-react';
import { ModelManager } from './WhisperModelManager';
import { ParakeetModelManager } from './ParakeetModelManager';


export interface TranscriptModelProps {
    provider: 'localWhisper' | 'parakeet' | 'deepgram' | 'elevenLabs' | 'groq' | 'openai';
    model: string;
    apiKey?: string | null;
}

export interface TranscriptSettingsProps {
    transcriptModelConfig: TranscriptModelProps;
    setTranscriptModelConfig: (config: TranscriptModelProps) => void;
    onModelSelect?: () => void;
}

// Mirrors Rust's DiarizationModelStatus (diarization/model.rs) as serialized by serde's
// default externally-tagged enum representation -- same convention already used
// elsewhere in this codebase for Whisper/Parakeet's ModelStatus (see
// transcription-model-readiness.ts's hasDownloadingModel).
type DiarizationModelStatus =
    | 'Available'
    | 'Missing'
    | { Downloading: { progress: number } }
    | { Corrupted: { file: string } }
    | { Error: string };

// Mirrors Rust's DenoisingModelStatus (denoising/model.rs), same serde convention as
// DiarizationModelStatus above.
type DenoisingModelStatus =
    | 'Available'
    | 'Missing'
    | { Downloading: { progress: number } }
    | { Corrupted: { file: string } }
    | { Error: string };

// Mirrors Rust's CustomVocabulary (database/models.rs, ADR-0029). Field names are plain
// snake_case on both sides -- no #[serde(rename)] on the Rust struct, unlike
// TranscriptSetting's camelCase API-key fields elsewhere in this file.
interface CustomVocabulary {
    id: string;
    name: string;
    terms: string;
    created_at: string;
    updated_at: string;
}

export function TranscriptSettings({ transcriptModelConfig, setTranscriptModelConfig, onModelSelect }: TranscriptSettingsProps) {
    const [apiKey, setApiKey] = useState<string | null>(transcriptModelConfig.apiKey || null);
    const [showApiKey, setShowApiKey] = useState<boolean>(false);
    const [isApiKeyLocked, setIsApiKeyLocked] = useState<boolean>(true);
    const [isLockButtonVibrating, setIsLockButtonVibrating] = useState<boolean>(false);
    const [uiProvider, setUiProvider] = useState<TranscriptModelProps['provider']>(transcriptModelConfig.provider);
    // Diarization toggle (ADR-0010): global opt-in, default off, separate from the
    // transcript provider/model config above -- see api_get/save_diarization_enabled.
    const [diarizationEnabled, setDiarizationEnabled] = useState<boolean>(false);
    const [isDiarizationToggleBusy, setIsDiarizationToggleBusy] = useState<boolean>(false);
    const [diarizationModelStatus, setDiarizationModelStatus] = useState<DiarizationModelStatus | null>(null);
    const [diarizationDownloadPercent, setDiarizationDownloadPercent] = useState<number>(0);
    const [isDiarizationDownloading, setIsDiarizationDownloading] = useState<boolean>(false);
    const [diarizationDownloadError, setDiarizationDownloadError] = useState<string | null>(null);
    // Upper bound passed to the spectral clustering method (NME-SC, ADR-0024), default 20
    // -- see api_get/save_diarization_max_speakers. Saved on blur, not per-keystroke.
    const [maxSpeakers, setMaxSpeakers] = useState<number>(20);

    // Denoising toggle (ADR-0027): global opt-in, default off, independent of the
    // diarization toggle above -- denoising also benefits plain ASR on its own.
    const [denoisingEnabled, setDenoisingEnabled] = useState<boolean>(false);
    const [isDenoisingToggleBusy, setIsDenoisingToggleBusy] = useState<boolean>(false);
    const [denoisingModelStatus, setDenoisingModelStatus] = useState<DenoisingModelStatus | null>(null);
    const [denoisingDownloadPercent, setDenoisingDownloadPercent] = useState<number>(0);
    const [isDenoisingDownloading, setIsDenoisingDownloading] = useState<boolean>(false);
    const [denoisingDownloadError, setDenoisingDownloadError] = useState<string | null>(null);
    // Opt-in sub-setting of the toggle above: whether to also save a debug copy of the
    // denoised ASR/diarization signal. Default off -- uncompressed WAV files, real
    // disk usage -- see api_get/save_denoising_save_debug_files.
    const [denoisingSaveDebugFiles, setDenoisingSaveDebugFiles] = useState<boolean>(false);
    const [isDenoisingSaveDebugFilesBusy, setIsDenoisingSaveDebugFilesBusy] = useState<boolean>(false);

    // Custom vocabulary (ADR-0029): settings-only Whisper `initial_prompt` bias, managed
    // here (create/edit/delete + pick the active one), never per-recording. `null` =
    // "None" (feature off, default) -- see api_get/save_active_vocabulary_id.
    const [vocabularies, setVocabularies] = useState<CustomVocabulary[]>([]);
    const [activeVocabularyId, setActiveVocabularyId] = useState<string | null>(null);
    const [isVocabularyDialogOpen, setIsVocabularyDialogOpen] = useState<boolean>(false);
    // `null` while the dialog is in "create" mode, the vocabulary being edited otherwise.
    const [editingVocabulary, setEditingVocabulary] = useState<CustomVocabulary | null>(null);
    const [vocabularyNameInput, setVocabularyNameInput] = useState<string>('');
    const [vocabularyTermsInput, setVocabularyTermsInput] = useState<string>('');
    const [isVocabularySaving, setIsVocabularySaving] = useState<boolean>(false);
    const [vocabularyError, setVocabularyError] = useState<string | null>(null);

    const refreshVocabularies = async () => {
        try {
            const list = await invoke<CustomVocabulary[]>('api_list_vocabularies');
            setVocabularies(list);
        } catch (err) {
            console.error('Error fetching vocabularies:', err);
        }
    };

    const handleActiveVocabularyChange = async (value: string) => {
        const newId = value === 'none' ? null : value;
        const previous = activeVocabularyId;
        setActiveVocabularyId(newId); // optimistic update
        try {
            await invoke('api_save_active_vocabulary_id', { vocabularyId: newId });
        } catch (err) {
            console.error('Error saving active_vocabulary_id:', err);
            setActiveVocabularyId(previous); // revert on failure
        }
    };

    const openCreateVocabularyDialog = () => {
        setEditingVocabulary(null);
        setVocabularyNameInput('');
        setVocabularyTermsInput('');
        setVocabularyError(null);
        setIsVocabularyDialogOpen(true);
    };

    const openEditVocabularyDialog = (vocabulary: CustomVocabulary) => {
        setEditingVocabulary(vocabulary);
        setVocabularyNameInput(vocabulary.name);
        setVocabularyTermsInput(vocabulary.terms);
        setVocabularyError(null);
        setIsVocabularyDialogOpen(true);
    };

    const handleSaveVocabulary = async () => {
        const name = vocabularyNameInput.trim();
        const terms = vocabularyTermsInput.trim();
        if (!name) {
            setVocabularyError('Name is required');
            return;
        }
        setIsVocabularySaving(true);
        setVocabularyError(null);
        try {
            if (editingVocabulary) {
                await invoke('api_update_vocabulary', { id: editingVocabulary.id, name, terms });
            } else {
                await invoke('api_create_vocabulary', { name, terms });
            }
            await refreshVocabularies();
            setIsVocabularyDialogOpen(false);
        } catch (err) {
            setVocabularyError(String(err));
        } finally {
            setIsVocabularySaving(false);
        }
    };

    const handleDeleteVocabulary = async (vocabulary: CustomVocabulary) => {
        if (!window.confirm(`Delete vocabulary "${vocabulary.name}"?`)) {
            return;
        }
        try {
            await invoke('api_delete_vocabulary', { id: vocabulary.id });
            setVocabularies((prev) => prev.filter((v) => v.id !== vocabulary.id));
            // Backend already falls back defensively at read time if the deleted vocabulary
            // was active (VocabularyRepository::resolve_active_terms), but the stored
            // active_vocabulary_id setting itself would otherwise be left pointing at a
            // vocabulary that no longer exists -- clear it explicitly so a later reload
            // doesn't show the Select with a value matching no option.
            if (activeVocabularyId === vocabulary.id) {
                setActiveVocabularyId(null);
                await invoke('api_save_active_vocabulary_id', { vocabularyId: null });
            }
        } catch (err) {
            console.error('Error deleting vocabulary:', err);
        }
    };

    const refreshDiarizationModelStatus = async () => {
        try {
            const status = await invoke<DiarizationModelStatus>('diarization_get_status');
            setDiarizationModelStatus(status);
        } catch (err) {
            console.error('Error fetching diarization model status:', err);
        }
    };

    // Check model status once diarization is toggled on (no need while it's off)
    useEffect(() => {
        if (diarizationEnabled) {
            refreshDiarizationModelStatus();
        }
    }, [diarizationEnabled]);

    // Listen for download progress/completion/error, same event-naming convention as
    // parakeet-model-download-*/builtin-ai-download-*
    useEffect(() => {
        const unlistenPromises = [
            listen<{ file: string; percent: number }>('diarization-model-download-progress', (event) => {
                setDiarizationDownloadPercent(event.payload.percent);
            }),
            listen('diarization-model-download-complete', () => {
                setIsDiarizationDownloading(false);
                setDiarizationDownloadError(null);
                refreshDiarizationModelStatus();
            }),
            listen<string>('diarization-model-download-error', (event) => {
                setIsDiarizationDownloading(false);
                setDiarizationDownloadError(event.payload);
            }),
        ];
        return () => {
            unlistenPromises.forEach((p) => p.then((unlisten) => unlisten()));
        };
    }, []);

    const handleDownloadDiarizationModels = async () => {
        setIsDiarizationDownloading(true);
        setDiarizationDownloadPercent(0);
        setDiarizationDownloadError(null);
        try {
            await invoke('diarization_download_models');
        } catch (err) {
            setIsDiarizationDownloading(false);
            setDiarizationDownloadError(String(err));
        }
    };

    const refreshDenoisingModelStatus = async () => {
        try {
            const status = await invoke<DenoisingModelStatus>('denoising_get_status');
            setDenoisingModelStatus(status);
        } catch (err) {
            console.error('Error fetching denoising model status:', err);
        }
    };

    // Check model status once denoising is toggled on (no need while it's off)
    useEffect(() => {
        if (denoisingEnabled) {
            refreshDenoisingModelStatus();
        }
    }, [denoisingEnabled]);

    useEffect(() => {
        const unlistenPromises = [
            listen<{ file: string; percent: number }>('denoising-model-download-progress', (event) => {
                setDenoisingDownloadPercent(event.payload.percent);
            }),
            listen('denoising-model-download-complete', () => {
                setIsDenoisingDownloading(false);
                setDenoisingDownloadError(null);
                refreshDenoisingModelStatus();
            }),
            listen<string>('denoising-model-download-error', (event) => {
                setIsDenoisingDownloading(false);
                setDenoisingDownloadError(event.payload);
            }),
        ];
        return () => {
            unlistenPromises.forEach((p) => p.then((unlisten) => unlisten()));
        };
    }, []);

    const handleDownloadDenoisingModels = async () => {
        setIsDenoisingDownloading(true);
        setDenoisingDownloadPercent(0);
        setDenoisingDownloadError(null);
        try {
            await invoke('denoising_download_models');
        } catch (err) {
            setIsDenoisingDownloading(false);
            setDenoisingDownloadError(String(err));
        }
    };

    // Sync uiProvider when backend config changes (e.g., after model selection or initial load)
    useEffect(() => {
        setUiProvider(transcriptModelConfig.provider);
    }, [transcriptModelConfig.provider]);

    useEffect(() => {
        invoke<boolean>('api_get_diarization_enabled')
            .then(setDiarizationEnabled)
            .catch((err) => console.error('Error fetching diarization_enabled:', err));
        invoke<number>('api_get_diarization_max_speakers')
            .then(setMaxSpeakers)
            .catch((err) => console.error('Error fetching diarization_max_speakers:', err));
        invoke<boolean>('api_get_denoising_enabled')
            .then(setDenoisingEnabled)
            .catch((err) => console.error('Error fetching denoising_enabled:', err));
        invoke<boolean>('api_get_denoising_save_debug_files')
            .then(setDenoisingSaveDebugFiles)
            .catch((err) => console.error('Error fetching denoising_save_debug_files:', err));
        refreshVocabularies();
        invoke<string | null>('api_get_active_vocabulary_id')
            .then(setActiveVocabularyId)
            .catch((err) => console.error('Error fetching active_vocabulary_id:', err));
    }, []);

    const handleDiarizationToggle = async (checked: boolean) => {
        setIsDiarizationToggleBusy(true);
        const previous = diarizationEnabled;
        setDiarizationEnabled(checked); // optimistic update
        try {
            await invoke('api_save_diarization_enabled', { enabled: checked });
        } catch (err) {
            console.error('Error saving diarization_enabled:', err);
            setDiarizationEnabled(previous); // revert on failure
        } finally {
            setIsDiarizationToggleBusy(false);
        }
    };

    const handleMaxSpeakersBlur = async (rawValue: string) => {
        const parsed = Math.max(2, Math.round(Number(rawValue)) || 20);
        setMaxSpeakers(parsed);
        try {
            await invoke('api_save_diarization_max_speakers', { maxSpeakers: parsed });
        } catch (err) {
            console.error('Error saving diarization_max_speakers:', err);
        }
    };

    const handleDenoisingToggle = async (checked: boolean) => {
        setIsDenoisingToggleBusy(true);
        const previous = denoisingEnabled;
        setDenoisingEnabled(checked); // optimistic update
        try {
            await invoke('api_save_denoising_enabled', { enabled: checked });
        } catch (err) {
            console.error('Error saving denoising_enabled:', err);
            setDenoisingEnabled(previous); // revert on failure
        } finally {
            setIsDenoisingToggleBusy(false);
        }
    };

    const handleDenoisingSaveDebugFilesToggle = async (checked: boolean) => {
        setIsDenoisingSaveDebugFilesBusy(true);
        const previous = denoisingSaveDebugFiles;
        setDenoisingSaveDebugFiles(checked); // optimistic update
        try {
            await invoke('api_save_denoising_save_debug_files', { enabled: checked });
        } catch (err) {
            console.error('Error saving denoising_save_debug_files:', err);
            setDenoisingSaveDebugFiles(previous); // revert on failure
        } finally {
            setIsDenoisingSaveDebugFilesBusy(false);
        }
    };

    useEffect(() => {
        if (transcriptModelConfig.provider === 'localWhisper' || transcriptModelConfig.provider === 'parakeet') {
            setApiKey(null);
        }
    }, [transcriptModelConfig.provider]);

    const fetchApiKey = async (provider: string) => {
        try {

            const data = await invoke('api_get_transcript_api_key', { provider }) as string;

            setApiKey(data || '');
        } catch (err) {
            console.error('Error fetching API key:', err);
            setApiKey(null);
        }
    };
    const modelOptions = {
        localWhisper: [], // Model selection handled by ModelManager component
        parakeet: [], // Model selection handled by ParakeetModelManager component
        deepgram: ['nova-2-phonecall'],
        elevenLabs: ['eleven_multilingual_v2'],
        groq: ['llama-3.3-70b-versatile'],
        openai: ['gpt-4o'],
    };
    const requiresApiKey = transcriptModelConfig.provider === 'deepgram' || transcriptModelConfig.provider === 'elevenLabs' || transcriptModelConfig.provider === 'openai' || transcriptModelConfig.provider === 'groq';

    const handleInputClick = () => {
        if (isApiKeyLocked) {
            setIsLockButtonVibrating(true);
            setTimeout(() => setIsLockButtonVibrating(false), 500);
        }
    };

    const handleWhisperModelSelect = (modelName: string) => {
        // Always update config when model is selected, regardless of current provider
        // This ensures the model is set when user switches back
        setTranscriptModelConfig({
            ...transcriptModelConfig,
            provider: 'localWhisper', // Ensure provider is set correctly
            model: modelName
        });
        // Close modal after selection
        if (onModelSelect) {
            onModelSelect();
        }
    };

    const handleParakeetModelSelect = (modelName: string) => {
        // Always update config when model is selected, regardless of current provider
        // This ensures the model is set when user switches back
        setTranscriptModelConfig({
            ...transcriptModelConfig,
            provider: 'parakeet', // Ensure provider is set correctly
            model: modelName
        });
        // Close modal after selection
        if (onModelSelect) {
            onModelSelect();
        }
    };

    return (
        <div>
            <div>
                {/* <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-semibold text-gray-900">Transcript Settings</h3>
                </div> */}
                <div className="space-y-4 pb-6">
                    <div>
                        <Label className="block text-sm font-medium text-gray-700 mb-1">
                            Transcript Model
                        </Label>
                        <div className="flex space-x-2 mx-1">
                            <Select
                                value={uiProvider}
                                onValueChange={(value) => {
                                    const provider = value as TranscriptModelProps['provider'];
                                    setUiProvider(provider);
                                    if (provider !== 'localWhisper' && provider !== 'parakeet') {
                                        fetchApiKey(provider);
                                    }
                                }}
                            >
                                <SelectTrigger className='focus:ring-1 focus:ring-blue-500 focus:border-blue-500'>
                                    <SelectValue placeholder="Select provider" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="parakeet">⚡ Parakeet (Recommended - Real-time / Accurate)</SelectItem>
                                    <SelectItem value="localWhisper">🏠 Local Whisper (High Accuracy)</SelectItem>
                                    {/* <SelectItem value="deepgram">☁️ Deepgram (Backup)</SelectItem>
                                    <SelectItem value="elevenLabs">☁️ ElevenLabs</SelectItem>
                                    <SelectItem value="groq">☁️ Groq</SelectItem>
                                    <SelectItem value="openai">☁️ OpenAI</SelectItem> */}
                                </SelectContent>
                            </Select>

                            {uiProvider !== 'localWhisper' && uiProvider !== 'parakeet' && (
                                <Select
                                    value={transcriptModelConfig.model}
                                    onValueChange={(value) => {
                                        const model = value as TranscriptModelProps['model'];
                                        setTranscriptModelConfig({ ...transcriptModelConfig, provider: uiProvider, model });
                                    }}
                                >
                                    <SelectTrigger className='focus:ring-1 focus:ring-blue-500 focus:border-blue-500'>
                                        <SelectValue placeholder="Select model" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {modelOptions[uiProvider].map((model) => (
                                            <SelectItem key={model} value={model}>{model}</SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            )}

                        </div>
                    </div>

                    {uiProvider === 'localWhisper' && (
                        <div className="mt-6">
                            <ModelManager
                                selectedModel={transcriptModelConfig.provider === 'localWhisper' ? transcriptModelConfig.model : undefined}
                                onModelSelect={handleWhisperModelSelect}
                                autoSave={true}
                            />
                        </div>
                    )}

                    {uiProvider === 'parakeet' && (
                        <div className="mt-6">
                            <ParakeetModelManager
                                selectedModel={transcriptModelConfig.provider === 'parakeet' ? transcriptModelConfig.model : undefined}
                                onModelSelect={handleParakeetModelSelect}
                                autoSave={true}
                            />
                        </div>
                    )}


                    {requiresApiKey && (
                        <div>
                            <Label className="block text-sm font-medium text-gray-700 mb-1">
                                API Key
                            </Label>
                            <div className="relative mx-1">
                                <Input
                                    type={showApiKey ? "text" : "password"}
                                    className={`pr-24 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 ${isApiKeyLocked ? 'bg-gray-100 cursor-not-allowed' : ''
                                        }`}
                                    value={apiKey || ''}
                                    onChange={(e) => setApiKey(e.target.value)}
                                    disabled={isApiKeyLocked}
                                    onClick={handleInputClick}
                                    placeholder="Enter your API key"
                                />
                                {isApiKeyLocked && (
                                    <div
                                        onClick={handleInputClick}
                                        className="absolute inset-0 flex items-center justify-center bg-gray-100 bg-opacity-50 rounded-md cursor-not-allowed"
                                    />
                                )}
                                <div className="absolute inset-y-0 right-0 pr-1 flex items-center">
                                    <Button
                                        type="button"
                                        variant="ghost"
                                        size="icon"
                                        onClick={() => setIsApiKeyLocked(!isApiKeyLocked)}
                                        className={`transition-colors duration-200 ${isLockButtonVibrating ? 'animate-vibrate text-red-500' : ''
                                            }`}
                                        title={isApiKeyLocked ? "Unlock to edit" : "Lock to prevent editing"}
                                    >
                                        {isApiKeyLocked ? <Lock className="h-4 w-4" /> : <Unlock className="h-4 w-4" />}
                                    </Button>
                                    <Button
                                        type="button"
                                        variant="ghost"
                                        size="icon"
                                        onClick={() => setShowApiKey(!showApiKey)}
                                    >
                                        {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                                    </Button>
                                </div>
                            </div>
                        </div>
                    )}

                    <div className="pt-2 border-t border-gray-100">
                        <div className="flex items-center justify-between mt-4">
                            <div className="pr-4">
                                <Label className="block text-sm font-medium text-gray-700">
                                    Speaker diarization
                                </Label>
                                <p className="text-xs text-gray-500 mt-0.5">
                                    Label who is speaking in the transcript. Runs entirely locally, CPU-only.
                                    Adds a short "diarizing" step when a recording stops. Requires downloading
                                    ~33MB of additional models on first use.
                                </p>
                            </div>
                            <Switch
                                checked={diarizationEnabled}
                                disabled={isDiarizationToggleBusy}
                                onCheckedChange={handleDiarizationToggle}
                            />
                        </div>

                        {diarizationEnabled && (
                            <div className="mt-3 mx-1 p-3 rounded-md border border-gray-200 bg-gray-50">
                                {isDiarizationDownloading ? (
                                    <div>
                                        <p className="text-xs text-gray-600 mb-1">
                                            Downloading diarization models... {diarizationDownloadPercent}%
                                        </p>
                                        <Progress value={diarizationDownloadPercent} />
                                    </div>
                                ) : diarizationModelStatus === 'Available' ? (
                                    <p className="text-xs text-emerald-700">✓ Diarization models ready</p>
                                ) : (
                                    <div className="flex items-center justify-between gap-3">
                                        <p className="text-xs text-gray-600">
                                            {diarizationDownloadError
                                                ? `Download failed: ${diarizationDownloadError}`
                                                : 'Diarization models not downloaded yet. Recording will be blocked while diarization is on until this completes.'}
                                        </p>
                                        <Button
                                            type="button"
                                            variant="outline"
                                            size="sm"
                                            onClick={handleDownloadDiarizationModels}
                                            className="shrink-0"
                                        >
                                            {diarizationDownloadError ? 'Retry' : 'Download'}
                                        </Button>
                                    </div>
                                )}

                                <div className="mt-3 pt-3 border-t border-gray-200 flex items-center justify-between gap-3">
                                    <div className="pr-4">
                                        <Label className="block text-xs font-medium text-gray-700">
                                            Max expected speakers
                                        </Label>
                                        <p className="text-xs text-gray-500 mt-0.5">
                                            Upper bound for the speaker-count estimate. Generous is better than
                                            tight -- it does not force this many speakers, only caps the estimate.
                                        </p>
                                    </div>
                                    <Input
                                        type="number"
                                        min={2}
                                        value={maxSpeakers}
                                        onChange={(e) => setMaxSpeakers(Number(e.target.value))}
                                        onBlur={(e) => handleMaxSpeakersBlur(e.target.value)}
                                        className="w-20 shrink-0"
                                    />
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="pt-2 border-t border-gray-100">
                        <div className="flex items-center justify-between mt-4">
                            <div className="pr-4">
                                <Label className="block text-sm font-medium text-gray-700">
                                    Audio denoising
                                </Label>
                                <p className="text-xs text-gray-500 mt-0.5">
                                    Reduce background noise before transcription and diarization. Runs
                                    entirely locally, CPU-only. Requires downloading an additional model
                                    (~11MB) on first use.
                                </p>
                            </div>
                            <Switch
                                checked={denoisingEnabled}
                                disabled={isDenoisingToggleBusy}
                                onCheckedChange={handleDenoisingToggle}
                            />
                        </div>

                        {denoisingEnabled && (
                            <div className="mt-3 mx-1 p-3 rounded-md border border-gray-200 bg-gray-50">
                                {isDenoisingDownloading ? (
                                    <div>
                                        <p className="text-xs text-gray-600 mb-1">
                                            Downloading denoising model... {denoisingDownloadPercent}%
                                        </p>
                                        <Progress value={denoisingDownloadPercent} />
                                    </div>
                                ) : denoisingModelStatus === 'Available' ? (
                                    <p className="text-xs text-emerald-700">✓ Denoising model ready</p>
                                ) : (
                                    <div className="flex items-center justify-between gap-3">
                                        <p className="text-xs text-gray-600">
                                            {denoisingDownloadError
                                                ? `Download failed: ${denoisingDownloadError}`
                                                : 'Denoising model not downloaded yet. Recording will be blocked while denoising is on until this completes.'}
                                        </p>
                                        <Button
                                            type="button"
                                            variant="outline"
                                            size="sm"
                                            onClick={handleDownloadDenoisingModels}
                                            className="shrink-0"
                                        >
                                            {denoisingDownloadError ? 'Retry' : 'Download'}
                                        </Button>
                                    </div>
                                )}

                                <div className="mt-3 pt-3 border-t border-gray-200 flex items-center justify-between gap-3">
                                    <div className="pr-4">
                                        <Label className="block text-xs font-medium text-gray-700">
                                            Save debug audio files
                                        </Label>
                                        <p className="text-xs text-gray-500 mt-0.5">
                                            Also save the exact denoised signal used by transcription and
                                            diarization, alongside the recording. Uncompressed WAV files --
                                            adds real disk usage (roughly 1.4GB/hour for live recordings),
                                            off by default.
                                        </p>
                                    </div>
                                    <Switch
                                        checked={denoisingSaveDebugFiles}
                                        disabled={isDenoisingSaveDebugFilesBusy}
                                        onCheckedChange={handleDenoisingSaveDebugFilesToggle}
                                    />
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="pt-2 border-t border-gray-100">
                        <div className="mt-4">
                            <Label className="block text-sm font-medium text-gray-700">
                                Custom vocabulary
                            </Label>
                            <p className="text-xs text-gray-500 mt-0.5">
                                Bias transcription toward expected names, acronyms, and technical
                                jargon. Whisper only -- Parakeet has no vocabulary-bias support.
                                Managed here, not per-recording.
                            </p>
                        </div>

                        <div className="mt-3 mx-1">
                            <Label className="block text-xs font-medium text-gray-700 mb-1">
                                Active vocabulary
                            </Label>
                            <Select
                                value={activeVocabularyId ?? 'none'}
                                onValueChange={handleActiveVocabularyChange}
                            >
                                <SelectTrigger className="focus:ring-1 focus:ring-blue-500 focus:border-blue-500">
                                    <SelectValue placeholder="None" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="none">None</SelectItem>
                                    {vocabularies.map((vocabulary) => (
                                        <SelectItem key={vocabulary.id} value={vocabulary.id}>
                                            {vocabulary.name}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="mt-3 mx-1 p-3 rounded-md border border-gray-200 bg-gray-50">
                            <div className="flex items-center justify-between mb-2">
                                <Label className="block text-xs font-medium text-gray-700">
                                    Saved vocabularies
                                </Label>
                                <Button type="button" variant="outline" size="sm" onClick={openCreateVocabularyDialog}>
                                    + New vocabulary
                                </Button>
                            </div>
                            {vocabularies.length === 0 ? (
                                <p className="text-xs text-gray-500">No vocabularies yet.</p>
                            ) : (
                                <ul className="space-y-1">
                                    {vocabularies.map((vocabulary) => (
                                        <li key={vocabulary.id} className="flex items-center justify-between gap-2 text-xs">
                                            <span className="truncate">{vocabulary.name}</span>
                                            <div className="flex items-center gap-1 shrink-0">
                                                <Button
                                                    type="button"
                                                    variant="ghost"
                                                    size="sm"
                                                    onClick={() => openEditVocabularyDialog(vocabulary)}
                                                >
                                                    Edit
                                                </Button>
                                                <Button
                                                    type="button"
                                                    variant="ghost"
                                                    size="sm"
                                                    onClick={() => handleDeleteVocabulary(vocabulary)}
                                                >
                                                    Delete
                                                </Button>
                                            </div>
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            <Dialog open={isVocabularyDialogOpen} onOpenChange={setIsVocabularyDialogOpen}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>{editingVocabulary ? 'Edit vocabulary' : 'New vocabulary'}</DialogTitle>
                    </DialogHeader>
                    <div className="space-y-3">
                        <div>
                            <Label className="block text-sm font-medium text-gray-700 mb-1">
                                Name
                            </Label>
                            <Input
                                value={vocabularyNameInput}
                                onChange={(e) => setVocabularyNameInput(e.target.value)}
                                placeholder="e.g. Project Atlas"
                            />
                        </div>
                        <div>
                            <Label className="block text-sm font-medium text-gray-700 mb-1">
                                Terms
                            </Label>
                            <Textarea
                                value={vocabularyTermsInput}
                                onChange={(e) => setVocabularyTermsInput(e.target.value)}
                                placeholder="Keycloak, ForgeRock, SSO, ..."
                                rows={4}
                            />
                            <p className="text-xs text-gray-500 mt-1">
                                Names, acronyms, or jargon expected in this vocabulary's recordings.
                            </p>
                        </div>
                        {vocabularyError && (
                            <p className="text-xs text-red-600">{vocabularyError}</p>
                        )}
                    </div>
                    <DialogFooter>
                        <Button type="button" variant="outline" onClick={() => setIsVocabularyDialogOpen(false)}>
                            Cancel
                        </Button>
                        <Button type="button" onClick={handleSaveVocabulary} disabled={isVocabularySaving}>
                            {isVocabularySaving ? 'Saving...' : 'Save'}
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div >
    )
}








