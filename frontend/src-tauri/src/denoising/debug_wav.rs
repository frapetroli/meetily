//! Minimal streaming WAV writer for the denoising debug audio files (mono, 32-bit
//! float PCM). Not a general-purpose WAV writer -- just enough to append samples
//! incrementally, without holding a whole recording in memory, then patch the
//! RIFF/data chunk sizes in the header once at the end.
//!
//! Used to save, alongside the raw recording, the exact signal handed to ASR and to
//! diarization after denoising (ADR-0027) -- useful for inspecting/listening to what
//! each consumer actually received. Only relevant when denoising is enabled: the
//! signals would otherwise be identical to the raw recording, not worth a second copy.
//!
//! For the batch paths (import/retranscription), where the whole buffer is already in
//! memory, `sherpa_onnx::write()` is used directly instead -- this writer exists only
//! for the live path, where the recording's own audio.mp4 is deliberately never held
//! in memory all at once (`IncrementalAudioSaver` checkpoints to disk), and the
//! denoised debug copies must not regress that.

use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

pub struct DebugWavWriter {
    writer: BufWriter<File>,
    data_bytes_written: u32,
}

impl DebugWavWriter {
    /// Creates the file and writes a placeholder 44-byte WAV header (mono, 32-bit
    /// float PCM) -- the size fields are patched in by `finalize()`.
    pub fn create(path: &Path, sample_rate: u32) -> io::Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        write_header(&mut writer, sample_rate, 0)?;
        Ok(Self {
            writer,
            data_bytes_written: 0,
        })
    }

    /// Appends samples as little-endian 32-bit float PCM.
    pub fn append(&mut self, samples: &[f32]) -> io::Result<()> {
        for &s in samples {
            self.writer.write_all(&s.to_le_bytes())?;
        }
        self.data_bytes_written = self
            .data_bytes_written
            .saturating_add((samples.len() as u64 * 4).min(u32::MAX as u64) as u32);
        Ok(())
    }

    /// Flushes buffered writes and patches the RIFF/data chunk sizes in the header.
    /// Consumes `self` -- call exactly once, after the last `append`.
    pub fn finalize(mut self) -> io::Result<()> {
        self.writer.flush()?;
        let mut file = self
            .writer
            .into_inner()
            .map_err(|e| e.into_error())?;
        // RIFF chunk size, offset 4: 36 + data size.
        file.seek(SeekFrom::Start(4))?;
        file.write_all(&(36u32 + self.data_bytes_written).to_le_bytes())?;
        // data subchunk size, offset 40.
        file.seek(SeekFrom::Start(40))?;
        file.write_all(&self.data_bytes_written.to_le_bytes())?;
        file.sync_all()?;
        Ok(())
    }
}

fn write_header<W: Write>(w: &mut W, sample_rate: u32, data_bytes: u32) -> io::Result<()> {
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 32;
    let byte_rate = sample_rate * num_channels as u32 * (bits_per_sample as u32 / 8);
    let block_align = num_channels * (bits_per_sample / 8);

    w.write_all(b"RIFF")?;
    w.write_all(&(36u32 + data_bytes).to_le_bytes())?;
    w.write_all(b"WAVE")?;
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?; // fmt chunk size
    w.write_all(&3u16.to_le_bytes())?; // format tag 3 = IEEE float
    w.write_all(&num_channels.to_le_bytes())?;
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&byte_rate.to_le_bytes())?;
    w.write_all(&block_align.to_le_bytes())?;
    w.write_all(&bits_per_sample.to_le_bytes())?;
    w.write_all(b"data")?;
    w.write_all(&data_bytes.to_le_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn writes_valid_wav_roundtrip() {
        let path = std::env::temp_dir().join(format!(
            "denoising_debug_wav_test_{}_{}.wav",
            std::process::id(),
            "roundtrip"
        ));
        let mut writer = DebugWavWriter::create(&path, 48000).unwrap();
        writer.append(&[0.0, 0.5, -0.5, 1.0]).unwrap();
        writer.append(&[0.25]).unwrap();
        writer.finalize().unwrap();

        let mut buf = Vec::new();
        File::open(&path).unwrap().read_to_end(&mut buf).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(&buf[0..4], b"RIFF");
        assert_eq!(&buf[8..12], b"WAVE");
        let riff_size = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let data_size = u32::from_le_bytes(buf[40..44].try_into().unwrap());
        assert_eq!(data_size, 5 * 4); // 5 f32 samples
        assert_eq!(riff_size, 36 + data_size);
        assert_eq!(buf.len(), 44 + 5 * 4);

        // Spot-check the sample data itself round-trips correctly.
        let first_sample = f32::from_le_bytes(buf[44..48].try_into().unwrap());
        assert_eq!(first_sample, 0.0);
        let last_sample = f32::from_le_bytes(buf[60..64].try_into().unwrap());
        assert_eq!(last_sample, 0.25);
    }

    #[test]
    fn empty_writer_produces_valid_zero_length_wav() {
        let path = std::env::temp_dir().join(format!(
            "denoising_debug_wav_test_{}_{}.wav",
            std::process::id(),
            "empty"
        ));
        let writer = DebugWavWriter::create(&path, 16000).unwrap();
        writer.finalize().unwrap();

        let mut buf = Vec::new();
        File::open(&path).unwrap().read_to_end(&mut buf).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(buf.len(), 44);
        let data_size = u32::from_le_bytes(buf[40..44].try_into().unwrap());
        assert_eq!(data_size, 0);
    }
}
