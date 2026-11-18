# 🎙️ Arabic Dialect Classifier (مُشخِّص اللهجات العربية)

A high-performance Python solution that identifies Arabic dialects (**Saudi**, **Bahraini**, **Lebanese**, and **MSA**, plus **Other**) from YouTube links and podcasts using **Google Gemini 3.1 Flash**.

---

## 🌟 Key Highlights

- **Direct In-Memory Streaming**: Uses `yt-dlp` to extract the live audio stream URL and pipes a targeted slice (e.g. 60–90 seconds) directly through `ffmpeg` in memory. **Zero video files downloaded to disk.**
- **Long-Form Podcast Support**: Set arbitrary start offsets (e.g., skip 60s of intro music/sponsors) and slice lengths.
- **Strict 4-Class + Other Focus**:
  - **Saudi** (Najdi, Hejazi, Sharqawi, Southern)
  - **Bahraini** (Bahrani, Sunni/Gulf Bahraini)
  - **Lebanese** (Levantine / Shami)
  - **MSA** (Modern Standard Arabic / الفصحى)
  - **Other** (Egyptian, Moroccan, Iraqi, etc., or Non-Arabic)
- **Percentage Distribution (Option C)**: Provides exact estimated percentages across all target classes that sum to 100%.
- **Arabic-First Evaluation**: Explicitly ignores foreign loanwords and code-switching (French, English), focusing strictly on Arabic phonology, morphology, vocabulary, and cadence.
- **Real-Time Streaming Output**: Watch Gemini stream its phonetic observations and linguistic evidence token-by-token.
- **Built-in Fallbacks**:
  - Direct local audio/video file upload (MP3, WAV, M4A, MP4) if YouTube restricts a video.
  - Automatic model fallback chain (`gemini-3.1-flash-lite` -> `gemini-3.5-flash` -> `gemini-3.8-flash`).

---

## 🛠️ Installation & Setup

1. **Prerequisites**: Python 3.10+ and a Gemini API Key.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements_dialect.txt
   ```
3. **Set your API Key** in `.env`:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   # or GOOGLE_API_KEY=your_key_here
   ```

---

## 🚀 How to Run

### 1. Launch the Gradio Web Interface
```bash
python app_dialect.py
```
Open `http://localhost:7860` in your browser. You can enter any YouTube link or click one of the pre-loaded examples.

### 2. Run via Command Line (CLI)
```bash
python cli_dialect.py --url "https://www.youtube.com/watch?v=F7-_TJ4R5YM" --start 60 --duration 90
```

---

## 📁 File Structure

- `dialect_engine.py`: Core engine handling in-memory `yt-dlp` stream piping, `ffmpeg` slicing, Gemini multimodal prompting, and result parsing.
- `app_dialect.py`: Full-featured Gradio web interface with interactive audio player, distribution charts, and live stream updates.
- `cli_dialect.py`: Standalone command-line tool for quick terminal-based analysis.
- `requirements_dialect.txt`: Dependencies list.
