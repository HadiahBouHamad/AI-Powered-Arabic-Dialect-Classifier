"""
Gradio Web Interface for Arabic Dialect Identification
Specialized for Saudi, Bahraini, Lebanese, MSA, and Other.
Powered by Gemini 3.1 Flash and in-memory audio extraction.
"""

import os
import tempfile
import static_ffmpeg

# Ensure ffmpeg and ffprobe are available in PATH for Gradio audio processing
static_ffmpeg.add_paths()

import gradio as gr
from dialect_engine import DialectDetector

# Verified, active public YouTube videos for quick one-click testing
EXAMPLES = [
    [
        "https://www.youtube.com/watch?v=F7-_TJ4R5YM",
        60,
        45
    ],
    [
        "https://www.youtube.com/watch?v=sox2griMZ7k",
        30,
        45
    ],
    [
        "https://www.youtube.com/watch?v=-ZS8rqaIN34",
        30,
        45
    ],
    [
        "https://www.youtube.com/watch?v=12h7hw_3DEk",
        30,
        45
    ]
]

def analyze_youtube(url, start_time, duration, custom_notes, model_choice):
    """
    Handler for YouTube link dialect identification with streaming updates.
    Yields 4 outputs:
      1. dominant_badge (gr.Markdown)
      2. dist_chart (gr.Label)
      3. audio_player (gr.Audio)
      4. analysis_markdown (gr.Markdown)
    """
    if not url or not url.strip():
        yield (
            "### ⚠️ Please enter a valid YouTube URL.",
            None,
            None,
            "**Error:** No YouTube URL provided. Please enter a link or click one of the examples below."
        )
        return

    status_log = f"⏳ Connecting to YouTube and extracting audio from {int(start_time)}s ({int(duration)}s slice)..."
    yield (
        "### ⏳ Processing YouTube audio in-memory...",
        None,
        None,
        status_log
    )

    try:
        current_detector = DialectDetector(model_name=model_choice)
        audio_bytes, meta = current_detector.extract_audio_from_youtube(
            url=url.strip(),
            start_seconds=int(start_time),
            duration_seconds=int(duration)
        )
    except Exception as e:
        err_msg = f"❌ Audio Extraction Failed: {str(e)}"
        yield (
            "### ❌ Extraction Error",
            None,
            None,
            f"### Error Details\n{err_msg}\n\n*Tip: Try using the 'File Upload' fallback tab if this YouTube video is region-restricted or unavailable.*"
        )
        return

    # Save audio slice to temporary file so user can listen to it in Gradio
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_audio.write(audio_bytes)
    temp_audio.close()

    info_header = (
        f"**Title:** {meta['title']}  \n"
        f"**Channel:** {meta['channel']}  \n"
        f"**Audio Slice:** {meta['start_time']}s to {meta['start_time'] + meta['slice_duration']}s "
        f"({meta['audio_size_bytes'] / 1024:.1f} KB in-memory)  \n\n"
    )

    status_log = info_header + "🎧 **Gemini 3.1 Flash is analyzing the dialect in real-time...**\n\n---\n\n"

    streamed_text = ""
    try:
        for chunk in current_detector.stream_dialect_identification(
            audio_bytes=audio_bytes,
            custom_instructions=custom_notes
        ):
            streamed_text += chunk
            parsed = current_detector.parse_analysis_result(streamed_text)
            hero_badge = f"### 🎙️ Dominant Dialect: **{parsed['dominant_dialect']}** ({parsed['confidence']})"
            dist_label = {k: v / 100.0 for k, v in parsed['distribution_raw'].items()}

            yield (
                hero_badge,
                dist_label,
                temp_audio.name,
                status_log + streamed_text
            )

    except Exception as e:
        yield (
            "### ❌ Gemini API Error",
            None,
            temp_audio.name,
            f"{status_log}\n\n❌ Error during Gemini inference: {str(e)}"
        )

def analyze_file_upload(file_obj, start_time, duration, custom_notes, model_choice):
    """
    Fallback handler for local audio/video file uploads with streaming updates.
    Yields 4 outputs:
      1. dominant_badge (gr.Markdown)
      2. dist_chart (gr.Label)
      3. audio_player (gr.Audio)
      4. analysis_markdown (gr.Markdown)
    """
    if file_obj is None:
        yield (
            "### ⚠️ Please upload an audio or video file.",
            None,
            None,
            "**Error:** No file selected. Please choose a local audio or video file."
        )
        return

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    status_log = f"⏳ Processing uploaded file: {os.path.basename(file_path)} (from {int(start_time)}s for {int(duration)}s)..."
    yield (
        "### ⏳ Processing audio file in-memory...",
        None,
        None,
        status_log
    )

    try:
        current_detector = DialectDetector(model_name=model_choice)
        audio_bytes, meta = current_detector.extract_audio_from_file(
            file_path=file_path,
            start_seconds=int(start_time),
            duration_seconds=int(duration)
        )
    except Exception as e:
        err_msg = f"❌ File Processing Failed: {str(e)}"
        yield (
            "### ❌ Processing Error",
            None,
            None,
            f"### Error Details\n{err_msg}"
        )
        return

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_audio.write(audio_bytes)
    temp_audio.close()

    info_header = (
        f"**File:** {meta['title']}  \n"
        f"**Audio Slice:** {meta['start_time']}s to {meta['start_time'] + meta['slice_duration']}s  \n\n"
    )

    status_log = info_header + "🎧 **Gemini 3.1 Flash is analyzing the dialect in real-time...**\n\n---\n\n"

    streamed_text = ""
    try:
        for chunk in current_detector.stream_dialect_identification(
            audio_bytes=audio_bytes,
            custom_instructions=custom_notes
        ):
            streamed_text += chunk
            parsed = current_detector.parse_analysis_result(streamed_text)
            hero_badge = f"### 🎙️ Dominant Dialect: **{parsed['dominant_dialect']}** ({parsed['confidence']})"
            dist_label = {k: v / 100.0 for k, v in parsed['distribution_raw'].items()}

            yield (
                hero_badge,
                dist_label,
                temp_audio.name,
                status_log + streamed_text
            )

    except Exception as e:
        yield (
            "### ❌ Gemini API Error",
            None,
            temp_audio.name,
            f"{status_log}\n\n❌ Error during Gemini inference: {str(e)}"
        )

# Gradio Styling
custom_css = """
.dominant-badge {
    text-align: center;
    padding: 14px;
    background: linear-gradient(135deg, #1e3a8a, #0284c7);
    color: white;
    border-radius: 12px;
    font-size: 1.4rem;
    font-weight: bold;
    margin-bottom: 12px;
}
.header-box {
    text-align: center;
    padding: 24px 16px;
    margin-bottom: 18px;
    background: linear-gradient(135deg, #0b1120 0%, #172554 50%, #1e3a8a 100%);
    border-radius: 16px;
    color: white;
    border: 1px solid rgba(56, 189, 248, 0.25);
    box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.25);
}
.header-box h1{
    color: #ffffff !important;
}
"""

def create_ui():
    with gr.Blocks(title="Arabic Dialect Classifier | مُشخِّص اللهجات العربية") as demo:
        with gr.Row():
            gr.HTML("""
                <div class="header-box">
                    <h1 style="font-size: 2.2rem; margin-bottom: 8px; color: #ffffff;">🎙️ مُشَخِّص اللهجات العربية | Arabic Dialect Classifier</h1>
                    <p style="font-size: 1.1rem; color: #e2e8f0; margin: 0;">
                        Identify Saudi (سعودي), Bahraini (بحريني), Lebanese (لبناني), and MSA (الفصحى) from YouTube podcasts and audio.
                    </p>
                    <p style="font-size: 0.95rem; color: #38bdf8; margin-top: 8px; font-weight: 500;">
                        Powered by Google Gemini 3.1 Flash • In-Memory Audio Streaming • Real-Time Dialect Breakdown
                    </p>
                </div>
            """)

        with gr.Row():
            # Left Column: Inputs & Podcast Controls
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.TabItem("📺 YouTube Link"):
                        yt_url_input = gr.Textbox(
                            label="YouTube Video / Podcast URL",
                            placeholder="https://www.youtube.com/watch?v=...",
                            lines=1
                        )
                        with gr.Row():
                            yt_start = gr.Slider(
                                minimum=0,
                                maximum=1800,
                                value=60,
                                step=15,
                                label="⏱️ Start Offset (Seconds)",
                                info="Skip intro music or ads (e.g. 60s = 1 min in)"
                            )
                            yt_duration = gr.Slider(
                                minimum=15,
                                maximum=180,
                                value=45,
                                step=15,
                                label="⏳ Sample Duration (Seconds)",
                                info="Audio slice length (45–90s is ideal)"
                            )

                        yt_analyze_btn = gr.Button("⚡ Analyze Video / Audio", variant="primary", size="lg")

                        gr.Markdown("#### 💡 Quick Test Examples (Click to Load):")
                        gr.Examples(
                            examples=EXAMPLES,
                            inputs=[yt_url_input, yt_start, yt_duration],
                            label="Verified Arabic Podcasts & Clips"
                        )

                    with gr.TabItem("📁 Direct File Upload (Fallback)"):
                        gr.Markdown("Use this if a YouTube video is age-restricted, blocked, or unavailable.")
                        file_input = gr.File(
                            label="Upload Audio or Video File (MP3, WAV, M4A, MP4)",
                            file_types=["audio", "video"]
                        )
                        with gr.Row():
                            file_start = gr.Slider(
                                minimum=0,
                                maximum=600,
                                value=0,
                                step=15,
                                label="⏱️ Start Offset (Seconds)"
                            )
                            file_duration = gr.Slider(
                                minimum=15,
                                maximum=180,
                                value=45,
                                step=15,
                                label="⏳ Sample Duration (Seconds)"
                            )
                        file_analyze_btn = gr.Button("⚡ Analyze Uploaded File", variant="primary", size="lg")

                with gr.Accordion("⚙️ Advanced Settings & Custom Context", open=False):
                    model_selector = gr.Dropdown(
                        choices=["gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"],
                        value="gemini-3.1-flash-lite",
                        label="Gemini Flash Model Version"
                    )
                    custom_notes_input = gr.Textbox(
                        label="Optional Context or Hints",
                        placeholder="e.g. 'Pay special attention to speaker 2' or 'Verify if speaker is from Hejaz'",
                        lines=2
                    )

            # Right Column: Results & Evidence
            with gr.Column(scale=6):
                dominant_badge = gr.Markdown("### 🎙️ Ready for audio input", elem_classes=["dominant-badge"])
                
                dist_chart = gr.Label(
                    label="📊 Dialect Distribution Breakdown (%)",
                    num_top_classes=5
                )

                audio_player = gr.Audio(
                    label="🎧 Extracted Audio Excerpt (Listen to what Gemini heard)",
                    type="filepath",
                    interactive=False
                )

                analysis_markdown = gr.Markdown(
                    value="*Analysis and linguistic evidence will stream here in real time...*",
                    label="📝 Linguistic Analysis & Detailed Breakdown"
                )

        # Wire YouTube Tab
        yt_analyze_btn.click(
            fn=analyze_youtube,
            inputs=[yt_url_input, yt_start, yt_duration, custom_notes_input, model_selector],
            outputs=[dominant_badge, dist_chart, audio_player, analysis_markdown]
        )

        # Wire File Upload Tab
        file_analyze_btn.click(
            fn=analyze_file_upload,
            inputs=[file_input, file_start, file_duration, custom_notes_input, model_selector],
            outputs=[dominant_badge, dist_chart, audio_player, analysis_markdown]
        )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(theme=gr.themes.Base(), css=custom_css, share=False)
