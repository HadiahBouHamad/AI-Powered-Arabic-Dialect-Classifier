"""
Gradio Web Interface for Arabic Dialect & Multi-Speaker Identification
Open Arabic dialect detection with per-speaker percentages.
Powered by the latest working Gemini model and in-memory audio extraction.
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
        "https://www.youtube.com/watch?v=2tYQo-kPs-Y",
        "01:05:40",
        60
    ],
    [
        "https://www.youtube.com/watch?v=sox2griMZ7k",
        "00:30",
        60
    ],
    [
        "https://www.youtube.com/watch?v=-ZS8rqaIN34",
        "00:30",
        60
    ],
    [
        "https://www.youtube.com/watch?v=12h7hw_3DEk",
        "00:30",
        60
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

    start_sec = DialectDetector.parse_time_offset(start_time)
    dur_sec = int(duration)
    formatted_start = DialectDetector.format_timestamp(start_sec)

    status_log = f"⏳ Connecting to YouTube and extracting audio from {formatted_start} ({start_sec}s, {dur_sec}s slice)..."
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
            start_seconds=start_sec,
            duration_seconds=dur_sec
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

    t_start_fmt = DialectDetector.format_timestamp(meta['start_time'])
    t_end_fmt = DialectDetector.format_timestamp(meta['start_time'] + meta['slice_duration'])
    info_header = (
        f"**Title:** {meta['title']}  \n"
        f"**Channel:** {meta['channel']}  \n"
        f"**Model:** `{current_detector.model_name}`  \n"
        f"**Audio Slice:** {t_start_fmt} to {t_end_fmt} "
        f"({meta['start_time']}s to {meta['start_time'] + meta['slice_duration']}s • "
        f"{meta['audio_size_bytes'] / 1024:.1f} KB in-memory)  \n\n"
    )

    status_log = info_header + f"🎧 **Gemini ({current_detector.model_name}) is analyzing the dialect in real-time...**\n\n---\n\n"

    streamed_text = ""
    try:
        for chunk in current_detector.stream_dialect_identification(
            audio_bytes=audio_bytes,
            custom_instructions=custom_notes
        ):
            streamed_text += chunk
            parsed = current_detector.parse_analysis_result(streamed_text)
            
            if parsed.get("speakers") and len(parsed["speakers"]) > 1:
                spk_parts = [f"{s['speaker']}: {s['dialect']} ({s['percentage']:.0f}%)" for s in parsed["speakers"]]
                spk_str = " • ".join(spk_parts)
                hero_badge = f"### 🎙️ Dominant: **{parsed['dominant_dialect']}** ({parsed['confidence']})  \n<span style='font-size:0.95rem; font-weight:normal; opacity:0.9;'>👥 {spk_str}</span>"
            elif parsed.get("speakers") and len(parsed["speakers"]) == 1:
                s = parsed["speakers"][0]
                hero_badge = f"### 🎙️ {s['speaker']} Dialect: **{s['dialect']}** ({s['percentage']:.0f}%)"
            elif parsed['dominant_dialect'] != "Analyzing...":
                hero_badge = f"### 🎙️ Dominant Dialect: **{parsed['dominant_dialect']}** ({parsed['confidence']})"
            else:
                hero_badge = "### 🎙️ Analyzing Dialect & Speakers..."

            dist_label = {k: v / 100.0 for k, v in parsed['distribution_raw'].items()} if parsed['distribution_raw'] else None

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

    start_sec = DialectDetector.parse_time_offset(start_time)
    dur_sec = int(duration)
    formatted_start = DialectDetector.format_timestamp(start_sec)

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    status_log = f"⏳ Processing uploaded file: {os.path.basename(file_path)} (from {formatted_start} / {start_sec}s for {dur_sec}s)..."
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
            start_seconds=start_sec,
            duration_seconds=dur_sec
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

    t_start_fmt = DialectDetector.format_timestamp(meta['start_time'])
    t_end_fmt = DialectDetector.format_timestamp(meta['start_time'] + meta['slice_duration'])
    info_header = (
        f"**File:** {meta['title']}  \n"
        f"**Model:** `{current_detector.model_name}`  \n"
        f"**Audio Slice:** {t_start_fmt} to {t_end_fmt} "
        f"({meta['start_time']}s to {meta['start_time'] + meta['slice_duration']}s)  \n\n"
    )

    status_log = info_header + f"🎧 **Gemini ({current_detector.model_name}) is analyzing the dialect in real-time...**\n\n---\n\n"

    streamed_text = ""
    try:
        for chunk in current_detector.stream_dialect_identification(
            audio_bytes=audio_bytes,
            custom_instructions=custom_notes
        ):
            streamed_text += chunk
            parsed = current_detector.parse_analysis_result(streamed_text)
            
            if parsed.get("speakers") and len(parsed["speakers"]) > 1:
                spk_parts = [f"{s['speaker']}: {s['dialect']} ({s['percentage']:.0f}%)" for s in parsed["speakers"]]
                spk_str = " • ".join(spk_parts)
                hero_badge = f"### 🎙️ Dominant: **{parsed['dominant_dialect']}** ({parsed['confidence']})  \n<span style='font-size:0.95rem; font-weight:normal; opacity:0.9;'>👥 {spk_str}</span>"
            elif parsed.get("speakers") and len(parsed["speakers"]) == 1:
                s = parsed["speakers"][0]
                hero_badge = f"### 🎙️ {s['speaker']} Dialect: **{s['dialect']}** ({s['percentage']:.0f}%)"
            elif parsed['dominant_dialect'] != "Analyzing...":
                hero_badge = f"### 🎙️ Dominant Dialect: **{parsed['dominant_dialect']}** ({parsed['confidence']})"
            else:
                hero_badge = "### 🎙️ Analyzing Dialect & Speakers..."

            dist_label = {k: v / 100.0 for k, v in parsed['distribution_raw'].items()} if parsed['distribution_raw'] else None

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
    with gr.Blocks(title="Arabic Dialect & Speaker Classifier | مُشخِّص اللهجات العربية") as demo:
        with gr.Row():
            gr.HTML("""
                <div class="header-box">
                    <h1 style="font-size: 2.2rem; margin-bottom: 8px; color: #ffffff;">🎙️ مُشَخِّص اللهجات والمتحدثين | Arabic Dialect & Speaker Identifier</h1>
                    <p style="font-size: 1.1rem; color: #e2e8f0; margin: 0;">
                        Open Arabic dialect identification and per-speaker percentage breakdowns from YouTube podcasts and audio.
                    </p>
                    <p style="font-size: 0.95rem; color: #38bdf8; margin-top: 8px; font-weight: 500;">
                        Powered by Google Gemini (Latest Working Model) • Open Dialect Classification • Multi-Speaker Diarization
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
                            yt_start = gr.Textbox(
                                label="⏱️ Start Time (Minute or Timestamp)",
                                value="01:00",
                                placeholder="e.g. 35:00, 1:15:00, 45m, or 45",
                                info="Start at ANY minute (supports 'MM:SS', 'HH:MM:SS', '45m', or plain minute like '45')"
                            )
                            yt_duration = gr.Slider(
                                minimum=15,
                                maximum=180,
                                value=60,
                                step=15,
                                label="⏳ Sample Duration (Seconds)",
                                info="45–60s for 1-2 speakers, 90–120s recommended for 3–4+ speaker panels"
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
                            file_start = gr.Textbox(
                                label="⏱️ Start Time (Minute or Timestamp)",
                                value="00:00",
                                placeholder="e.g. 35:00, 1:15:00, 45m, or 0",
                                info="Start at ANY minute (supports 'MM:SS', 'HH:MM:SS', '45m', or plain minute like '45')"
                            )
                            file_duration = gr.Slider(
                                minimum=15,
                                maximum=180,
                                value=60,
                                step=15,
                                label="⏳ Sample Duration (Seconds)",
                                info="45–60s for 1-2 speakers, 90–120s recommended for 3–4+ speaker panels"
                            )
                        file_analyze_btn = gr.Button("⚡ Analyze Uploaded File", variant="primary", size="lg")

                with gr.Accordion("⚙️ Advanced Settings & Custom Context", open=False):
                    model_selector = gr.Dropdown(
                        choices=["Auto (Latest Working Model)", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
                        value="Auto (Latest Working Model)",
                        label="Gemini Model"
                    )
                    custom_notes_input = gr.Textbox(
                        label="Optional Context or Hints",
                        placeholder="e.g. 'This video has 4 speakers in a panel discussion' or 'Pay special attention to speaker 3'",
                        lines=2
                    )

            # Right Column: Results & Evidence
            with gr.Column(scale=6):
                dominant_badge = gr.Markdown("### 🎙️ Ready for audio input", elem_classes=["dominant-badge"])
                
                dist_chart = gr.Label(
                    label="📊 Speaker & Dialect Breakdown (%)",
                    num_top_classes=8
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
