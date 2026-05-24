import gradio as gr
from evonet_studio.engine import StudioEngine

def build_synthesis_ui(engine: StudioEngine):
    with gr.Tab("🧬 Data Synthesis"):
        gr.Markdown("### AI Data Synthesis & Distillation")
        gr.Markdown("Automatically generate high-quality `.jsonl` training datasets using LLMs (Knowledge Distillation).")
        
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("#### 1. Teacher Model Configuration")
                provider = gr.Dropdown(choices=["Google Gemini", "OpenAI", "Anthropic", "Local Ollama"], value="Google Gemini", label="Provider")
                api_key = gr.Textbox(label="API Key (Leave empty for Local)", type="password")
                
                with gr.Row():
                    fetch_models_btn = gr.Button("🔄 Fetch Models")
                    teacher_model = gr.Dropdown(choices=["gemini-1.5-flash"], value="gemini-1.5-flash", label="Teacher Model", allow_custom_value=True)
                
                gr.Markdown("#### 2. Dataset Requirements")
                topic = gr.Textbox(label="Topic / Description", placeholder="e.g., A psychological assistant that uses cognitive behavioral therapy.", lines=3)
                dataset_format = gr.Dropdown(choices=["Alpaca (Instruction, Input, Output)", "ChatML (Role, Content)"], value="Alpaca (Instruction, Input, Output)", label="Format")
                count = gr.Slider(minimum=10, maximum=1000, step=10, value=50, label="Number of Samples to Generate")
                
                output_file = gr.Textbox(label="Output JSONL File Path", value="dataset_synthetic.jsonl")
                
                generate_btn = gr.Button("🚀 Generate Dataset", elem_classes=["primary"])
                
            with gr.Column(scale=4):
                gr.Markdown("#### 📈 Synthesis Console")
                log_output = gr.TextArea(
                    label="Live Logs",
                    interactive=False,
                    lines=25,
                    max_lines=30,
                    autoscroll=True,
                    elem_classes=["console-box"]
                )
                refresh_btn = gr.Button("🔄 Refresh Logs")

        def fetch_models(prov, key):
            models = engine.fetch_llm_models(prov, key)
            if not models:
                return gr.update(choices=["gemini-1.5-flash", "gpt-4o"], value="gemini-1.5-flash")
            return gr.update(choices=models, value=models[0] if models else "")
            
        fetch_models_btn.click(fn=fetch_models, inputs=[provider, api_key], outputs=[teacher_model])
                
        def start_synthesis(prov, key, model, t, count_val, fmt, out):
            msg = engine.synthesize_data(prov, key, model, t, int(count_val), fmt, out)
            return msg + "\n\n" + engine.get_logs()
            
        def update_logs():
            return engine.get_logs()

        generate_btn.click(
            fn=start_synthesis,
            inputs=[provider, api_key, teacher_model, topic, count, dataset_format, output_file],
            outputs=[log_output]
        )
        refresh_btn.click(fn=update_logs, outputs=[log_output])
