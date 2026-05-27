import gradio as gr
from evonet_studio.engine import StudioEngine

def build_export_ui(engine: StudioEngine):
    with gr.Tab("📦 Export & Hub"):
        gr.Markdown("### Model Merging, Exporting, and Hub Push")
        gr.Markdown("Merge LoRA adapters into base models, export to GGUF, or push to HuggingFace Hub.")
        
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("#### 1. LoRA Merging")
                base_model_path = gr.Textbox(label="Base Model (Path or ID)", value=engine.get_models()[0] if engine.get_models() else "")
                lora_path = gr.Textbox(label="LoRA Adapter Path", value="outputs/evonet_model")
                merged_output_dir = gr.Textbox(label="Merged Output Directory", value="outputs/merged_model")
                merge_btn = gr.Button("🔗 Merge LoRA to Base Model", elem_classes=["primary"])
                
                gr.Markdown("#### 2. HuggingFace Hub Push")
                hf_token = gr.Textbox(label="HuggingFace Token (Write Access)", type="password")
                repo_id = gr.Textbox(label="Repository ID", placeholder="your-username/evonet-awesome-model")
                push_btn = gr.Button("☁️ Push Model to Hub")
                
                gr.Markdown("#### 3. GGUF Export (Requires Unsloth GPU)")
                gguf_model_path = gr.Textbox(label="Model Path to Export", value="outputs/merged_model")
                quantization = gr.Dropdown(choices=["q4_k_m", "q5_k_m", "q8_0", "f16"], value="q4_k_m", label="Quantization Method")
                export_btn = gr.Button("💾 Export to GGUF")
                
                gr.Markdown("#### 4. Push to Ollama (Local)")
                ollama_model_name = gr.Textbox(label="Ollama Model Name", value="evonet-model")
                ollama_btn = gr.Button("🦙 Push to Ollama")
            
            with gr.Column(scale=4):
                gr.Markdown("#### 📈 Export Console")
                log_output = gr.TextArea(
                    label="Status Logs",
                    interactive=False,
                    lines=25,
                    max_lines=30,
                    autoscroll=True,
                    elem_classes=["console-box"]
                )
                refresh_btn = gr.Button("🔄 Refresh Logs")

        def do_merge(base, lora, out):
            msg = engine.merge_lora(base, lora, out)
            return msg + "\n\n" + engine.get_logs()
            
        def do_push(model, token, repo):
            msg = engine.push_to_hub(model, token, repo)
            return msg + "\n\n" + engine.get_logs()

        def do_export(path, quant):
            msg = engine.export_gguf(path, quant)
            return msg + "\n\n" + engine.get_logs()
            
        def do_ollama(path, name):
            msg = engine.push_to_ollama(path, name)
            return msg + "\n\n" + engine.get_logs()
            
        def update_logs():
            return engine.get_logs()

        merge_btn.click(fn=do_merge, inputs=[base_model_path, lora_path, merged_output_dir], outputs=[log_output])
        push_btn.click(fn=do_push, inputs=[merged_output_dir, hf_token, repo_id], outputs=[log_output])
        export_btn.click(fn=do_export, inputs=[gguf_model_path, quantization], outputs=[log_output])
        ollama_btn.click(fn=do_ollama, inputs=[gguf_model_path, ollama_model_name], outputs=[log_output])
        refresh_btn.click(fn=update_logs, outputs=[log_output])
