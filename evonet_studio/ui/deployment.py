import gradio as gr
from evonet_studio.engine import StudioEngine

def build_deployment_ui(engine: StudioEngine):
    with gr.Tab("⚡ API Deployment"):
        gr.Markdown("### 1-Click Production API Server")
        gr.Markdown("Deploy your fine-tuned model as an OpenAI-compatible API server (`http://localhost:8000/v1/chat/completions`).")
        
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("#### 1. Server Configuration")
                model_path = gr.Textbox(label="Base Model Path", value="outputs/merged_model")
                
                gr.Markdown("#### 2. Multi-LoRA MoE (Optional)")
                gr.Markdown("Load multiple LoRA adapters into the Base Model. The Native Server will dynamically route questions to the right expert based on keywords.")
                loras_json = gr.Textbox(
                    label="LoRA Experts (JSON mapping: name -> path)",
                    placeholder='{"code": "outputs/lora_code", "math": "outputs/lora_math"}',
                    lines=2
                )
                
                engine_type = gr.Dropdown(choices=["vLLM (GPU Only - Extremely Fast)", "FastAPI + Transformers (CPU/GPU Fallback)"], value="FastAPI + Transformers (CPU/GPU Fallback)", label="Deployment Engine")
                
                with gr.Row():
                    start_btn = gr.Button("🚀 Start Server", elem_classes=["primary"])
                    stop_btn = gr.Button("⏹️ Stop Server")
                    
                server_status = gr.Markdown("🔴 **Status:** Offline")
                
                gr.Markdown("#### 2. Test Snippet")
                gr.Code(
                    language="python",
                    value='''import requests\n\nresp = requests.post(\n    "http://localhost:8000/v1/chat/completions",\n    json={"messages": [{"role": "user", "content": "Hello!"}]}\n)\nprint(resp.json())'''
                )
                
            with gr.Column(scale=4):
                gr.Markdown("#### 📈 Server Logs")
                log_output = gr.TextArea(
                    label="Live Logs",
                    interactive=False,
                    lines=20,
                    max_lines=25,
                    autoscroll=True,
                    elem_classes=["console-box"]
                )
                refresh_btn = gr.Button("🔄 Refresh Server Logs")

        def start_server(path, loras, etype):
            is_vllm = "vLLM" in etype
            msg = engine.start_server(path, is_vllm, loras)
            return "🟢 **Status:** " + msg, engine.get_server_logs()
            
        def stop_server():
            msg = engine.stop_server()
            return "🔴 **Status:** " + msg, engine.get_server_logs()
            
        def update_logs():
            return engine.get_server_logs()

        start_btn.click(fn=start_server, inputs=[model_path, loras_json, engine_type], outputs=[server_status, log_output])
        stop_btn.click(fn=stop_server, outputs=[server_status, log_output])
        refresh_btn.click(fn=update_logs, outputs=[log_output])
