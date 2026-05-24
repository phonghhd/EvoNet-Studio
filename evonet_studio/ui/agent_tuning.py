import gradio as gr
from evonet_studio.engine import StudioEngine

def build_agent_tuning_ui(engine: StudioEngine):
    with gr.Tab("🤖 Agent & Swarm Studio"):
        gr.Markdown("### Agentic Fine-Tuning & Multi-Agent Swarm")
        gr.Markdown("Train models for tool calling, or test them in a Multi-Agent Swarm environment.")
        
        with gr.Row():
            with gr.Column(scale=5):
                with gr.Accordion("✨ Auto-Prompt Optimizer (DSPy Style)", open=False):
                    gr.Markdown("Auto-generate a highly detailed System Prompt using an LLM API.")
                    with gr.Row():
                        opt_provider = gr.Dropdown(choices=["Gemini"], value="Gemini", label="Provider")
                        opt_model = gr.Textbox(value="gemini-1.5-flash", label="Model")
                        opt_api_key = gr.Textbox(label="API Key", type="password")
                    
                    raw_idea = gr.Textbox(label="Raw Idea (e.g. Bot đặt vé máy bay)", lines=2)
                    optimize_btn = gr.Button("✨ Generate Super Prompt")
                    optimized_prompt = gr.TextArea(label="Optimized System Prompt", interactive=True, lines=5)
                    
                    optimize_btn.click(
                        fn=lambda p, k, m, r: engine.optimize_prompt(r, p, k, m),
                        inputs=[opt_provider, opt_api_key, opt_model, raw_idea],
                        outputs=[optimized_prompt]
                    )
            
                gr.Markdown("#### 1. Model & Tool Dataset")
                with gr.Row():
                    model_name = gr.Dropdown(
                        choices=engine.get_models(),
                        value=engine.get_models()[0] if engine.get_models() else "",
                        label="Base Model",
                        allow_custom_value=True,
                        scale=4
                    )
                    refresh_hub_btn = gr.Button("🔄 Fetch Latest Models (HF Hub)", scale=1)
                
                dataset_path = gr.Textbox(
                    label="Tool-Calling Dataset",
                    placeholder="e.g., glaiveai/glaive-function-calling-v2",
                    value="glaiveai/glaive-function-calling-v2"
                )
                
                gr.Markdown("#### 2. Training Parameters")
                with gr.Row():
                    epochs = gr.Number(value=3, label="Epochs", minimum=1)
                    batch_size = gr.Number(value=2, label="Batch Size", minimum=1)
                    learning_rate = gr.Number(value=2e-5, label="Learning Rate")
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    with gr.Row():
                        lora_rank = gr.Slider(minimum=8, maximum=128, step=8, value=32, label="LoRA Rank (r)")
                        lora_alpha = gr.Slider(minimum=8, maximum=256, step=8, value=32, label="LoRA Alpha")
                    with gr.Row():
                        warmup_steps = gr.Number(value=10, label="Warmup Steps")
                        max_seq_length = gr.Number(value=2048, label="Max Sequence Length")
                    
                    tool_format = gr.Dropdown(
                        choices=["ChatML", "Llama-3-Instruct", "OpenAI-JSON"],
                        value="ChatML",
                        label="System Prompt Format"
                    )
                
                output_dir = gr.Textbox(label="Output Directory", value="outputs/evonet_agent_model")
                
                train_btn = gr.Button("🚀 Start Agent Tuning", elem_classes=["primary"])
            
            with gr.Column(scale=4):
                gr.Markdown("#### 📈 Training Console")
                log_output = gr.TextArea(
                    label="Live Logs",
                    interactive=False,
                    lines=25,
                    max_lines=30,
                    autoscroll=True,
                    elem_classes=["console-box"]
                )
                refresh_btn = gr.Button("🔄 Refresh Logs")
                
        gr.Markdown("---")
        gr.Markdown("### 🐝 Multi-Agent Swarm Playground")
        with gr.Row():
            with gr.Column(scale=5):
                swarm_prompt = gr.Textbox(label="Swarm Task", placeholder="e.g., Lên kế hoạch marketing cho một sản phẩm AI mới", lines=3)
                swarm_model = gr.Textbox(label="Model to run Swarm", value="outputs/evonet_agent_model")
                run_swarm_btn = gr.Button("🧠 Run Swarm Optimization", elem_classes=["primary"])
                
            with gr.Column(scale=4):
                swarm_output = gr.TextArea(label="Swarm Collaboration Output", interactive=False, lines=10)

        def start_agent_train(m, d, e, b, lr, r, a, w, max_len, fmt, out):
            msg = engine.start_agent_tuning(m, d, int(e), int(b), lr, int(r), int(a), int(w), int(max_len), fmt, out)
            return msg + "\n\n" + engine.get_logs()
            
        def run_swarm(task, model_path):
            return engine.run_swarm_test(task, model_path)
            
        def update_logs():
            return engine.get_logs()

        train_btn.click(
            fn=start_agent_train,
            inputs=[model_name, dataset_path, epochs, batch_size, learning_rate, lora_rank, lora_alpha, warmup_steps, max_seq_length, tool_format, output_dir],
            outputs=[log_output]
        )
        refresh_btn.click(fn=update_logs, outputs=[log_output])
        run_swarm_btn.click(fn=run_swarm, inputs=[swarm_prompt, swarm_model], outputs=[swarm_output])
        
        def fetch_models():
            new_list = engine.fetch_trending_models()
            return gr.Dropdown(choices=new_list, value=new_list[0] if new_list else "")
            
        refresh_hub_btn.click(fn=fetch_models, outputs=[model_name])
