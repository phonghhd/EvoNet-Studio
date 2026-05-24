import gradio as gr
from evonet_studio.engine import StudioEngine

def build_agent_tuning_ui(engine: StudioEngine):
    with gr.Tab("🤖 Agent Tuning"):
        gr.Markdown("### Agentic Fine-Tuning (Tool Calling / ReAct)")
        gr.Markdown("Train models to understand JSON schemas and execute tool calls accurately.")
        
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("#### 1. Model & Tool Dataset")
                model_name = gr.Dropdown(
                    choices=engine.get_models(),
                    value=engine.get_models()[0] if engine.get_models() else "",
                    label="Base Model",
                    allow_custom_value=True
                )
                
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

        def start_agent_train(m, d, e, b, lr, r, a, w, max_len, fmt, out):
            msg = engine.start_agent_tuning(m, d, int(e), int(b), lr, int(r), int(a), int(w), int(max_len), fmt, out)
            return msg + "\n\n" + engine.get_logs()
            
        def update_logs():
            return engine.get_logs()

        train_btn.click(
            fn=start_agent_train,
            inputs=[model_name, dataset_path, epochs, batch_size, learning_rate, lora_rank, lora_alpha, warmup_steps, max_seq_length, tool_format, output_dir],
            outputs=[log_output]
        )
        refresh_btn.click(fn=update_logs, outputs=[log_output])
