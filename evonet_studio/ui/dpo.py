import gradio as gr
from evonet_studio.engine import StudioEngine

def build_alignment_ui(engine: StudioEngine):
    with gr.Tab("⚖️ Alignment Tuning"):
        gr.Markdown("### Advanced Alignment Tuning")
        gr.Markdown("Align your model to human preferences using DPO, ORPO, or KTO.")
        
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("#### 1. Model & Data")
                with gr.Row():
                    model_name = gr.Dropdown(
                        choices=engine.get_models(),
                        value=engine.get_models()[0] if engine.get_models() else "",
                        label="Policy Model",
                        allow_custom_value=True,
                        scale=4
                    )
                    refresh_hub_btn = gr.Button("🔄 Fetch Latest Models (HF Hub)", scale=1)
                
                algorithm = gr.Dropdown(
                    choices=["DPO (Direct Preference Optimization)", "ORPO (Odds Ratio Preference Optimization)", "KTO (Kahneman-Tversky Optimization)"],
                    value="DPO (Direct Preference Optimization)",
                    label="Alignment Algorithm"
                )
                
                dataset_path = gr.Textbox(
                    label="Preference Dataset",
                    placeholder="e.g., Anthropic/hh-rlhf (DPO/ORPO) or argilla/kto-mix-15k (KTO)",
                    value="Anthropic/hh-rlhf"
                )
                
                gr.Markdown("#### 2. Training Parameters")
                with gr.Row():
                    epochs = gr.Number(value=1, label="Epochs", minimum=1)
                    batch_size = gr.Number(value=1, label="Batch Size", minimum=1)
                    learning_rate = gr.Number(value=1e-5, label="Learning Rate")
                    beta = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.1, label="Beta (KL Penalty)")
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    with gr.Row():
                        lora_rank = gr.Slider(minimum=8, maximum=128, step=8, value=16, label="LoRA Rank (r)")
                        lora_alpha = gr.Slider(minimum=8, maximum=256, step=8, value=16, label="LoRA Alpha")
                    with gr.Row():
                        warmup_steps = gr.Number(value=5, label="Warmup Steps")
                        max_seq_length = gr.Number(value=1024, label="Max Sequence Length")
                
                output_dir = gr.Textbox(label="Output Directory", value="outputs/evonet_aligned_model")
                
                train_btn = gr.Button("🚀 Start Alignment Training", elem_classes=["primary"])
            
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

        def start_train(algo, m, d, e, b, lr, r, a, beta, w, max_len, out):
            is_orpo = "ORPO" in algo
            is_kto = "KTO" in algo
            
            msg = engine.start_alignment(m, d, int(e), int(b), lr, int(r), int(a), float(beta), int(w), int(max_len), out, is_orpo=is_orpo, is_kto=is_kto)
            return msg + "\n\n" + engine.get_logs()
            
        def update_logs():
            return engine.get_logs()

        train_btn.click(
            fn=start_train,
            inputs=[algorithm, model_name, dataset_path, epochs, batch_size, learning_rate, lora_rank, lora_alpha, beta, warmup_steps, max_seq_length, output_dir],
            outputs=[log_output]
        )
        refresh_btn.click(fn=update_logs, outputs=[log_output])
        
        def fetch_models():
            new_list = engine.fetch_trending_models()
            return gr.Dropdown(choices=new_list, value=new_list[0] if new_list else "")
            
        refresh_hub_btn.click(fn=fetch_models, outputs=[model_name])
