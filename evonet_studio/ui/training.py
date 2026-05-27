import gradio as gr
from evonet_studio.engine import StudioEngine

def build_training_ui(engine: StudioEngine):
    with gr.Tab("🚀 Supervised Fine-Tuning"):
        gr.Markdown("### SFT Training Pipeline")
        gr.Markdown("Train models on instruction datasets using Unsloth optimized kernels.")
        
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("#### 1. Model & Data")
                with gr.Row():
                    model_name = gr.Dropdown(
                        choices=engine.get_models(),
                        value=engine.get_models()[0] if engine.get_models() else "",
                        label="Base Model",
                        allow_custom_value=True,
                        scale=4
                    )
                    refresh_hub_btn = gr.Button("🔄 Fetch Latest Models (HF Hub)", scale=1)
                
                gr.Markdown("#### 2. Training Mode & Dataset")
                training_mode = gr.Radio(
                    choices=["Supervised Fine-Tuning (SFT / Q&A)", "Continued Pre-Training (CPT / Raw Text)"],
                    value="Supervised Fine-Tuning (SFT / Q&A)",
                    label="Select Training Mode"
                )
                dataset_path = gr.Textbox(
                    label="Dataset Path or HuggingFace ID",
                    placeholder="e.g., yahma/alpaca-cleaned (SFT) or outputs/raw_corpus.txt (CPT)",
                    value="yahma/alpaca-cleaned"
                )
                
                with gr.Row():
                    is_vision = gr.Checkbox(label="Is Vision-Language Model (VLM)?", value=False)
                    is_bitnet = gr.Checkbox(label="🔥 Enable 1.58-bit Compression (BitNet)", value=False)
                
                gr.Markdown("#### 🌟 Enterprise Features")
                with gr.Row():
                    is_distributed = gr.Checkbox(label="🚀 Enable Distributed Training (DeepSpeed ZeRO-3)", value=False, elem_classes=["enterprise-feature"])

                gr.Markdown("#### 3. Basic Hyperparameters")
                suggest_hp_btn = gr.Button("✨ Auto-Suggest Hyperparameters")
                with gr.Row():
                    epochs = gr.Number(value=3, label="Epochs", minimum=1)
                    batch_size = gr.Number(value=2, label="Batch Size", minimum=1)
                    learning_rate = gr.Number(value=2e-4, label="Learning Rate")
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    with gr.Row():
                        lora_rank = gr.Slider(minimum=8, maximum=128, step=8, value=16, label="LoRA Rank (r)")
                        lora_alpha = gr.Slider(minimum=8, maximum=256, step=8, value=16, label="LoRA Alpha")
                    with gr.Row():
                        warmup_steps = gr.Number(value=10, label="Warmup Steps")
                        max_seq_length = gr.Number(value=2048, label="Max Sequence Length")
                
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value="outputs/evonet_model"
                )
                
                train_btn = gr.Button("🚀 Start Training", elem_classes=["primary"])
            
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

        # Callbacks
        def do_suggest_hp(path):
            e, b, lr = engine.suggest_hyperparameters(path)
            return e, b, lr, engine.get_logs()
            
        suggest_hp_btn.click(
            fn=do_suggest_hp,
            inputs=[dataset_path],
            outputs=[epochs, batch_size, learning_rate, log_output]
        )

        def start_train(m, mode, d, is_vlm, is_bitnet_val, is_dist, e, b, lr, r, a, w, max_len, out):
            is_cpt = "CPT" in mode
            msg = engine.start_training(m, d, int(e), int(b), lr, int(r), int(a), int(w), int(max_len), out, is_vision=is_vlm, is_bitnet=is_bitnet_val, is_cpt=is_cpt, is_distributed=is_dist)
            return msg + "\n\n" + engine.get_logs()
            
        def update_logs():
            return engine.get_logs()

        train_btn.click(
            fn=start_train,
            inputs=[model_name, training_mode, dataset_path, is_vision, is_bitnet, is_distributed, epochs, batch_size, learning_rate, lora_rank, lora_alpha, warmup_steps, max_seq_length, output_dir],
            outputs=[log_output]
        )
        
        refresh_btn.click(
            fn=update_logs,
            outputs=[log_output]
        )
        
        def fetch_models():
            new_list = engine.fetch_trending_models()
            return gr.Dropdown(choices=new_list, value=new_list[0] if new_list else "")
            
        refresh_hub_btn.click(fn=fetch_models, outputs=[model_name])
