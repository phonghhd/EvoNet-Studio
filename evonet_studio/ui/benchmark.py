import gradio as gr
from evonet_studio.engine import StudioEngine

def build_benchmark_ui(engine: StudioEngine):
    with gr.Tab("📊 Automated Benchmark"):
        gr.Markdown("### Academic Evaluation Suite (Powered by lm-eval)")
        gr.Markdown("Test your fine-tuned models on standardized academic benchmarks (MMLU, GSM8k, etc.) to get an objective score.")
        
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("#### 1. Target Model")
                model_path = gr.Textbox(
                    label="Local Model Path or HuggingFace ID",
                    value="outputs/evonet_model"
                )
                
                gr.Markdown("#### 2. Benchmark Tasks")
                tasks = gr.CheckboxGroup(
                    choices=["mmlu", "gsm8k", "arc_easy", "arc_challenge", "hellaswag", "truthfulqa_mc2"],
                    value=["arc_easy"],
                    label="Select Academic Tasks"
                )
                
                with gr.Row():
                    num_fewshot = gr.Number(value=0, label="Few-shot Examples (k)", minimum=0)
                    limit = gr.Number(value=10, label="Limit (Number of questions per task, set 0 for all)", minimum=0)
                
                gr.Markdown("> *Note: Running full benchmarks requires installing the `lm_eval` package.*")
                
                run_btn = gr.Button("🚀 Run Benchmark", elem_classes=["primary"])
                
            with gr.Column(scale=4):
                gr.Markdown("#### 📈 Benchmark Results")
                log_output = gr.TextArea(
                    label="Live Logs",
                    interactive=False,
                    lines=15,
                    max_lines=20,
                    autoscroll=True,
                    elem_classes=["console-box"]
                )
                refresh_btn = gr.Button("🔄 Refresh Logs")
                
        def run_bench(path, t, k, l):
            msg = engine.run_benchmark(path, t, int(k), int(l))
            return msg + "\n\n" + engine.get_logs()
            
        def update_logs():
            return engine.get_logs()
            
        run_btn.click(
            fn=run_bench,
            inputs=[model_path, tasks, num_fewshot, limit],
            outputs=[log_output]
        )
        refresh_btn.click(fn=update_logs, outputs=[log_output])
