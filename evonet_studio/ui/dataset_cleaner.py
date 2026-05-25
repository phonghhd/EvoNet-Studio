import gradio as gr
from evonet_studio.engine import StudioEngine

def build_dataset_cleaner_ui(engine: StudioEngine):
    with gr.Tab("🧹 Data Cleaner"):
        gr.Markdown("### Dataset Diagnostics & Preprocessing")
        gr.Markdown("Analyze your `.jsonl` or `.json` datasets for duplicates and structural errors before training.")
        
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("#### 1. Analyze Dataset")
                dataset_path = gr.Textbox(
                    label="Local Dataset Path (e.g., outputs/my_data.jsonl)",
                    placeholder="outputs/data.jsonl"
                )
                analyze_btn = gr.Button("🔍 Analyze Dataset", elem_classes=["primary"])
                
                gr.Markdown("#### 2. Clean & Format")
                with gr.Row():
                    remove_dups = gr.Checkbox(label="Remove Exact Duplicates", value=True)
                    remove_short = gr.Checkbox(label="Remove Extremely Short Answers (< 5 words)", value=True)
                
                output_path = gr.Textbox(
                    label="Cleaned Output Path",
                    value="outputs/dataset_cleaned.jsonl"
                )
                clean_btn = gr.Button("✨ Clean & Export Dataset")
                
            with gr.Column(scale=4):
                gr.Markdown("#### 📊 Health Report")
                report_output = gr.Markdown("*Enter a dataset path and click Analyze...*")
                
        def run_analysis(path):
            return engine.analyze_dataset(path)
            
        def run_cleaning(path, out_path, rm_dup, rm_short):
            return engine.clean_dataset(path, out_path, rm_dup, rm_short)
            
        analyze_btn.click(fn=run_analysis, inputs=[dataset_path], outputs=[report_output])
        clean_btn.click(fn=run_cleaning, inputs=[dataset_path, output_path, remove_dups, remove_short], outputs=[report_output])
