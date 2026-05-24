import gradio as gr
from evonet_studio.engine import StudioEngine

def build_evaluation_ui(engine: StudioEngine):
    with gr.Tab("📈 Benchmark & Eval"):
        gr.Markdown("### Model Benchmark & Evaluation")
        gr.Markdown("Evaluate your fine-tuned models using statistical metrics or LLM-as-a-Judge.")
        
        with gr.Row():
            with gr.Column(scale=5):
                model_name = gr.Textbox(label="Model Path", value="outputs/evonet_model")
                dataset_path = gr.Textbox(label="Benchmark Dataset", value="wikitext")
                
                eval_method = gr.Radio(
                    choices=["Statistical (Perplexity)", "LLM-as-a-Judge"], 
                    value="Statistical (Perplexity)", 
                    label="Evaluation Method"
                )
                
                with gr.Group(visible=True) as stat_group:
                    metric = gr.Dropdown(choices=["Perplexity", "ROUGE"], value="Perplexity", label="Metric")
                
                with gr.Group(visible=False) as llm_group:
                    judge_provider = gr.Dropdown(choices=["Google Gemini", "OpenAI", "Anthropic", "Local (Ollama)"], value="Google Gemini", label="API Provider")
                    api_key = gr.Textbox(label="API Key (Leave empty for Local)", type="password")
                    
                    with gr.Row():
                        fetch_models_btn = gr.Button("🔄 Fetch Available Models")
                        judge_model = gr.Dropdown(choices=["gemini-1.5-flash"], value="gemini-1.5-flash", label="Judge Model", allow_custom_value=True)
                    
                    sample_size = gr.Slider(minimum=1, maximum=100, step=1, value=10, label="Sample Size (Prompts to eval)")
                
                eval_btn = gr.Button("📊 Run Evaluation", elem_classes=["primary"])
                
            with gr.Column(scale=4):
                eval_result = gr.Markdown("### Evaluation Results\n*No evaluation run yet.*")
        
        def toggle_groups(method):
            if method == "LLM-as-a-Judge":
                return gr.update(visible=False), gr.update(visible=True)
            return gr.update(visible=True), gr.update(visible=False)
            
        eval_method.change(fn=toggle_groups, inputs=[eval_method], outputs=[stat_group, llm_group])
        
        def fetch_models(provider, key):
            models = engine.fetch_llm_models(provider, key)
            if not models:
                return gr.update(choices=["gemini-1.5-flash", "gpt-4o"], value="gemini-1.5-flash")
            return gr.update(choices=models, value=models[0] if models else "")
            
        fetch_models_btn.click(fn=fetch_models, inputs=[judge_provider, api_key], outputs=[judge_model])
                
        def run_eval(m, d, method, met, prov, j_model, key, size):
            return engine.evaluate_model(m, d, method, met, prov, j_model, key, int(size))
            
        eval_btn.click(
            fn=run_eval, 
            inputs=[model_name, dataset_path, eval_method, metric, judge_provider, judge_model, api_key, sample_size], 
            outputs=[eval_result]
        )
