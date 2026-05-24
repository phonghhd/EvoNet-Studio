import gradio as gr
from evonet_studio.engine import StudioEngine

def build_arena_ui(engine: StudioEngine):
    with gr.Tab("⚔️ LLM Arena (DPO Synthesis)"):
        gr.Markdown("### Blind Test Models & Generate DPO Datasets")
        gr.Markdown("Compare two models blindly. Your votes are automatically saved as a high-quality DPO dataset.")
        
        with gr.Row():
            provider = gr.Dropdown(choices=["Gemini"], value="Gemini", label="API Provider")
            model_name = gr.Textbox(value="gemini-1.5-flash", label="Model ID")
            api_key = gr.Textbox(label="API Key", type="password")
            
        with gr.Row():
            prompt_input = gr.TextArea(label="Enter Prompt", lines=3, placeholder="Ask a complex question...")
            
        btn_fight = gr.Button("⚔️ Send to Arena", elem_classes=["primary"])
        
        with gr.Row():
            model_a_output = gr.TextArea(label="Model A", interactive=False, lines=10)
            model_b_output = gr.TextArea(label="Model B", interactive=False, lines=10)
            
        with gr.Row():
            btn_a = gr.Button("👈 Model A is Better")
            btn_tie = gr.Button("🤝 Tie (Skip)")
            btn_b = gr.Button("👉 Model B is Better")
            
        vote_status = gr.Markdown("Ready to vote.")
        
        # State to store current responses
        current_res_a = gr.State("")
        current_res_b = gr.State("")
        
        def run_arena(prompt, prov, key, model):
            res_a, res_b = engine.arena_chat(prompt, prov, key, model)
            return res_a, res_b, res_a, res_b
            
        btn_fight.click(
            fn=run_arena,
            inputs=[prompt_input, provider, api_key, model_name],
            outputs=[model_a_output, model_b_output, current_res_a, current_res_b]
        )
        
        def vote_a(prompt, a, b):
            if not a or not b: return "Cannot vote on empty responses."
            return engine.save_arena_vote(prompt, chosen=a, rejected=b)
            
        def vote_b(prompt, a, b):
            if not a or not b: return "Cannot vote on empty responses."
            return engine.save_arena_vote(prompt, chosen=b, rejected=a)
            
        btn_a.click(fn=vote_a, inputs=[prompt_input, current_res_a, current_res_b], outputs=[vote_status])
        btn_b.click(fn=vote_b, inputs=[prompt_input, current_res_a, current_res_b], outputs=[vote_status])
        btn_tie.click(fn=lambda: "Tie recorded (skipped).", outputs=[vote_status])
