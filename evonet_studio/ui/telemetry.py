import gradio as gr
from evonet_studio.engine import StudioEngine

def build_telemetry_ui(engine: StudioEngine):
    with gr.Tab("🔄 Data Flywheel (Telemetry)"):
        gr.Markdown("### Production Telemetry & Correction Station")
        gr.Markdown("Review poorly rated responses (Thumbs Down) from your production API, correct them, and instantly push them to the Alignment Dataset to close the feedback loop.")
        
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("#### 1. Poorly Rated Logs (Score = -1)")
                
                log_list = gr.Dataframe(
                    headers=["Log ID", "Prompt", "Score"],
                    interactive=False,
                    type="array"
                )
                refresh_logs_btn = gr.Button("🔄 Fetch Latest Logs")
                
            with gr.Column(scale=5):
                gr.Markdown("#### 2. Correction Station")
                selected_log_id = gr.Textbox(label="Selected Log ID", interactive=False)
                prompt_text = gr.TextArea(label="User Prompt", interactive=False, lines=3)
                rejected_text = gr.TextArea(label="AI's Bad Response (Rejected)", interactive=False, lines=4)
                
                gr.Markdown("✍️ **Write the Perfect Response:**")
                chosen_text = gr.TextArea(label="Your Corrected Response (Chosen)", interactive=True, lines=4)
                
                push_btn = gr.Button("🔄 Push to DPO Dataset", elem_classes=["primary"])
                status_out = gr.Markdown("")
                
        # Hidden state to store full logs
        raw_logs = gr.State([])
        
        def load_logs():
            logs = engine.get_telemetry_logs()
            bad_logs = [l for l in logs if l.get("score") == -1]
            table_data = []
            for l in bad_logs:
                # truncate prompt for table
                p = l.get("prompt", "")
                p_short = p[:50] + "..." if len(p) > 50 else p
                table_data.append([l.get("log_id"), p_short, str(l.get("score"))])
            return table_data, logs
            
        refresh_logs_btn.click(fn=load_logs, outputs=[log_list, raw_logs])
        
        def select_log(evt: gr.SelectData, logs):
            row_idx = evt.index[0]
            # Find the corresponding log
            bad_logs = [l for l in logs if l.get("score") == -1]
            if row_idx < len(bad_logs):
                target = bad_logs[row_idx]
                return target.get("log_id"), target.get("prompt"), target.get("response"), ""
            return "", "", "", ""
            
        log_list.select(fn=select_log, inputs=[raw_logs], outputs=[selected_log_id, prompt_text, rejected_text, chosen_text])
        
        def push_to_dpo(prompt, chosen, rejected):
            if not prompt or not chosen or not rejected:
                return "❌ Error: Missing data."
            msg = engine.convert_log_to_dpo(prompt, chosen, rejected)
            return msg
            
        push_btn.click(fn=push_to_dpo, inputs=[prompt_text, chosen_text, rejected_text], outputs=[status_out])
