import gradio as gr
from evonet_studio.engine import StudioEngine

def build_cloud_sync_ui(engine: StudioEngine):
    with gr.Tab("☁️ Cloud Sync"):
        gr.Markdown("### Workspace Backup & Restore (HuggingFace Hub)")
        gr.Markdown("Safely backup your entire `outputs/` folder (datasets, logs, LoRA adapters) to a Private HuggingFace Dataset repository so you never lose your work.")
        
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("#### 1. HuggingFace Credentials")
                hf_token = gr.Textbox(
                    label="HuggingFace Access Token (Write Permission)",
                    placeholder="hf_...",
                    type="password"
                )
                repo_id = gr.Textbox(
                    label="HuggingFace Dataset Repo ID (e.g. username/my-evonet-backup)",
                    placeholder="username/my-evonet-backup"
                )
                
                gr.Markdown("> *Note: Please create an empty **Dataset** repository on HuggingFace first. It is highly recommended to set it to **Private**.*")
                
                with gr.Row():
                    backup_btn = gr.Button("🚀 Backup Workspace to Cloud", elem_classes=["primary"])
                    restore_btn = gr.Button("📥 Restore Workspace from Cloud")
                    
            with gr.Column(scale=4):
                gr.Markdown("#### 📈 Sync Console")
                log_output = gr.TextArea(
                    label="Live Logs",
                    interactive=False,
                    lines=15,
                    max_lines=20,
                    autoscroll=True,
                    elem_classes=["console-box"]
                )
                refresh_btn = gr.Button("🔄 Refresh Logs")
                
        def run_backup(token, repo):
            if not token or not repo:
                return "❌ Please provide both HF Token and Repo ID."
            # Start in background
            import threading
            def bg_task():
                engine.backup_workspace(token, repo)
            threading.Thread(target=bg_task).start()
            return "Backup process started in background. Check logs."
            
        def run_restore(token, repo):
            if not token or not repo:
                return "❌ Please provide both HF Token and Repo ID."
            import threading
            def bg_task():
                engine.restore_workspace(token, repo)
            threading.Thread(target=bg_task).start()
            return "Restore process started in background. Check logs."
            
        def update_logs():
            return engine.get_logs()
            
        backup_btn.click(fn=run_backup, inputs=[hf_token, repo_id], outputs=[log_output])
        restore_btn.click(fn=run_restore, inputs=[hf_token, repo_id], outputs=[log_output])
        refresh_btn.click(fn=update_logs, outputs=[log_output])
