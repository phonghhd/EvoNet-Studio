import gradio as gr
from evonet_studio.engine import StudioEngine

def build_lora_manager_ui(engine: StudioEngine):
    with gr.Tab("🗂️ LoRA Manager"):
        gr.Markdown("### LoRA Adapter Registry")
        gr.Markdown("Manage, review, and organize your trained LoRA adapters.")
        
        with gr.Row():
            with gr.Column(scale=5):
                refresh_btn = gr.Button("🔄 Scan for Adapters", elem_classes=["primary"])
                lora_table = gr.Dataframe(
                    headers=["Adapter Path", "Size (MB)"],
                    datatype=["str", "str"],
                    col_count=(2, "fixed"),
                    interactive=False
                )
            
            with gr.Column(scale=4):
                gr.Markdown("#### 💡 Instructions")
                gr.Markdown(
                    "1. Click **Scan for Adapters** to search the `outputs/` directory for trained models.\n"
                    "2. You can use these paths in the **Export & Hub** tab to merge them with your Base Model.\n"
                    "3. You can also use them in the **Deployment** tab to enable the Multi-LoRA MoE Router."
                )

        def fetch_loras():
            adapters = engine.get_lora_adapters()
            if not adapters:
                return [["No adapters found", "0 MB"]]
            return adapters

        refresh_btn.click(fn=fetch_loras, outputs=[lora_table])
