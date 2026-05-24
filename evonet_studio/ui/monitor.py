import gradio as gr
from evonet_studio.engine import StudioEngine
import json

def build_monitor_ui(engine: StudioEngine):
    with gr.Tab("🖥️ System Monitor"):
        gr.Markdown("### Real-time Hardware Resources")
        
        with gr.Row():
            cpu_text = gr.Textbox(label="CPU Usage", interactive=False)
            ram_text = gr.Textbox(label="RAM Usage", interactive=False)
            
        gpu_json = gr.JSON(label="GPU Statistics")
        
        refresh_btn = gr.Button("🔄 Refresh System Stats")
        
        def update_stats():
            stats = engine.get_system_stats()
            cpu = f"{stats['cpu_percent']}%"
            ram = f"{stats['ram_used']:.1f} GB / {stats['ram_total']:.1f} GB ({stats['ram_percent']}%)"
            return cpu, ram, stats['gpus']
            
        refresh_btn.click(fn=update_stats, outputs=[cpu_text, ram_text, gpu_json])
