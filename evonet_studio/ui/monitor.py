import gradio as gr
from evonet_studio.engine import StudioEngine
import json

def build_monitor_ui(engine: StudioEngine):
    with gr.Tab("🖥️ System Monitor"):
        gr.Markdown("### Real-time Hardware Resources")
        
        with gr.Row():
            monitor_html = gr.HTML(label="Dashboard")
            
        gpu_json = gr.JSON(label="GPU Statistics")
        
        refresh_btn = gr.Button("🔄 Refresh System Stats")
        
        def update_stats():
            stats = engine.get_system_stats()
            cost = engine.get_gpu_cost_stats()
            
            try:
                from vietnamese_ai.ui.components import BangThongKe, TinhToanGPU
                
                # We render multiple BangThongKe for CPU, RAM, and Cost
                cpu_comp = BangThongKe("CPU Usage", lambda: {"CPU": f"{stats['cpu_percent']}%"})
                ram_comp = BangThongKe("RAM Usage", lambda: {"RAM": f"{stats['ram_percent']}%", "Used": f"{stats['ram_used']:.1f} GB"})
                cost_comp = BangThongKe("Estimated GPU Cost (A100)", lambda: {"Cost": cost})
                
                html_content = (
                    f"<div style='display: flex; gap: 10px; width: 100%;'>"
                    f"<div style='flex: 1;'>{cpu_comp.render_html()}</div>"
                    f"<div style='flex: 1;'>{ram_comp.render_html()}</div>"
                    f"<div style='flex: 1;'>{cost_comp.render_html()}</div>"
                    f"</div>"
                    f"<script>{cpu_comp.render_js()} {ram_comp.render_js()} {cost_comp.render_js()}</script>"
                )
            except Exception as e:
                html_content = f"<i>EvoNet AI Dashboard Error: {str(e)}</i>"
                
            return html_content, stats['gpus']
            
        refresh_btn.click(fn=update_stats, outputs=[monitor_html, gpu_json])
