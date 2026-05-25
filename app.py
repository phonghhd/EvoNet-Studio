import sys
import os

# Add local vietnamese-ai if it exists in deps to sys.path so we can import it
local_deps_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deps', 'vietnamese-ai')
if os.path.exists(local_deps_path) and local_deps_path not in sys.path:
    sys.path.insert(0, local_deps_path)

import gradio as gr
from evonet_studio.engine import StudioEngine
from evonet_studio.ui.training import build_training_ui
from evonet_studio.ui.export import build_export_ui
from evonet_studio.ui.chat import build_chat_ui
from evonet_studio.ui.dataset import build_dataset_ui
from evonet_studio.ui.dpo import build_alignment_ui
from evonet_studio.ui.agent_tuning import build_agent_tuning_ui
from evonet_studio.ui.evaluation import build_evaluation_ui
from evonet_studio.ui.monitor import build_monitor_ui
from evonet_studio.ui.synthesis import build_synthesis_ui
from evonet_studio.ui.rag import build_rag_ui
from evonet_studio.ui.deployment import build_deployment_ui
from evonet_studio.ui.arena import build_arena_ui
from evonet_studio.ui.dataset_cleaner import build_dataset_cleaner_ui
from evonet_studio.ui.benchmark import build_benchmark_ui
from evonet_studio.ui.telemetry import build_telemetry_ui
from evonet_studio.ui.cloud_sync import build_cloud_sync_ui

def create_app():
    # Initialize Engine
    engine = StudioEngine()
    
    # Custom CSS for modern/premium look with glassmorphism and animations
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@400;500;600&display=swap');
    
    :root {
        --primary-400: #a78bfa;
        --primary-500: #8b5cf6;
        --primary-600: #7c3aed;
        --accent: #ec4899;
        --bg-grad: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }
    
    .gradio-container {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Header Animation */
    .evo-header {
        background: linear-gradient(to right, #8b5cf6, #ec4899, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: textShine 5s ease-in-out infinite alternate;
        font-family: 'Outfit', sans-serif;
    }
    
    @keyframes textShine {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    
    /* Glassmorphism Panels */
    .glass-panel {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
    }
    
    .dark .gradio-container {
        background: var(--bg-grad) !important;
        color: #f8fafc;
    }
    
    .dark .glass-panel {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Buttons */
    button.primary {
        background: linear-gradient(135deg, var(--primary-500), var(--primary-600)) !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4) !important;
    }
    
    button.primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6) !important;
        filter: brightness(1.1) !important;
    }
    
    /* Tabs */
    .tabs > .tab-nav {
        border-bottom: 2px solid rgba(139, 92, 246, 0.2) !important;
    }
    .tabs > .tab-nav > button.selected {
        color: var(--primary-400) !important;
        border-bottom-color: var(--primary-500) !important;
    }
    """
    
    theme = gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="pink",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
    ).set(
        body_background_fill="*neutral_950",
        body_text_color="*neutral_100",
        block_background_fill="rgba(30, 41, 59, 0.6)",
        block_border_width="1px",
        block_border_color="rgba(255,255,255,0.05)",
        button_primary_background_fill="linear-gradient(90deg, *primary_500, *primary_600)",
        button_primary_background_fill_hover="linear-gradient(90deg, *primary_600, *primary_700)",
        panel_background_fill="rgba(15, 23, 42, 0.7)",
    )
    
    with gr.Blocks(title="EvoNet-Studio Pro", theme=theme, css=custom_css) as demo:
        gr.HTML(
            """
            <div style="text-align: center; max-width: 900px; margin: 0 auto; padding: 30px 0;">
                <h1 class="evo-header" style="font-weight: 800; font-size: 3.5rem; margin-bottom: 0.5rem; line-height: 1.2;">
                    🚀 EvoNet-Studio Pro
                </h1>
                <p style="font-size: 1.2rem; color: #94a3b8; font-weight: 300;">
                    The ultimate AI fine-tuning & inference studio. Powered by <b style="color: #c084fc;">vietnamese-ai</b> & <b style="color: #f472b6;">Unsloth</b>.
                </p>
            </div>
            """
        )
        
        with gr.Tabs(elem_classes=["glass-panel"]):
            build_synthesis_ui(engine)
            build_training_ui(engine)
            build_alignment_ui(engine)
            build_agent_tuning_ui(engine)
            build_dataset_cleaner_ui(engine)
            build_dataset_ui(engine)
            build_arena_ui(engine)
            build_telemetry_ui(engine)
            build_evaluation_ui(engine)
            build_benchmark_ui(engine)
            build_rag_ui(engine)
            build_chat_ui(engine)
            build_cloud_sync_ui(engine)
            build_export_ui(engine)
            build_deployment_ui(engine)
            build_monitor_ui(engine)
        
        gr.Markdown("---")
        gr.Markdown("<div style='text-align: center; color: #64748b; font-size: 0.9rem;'>Developed with ❤️ using EvoNet Framework</div>")

    return demo, theme, custom_css

if __name__ == "__main__":
    app, theme, custom_css = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
