import gradio as gr
from evonet_studio.engine import StudioEngine

def build_chat_ui(engine: StudioEngine):
    with gr.Tab("💬 Chat & Inference"):
        gr.Markdown("### Interactive Model Inference")
        gr.Markdown("Test your fine-tuned model instantly with real-time streaming.")
        
        with gr.Row():
            with gr.Column(scale=3):
                model_path = gr.Textbox(
                    label="Model Path (e.g. outputs/evonet_model)", 
                    value="outputs/evonet_model"
                )
            with gr.Column(scale=1):
                load_btn = gr.Button("🔌 Load Model", elem_classes=["primary"])
                load_status = gr.Markdown("*Model not loaded*")
                
        def load_model_callback(path):
            status = engine.load_chat_model(path)
            return f"**Status:** {status}"
            
        load_btn.click(fn=load_model_callback, inputs=[model_path], outputs=[load_status])
        
        # Use ChatInterface with streaming function
        chatbot = gr.ChatInterface(
            fn=engine.chat_inference_stream,
            title="EvoNet-Studio Chat",
            description="Chat with your model in real-time. (Make sure to load it first!)"
        )
