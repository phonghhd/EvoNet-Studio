import gradio as gr
from evonet_studio.engine import StudioEngine

def build_rag_ui(engine: StudioEngine):
    with gr.Tab("📚 RAG Studio"):
        gr.Markdown("### Retrieval-Augmented Generation (RAG) Testing")
        gr.Markdown("Test your fine-tuned model's ability to extract information from documents.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 1. Configure Model")
                model_name = gr.Textbox(label="Model Path", value="outputs/evonet_model")
                load_btn = gr.Button("🔄 Load Model into Memory")
                load_status = gr.Markdown("*Model not loaded.*")
                
                gr.Markdown("#### 2. Vector Database")
                doc_file = gr.File(label="Upload Document (PDF or TXT)", file_types=[".pdf", ".txt"])
                chunk_size = gr.Slider(minimum=100, maximum=1000, step=50, value=300, label="Chunk Size (Characters)")
                build_db_btn = gr.Button("🔨 Build Vector Database")
                db_status = gr.Markdown("*No database built.*")
                
            with gr.Column(scale=2):
                gr.Markdown("#### 3. RAG Chat Console")
                chat_history = gr.Chatbot(label="RAG Assistant", height=400)
                user_input = gr.Textbox(label="Ask a question about your document...", placeholder="Type here and press enter...")
                
                with gr.Accordion("Advanced Options", open=False):
                    top_k = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Top-K Chunks to Retrieve")
                    
                clear_btn = gr.Button("🗑️ Clear Chat")

        def load_model(path):
            msg = engine.load_chat_model(path)
            return f"**Status:** {msg}"
            
        def build_db(file_obj, size):
            if file_obj is None: return "**Status:** Please upload a file."
            msg = engine.build_vector_db(file_obj.name, size)
            return f"**Status:** {msg}"
            
        def rag_chat(user_text, history, k):
            history = history or []
            # We use yield to support streaming if engine supports it, or just return
            bot_reply = ""
            for partial_reply in engine.rag_query_stream(user_text, history, k):
                bot_reply = partial_reply
                yield "", history + [(user_text, bot_reply)]

        load_btn.click(fn=load_model, inputs=[model_name], outputs=[load_status])
        build_db_btn.click(fn=build_db, inputs=[doc_file, chunk_size], outputs=[db_status])
        user_input.submit(fn=rag_chat, inputs=[user_input, chat_history, top_k], outputs=[user_input, chat_history])
        clear_btn.click(lambda: None, None, chat_history, queue=False)
