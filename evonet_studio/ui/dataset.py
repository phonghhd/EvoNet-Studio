import gradio as gr
import pandas as pd

def build_dataset_ui(engine):
    with gr.Tab("📊 Dataset Management"):
        gr.Markdown("### Preview & Format Datasets")
        gr.Markdown("Load local or HuggingFace datasets to verify their structure before fine-tuning.")
        
        with gr.Row():
            with gr.Column(scale=1):
                upload_file = gr.File(label="Upload CSV or JSONL")
                dataset_path = gr.Textbox(label="Or enter HuggingFace Dataset Path", placeholder="yahma/alpaca-cleaned")
                load_data_btn = gr.Button("🔍 Load & Preview", elem_classes=["primary"])
                
                with gr.Accordion("Dataset Info", open=True):
                    ds_info = gr.Markdown("*No dataset loaded*")
                
            with gr.Column(scale=3):
                preview_df = gr.Dataframe(label="Dataset Preview", interactive=False, max_height=600)
                
        def load_preview(file_obj, path):
            try:
                if file_obj is not None:
                    fname = file_obj.name
                    if fname.endswith('.csv'):
                        df = pd.read_csv(fname)
                    else:
                        df = pd.read_json(fname, lines=True)
                elif path:
                    from datasets import load_dataset
                    ds = load_dataset(path, split='train')
                    df = ds.to_pandas()
                else:
                    return pd.DataFrame([{"Error": "Provide a file or path"}]), "Error"
                
                info = f"**Total Rows:** {len(df):,}\n\n**Columns:** {', '.join(df.columns)}"
                return df.head(50), info
            except Exception as e:
                return pd.DataFrame([{"Error": str(e)}]), f"**Error:** {str(e)}"
                
        load_data_btn.click(fn=load_preview, inputs=[upload_file, dataset_path], outputs=[preview_df, ds_info])
