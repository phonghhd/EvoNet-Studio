import gradio as gr

def build_workflow_ui():
    with gr.Tab("🏗️ Workflow Builder (Beta)"):
        gr.Markdown("### EvoNet AI Visual Pipeline Builder")
        gr.Markdown("Kéo thả các khối logic (Nodes) để thiết kế quy trình huấn luyện và đánh giá AI của bạn mà không cần viết code.")
        
        # Nhúng Studio Kéo Thả (Node-based UI) từ vietnamese-ai
        try:
            from vietnamese_ai.studio.builder import StudioKeoTha
            
            # Khởi tạo backend engine cho builder
            studio = StudioKeoTha()
            
            # HTML hiển thị khu vực làm việc giả lập (Canvas)
            html_content = (
                f"<div style='width: 100%; height: 400px; border: 2px dashed #4b5563; border-radius: 12px; display: flex; align-items: center; justify-content: center; background-color: #1f2937; margin-bottom: 20px;'>"
                f"   <div class='text-center' style='text-align: center;'>"
                f"       <h3 style='color: #9ca3af; font-size: 24px; margin-bottom: 10px;'>EvoNet-Studio Workspace</h3>"
                f"       <p style='color: #6b7280;'>Drag and drop nodes from the left panel to start building.</p>"
                f"       <p style='color: #10b981; font-weight: bold;'>Status: EvoNet AI Builder Engine Loaded ({len(studio.danh_sach_loai_node())} Node Types Available)</p>"
                f"   </div>"
                f"</div>"
            )
            gr.HTML(html_content)
            
            with gr.Row():
                with gr.Column(scale=1):
                    node_type = gr.Dropdown(choices=studio.danh_sach_loai_node(), label="Chọn Loại Node", value="DU_LIEU")
                    add_node_btn = gr.Button("➕ Thêm Node")
                with gr.Column(scale=1):
                    run_pipeline_btn = gr.Button("🚀 Chạy Pipeline", variant="primary")
                    clear_btn = gr.Button("🗑️ Xóa Canvas")
                    
            status_text = gr.Markdown("**Trạng thái:** Sẵn sàng.")
            canvas_state = gr.JSON(label="Dữ liệu Canvas (Backend)")
            
            def add_node(loai_node):
                node_id = studio.them_node(loai_node, {"name": f"Node_{loai_node}"})
                return f"**Trạng thái:** Đã thêm thành công Node ID: `{node_id}` ({loai_node})", studio.lay_canvas()
                
            def run_pipeline():
                try:
                    result = studio.chay()
                    return f"**Trạng thái:** Pipeline chạy thành công! Kết quả: {result}", studio.lay_canvas()
                except Exception as e:
                    return f"**Trạng thái:** Lỗi khi chạy Pipeline: {str(e)}", studio.lay_canvas()
                    
            def clear_canvas():
                # Re-init studio to clear
                nonlocal studio
                studio = StudioKeoTha()
                return "**Trạng thái:** Đã xóa toàn bộ Canvas.", studio.lay_canvas()
                
            add_node_btn.click(fn=add_node, inputs=[node_type], outputs=[status_text, canvas_state])
            run_pipeline_btn.click(fn=run_pipeline, outputs=[status_text, canvas_state])
            clear_btn.click(fn=clear_canvas, outputs=[status_text, canvas_state])
            
        except Exception as e:
            gr.HTML(f"<div style='color: red;'>Failed to load vietnamese_ai.studio: {str(e)}</div>")
