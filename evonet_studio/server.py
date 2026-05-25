import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="EvoNet API Server Wrapper")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--engine", type=str, default="native", choices=["vllm", "native"], help="Engine to use")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--loras", type=str, default="", help="JSON string mapping expert name to LoRA path")
    args = parser.parse_args()

    print(f"Starting API Server on {args.host}:{args.port} using engine '{args.engine}'...")

    if args.engine == "vllm":
        try:
            import vllm.entrypoints.openai.api_server
            # vLLM API server is designed to be run from CLI, but we can override sys.argv
            sys.argv = ["vllm", "--model", args.model, "--host", args.host, "--port", str(args.port)]
            print("Handing over to vLLM OpenAI API Server...")
            vllm.entrypoints.openai.api_server.main()
        except ImportError:
            print("Error: vLLM is not installed. Please install it using `pip install vllm`")
            sys.exit(1)
        except Exception as e:
            print(f"vLLM Server Error: {e}")
            sys.exit(1)
    else:
        # Fallback Native Server using FastAPI and Transformers
        try:
            from fastapi import FastAPI, Request
            from fastapi.responses import JSONResponse
            import uvicorn
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import time
            import uuid
            import json
            import os
            from pydantic import BaseModel

            app = FastAPI(title="EvoNet Native API")
            
            # Request Models
            class FeedbackRequest(BaseModel):
                log_id: str
                score: int

            print("Loading Model into memory...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_kwargs = {}
            if device == "cuda":
                model_kwargs["load_in_4bit"] = True
                model_kwargs["device_map"] = "auto"

            tokenizer = AutoTokenizer.from_pretrained(args.model)
            model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
            
            lora_experts = {}
            if args.loras:
                try:
                    lora_experts = json.loads(args.loras)
                    print(f"Loading LoRA Experts for MoE Routing: {lora_experts}")
                    from peft import PeftModel
                    # Load the first one to initialize PEFT
                    first_name, first_path = list(lora_experts.items())[0]
                    model = PeftModel.from_pretrained(model, first_path, adapter_name=first_name)
                    # Load the rest
                    for name, path in list(lora_experts.items())[1:]:
                        model.load_adapter(path, adapter_name=name)
                    # Set back to base initially
                    model.disable_adapter_layers()
                except Exception as e:
                    print(f"Failed to load LoRA experts: {e}")
                    
            model.eval()

            print("Model loaded. API is ready.")

            @app.post("/v1/chat/completions")
            async def chat_completions(request: Request):
                data = await request.json()
                messages = data.get("messages", [])
                
                # Simple prompt construction
                prompt = ""
                for msg in messages:
                    prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
                prompt += "Assistant: "

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                # --- MoE ROUTING ---
                active_expert = None
                if lora_experts:
                    prompt_lower = prompt.lower()
                    for expert_name in lora_experts.keys():
                        if expert_name.lower() in prompt_lower:
                            active_expert = expert_name
                            break
                            
                try:
                    if active_expert:
                        print(f"MoE Router: Activating Expert '{active_expert}'")
                        model.enable_adapter_layers()
                        model.set_adapter(active_expert)
                    else:
                        if lora_experts:
                            model.disable_adapter_layers()
                            
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=data.get("max_tokens", 256))
                finally:
                    if lora_experts:
                        model.disable_adapter_layers()
                # -------------------
                
                generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

                log_id = f"chatcmpl-{uuid.uuid4().hex}"
                
                response = {
                    "id": log_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": args.model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": inputs.input_ids.shape[1],
                        "completion_tokens": outputs.shape[1] - inputs.input_ids.shape[1],
                        "total_tokens": outputs.shape[1]
                    }
                }
                
                # --- TELEMETRY INTERCEPTION ---
                try:
                    os.makedirs("outputs", exist_ok=True)
                    with open("outputs/production_logs.jsonl", "a", encoding="utf-8") as f:
                        log_record = {
                            "log_id": log_id,
                            "timestamp": response["created"],
                            "prompt": prompt,
                            "response": generated_text,
                            "score": 0 # 0 = neutral, 1 = upvote, -1 = downvote
                        }
                        f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Failed to write telemetry log: {e}")
                # ------------------------------
                
                return JSONResponse(content=response)
                
            @app.post("/v1/feedback")
            async def submit_feedback(req: FeedbackRequest):
                """Endpoint for external apps to send Thumbs Up/Down (-1, 1)."""
                try:
                    file_path = "outputs/production_logs.jsonl"
                    if not os.path.exists(file_path):
                        return JSONResponse(status_code=404, content={"status": "error", "message": "Log file not found."})
                        
                    updated_logs = []
                    found = False
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line)
                            if data.get("log_id") == req.log_id:
                                data["score"] = req.score
                                found = True
                            updated_logs.append(data)
                            
                    if found:
                        with open(file_path, "w", encoding="utf-8") as f:
                            for data in updated_logs:
                                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                        return {"status": "success", "message": f"Updated score for {req.log_id} to {req.score}"}
                    else:
                        return JSONResponse(status_code=404, content={"status": "error", "message": "Log ID not found."})
                except Exception as e:
                    return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

            uvicorn.run(app, host=args.host, port=args.port)

        except ImportError as e:
            print(f"Dependencies missing for Native API server: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Server Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
