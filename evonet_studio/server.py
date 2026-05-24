import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="EvoNet API Server Wrapper")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--engine", type=str, default="native", choices=["vllm", "native"], help="Engine to use")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
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

            app = FastAPI(title="EvoNet Native API")

            print("Loading Model into memory...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_kwargs = {}
            if device == "cuda":
                model_kwargs["load_in_4bit"] = True
                model_kwargs["device_map"] = "auto"

            tokenizer = AutoTokenizer.from_pretrained(args.model)
            model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
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
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=data.get("max_tokens", 256))
                
                generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

                response = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
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
                return JSONResponse(content=response)

            uvicorn.run(app, host=args.host, port=args.port)

        except ImportError as e:
            print(f"Dependencies missing for Native API server: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Server Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
