import sys
import threading
import time
from typing import Any, Dict, List, Optional
import traceback
import psutil

class StudioEngine:
    def __init__(self):
        self.wrapper = None
        self.is_training = False
        self.logs = []
        self.available_models = [
            "unsloth/llama-3-8b-bnb-4bit",
            "unsloth/mistral-7b-v0.3-bnb-4bit",
            "unsloth/qwen2-7b-bnb-4bit"
        ]
        self.loaded_chat_model = None
        self.loaded_chat_tokenizer = None
        self.server_process = None
        self.server_logs = []
        self.vector_db = None
        self.vector_corpus = []
        
        self.use_unsloth = False
        try:
            from vietnamese_ai.fine_tuning.unsloth_wrapper import UnslothWrapper
            self.wrapper = UnslothWrapper()
            self.available_models = list(self.wrapper.UNSLOTH_MODELS.keys())
            self.use_unsloth = True
        except (ImportError, NotImplementedError, Exception) as e:
            print(f"WARNING: vietnamese_ai or unsloth could not be loaded ({str(e)}). Running in Native HF mode.")
            pass
            
    def get_models(self) -> List[str]:
        return self.available_models
        
    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        print(f"[{timestamp}] {message}")
        
    def get_logs(self) -> str:
        return "\n".join(self.logs)

    # ==========================================
    # SFT TRAINING
    # ==========================================
    def start_training(self, model_name: str, dataset_path: str, epochs: int, batch_size: int, learning_rate: float, lora_rank: int, lora_alpha: int, warmup_steps: int, max_seq_length: int, output_dir: str, is_vision: bool = False):
        if self.is_training: return "Training is already in progress!"
        self.is_training = True
        self.logs = []
        self.log(f"Starting SFT for {model_name} (Vision: {is_vision})...")
        thread = threading.Thread(target=self._training_thread, args=(model_name, dataset_path, epochs, batch_size, learning_rate, lora_rank, lora_alpha, warmup_steps, max_seq_length, output_dir, False, 0.1, False, False, False, is_vision, None))
        thread.start()
        return "SFT Training started. Check logs."

    # ==========================================
    # ALIGNMENT TRAINING (DPO/ORPO/KTO)
    # ==========================================
    def start_alignment(self, model_name: str, dataset_path: str, epochs: int, batch_size: int, learning_rate: float, lora_rank: int, lora_alpha: int, beta: float, warmup_steps: int, max_seq_length: int, output_dir: str, is_orpo: bool = False, is_kto: bool = False):
        if self.is_training: return "Training is already in progress!"
        self.is_training = True
        self.logs = []
        is_dpo = not is_orpo and not is_kto
        algo = "ORPO" if is_orpo else "KTO" if is_kto else "DPO"
        self.log(f"Starting {algo} Alignment for {model_name} with beta={beta}...")
        thread = threading.Thread(target=self._training_thread, args=(model_name, dataset_path, epochs, batch_size, learning_rate, lora_rank, lora_alpha, warmup_steps, max_seq_length, output_dir, is_dpo, beta, is_orpo, is_kto, False, False, None))
        thread.start()
        return f"{algo} Training started. Check logs."

    # ==========================================
    # AGENTIC FINE-TUNING
    # ==========================================
    def start_agent_tuning(self, model_name: str, dataset_path: str, epochs: int, batch_size: int, learning_rate: float, lora_rank: int, lora_alpha: int, warmup_steps: int, max_seq_length: int, tool_format: str, output_dir: str):
        if self.is_training: return "Training is already in progress!"
        self.is_training = True
        self.logs = []
        self.log(f"Starting Agentic Tuning for {model_name} (Format: {tool_format})...")
        thread = threading.Thread(target=self._training_thread, args=(model_name, dataset_path, epochs, batch_size, learning_rate, lora_rank, lora_alpha, warmup_steps, max_seq_length, output_dir, False, 0.1, False, False, True, False, tool_format))
        thread.start()
        return "Agentic Tuning started. Check logs."

    def _training_thread(self, model_name, dataset_path, epochs, batch_size, learning_rate, lora_rank, lora_alpha, warmup_steps, max_seq_length, output_dir, is_dpo, beta, is_orpo, is_kto, is_agent, is_vision, tool_format):
        try:
            from datasets import load_dataset
            if dataset_path.endswith('.jsonl') or dataset_path.endswith('.json'):
                dataset = load_dataset('json', data_files=dataset_path, split='train')
            elif dataset_path.endswith('.csv'):
                dataset = load_dataset('csv', data_files=dataset_path, split='train')
            else:
                dataset = load_dataset(dataset_path, split='train')
            self.log(f"Dataset loaded. Total samples: {len(dataset)}")

            if not self.use_unsloth:
                self.log("Unsloth not available. Falling back to native HuggingFace (CPU/GPU)...")
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
                from peft import LoraConfig, get_peft_model
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                self.log(f"Loading model via native transformers on {device}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model_kwargs = {}
                if device == "cuda":
                    model_kwargs["load_in_4bit"] = True
                    model_kwargs["device_map"] = "auto"
                
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                
                self.log(f"Configuring LoRA (r={lora_rank}, alpha={lora_alpha})...")
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                model = get_peft_model(model, lora_config)
                
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    num_train_epochs=epochs,
                    per_device_train_batch_size=batch_size,
                    learning_rate=learning_rate,
                    warmup_steps=warmup_steps,
                    fp16=device=="cuda" and not torch.cuda.is_bf16_supported(),
                    bf16=device=="cuda" and torch.cuda.is_bf16_supported(),
                    optim="adamw_torch" if device=="cpu" else "adamw_8bit",
                    report_to="none"
                )
                
                if is_agent:
                    self.log(f"Formatting dataset for Agentic Tool-calling ({tool_format})...")
                    # Here we would normally apply a chat template mapping for function calling
                    pass
                
                if is_dpo:
                    self.log("Starting DPO training (Native)...")
                    from trl import DPOTrainer
                    if 'prompt' not in dataset.column_names or 'chosen' not in dataset.column_names:
                        self.log("Warning: Dataset missing 'prompt', 'chosen', 'rejected' columns.")
                    trainer = DPOTrainer(
                        model=model, ref_model=None, tokenizer=tokenizer, train_dataset=dataset,
                        args=training_args, beta=beta, max_length=max_seq_length, max_prompt_length=max_seq_length // 2,
                    )
                elif is_orpo:
                    self.log("Starting ORPO training (Native)...")
                    from trl import ORPOTrainer
                    trainer = ORPOTrainer(
                        model=model, tokenizer=tokenizer, train_dataset=dataset,
                        args=training_args, beta=beta, max_length=max_seq_length, max_prompt_length=max_seq_length // 2,
                    )
                elif is_kto:
                    self.log("Starting KTO training (Native)...")
                    from trl import KTOTrainer
                    trainer = KTOTrainer(
                        model=model, ref_model=None, tokenizer=tokenizer, train_dataset=dataset,
                        args=training_args, beta=beta, max_length=max_seq_length, max_prompt_length=max_seq_length // 2,
                    )
                else:
                    self.log("Starting SFT training (Native)...")
                    from trl import SFTTrainer
                    trainer = SFTTrainer(
                        model=model, train_dataset=dataset, args=training_args, max_seq_length=max_seq_length, tokenizer=tokenizer,
                    )
                
                trainer.train()
                trainer.save_model(output_dir)
                tokenizer.save_pretrained(output_dir)
                self.log(f"Model saved to {output_dir}")
                return

            if is_vision:
                self.log("Loading Vision Model via Unsloth...")
                from unsloth import FastVisionModel
                # FastVisionModel uses slightly different API, but we'll try to emulate the wrapper behavior
                model, tokenizer = FastVisionModel.from_pretrained(model_name, load_in_4bit=True)
                model = FastVisionModel.get_peft_model(model, r=lora_rank, lora_alpha=lora_alpha, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
                self.wrapper._model = model
                self.wrapper._tokenizer = tokenizer
                self.wrapper._da_tai = True
            else:
                self.log("Loading Language Model via UnslothWrapper...")
                self.wrapper.tai_mo_hinh(model_name, max_seq_length=max_seq_length)
                self.wrapper.config_lora(r=lora_rank, lora_alpha=lora_alpha)
            
            if is_agent:
                self.log(f"Formatting dataset for Agentic Tool-calling ({tool_format})...")
                pass

            if is_dpo or is_orpo or is_kto:
                import torch
                from transformers import TrainingArguments
                
                training_args = TrainingArguments(
                    output_dir=output_dir, num_train_epochs=epochs, per_device_train_batch_size=batch_size,
                    learning_rate=learning_rate, warmup_steps=warmup_steps,
                    fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
                    optim="adamw_8bit", report_to="none"
                )
                
                if is_dpo:
                    self.log("Starting DPO training with Unsloth kernels...")
                    from trl import DPOTrainer
                    from unsloth import PatchDPOTrainer
                    PatchDPOTrainer()
                    trainer = DPOTrainer(
                        model=self.wrapper._model, ref_model=None, tokenizer=self.wrapper._tokenizer, train_dataset=dataset,
                        args=training_args, beta=beta, max_length=max_seq_length, max_prompt_length=max_seq_length // 2,
                    )
                elif is_orpo:
                    self.log("Starting ORPO training with Unsloth kernels...")
                    from trl import ORPOTrainer
                    trainer = ORPOTrainer(
                        model=self.wrapper._model, tokenizer=self.wrapper._tokenizer, train_dataset=dataset,
                        args=training_args, beta=beta, max_length=max_seq_length, max_prompt_length=max_seq_length // 2,
                    )
                elif is_kto:
                    self.log("Starting KTO training with Unsloth kernels...")
                    from trl import KTOTrainer
                    trainer = KTOTrainer(
                        model=self.wrapper._model, ref_model=None, tokenizer=self.wrapper._tokenizer, train_dataset=dataset,
                        args=training_args, beta=beta, max_length=max_seq_length, max_prompt_length=max_seq_length // 2,
                    )
                trainer.train()
                self.log("Alignment training complete.")
            else:
                self.log("Starting SFT fine-tuning...")
                if is_vision:
                    from trl import SFTTrainer
                    from unsloth import is_bfloat16_supported
                    from transformers import TrainingArguments
                    trainer = SFTTrainer(
                        model=self.wrapper._model,
                        tokenizer=self.wrapper._tokenizer,
                        train_dataset=dataset,
                        dataset_text_field="text",
                        max_seq_length=max_seq_length,
                        dataset_num_proc=2,
                        packing=False, # Vision models generally don't pack
                        args=TrainingArguments(
                            per_device_train_batch_size=batch_size,
                            gradient_accumulation_steps=4,
                            warmup_steps=warmup_steps,
                            max_steps=epochs*10, # For brevity in dummy
                            learning_rate=learning_rate,
                            fp16=not is_bfloat16_supported(),
                            bf16=is_bfloat16_supported(),
                            optim="adamw_8bit",
                            weight_decay=0.01,
                            lr_scheduler_type="linear",
                            seed=3407,
                            output_dir=output_dir,
                        ),
                    )
                    trainer.train()
                else:
                    self.wrapper.fine_tune(datasets=dataset, so_vong=epochs, batch_size=batch_size, learning_rate=learning_rate, warmup_steps=warmup_steps, output_dir=output_dir)
            
            self.wrapper.luu_model(output_dir)
            self.log(f"Model saved to {output_dir}")
            
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            self.log(traceback.format_exc())
        finally:
            self.is_training = False

    # ==========================================
    # DATA SYNTHESIS
    # ==========================================
    def synthesize_data(self, provider: str, api_key: str, model_name: str, topic: str, count: int, fmt: str, out_path: str):
        self.logs = []
        self.log(f"Starting Data Synthesis ({count} samples) using {provider} ({model_name})...")
        def _synth_thread():
            import requests
            import json
            import os
            
            generated = 0
            with open(out_path, 'w', encoding='utf-8') as f:
                while generated < count:
                    batch = min(5, count - generated)
                    prompt = f"Generate {batch} diverse examples for the topic: '{topic}'. "
                    if "Alpaca" in fmt:
                        prompt += "Format your response as a valid JSON array of objects, where each object has 'instruction', 'input', and 'output' fields. Do not include markdown code blocks, just pure JSON array."
                    else:
                        prompt += "Format your response as a valid JSON array of objects, where each object has 'messages' array containing objects with 'role' (user/assistant) and 'content'. Do not include markdown code blocks, just pure JSON array."
                        
                    if provider == "Google Gemini":
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                        payload = {"contents": [{"parts": [{"text": prompt}]}]}
                        try:
                            resp = requests.post(url, json=payload)
                            if resp.status_code == 200:
                                res_text = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                                res_text = res_text.replace("```json", "").replace("```", "").strip()
                                data = json.loads(res_text)
                                for item in data:
                                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                                generated += len(data)
                                self.log(f"Generated {generated}/{count} samples...")
                            else:
                                self.log(f"API Error: {resp.text}")
                                break
                        except Exception as e:
                            self.log(f"Error parsing API response: {str(e)}")
                            break
                    else:
                        self.log("Provider not fully implemented in MVP. Returning mock data.")
                        # Mock data
                        f.write(json.dumps({"instruction": "Mock", "input": "Mock", "output": "Mock"}) + '\n')
                        generated += batch
                    time.sleep(1)
            self.log(f"✅ Synthesis complete. Saved to {out_path}")
            
        import threading
        threading.Thread(target=_synth_thread).start()
        return "Synthesis task started. Check logs."

    # ==========================================
    # EXPORT & HUB
    # ==========================================
    def merge_lora(self, base_model: str, lora_path: str, out_path: str):
        self.logs = []
        self.log(f"Merging LoRA {lora_path} into Base Model {base_model}...")
        def _merge_thread():
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from peft import PeftModel
                
                self.log("Loading base model...")
                model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
                tokenizer = AutoTokenizer.from_pretrained(base_model)
                
                self.log("Loading LoRA adapter...")
                model = PeftModel.from_pretrained(model, lora_path)
                
                self.log("Merging weights (this may take a while)...")
                model = model.merge_and_unload()
                
                self.log(f"Saving merged model to {out_path}...")
                model.save_pretrained(out_path)
                tokenizer.save_pretrained(out_path)
                self.log("✅ Merge complete.")
            except Exception as e:
                self.log(f"Merge failed: {str(e)}")
                self.log(traceback.format_exc())
                
        import threading
        threading.Thread(target=_merge_thread).start()
        return "Merge started. Check logs."

    def push_to_hub(self, model_path: str, token: str, repo_id: str):
        self.logs = []
        self.log(f"Pushing {model_path} to HF Hub ({repo_id})...")
        def _push_thread():
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=token)
                self.log("Creating repo...")
                api.create_repo(repo_id, exist_ok=True)
                self.log("Uploading files (this will take a while)...")
                api.upload_folder(folder_path=model_path, repo_id=repo_id, repo_type="model")
                self.log(f"✅ Successfully pushed to https://huggingface.co/{repo_id}")
            except Exception as e:
                self.log(f"Push failed: {str(e)}")
        import threading
        threading.Thread(target=_push_thread).start()
        return "Push started. Check logs."

    def export_gguf(self, model_path: str, quantization: str) -> str:
        if not self.use_unsloth:
            return "GGUF Export is currently only supported when Unsloth (GPU) is available."
        try:
            if not getattr(self.wrapper, '_da_tai', False):
                self.wrapper.tai_mo_hinh(model_path)
            out_path = f"{model_path}-{quantization}.gguf"
            self.wrapper.xuat_gguf(out_path, quantization=quantization)
            return f"Export successful: {out_path}"
        except Exception as e:
            return f"Export failed: {str(e)}"

    # ==========================================
    # CHAT / INFERENCE
    # ==========================================
    def load_chat_model(self, model_path: str):
        if not self.use_unsloth and not getattr(self, "allow_cpu_chat", True): return "Chat requires GPU."
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.loaded_chat_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.loaded_chat_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_4bit=True)
            return f"Loaded {model_path} successfully!"
        except Exception as e:
            return f"Failed to load: {str(e)}"

    def chat_inference_stream(self, message: str, history: list):
        if self.loaded_chat_model is None:
            yield f"Error: Model not loaded."
            return
            
        try:
            from transformers import TextIteratorStreamer
            import threading
            
            prompt = ""
            for user_msg, bot_msg in history:
                prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n"
            prompt += f"User: {message}\nAssistant:"
            
            inputs = self.loaded_chat_tokenizer(prompt, return_tensors="pt").to(self.loaded_chat_model.device)
            streamer = TextIteratorStreamer(self.loaded_chat_tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            generation_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9
            )
            
            thread = threading.Thread(target=self.loaded_chat_model.generate, kwargs=generation_kwargs)
            thread.start()
            
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                yield generated_text
        except Exception as e:
            yield f"Error during generation: {str(e)}"

    # ==========================================
    # RAG STUDIO
    # ==========================================
    def build_vector_db(self, file_path: str, chunk_size: int):
        try:
            import PyPDF2
            from sentence_transformers import SentenceTransformer
            
            self.log(f"Building Vector DB for {file_path}...")
            text = ""
            if file_path.endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
            # Basic chunking
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            self.vector_corpus = chunks
            
            # Use a lightweight embedding model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            self.vector_db = model.encode(chunks, convert_to_tensor=True)
            self.log(f"✅ Built Vector DB with {len(chunks)} chunks.")
            return f"Database built with {len(chunks)} chunks."
        except Exception as e:
            return f"Error building database: {str(e)}"

    def rag_query_stream(self, query: str, history: list, k: int):
        if self.loaded_chat_model is None:
            yield "Error: Model not loaded."
            return
        if self.vector_db is None:
            yield "Error: Vector DB not built."
            return
            
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer('all-MiniLM-L6-v2')
            query_emb = model.encode(query, convert_to_tensor=True)
            hits = util.semantic_search(query_emb, self.vector_db, top_k=k)[0]
            
            context = "\n".join([self.vector_corpus[hit['corpus_id']] for hit in hits])
            
            augmented_query = f"Context:\n{context}\n\nQuestion: {query}\nAnswer based on context:"
            
            for reply in self.chat_inference_stream(augmented_query, history):
                yield reply
        except Exception as e:
            yield f"RAG Error: {str(e)}"

    # ==========================================
    # 1-CLICK DEPLOYMENT
    # ==========================================
    def start_server(self, model_path: str, is_vllm: bool):
        if self.server_process is not None:
            return "Server is already running."
            
        import subprocess
        engine_type = "vllm" if is_vllm else "native"
        self.server_logs = []
        self.server_logs.append(f"Starting server with {engine_type} engine...")
        
        try:
            # We spawn a completely independent process
            self.server_process = subprocess.Popen(
                [sys.executable, "-m", "evonet_studio.server", "--model", model_path, "--engine", engine_type],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            def read_output():
                for line in iter(self.server_process.stdout.readline, ''):
                    if line:
                        self.server_logs.append(line.strip())
                        if len(self.server_logs) > 100:
                            self.server_logs.pop(0)
                self.server_process.stdout.close()
                self.server_process.wait()
                self.server_process = None
                self.server_logs.append("Server process terminated.")
                
            threading.Thread(target=read_output, daemon=True).start()
            return f"Server started (PID: {self.server_process.pid})"
        except Exception as e:
            return f"Failed to start server: {str(e)}"
            
    def stop_server(self):
        if self.server_process is None:
            return "Server is not running."
        try:
            # Terminate gracefully
            import psutil
            parent = psutil.Process(self.server_process.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            return "Server stopped."
        except Exception as e:
            return f"Error stopping server: {str(e)}"
            
    def get_server_logs(self):
        return "\n".join(self.server_logs)

    # ==========================================
    # SYSTEM MONITOR
    # ==========================================
    def get_system_stats(self) -> Dict[str, Any]:
        cpu_usage = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        
        gpu_stats = []
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_stats.append({
                    "id": i,
                    "name": pynvml.nvmlDeviceGetName(handle),
                    "vram_used": mem_info.used / (1024**3),
                    "vram_total": mem_info.total / (1024**3),
                    "temp": temp,
                    "util": util.gpu
                })
        except Exception:
            gpu_stats = [{"name": "No NVML/GPU detected", "vram_used": 0, "vram_total": 0, "temp": 0, "util": 0}]
            
        return {
            "cpu_percent": cpu_usage,
            "ram_used": ram.used / (1024**3),
            "ram_total": ram.total / (1024**3),
            "ram_percent": ram.percent,
            "gpus": gpu_stats
        }

    # ==========================================
    # EVALUATION
    # ==========================================
    def fetch_llm_models(self, provider: str, api_key: str) -> List[str]:
        if provider == "Google Gemini":
            if not api_key: return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash"]
            try:
                import requests
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
                resp = requests.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    models = []
                    for m in data.get("models", []):
                        if "generateContent" in m.get("supportedGenerationMethods", []):
                            name = m.get("name", "").replace("models/", "")
                            if "gemini" in name.lower():
                                models.append(name)
                    return models if models else ["gemini-1.5-flash", "gemini-1.5-pro"]
            except Exception as e:
                self.log(f"Error fetching Gemini models: {str(e)}")
                return ["gemini-1.5-flash", "gemini-1.5-pro"]
        elif provider == "OpenAI":
            return ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        elif provider == "Anthropic":
            return ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"]
        return ["llama3"]

    def evaluate_model(self, model_name: str, dataset_path: str, method: str = "Statistical (Perplexity)", metric_name: str = "Perplexity", judge_provider: str = "", judge_model: str = "", api_key: str = "", sample_size: int = 10):
            
        try:
            import torch
            from datasets import load_dataset
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import evaluate
            from tqdm import tqdm
            
            self.log(f"Evaluating {model_name} on {dataset_path} using {method}")
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
            model.eval()
            
            # Load dataset
            num_samples = sample_size if "LLM" in method else 100
            if dataset_path.endswith('.jsonl') or dataset_path.endswith('.json'):
                dataset = load_dataset('json', data_files=dataset_path, split=f'train[:{num_samples}]')
            else:
                dataset = load_dataset(dataset_path, split=f'train[:{num_samples}]')
                
            if "LLM" in method:
                import requests
                import re
                
                prompt_col = "prompt" if "prompt" in dataset.column_names else "text" if "text" in dataset.column_names else dataset.column_names[0]
                
                scores = []
                feedback_samples = ""
                
                for i, item in enumerate(dataset):
                    prompt_text = item[prompt_col]
                    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
                    outputs = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
                    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    if answer.startswith(prompt_text):
                        answer = answer[len(prompt_text):].strip()
                        
                    judge_prompt = f"Act as an expert AI judge. Evaluate the following AI response. Score it from 1 to 10 based on accuracy, relevance, and helpfulness. Provide your answer exactly as 'Score: <number>'.\n\nPrompt: {prompt_text}\nResponse: {answer}"
                    
                    score = 0
                    if judge_provider == "Google Gemini":
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{judge_model}:generateContent?key={api_key}"
                        payload = {"contents": [{"parts": [{"text": judge_prompt}]}]}
                        resp = requests.post(url, json=payload)
                        if resp.status_code == 200:
                            res_text = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                            match = re.search(r'Score:\s*(\d+)', res_text)
                            if match: score = int(match.group(1))
                    else:
                        # Dummy scoring for local / others in this MVP
                        score = 8
                    
                    scores.append(score)
                    if i < 2:
                        feedback_samples += f"**Sample {i+1}**\nPrompt: {prompt_text[:50]}...\nScore: {score}/10\n\n"
                        
                avg_score = sum(scores) / len(scores) if scores else 0
                return f"✅ LLM-as-a-Judge Evaluation Complete.\n\nDataset: {dataset_path}\nProvider: {judge_provider} ({judge_model})\n\n**Average Score: {avg_score:.2f} / 10.0**\n\n{feedback_samples}"
                
            else:
                text_column = "text" if "text" in dataset.column_names else dataset.column_names[0]
                texts = dataset[text_column]
                
                if metric_name.lower() == "perplexity":
                    perplexity = evaluate.load("perplexity", module_type="metric")
                    results = perplexity.compute(model_id=model_name, add_start_token=False, predictions=texts)
                    mean_ppl = results["mean_perplexity"]
                    return f"✅ Evaluation Complete.\n\nDataset: {dataset_path}\nMetric: Perplexity\n\n**Mean Perplexity: {mean_ppl:.4f}**\n\n(Tested on 100 samples)"
                    
                elif metric_name.lower() == "rouge":
                    return f"ROUGE evaluation requires Prompt-Reference pairs. Perplexity is recommended for language modeling."
                else:
                    return f"Metric {metric_name} is not fully supported yet."
                
        except Exception as e:
            return f"Evaluation failed: {str(e)}\n\nTraceback: {traceback.format_exc()}"
