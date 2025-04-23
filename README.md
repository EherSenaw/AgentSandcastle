# ASC: AgentSandCastle
[![Development Status](https://img.shields.io/badge/status-in%20development-yellow.svg)](https://github.com/EherSenaw/AgentSandCastle)

> **TL;DR;** A on-device framework for rapid R&D of local LLMs and agent systems under constrained resources (Linux/WSL, macOS). 
> 
> **📢 Development status**: Currently supports single-engine, single-agent. Aiming for multi-modal, multi-engine, multi-agent support through sequential feature additions (optimizations + new capabilities).

---
## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/EherSenaw/AgentSandCastle.git
cd AgentSandCastle

# (Suggested) Create and activate a virtual env (Python >= 3.9, tested on 3.10)
conda env create -n asc "python=3.10"
conda activate asc

# Install dependencies
# [macOS users] use `requirements.txt`. This will automatilcally install MLX.
pip install -r requirements.txt
# [Linux/WSL users] ensure a CUDA-enabled environment for vLLM, Then, check requirements.txt for additional installation.
```

## 🚀 Usage

```bash
# CLI mode
. ./run.sh

# Web interface (FastAPI)
. ./run_fast.sh
```

## ⭐️ Key Features
    Try R&D your own (or recent) study/method/skill in ML/MLOps field training by plugging that in to this framework directly. Categories can be Post-training, Inference optimization, Serving optimization, ...

    The framework is tailored for researchers and developers who want to prototype and experiment with multi-agent systems under constrained computing resources.
- **On-device LLM Engine**: Fully local LLMs for affordable hardware, on MLX (macOS) or vLLM (other platforms). Tested on MacBook Air(**M1**) and Desktop(**GTX1660**)
- **Commandline interface**: Interact with agents via terminal.
- **Web interface**: FastAPI-based interactive UI.
>

- **Simple ReAct-style Agent Flow**: Built-in reasoning loop (see ```mlx_engine.py```)
- **Tool auto-parsing**: Assign tools via the ```@tool(parse_docstring=True)``` decorator. 
- **Structured output**: Auto-convert Google-style docstrings into Pydantic(JSON) schemas for structured output guidance, using the decorator.
>

- **Minimal Dependencies**: Lightweight design with minimal external libraries for easier maintenance and greater control. (e.g. LangChain)
- **End-to-End Usability**: Seamless path from prototype to production-level experiments.
- **Easy-to-adapt**: Aim for as platform-agnostic design as possible, for easy testing of the optimization methods (currently wraps MLX/vLLM; full PyTorch backend customization planned)
>

## 🗺️ Roadmap
- [x] PoC: single-agent, single-LLM, single-turn chat, CLI. (w/ vLLM, MLX-LM) 
- [x] Define decorator for: Tool auto parsing & Structured output guidance
- [x] Add FastAPI-based web interface
> 
### Short-term (1-2 weeks)
- [ ] Streaming text output
- [ ] KV cache
- [ ] Prompt cache
- [ ] Speculative decoding
> 
### Mid- to Long-term
- [ ] Cache management with VectorDB: Chroma or FAISS?
- [ ] Custom quantization
- [ ] KV compression / eviction
- [ ] Context compression / eviction
- [ ] PyTorch backend customization (MPS & CUDA) ...
- [ ] Add graph-based state control over ReAct agent architecture.
- [ ] Multi-LLM engines
- [ ] Multi-agents
- [ ] Scheduling concerns ...
- [ ] Preference optimization support ...
- [ ] Documentation and tutorials
> 


## 🤝 Contributing

0. Create an Issue: Describe the feature and its impact on the codebase (MUST: pros/cons).
1. Fork the repository.
2. Create a branch: ```git checkout -b feature/<your_feature>```
3. Commit changes: ```git commit -m "Add <your_feature>"```
4. Open a Pull Request.

## 📝 Changelog
### 2025-04-23
- Resume development.
- Add FastAPI-based web interface
### 2025-03-05
- CLI-based PoC
- Define decorator for: Tool auto parsing & Structured output guidance
- Pause until next update

## 📄 License
MIT © Taewoo Jeong