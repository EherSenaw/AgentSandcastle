# AgentSandcastle
Begin your research & engineering journey of building and advancing your own agent framework, referring this repository as a starting point.
AgentSandcastle is a experimental framework with an AIM on:

* The framework is tailored for researchers and developers who want to prototype and experiment with multi-agent systems under constrained computing resources.
* Run (Multimodal) LLM fully local on affordable hardware (for me, a MacBook Air\[M1\] and Desktop\[GTX1660\])
* Maintain maximal control of the full system by suppressing external dependencies (e.g., LangChain) required for your own agent service be minimal.
* Try R&D your own (or recent) study/method/skill in ML/MLOps field training by plugging that in to this framework directly. Categories can be Post-training, Inference optimization, Serving optimization, ...

## Overview
AgentSandcastle aims to bridge the gap between cutting-edge agent research and practical, local experimentation by providing a lightweight, end-to-end environment that minimizes external dependencies and offers a controllable system for deploying both single and multi-agent architectures.

## Motivation
The inspiration came from my personal journey in exploring optimization across the full stack of (Multimodal) LLM development—spanning training, inference, and serving. While following the research trends related to Agentic AI, I found that available codebases often had heavy dependencies (e.g. Langchain), which might add difficulty of applying one's own optimization method across system and were not designed to run efficiently on low-resource systems. With AgentSandcastle, I wanted to create a platform that:

* Runs on a personal machine such as a MacBook or desktop.
* Provides a system-level approach for rapid prototyping of new methods or hypotheses.
* Supports multi-modal input/output processing in a user-friendly, end-to-end framework.

## Key Features
* AIMs:
  * Local LLM Engine: Optimized for low-resource hardware while still enabling experimentation with state-of-the-art (multimodal) LLMs.
  * Multi-Agent System: Easily extendable framework to simulate and test various agent interactions.
  * Minimal Dependencies: Designed to run with minimal external libraries for easier maintenance and greater control.
  * Modular Architecture: Components are designed to be plug-and-play, allowing researchers to integrate new optimization techniques seamlessly. One might be intrested in optimizing local LLM inference, the other might be interested in prompt engineering, and so on. In these cases, each of them can easily plug-and-play on each component by replacing corresponding part in the system.
  * End-to-End Usability: Aiming to deliver a smooth experience from prototype to production-level experiments.

## Usage
Instructions on setting up and running AgentSandcastle on your local machine:
* Installation:
First, clone this repository.
  > [Apple Silicon]()
  > (Suggested) Use conda or virtual environment, with `python>=3.9`. Tested with `python==3.10`.
  > NOTE: for local run of LLM part, currently using `MLX` developed by Apple.
  > Install requirements. <Subject to change> in the later release by incremental update to minimize dependency.
    ```python
    pip install -r requirements.txt
    ```

  > [Ubuntu]()
  > NOTE: The support for non-"Apple Silicon" is not our primary goal; but will test the functionallities some times if possible.
  > NOTE: For local run of LLM part, currently using `vLLM`.
  > Build your own cuda-available environment.
  > Check compatibilities in `requirements.txt` and install.
 
* Run:
  > Follow `run_*.sh` files to know how to try the system.
  > The `tests` directory is designed to contain several tests used during development, including unit tests and integration tests. These might be helpful to dig in the details of the framework. 

* Customization:
  > Make a branch and do whatever you want!
  > (Suggested) If possible, feel free to PR or ISSUE: your use-cases / new feature. 

## Roadmap
* Proof of Concept: Launch the initial PoC with minimal optimization considerations.
  * NOW: (MAC) Local Single-agent with single-(m)LLM-engine tested. Test on-going for the integration of tool calling by auto-parsing and structured output control.
  *       (Non-MAC) Local Single-agent with single-(m)LLM-engine tested.
* Optimization Iterations: Gradually integrate and test various ML/MLOps optimization techniques.
* Community Contributions: Encourage community-driven enhancements and experiments to further refine the framework.

## Contributing
Contributions are welcome! If you have ideas or improvements related to system optimization, multi-agent orchestration, or any other aspect, please feel free to open an issue or pull request. Our goal is to evolve AgentSandcastle into a robust platform that benefits the research community.

## Keywords
Local LLM Engine, Multimodal LLM, Multi-Agent System, Optimization, ML/MLOps, End-to-End Framework, Lightweight, Low-Dependency, Prototyping, Research-Driven
