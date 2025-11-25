# SQL-to-AST: Intelligent SQL Parsing with CodeT5 & FastAPI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📄 Abstract
This project addresses the fragility of traditional grammar-based SQL parsers by introducing a machine-learning-based **SQL-to-AST model**. By leveraging **Parameter-Efficient Fine-Tuning (LoRA)** on the **CodeT5p-220m** transformer model, we generate parser-compatible Abstract Syntax Trees (ASTs) directly from SQL queries. The model is exposed via a production-ready **REST API** built with **FastAPI**, enabling real-time inference, CRUD operations, and seamless integration into developer tools.

---

## 🚀 Key Features
- **Intelligent Parsing:** Converts SQL queries to JSON ASTs with high semantic fidelity using Deep Learning.
- **Efficient Training:** Utilizes **LoRA (Low-Rank Adaptation)** for GPU-efficient fine-tuning of CodeT5p-220m.
- **Scalable API:** Asynchronous REST endpoints (`/predict`, `/create`, `/health`) built on **FastAPI**.
- **Robust Validation:** Automated data validation using **Pydantic** schemas.
- **Documentation:** Auto-generated API docs via Swagger UI and ReDoc.

---

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Deep Learning:** PyTorch 2.0+, Hugging Face Transformers, PEFT (LoRA)
- **Model:** CodeT5p-220m
- **Web Framework:** FastAPI, Uvicorn, Starlette
- **Database/ORM:** SQLAlchemy
- **Dataset:** Spider Dataset (Yale University)

---

## 📂 Folder Structure
```text
SQL-to-AST-Model/
├── data/                   # Dataset storage (Spider dataset)
│   ├── raw/                # Original SQL-AST pairs
│   └── processed/          # Tokenized datasets
├── model_training/         # Training pipelines
│   ├── fine_tune.py        # Main LoRA training script
│   └── evaluate.py         # Script for SMA and ASS metrics
├── app/                    # FastAPI Application
│   ├── main.py             # Entry point for the API
│   ├── api/                # Route handlers
│   ├── models/             # Pydantic models & DB schemas
│   └── utils/              # Helper functions (AST parsing)
├── notebooks/              # Jupyter/Colab notebooks for experiments
├── tests/                  # Unit tests (Pytest)
├── saved_models/           # Checkpoints and LORA adapters
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
└── README.md               # Project Documentation
