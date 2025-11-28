# DPR Fine-Tuning Guide

This repository contains scripts for fine-tuning a Dense Passage Retrieval (DPR) model. 

## 🎯 Choose Your Fine-Tuning Setting

There are two available modes depending on your hardware resources and training requirements.

### 1. Basic Mode (`--mode basic`)

* **Best for:** machines with high RAM (64GB+).
* **How it works:** Loads the entire corpus into memory.
* **Negative Sampling:** Uses **Random Negatives** (samples random passages from the corpus as negatives).
* **Precision:** Standard FP32 training.

### 2. Advanced Mode (`--mode advanced`)

* **Best for:** GPU acceleration.
* **How it works:** Uses a temporary SQLite database to stream data from disk, saving RAM.
* **Negative Sampling:** Uses **Hard Negatives**. It mines difficult negatives using BM25 (passages that look similar to the query but are not relevant) to create a stronger training signal.
* **Precision:** Uses Mixed Precision (FP16) for faster and more memory-efficient training.

---

## 🚀 How to Run

Make sure you are in the project root directory and have installed the requirements (`pip install -r requirements.txt`).

**To run the Basic setting:**
```bash
python run.py --mode basic
```
**To run the advanced setting:**
```bash
python run.py --mode advanced
```