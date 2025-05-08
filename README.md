# 🩻 SKT_AI_Fellowship7 Codebase
This repository provides code for reproducing medical VQA and radiology report generation that we made for SKT_AI_Fellowship7. 

> ⚠️ **Note**: This repository assumes [LLaVA](https://github.com/haotian-liu/LLaVA) is already installed and used as the default model.  
> If you want to use [LLaVA-Med](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b), simply change the `model_path` accordingly.

---

## 🔧 1. Data Preprocessing

We use the IU [X-Ray](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view) dataset for our toy experiments.

To generate the preprocessed data file, run:

```bash
python iu_xray_preprocessing.py
```

This will create iuxray_preprocessed.jsonl containing cleaned VQA dataset. You can use it for VQA task!
(✅ However, running this script is optional – the preprocessed file is already provided.)

---

## 2. VQA

To reproduce the VQA results, run:

```bash
# VQA without RAG
python llava_vqa.py

# VQA with RAG (varying the number of retrieved documents) 
python llava_vqa_rag_k_variants.py

# VQA with adaptive RAG (proposed by [MMed_RAG](https://arxiv.org/abs/2410.13085))
python llava_vqa_adaptive_rag.py 
```
