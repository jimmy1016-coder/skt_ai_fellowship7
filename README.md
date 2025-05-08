# 🩻 SKT_AI_Fellowship7 VQA and Report Generation
This repository provides code for reproducing medical VQA and radiology report generation that we made for SKT_AI_Fellowship7. 

> ⚠️ **Note**: This repository assumes LLaVA is already installed and used as the default model.  
> If you want to use [LLaVA-Med](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b), simply change the `model_path` accordingly.

---

## 🔧 1. Data Preprocessing

We use the IU X-Ray dataset as our primary source.

To generate the preprocessed data file, run:

```bash
python iu_xray_preprocessing.py
