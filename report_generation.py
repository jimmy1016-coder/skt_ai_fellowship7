#!/usr/bin/env python3
# evaluate_llava_med_reload_each.py

import os
import json
import gc
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast

# inference
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# metrics
import nltk
nltk.download('wordnet', quiet=True)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

def eval_model_with_components(model, tokenizer, image_processor, args):
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
        IMAGE_PLACEHOLDER,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images
    import re

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode or "llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    from PIL import Image
    image = Image.open(args.image_file).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
    image_size = [image.size]

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_size,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

def run_single_inference(model, tokenizer, image_processor, img_path, prompt):
    args = type("Args", (), {
        "query": prompt,
        "conv_mode": None,
        "image_file": img_path,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512,
    })()

    with torch.no_grad():
        with autocast():
            pred = eval_model_with_components(model, tokenizer, image_processor, args)
    return pred

def main():
    model_path = "liuhaotian/llava-v1.5-7b"
    prompt = (
        "You are a professional radiologist. You are provided with an X-ray image. "
        "Please generate a diagnostic report based solely on the image. "
        "Do not refer to any external knowledge or previous reports. "
        "Your report should describe the condition of the lungs, heart size, "
        "presence of pneumothorax, and any abnormal findings."
    )

    # Load test data
    json_path = "/home/sooyoung/skt_ai_fellowship/MMed-RAG/data/test/report/iuxray_test.json"
    with open(json_path, "r") as f:
        test_data = json.load(f)

    base_img_dir = "/home/sooyoung/skt_ai_fellowship/iu_xray/images"
    refs, hyps = [], []

    # Load model, tokenizer, image_processor once
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device="cuda"
    )
    model.config.use_cache = False
    model.eval()
    model.half()

    results = []
    smoothie = SmoothingFunction().method4
    rouge = Rouge()

    for item in tqdm(test_data, desc="Running inference"):
        rel_path = item["image_path"][0]
        img_path = os.path.join(base_img_dir, rel_path)
        ref = item["report"].strip()
        hyp = run_single_inference(model, tokenizer, image_processor, img_path, prompt)

        refs.append(ref)
        hyps.append(hyp)

        bleu = sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothie)
        meteor = meteor_score([ref.split()], hyp.split())
        rouge_score = rouge.get_scores(hyp, ref)[0]["rouge-l"]["f"]

        results.append({
            "image_path": rel_path,
            "ground_truth": ref,
            "prediction": hyp,
            "bleu": bleu,
            "meteor": meteor,
            "rouge_l": rouge_score
        })

    print("\n===== Evaluation Results =====")
    print(f"Samples evaluated: {len(results)}")
    print(f"BLEU   : {sum(r['bleu'] for r in results) / len(results):.4f}")
    print(f"METEOR : {sum(r['meteor'] for r in results) / len(results):.4f}")
    print(f"ROUGE-L F1: {sum(r['rouge_l'] for r in results) / len(results):.4f}")

    # Save predictions and references to a JSON file for qualitative analysis
    with open("llava_predictions.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()