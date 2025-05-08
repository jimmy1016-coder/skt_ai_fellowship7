import os
import json
import re
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.utils import disable_torch_init
from PIL import Image

def strip_image_tokens(s: str) -> str:
    return re.sub(r"<image(_\d+)?>", "", s, flags=re.IGNORECASE).strip()

def image_parser(args):
    return args.image_file.split(args.sep)

def load_image(image_file):
    return Image.open(image_file).convert("RGB")

def load_images(image_files):
    return [load_image(f) for f in image_files]

def eval_model(args, tokenizer, model, image_processor, context_len):
    disable_torch_init()

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        qs = re.sub(IMAGE_PLACEHOLDER, image_token_se if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN, qs)
    else:
        qs = (image_token_se if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN) + "\n" + qs

    conv_mode = args.conv_mode or "llava_v1"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

# Load data
with open("/home/heawon/LLaVA-Med/iuxray_preprocessed.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f]

# Load RAG retrieval results
with open("/home/heawon/LLaVA-Med/retrieved_reports_by_image.json", "r") as f:
    retrieval_dict = json.load(f)

# Result tracking for different k values
k_values = [1, 2, 3, 5]
results = {k: {'y_true': [], 'y_pred': [], 'y_score': []} for k in k_values}

# Load model
model_path = "microsoft/llava-med-v1.5-mistral-7b"
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="cuda" if torch.cuda.is_available() else "cpu"
)
model.eval()

# Evaluation loop
for item in tqdm(dataset):
    question = strip_image_tokens(item["text"])
    rel_path = item["image_path"]
    image_file = os.path.abspath(rel_path)
    label = item["answer"].strip().lower()

    # Construct retrieval key
    retrieval_key = os.path.join(os.path.basename(os.path.dirname(rel_path)), os.path.basename(rel_path))

    if not os.path.exists(image_file):
        print(f"🚫 Image not found: {image_file}")
        continue

    if retrieval_key not in retrieval_dict:
        print(f"⚠️ Skipping: {retrieval_key} not found in retrieval dictionary.")
        continue

    retrieved_reports = retrieval_dict[retrieval_key]

    for k in k_values:
        # Get top-k reports
        selected_reports = retrieved_reports[:k]

        # Construct prompt
        ref_text = "\n".join([f"{i+1}. {r}" for i, r in enumerate(selected_reports)])
        ref_prompt = (
            f"You are a professional radiologist. You are provided with an X-ray image and {len(selected_reports)} reference report(s):\n{ref_text}\n"
            f"Please answer the following question based on the image and the reference reports. "
            f"It should be noted that the diagnostic information in the reference reports cannot be directly used as the basis for diagnosis, "
            f"but should only be used for reference and comparison.\n\n"
            f"Question: {question}\nPlease answer strictly with 'yes' or 'no'."
        )

        # Create args
        args = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "model_name": model_name,
            "query": ref_prompt,
            "conv_mode": None,
            "image_file": image_file,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()

        # Run model
        output = eval_model(args, tokenizer, model, processor, context_len)

        # Process output
        output_text = output.strip().lower() if isinstance(output, str) else str(output).strip().lower()
        pred = 1 if "yes" in output_text and "no" not in output_text else (0 if "no" in output_text else -1)

        # Skip invalid predictions
        if label not in ["yes", "no"] or pred == -1:
            continue

        # Store results
        results[k]['y_true'].append(1 if label == "yes" else 0)
        results[k]['y_pred'].append(pred)
        results[k]['y_score'].append(1 if pred == 1 else 0)

# Show results for each k
for k in k_values:
    y_true = results[k]['y_true']
    y_pred = results[k]['y_pred']
    y_score = results[k]['y_score']

    print(f"\n✅ Results for k={k}:")
    if y_true:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        try:
            auroc = roc_auc_score(y_true, y_score)
        except:
            auroc = "Cannot be computed (only one class present)"
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUROC: {auroc}")
    else:
        print("⚠️ No valid samples evaluated.")