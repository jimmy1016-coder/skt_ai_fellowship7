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

# Result tracking
y_true, y_pred, y_score = [], [], []

# Load model
model_path = "liuhaotian/llava-v1.5-7b"
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

# Eval loop
for item in tqdm(dataset):
    question = strip_image_tokens(item["text"])
    rel_path = item["image_path"]
    image_file = os.path.abspath(rel_path)
    label = item["answer"].strip().lower()
    if not os.path.exists(image_file):
        print(f"🚫 Image not found: {image_file}")
        continue

    # Build RAG-style prompt
    image_name = os.path.basename(image_file)
    retrieved_reports = retrieval_dict.get(image_name, [])
    ref_text = "\n".join([f"{i+1}. {r}" for i, r in enumerate(retrieved_reports)])
    ref_prompt = (
        f"You are a professional radiologist. You are provided with an X-ray image and {len(retrieved_reports)} reference report(s):\n{ref_text}\n"
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
    if label not in ["yes", "no"] or pred == -1:
        continue

    y_true.append(1 if label == "yes" else 0)
    y_pred.append(pred)
    y_score.append(1 if pred == 1 else 0)

# Show results
print("\n Evaluation Results")
if y_true:
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("F1 Score:", round(f1_score(y_true, y_pred), 4))
    try:
        print("AUROC:", round(roc_auc_score(y_true, y_score), 4))
    except:
        print("AUROC: Cannot be computed (only one class present)")
else:
    print("No valid samples evaluated.")

print("\n Detailed Predictions (True vs Pred):")
for i, (yt, yp) in enumerate(zip(y_true, y_pred), 1):
    print(f"{i:02d}: True = {yt}, Pred = {yp}")
