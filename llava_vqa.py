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
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from PIL import Image

def strip_image_tokens(s: str) -> str:
    return re.sub(r"<image(_\d+)?>", "", s, flags=re.IGNORECASE).strip()

# eval_model인데 모델 리로딩만 바꾼버전
def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def eval_model(args, tokenizer, model, image_processor, context_len):
    disable_torch_init()

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

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs



# 1. 데이터 로드
with open("iuxray_preprocessed.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f]

# 2. 결과 저장용 리스트
y_true = []
y_pred = []
y_score = []

# 3. 모델 로드
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
model.eval()  # ⭕️ 진짜 모델에만 eval() 적용

# 4. 평가 루프
for item in tqdm(dataset):
    question = strip_image_tokens(item["text"])
    rel_path = item["image_path"]
    image_file = os.path.abspath(rel_path)
    label = item["answer"].strip().lower()

    # 이미지 파일 존재 확인
    if not os.path.exists(image_file):
        print(f"🚫 Image not found: {image_file}")
        continue

    # LLaVA-Med 실행용 인자 구성
    args = type('Args', (), {
        "model_path": model_path, # 안씀
        "model_base": None,       # 안씀
        "model_name": model_name, # 안씀
        "query": question,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    output = eval_model(args, tokenizer, model, processor, context_len)

    # 출력 후처리
    if isinstance(output, str):
        output_text = output.strip().lower()
    elif isinstance(output, dict) and "text" in output:
        output_text = output["text"].strip().lower()
    else:
        output_text = str(output).strip().lower()

    # "yes"/"no" 판단
    if "yes" in output_text and "no" not in output_text:
        pred = 1
    elif "no" in output_text:
        pred = 0
    else:
        pred = -1  # 애매하거나 이상한 응답

    # 정답도 yes/no가 아니면 스킵
    if label not in ["yes", "no"] or pred == -1:
        continue

    y_true.append(1 if label == "yes" else 0)
    y_pred.append(pred)
    y_score.append(1 if pred == 1 else 0)

# 5. 결과 출력
print("\n✅ Evaluation Results")
if y_true:
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("F1 Score:", round(f1_score(y_true, y_pred), 4))
    try:
        print("AUROC:", round(roc_auc_score(y_true, y_score), 4))
    except:
        print("AUROC: Cannot be computed (only one class present)")
else:
    print("⚠️ No valid samples evaluated.")

# 6. 개별 결과 출력
print("\n🧾 Detailed Predictions (True vs Pred):")
for i, (yt, yp) in enumerate(zip(y_true, y_pred), 1):
    print(f"{i:02d}: True = {yt}, Pred = {yp}")
