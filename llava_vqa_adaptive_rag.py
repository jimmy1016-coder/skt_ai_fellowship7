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
    """이미지 토큰을 제거하고 텍스트만 남기는 함수"""
    return re.sub(r"<image(_\d+)?>", "", s, flags=re.IGNORECASE).strip()

def load_image(image_file):
    """이미지 파일을 불러오는 함수"""
    return Image.open(image_file).convert("RGB")

def load_images(image_files):
    """이미지 파일 리스트를 불러와 PIL 이미지 객체 리스트로 반환"""
    return [load_image(f) for f in image_files]

def retrieve_topk_per_image(logits, k, retrieve_threshold=None):
    """
    Adaptive Retrieval 함수
    - logits: 이미지-텍스트 유사도 점수 텐서
    - k: 최대 top-k 개수
    - retrieve_threshold: 선택 임계값 (없으면 고정된 k만 사용)
    """
    pred_per_val_image = []

    for logit_values in logits["image_to_text"]:
        # top-1 logit
        top1_logit = logit_values.max()
        sorted_logits, sorted_indices = torch.sort(logit_values, descending=True)

        # 임계값 기반 동적 top-k 선택
        if retrieve_threshold:
            ratios = top1_logit / sorted_logits
            selected_indices = sorted_indices[ratios < retrieve_threshold]

            # 선택된 개수가 k보다 많을 경우 잘라내기
            if len(selected_indices) > k:
                selected_indices = selected_indices[:k]
            # 선택된 개수 출력
            print(f"🔍 Retrieved {len(selected_indices)} reports using threshold {retrieve_threshold} (max k={k})")
            pred_per_val_image.append(selected_indices)
        else:
            # 고정된 top-k 선택
            pred_per_val_image.append(sorted_indices[:k])

            # 선택된 개수 출력
            print(f"🔍 Retrieved {len(selected_indices)} reports (fixed k={k})")
            pred_per_val_image.append(selected_indices)

    return pred_per_val_image

def eval_model(args, tokenizer, model, image_processor, context_len):
    """모델 평가 함수 - LLaVA 모델에 프롬프트와 이미지를 입력하여 응답 생성"""
    disable_torch_init()

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    # 프롬프트 내 이미지 토큰 처리
    if IMAGE_PLACEHOLDER in qs:
        qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
    else:
        qs = image_token_se + "\n" + qs

    # 대화 모드 설정
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 이미지 로드
    images = load_images([args.image_file])
    image_sizes = [img.size for img in images]
    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)

    # 입력 토큰 생성
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

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

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

# 1. 데이터 로드
with open("/home/heawon/LLaVA-Med/iuxray_preprocessed.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f]

# 2. Retrieval 결과 로드
with open("/home/heawon/LLaVA-Med/retrieved_reports_by_image.json", "r") as f:
    retrieval_dict = json.load(f)

# 3. 결과 저장용 리스트
y_true, y_pred, y_score = [], [], []

# 4. 모델 로드
model_path = "microsoft/llava-med-v1.5-mistral-7b"
tokenizer, model, processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    load_8bit=False,
    load_4bit=False,
    device_map="cuda" if torch.cuda.is_available() else "cpu"
)
model.eval()

# 5. 평가 루프
for item in tqdm(dataset):
    question = strip_image_tokens(item["text"])
    rel_path = item["image_path"]
    image_file = os.path.abspath(rel_path)
    label = item["answer"].strip().lower()

    # 이미지 파일이 존재하지 않으면 스킵
    if not os.path.exists(image_file):
        print(f"🚫 Image not found: {image_file}")
        continue

    # retrieval_dict 키 생성
    retrieval_key = os.path.join(os.path.basename(os.path.dirname(rel_path)), os.path.basename(rel_path))
    
    if retrieval_key not in retrieval_dict:
        print(f"⚠️ Skipping: {retrieval_key} not found in retrieval dictionary.")
        continue

    # Retrieval 점수 계산 (모의 데이터 생성)
    logit_values = torch.rand(len(retrieval_dict[retrieval_key]))  # 예시: 무작위 점수 생성

    # Adaptive Retrieval 적용
    selected_indices = retrieve_topk_per_image({"image_to_text": [logit_values]}, k=5, retrieve_threshold=1.5)[0] # retrieve_threshold = 0.9
    selected_reports = [retrieval_dict[retrieval_key][idx] for idx in selected_indices]

    # Prompt 생성
    ref_text = "\n".join([f"{i+1}. {r}" for i, r in enumerate(selected_reports)])
    ref_prompt = (
        f"You are a professional radiologist. You are provided with an X-ray image and {len(selected_reports)} reference report(s):\n{ref_text}\n"
        "Please answer the following question based on the image and the reference reports. "
        "It should be noted that the diagnostic information in the reference reports cannot be directly used as the basis for diagnosis, "
        "but should only be used for reference and comparison.\n\n"
        f"Question: {question}\nPlease answer strictly with 'yes' or 'no'."
    )

    # LLaVA-Med 실행
    args = type('Args', (), {
        "query": ref_prompt,
        "image_file": image_file,
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    # 모델 응답 생성
    output = eval_model(args, tokenizer, model, processor, context_len)
    output_text = output.strip().lower()

    # "yes"/"no" 판단
    if "yes" in output_text and "no" not in output_text:
        pred = 1
    elif "no" in output_text:
        pred = 0
    else:
        pred = -1  # 애매한 응답은 무시

    if label not in ["yes", "no"] or pred == -1:
        continue

    y_true.append(1 if label == "yes" else 0)
    y_pred.append(pred)
    y_score.append(1 if pred == 1 else 0)

# 6. 결과 출력
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

# 7. 개별 결과 출력
print("\n🧾 Detailed Predictions (True vs Pred):")
for i, (yt, yp) in enumerate(zip(y_true, y_pred), 1):
    print(f"{i:02d}: True = {yt}, Pred = {yp}")