import json

input_path = "iuxray_test.jsonl"
output_path = "iuxray_preprocessed.jsonl"

with open(input_path, "r") as infile, open(output_path, "w") as outfile:
    for line in infile:
        item = json.loads(line)

        # 구성요소 추출 및 재구성
        new_item = {
            "question_id": item.get("question_id"),
            "question": item.get("question", "").strip(),
            "image_path": f"iu_xray/images/{item.get('image')}",  # 경로 추가
            "answer": item.get("answer", "").strip(),
            "text": item.get("text", "").strip()
        }

        # 저장
        outfile.write(json.dumps(new_item) + "\n")

print("✅ Preprocessing complete! Saved to", output_path)