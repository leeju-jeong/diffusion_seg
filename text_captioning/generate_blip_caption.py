# generate_blip_caption.py

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import json
from tqdm import tqdm

def generate_blip_captions(image_dir, id_txt_path, output_json, use_gpu=False):
    device = "cuda" if use_gpu else "cpu"

    # BLIP 모델 로딩
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    # 이미지 파일 리스트 불러오기
    with open(id_txt_path, 'r') as f:
        image_ids = [line.strip() for line in f]

    captions = {}
    for img_id in tqdm(image_ids):
        file_name = img_id
        img_path = os.path.join(image_dir, file_name)

        if not os.path.exists(img_path):
            print(f"[!] 파일 없음: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        captions[file_name] = caption

    # JSON 저장
    with open(output_json, 'w') as f:
        json.dump(captions, f, indent=2)

    print(f"✅ 캡션 저장 완료: {output_json}")

if __name__ == "__main__":
    # train
    # generate_blip_captions(
    #     image_dir="my/CamVid/CamVid_RGB",
    #     id_txt_path="my/CamVid/camvid_train.txt",
    #     output_json="train_captions.json",
    #     use_gpu=True
    # )

    # val
    generate_blip_captions(
    image_dir="my/CamVid/CamVid_RGB",      
    id_txt_path="my/CamVid/blip_val.txt",
    output_json="val_captions.json",
    use_gpu=False
)


    # test
    # generate_blip_captions(
    #     image_dir="my/CamVid/CamVid_RGB",
    #     id_txt_path="my/CamVid/camvid_test.txt",
    #     output_json="test_captions.json",
    #     use_gpu=True
    # )
