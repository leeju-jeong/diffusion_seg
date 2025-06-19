#text 파일 blip에 사용할 수 있도록 정제.

import os

def extract_ids_from_split(split_path, output_path):
    with open(split_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            rgb_path = line.strip().split()[0]  # "CamVid_RGB/0016E5_08101.png"
            base = os.path.basename(rgb_path)  
            fout.write(base + '\n')
    print(f"✅ 저장 완료: {output_path}")

# 예시 실행
if __name__ == "__main__":
    # extract_ids_from_split(
    #     split_path="my/CamVid/camvid_train.txt",
    #     output_path="my/CamVid/blip_train.txt"
    # )
    
    # extract_ids_from_split(
    #     split_path="my/CamVid/camvid_test.txt",
    #     output_path="my/CamVid/blip_test.txt"
    # )

    extract_ids_from_split(
        split_path="my/CamVid/camvid_val.txt",
        output_path="my/CamVid/blip_val.txt"
    )
