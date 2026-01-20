import os
from jiwer import wer

truth_dir = r"E:\Downloads\AlloSat_corpus\AlloSat_corpus\usager_text"
pred_dir = r"E:\Downloads\AlloSat_corpus\AlloSat_corpus\outputs"

total_wer = 0
file_count = 0

for pred_file in os.listdir(pred_dir):  
    if pred_file.endswith(".txt"):
        pred_path = os.path.join(pred_dir, pred_file)
        truth_path = os.path.join(truth_dir, pred_file)  

        if not os.path.exists(truth_path):
            print(f"⚠️  Warning: {pred_file} missing in ground truth, skipping")
            continue

        with open(pred_path, "r", encoding="utf-8") as f:
            pred_text = f.read().strip()
        with open(truth_path, "r", encoding="utf-8") as f:
            truth_text = f.read().strip()
            print(truth_text)

        score = wer(truth_text, pred_text)
        print(f"{pred_file} WER: {score:.3f}")
        total_wer += score
        file_count += 1

if file_count > 0:
    avg_wer = total_wer / file_count
    print(f"\n✅ Average WER on {file_count} files: {avg_wer:.3f}")
