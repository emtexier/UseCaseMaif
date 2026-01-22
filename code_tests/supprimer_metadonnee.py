import os
import re

input_dir = r"E:\Downloads\AlloSat_corpus\AlloSat_corpus\stm"
output_dir = r"E:\Downloads\AlloSat_corpus\AlloSat_corpus\usager_text"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".stm"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".txt")

        clean_lines = []

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = re.search(r"> ?(.*)", line)
                if match:
                    text = match.group(1).strip()
                    text = re.sub(r"\[.*?\]|【.*?】", "", text)
                    if text:
                        clean_lines.append(text)

        if clean_lines:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(" ".join(clean_lines))

        print(f"Processed {file} → {output_path}")

print("✅ All STM files cleaned and saved as TXT!")
