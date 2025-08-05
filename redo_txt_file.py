in_path  = "yiddish_tacotron2_train_data_FIXED.txt"
out_path = "yiddish_tacotron2_train_data_FIXED_2col.txt"

with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
    for line in fin:
        parts = line.rstrip("\n").split("|")
        if len(parts) < 2:
            continue
        wav, text = parts[0], parts[1]
        fout.write(f"{wav}|{text.strip()}\n")
