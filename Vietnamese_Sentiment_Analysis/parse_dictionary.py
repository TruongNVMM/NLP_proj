import os
import json
import re

DICTIONARY_DIR = "dictionary/dictionaries"


def parse_vietnamese_dictionary_entry(line: str):
    text = line.strip()
    if not text:
        return None
    parts = text.split(" ", 1)
    word = parts[0].strip()

    synonyms = []
    antonyms = []

    m_syn = re.search(r"\[(.*?)\]", text)
    if m_syn:
        synonyms_text = m_syn.group(1)
        synonyms = [w.strip() for w in synonyms_text.split(",") if w.strip()]

    m_ant = re.search(r"trái nghĩa[:]\s*([^;]+)", text)
    if m_ant:
        antonyms_text = m_ant.group(1)
        antonyms = [w.strip() for w in antonyms_text.split(",") if w.strip()]

    return {
        "word": word,
        "synonyms": synonyms,
        "antonyms": antonyms,
    }

vn_synonym_dict = {}
vn_antonym_dict = {}

for root, dirs, files in os.walk(DICTIONARY_DIR):
    for file in files:
        file_path = os.path.join(root, file)
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                entry = parse_vietnamese_dictionary_entry(line)
                if entry is None:
                    continue

                w = entry["word"]
                if entry["synonyms"]:
                    vn_synonym_dict.setdefault(w, []).extend(entry["synonyms"])

                if entry["antonyms"]:
                    vn_antonym_dict.setdefault(w, []).extend(entry["antonyms"])

for w in list(vn_synonym_dict):
    vn_synonym_dict[w] = list(set(vn_synonym_dict[w]))

for w in list(vn_antonym_dict):
    vn_antonym_dict[w] = list(set(vn_antonym_dict[w]))

with open("sentiment_lexicon/vn_synonym_dict.json", "w", encoding="utf-8") as f:
    json.dump(vn_synonym_dict, f, ensure_ascii=False, indent=2)

with open("sentiment_lexicon/vn_antonym_dict.json", "w", encoding="utf-8") as f:
    json.dump(vn_antonym_dict, f, ensure_ascii=False, indent=2)
