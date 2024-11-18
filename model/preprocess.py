from datasets import load_dataset
import random
import tpt
import pyconll
import re

def clean_hf_dataset(x):
    x["la"] = clean_str(x["la"])
    return x

def find_matches(hf_dataset, tree_dataset):
    hf_dict = ds.select_columns(["la", "en"]).to_dict()
                       
    hf_dict = dict(zip(hf_dict["la"], hf_dict["en"]))
    
    hf_dict_copy = dict(hf_dict)

    for la_key in hf_dict.keys():
        if ('"' or '.' or ';' or '?' or '!') in hf_dict[la_key] and ('"' or '.' or ';' or '?' or '!') in la_key:
            la_split = re.split(r'|"[a-zA-Z]|;|\? |\!', la_key)
            en_split = re.split(r'\. |"[a-zA-Z]|;|\? |\!', hf_dict[la_key])

            if len(la_split) != len(en_split):
                continue

            del hf_dict_copy[la_key]

            for i in range(len(la_split)):
                hf_dict_copy[la_split[i]] = en_split[i]

    hf_dict_cleaned = {}
    
    for i in hf_dict_copy:
        hf_dict_cleaned[clean_str(i)] = clean_str(hf_dict_copy[i])
    
    hf_dict = dict(hf_dict_cleaned)

    tree_set = set([clean_str(s.text) for s in treebank])

    matches = hf_dict.keys() & tree_set
    result = {match : hf_dict[match] for match in matches}

    print(len(matches))

    non_matches = tree_set.difference(hf_dict.keys())
    non_matches_2 = hf_dict.keys() - tree_set

    return result, non_matches, non_matches_2

def clean_str(str):
    return re.sub(r'[^a-zA-Z0-9\s]', '', str).lower().strip()

if __name__ == "__main__":
    ds = load_dataset("grosenthal/latin_english_parallel", split='train')
    a = ds[0]

    treebank_filepath = 'UD/UD_Latin-Perseus/la_perseus-ud-train.conllu'
    treebank = pyconll.load_from_file(treebank_filepath)

    matches, non_matches, non_matches_2 = find_matches(ds, treebank)

    print(len(matches))

    for i in range(5):
        match = random.choice(list(matches))
        print()
        print(f"la: {match}")
        print(f"en: {matches[match]}")

    print("===================")

    i = 0
    for s in non_matches:
        found = False
        for g in non_matches_2:
            if s in g and len(s) > 10:
                # print()
                # print(f"tr: {s}")
                # print(f"hf: {g}")
                found = True
            if found == True:
                i += 1
                break
        # if i >= 25:
        #     break
    
    print(i)