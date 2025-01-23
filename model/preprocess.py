from datasets import Dataset, load_dataset
import random
# import tpt
import pyconll
import re
from tqdm import tqdm

def clean_hf_dataset(x):
    x["la"] = clean_str(x["la"])
    return x

def find_matches(sent_data: Dataset | list[str], tree_dataset):
    if isinstance(sent_data, Dataset):
        _hf_dict = sent_data.select_columns(["la", "en", "id"]).to_dict()
                        
        hf_dict = dict(zip(_hf_dict["la"], _hf_dict["en"]))
        hf_ids = dict(zip(_hf_dict["la"], _hf_dict["id"]))
        hf_ids = {clean_str(i) : hf_ids[i] for i in hf_ids}
    else:
        hf_dict = {"la": sent_data, "en": ""}
        
    hf_dict_copy = dict(hf_dict)

    for la_key in tqdm(hf_dict.keys()):
        if any(x in hf_dict[la_key] for x in ['"', '.', ';', '?', '!'] )  and any(x in la_key for x in ['"', ':', '.', ';', '?', '!']):
            la_split = re.split(r'[;?!:.]|(?<=")[a-zA-Z]', la_key)
            en_split = re.split(r'[;?!:.]|(?<=")[a-zA-Z]', hf_dict[la_key])

            if len(la_split) != len(en_split):
                # for i in la_split:
                #     hf_dict_copy[la_split[i]] = en_split[i]
                continue

            del hf_dict_copy[la_key]

            for i in range(len(la_split)):
                hf_dict_copy[la_split[i]] = en_split[i]

    hf_dict_cleaned = {}
    
    for i in hf_dict_copy:
        hf_dict_cleaned[clean_str(i)] = clean_str(hf_dict_copy[i])
    
    hf_dict = dict(hf_dict_cleaned)

    UD_str_set = set([clean_str(s.text) for s in tree_dataset])

    matches = sorted(hf_dict.keys() & UD_str_set)

    trees = {clean_str(s.text): s for s in tree_dataset}

    base_matches = {}
    extra_matches = {}
    extra_ids = len(sent_data)

    for la in matches:
        if la in hf_ids:
            base_matches[hf_ids[la]] = {"la": [], "en": [], "tree": []}
            base_matches[hf_ids[la]]["la"] = trees[la].text
            base_matches[hf_ids[la]]["en"] = hf_dict[la]
            base_matches[hf_ids[la]]["tree"] = trees[la]
        else:
            extra_matches[extra_ids] = {}
            extra_matches[extra_ids]["la"] = trees[la].text
            extra_matches[extra_ids]["en"] = hf_dict[la]
            extra_matches[extra_ids]["tree"] = trees[la]
            
            extra_ids += 1
    
    # result = [i for i in hf_dict if i in matches]
    
    # matches = [x for _, x in sorted(zip(tree_matches, matches), key=lambda pair: pair[0].text)]

    # print(len(matches))

    non_matches = UD_str_set.difference(hf_dict.keys())
    non_matches_2 = hf_dict.keys() - UD_str_set

    return base_matches, extra_matches

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