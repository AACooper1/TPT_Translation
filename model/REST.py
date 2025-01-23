import requests, json
from tqdm import tqdm

url = 'https://lindat.mff.cuni.cz/services/udpipe/api/process'

open("data/output.conllu", "w")

with open("data/dataset.txt", 'r') as in_file:
    s = in_file.readlines()

for i in tqdm(range(0, len(s), 10000)):

    r = {
            "data": ''.join(s[i:i+10000],),
            "model": "latin-evalatin24-240520",
            "tokenizer": "",
            "tagger": "",
            "parser": "",
            "output": "conllu"
    }

    response = requests.post(url, data=r)

    result = json.loads(response.content)['result']

    with open("data/output.conllu", "a") as out_file:
        out_file.write(result + "\n\n")