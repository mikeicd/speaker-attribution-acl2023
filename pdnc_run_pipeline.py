from booknlpen.english.english_booknlp import EnglishBookNLP as BookNLP
import os
from pathlib import Path

# model_params = {"pipeline": "entity,quote,supersense,event,coref", "model": "big"}

entityName="entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model"
corefName="coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model"
quoteAttribName="speaker_bert-base-uncased.model"

home = str(Path.home())
repo_root = Path(__file__).resolve().parent
modelsPath=os.path.join(home, "booknlp_models")
repo_modelsPath=os.path.join(repo_root, "training/trained_models/lxo_split_0_uncased")

model_params={
		"pipeline":"entity,quote,supersense,event,coref",
		"model":"custom",
        "entity_model_path": os.path.join(modelsPath, entityName),
        "coref_model_path": os.path.join(modelsPath, corefName),
        "quote_attribution_model_path": os.path.join(repo_modelsPath, quoteAttribName)
	}

booknlp = BookNLP(model_params)

IN_ROOT = "data/pdnc_source"
OUT_ROOT = "booknlpen/pdnc_output"
os.makedirs(OUT_ROOT, exist_ok=True)

novels = []
ip_paths = []
op_paths = []

for novel in os.scandir(IN_ROOT):
    if os.path.isdir(novel) and novel.name[0] not in [".", "_"]:
        novel_name = novel.name
        novels.append(novel_name)
        ip_paths.append(os.path.join(novel.path, "novel.txt"))
        op_path = os.path.join(OUT_ROOT, novel_name)
        if not os.path.isdir(op_path):
            os.mkdir(op_path)

        op_paths.append(op_path)

assert len(novels) == len(ip_paths) == len(op_paths), print("Length Mismatch!")

for bname, ip, op in zip(novels, ip_paths, op_paths):
    print("Processing: ", bname)
    booknlp.process(ip, op, bname)
    print("Done")
