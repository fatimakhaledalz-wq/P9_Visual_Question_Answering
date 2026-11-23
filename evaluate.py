"""Evaluate the same pretrained ViLT model on VQA v2 and CLEVR.
Dataset files are NOT bundled. Results are only generated after real inference.
"""
import argparse, json, csv, os, random
from pathlib import Path
from PIL import Image
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering

MODEL_ID = "dandelin/vilt-b32-finetuned-vqa"

def normalize(s):
    return " ".join(str(s).lower().strip().split())

def load_examples(kind, images, questions, annotations, limit, seed):
    qdata=json.loads(Path(questions).read_text())
    qrows=qdata["questions"]
    amap={}
    if annotations:
        adata=json.loads(Path(annotations).read_text())
        amap={int(a["question_id"]):a for a in adata["annotations"]}
    examples=[]
    for q in qrows:
        if kind=="vqa":
            qid=int(q["question_id"])
            if qid not in amap: continue
            img=Path(images)/("COCO_val2014_%012d.jpg"%int(q["image_id"]))
            if not img.exists(): continue
            answers=[a["answer"] for a in amap[qid]["answers"]]
            examples.append((str(img),q["question"],answers,qid))
        else:
            img=Path(images)/q["image_filename"]
            if not img.exists() or "answer" not in q: continue
            examples.append((str(img),q["question"],[str(q["answer"])],q.get("question_index",len(examples))))
    random.Random(seed).shuffle(examples)
    return examples[:limit] if limit>0 else examples

def score(kind,pred,answers):
    p=normalize(pred)
    if kind=="clevr": return float(p==normalize(answers[0]))
    # Standard VQA leave-one-out accuracy, averaged across 10 annotators.
    return sum(min(1.,sum(normalize(a)==p for j,a in enumerate(answers) if j!=i)/3.)
               for i in range(len(answers)))/len(answers)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=["vqa","clevr"],required=True)
    ap.add_argument("--images",required=True)
    ap.add_argument("--questions",required=True)
    ap.add_argument("--annotations",default=None)
    ap.add_argument("--limit",type=int,default=100)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--model-path",default=MODEL_ID)
    ap.add_argument("--output",default="results")
    args=ap.parse_args()
    if args.dataset=="vqa" and not args.annotations:
        ap.error("--annotations required for VQA v2 scoring")
    data=load_examples(args.dataset,args.images,args.questions,args.annotations,args.limit,args.seed)
    if not data: raise RuntimeError("No labeled examples found. Check paths, split and filenames.")
    print("Loading model:",args.model_path)
    processor=ViltProcessor.from_pretrained(args.model_path)
    model=ViltForQuestionAnswering.from_pretrained(args.model_path)
    model.eval()
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,(path,question,answers,qid) in enumerate(data,1):
        with Image.open(path) as im:
            encoded=processor(images=im.convert("RGB"),text=question,return_tensors="pt",truncation=True)
        encoded={k:v.to(device) for k,v in encoded.items()}
        with torch.inference_mode():
            logits=model(**encoded).logits
        prediction=model.config.id2label[int(logits.argmax(-1).item())]
        accuracy=score(args.dataset,prediction,answers)
        rows.append(dict(id=qid,image=path,question=question,prediction=prediction,
                         ground_truth=" | ".join(answers),score=round(accuracy,4)))
        if i%10==0:print(f"{i}/{len(data)} completed")
    with (out/f"{args.dataset}_predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={"dataset":args.dataset,"model":args.model_path,"n":len(rows),
             "accuracy":round(sum(r["score"] for r in rows)/len(rows),4),
             "metric":"VQA leave-one-out" if args.dataset=="vqa" else "exact match",
             "seed":args.seed}
    (out/f"{args.dataset}_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
