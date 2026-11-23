# P9 — Visual Question Answering (VQA)

**Method:** pretrained ViLT (`dandelin/vilt-b32-finetuned-vqa`), the *same frozen model* evaluated on two datasets.

**Datasets:** [VQA v2.0](https://visualqa.org/download.html) (real COCO images, human questions) and [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) (synthetic reasoning scenes).

**Original model code:** https://github.com/dandelin/ViLT  
**Pretrained checkpoint:** https://huggingface.co/dandelin/vilt-b32-finetuned-vqa

## Run (Google Colab, Kaggle with Internet, or local Python)

1. Open a Python environment with internet and install `pip install -r requirements.txt`.
2. Download **VQA v2 validation images**, **validation questions**, and **validation annotations** from visualqa.org. Unzip them.
3. Download **CLEVR v1.0 validation images** and **validation questions** from Stanford. Unzip them.
4. Run these commands, replacing paths with the actual extracted folders:

```bash
python evaluate.py --dataset vqa --images /path/to/val2014 --questions /path/to/v2_OpenEnded_mscoco_val2014_questions.json --annotations /path/to/v2_mscoco_val2014_annotations.json --limit 100
python evaluate.py --dataset clevr --images /path/to/CLEVR_v1.0/images/val --questions /path/to/CLEVR_v1.0/questions/CLEVR_val_questions.json --limit 100
```

5. Check `results/vqa_summary.json`, `results/clevr_summary.json`, and the two prediction CSV files. Enter the **measured** scores in the presentation. Do not claim results before running the experiments.

**CPU works**, but inference can be slow. Use `--limit 10` for a smoke test, then increase to 100 or more. The checkpoint download needs internet once; use `--model-path /local/model/folder` if it is already available offline.

## Experimental caveats

- The checkpoint was fine-tuned for VQA, **not CLEVR**. Lower CLEVR accuracy would reflect domain shift, not a fair head-to-head benchmark.
- VQA uses leave-one-out agreement over human annotators; CLEVR uses exact match. **The accuracy numbers are not directly comparable** because metrics differ.
- We do not train or fine-tune on either dataset; this is an inference-only evaluation.
- VQA normalization here is simplified (case/whitespace), so results may differ from the official evaluation server (which also normalizes punctuation, numbers, articles).
- Dataset images and checkpoint weights are not committed because they are large and have separate licenses.

## Files
- `evaluate.py`: complete evaluation script
- `requirements.txt`: dependencies
- `PROJECT_REPORT.md`: presentation-ready method, datasets, results placeholders and conclusion

## References
Kim et al., *ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision*, ICML 2021.  
Goyal et al., *Making the V in VQA Matter*, CVPR 2017.  
Johnson et al., *CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning*, CVPR 2017.
