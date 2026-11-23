# P9 — Visual Question Answering: Slide Content

## Slide 1 — Title
Visual Question Answering (VQA): Evaluation of ViLT on VQA v2 and CLEVR.

## Slide 2 — Problem
Input: an image and a natural-language question. Output: an answer grounded in the image.

## Slide 3 — Method
ViLT is a vision-and-language transformer. Load pretrained checkpoint `dandelin/vilt-b32-finetuned-vqa`, preprocess each image-question pair, run inference, choose highest-scoring answer label. No additional training.

## Slide 4 — Dataset 1
VQA v2: COCO photographs, natural-language questions, 10 human answers per question. Use validation split.

## Slide 5 — Dataset 2
CLEVR: synthetic 3D scenes with questions involving attributes, counts, spatial relationships, and reasoning. Use validation split.

## Slide 6 — Experiment
Same checkpoint on both datasets; random seed 42, sample size [FILL FROM RUN]. VQA metric: leave-one-out human agreement. CLEVR metric: exact-match accuracy.

## Slide 7 — Results (fill after execution)
| Dataset | Evaluated samples | Accuracy |
|---|---:|---:|
| VQA v2 validation | NOT RUN | NOT RUN |
| CLEVR validation | NOT RUN | NOT RUN |

Include 2 correct and 2 incorrect predictions from the generated CSVs.

## Slide 8 — Discussion
Consider pretrained VQA answer vocabulary, synthetic-to-real domain shift, differences in reasoning complexity and non-equivalent scoring rules. Do not infer superiority directly from different metric percentages.

## Slide 9 — Conclusion
Summarize actual results after running. Limitations: no fine-tuning, simplified VQA normalization, subset evaluation, potentially large dataset downloads.

## Slide 10 — References
https://github.com/dandelin/ViLT
https://huggingface.co/dandelin/vilt-b32-finetuned-vqa
https://visualqa.org/download.html
https://cs.stanford.edu/people/jcjohns/clevr/
