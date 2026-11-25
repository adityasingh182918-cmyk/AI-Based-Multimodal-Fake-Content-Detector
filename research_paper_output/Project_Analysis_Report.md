# Project analysis report

Audit date: 2026-11-25. Phase 1 is a ResNet-18 image classifier; Phase 2 contains HC3 SVM/DistilBERT and expanded HC3+MAGE candidates; Phase 3 is deterministic rule-based combination. No neural fusion checkpoint or paired training data exists. FastAPI source defaults to the strict CIFAKE image checkpoint and HC3 SVM unless environment variables override them.

## Datasets
- **CIFAKE subset:** pinned `dragonintelligence/CIFAKE-image-dataset` revision `6eebd6090fe3c0d9d44f899a25f0a34e3c2dad31`; 2,000 32×32 images, train 1,600 (800/class), validation 200 (100/class), test 200 (100/class). RGB, pad resize to 224×224, ImageNet normalization.
- **HC3:** `reddit_eli5` + `wiki_csai`, revision `4d0ff18143b5a7e1b1e79beb540c04549d1e59d3`; group split: train 2,476 (1,226 human/1,250 AI), val 531 (253/278), test 531 (253/278).
- **Expanded HC3+MAGE:** train 26,954 (13,224 human/13,730 AI), val 6,650 (3,252/3,398), test 5,727 (2,569/3,158). Exact duplicates removed; MAGE source IDs unavailable, so semantic leakage is not fully ruled out.
- **OOD:** MAGE GPT-4 n=1,562; GRADTEX Test B n=2,642 and C n=3,529 (human-written vs AI-involved MIX); Beemo n=656 positive-only edited-AI probe.

## Recovered results
- Strict CIFAKE ResNet-18: accuracy 0.9750, AI precision 0.9612, recall 0.9900, F1 0.9754; matrix `[[96,4],[1,99]]` in REAL/AI order. Validation-selected abstention test coverage was 85%, covered accuracy 100%, 30 uncertain; distinct from full-set accuracy. Softmax is uncalibrated.
- Expanded same test n=5,727: SVM accuracy/macro-F1 0.7524/0.7523; DistilBERT 0.8537/0.8479. DistilBERT human false-positive rate 0.2655, AI recall 0.9506.
- HC3-only MAGE GPT-4 macro-F1: SVM 0.6367; DistilBERT 0.7144. GRADTEX C 0.4652/0.4124; B 0.4561/0.4361. Beemo positive-only recall 0.3003/0.3476; two-class metrics invalid.
- Saved project record: 84 software tests passed, one non-failing PyTorch sparse warning; tests are not accuracy and were not rerun for this paper.

## Limits
CIFAKE evidence is a small same-benchmark low-resolution split. Later fine-tune `resnet18_realworld_v1.pt` includes four user-labeled REAL images in training; not independent evidence. Text transfer degrades substantially; GRADTEX MIX includes AI assistance, and Beemo has only positives. No human grammar-correction set or eligible paired multimodal data is present. Phase 3 has no learned fusion metrics. All output derives from existing reports/code; project models and code were not modified.

## Output verification

The DOCX and XLSX ZIP packages pass integrity checks; the DOCX contains all four author names and IDs, two A4 sections, two tables, and ten embedded figures. The XLSX contains eight worksheets. The PDF was generated locally from the DOCX content with a two-column renderer because LibreOffice was unavailable and Word export was interrupted by the desktop Escape key. All three rendered PDF pages were visually inspected. Native Microsoft Word pagination has not been visually verified and may differ from the supplied PDF.
