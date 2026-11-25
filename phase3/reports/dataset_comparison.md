# Phase 3 dataset comparison

**Audit date:** 2026-11-25. This is a documented-source review only. No dataset
was downloaded. Categories are evaluated for this project's two independent
binary labels (image origin and text authorship) plus authentic image-text
pairs, not for generic misinformation/fake-news classification.

| Dataset | Official source / size | Image and text availability | Image label | Text label | Pairing / metadata | Terms and access | Decision |
|---|---|---|---|---|---|---|---|
| MiRAGeNews | [HF card](https://huggingface.co/datasets/anson-huang/mirage-news); card says 15,000 records, 3.37 GB, 10k train, 2.5k validation, 5×500 test; paper says 12,500 image-caption pairs | Both; source page exposes image, label, caption | Binary `real`/`fake` multimodal news; not a proved independent image-origin label | Caption text available; no separate human/AI authorship label verified | News pairs with publisher/generator OOD partitions; card lists train/val NYT + Midjourney and held-out BBC/CNN with DALL-E 3/SDXL | No clear dataset license visible on card/repo page; not downloaded | **RESEARCH LEAD** — promising pairing, but label semantics and terms are unresolved |
| OpenFake | [HF card](https://huggingface.co/datasets/ComplexDataLab/OpenFake), [paper](https://arxiv.org/abs/2509.09495); core 2.45M rows, nearly 4M total images in paper; exact selected subset size depends on config | Images and prompt/caption fields; generator, type, release date, hashes and prompt metadata | Real vs synthetic image, with generator/source metadata in configurations | Prompts and generated descriptive captions exist, but not human-written vs AI-written labels for the paired text | Synthetic images use prompts related to source real images; text authorship origin does not supply the needed four-way set | Dataset card says CC BY-NC 4.0 overall; subsets may carry additional per-row terms and real-image rights. User previously indicated commercial/uncertain use, so NC data is excluded. No download. | **REJECTED** — licensing constraint and missing paired text-authorship labels; possible research-only image evaluation only if rights are acceptable |
| DGM4 | [HF dataset](https://huggingface.co/datasets/rshaojimmy/DGM4), [official code](https://github.com/rshaojimmy/MultiModal-DeepFake); viewer shows 208k train rows, total in 100k–1M band | Image paths and manipulated captions; boxes/token positions | Authentic/or manipulated media; examples include face swap/attribute manipulations | Manipulated text tokens and manipulation class, not AI authorship | Matched news ID; explicitly built for locating image/text manipulations | HF card labels Apache-2.0; source images may retain upstream source restrictions; no data downloaded | **REJECTED** — manipulation is not equivalent to AI generation/authorship |
| MMFakeBench | [official repo](https://github.com/liuxuannan/MMFakeBench), [HF access page](https://huggingface.co/datasets/liuxuannan/MMFakeBench); 7.46 GB | Paired image/text annotations; source fields, `gt_answers`, `fake_cls` | `image_source` may identify AI-generated images in some rows | `text_source` identifies source dataset and examples contain manipulated captions; it does not certify AI authorship | Mixed-source misinformation pairs and provenance-like source metadata | Access requires agreeing to a data usage protocol; HF states non-commercial research only and restrictions on copying/distribution/derived data. Excluded given user's licensing preference. | **REJECTED** — task labels measure misinformation, not independent text authorship; gated non-commercial terms |
| Fakeddit | [official repo](https://github.com/entitize/Fakeddit), [paper](https://arxiv.org/abs/1911.03854); over 1M records in paper | Text/metadata and downloadable image data; multimodal subset identified by authors | No real-vs-AI image-origin label | Fake-news category labels use distant supervision, not human-vs-AI text authorship | Reddit title/image pairs, but truthfulness categories and post relationships are not modality provenance | Repository requires consulting Usage instructions before download; no applicable license/use terms verified in this audit; no download | **REJECTED** — misinformation labels do not mean generated content; rights/terms unresolved |
| M4FC | [official repo](https://github.com/UKPLab/M4FC), [data README](https://github.com/UKPLab/M4FC/blob/main/data/README.md); 4,982 images paired with 6,980 claims | Images and claims, with task-specific fields | Authentic/manipulated fake image for fact-checking; not AI-generated-image provenance | Claims and GPT-4-derived annotations/captions are present, not an independent human/AI authorship class | Fact-checking article and image/claim relations, multiple task labels | Dataset CC BY-SA 4.0; image URLs are fetched from upstream and retain source concerns | **REJECTED** — fact-check verdict/manipulation labels are not image or text authorship labels |
| MS COCOAI / Defactify | [official HF card](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset), [paper](https://arxiv.org/abs/2601.00553); 96,000 rows, 7.51 GB; listed 42k/9k/45k train/validation/test | Image plus original COCO caption | Explicit real-photo vs AI-image labels; five named image generators | Captions are human-written COCO captions in both image classes; no AI-authored text labels | Synthetic images are generated from source human captions; card lists caption/image/image-label fields but no stable source-group ID for safe split reconstruction | HF card directs licensing questions to the authors; no dataset license is declared. COCO annotations are CC BY 4.0, while image rights remain governed by per-image Flickr terms. Not downloaded. | **RESEARCH LEAD** — supplies real/AI image × human text examples, but lacks AI-text labels and clear dataset/image rights; group-safe splitting needs verification |
| GenImage | [official repo](https://github.com/GenImage-Dataset/GenImage), [homepage](https://genimage-dataset.github.io/); over 1M real/fake pairs | Images only | `ai` vs `nature` (ImageNet-collected real images) and known generator subsets | No paired text | Generator and source partitions; current project archive is incomplete | Homepage states CC BY-NC-SA 4.0 plus additional terms; archive also has 12 missing parts. No download. | **REJECTED** — no text modality, licensing constraint, and incomplete local archive; it remains Phase 1-only |
| VisualNews | [official repo](https://github.com/FuxiaoLiu/VisualNews-Repository), [publisher page](https://www.cs.rice.edu/~vo9/visualnews/); over 1M news images and article/caption metadata; Rice page lists 4.2 GB articles and 91 GB image archive | Human-produced news text/captions and images | No AI-origin labels | Human/source-authored text is not an AI-authorship comparison set | Strong news pair/source metadata | No blanket dataset license identified; underlying publishers and image rights apply | **RESEARCH LEAD** — could contribute a documented REAL_IMAGE + HUMAN_TEXT slice only after rights review, but cannot supply remaining classes |
| RU-AI | [official repo](https://github.com/ZhihaoZhang97/RU-AI), [Zenodo](https://zenodo.org/records/11406538), [HF](https://huggingface.co/datasets/zzha6204/RU-AI-origin); 1.475M instances; Zenodo files total 157.5 GB and full extraction reported up to 500 GB | Aligned image/text/voice variants; Flickr8K subset download shown as 1.22 GB images + ~1 MB text (plus 2.73 GB audio, unnecessary to our modality pair) | Human originals and AI image variants with generator metadata | Original human captions and AI-generated caption variants (GPT/PaLM/Llama/Mixtral/RWKV) | Same source image IDs align originals and generated modality variants, which could support four source-ID-preserving combinations after per-record review | GitHub MIT license is for code; Zenodo rights field is blank and HF's license tag is only `cc`. Underlying Flickr8K/COCO/Places image/text rights are heterogeneous and not resolved for this project. No download. | **RESEARCH LEAD** — strongest structural match discovered, but cannot use while rights are unclear |

## Selection

**No dataset is approved for training or independent evaluation at this time.**
RU-AI remains the strongest structural candidate because its example notebook
shows same-image-ID human captions, generated caption variants, real images,
and generated images. MS COCOAI/Defactify is another useful alignment lead for
real and generated images paired with human-written COCO captions, but it has
no AI-authored text class and its dataset license is not stated on the card.
COCO image rights also remain per-image Flickr terms, and the public schema
shown does not establish source-group IDs for leakage-safe splitting.

The RU-AI dataset and upstream rights remain unresolved. MS COCOAI's HF card
says to contact the authors for license and permitted-use terms; no such
written permission is present in this workspace. Both remain research leads,
not approved data. No archive was downloaded, and no training or feature
extraction was run.

MiRAGeNews remains an alternate pairing lead, but its real/fake label does not
independently establish image origin and text authorship, and its visible
source pages do not establish a usable dataset license. The other screened
sources primarily measure fact-checking, misinformation, or manipulation,
not the two independent origin labels required here.

## Smallest additional data requirement

To unblock a trustworthy first run, obtain one written-cleared aligned source
with for each source-group ID: a human-authored caption, its documented AI
rewrites/variants (or AI-authored captions), the corresponding real image, and
an AI-generated image variant made from the same source description. Retain
the source ID, generation method, image/text origin evidence, license reference,
and grouping ID. This can produce four real combinations without pairing
unrelated examples. Each group must remain within a single train/validation/
test partition. Alternatively, obtain written confirmation for the RU-AI Flickr8K subset, or
MS COCOAI plus per-image COCO/Flickr rights and source-group metadata, under
the intended project terms.
