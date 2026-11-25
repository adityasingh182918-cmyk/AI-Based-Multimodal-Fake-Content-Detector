# Phase 3 dataset license audit

Audit date: 2026-11-25. This is a screening record, not legal advice. A dataset
license on code does not automatically license its images, captions, or
upstream source material. Prior user preference is to avoid non-commercial or
uncertain terms when commercial use may occur.

| Candidate | Dataset terms visible in official source | Access conditions / source caveat | Decision |
|---|---|---|---|
| MiRAGeNews | No license declaration visible on the HF dataset page or linked repo README during this audit | 3.37 GB HF card; image-caption pairs aggregate news publisher and generated image rights | HOLD — obtain rights clarification |
| OpenFake | HF card says CC BY-NC 4.0 overall; generated subsets may have per-subset or per-row terms, including source-image inherited terms | Real photos retain rights of Flickr/LAION/Pexels/etc.; read row license fields before any subset use | EXCLUDE because non-commercial/rights uncertainty is incompatible with current preference |
| DGM4 | HF dataset card says Apache-2.0; official repository has a code license | Data include VisualNews news content and manipulated images; confirm underlying image/text redistribution rights separately | Not selected; task mismatch regardless of code license |
| MMFakeBench | HF data card says CC BY 4.0 but access agreement restricts to non-commercial research, restricts redistribution/copying/derived data, requires data-use acceptance and sharing HF user info | Gated: user must accept explicit protocol; 7.46 GB | EXCLUDE; explicit non-commercial terms conflict with current preference |
| Fakeddit | Repository directs users to read a separate Usage section and download via provided Drive files; no clearly applicable dataset license was confirmed from the inspected page | Reddit images and text remain third-party/user-posted material; public test release | HOLD/REJECT for this task; no rights approval, task labels are fake-news labels |
| M4FC | Repo states code Apache-2.0, dataset CC BY-SA 4.0 | Images sourced from fact-check articles/URLs; labels include manipulated-image and GPT-4-derived task fields | Not selected; task mismatch; retain source-image rights caveat |
| MS COCOAI / Defactify | HF card does not declare a license and directs formal questions to the dataset contact | 96,000 rows / 7.51 GB; built on COCO captions and images; official COCO terms license annotations CC BY 4.0 but leave image copyright to Flickr terms and each image source | HOLD — no download/use until written dataset permission and per-image rights are verified; no AI-text label regardless |
| GenImage | Homepage states CC BY-NC-SA 4.0 with additional terms | Over 1M image pairs, Baidu download, local 12-part incompleteness; source ImageNet rights also relevant | EXCLUDE from new Phase 3 use under current preference; existing Phase 1 remains frozen |
| VisualNews | No blanket dataset license found on inspected official repository/page | Publisher and Flickr image/caption rights; image archive 91 GB, articles 4.2 GB | HOLD; do not download absent rights review |
| RU-AI | GitHub repo is MIT-licensed; that is its code. Zenodo rights/license field was blank and HF displayed only a generic `cc` tag; no precise dataset license was verified | Dataset combines Flickr8K/COCO/Places205 with generated variants. Flickr8K sample archive is 1.22 GB images + ~1 MB text + 2.73 GB audio; audio not needed. Underlying rights differ by source. | HOLD — strongest data schema, but do not download/use while dataset and upstream rights are unresolved |

## Consequence

No candidate passed all required gates (usable access, clear intended-use
license, both modality-origin labels, meaningful pairs, and leakage-safe source
groups). MS COCOAI/Defactify is now recorded as an additional research lead,
not an approved source: its card requests direct licensing questions, its text
is human-only, and image rights remain per-image. No archive download, extraction, dataset modification, or training
was performed. The blocker is rights confirmation and label/pair validation,
not a missing fusion implementation. Ask the RU-AI authors or a rights-cleared
data provider to confirm terms covering the images, captions, generated
variants, model training, derived features/checkpoints, and intended
distribution/commercial use. If only an academic non-commercial license is
available, do not use it for this project without a revised user instruction.
