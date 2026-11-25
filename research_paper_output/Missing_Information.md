# Missing information and confirmations

- Author emails were not provided.
- No independent high-resolution camera-photo or unseen-image-generator test corpus; image test has 200 32×32 CIFAKE items. Four user-labeled REAL images are in later fine-tune training and provenance was not authenticated.
- MAGE original document IDs are unavailable; exact duplicate removal cannot exclude semantic/source overlap.
- No independently labeled, rights-cleared human grammar-correction benchmark. Beemo is positive-only, so no two-class accuracy, precision, macro-F1, or human false-positive rate.
- No paired multimodal dataset, trained fusion model, or joint metric.
- Image softmax is uncalibrated. Expanded DistilBERT temperature scaling was validation-fitted, but OOD calibration is weak. Expanded abstention limits [0,1] yield zero confident coverage.
- Repeat-seed variation and complete historical optimizer/scheduler details are not consistently recorded. Runtime environment overrides may differ from code defaults.
- The final DOCX was not visually rendered inside Microsoft Word: LibreOffice was unavailable, and the Word UI export attempt was interrupted by the desktop Escape key. The supplied PDF was generated from the DOCX content using a local two-column renderer and visually reviewed; Word-specific pagination can differ.
