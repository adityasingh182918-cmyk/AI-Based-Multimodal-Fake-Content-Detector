# Phase 3 models

The `image_copy/` and `text_copy/` directories are frozen copies of the
currently active Phase 1 and Phase 2 model artifacts. Phase 3 uses these copies
for feature extraction and independent outputs. New fusion checkpoints belong
in `fusion/` and must be created only by an explicit training command.

No fusion checkpoint is present until an eligible paired dataset is prepared
and training is explicitly run. Model artifacts are local and Git-ignored.
