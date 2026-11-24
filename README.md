# FreshProof

Produce quality grading that shows its evidence.

Take a photo of a fruit or vegetable. FreshProof returns a grade, an estimate of
the days of shelf life left, and a highlight of the exact defect it based the
grade on. When the model is unsure, it says so and asks for a manual check.

## Why

Most quality tools give a grade and ask you to trust it. A grade is hard to use
in a dispute with a supplier or a customer. A grade with verified visual proof
is much easier to act on.

The research question behind the project: does the highlighted region actually
match the real defect? FreshProof measures this instead of assuming it.

## Status

Early stage.

- Done: label schema and loader, item-based data splits, evidence metrics,
  baseline model (ResNet18 with produce type and grade heads), training script,
  calibration and "check by hand" threshold, synthetic data generator, scan API
  with a stand-in model
- Next: collect the first real photos, train the baseline, add shelf life
  regression, plug the trained model into the API

## Project layout

    docs/            data collection protocol and roadmap
    configs/         default settings
    data/            dataset layout (the data itself is not in git)
    scripts/         training script and local API server
    src/freshproof/
      data/          label schema, loader, item-based splits, torch dataset
      models/        baseline classifier
      evaluation/    evidence metrics (does the highlight match the defect)
      uncertainty/   calibration, temperature scaling, abstain threshold
      labeling/      local tool for painting defects and saving labels
      service/       scan API, predictor interface, evidence overlay
      synthetic.py   fake produce photos for testing the pipeline
    tests/           unit tests

## Label your photos

A small local tool lets you paint over defects and save grades without
touching the CSV by hand. See docs/labeling.md.

    pip install -r requirements-api.txt
    PYTHONPATH=src python scripts/label_tool.py --data-root data

## Try it without real photos

The synthetic generator draws fake produce with known defects. It is only for
checking that the pipeline runs. Never report its results as real results.

    PYTHONPATH=src python -m freshproof.synthetic --out data_synth --items 8 --days 6

To try the scan API with the demo stand-in model:

    pip install -r requirements-api.txt
    PYTHONPATH=src python scripts/serve.py

Then open http://127.0.0.1:8000/docs and upload a photo to /scan. The response
has a grade, a confidence, a `needs_manual_check` flag, and a PNG overlay that
shows the evidence.

## Quick start

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    PYTHONPATH=src pytest

To train, install the extra requirements and point the script at your data:

    pip install -r requirements-train.txt
    PYTHONPATH=src python scripts/train.py --data-root data --epochs 10 --pretrained

Splits are made by item, not by photo, so daily photos of the same tomato never
end up on both sides of the split. The holdout set is only scored when you pass
`--eval-holdout`, and that should happen once, at the very end.

## Evidence metrics

For each image, with a saliency map and a ground truth defect mask:

- energy in mask: share of the highlight that falls on the real defect
- pointing game: whether the strongest point of the highlight is on the defect
- IoU: overlap between the thresholded highlight and the defect

## License

To be decided.
