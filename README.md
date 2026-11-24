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

Early stage. Repo structure, label schema, and evidence metrics are in place.
Data collection comes next.

## Project layout

    docs/            data collection protocol and roadmap
    configs/         default settings
    data/            dataset layout (data itself is not in git)
    src/freshproof/  data schema and evaluation metrics
    tests/           unit tests

## Quick start

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    PYTHONPATH=src pytest

## Evidence metrics

For each image, with a saliency map and a ground truth defect mask:

- energy in mask: share of the highlight that falls on the real defect
- pointing game: whether the strongest point of the highlight is on the defect
- IoU: overlap between the thresholded highlight and the defect

## License

To be decided.
