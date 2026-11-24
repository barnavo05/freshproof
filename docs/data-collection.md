# Data collection protocol

The model is only as good as the photos. This protocol keeps the dataset
consistent so results can be trusted and the dataset can be released later.

## Target for v1

- 2,000 to 5,000 photos
- 8 to 12 produce types that Indian shoppers buy most (start with tomato,
  potato, onion, banana, mango, apple, orange, capsicum)
- Every type covered across fresh, aging, and spoiled stages

## How to shoot

1. Use a normal phone camera, not a studio setup. Real conditions are the point.
2. Vary the light: daylight, shade, tube light, shop lighting, evening.
3. Vary the background: market stall, crate, kitchen counter, plastic bag.
4. Take 1 close photo per item and 1 photo of a small group per lot.
5. Never edit or filter photos. No portrait mode, no beauty mode.
6. Do not include faces or identifiable people in the frame.

## What to record for every photo

See `src/freshproof/data/schema.py`. In short:

- produce type
- grade (A, B, C, reject)
- days since purchase or harvest, if known
- days until visibly spoiled, recorded later by re-photographing the same item
  daily (this is the ground truth for shelf life)
- a defect mask: the pixels where the defect actually is

Use the labeling tool for all of this. See docs/labeling.md.

## Shelf life ground truth

Pick 20 to 30 items per type, photograph them daily until they spoil. This gives
real "days left" labels instead of guesses.

## Rules

- One person labels, a second person re-checks 10 percent. Track disagreement.
- Keep a holdout set that is never used for training.
- Record the phone model and light condition. We will need them to test
  robustness.
