# Labeling guide

The labeling tool is a small page that runs on your own computer. You see a photo,
paint over the defect, pick a grade, and it saves everything.

## Start it

    pip install -r requirements-api.txt
    PYTHONPATH=src python scripts/label_tool.py --data-root data

On Windows PowerShell, set the path first:

    $env:PYTHONPATH="src"; python scripts/label_tool.py --data-root data

Your browser opens at http://127.0.0.1:8001. Only your own computer can reach
that address.

## Before you start

1. Copy your photos into `data/raw/`. Use .jpg, .jpeg or .png.
2. Each file name, without the extension, becomes the photo_id. Use clear names
   such as `tomato_01_day0.jpg`. Names must be unique.
3. iPhone photos in HEIC format are not supported. In the iPhone camera
   settings choose "Most Compatible" so photos are saved as JPG, or convert them first.

## For each photo

1. **Paint the defect.** Cover only the area that is actually bad: the rot spot,
   bruise, mold or blackened skin. Use Erase to fix mistakes.
2. **Pick the grade.** Keys 1 to 4.
   - A: no visible defect. Leave the mask empty.
   - B: small defect, still fine to sell.
   - C: clear defect, sell at a discount or use soon.
   - Reject: not fit to sell.
3. **Fill the details.** Produce name, light, phone model, item id, days left.
   The tool remembers produce, light, phone and your name for the next photo.
4. **Save and next.** Ctrl+Enter, or the green button.

Write the rules for B, C and reject once and stick to them. Consistent grading
matters more than perfect grading. If a second person labels a sample, compare
their grades with yours and note how often you disagree. The paper should report that.

## Item id and days left

If you photograph the same tomato every day, give every photo of it the same
item id, for example `tomato_01`. This keeps all photos of one item in the same
split, so the model is never tested on an item it already saw.

Days left is the number of days from this photo until the item became unfit.
Fill it in after the item has spoiled and you know the real number.
Leave it empty if you do not know.

## Keyboard shortcuts

| Key | Action |
|---|---|
| 1, 2, 3, 4 | Grade A, B, C, reject |
| E | Switch between paint and erase |
| [ and ] | Smaller and bigger brush |
| X | Clear the mask |
| Left and right arrows | Previous and next photo |
| Ctrl+Enter | Save and go to the next unlabeled photo |

## Where things are saved

- `data/labels.csv`: one row per photo
- `data/masks/<photo_id>.png`: white where the defect is, black elsewhere, the
  same size as the upright photo

You can stop and restart any time. Photos you already labeled open with their
saved values, and you can fix them and save again.

## Phone rotation

Phones often store photos sideways with a hidden rotation flag. FreshProof
applies that flag everywhere, so the mask always matches what you saw on screen.
