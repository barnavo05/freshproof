# Data folder

Raw photos and processed data are not stored in git. They stay on your machine
or in cloud storage.

Expected layout:

    data/
      raw/          original photos exactly as shot
      masks/        defect masks, one per photo that has a visible defect
      labels.csv    one row per photo, following src/freshproof/data/schema.py
      processed/    resized and split data created by scripts

See docs/data-collection.md for how to shoot and label.
