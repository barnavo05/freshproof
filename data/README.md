# Data folder

Raw photos and processed data are not stored in git. They stay on your machine
or in cloud storage.

Expected layout:

    data/
      raw/          original photos exactly as shot, named <photo_id>.jpg
      masks/        defect masks, one per photo that has a visible defect
      labels.csv    one row per photo
      processed/    resized and split data created by scripts

## labels.csv columns

Required: `photo_id, produce_type, grade, light, phone_model`

Optional: `item_id, defect_mask_path, days_until_spoiled, labeler`

- `photo_id` must match the file name in raw/ without the extension (.jpg,
  .jpeg or .png). Every photo_id must be unique.
- `grade` is one of A, B, C, reject.
- `light` is one of daylight, shade, tube_light, shop_light, evening, other.
- `item_id` names the physical item. If you photograph the same tomato every
  day, give all those photos the same item_id. This keeps them in one split.
  Leave it blank for single photos.
- A grade A photo must not have a defect mask.

Example row:

    photo_id,produce_type,grade,light,phone_model,item_id,defect_mask_path,days_until_spoiled,labeler
    tom_001,tomato,B,daylight,pixel7,tom_item_01,masks/tom_001.png,3,barnavo

See docs/data-collection.md for how to shoot and label.
