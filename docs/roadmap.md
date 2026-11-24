# Roadmap

## Weeks 1 to 2: data and gap check
- Read prior work on explanation reliability for produce quality
- Start shooting the dataset following docs/data-collection.md
- Set up labeling and a second-person check

## Weeks 3 to 5: model and evidence
- Baseline classifier for produce type and grade
- Shelf life regression from daily re-photography
- Compare saliency methods using the evidence metrics in
  src/freshproof/evaluation
- Calibrated uncertainty and a "check by hand" flag

## Weeks 4 to 6: app
- Backend with FastAPI
- Phone web app: scan, grade, proof, days left
- Works on cheap phones and slow networks

## Weeks 6 to 8: pilots and paper
- Free pilot with 2 to 3 local sellers
- Measure waste and disputes avoided
- Write the paper and post it on arXiv
- Choose a venue with travel support or online presentation

## Later
- Batch scanning, supplier scorecards, dispute reports
- Wrap the web app for the Play Store, then the App Store
