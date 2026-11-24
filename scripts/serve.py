"""Run the API locally with the demo stand-in model.

    pip install -r requirements-api.txt
    PYTHONPATH=src python scripts/serve.py

Then open http://127.0.0.1:8000/docs to try /scan in the browser.
"""

import uvicorn

from freshproof.service.api import create_app
from freshproof.service.predictor import StubPredictor

app = create_app(StubPredictor(), abstain_threshold=0.6)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
