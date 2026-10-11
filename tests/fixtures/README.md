# Test fixtures

Fixed test data for `pytest tests/`. Do not put real data here.

- `raw/image_evoDJI_0001_frame12.json`, `raw/audio_videoplayback_14.json` – two real records
- `raw/fixture_*`, `raw/weather.csv` – invented test records (event `ev_demo_081`)
- `scenes/*.json` – hand-written examples of VLM scene-graph output

The expected answers in `tests/cq_expectations.py` are written for exactly this data.
Your real data lives in `data/raw` and `data/scenes` and is evaluated separately.
