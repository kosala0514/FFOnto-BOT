"""
Expected answers for the CQ test suite (tests/test_cq_suite.py).

Each CQ maps to a dict:
    "min_rows":  minimum number of result rows            (default 1)
    "rows":      list of row patterns that MUST appear in the result
    "absent":    list of row patterns that must NOT appear

A row pattern is {column: expected}. Expected values are compared with the
displayed value (as in results/report.md):
    "ffontov:Smoke"   exact match
    "~3f6d562f"   '~' = substring match
    "True"/"False" for booleans

The values below are for the sample/fixture data shipped in data/raw and data/scenes.
When you add real data, add expectations taken from the ForestFire-VQA ground-truth
answers – that turns the dataset itself into a regression test for the ontology.
"""

REAL_IMAGE = "~3f6d562f-33d3-49ee-a40b-3a8b6cf07823"
REAL_AUDIO = "~7cd6773c-c37e-4acc-a594-20a8d0a5c74a"
FIX_IMAGE = "~00000000-0000-4000-8000-000000000081"

EXPECT = {
    "CQ01": {"rows": [{"eventCode": "ev_demo_081", "nRecords": "3", "nImages": "1", "nAudio": "1", "nWeatherRows": "1"}]},
    "CQ02": {"rows": [{"record": REAL_IMAGE, "missingFields": "~capture_time"},
                      {"record": REAL_IMAGE, "missingFields": "~crs"},
                      {"record": REAL_IMAGE, "checksum": "~df5124eb", "width": "2048", "height": "1080"}]},
    "CQ03": {"rows": [{"record": REAL_IMAGE, "output": "caption", "modelName": "Salesforce/blip-image-captioning-large"},
                      {"record": REAL_AUDIO, "modelName": "google/yamnet", "modelVersion": "1"}]},
    "CQ04": {"rows": [{"record": REAL_IMAGE, "class": "ffontov:NoFire"},
                      {"record": REAL_IMAGE, "class": "ffontov:Fire", "severity": "ffontov:Low"}]},
    "CQ05": {"rows": [{"record": REAL_IMAGE, "class": "ffontov:Smoke", "aboveThreshold": "True", "systemLabel": "Smoke"},
                      {"record": REAL_IMAGE, "class": "ffontov:Fire", "aboveThreshold": "False"},
                      {"record": REAL_AUDIO, "class": "ffontov:Explosion", "aboveThreshold": "True"}]},
    "CQ06": {"rows": [{"record": REAL_IMAGE, "mentionsFire": "True", "mentionsWater": "True", "mentionsSmoke": "False"}]},
    "CQ07": {"rows": [{"conflictType": "~Caption mentions", "record": REAL_IMAGE},
                      {"conflictType": "~Audio indicates", "eventCode": "ev_demo_081"}]},
    "CQ07b": {"rows": [{"eventCode": "ev_demo_081", "modality": "AudioRecord", "hasTimestamp": "True"},
                       {"record": REAL_IMAGE, "hasTimestamp": "False"}]},
    "CQ08": {"rows": [{"site": "~demo_site_A", "date": "~2012-08-28", "rhPct": "43"}]},
    "CQ09": {"rows": [{"date": "~2012-08-28", "fwiTrend": "increasing", "fireOccurred": "True"}]},
    "CQ10": {"rows": [{"date": "~2012-08-28", "condition": "Elevated fire danger"},
                      {"date": "~2012-08-28", "condition": "~Rapid fine-fuel spread"}],
             "absent": [{"condition": "Aerial drop effectiveness reduced"}]},
    "CQ11": {"rows": [{"record": FIX_IMAGE, "phenomenon": "Flame front", "count": "2"}]},
    "CQ12": {"rows": [{"record": REAL_IMAGE, "property": "Smoke colour", "value": "Gray"}]},
    "CQ13": {"rows": [{"record": FIX_IMAGE, "property": "Fire intensity", "value": "High"}]},
    "CQ14": {"rows": [{"record": REAL_IMAGE, "kind": "Lake"}, {"record": FIX_IMAGE, "kind": "Coniferous forest", "states": "Dry"}]},
    "CQ15": {"rows": [{"record": FIX_IMAGE, "property": "Visibility level", "value": "Low"}]},
    "CQ16": {"rows": [{"record": FIX_IMAGE, "kind": "Firefighter", "activities": "~Applying water"}]},
    "CQ17": {"rows": [{"record": FIX_IMAGE, "missingRequiredPPE": "~Gloves"}]},
    "CQ18": {"rows": [{"record": FIX_IMAGE, "kind": "Fire engine", "count": "1", "states": "Emergency lights on"}]},
    "CQ19": {"rows": [{"record": FIX_IMAGE, "activity": "~Applying water", "tactic": "Direct attack"}]},
    "CQ20": {"rows": [{"record": FIX_IMAGE, "kind": "House", "states": "~Scorched"}]},
    "CQ21": {"rows": [{"subject": "Flame front", "relation": "advancing toward", "object": "House"}]},
    "CQ22": {"rows": [{"record": FIX_IMAGE, "kind": "Foam"}]},
    "CQ23": {"rows": [{"record": REAL_AUDIO, "soundEvent": "Explosion", "aboveThreshold": "True", "longerThan60s": "False"}]},
    "CQ24": {"rows": [{"hazardType": "Missing PPE", "eventCode": "ev_demo_081"},
                      {"hazardType": "Explosion hazard", "eventCode": "ev_demo_081"},
                      {"hazardType": "Flame proximity to people", "threatened": "Firefighter"}]},
    "CQ25": {"rows": [{"hazardType": "Explosion hazard", "recommendedActions": "~Cool and isolate pressurised tank"}]},
    "CQ26": {"rows": [{"ecosystem": "Coniferous forest", "typicalImpacts": "~Tree mortality"},
                      {"ecosystem": "Lake", "typicalImpacts": "~Water quality degradation"}]},
}
