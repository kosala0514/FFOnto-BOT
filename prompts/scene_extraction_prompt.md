# Scene-graph extraction prompt (generated from ffonto-vocab.ttl)

Use this as the system / user prompt for a vision-language model (e.g. Qwen2-VL, LLaVA, GPT-4o, Gemini).
Ask for JSON only. Save each answer as `data/scenes/<image_id>.json`, then run `python scripts/lift.py`.

---

You are annotating ONE wildfire image for a knowledge graph. Describe only what is VISIBLE.
Return ONLY a JSON object with this structure:

```json
{
  "scene_id": "scene_<image_id>",
  "image_record_uuid": "<record_uuid of the image record>",
  "extractor": {"model_name": "<your model name>", "model_version": "<version>"},
  "human_validated": false,
  "entities": [
    {"id": "e1", "kind": "<KIND>", "count": 1,
     "state": ["<STATE>", ...], "activity": ["<ACTIVITY>", ...], "wears": ["<PPE>", ...],
     "position": "<POSITION>",
     "near": ["e2"], "advancing_toward": [], "reached": [], "protects": [], "blocks": []}
  ],
  "attributes": {"<PROPERTY>": "<VALUE>"}
}
```

Rules:
- Use ONLY the exact names listed below. If something does not fit, leave it out.
- One entity per distinct object or group; use "count" for groups (e.g. 3 firefighters).
- Relations ("near", "advancing_toward", "reached", "protects", "blocks") refer to entity ids.
- Positions are frame-relative to the fire.

## KIND (entity kinds)

- **FirePhenomenon**: FlameFront, SpotFire, CrownFire, SurfaceFire, Torching, Smouldering, Hotspot, FireCluster, SmokePlume, Pyrocumulus, Ember, PileBurn
- **Agent**: Person, Firefighter, Civilian, Responder, PoliceOfficer, EMSWorker, UtilityWorker, Official, Lookout, MediaObserver, Animal, Pet, Livestock, Wildlife
- **Equipment**: HandTool, Pulaski, McLeod, Shovel, Axe, FireBeater, Chainsaw, DripTorch, Fusee, HoseLine, Nozzle, PortablePump, PortableWaterTank, Sprinkler, HandheldRadio, MapBoard, Generator, PortableLighting, TrafficControlDevice, TrafficCone, CautionTape, Barricade, FuelCan, FireShelterPack
- **Vehicle**: GroundVehicle, FireEngine, BrushEngine, WaterTender, Dozer, Excavator, PoliceCar, Ambulance, Bus, PrivateCar, LivestockTrailer, Aircraft, Helicopter, AirTanker, Drone
- **BuiltAsset**: Building, House, Garage, Barn, Shed, CommandPost, Road, Bridge, PowerLine, PowerPole, Transformer, Hydrant, PropaneTank, Fence, Sign, CommunicationTower, SolarPanel, Culvert, Helispot
- **EnvironmentFeature**: Vegetation, Forest, ConiferousForest, DeciduousForest, MixedForest, Grassland, Shrubland, Tree, DownedLog, LadderFuel, LeafLitter, Terrain, Ridge, Slope, Valley, Canyon, Mountain, RockyTerrain, WaterBody, River, Lake, Pond, Stream, Sea, SnowCover, AgriculturalLand, UrbanArea, BareGround
- **BurnEvidence**: AshDeposit, CharredTrunk, ScorchedCrown, Snag, ExposedMineralSoil, ErosionRill, UnburnedPatch, Regrowth, RetardantLine, Foam, ControlLine, ErosionControlMeasure

## STATE

- **DamageState**: Intact, Scorched, Charred, Melted, Damaged, Destroyed, Collapsed
- **FireState**: OnFire, SmoulderingState, Threatened
- **AccessState**: Open, Blocked, Passable, Accessible
- **OperationalState**: Deployed, Staged, Charged, Flowing, LightsOn, Down, Leaning
- **VegetationState**: Green, Dry, Cured, Dead, Consumed, Patchy, Continuous, Dense, Sparse

## ACTIVITY

Evacuating, ShelteringInPlace, Gathering, AssistingVulnerablePerson, LoadingAnimals, DirectingTraffic, EstablishingPerimeter, Briefing, Communicating, AssistingCivilian, StandingBy, Retreating, TakingCover, Patrolling, ActingAsLookout, ApplyingWater, AdvancingHoseLine, BeatingOutFlames, SpotFireTriage, DiggingHandLine, CuttingControlLine, Burnout, RetardantLineConstruction, WaterDrop, RetardantDrop, Drafting, AerialReconnaissance, StructureProtection, MopUp, Pumping, FellingTrees

## PPE

Helmet, FireResistantClothing, Gloves, EyeProtection, Respirator, HighVisibilityVest, Headlamp, RadioChestHarness, HearingProtection, FireShelter

## POSITION

LeftFlank, RightFlank, HeadOfFire, AheadOfFire, OnFireEdge, InsideBlack, RidgeTop, ValleyBottom, BetweenFireAndAsset

## PROPERTY → allowed VALUE

- **SmokeColour**: White, LightGray, Gray, DarkGray, Black, Brown, Orange, Blue
- **AshColour**: White, LightGray, Gray, DarkGray, Black, Brown, Orange, Blue
- **SmokeDensity**: Low, Medium, High, Extreme
- **PlumeHeight**: Low, Medium, High, Extreme
- **VisibilityLevel**: Low, Medium, High, Extreme
- **AirQualityLevel**: Low, Medium, High, Extreme
- **FireIntensity**: Low, Medium, High, Extreme
- **BurnSeverity**: Low, Medium, High, Extreme
- **VegetationDensity**: Low, Medium, High, Extreme
- **PlumeDirection**: Vertical, Drifting
- **DominantAspect**: North, East, South, West
- **SkyCondition**: Clear, Cloudy, Hazy, Foggy, SunVisibleThroughSmoke
- **LightCondition**: Daytime, Night, Dusk
- **FireType**: SurfaceFireType, CrownFireType, StructureFireType, GrassFireType, PileFireType
- **FireSpreadStatus**: Expanding, Stable, Contained, Increasing, Decreasing
- **FireActivityTrend**: Expanding, Stable, Contained, Increasing, Decreasing
- **ScenePurpose**: Wildfire, PrescribedBurn
- **VegetationDistribution**: Uniform, PatchyDistribution, Fragmented
