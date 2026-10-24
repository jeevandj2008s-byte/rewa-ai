# Model Card: YOLOv8n Waste Detection (12-Class)

## Model Overview
- **Model Name:** `kendrickfff/waste-classification-yolov8-ken`
- **Weights File:** `yolov8n-waste-12cls-best.pt`
- **Architecture:** Ultralytics YOLOv8n (Nano detection backbone and head)
- **Parameters:** ~3.0 Million
- **File Size:** ~51 MB (checkpoint with optimizer states)
- **License:** Creative Commons Attribution 4.0 International (CC-BY-4.0)
- **Hugging Face Repository:** [kendrickfff/waste-classification-yolov8-ken](https://huggingface.co/kendrickfff/waste-classification-yolov8-ken)

## Training Details
- **Training Environment:** NVIDIA A100 GPU
- **Framework:** PyTorch & Ultralytics YOLOv8
- **Dataset:** Multi-class municipal solid waste, recycling, and litter imagery
- **Epochs:** 30 epochs
- **Input Resolution:** 640x640

## Classes (12 Detection Categories)
1. `battery` (Household batteries, e-waste)
2. `biological` (Organic food waste, peels, kitchen scraps)
3. `brown-glass` (Amber/brown beverage bottles, glass medicine bottles)
4. `cardboard` (Corrugated boxes, cardboard packaging)
5. `clothes` (Textiles, fabrics, apparel)
6. `green-glass` (Green glass bottles, olive oil bottles)
7. `metal` (Aluminium cans, tin cans, foil trays)
8. `paper` (Paper sheets, office paper, newspapers)
9. `plastic` (PET bottles, HDPE containers, plastic wrappers/pouches)
10. `shoes` (Footwear, sneakers, slippers)
11. `trash` (General reject refuse, sanitary waste)
12. `white-glass` (Clear/transparent glass jars, beverage bottles)

## Integration in ReWA.AI
In ReWA.AI, this model forms Stage 1 of the multi-object waste sorting pipeline:
1. **Stage 1 (YOLO):** Detects multiple items in an image (e.g. plastic bottle + pouch side-by-side) with bounding boxes, spatial coordinates, and initial waste class.
2. **Stage 2 (CLIP Refinement):** Crops each detected item and uses ReWA.AI's existing zero-shot CLIP classifier with targeted candidate labels (from `data/yolo_class_map.json`) to distinguish fine-grained categories (e.g., rigid bottle vs soft wrapper).
3. **Bin Assignment:** Maps each item to canonical digital twin bins (`Plastic`, `Paper`, `Metal`, `Glass`, `Hazardous`, `Biodegradable`).

## Limitations
- Performance on extremely dark or heavily blurry camera shots may produce false negatives.
- Nested or tightly crushed objects (e.g., wrapper inside a crushed can) may be detected as a single item; users are advised to separate items.
- Model scores are similarity/detection scores, never guarantees of absolute classification accuracy.

