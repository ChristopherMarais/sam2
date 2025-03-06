from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import torch
import uvicorn
import json

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = FastAPI()

# Initialize the SAM2 predictor using a local checkpoint and config.
checkpoint_path = "../checkpoints/sam2.1_hiera_large.pt"
config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(config_path, checkpoint_path))

@app.post("/segment_box")
async def segment_image_box(
    file: UploadFile = File(...),
    box: str = Form(...)  # JSON-encoded bounding box, e.g. "[xmin, ymin, xmax, ymax]"
):
    # Read and decode the image.
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Parse the bounding box.
    try:
        # Expects a JSON list like: [xmin, ymin, xmax, ymax]
        box_coords = np.array(json.loads(box), dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid box format. Ensure it's valid JSON.")

    # Run SAM 2 inference with the bounding box prompt.
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image)
            # Pass the box prompt to the predictor.
            masks, scores, logits = predictor.predict(box=box_coords)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

    # Convert results to JSON-serializable format.
    mask_list = [mask.tolist() for mask in masks] if isinstance(masks, list) else masks.tolist()
    score_list = scores.tolist() if hasattr(scores, "tolist") else scores

    return JSONResponse(content={"masks": mask_list, "scores": score_list})

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)
