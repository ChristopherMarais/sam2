from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import torch
import uvicorn
import json

# Import the SAM2ImagePredictor from the SAM 2 repo.
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = FastAPI()

# Initialize the SAM2 predictor.
# Option 1: Load from Hugging Face (if you have installed the SAM2 package)
# predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")
# Option 2: Load using a local checkpoint and config
# from sam2.build_sam import build_sam2
checkpoint_path = "../checkpoints/sam2.1_hiera_large.pt"
config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(config_path, checkpoint_path))

@app.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    points: str = Form(None),   # JSON-encoded list of click coordinates, e.g. "[[100,150],[200,250]]"
    labels: str = Form(None)    # JSON-encoded list of labels (1 for positive, 0 for negative), e.g. "[1,1]"
):
    # Read and decode the image.
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Parse the interactive prompt data.
    try:
        if points:
            point_coords = np.array(json.loads(points), dtype=np.float32)
        else:
            point_coords = None

        if labels:
            point_labels = np.array(json.loads(labels), dtype=np.int32)
        else:
            point_labels = None
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid prompt format. Ensure points and labels are valid JSON.")

    # Run SAM 2 inference with the provided prompts.
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image)
            if point_coords is not None and point_labels is not None:
                # This call uses the click prompts to focus segmentation on selected objects.
                masks, scores, logits = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True  # Returns multiple mask options per prompt.
                )
            else:
                # Fallback to automatic mask generation if no prompts are provided.
                masks, scores, logits = predictor.predict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

    # Convert the masks and scores to a JSON-serializable format.
    mask_list = [mask.tolist() for mask in masks] if isinstance(masks, list) else masks.tolist()
    score_list = scores.tolist() if hasattr(scores, "tolist") else scores

    return JSONResponse(content={"masks": mask_list, "scores": score_list})

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)
