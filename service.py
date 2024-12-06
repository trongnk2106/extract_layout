import io
import base64
import torch
from PIL import Image
import numpy as np
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM
import openai
from openai import OpenAI
import os 

yolo_model = YOLO('weights/icon_detect/best.pt').to('cuda')
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("weights/icon_caption_florence", 
                                             torch_dtype=torch.float16, 
                                             trust_remote_code=True).to('cuda')
caption_model_processor = {'processor': processor, 'model': model}

# Import utility functions
from utils import check_ocr_box, get_som_labeled_img

# FastAPI application
app = FastAPI(
    title="OmniParser GUI Screen Parser",
    description="A screen parsing tool to convert general GUI screens to structured elements",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for parsing
class ParseRequest(BaseModel):
    box_threshold: float = 0.05
    iou_threshold: float = 0.1

async def parse(image, box_threshold, iou_threshold, image_save_path):
    
    """
    Parse a screen image and return labeled image, parsed content, and coordinates
    
    - **file**: Image file to be parsed
    - **box_threshold**: Confidence threshold for bounding boxes (default: 0.05)
    - **iou_threshold**: Intersection over Union threshold (default: 0.1)
    """
    try:
        # Read image
        
        box_overlay_ratio = image.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        # OCR processing
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            image_save_path, 
            display_img=False, 
            output_bb_format='xyxy', 
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold':0.9}, 
            use_paddleocr=True
        )
        text, ocr_bbox = ocr_bbox_rslt

        # Process image with YOLO and caption model
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_save_path, 
            yolo_model, 
            BOX_TRESHOLD=box_threshold, 
            output_coord_in_ratio=True, 
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, 
            caption_model_processor=caption_model_processor, 
            ocr_text=text,
            iou_threshold=iou_threshold,
        )

        # Decode labeled image
        labeled_image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        
        # Convert labeled image to base64 for response
        buffered = io.BytesIO()
        labeled_image.save(buffered, format="PNG")
        labeled_image_base64 = base64.b64encode(buffered.getvalue()).decode()

        return labeled_image_base64, parsed_content_list, label_coordinates
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/parse_screen/")
async def process_screen(
    file: UploadFile = File(...), 
    box_threshold: float = 0.05, 
    iou_threshold: float = 0.1
):
    try: 
        contents = await file.read()
        image_save_path = 'imgs/saved_image_demo.png'
        
        with open(image_save_path, 'wb') as f:
            f.write(contents)
        
        image = Image.open(image_save_path)
        labeled_image_base64, parsed_content_list, label_coordinates = await parse(image, 
                                                                             box_threshold, 
                                                                             iou_threshold,
                                                                             image_save_path)
        
        return {
            "labeled_image": labeled_image_base64,
            "parsed_content": "\n".join(parsed_content_list),
            "label_coordinates": str(label_coordinates)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/parse_with_prompt")
async def process_screen_prompt(
    prompt: str = "",
    file: UploadFile = File(...),
    box_threshold: float = 0.05, 
    iou_threshold: float = 0.1
):

    try:
        contents = await file.read()
        image_save_path = 'imgs/saved_image_demo.png'
        
        with open(image_save_path, 'wb') as f:
            f.write(contents)
        
        image = Image.open(image_save_path)
        labeled_image_base64, parsed_content_list, label_coordinates = await parse(image, 
                                                                             box_threshold, 
                                                                             iou_threshold,
                                                                             image_save_path)
        icon_box = {}
        for idx, icon in enumerate(parsed_content_list):
            icon_box[icon] = label_coordinates[str(idx)]
        
        breakpoint()
        return {
            "labeled_image": labeled_image_base64,
            "parsed_content": "\n".join(parsed_content_list),
            "label_coordinates": str(label_coordinates)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add a simple health check endpoint
@app.get("/")
async def health_check():
    return {"status": "OmniParser is running", "version": "0.1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)