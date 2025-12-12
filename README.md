# AI-Powered-image-inpainting
This project implements an end-to-end image inpainting and restoration pipeline using state-of-the-art deep learning models. It reconstructs missing or corrupted regions in images using a combination of automatic mask generation and diffusion-based inpainting. The system ensures that the restored content blends naturally with the original image.

The workflow integrates Segment Anything Model (SAM) for object/region segmentation and Stable Diffusion Inpainting for high-quality restoration.

## Features

- Automatic Mask Generation using META’s SAM model

- High-quality Image Inpainting with Stable Diffusion

- Manual Masking support for custom edits

- Real-time visualization of input, mask, and output

- Handles complex textures, backgrounds, and fine details

- Works entirely on Google Colab, no local setup required

## Workflow Overview
1. **Upload an input image** to the pipeline
2. **SAM generates segmentation masks** automatically
3. User selects a mask **or uploads a custom mask**
4. Mask + image are passed to **Stable Diffusion Inpainting**
5. The model predicts a **cLean and realistic reconstruction**
6. Output is displayed and saved

## Usage Instructions (Google Colab)
### 1. Install dependencies 
```python
!pip install 'git+https://github.com/facebookresearch/segment-anything.git'
!pip install diffusers accelerate transformers
!pip install opencv-python pillow
```

### 2. Load SAM model
```python
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)
```

### 3. Generate Mask
```python
predictor.set_image(input_image)
masks, _, _ = predictor.predict(point_coords=None, point_labels=None)
```

### 4. Run Stable Diffusion Inpainting
```python
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
result = pipe(prompt="restore damaged area", image=input_image, mask_image=mask)
result.images[0]
```

## Applications

- Old photo restoration

- Removing unwanted objects

- Filling missing/corrupted regions

- Background reconstruction

- Image editing pipelines

- Preprocessing for computer vision datasets

## Future Improvements
- Fine-tuning diffusion model for domain-specific restoration

- Add Gradio / Streamlit UI

- Support for video inpainting

- Improve mask-selection UX

- Deploy as an API
