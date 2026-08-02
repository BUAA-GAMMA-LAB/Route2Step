import os
import sys

# Avoid loading a local swift/ directory in place of the installed ms-swift package.
local_swift_path = os.path.abspath('swift')
sys.path = [p for p in sys.path if p != '' and os.path.abspath(p) != local_swift_path]

import random
import re
import PIL.Image
from torchvision import transforms
from swift.llm import Template

# Online image augmentation parameters.
MASK_PROB = 0.02
JITTER_BRIGHTNESS = 0.2
JITTER_CONTRAST = 0.2

print(
    f"Online data augmentation enabled: mask_prob={MASK_PROB}, "
    f"brightness={JITTER_BRIGHTNESS}, contrast={JITTER_CONTRAST}",
    flush=True,
)

# Create a black placeholder image for masked history frames.
black_img_path = os.path.abspath('data/black_mask_placeholder.jpg')
if not os.path.exists(black_img_path):
    os.makedirs('data', exist_ok=True)
    PIL.Image.new('RGB', (640, 480), color=0).save(black_img_path)

# Inject temporal masking when Template.encode processes a training example.
_original_encode = Template.encode

def patched_encode(self, inputs, *args, **kwargs):
    # The training pipeline normally passes a dictionary here. Keep other
    # input types unchanged.
    if isinstance(inputs, dict):
        messages = inputs.get('messages', [])
        images = inputs.get('images', [])
        
        if images and messages and isinstance(messages, list):
            # Collect the user prompt text.
            content = ""
            for msg in messages:
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    content += str(msg.get('content', ''))
            
            # Read the number of history frames from the prompt.
            match = re.search(r"The first (\d+) images are the History trajectory", content, re.IGNORECASE)
            
            if match:
                num_history = int(match.group(1))
                new_images = list(images)
                changed = False
                for i in range(min(num_history, len(new_images))):
                    if random.random() < MASK_PROB:
                        new_images[i] = black_img_path
                        changed = True
                if changed:
                    inputs['images'] = new_images
            
    return _original_encode(self, inputs, *args, **kwargs)

Template.encode = patched_encode

# Apply color perturbation when dataset images are opened.
_original_open = PIL.Image.open

def patched_open(fp, mode='r', formats=None):
    # Use the source path to decide whether augmentation applies.
    fp_name = ""
    if isinstance(fp, str):
        fp_name = fp
    elif hasattr(fp, 'name') and isinstance(fp.name, str):
        fp_name = fp.name

    if fp_name == black_img_path:
        return _original_open(fp, mode, formats)

    # Open the image once so file-like inputs are not consumed twice.
    img = _original_open(fp, mode, formats)
    
    # Apply perturbation only to images from the dataset directories.
    if fp_name and ('images/' in fp_name or 'StreamVLN-Trajectory-Data' in fp_name):
        try:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            jitter = transforms.ColorJitter(brightness=JITTER_BRIGHTNESS, contrast=JITTER_CONTRAST)
            return jitter(img)
        except Exception:
            # Fall back to the original image if augmentation fails.
            return img
            
    return img

PIL.Image.open = patched_open

if __name__ == '__main__':
    from swift.llm import sft_main
    sft_main()
