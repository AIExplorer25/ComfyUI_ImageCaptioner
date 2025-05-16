import torch
import os
from PIL import Image
import os
import time
from transformers import pipeline

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor
)

class ImageCaptioner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rename_images_sequencially": ("BOOLEAN", {
                    "default": True, "label_on": "Yes", "label_off": "No"
                }),
                 "image_folder_path": ("STRING", {"multiline": True, "default": ""}),
                "model_id": ("STRING", {"multiline": True, "default": ""}),
                "system_message": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("updated_status",)
    FUNCTION = "generate_captions"
    OUTPUT_NODE = True
    CATEGORY = "utils"

    # Class variable to store the last inputs and outputs
    _cache = {}
    
    def generate_captions(self,rename_images_sequencially,image_folder_path, model_id, system_message,):
        # Handle case where inputs are not lists
        print("inside generate_captions function...")
        if rename_images_sequencially:
            # Rename images in the folder sequentially
            for i, image_name in enumerate(os.listdir(image_folder_path)):
                if image_name.endswith(".jpg") or image_name.endswith(".png"):
                    new_image_name = f"{i+1}.jpg"
                    os.rename(os.path.join(image_folder_path, image_name), os.path.join(image_folder_path, new_image_name))
                    print(f"Renamed {image_name} to {new_image_name}")
        else:
            # Rename images in the folder sequentially
            for i, image_name in enumerate(os.listdir(image_folder_path)):
                if image_name.endswith(".jpg") or image_name.endswith(".png"):
                    new_image_name = f"{i+1}.jpg"
                    os.rename(os.path.join(image_folder_path, image_name), os.path.join(image_folder_path, new_image_name))
                    print(f"Renamed {image_name} to {new_image_name}")
        # Load the model and processor
        model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", # Use "flash_attention_2" when running on Ampere or newer GPU or use "eager" for older GPUs
        )
        min_pixels = 512*28*28
        max_pixels = 1600*28*28

        processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels, min_pixels=min_pixels)
        status=[]
        system_message = "You are an expert image describer."
        for image in os.listdir(image_folder_path):
            if image.endswith(".jpg") or image.endswith(".png"):
                image_path = os.path.join(image_folder_path, image)
                image = Image.open(image_path).convert("RGB")
                messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {"type": "image", "image": image},
                    ],
                },
                ]
                text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                text=[text],
                images=image,
                padding=True,
                return_tensors="pt",
                )
                inputs = inputs.to(model.device)
                generated_ids = model.generate(**inputs, max_new_tokens=512, min_p=0.1, do_sample=True, temperature=1.5)
                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                caption = output_text[0]
                print(f"Caption for {image}: {caption}")
                # Save the caption to a file
                caption_file_path = os.path.join(image_folder_path, f"{os.path.splitext(image)[0]}.txt")
                with open(caption_file_path, "w") as caption_file:
                    caption_file.write(caption)
                status.append(f"Caption for {image}: {caption}")




        
        return ", ".join(status),

   

NODE_CLASS_MAPPINGS = {
    "ImageCaptioner": ImageCaptioner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCaptioner": "Image Captioner",
}
