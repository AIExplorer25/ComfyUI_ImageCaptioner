import torch
import os
from PIL import Image
import os
import time
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
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


    
    def generate_captions(self,rename_images_sequencially,image_folder_path, model_id, system_message,):
        # Handle case where inputs are not lists
        print("inside generate_captions function...")
        if rename_images_sequencially:
            # Rename images in the folder sequentially
            print("inside rename image if ...")
            for i, image_name in enumerate(os.listdir(image_folder_path)):
                if image_name.endswith(".jpg") or image_name.endswith(".png") or image_name.endswith(".jpeg"):
                    new_image_name = f"{i+1}.jpg"
                    os.rename(os.path.join(image_folder_path, image_name), os.path.join(image_folder_path, new_image_name))
                    print(f"Renamed {image_name} to {new_image_name}")
        else:
            # Rename images in the folder sequentially
            print("inside rename image else ...")
            for i, image_name in enumerate(os.listdir(image_folder_path)):
                if image_name.endswith(".jpg") or image_name.endswith(".png") or image_name.endswith(".jpeg"):
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
            if image.endswith(".jpg") or image.endswith(".png") or image_name.endswith(".jpeg"):
                image_path = os.path.join(image_folder_path, image)
                image_name = os.path.basename(image_path)
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
                print(f"Caption for {image_name}: {caption}")
                # Save the caption to a file
                caption_file_path = os.path.join(image_folder_path, f"{os.path.splitext(image_name)[0]}.txt")
                with open(caption_file_path, "w") as caption_file:
                    caption_file.write(caption)
                status.append(f"Caption for {image_name}: {caption}")




        
        return ", ".join(status),



class ImageCaptionerPostProcessing:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                 "image_folder_path": ("STRING", {"multiline": True, "default": ""}),
                "trigger_word": ("STRING", {"multiline": True, "default": ""}),
                "text_replace": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("updated_status",)
    FUNCTION = "postprocessing_captions"
    OUTPUT_NODE = True
    CATEGORY = "utils"


    
    def postprocessing_captions(self,image_folder_path, trigger_word, text_replace,):
        # Handle case where inputs are not lists
        print("inside generate_captions function...")
            # Rename images in the folder sequentially
        print("inside rename image if ...")
        status=[]
        for i, text_name in enumerate(os.listdir(image_folder_path)):
            if text_name.endswith(".txt") or text_name.endswith(".text"):
                    caption_file_path = os.path.join(image_folder_path, text_name)
                    with open(caption_file_path, "r") as caption_file:
                        captiontext = caption_file.read()
                    # Replace the trigger word with the new text
                    text_replace_splitted=text_replace.split(",")
                    for text2replace in text_replace_splitted:
                        if captiontext.find(text2replace) != -1:
                            captiontext = captiontext.replace(text2replace, "")
                    captiontextFinal = captiontext +" , "+ trigger_word
                    
                    # Save the modified caption back to the file
                    with open(caption_file_path, "w") as caption_file:
                        caption_file.write(captiontextFinal)
                    print(f"Updated caption in {text_name} to: {captiontextFinal}")
                    status.append(f"Updated caption in {text_name} to: {captiontextFinal}")
            else:
                print(f"No caption found in {text_name} to update.")

        
        return ", ".join(status),        


   

class CheckImageCaptionsData:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                 "image_folder_path": ("STRING", {"multiline": True, "default": ""}),
                 "display_first_nchars": ("INT", { "default": "20", "min": 0, "max": 100}),
                 
            }
        }

    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("updated_status",)
    FUNCTION = "postprocessing_checkcaptionsdata"
    OUTPUT_NODE = True
    CATEGORY = "utils"


    
    def postprocessing_checkcaptionsdata(self,image_folder_path,display_first_nchars,):
        # Handle case where inputs are not lists
        print("inside generate_captions function...")
            # Rename images in the folder sequentially
        print("inside rename image if ...")
        status=[]
        for i, text_name in enumerate(os.listdir(image_folder_path)):
            if text_name.endswith(".txt") or text_name.endswith(".text"):
                    caption_file_path = os.path.join(image_folder_path, text_name)
                    with open(caption_file_path, "r") as caption_file:
                        captiontext = caption_file.read()
                    first_n_chars = captiontext[:display_first_nchars]
                    status.append(first_n_chars)

            else:
                print(f"No caption found in {text_name} to update.")

        
        return "-------/r/n ".join(status),    

class Quen3Helper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "your_prompt": ("STRING", {"multiline": True, "default": ""}),
                "model_id": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("quen3_response",)
    FUNCTION = "generate_response"
    OUTPUT_NODE = True
    CATEGORY = "utils"
    def generate_response(self,your_prompt, model_id,):
        # Load the model and processor
        model_name = model_id
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        # prepare the model input
        prompt =your_prompt or "Give me a short introduction to large language model."
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        print("thinking content:", thinking_content)
        print("content:", content)
        caption_file_path = os.path.join("./", "quenresponse.txt")
        with open(caption_file_path, "w") as caption_file:
            caption_file.write(content)

        return content,




NODE_CLASS_MAPPINGS = {
    "ImageCaptioner": ImageCaptioner,
    "Quen3Helper": Quen3Helper,
    "CheckImageCaptionsData": CheckImageCaptionsData,
    "ImageCaptionerPostProcessing": ImageCaptionerPostProcessing,
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCaptioner": "Image Captioner",
    "Quen3Helper": "Quen3 Helper",
    "CheckImageCaptionsData": "Check Image Captions Data",
    "ImageCaptionerPostProcessing": "Image Captioner PostProcessing",
}
