import torch
import os
from PIL import Image
import os
import time
from transformers import pipeline
from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor
)
import json



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
                if image_name.endswith(".JPG") or image_name.endswith(".jpg") or image_name.endswith(".png") or image_name.endswith(".jpeg"):
                    new_image_name = f"{i+1}.jpg"
                    os.rename(os.path.join(image_folder_path, image_name), os.path.join(image_folder_path, new_image_name))
                    print(f"Renamed {image_name} to {new_image_name}")
        else:
            # Rename images in the folder sequentially
            print("inside rename image else ...")
            for i, image_name in enumerate(os.listdir(image_folder_path)):
                if image_name.endswith(".JPG") or image_name.endswith(".jpg") or image_name.endswith(".png") or image_name.endswith(".jpeg"):
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
            if image_name.endswith(".JPG") or image.endswith(".jpg") or image.endswith(".png") or image_name.endswith(".jpeg"):
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
                    captiontextFinal = trigger_word +" , "+ captiontext 
                    
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
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        ).eval()
        # prepare the model input
        prompt =your_prompt or "Give me a short introduction to large language model."
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True. .to(model.device)
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=1,
            early_stopping=True
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


class Quen3HelperGGUF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_message": ("STRING", {"multiline": True, "default": ""}),
                "your_prompt": ("STRING", {"multiline": True, "default": ""}),
                "model_path": ("STRING", {"multiline": True, "default": ""}),
                "model_name": ("STRING", {"multiline": True, "default": ""}),
                "n_gpu_layers": ("INT", {"default": 80, "min": 24,"max": 20000}),
                "n_ctx": ("INT", {"default": 8192,"min": 24,"max": 20000}),
                "max_tokens": ("INT", {"default": 8192,"min": 24,"max": 20000}),
                
                
            }
        }

    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("quen3_response",)
    FUNCTION = "generate_response_gguf"
    OUTPUT_NODE = True
    CATEGORY = "utils"
    def generate_response_gguf(self,system_message,your_prompt, model_path,model_name,n_gpu_layers,n_ctx,max_tokens,):
        # Load the model and processor  .from_pretrained("/path/to/ggml-model.bin", model_type="gpt2")


        # model = AutoModelForCausalLM.from_pretrained("/workspace/ComfyUI/models/Qwen3gguf/Qwen3-30B-A6B-16-Extreme.Q6_K.gguf", model_type="gpt2")
        full_model_path = os.path.join(model_path, model_name)
        llm = Llama(
        model_path=full_model_path,
        n_gpu_layers=n_gpu_layers, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
        n_ctx=n_ctx, # Uncomment to increase the context window
)
        # prepare the model input  model_type="qwen3"
        print("supports_gpu........")
        

        system_message = system_message or "You are a knowledgeable and creative historian. You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."
        user_message =your_prompt or "Write a long, detailed story on the history of the Kohinoor diamond and how it ended up in Britain."
        
        prompt = f"""<|system|>
        {system_message}</s>
        <|user|>
        {user_message}</s>
        <|assistant|>"""
        
        
        output =llm(
                  prompt, # Prompt
                  max_tokens=max_tokens,  # Generate up to 512 tokens
                  stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
                  echo=False        # Whether to echo the prompt
                )
        response_text = output["choices"][0]["text"]
         # response=model(prompt)
        caption_file_path = os.path.join(model_path, "quenresponse.txt")
        with open(caption_file_path, "w") as caption_file:
            json.dump(output, caption_file, indent=2)
        tosend=json.dumps(output)
        return response_text,

NODE_CLASS_MAPPINGS = {
    "ImageCaptioner": ImageCaptioner,
    "Quen3Helper": Quen3Helper,
    "CheckImageCaptionsData": CheckImageCaptionsData,
    "ImageCaptionerPostProcessing": ImageCaptionerPostProcessing,
    "Quen3HelperGGUF": Quen3HelperGGUF,
    
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCaptioner": "Image Captioner",
    "Quen3Helper": "Quen3 Helper",
    "CheckImageCaptionsData": "Check Image Captions Data",
    "ImageCaptionerPostProcessing": "Image Captioner PostProcessing",
    "Quen3HelperGGUF": "Quen3Helper GGUF",
}
