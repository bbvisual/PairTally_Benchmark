#!/usr/bin/env python3
"""
Test script for Llama Vision and InternVL basic vision capabilities
Requires: llama-vision environment
Usage: conda activate llama-vision && python test_llama_internvl_vision.py
"""

import os
import sys
from PIL import Image
import traceback
import torch
import torchvision.transforms as T

def find_test_image():
    """Find a test image from the dataset"""
    base_paths = [
        "/home/khanhnguyen/DICTA25/test_bbx_frames/images_384_VarV2",
        "/home/khanhnguyen/DICTA25/test_bbx_frames/images_384_VarV2"
    ]
    
    for base_path in base_paths:
        if os.path.exists(base_path):
            images = [f for f in os.listdir(base_path) if f.endswith('.jpg')]
            if images:
                image_path = os.path.join(base_path, images[0])
                print(f"✅ Found image: {image_path}")
                return image_path
    
    print("❌ No test images found")
    return None

def test_llama_vision():
    """Test Llama Vision capabilities"""
    try:
        print("Loading Llama Vision libraries...")
        from transformers import MllamaForConditionalGeneration, AutoProcessor
        
        # Load model exactly as in evaluation script
        model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        print(f"Loading {model_name}...")
        
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_name)
        
        print("Model loaded successfully!")
        
        # Test image
        image_path = find_test_image()
        if not image_path:
            return
            
        image = Image.open(image_path)
        print(f"Image loaded: {image.size}")
        
        # Test basic vision questions
        questions = [
            "What do you see in this image?",
            "Describe what objects are in this image.",
            "What types of coins are visible in this image?",
            "How would you describe the arrangement of objects in this image?",
            "Are there different types of coins? If so, what types?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}: {question}")
            
            # Create messages as in evaluation script
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]}
            ]
            
            # Process with Llama Vision
            input_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                image, 
                input_text, 
                add_special_tokens=False, 
                return_tensors="pt"
            ).to(model.device)
            
            # Generate response
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.1,
                    top_p=0.9,
                )
            
            # Decode response
            response = processor.decode(output[0], skip_special_tokens=True)
            # Extract only the assistant's response
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()
            
            print(f"Answer: {response}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error with Llama Vision: {e}")
        traceback.print_exc()

def build_transform(input_size):
    """Build transform for InternVL"""
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    else:
        image = image_file.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def test_internvl():
    """Test InternVL vision capabilities"""
    try:
        print("Loading InternVL libraries...")
        from transformers import AutoModel, AutoTokenizer
        
        # Load model exactly as in evaluation script
        model_name = "OpenGVLab/InternVL3-38B-Instruct"
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        
        # Use device_map="auto" for 38B model
        model = AutoModel.from_pretrained(
            model_name,
            device_map="auto",  # Smart distribution across GPUs and CPU
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
        
        print("Model loaded successfully!")
        
        # Test image
        image_path = find_test_image()
        if not image_path:
            return
            
        image = Image.open(image_path)
        print(f"Image loaded: {image.size}")
        
        # Test basic vision questions
        questions = [
            "What do you see in this image?",
            "Describe what objects are in this image.",
            "What types of coins are visible in this image?",
            "How would you describe the arrangement of objects in this image?",
            "Are there different types of coins? If so, what types?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}: {question}")
            
            # Create InternVL format question
            formatted_question = f"<image>\n{question}"
            
            # Load and preprocess image using InternVL's method
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16)
            
            # Move to the appropriate device
            if hasattr(model, 'device'):
                pixel_values = pixel_values.to(model.device)
            else:
                # For models with device_map="auto", get device from parameters
                device = next(model.parameters()).device
                pixel_values = pixel_values.to(device)
            
            # Create generation config
            generation_config = dict(
                max_new_tokens=256, 
                do_sample=False
            )
            
            # Use InternVL's chat method
            response = model.chat(tokenizer, pixel_values, formatted_question, generation_config)
            
            print(f"Answer: {response}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error with InternVL: {e}")
        traceback.print_exc()

def main():
    """Main test function"""
    image_path = find_test_image()
    if not image_path:
        print("❌ No test image found, exiting")
        return
    
    print(f"\n🔍 Testing VLM Basic Vision Capabilities")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Path: {image_path}")
    print("=" * 50)
    
    print("Testing Llama Vision")
    print("=" * 50)
    test_llama_vision()
    
    print("\n" + "=" * 80 + "\n")
    
    print("Testing InternVL")
    print("=" * 50)
    test_internvl()

if __name__ == "__main__":
    main() 