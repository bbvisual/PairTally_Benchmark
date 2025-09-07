#!/usr/bin/env python3
"""
Test script for Qwen2.5-VL basic vision capabilities
Requires: qwen-vl environment with specific transformers version
Usage: conda activate qwen-vl && python test_qwen_vision.py
"""

import os
import sys
from PIL import Image
import traceback

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

def test_qwen2_5vl():
    """Test Qwen2.5-VL vision capabilities"""
    try:
        print("Loading Qwen2.5-VL libraries...")
        from transformers import Qwen2_5VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        import torch
        
        # Load model exactly as in evaluation script
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        print(f"Loading {model_name}...")
        
        model = Qwen2_5VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
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
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Process with Qwen2.5-VL
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            # Generate response
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            print(f"Answer: {output_text[0]}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error with Qwen2.5-VL: {e}")
        traceback.print_exc()

def main():
    """Main test function"""
    image_path = find_test_image()
    if not image_path:
        print("❌ No test image found, exiting")
        return
    
    print(f"\n🔍 Testing Qwen2.5-VL Vision Capabilities")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Path: {image_path}")
    print("=" * 50)
    
    test_qwen2_5vl()

if __name__ == "__main__":
    main() 