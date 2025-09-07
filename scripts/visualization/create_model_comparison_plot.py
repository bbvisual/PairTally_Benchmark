import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import random
import argparse
from pathlib import Path
import seaborn as sns

# Set minimal style for conference paper
plt.style.use('default')  # Use clean default style

class ModelComparisonVisualizer:
    def __init__(self, results_dir="/home/khanhnguyen/DICTA25-RESULTS", dataset_name="final_dataset_default"):
        self.results_dir = Path(results_dir)
        self.dataset_name = dataset_name
        
        # Define models and their types (all 6 models including LLMDet)
        self.models = {
            'DAVE': {
                'type': 'bbox',
                'dir': 'DAVE-qualitative',
                'color': '#FF0000'
            },
            'GeCo': {
                'type': 'bbox',
                'dir': 'GeCo-qualitative',
                'color': '#00FF00'
            },
            'CountGD': {
                'type': 'bbox',
                'dir': 'CountGD-qualitative',
                'color': '#00FFFF'
            },
            'FamNet': {
                'type': 'density',
                'dir': 'LearningToCountEverything-qualitative',
                'color': '#96CEB4'
            },
            'LOCA': {
                'type': 'density',
                'dir': 'LOCA-qualitative',
                'color': '#FFEAA7'
            },
            'LLMDet': {
                'type': 'bbox',
                'dir': 'LLMDet',
                'color': '#FF6B6B'
            }
        }
        
    def load_model_data(self, model_name, model_info):
        """Load qualitative data for a specific model"""
        if model_name == 'LLMDet':
            # LLMDet has a different data structure
            data_path = self.results_dir / model_info['dir'] / 'result_llmdet.json'
        else:
        data_path = self.results_dir / model_info['dir'] / self.dataset_name / 'positive_qualitative_data.json'
        
        if not data_path.exists():
            print(f"⚠️  Warning: {model_name} data not found at {data_path}")
            return None
            
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            if model_name == 'LLMDet':
                # Convert LLMDet format to standard format
                converted_data = {'images': []}
                for filename, entry in data.items():
                    # Extract prediction and ground truth
                    pred_count = entry.get('pred_count', 0)
                    obj_1_num = entry.get('obj_1_num', 0)
                    obj_2_num = entry.get('obj_2_num', 0)
                    
                    # Determine which class this prediction is for based on caption
                    caption = entry.get('caption', '').lower()
                    
                    # Parse filename to get object names
                    parsed = self.parse_filename(filename)
                    if parsed:
                        obj1_name = parsed['obj1_name'].lower()
                        obj2_name = parsed['obj2_name'].lower()
                        
                        # Determine target class
                        if caption in obj1_name or obj1_name in caption:
                            gt_count = obj_1_num
                        elif caption in obj2_name or obj2_name in caption:
                            gt_count = obj_2_num
                        else:
                            # Skip if we can't determine
                            continue
                        
                        # Convert boxes format if present
                        boxes = entry.get('boxes', [])
                        predicted_boxes = []
                        if boxes:
                            # LLMDet boxes are in [x1, y1, x2, y2] format
                            predicted_boxes = boxes
                        
                        converted_entry = {
                            'image_name': filename,
                            'pred_count': pred_count,
                            'gt_count': gt_count,
                            'predicted_boxes': predicted_boxes
                        }
                        converted_data['images'].append(converted_entry)
                
                print(f"✅ Loaded {model_name}: {len(converted_data['images'])} images")
                return converted_data
            else:
            print(f"✅ Loaded {model_name}: {len(data['images'])} images")
            return data
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            return None
    
    def parse_filename(self, filename):
        """
        Parse filename according to the format:
        {obj1_name}_{obj2_name}_{test_type}_{super_category}_{pos_code}_{neg_code}_{pos_count}_{neg_count}_{id1}_{id2}.jpg
        """
        # Remove .jpg extension
        name = filename.replace('.jpg', '')
        
        # Split by underscores
        parts = name.split('_')
        
        if len(parts) < 10:
            return None
        
        # Find the indices for test_type, super_category, etc.
        # Look for INTRA or INTER
        test_type_idx = None
        for i, part in enumerate(parts):
            if part in ['INTRA', 'INTER']:
                test_type_idx = i
                break
        
        if test_type_idx is None:
            return None
        
        # Reconstruct object names (they might contain underscores)
        obj1_parts = parts[:test_type_idx-2]
        obj2_parts = [parts[test_type_idx-2]]
        
        # Find where obj2 ends by looking for the super category
        super_cat_idx = test_type_idx + 1
        if super_cat_idx >= len(parts):
            return None
        
        super_category = parts[super_cat_idx]
        
        # The remaining parts should be pos_code, neg_code, pos_count, neg_count, id1, id2
        if len(parts) - super_cat_idx < 6:
            return None
        
        pos_code = parts[super_cat_idx + 1]
        neg_code = parts[super_cat_idx + 2]
        pos_count = parts[super_cat_idx + 3]
        neg_count = parts[super_cat_idx + 4]
        id1 = parts[super_cat_idx + 5]
        id2 = parts[super_cat_idx + 6]
        
        # Reconstruct object names
        obj1_name = '-'.join(obj1_parts)
        obj2_name = '-'.join(obj2_parts)
        
        return {
            'obj1_name': obj1_name,
            'obj2_name': obj2_name,
            'test_type': parts[test_type_idx],
            'super_category': super_category,
            'pos_code': pos_code,
            'neg_code': neg_code,
            'pos_count': int(pos_count),
            'neg_count': int(neg_count),
            'id1': id1,
            'id2': id2
        }
    
    def find_common_images(self):
        """Find images that exist in all models"""
        all_data = {}
        
        # Load data for all models
        for model_name, model_info in self.models.items():
            data = self.load_model_data(model_name, model_info)
            if data:
                all_data[model_name] = data
        
        if not all_data:
            raise ValueError("No model data found!")
        
        # Find common image names
        image_sets = []
        for model_name, data in all_data.items():
            image_names = {img['image_name'] for img in data['images']}
            image_sets.append(image_names)
            print(f"{model_name}: {len(image_names)} images")
        
        common_images = set.intersection(*image_sets)
        print(f"\n🔍 Found {len(common_images)} common images across all models")
        
        if not common_images:
            raise ValueError("No common images found across all models!")
        
        return all_data, list(common_images)
    
    def get_image_data(self, all_data, image_name):
        """Extract data for a specific image from all models"""
        image_data = {}
        
        for model_name, data in all_data.items():
            for img in data['images']:
                if img['image_name'] == image_name:
                    image_data[model_name] = img
                    break
        
        return image_data
    
    def load_image(self, image_name, model_name=None):
        """Load the original image or model-specific image"""
        if model_name == 'DAVE':
            # Load DAVE's processed image from DAVE results directory
            # DAVE saves images with .jpg.png extension (PNG versions of JPG files)
            dave_image_path = Path(f"/home/khanhnguyen/DICTA25-RESULTS/DAVE-qualitative/{self.dataset_name}/images/{image_name}.png")
            print(f"🔍 Loading DAVE image from: {dave_image_path}")
            if dave_image_path.exists():
                return Image.open(dave_image_path)
            else:
                print(f"⚠️  DAVE image not found at: {dave_image_path}")
                # Fallback to original image
                return self.load_original_image(image_name)
        
        # For all other models, load from standard test frames directory
        standard_image_path = Path(f"/home/khanhnguyen/DICTA25/{self.dataset_name}/images/{image_name}")
        if standard_image_path.exists():
            return Image.open(standard_image_path)
        
        # Fallback to original image loading method
        return self.load_original_image(image_name)
    
    def load_original_image(self, image_name):
        """Load the original image from various possible paths"""
        # Primary path for test frames
        primary_path = Path(f"/home/khanhnguyen/DICTA25/{self.dataset_name}/images/{image_name}")
        
        # Try different possible image paths for original images
        possible_paths = [
            primary_path,
            self.results_dir / 'LearningToCountEverything-qualitative' / self.dataset_name / 'images' / image_name,
            Path(f"/home/khanhnguyen/DICTA25/{self.dataset_name}/images_384_VarV2/{image_name}")
        ]
        
        for path in possible_paths:
            if path.exists():
                return Image.open(path)
        
        raise FileNotFoundError(f"Could not find image: {image_name}")
    
    def load_density_map(self, model_name, image_name, class_type='positive'):
        """Load density map for density-based models"""
        model_info = self.models[model_name]
        
        # Remove file extension and add .npy
        img_name_no_ext = os.path.splitext(image_name)[0]
        density_path = self.results_dir / model_info['dir'] / self.dataset_name / f'density_maps_{class_type}' / f'{img_name_no_ext}.npy'
        
        if not density_path.exists():
            # Try with full filename
            density_path = self.results_dir / model_info['dir'] / self.dataset_name / f'density_maps_{class_type}' / f'{image_name}.npy'
        
        if density_path.exists():
            return np.load(density_path)
        else:
            print(f"⚠️  Density map not found for {model_name}: {density_path}")
            return None
    
    def draw_bounding_boxes(self, ax, boxes, color, alpha=0.7, linewidth=1):
        """Draw bounding boxes on an axis"""
        for box in boxes:
            if len(box) == 4:
                x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
                rect = patches.Rectangle((x, y), w, h, linewidth=linewidth, 
                                       edgecolor=color, facecolor='none', alpha=alpha)
                ax.add_patch(rect)
    
    def draw_points(self, ax, points, color, alpha=0.9, size=50):
        """Draw points for VLM predictions"""
        if not points:
            return
            
        for point in points:
            if len(point) == 2:
                x, y = point[0], point[1]
                ax.scatter(x, y, c=color, s=size, alpha=alpha, marker='o', 
                          edgecolors='white', linewidth=1)
            elif len(point) == 4:  # Handle as bounding box center
                x, y = (point[0] + point[2]) / 2, (point[1] + point[3]) / 2
                ax.scatter(x, y, c=color, s=size, alpha=alpha, marker='o', 
                          edgecolors='white', linewidth=1)
    
    def create_comparison_plot(self, image_name, image_data, save_path=None):
        # --- 1. load image as before ---
        original_img = self.load_image(image_name)
        img_array = np.array(original_img)

        # --- 2. decide which models will be shown ---
        model_order = ['DAVE', 'GeCo', 'CountGD', 'FamNet', 'LOCA', 'LLMDet']
        available_models = [m for m in model_order if m in image_data]

        # --- 3. build a 2 × 3 grid instead of 1 × 6 ---
        n_rows, n_cols = 2, 3                       # ← 2 rows, 3 columns
        fig = plt.figure(figsize=(12, 8))           # taller, slightly narrower
        gs  = fig.add_gridspec(n_rows, n_cols,
                            wspace=0.02, hspace=0.1)

        # --- 4. draw each panel ---
        gt_count = list(image_data.values())[0].get('gt_count', 'N/A')
        total_slots = n_rows * n_cols

        for idx in range(total_slots):
            r, c = divmod(idx, n_cols)              # row, col indices
            ax = fig.add_subplot(gs[r, c])

            if idx < len(available_models):         # real model output
                model_name  = available_models[idx]
                model_info  = self.models[model_name]
                pred_count  = image_data[model_name].get('pred_count', 'N/A')

                # choose image background
                if model_name == 'DAVE':
                    ax.imshow(np.array(self.load_image(image_name, 'DAVE')))
                else:
                    ax.imshow(img_array)
                
                data = image_data[model_name]
                model_info = self.models[model_name]
                pred_count = data.get('pred_count', 'N/A')
                
                # Draw predictions based on model type
                if model_info['type'] == 'bbox' and 'predicted_boxes' in data:
                    predicted_boxes = data['predicted_boxes']
                    
                    # Handle different models with their specific coordinate systems
                    if model_name == 'DAVE':
                        if predicted_boxes and len(predicted_boxes) > 0:
                            # Load DAVE image to get its actual dimensions
                            dave_img = self.load_image(image_name, model_name='DAVE')
                            if dave_img:
                                dave_img_array = np.array(dave_img)
                                # DAVE boxes are in 512x512 coordinates, scale them to the DAVE image size
                                scaled_boxes = self.scale_dave_coordinates(predicted_boxes, dave_img_array.shape)
                            else:
                                # Fallback to original image dimensions
                                scaled_boxes = self.scale_dave_coordinates(predicted_boxes, img_array.shape)
                            self.draw_bounding_boxes(ax, scaled_boxes, model_info['color'], alpha=0.9, linewidth=1)
                    
                    elif model_name in ['GeCo', 'CountGD', 'LLMDet']:
                        if predicted_boxes:
                            # Convert coordinates if needed
                            if all(0 <= coord <= 1 for box in predicted_boxes for coord in box):
                                h, w = img_array.shape[:2]
                                predicted_boxes = [[x*w, y*h, x2*w, y2*h] for x, y, x2, y2 in predicted_boxes]
                            
                            # Use model-specific colors
                            self.draw_bounding_boxes(ax, predicted_boxes, model_info['color'], alpha=0.9, linewidth=1)
                
                elif model_info['type'] == 'density':
                    # For density models (FAMNet, LOCA) - show original image with density overlay
                    density_map = self.load_density_map(model_name, image_name)
                    if density_map is not None:
                        # Resize density map to match image if needed
                        if density_map.shape != img_array.shape[:2]:
                            from scipy.ndimage import zoom
                            zoom_factors = (img_array.shape[0] / density_map.shape[0], 
                                          img_array.shape[1] / density_map.shape[1])
                            density_map = zoom(density_map, zoom_factors, order=1)
                        
                        # Overlay density map with vibrant colors
                        ax.imshow(density_map, cmap='hot', alpha=0.6, interpolation='bilinear')
                
                # Minimalistic title - small and subtle
                ax.set_title(f'{model_name}\nPred: {pred_count} GT: {gt_count}', 
                            fontsize=10, pad=2)
                ax.axis('off')
            else:                                    # placeholder if < 6 models
                ax.imshow(img_array, alpha=0.15)
                ax.text(0.5, 0.5, 'Empty', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='lightgray',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.set_title('Empty\nPred:__ GT: __', fontsize=10, pad=2, color='lightgray')
            ax.axis('off')

        # --- 5. tighten layout & save if asked ---
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05,
                            hspace=0.05, wspace=0.02)
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
            print(f"📊 Saved comparison plot: {save_path}")
        return fig
    
    def create_specific_image_comparison(self, image_name, save_path=None):
        """Create comparison for a specific image"""
        print(f"🎯 Creating comparison for specific image: {image_name}")
        
        # Load data for all models
        all_data = {}
        for model_name, model_info in self.models.items():
            data = self.load_model_data(model_name, model_info)
            if data:
                all_data[model_name] = data
        
        if not all_data:
            raise ValueError("No model data found!")
        
        # Find the specific image in the data
        image_data = {}
        for model_name, data in all_data.items():
            for img in data['images']:
                if img['image_name'] == image_name:
                    image_data[model_name] = img
                    break
        
        if not image_data:
            # List available images
            available_images = set()
            for model_name, data in all_data.items():
                for img in data['images']:
                    available_images.add(img['image_name'])
            
            print(f"❌ Image '{image_name}' not found!")
            print(f"Available images (first 10): {list(available_images)[:10]}")
            return None, None, None
        
        print(f"📊 Found data for {len(image_data)} models")
        
        # Create visualization
        fig = self.create_comparison_plot(image_name, image_data, save_path)
        
        return fig, image_name, image_data
    
    def run_comparison(self, random_seed=None, save_dir=None):
        """Run the complete comparison workflow"""
        print("🚀 Starting Model Comparison Visualization\n")
        
        # Set random seed for reproducibility
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Find common images
        all_data, common_images = self.find_common_images()
        
        # Select random image
        selected_image = random.choice(common_images)
        print(f"\n🎲 Randomly selected image: {selected_image}")
        
        # Get data for selected image
        image_data = self.get_image_data(all_data, selected_image)
        print(f"📊 Found data for {len(image_data)} models")
        
        # Create visualization
        fig = self.create_comparison_plot(selected_image, image_data)
        
        # Save if requested
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / f"model_comparison_{os.path.splitext(selected_image)[0]}.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to: {save_path}")
        
        plt.show()
        return fig, selected_image, image_data
    
    def scale_dave_coordinates(self, boxes, original_image_shape):
        h, w = original_image_shape[:2]
        scaled = []
        for x1, y1, x2, y2 in boxes:
            # if these are normalized:
            if 0.0 <= x1 <= 1.0:
                x1 *= w;  x2 *= w
                y1 *= h;  y2 *= h
            # otherwise assume they're already in pixel coords
            scaled.append([x1, y1, x2, y2])
        return scaled


def main():
    parser = argparse.ArgumentParser(description='Create professional model comparison visualization for 6 counting models including LLMDet')
    parser.add_argument('--dataset', default='final_dataset_default', help='Dataset name to analyze')
    parser.add_argument('--results_dir', default='/home/khanhnguyen/DICTA25-RESULTS', help='Results directory')
    parser.add_argument('--save_dir', default='./comparison_plots', help='Directory to save plots')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible image selection')
    parser.add_argument('--image_name', help='Specific image name to visualize (instead of random)')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ModelComparisonVisualizer(args.results_dir, args.dataset)
    
    if args.image_name:
        # Load data and visualize specific image
        fig, image_name, image_data = visualizer.create_specific_image_comparison(args.image_name)
        
        if fig is None:
            return
        
        if args.save_dir:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / f"model_comparison_{os.path.splitext(args.image_name)[0]}.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
            print(f"💾 Saved to: {save_path}")
        
        plt.show()
    else:
        # Run random comparison
        visualizer.run_comparison(random_seed=args.seed, save_dir=args.save_dir)

if __name__ == '__main__':
    main()