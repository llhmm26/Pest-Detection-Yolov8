import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Optional


class SplitMergeInference:
    """
    A class to handle splitting high-resolution images, performing YOLOv8 inference on splits,
    and merging results back together.
    """
    
    def __init__(self, model_path: str, grid_size: Tuple[int, int] = (4, 4), 
                 confidence_threshold: float = 0.3, iou_threshold: float = 0.5,
                 device: Optional[str] = None):
        """
        Initialize the SplitMergeInference model.
        
        Args:
            model_path: Path to the YOLOv8 model weights
            grid_size: Number of rows and columns to split the image into
            confidence_threshold: Minimum confidence score for detection
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cuda', 'cpu', etc.)
        """
        self.model = YOLO(model_path)
        self.grid_size = grid_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set model parameters
        self.model.conf = confidence_threshold
        self.model.iou = iou_threshold
        
    def split_image(self, image: np.ndarray) -> List[Dict]:
        """
        Split an image into a grid of smaller images.
        
        Args:
            image: Input image as numpy array (H,W,C)
            
        Returns:
            List of dictionaries containing split images and their positions
        """
        h, w = image.shape[:2]
        rows, cols = self.grid_size
        
        # Calculate the size of each split
        split_h, split_w = h // rows, w // cols
        
        # Ensure we cover the entire image by adjusting last row/column if needed
        splits = []
        
        for i in range(rows):
            for j in range(cols):
                # Calculate start and end coordinates
                start_y = i * split_h
                start_x = j * split_w
                
                # Adjust end coordinates for last row/column
                end_y = start_y + split_h if i < rows - 1 else h
                end_x = start_x + split_w if j < cols - 1 else w
                
                # Create the split image
                split_img = image[start_y:end_y, start_x:end_x].copy()
                
                splits.append({
                    'image': split_img,
                    'position': (start_x, start_y, end_x, end_y)
                })
                
        return splits
    
    def run_inference(self, splits: List[Dict]) -> List[Dict]:
        """
        Run YOLOv8 inference on each split.
        
        Args:
            splits: List of split image dictionaries
            
        Returns:
            List of dictionaries with original splits and detection results
        """
        results = []
        
        for i, split in enumerate(splits):
            # Run inference
            detections = self.model(split['image'], verbose=False)[0]
            
            # Add results to the split dictionary
            results.append({
                **split,
                'detections': detections
            })
            
        return results
    
    def merge_results(self, results: List[Dict], original_image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Merge detection results from splits back to original image coordinates.
        
        Args:
            results: List of dictionaries with split images and their detection results
            original_image: The original input image
            
        Returns:
            Tuple of (visualization image, list of adjusted detection dictionaries)
        """
        merged_detections = []
        
        # Create a copy of the original image for visualization
        vis_image = original_image.copy()
        
        for result in results:
            position = result['position']
            detections = result['detections']
            offset_x, offset_y = position[0], position[1]
            
            if len(detections.boxes.xyxy) > 0:
                # Get boxes and convert to numpy
                boxes = detections.boxes.xyxy.cpu().numpy()
                scores = detections.boxes.conf.cpu().numpy()
                class_ids = detections.boxes.cls.cpu().numpy().astype(int)
                
                # Adjust coordinates to original image
                for box_idx in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[box_idx]
                    
                    # Adjust coordinates
                    adj_x1 = x1 + offset_x
                    adj_y1 = y1 + offset_y
                    adj_x2 = x2 + offset_x
                    adj_y2 = y2 + offset_y
                    
                    class_id = class_ids[box_idx]
                    score = scores[box_idx]
                    
                    # Filter by confidence threshold
                    if score >= self.confidence_threshold:
                        # Add to merged detections
                        merged_detections.append({
                            'bbox': (adj_x1, adj_y1, adj_x2, adj_y2),
                            'class_id': class_id,
                            'score': score,
                            'class_name': detections.names[class_id]
                        })
                        
                        # Draw on visualization image
                        color = self._get_color(class_id)
                        cv2.rectangle(
                            vis_image, 
                            (int(adj_x1), int(adj_y1)), 
                            (int(adj_x2), int(adj_y2)), 
                            color, 
                            2
                        )
                        
                        # Add label
                        label = f"{detections.names[class_id]}: {score:.2f}"
                        cv2.putText(
                            vis_image, 
                            label, 
                            (int(adj_x1), int(adj_y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            color, 
                            2
                        )
        
        # Apply non-maximum suppression to remove duplicate detections
        merged_detections = self._apply_nms(merged_detections)
        
        return vis_image, merged_detections
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
            
        # Group detections by class
        detections_by_class = {}
        for det in detections:
            class_id = det['class_id']
            if class_id not in detections_by_class:
                detections_by_class[class_id] = []
            detections_by_class[class_id].append(det)
        
        # Apply NMS for each class
        filtered_detections = []
        for class_id, dets in detections_by_class.items():
            # Extract bboxes, scores
            boxes = np.array([det['bbox'] for det in dets])
            scores = np.array([det['score'] for det in dets])
            
            # Prepare boxes for NMS (xmin, ymin, xmax, ymax)
            boxes_for_nms = boxes
            
            # Apply NMS
            keep_indices = self._nms(boxes_for_nms, scores, self.iou_threshold)
            
            # Keep selected detections
            for idx in keep_indices:
                filtered_detections.append(dets[idx])
        
        return filtered_detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, 
             iou_threshold: float) -> List[int]:
        """
        Apply Non-Maximum Suppression to boxes.
        
        Args:
            boxes: Array of boxes in format (x1, y1, x2, y2)
            scores: Array of confidence scores
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of indices to keep
        """
        # Sort by score
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculate IoU of the selected box with the rest
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            # IoU = intersection / (areas[i] + areas[order[1:]] - intersection)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            union = area_i + area_others - inter
            iou = inter / union
            
            # Get indices of boxes with IoU <= threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
            
        return keep
    
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """
        Generate a consistent color for a class ID.
        
        Args:
            class_id: Class ID
            
        Returns:
            BGR color tuple
        """
        colors = [
            (0, 255, 0),   # Green
            (0, 0, 255),   # Red
            (255, 0, 0),   # Blue
            (0, 255, 255), # Yellow
            (255, 0, 255), # Magenta
            (255, 255, 0), # Cyan
            (128, 0, 0),   # Dark blue
            (0, 128, 0),   # Dark green
            (0, 0, 128),   # Dark red
            (128, 128, 0), # Dark cyan
        ]
        return colors[class_id % len(colors)]
    
    def process_image(self, image_path: str, output_path: Optional[str] = None, 
                     visualize: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single image through the split-infer-merge pipeline.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the output visualization
            visualize: Whether to display the result
            
        Returns:
            Tuple of (visualization image, list of detection dictionaries)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert to RGB for visualization
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Split image
        splits = self.split_image(image_rgb)
        
        # Run inference on splits
        results = self.run_inference(splits)
        
        # Merge results
        vis_image, detections = self.merge_results(results, image_rgb)
        
        # Save result if output path is provided
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert back to BGR for saving with OpenCV
            vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, vis_image_bgr)
            print(f"Results saved to {output_path}")
        
        # Visualize if requested
        if visualize:
            plt.figure(figsize=(12, 10))
            plt.imshow(vis_image)
            plt.axis('off')
            plt.title(f"Detected {len(detections)} objects")
            plt.tight_layout()
            plt.show()
        
        return vis_image, detections


def main():
    """
    Example usage of the SplitMergeInference class.
    """
    # Initialize the model
    model_path = "/home/ilham/playground/yolov8/runs/detect/train18/weights/best.pt"  # Use yolov8n.pt or your custom-trained model
    pipeline = SplitMergeInference(
        model_path=model_path,
        grid_size=(4, 4),
        confidence_threshold=0.1,
        iou_threshold=0.5
    )
    
    # Process an image 
    image_path = "/home/ilham/playground/yolov8/image-test/PSA3P2B /iScout_20250217_0706_07214BBA_PIC_46_CAM_1.xml.jpg"  # Replace with your image path
    output_path = "/home/ilham/playground/yolov8/results/pest_detection_result.jpg"
    
    try:
        vis_image, detections = pipeline.process_image(
            image_path=image_path,
            output_path=output_path,
            visualize=True
        )
        
        # Print detection summary
        print(f"Found {len(detections)} pests in the image")
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        
        for class_name, count in class_counts.items():
            print(f"  - {class_name}: {count}")
            
    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    main()
