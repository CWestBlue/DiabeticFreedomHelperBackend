"""Per-mask food recognition service.

MVP-2.9: Uses segmentation masks to crop individual food regions
and sends each to LLaVA for improved multi-item detection.
"""

import asyncio
import base64
import io
import logging
import time
from typing import Sequence

import cv2
import numpy as np

from diabetic_api.models.food_scan import SegmentationMask, SegmentationResult

from .base import (
    EstimatedMacros,
    FoodRecognitionResult,
    RecognizedFood,
    RecognizedFoodWithMask,
)
from .ollama_provider import OllamaFoodRecognition

logger = logging.getLogger(__name__)


# Configuration
MIN_MASK_AREA_PIXELS = 5000  # Filter out tiny masks (noise)
MAX_MASKS_TO_PROCESS = 5  # Limit for performance
CROP_PADDING_RATIO = 0.1  # Add 10% padding around bounding box


class PerMaskFoodRecognition:
    """
    Orchestrates per-mask food recognition.
    
    Takes segmentation masks and RGB image, crops individual food regions,
    and sends each to LLaVA for recognition.
    """
    
    def __init__(self, recognition_service: OllamaFoodRecognition) -> None:
        """
        Initialize per-mask recognition.
        
        Args:
            recognition_service: The underlying food recognition service
        """
        self._recognition_service = recognition_service
    
    async def recognize_with_masks(
        self,
        rgb_data: bytes,
        segmentation_result: SegmentationResult,
        max_masks: int = MAX_MASKS_TO_PROCESS,
        min_mask_area: int = MIN_MASK_AREA_PIXELS,
        parallel: bool = True,
    ) -> FoodRecognitionResult:
        """
        Recognize foods using segmentation masks.
        
        For each mask:
        1. Crop the RGB image using the bounding box
        2. Send to LLaVA for single-item recognition
        3. Aggregate results with mask linkage
        
        Args:
            rgb_data: Raw RGB image bytes (JPEG)
            segmentation_result: Result from segmentation service
            max_masks: Maximum masks to process
            min_mask_area: Minimum mask area to process (filters noise)
            parallel: Whether to process masks in parallel
            
        Returns:
            FoodRecognitionResult with per-mask identified foods
        """
        start_time = time.time()
        
        # Decode RGB image
        rgb_array = self._decode_image(rgb_data)
        if rgb_array is None:
            logger.error("Failed to decode RGB image for per-mask recognition")
            return self._empty_result("Failed to decode image")
        
        height, width = rgb_array.shape[:2]
        
        # Filter and sort masks
        valid_masks = self._filter_masks(
            segmentation_result.masks,
            min_area=min_mask_area,
            max_count=max_masks,
        )
        
        if not valid_masks:
            logger.warning("No valid masks after filtering")
            return self._empty_result("No valid masks")
        
        logger.info(f"Processing {len(valid_masks)} masks for per-mask recognition")
        
        # Process each mask
        if parallel:
            foods = await self._process_masks_parallel(rgb_array, valid_masks)
        else:
            foods = await self._process_masks_sequential(rgb_array, valid_masks)
        
        # Filter out failed recognitions
        recognized_foods = [f for f in foods if f is not None]
        
        if not recognized_foods:
            logger.warning("All per-mask recognitions failed")
            return self._empty_result("All recognitions failed")
        
        # Calculate overall confidence (average of individual confidences)
        overall_confidence = sum(f.confidence for f in recognized_foods) / len(recognized_foods)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"Per-mask recognition complete: {len(recognized_foods)} foods "
            f"identified in {processing_time_ms}ms"
        )
        
        return FoodRecognitionResult(
            foods=recognized_foods,
            overall_confidence=overall_confidence,
            is_multi_food_plate=len(recognized_foods) > 1,
            raw_response=f"Per-mask recognition: {len(recognized_foods)} items",
            provider=f"{self._recognition_service.provider_name}/per-mask",
            processing_time_ms=processing_time_ms,
        )
    
    async def _process_masks_parallel(
        self,
        rgb_array: np.ndarray,
        masks: list[tuple[int, SegmentationMask]],
    ) -> list[RecognizedFoodWithMask | None]:
        """Process all masks in parallel using asyncio.gather."""
        
        tasks = [
            self._process_single_mask(rgb_array, idx, mask)
            for idx, mask in masks
        ]
        
        return await asyncio.gather(*tasks)
    
    async def _process_masks_sequential(
        self,
        rgb_array: np.ndarray,
        masks: list[tuple[int, SegmentationMask]],
    ) -> list[RecognizedFoodWithMask | None]:
        """Process masks sequentially."""
        
        results = []
        for idx, mask in masks:
            result = await self._process_single_mask(rgb_array, idx, mask)
            results.append(result)
        return results
    
    async def _process_single_mask(
        self,
        rgb_array: np.ndarray,
        mask_index: int,
        mask: SegmentationMask,
    ) -> RecognizedFoodWithMask | None:
        """
        Process a single mask: crop and recognize.
        
        Args:
            rgb_array: Full RGB image as numpy array
            mask_index: Original index of the mask
            mask: Segmentation mask with bbox
            
        Returns:
            RecognizedFoodWithMask if successful, None otherwise
        """
        try:
            # Crop the image using bounding box
            cropped = self._crop_with_bbox(rgb_array, mask.bbox)
            
            if cropped is None or cropped.size == 0:
                logger.warning(f"Mask {mask_index}: empty crop")
                return None
            
            # Encode crop as JPEG for sending to LLaVA
            crop_bytes = self._encode_image(cropped)
            
            if crop_bytes is None:
                logger.warning(f"Mask {mask_index}: failed to encode crop")
                return None
            
            # Recognize the single food item
            recognized = await self._recognition_service.recognize_single(
                image_data=crop_bytes,
                mask_index=mask_index,
            )
            
            if recognized is None:
                logger.debug(f"Mask {mask_index}: recognition returned None")
                return None
            
            # Create RecognizedFoodWithMask with mask linkage
            return RecognizedFoodWithMask(
                # Copy all fields from RecognizedFood
                label=recognized.label,
                confidence=recognized.confidence,
                estimated_grams=recognized.estimated_grams,
                estimated_macros=recognized.estimated_macros,
                category=recognized.category,
                is_mixed_dish=recognized.is_mixed_dish,
                visible_components=recognized.visible_components,
                possible_usda_matches=recognized.possible_usda_matches,
                # Add mask-specific fields
                mask_index=mask_index,
                bbox=mask.bbox,
                mask_area_pixels=mask.area_pixels,
            )
            
        except Exception as e:
            logger.warning(f"Mask {mask_index}: processing failed: {e}")
            return None
    
    def _filter_masks(
        self,
        masks: Sequence[SegmentationMask],
        min_area: int,
        max_count: int,
    ) -> list[tuple[int, SegmentationMask]]:
        """
        Filter and sort masks for processing.
        
        Args:
            masks: List of segmentation masks
            min_area: Minimum area in pixels
            max_count: Maximum masks to return
            
        Returns:
            List of (original_index, mask) tuples, sorted by area (largest first)
        """
        # Filter by area and create indexed tuples
        valid = [
            (i, m) for i, m in enumerate(masks)
            if m.area_pixels >= min_area
        ]
        
        # Sort by area (largest first)
        valid.sort(key=lambda x: x[1].area_pixels, reverse=True)
        
        # Limit count
        return valid[:max_count]
    
    def _crop_with_bbox(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        padding_ratio: float = CROP_PADDING_RATIO,
    ) -> np.ndarray | None:
        """
        Crop image using bounding box with optional padding.
        
        Args:
            image: Input image (H, W, C)
            bbox: Bounding box as (x, y, width, height)
            padding_ratio: Padding to add around bbox (0.1 = 10%)
            
        Returns:
            Cropped image array or None if invalid
        """
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]
        
        # Add padding
        pad_w = int(w * padding_ratio)
        pad_h = int(h * padding_ratio)
        
        # Calculate padded bounds (clamped to image boundaries)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)
        
        # Validate
        if x2 <= x1 or y2 <= y1:
            return None
        
        return image[y1:y2, x1:x2].copy()
    
    def _decode_image(self, image_data: bytes) -> np.ndarray | None:
        """Decode JPEG/PNG bytes to numpy array."""
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            return None
    
    def _encode_image(self, image: np.ndarray, quality: int = 85) -> bytes | None:
        """Encode numpy array to JPEG bytes."""
        try:
            # Convert RGB to BGR for cv2
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buffer.tobytes()
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None
    
    def _empty_result(self, reason: str) -> FoodRecognitionResult:
        """Create an empty result for failure cases."""
        return FoodRecognitionResult(
            foods=[],
            overall_confidence=0.0,
            is_multi_food_plate=False,
            raw_response=f"Per-mask recognition failed: {reason}",
            provider=f"{self._recognition_service.provider_name}/per-mask",
            processing_time_ms=0,
        )


def create_per_mask_service(
    recognition_service: OllamaFoodRecognition,
) -> PerMaskFoodRecognition:
    """Factory function to create per-mask recognition service."""
    return PerMaskFoodRecognition(recognition_service)
