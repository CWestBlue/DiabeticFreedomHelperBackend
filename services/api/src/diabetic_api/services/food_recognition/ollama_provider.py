"""
Ollama/LLaVA provider for food recognition.

Uses local Ollama instance with LLaVA vision model to identify foods in images.
"""

import base64
import json
import logging
import time
from typing import Any

import httpx

from .base import (
    EstimatedMacros,
    FoodCategory,
    FoodRecognitionError,
    FoodRecognitionResult,
    FoodRecognitionService,
    RecognizedFood,
)

logger = logging.getLogger(__name__)


# Prompt for food recognition - Multi-food detection (MVP-3.4)
FOOD_RECOGNITION_PROMPT = """You are a food identification expert. Analyze this specific image and identify ALL DISTINCT FOODS you can see.

CRITICAL: Only identify foods that are ACTUALLY VISIBLE in this image. Do not copy examples.

ANALYSIS STEPS:
1. Look carefully at the image - what foods do you actually see?
2. Look for ANY visible text, logos, or brand names on packaging
3. Count how many DIFFERENT food items are visible
4. For each distinct food, identify it separately with realistic macros

For EACH distinct food you see, provide:
1. Food name - BE SPECIFIC! Include brand if visible on packaging
2. Estimated portion size in grams
3. Confidence level (0.0 to 1.0)
4. Food category (protein, carbohydrate, vegetable, fruit, dairy, fat, beverage, mixed, snack)
5. Whether this item is a mixed dish (true/false)
6. If mixed dish: list visible_components if identifiable
7. Realistic macronutrients for the estimated portion

Respond ONLY with valid JSON in this format:
{
  "foods": [
    {
      "label": "<actual food name you see>",
      "confidence": <0.0-1.0>,
      "estimated_grams": <number>,
      "category": "<category>",
      "is_mixed_dish": <true/false>,
      "visible_components": <null or array of strings>,
      "macros": {"carbs": <number>, "protein": <number>, "fat": <number>, "fiber": <number>}
    }
  ],
  "is_multi_food_plate": <true if multiple distinct foods, false otherwise>,
  "overall_confidence": <0.0-1.0>
}

RULES:
- ONLY return foods you can ACTUALLY SEE in this image
- Do NOT invent or assume foods that aren't visible
- Return 1-8 foods maximum
- Include brand names if visible (e.g., "Quest protein bar", "KIND bar", "Blue Diamond almonds")
- Use realistic macros for the specific food identified
- Sort by confidence (highest first)

Do not include any text outside the JSON."""


# Simplified prompt for single food item recognition (MVP-2.9)
SINGLE_FOOD_RECOGNITION_PROMPT = """You are a food identification expert. This image shows a SINGLE food item that has been cropped from a larger image.

Identify this ONE food item. Be specific about what you see.

Provide:
1. Food name - be specific, include brand if visible
2. Estimated portion size in grams
3. Confidence level (0.0 to 1.0)
4. Food category (protein, carbohydrate, vegetable, fruit, dairy, fat, beverage, mixed, snack)
5. Whether this is a mixed dish (true/false)
6. Realistic macronutrients for the estimated portion

Respond ONLY with valid JSON:
{
  "label": "<food name>",
  "confidence": <0.0-1.0>,
  "estimated_grams": <number>,
  "category": "<category>",
  "is_mixed_dish": <true/false>,
  "macros": {"carbs": <number>, "protein": <number>, "fat": <number>, "fiber": <number>}
}

RULES:
- Identify the SINGLE food item shown
- Use realistic macros for the portion size
- If you cannot identify the food, set label to "Unknown food" with low confidence

Do not include any text outside the JSON."""


class OllamaFoodRecognition(FoodRecognitionService):
    """
    Food recognition using Ollama with LLaVA vision model.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llava:7b",
        timeout: float = 60.0,
    ):
        """
        Initialize Ollama provider.
        
        Args:
            base_url: Ollama API base URL
            model: Vision model to use (default: llava:7b)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
    
    @property
    def provider_name(self) -> str:
        return f"ollama/{self.model}"
    
    async def recognize(
        self,
        image_data: bytes,
        *,
        max_foods: int = 5,
        include_macros: bool = True,
        include_portions: bool = True,
    ) -> FoodRecognitionResult:
        """
        Recognize foods in an image using LLaVA.
        """
        start_time = time.time()
        
        try:
            # Encode image to base64
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            
            # Build request
            request_body = {
                "model": self.model,
                "prompt": FOOD_RECOGNITION_PROMPT,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Slightly higher for better analysis variety
                    "num_predict": 1500,  # More tokens for detailed analysis
                    "top_p": 0.9,  # Allow some creativity in identification
                },
            }
            
            logger.info(f"Sending food recognition request to Ollama ({self.model})")
            
            # Call Ollama API
            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json=request_body,
            )
            
            if response.status_code != 200:
                raise FoodRecognitionError(
                    message=f"Ollama API error: {response.status_code}",
                    error_code="PROVIDER_ERROR",
                    provider=self.provider_name,
                    details={"status_code": response.status_code, "body": response.text},
                )
            
            result_data = response.json()
            raw_response = result_data.get("response", "")
            
            logger.debug(f"Raw Ollama response: {raw_response[:500]}...")
            
            # Parse JSON from response (MVP-3.4: now returns is_multi_food_plate)
            foods, overall_confidence, is_multi_food_plate = self._parse_response(raw_response)
            
            # Limit to max_foods
            foods = foods[:max_foods]
            
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Parsed {len(foods)} foods, is_multi_food_plate={is_multi_food_plate}"
            )
            
            return FoodRecognitionResult(
                foods=foods,
                overall_confidence=overall_confidence,
                is_multi_food_plate=is_multi_food_plate,
                raw_response=raw_response,
                provider=self.provider_name,
                processing_time_ms=processing_time,
            )
            
        except httpx.RequestError as e:
            raise FoodRecognitionError(
                message=f"Failed to connect to Ollama: {e}",
                error_code="CONNECTION_ERROR",
                provider=self.provider_name,
            ) from e
        except FoodRecognitionError:
            raise
        except Exception as e:
            logger.exception("Unexpected error in food recognition")
            raise FoodRecognitionError(
                message=f"Unexpected error: {e}",
                error_code="UNEXPECTED_ERROR",
                provider=self.provider_name,
            ) from e
    
    def _parse_response(
        self, raw_response: str
    ) -> tuple[list[RecognizedFood], float, bool]:
        """Parse LLaVA response into structured food data.
        
        Returns:
            tuple of (foods list, overall_confidence, is_multi_food_plate)
        """
        
        # Try to extract JSON from response
        json_str = self._extract_json(raw_response)
        
        if not json_str:
            logger.warning("Could not extract JSON from response")
            return [], 0.0, False
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return [], 0.0, False
        
        foods = []
        for item in data.get("foods", []):
            try:
                # Parse macros if present
                macros = None
                if "macros" in item:
                    macros = EstimatedMacros(
                        carbs=float(item["macros"].get("carbs", 0)),
                        protein=float(item["macros"].get("protein", 0)),
                        fat=float(item["macros"].get("fat", 0)),
                        fiber=float(item["macros"].get("fiber", 0)),
                    )
                
                # Parse category
                category_str = item.get("category", "unknown").lower()
                try:
                    category = FoodCategory(category_str)
                except ValueError:
                    category = FoodCategory.UNKNOWN
                
                # Parse visible components for mixed dishes (MVP-3.4)
                visible_components = None
                if item.get("visible_components"):
                    components = item.get("visible_components")
                    if isinstance(components, list):
                        visible_components = [str(c) for c in components if c]
                
                food = RecognizedFood(
                    label=item.get("label", "Unknown Food"),
                    confidence=float(item.get("confidence", 0.5)),
                    estimated_grams=float(item["estimated_grams"]) if item.get("estimated_grams") else None,
                    estimated_macros=macros,
                    category=category,
                    is_mixed_dish=bool(item.get("is_mixed_dish", False)),
                    visible_components=visible_components,
                )
                foods.append(food)
                
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Failed to parse food item: {e}")
                continue
        
        overall_confidence = float(data.get("overall_confidence", 0.5))
        
        # Determine multi-food status: if we have multiple foods, it's multi-food
        # regardless of what the LLM returns for is_multi_food_plate
        # This ensures we don't miss multi-food detection due to LLM error
        llm_says_multi = bool(data.get("is_multi_food_plate", False))
        is_multi_food_plate = len(foods) > 1 or llm_says_multi
        
        logger.info(
            f"Multi-food detection: foods_count={len(foods)}, "
            f"llm_says_multi={llm_says_multi}, is_multi_food_plate={is_multi_food_plate}"
        )
        
        return foods, overall_confidence, is_multi_food_plate
    
    def _extract_json(self, text: str) -> str | None:
        """Extract JSON object from text response."""
        
        # Try to find JSON in the response
        text = text.strip()
        
        # If it starts with {, try to parse directly
        if text.startswith("{"):
            # Find matching closing brace
            brace_count = 0
            for i, char in enumerate(text):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return text[: i + 1]
        
        # Try to find JSON block in the text
        start = text.find("{")
        if start == -1:
            return None
        
        # Find the matching end brace
        brace_count = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[start : i + 1]
        
        return None
    
    async def recognize_single(
        self,
        image_data: bytes,
        mask_index: int = 0,
    ) -> RecognizedFood | None:
        """
        Recognize a single food item in a cropped image (MVP-2.9).
        
        This uses a simplified prompt optimized for single-item recognition,
        which is more accurate when the image contains only one food item.
        
        Args:
            image_data: Raw cropped image bytes (JPEG or PNG)
            mask_index: Index of the mask this crop came from (for logging)
            
        Returns:
            RecognizedFood if successful, None if recognition fails
        """
        start_time = time.time()
        
        try:
            # Encode image to base64
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            
            # Build request with single-item prompt
            request_body = {
                "model": self.model,
                "prompt": SINGLE_FOOD_RECOGNITION_PROMPT,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.2,  # Lower temp for more consistent results
                    "num_predict": 500,  # Fewer tokens needed for single item
                    "top_p": 0.9,
                },
            }
            
            logger.debug(f"Recognizing single food item (mask {mask_index})")
            
            # Call Ollama API
            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json=request_body,
            )
            
            if response.status_code != 200:
                logger.warning(
                    f"Ollama API error for mask {mask_index}: {response.status_code}"
                )
                return None
            
            result_data = response.json()
            raw_response = result_data.get("response", "")
            
            # Parse the single food response
            food = self._parse_single_response(raw_response)
            
            if food:
                processing_time = int((time.time() - start_time) * 1000)
                logger.info(
                    f"Mask {mask_index}: identified '{food.label}' "
                    f"(conf={food.confidence:.2f}) in {processing_time}ms"
                )
            
            return food
            
        except Exception as e:
            logger.warning(f"Single food recognition failed for mask {mask_index}: {e}")
            return None
    
    def _parse_single_response(self, raw_response: str) -> RecognizedFood | None:
        """Parse LLaVA response for single food item."""
        
        json_str = self._extract_json(raw_response)
        if not json_str:
            logger.warning("Could not extract JSON from single-item response")
            return None
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse single-item JSON: {e}")
            return None
        
        try:
            # Parse macros if present
            macros = None
            if "macros" in data:
                macros = EstimatedMacros(
                    carbs=float(data["macros"].get("carbs", 0)),
                    protein=float(data["macros"].get("protein", 0)),
                    fat=float(data["macros"].get("fat", 0)),
                    fiber=float(data["macros"].get("fiber", 0)),
                )
            
            # Parse category
            category_str = data.get("category", "unknown").lower()
            try:
                category = FoodCategory(category_str)
            except ValueError:
                category = FoodCategory.UNKNOWN
            
            return RecognizedFood(
                label=data.get("label", "Unknown Food"),
                confidence=float(data.get("confidence", 0.5)),
                estimated_grams=float(data["estimated_grams"]) if data.get("estimated_grams") else None,
                estimated_macros=macros,
                category=category,
                is_mixed_dish=bool(data.get("is_mixed_dish", False)),
            )
            
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse single food item: {e}")
            return None

    async def health_check(self) -> bool:
        """Check if Ollama is available and has the required model."""
        try:
            # Check if Ollama is running
            response = await self._client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return False
            
            # Check if the model is available
            tags = response.json()
            models = [m.get("name", "") for m in tags.get("models", [])]
            
            # Check for exact match or partial match (e.g., "llava:7b" in "llava:7b-v1.6")
            model_available = any(
                self.model in m or m.startswith(self.model.split(":")[0])
                for m in models
            )
            
            if not model_available:
                logger.warning(
                    f"Model {self.model} not found. Available: {models}"
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
