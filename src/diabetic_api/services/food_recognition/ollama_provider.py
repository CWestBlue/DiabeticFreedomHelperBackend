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
FOOD_RECOGNITION_PROMPT = """You are a food identification expert. Analyze this food image and identify ALL DISTINCT FOODS visible.

ANALYSIS STEPS:
1. Count how many DIFFERENT food items are visible on the plate/table
2. For each distinct food, identify it separately
3. Look for ANY visible text, logos, or brand names on packaging
4. For mixed dishes: identify visible components if possible, OR return as single mixed dish

MULTI-FOOD DETECTION RULES:
- If you see chicken, rice, and broccoli on a plate: return 3 separate food items
- If you see a burger: return as 1 item (mixed dish) unless components are clearly separated
- If you see a stir-fry: try to identify visible ingredients (e.g., chicken pieces, vegetables, rice) OR return as single "stir-fry" with aggregate macros
- Maximum 8 distinct foods per scan

For EACH distinct food provide:
1. Food name - BE SPECIFIC! Include brand if visible
2. Estimated portion size in grams for THAT food only
3. Confidence level (0.0 to 1.0)
4. Food category (protein, carbohydrate, vegetable, fruit, dairy, fat, beverage, mixed)
5. Whether this item is a mixed dish (true/false)
6. If mixed dish: list visible_components if identifiable
7. Macronutrients for THAT food's portion:
   - Carbohydrates (grams)
   - Protein (grams)
   - Fat (grams)
   - Fiber (grams)

Respond ONLY with valid JSON:
{
  "foods": [
    {
      "label": "grilled chicken breast",
      "confidence": 0.85,
      "estimated_grams": 150,
      "category": "protein",
      "is_mixed_dish": false,
      "visible_components": null,
      "macros": {"carbs": 0, "protein": 35, "fat": 4, "fiber": 0}
    },
    {
      "label": "steamed white rice",
      "confidence": 0.90,
      "estimated_grams": 200,
      "category": "carbohydrate",
      "is_mixed_dish": false,
      "visible_components": null,
      "macros": {"carbs": 45, "protein": 4, "fat": 0, "fiber": 1}
    },
    {
      "label": "steamed broccoli",
      "confidence": 0.88,
      "estimated_grams": 100,
      "category": "vegetable",
      "is_mixed_dish": false,
      "visible_components": null,
      "macros": {"carbs": 7, "protein": 3, "fat": 0, "fiber": 3}
    }
  ],
  "is_multi_food_plate": true,
  "overall_confidence": 0.85
}

Example for MIXED DISH with decomposition:
{
  "foods": [
    {
      "label": "chicken stir-fry",
      "confidence": 0.75,
      "estimated_grams": 350,
      "category": "mixed",
      "is_mixed_dish": true,
      "visible_components": ["chicken pieces", "bell peppers", "onions", "sauce"],
      "macros": {"carbs": 15, "protein": 30, "fat": 12, "fiber": 3}
    }
  ],
  "is_multi_food_plate": false,
  "overall_confidence": 0.75
}

Rules:
- Return 1-8 foods based on what you ACTUALLY SEE
- Each food gets its own macros based on its visible portion
- Do NOT make up foods that aren't visible
- Sort by confidence (highest first)
- Be realistic with portion estimates

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
        is_multi_food_plate = bool(data.get("is_multi_food_plate", len(foods) > 1))
        
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
