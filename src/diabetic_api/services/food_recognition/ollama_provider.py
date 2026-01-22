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


# Prompt for food recognition
FOOD_RECOGNITION_PROMPT = """You are a food identification expert. Analyze this food image carefully.

IMPORTANT ANALYSIS STEPS:
1. Look for ANY visible text, logos, or brand names on packaging
2. Look for nutrition labels or product descriptions
3. Consider the shape, texture, and color

Provide EXACTLY 3 possible identifications ranked by confidence.

For each food item provide:
1. Food name - BE SPECIFIC! Include brand if visible (e.g., "Quest protein bar" not just "chocolate bar")
2. Estimated portion size in grams
3. Confidence level (0.0 to 1.0)
4. Food category (protein, carbohydrate, vegetable, fruit, dairy, fat, beverage, mixed)
5. Macronutrients per YOUR estimated portion (DIFFERENT for each food):
   - Carbohydrates (grams)
   - Protein (grams)
   - Fat (grams)
   - Fiber (grams)

Respond ONLY with valid JSON:
{
  "foods": [
    {
      "label": "protein bar",
      "confidence": 0.80,
      "estimated_grams": 60,
      "category": "protein",
      "is_mixed_dish": false,
      "macros": {"carbs": 22, "protein": 20, "fat": 8, "fiber": 5}
    },
    {
      "label": "chocolate candy bar",
      "confidence": 0.35,
      "estimated_grams": 50,
      "category": "carbohydrate",
      "is_mixed_dish": false,
      "macros": {"carbs": 35, "protein": 3, "fat": 14, "fiber": 1}
    },
    {
      "label": "granola bar",
      "confidence": 0.20,
      "estimated_grams": 40,
      "category": "carbohydrate",
      "is_mixed_dish": false,
      "macros": {"carbs": 25, "protein": 4, "fat": 6, "fiber": 2}
    }
  ],
  "overall_confidence": 0.80
}

Rules:
- ALWAYS return EXACTLY 3 foods with DIFFERENT macros
- Read any visible text/branding carefully
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
            
            # Parse JSON from response
            foods, overall_confidence = self._parse_response(raw_response)
            
            # Limit to max_foods
            foods = foods[:max_foods]
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return FoodRecognitionResult(
                foods=foods,
                overall_confidence=overall_confidence,
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
    ) -> tuple[list[RecognizedFood], float]:
        """Parse LLaVA response into structured food data."""
        
        # Try to extract JSON from response
        json_str = self._extract_json(raw_response)
        
        if not json_str:
            logger.warning("Could not extract JSON from response")
            return [], 0.0
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return [], 0.0
        
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
                
                food = RecognizedFood(
                    label=item.get("label", "Unknown Food"),
                    confidence=float(item.get("confidence", 0.5)),
                    estimated_grams=float(item["estimated_grams"]) if item.get("estimated_grams") else None,
                    estimated_macros=macros,
                    category=category,
                    is_mixed_dish=bool(item.get("is_mixed_dish", False)),
                )
                foods.append(food)
                
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Failed to parse food item: {e}")
                continue
        
        overall_confidence = float(data.get("overall_confidence", 0.5))
        
        return foods, overall_confidence
    
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
