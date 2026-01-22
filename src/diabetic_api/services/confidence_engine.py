"""
Confidence Engine for Meal Vision scans.

MVP-2.8: Calculates confidence scores and macro ranges based on multiple factors:
- Food recognition confidence (LLaVA)
- Nutrition database match quality (USDA)
- Depth data availability
- Mixed dish detection
"""

import logging
from dataclasses import dataclass
from enum import Enum

from diabetic_api.models.food_scan import (
    MacroRanges,
    MacroSource,
    Macros,
    ScanQuality,
    UncertaintyReason,
)

logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """Confidence level categories."""
    
    HIGH = "high"      # 0.8+ - Ready to log
    MEDIUM = "medium"  # 0.5-0.8 - Review recommended
    LOW = "low"        # <0.5 - Consider re-scanning


@dataclass
class ConfidenceFactors:
    """Individual factors contributing to confidence score."""
    
    recognition_confidence: float  # LLaVA's confidence (0-1)
    usda_match_score: float | None  # USDA search relevance (0-1, None if no lookup)
    has_depth_data: bool  # Whether depth sensor provided data
    is_mixed_dish: bool  # Mixed dishes are harder to estimate
    has_usda_match: bool  # Whether USDA found a match
    portion_confidence: float  # Confidence in gram estimate (0-1)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "recognition": self.recognition_confidence,
            "usda_match": self.usda_match_score,
            "has_depth": self.has_depth_data,
            "is_mixed_dish": self.is_mixed_dish,
            "has_usda": self.has_usda_match,
            "portion": self.portion_confidence,
        }


@dataclass
class ConfidenceResult:
    """Complete confidence assessment result."""
    
    overall_score: float  # Combined confidence (0-1)
    level: ConfidenceLevel  # Categorical level
    scan_quality: ScanQuality  # Quality indicator for UI
    macro_variance: float  # Variance percentage for ranges (e.g., 0.15 = ±15%)
    factors: ConfidenceFactors  # Individual factor breakdown
    uncertainty_reasons: list[UncertaintyReason]  # Reasons for uncertainty


class ConfidenceEngine:
    """
    Engine for calculating scan confidence and macro ranges.
    
    Weights different factors to produce an overall confidence score
    and appropriate macro ranges.
    """
    
    # Factor weights (must sum to 1.0)
    WEIGHT_RECOGNITION = 0.30  # How well LLaVA identified the food
    WEIGHT_USDA_MATCH = 0.25   # Quality of USDA database match
    WEIGHT_DATA_SOURCE = 0.25  # Whether using USDA vs LLaVA-only macros
    WEIGHT_PORTION = 0.15      # Confidence in portion/gram estimate
    WEIGHT_DEPTH = 0.05        # Whether depth data available
    
    # Variance ranges based on confidence
    VARIANCE_HIGH = 0.10   # ±10% for high confidence
    VARIANCE_MEDIUM = 0.20  # ±20% for medium confidence
    VARIANCE_LOW = 0.35     # ±35% for low confidence
    
    def calculate_confidence(
        self,
        recognition_confidence: float,
        usda_match_score: float | None,
        has_usda_match: bool,
        macro_source: MacroSource,
        has_depth_data: bool,
        is_mixed_dish: bool,
        estimated_grams: float | None,
    ) -> ConfidenceResult:
        """
        Calculate overall confidence score and macro ranges.
        
        Args:
            recognition_confidence: LLaVA's confidence (0-1)
            usda_match_score: USDA search match score (0-1, None if no lookup)
            has_usda_match: Whether USDA found a matching food
            macro_source: Source of primary macro data
            has_depth_data: Whether depth sensor data was available
            is_mixed_dish: Whether the food is a mixed/composite dish
            estimated_grams: Estimated weight in grams
            
        Returns:
            ConfidenceResult with overall score, ranges, and factors
        """
        # Calculate portion confidence based on data availability
        portion_confidence = self._calculate_portion_confidence(
            has_depth_data, estimated_grams, is_mixed_dish
        )
        
        # Build factors
        factors = ConfidenceFactors(
            recognition_confidence=recognition_confidence,
            usda_match_score=usda_match_score,
            has_depth_data=has_depth_data,
            is_mixed_dish=is_mixed_dish,
            has_usda_match=has_usda_match,
            portion_confidence=portion_confidence,
        )
        
        # Calculate weighted score
        overall_score = self._calculate_weighted_score(factors, macro_source)
        
        # Determine level and quality
        level = self._score_to_level(overall_score)
        scan_quality = self._level_to_quality(level)
        
        # Calculate variance for macro ranges
        macro_variance = self._calculate_variance(overall_score, macro_source)
        
        # Collect uncertainty reasons
        uncertainty_reasons = self._collect_uncertainty_reasons(factors, macro_source)
        
        logger.debug(
            f"Confidence calculated: {overall_score:.2f} ({level.value}), "
            f"variance=±{macro_variance*100:.0f}%"
        )
        
        return ConfidenceResult(
            overall_score=overall_score,
            level=level,
            scan_quality=scan_quality,
            macro_variance=macro_variance,
            factors=factors,
            uncertainty_reasons=uncertainty_reasons,
        )
    
    def calculate_macro_ranges(
        self,
        macros: Macros,
        variance: float,
    ) -> MacroRanges:
        """
        Calculate P10-P90 confidence ranges for macros.
        
        Args:
            macros: Base macro values
            variance: Variance percentage (e.g., 0.15 for ±15%)
            
        Returns:
            MacroRanges with P10 and P90 values
        """
        return MacroRanges(
            carbs_p10=max(0, macros.carbs * (1 - variance)),
            carbs_p90=macros.carbs * (1 + variance),
            protein_p10=max(0, macros.protein * (1 - variance)),
            protein_p90=macros.protein * (1 + variance),
            fat_p10=max(0, macros.fat * (1 - variance)),
            fat_p90=macros.fat * (1 + variance),
            fiber_p10=max(0, macros.fiber * (1 - variance)),
            fiber_p90=macros.fiber * (1 + variance),
        )
    
    def _calculate_portion_confidence(
        self,
        has_depth: bool,
        estimated_grams: float | None,
        is_mixed_dish: bool,
    ) -> float:
        """Calculate confidence in portion/gram estimate."""
        base = 0.5  # Start at 50%
        
        # Depth data significantly improves portion accuracy
        if has_depth:
            base += 0.3
        
        # Having any gram estimate is better than nothing
        if estimated_grams and estimated_grams > 0:
            base += 0.1
        
        # Mixed dishes are harder to portion
        if is_mixed_dish:
            base -= 0.15
        
        return max(0.1, min(1.0, base))
    
    def _calculate_weighted_score(
        self,
        factors: ConfidenceFactors,
        macro_source: MacroSource,
    ) -> float:
        """Calculate weighted overall confidence score."""
        score = 0.0
        
        # Recognition confidence
        score += factors.recognition_confidence * self.WEIGHT_RECOGNITION
        
        # USDA match score (use 0.5 baseline if no lookup attempted)
        usda_score = factors.usda_match_score if factors.usda_match_score is not None else 0.5
        if factors.has_usda_match:
            score += usda_score * self.WEIGHT_USDA_MATCH
        else:
            # Penalty for no USDA match
            score += 0.3 * self.WEIGHT_USDA_MATCH
        
        # Data source reliability
        if macro_source == MacroSource.USDA:
            score += 1.0 * self.WEIGHT_DATA_SOURCE
        elif macro_source == MacroSource.LLAVA:
            score += 0.6 * self.WEIGHT_DATA_SOURCE
        else:
            score += 0.2 * self.WEIGHT_DATA_SOURCE
        
        # Portion confidence
        score += factors.portion_confidence * self.WEIGHT_PORTION
        
        # Depth data bonus
        if factors.has_depth_data:
            score += 1.0 * self.WEIGHT_DEPTH
        else:
            score += 0.5 * self.WEIGHT_DEPTH
        
        # Mixed dish penalty
        if factors.is_mixed_dish:
            score *= 0.85  # 15% reduction for mixed dishes
        
        return max(0.0, min(1.0, score))
    
    def _score_to_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level."""
        if score >= 0.75:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _level_to_quality(self, level: ConfidenceLevel) -> ScanQuality:
        """Convert confidence level to scan quality."""
        if level == ConfidenceLevel.HIGH:
            return ScanQuality.GOOD
        elif level == ConfidenceLevel.MEDIUM:
            return ScanQuality.OK
        else:
            return ScanQuality.POOR
    
    def _calculate_variance(
        self,
        score: float,
        macro_source: MacroSource,
    ) -> float:
        """
        Calculate appropriate variance for macro ranges.
        
        Lower variance = tighter ranges = more confident.
        """
        # Base variance on confidence score
        if score >= 0.75:
            base_variance = self.VARIANCE_HIGH
        elif score >= 0.5:
            base_variance = self.VARIANCE_MEDIUM
        else:
            base_variance = self.VARIANCE_LOW
        
        # USDA data is more reliable, can tighten ranges
        if macro_source == MacroSource.USDA:
            base_variance *= 0.8  # 20% tighter
        elif macro_source == MacroSource.LLAVA:
            base_variance *= 1.1  # 10% wider for AI-only
        
        return min(0.5, base_variance)  # Cap at ±50%
    
    def _collect_uncertainty_reasons(
        self,
        factors: ConfidenceFactors,
        macro_source: MacroSource,
    ) -> list[UncertaintyReason]:
        """Collect applicable uncertainty reasons."""
        reasons = []
        
        if factors.recognition_confidence < 0.7:
            reasons.append(UncertaintyReason.LOW_RECOGNITION_CONFIDENCE)
        
        if not factors.has_depth_data:
            reasons.append(UncertaintyReason.NO_DEPTH_DATA)
        
        if not factors.has_usda_match and macro_source != MacroSource.USDA:
            reasons.append(UncertaintyReason.WEAK_FOOD_MAPPING)
        
        if factors.is_mixed_dish:
            reasons.append(UncertaintyReason.MIXED_DISH)
        
        return reasons


# Singleton instance
_engine: ConfidenceEngine | None = None


def get_confidence_engine() -> ConfidenceEngine:
    """Get the confidence engine singleton."""
    global _engine
    if _engine is None:
        _engine = ConfidenceEngine()
    return _engine
