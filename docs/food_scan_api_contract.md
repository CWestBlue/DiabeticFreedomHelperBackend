# Meal Vision: Food Scan API Contract v1.0

**Ticket:** MVP-0.1  
**Version:** 1.0  
**Last Updated:** 2026-01-15

## Overview

The Food Scan API enables nutritional estimation from food images using RGB + depth data captured from mobile devices with ARCore (Android) or ARKit (iOS) support.

### MVP Constraints

- Food must be on a **plate/table**
- **Table surface must be visible** in the frame
- **Plate must be fully in frame**
- Single-frame capture only (multi-frame fusion is V2)

---

## Endpoint

```
POST /food/scan
Content-Type: multipart/form-data
```

---

## Request

### Multipart Form Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `rgb` | File (JPEG) | Yes | RGB image from camera |
| `depth_u16` | File (PNG) | Yes | 16-bit depth map in millimeters |
| `confidence_u8` | File (PNG) | No | 8-bit confidence map (0-255) |
| `metadata` | JSON string | Yes | Request metadata (see below) |

### Metadata JSON Schema

```json
{
  "intrinsics": {
    "fx": 1000.0,
    "fy": 1000.0,
    "cx": 540.0,
    "cy": 960.0,
    "width": 1080,
    "height": 1920
  },
  "orientation": {
    "pitch": -45.0,
    "roll": 0.0,
    "yaw": 0.0
  },
  "device": {
    "platform": "android",
    "model": "Pixel 9",
    "os_version": "15",
    "depth_sensor": "arcore_raw"
  },
  "scan_version": "1.0",
  "user_id": "user_123",
  "client_timestamp": "2026-01-15T12:30:00Z",
  "opt_in_store_artifacts": false
}
```

### Field Definitions

#### Camera Intrinsics
| Field | Type | Description |
|-------|------|-------------|
| `fx` | float | Focal length X in pixels |
| `fy` | float | Focal length Y in pixels |
| `cx` | float | Principal point X in pixels |
| `cy` | float | Principal point Y in pixels |
| `width` | int | Image width in pixels |
| `height` | int | Image height in pixels |

#### Device Orientation
| Field | Type | Description |
|-------|------|-------------|
| `pitch` | float | Pitch angle in degrees |
| `roll` | float | Roll angle in degrees |
| `yaw` | float | Yaw angle in degrees |

#### Device Info
| Field | Type | Description |
|-------|------|-------------|
| `platform` | string | `"android"` or `"ios"` |
| `model` | string | Device model (e.g., `"Pixel 9"`) |
| `os_version` | string | OS version string |
| `depth_sensor` | string | Depth sensor type (e.g., `"arcore_raw"`, `"arcore_smoothed"`) |

---

## Response

### Success Response (200 OK)

```json
{
  "scan_id": "scan_abc123def456",
  "food_candidates": [
    {
      "canonical_food_id": "usda_12345",
      "label": "Grilled Chicken Breast",
      "probability": 0.85,
      "is_mixed_dish": false
    },
    {
      "canonical_food_id": "usda_12346",
      "label": "Roasted Turkey Breast",
      "probability": 0.10,
      "is_mixed_dish": false
    }
  ],
  "selected_food": {
    "canonical_food_id": "usda_12345",
    "label": "Grilled Chicken Breast",
    "probability": 0.85,
    "is_mixed_dish": false
  },
  "volume_ml": 180.5,
  "grams_est": 150.0,
  "macros": {
    "carbs": 0.0,
    "protein": 31.0,
    "fat": 3.6,
    "fiber": 0.0
  },
  "macro_ranges": {
    "carbs_p10": 0.0,
    "carbs_p90": 2.0,
    "protein_p10": 25.0,
    "protein_p90": 38.0,
    "fat_p10": 2.0,
    "fat_p90": 6.0,
    "fiber_p10": 0.0,
    "fiber_p90": 0.5
  },
  "confidence_score": 0.82,
  "scan_quality": "good",
  "uncertainty_reasons": [],
  "debug": null,
  "processed_at": "2026-01-15T12:30:05Z"
}
```

### Response Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `scan_id` | string | Unique identifier for this scan |
| `food_candidates` | array | Ranked list of food candidates (1-10) |
| `selected_food` | object | Top/selected food candidate |
| `volume_ml` | float | Estimated volume in milliliters |
| `grams_est` | float | Estimated weight in grams |
| `macros` | object | Estimated macronutrients (carbs, protein, fat, fiber in grams) |
| `macro_ranges` | object | P10-P90 confidence ranges for each macro |
| `confidence_score` | float | Overall confidence (0-1) |
| `scan_quality` | string | `"good"`, `"ok"`, or `"poor"` |
| `uncertainty_reasons` | array | List of uncertainty reason codes |
| `debug` | object | Debug info (only if `opt_in_store_artifacts=true`) |
| `processed_at` | string | Server processing timestamp (ISO 8601) |

### Scan Quality Enum
| Value | Description |
|-------|-------------|
| `good` | High confidence, proceed to confirm |
| `ok` | Moderate confidence, may want to re-scan |
| `poor` | Low confidence, recommend re-scan |

### Uncertainty Reasons Enum
| Code | Description |
|------|-------------|
| `mixed_dish` | Multiple foods detected |
| `low_segmentation_confidence` | Food segmentation unclear |
| `poor_depth_quality` | Depth map has many invalid pixels |
| `insufficient_table_visible` | Not enough table surface for plane fit |
| `weak_food_mapping` | Low confidence in food identification |
| `unknown_food` | Food not in database |
| `partial_occlusion` | Food partially hidden |
| `low_lighting` | Image too dark |
| `motion_blur` | Image blurry |

---

## Error Response (400/500)

```json
{
  "error_code": "NEEDS_TABLE_VISIBLE",
  "message": "Table surface must be visible in the frame for accurate volume estimation",
  "details": {
    "depth_valid_ratio": 0.15,
    "min_required": 0.40
  },
  "scan_id": null
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `NEEDS_TABLE_VISIBLE` | 400 | Table surface not visible enough |
| `DEPTH_TOO_SPARSE` | 400 | Depth map has too few valid pixels |
| `NO_FOOD_DETECTED` | 400 | No food found in image |
| `INVALID_INTRINSICS` | 400 | Camera intrinsics invalid |
| `IMAGE_TOO_DARK` | 400 | Image too dark for processing |
| `IMAGE_TOO_BLURRY` | 400 | Image too blurry |
| `PLATE_NOT_IN_FRAME` | 400 | Plate not fully visible |
| `PROCESSING_ERROR` | 500 | Server processing error |

---

## Example curl Commands

### Basic Scan Request

```bash
curl -X POST "https://api.example.com/food/scan" \
  -F "rgb=@food_image.jpg" \
  -F "depth_u16=@depth_map.png" \
  -F 'metadata={
    "intrinsics": {
      "fx": 1000.0,
      "fy": 1000.0,
      "cx": 540.0,
      "cy": 960.0,
      "width": 1080,
      "height": 1920
    },
    "orientation": {
      "pitch": -45.0,
      "roll": 0.0,
      "yaw": 0.0
    },
    "device": {
      "platform": "android",
      "model": "Pixel 9",
      "os_version": "15",
      "depth_sensor": "arcore_raw"
    },
    "scan_version": "1.0",
    "user_id": "user_123",
    "client_timestamp": "2026-01-15T12:30:00Z",
    "opt_in_store_artifacts": false
  }'
```

### With Confidence Map and Debug Opt-in

```bash
curl -X POST "https://api.example.com/food/scan" \
  -F "rgb=@food_image.jpg" \
  -F "depth_u16=@depth_map.png" \
  -F "confidence_u8=@confidence_map.png" \
  -F 'metadata={
    "intrinsics": {
      "fx": 1000.0,
      "fy": 1000.0,
      "cx": 540.0,
      "cy": 960.0,
      "width": 1080,
      "height": 1920
    },
    "orientation": {
      "pitch": -45.0,
      "roll": 0.0,
      "yaw": 0.0
    },
    "device": {
      "platform": "android",
      "model": "Pixel 9",
      "os_version": "15",
      "depth_sensor": "arcore_raw"
    },
    "scan_version": "1.0",
    "user_id": "user_123",
    "client_timestamp": "2026-01-15T12:30:00Z",
    "opt_in_store_artifacts": true
  }'
```

### Local Development

```bash
curl -X POST "http://localhost:8000/food/scan" \
  -F "rgb=@test_food.jpg" \
  -F "depth_u16=@test_depth.png" \
  -F 'metadata={"intrinsics":{"fx":1000,"fy":1000,"cx":540,"cy":960,"width":1080,"height":1920},"orientation":{"pitch":-45,"roll":0,"yaw":0},"device":{"platform":"android","model":"Pixel 9","os_version":"15"},"scan_version":"1.0","user_id":"test_user","client_timestamp":"2026-01-15T12:30:00Z","opt_in_store_artifacts":false}'
```

---

## MealEstimate Storage Schema

Confirmed scans are stored in the `meal_estimates` collection (separate from `pump_data`).

```json
{
  "_id": "meal_xyz789",
  "scan_id": "scan_abc123",
  "user_id": "user_123",
  "source": "vision",
  "canonical_food_id": "usda_12345",
  "food_label": "Grilled Chicken Breast",
  "macros": {
    "carbs": 0.0,
    "protein": 31.0,
    "fat": 3.6,
    "fiber": 0.0
  },
  "macro_ranges": {
    "carbs_p10": 0.0,
    "carbs_p90": 2.0,
    "protein_p10": 25.0,
    "protein_p90": 38.0,
    "fat_p10": 2.0,
    "fat_p90": 6.0,
    "fiber_p10": 0.0,
    "fiber_p90": 0.5
  },
  "confidence": 0.82,
  "uncertainty_reasons": [],
  "user_overrides": null,
  "created_at": "2026-01-15T12:30:05Z",
  "updated_at": null
}
```

---

## Versioning

The API uses `scan_version` in requests to track contract versions:

| Version | Status | Changes |
|---------|--------|---------|
| `1.0` | Current | Initial MVP release |

Future versions will maintain backward compatibility where possible.

---

## Related Tickets

- **MVP-0.1**: This contract definition
- **MVP-0.2**: MealEstimate object + storage schema
- **MVP-2.1**: Endpoint implementation
- **MVP-2.3-2.8**: ML pipeline implementation
