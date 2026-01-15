# Meal Vision: Data Schema Documentation

**Ticket:** MVP-0.2  
**Version:** 1.0  
**Last Updated:** 2026-01-15

## Overview

This document describes the MongoDB collections used by the Meal Vision feature. All food scan data is stored **separately** from CareLink pump data to ensure isolation and prevent interference with existing sync workflows.

---

## Collection: `meal_estimates`

Stores confirmed meal estimates from food scans. This is the primary collection for user-facing meal data.

### Schema

```json
{
  "_id": ObjectId,
  "scan_id": "string (required)",
  "user_id": "string (required)",
  "source": "string (enum: vision, manual, barcode)",
  "canonical_food_id": "string (required)",
  "food_label": "string (required)",
  "macros": {
    "carbs": "float (grams)",
    "protein": "float (grams)",
    "fat": "float (grams)",
    "fiber": "float (grams)"
  },
  "macro_ranges": {
    "carbs_p10": "float (optional)",
    "carbs_p90": "float (optional)",
    "protein_p10": "float (optional)",
    "protein_p90": "float (optional)",
    "fat_p10": "float (optional)",
    "fat_p90": "float (optional)",
    "fiber_p10": "float (optional)",
    "fiber_p90": "float (optional)"
  },
  "confidence": "float (0-1, required)",
  "uncertainty_reasons": ["string array"],
  "user_overrides": {
    "corrected_macros": { "...": "optional" },
    "selected_food_id": "string (optional)",
    "notes": "string (optional)"
  },
  "created_at": "datetime (required)",
  "updated_at": "datetime (optional)"
}
```

### Example Document

```json
{
  "_id": { "$oid": "679f8a1b2c3d4e5f67890abc" },
  "scan_id": "scan_abc123def456",
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
  "created_at": { "$date": "2026-01-15T12:30:05.000Z" },
  "updated_at": null
}
```

### Indexes

```javascript
// Primary queries
db.meal_estimates.createIndex({ "user_id": 1, "created_at": -1 })
db.meal_estimates.createIndex({ "scan_id": 1 }, { unique: true })

// Timeline queries
db.meal_estimates.createIndex({ "user_id": 1, "created_at": 1 })
```

---

## Collection: `food_scans` (Future - MVP-2.2)

Stores raw scan metadata for debugging and analytics. Only populated when `opt_in_store_artifacts=true`.

### Schema (Preview)

```json
{
  "_id": ObjectId,
  "scan_id": "string (required, unique)",
  "user_id": "string (required)",
  "device": {
    "platform": "string",
    "model": "string",
    "os_version": "string",
    "depth_sensor": "string"
  },
  "intrinsics": {
    "fx": "float",
    "fy": "float",
    "cx": "float",
    "cy": "float",
    "width": "int",
    "height": "int"
  },
  "scan_quality": "string (enum: good, ok, poor)",
  "processing_time_ms": "int",
  "created_at": "datetime",
  "ttl_expires_at": "datetime (TTL index)"
}
```

---

## Collection: `scan_artifacts` (Future - MVP-2.2)

Stores binary artifacts (images, depth maps) with TTL expiration. Only populated when `opt_in_store_artifacts=true`.

### Schema (Preview)

```json
{
  "_id": ObjectId,
  "scan_id": "string (required)",
  "artifact_type": "string (enum: rgb, depth_u16, confidence_u8, segmentation_mask)",
  "storage_uri": "string (GridFS or blob storage URI)",
  "size_bytes": "int",
  "created_at": "datetime",
  "ttl_expires_at": "datetime (TTL index, default 7 days)"
}
```

---

## Data Architecture Summary

| Collection | Purpose | Source | Isolation |
|------------|---------|--------|-----------|
| `pump_data` | CareLink glucose/bolus/basal data | CareLink sync | Existing, unchanged |
| `meal_estimates` | Confirmed food scan results | Meal Vision | **New, separate** |
| `food_scans` | Scan metadata (opt-in) | Meal Vision | **New, separate** |
| `scan_artifacts` | Binary artifacts (opt-in) | Meal Vision | **New, separate** |
| `ChatHistory` | Chat sessions/messages | AI Chat | Existing, unchanged |

---

## Timeline View Integration

The timeline view displays data from **both** collections but queries them separately:

```python
# Pseudo-code for unified timeline
async def get_timeline(user_id: str, start: datetime, end: datetime):
    # Query pump data (existing)
    pump_data = await uow.pump_data.get_boluses(start, end)
    
    # Query meal estimates (new, separate collection)
    meals = await uow.meal_estimates.get_timeline_data(user_id, start, end)
    
    # Merge and sort by timestamp
    timeline = sorted(pump_data + meals, key=lambda x: x["timestamp"])
    return timeline
```

This approach ensures:
- CareLink sync is **not affected** by meal scan data
- Data can be queried independently
- Different retention policies can be applied
- Schema changes don't impact existing data

---

## Source Field Values

| Value | Description |
|-------|-------------|
| `vision` | AI-powered food scan (RGB + depth) |
| `manual` | User manually entered meal (future) |
| `barcode` | Barcode scan lookup (V4) |

---

## Uncertainty Reasons

| Code | Description |
|------|-------------|
| `mixed_dish` | Multiple foods detected |
| `low_segmentation_confidence` | Food segmentation unclear |
| `poor_depth_quality` | Depth map quality issues |
| `insufficient_table_visible` | Table plane fit failed |
| `weak_food_mapping` | Low food ID confidence |
| `unknown_food` | Food not in database |
| `partial_occlusion` | Food partially hidden |
| `low_lighting` | Image too dark |
| `motion_blur` | Image blurry |

---

## Related Documentation

- [Food Scan API Contract](food_scan_api_contract.md) - API request/response schemas
- [README - Meal Vision](../README.md#meal-vision-food-scan-feature) - MVP constraints
