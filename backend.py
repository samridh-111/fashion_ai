"""
Enhanced FastAPI Backend
Complete outfit recommendation system with ML, feedback, and scoring
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import os
import uuid
import json
from pathlib import Path

from ml_features import FeatureExtractor, ClothingItem
from recommendation_engine import RecommendationEngine

# Initialize FastAPI
app = FastAPI(title="AI Fashion Recommender")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
DATA_DIR = "user_clothes"
TOP_DIR = os.path.join(DATA_DIR, "tops")
BOTTOM_DIR = os.path.join(DATA_DIR, "bottoms")
METADATA_FILE = "wardrobe_metadata.json"

# Create directories
os.makedirs(TOP_DIR, exist_ok=True)
os.makedirs(BOTTOM_DIR, exist_ok=True)

# Serve uploaded images
app.mount("/images", StaticFiles(directory=DATA_DIR), name="images")

# Initialize ML components
print("Initializing ML components...")
feature_extractor = FeatureExtractor()
recommendation_engine = RecommendationEngine(feature_extractor)

# In-memory wardrobe storage
wardrobe = {
    'tops': {},
    'bottoms': {}
}


def load_wardrobe():
    """Load wardrobe from disk"""
    global wardrobe

    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)

            # Reconstruct clothing items
            for category in ['tops', 'bottoms']:
                for item_data in metadata.get(category, []):
                    item_path = item_data['path']
                    if os.path.exists(item_path):
                        # Re-extract features
                        features = feature_extractor.extract_all_features(item_path)
                        if features:
                            item = ClothingItem.from_dict(item_data, features)
                            wardrobe[category][item.id] = item

            print(f"Loaded wardrobe: {len(wardrobe['tops'])} tops, {len(wardrobe['bottoms'])} bottoms")
        except Exception as e:
            print(f"Error loading wardrobe: {e}")


def save_wardrobe():
    """Save wardrobe metadata to disk"""
    try:
        metadata = {
            'tops': [item.to_dict() for item in wardrobe['tops'].values()],
            'bottoms': [item.to_dict() for item in wardrobe['bottoms'].values()]
        }
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error saving wardrobe: {e}")


# Load existing wardrobe on startup
load_wardrobe()


# Pydantic models
class RecommendationRequest(BaseModel):
    occasion: str
    num_results: Optional[int] = 5


class FeedbackRequest(BaseModel):
    top_id: str
    bottom_id: str
    liked: bool
    occasion: str


class AlternativeRequest(BaseModel):
    rejected_top_id: str
    rejected_bottom_id: str
    occasion: str


# API Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend"""
    frontend_path = Path(__file__).parent / "frontend.html"
    if frontend_path.exists():
        return HTMLResponse(content=frontend_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "online",
        "wardrobe": {
            "tops": len(wardrobe['tops']),
            "bottoms": len(wardrobe['bottoms'])
        }
    }


@app.post("/api/upload")
async def upload_clothing(
    file: UploadFile = File(...),
    category: str = Form(...)
):
    """
    Upload a clothing item

    Args:
        file: Image file
        category: 'top' or 'bottom'
    """
    if category not in ['top', 'bottom']:
        raise HTTPException(
            status_code=400,
            detail="Category must be 'top' or 'bottom'"
        )

    # Generate unique ID
    item_id = str(uuid.uuid4())

    # Save file
    save_dir = TOP_DIR if category == 'top' else BOTTOM_DIR
    file_extension = os.path.splitext(file.filename)[1]
    save_filename = f"{item_id}{file_extension}"
    save_path = os.path.join(save_dir, save_filename)

    try:
        # Write file
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Extract features
        features = feature_extractor.extract_all_features(save_path)

        if features is None:
            os.remove(save_path)
            raise HTTPException(
                status_code=400,
                detail="Failed to extract features from image"
            )

        # Create clothing item
        item = ClothingItem(
            id=item_id,
            filename=save_filename,
            path=save_path,
            category=category,
            features=features
        )

        # Add to wardrobe
        category_plural = category + 's'
        wardrobe[category_plural][item_id] = item

        # Save metadata
        save_wardrobe()

        return {
            "success": True,
            "item_id": item_id,
            "filename": save_filename,
            "category": category,
            "message": f"{category.capitalize()} uploaded successfully!"
        }

    except Exception as e:
        # Clean up file if it exists
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/wardrobe")
async def get_wardrobe():
    """Get all wardrobe items"""
    return {
        "tops": [
            {
                **item.to_dict(),
                "image_url": f"/images/tops/{item.filename}"
            }
            for item in wardrobe['tops'].values()
        ],
        "bottoms": [
            {
                **item.to_dict(),
                "image_url": f"/images/bottoms/{item.filename}"
            }
            for item in wardrobe['bottoms'].values()
        ]
    }


@app.delete("/api/wardrobe/{item_id}")
async def delete_item(item_id: str):
    """Delete a wardrobe item"""
    # Find and delete item
    for category in ['tops', 'bottoms']:
        if item_id in wardrobe[category]:
            item = wardrobe[category][item_id]

            # Delete file
            if os.path.exists(item.path):
                os.remove(item.path)

            # Remove from wardrobe
            del wardrobe[category][item_id]

            # Save metadata
            save_wardrobe()

            return {"success": True, "message": "Item deleted"}

    raise HTTPException(status_code=404, detail="Item not found")


@app.post("/api/recommend")
async def recommend_outfits(request: RecommendationRequest):
    """
    Get outfit recommendations

    Args:
        occasion: Description of the occasion
        num_results: Number of recommendations (default: 5)
    """
    if not wardrobe['tops'] or not wardrobe['bottoms']:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one top and one bottom"
        )

    # Get recommendations
    tops_list = list(wardrobe['tops'].values())
    bottoms_list = list(wardrobe['bottoms'].values())

    recommendations = recommendation_engine.recommend_outfits(
        tops=tops_list,
        bottoms=bottoms_list,
        occasion=request.occasion,
        num_results=request.num_results,
        exclude_rejected=True
    )

    if not recommendations:
        return {
            "recommendations": [],
            "message": "No suitable outfits found. Try uploading more clothes or adjusting your occasion."
        }

    # Format response
    result = []
    for rec in recommendations:
        result.append({
            "top": {
                **rec.top.to_dict(),
                "image_url": f"/images/tops/{rec.top.filename}"
            },
            "bottom": {
                **rec.bottom.to_dict(),
                "image_url": f"/images/bottoms/{rec.bottom.filename}"
            },
            "match_score": rec.percentage,
            "explanation": rec.explanation,
            "score_breakdown": rec.breakdown
        })

    return {
        "occasion": request.occasion,
        "recommendations": result
    }


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback on an outfit

    Args:
        top_id: ID of the top
        bottom_id: ID of the bottom
        liked: True if liked, False if rejected
        occasion: The occasion context
    """
    # Validate IDs
    if request.top_id not in wardrobe['tops']:
        raise HTTPException(status_code=404, detail="Top not found")
    if request.bottom_id not in wardrobe['bottoms']:
        raise HTTPException(status_code=404, detail="Bottom not found")

    # Record feedback
    if request.liked:
        recommendation_engine.record_like(
            request.top_id,
            request.bottom_id,
            request.occasion
        )
        message = "Thanks for the feedback! We'll show you more outfits like this."
    else:
        recommendation_engine.feedback_system.record_feedback(
            request.top_id,
            request.bottom_id,
            liked=False,
            occasion=request.occasion
        )
        message = "Noted! We'll avoid this combination in the future."

    return {
        "success": True,
        "message": message
    }


@app.post("/api/alternative")
async def get_alternative(request: AlternativeRequest):
    """
    Get alternative outfit after rejection

    Args:
        rejected_top_id: ID of rejected top
        rejected_bottom_id: ID of rejected bottom
        occasion: The occasion context
    """
    tops_list = list(wardrobe['tops'].values())
    bottoms_list = list(wardrobe['bottoms'].values())

    alternative = recommendation_engine.get_alternative_outfit(
        rejected_top_id=request.rejected_top_id,
        rejected_bottom_id=request.rejected_bottom_id,
        tops=tops_list,
        bottoms=bottoms_list,
        occasion=request.occasion
    )

    if not alternative:
        raise HTTPException(
            status_code=404,
            detail="No alternative outfits available. Try uploading more clothes!"
        )

    return {
        "top": {
            **alternative.top.to_dict(),
            "image_url": f"/images/tops/{alternative.top.filename}"
        },
        "bottom": {
            **alternative.bottom.to_dict(),
            "image_url": f"/images/bottoms/{alternative.bottom.filename}"
        },
        "match_score": alternative.percentage,
        "explanation": alternative.explanation,
        "score_breakdown": alternative.breakdown
    }


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    feedback_system = recommendation_engine.feedback_system

    return {
        "wardrobe": {
            "tops": len(wardrobe['tops']),
            "bottoms": len(wardrobe['bottoms']),
            "total_items": len(wardrobe['tops']) + len(wardrobe['bottoms'])
        },
        "feedback": {
            "total_feedback": len(feedback_system.feedback_history),
            "liked_pairs": len(feedback_system.liked_pairs),
            "rejected_pairs": len(feedback_system.rejected_pairs)
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
