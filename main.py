import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import ee

app = FastAPI(
    title="Sri Lanka Urban Heat Island (UHI) API",
    description="Remote sensing API calculating Land Surface Temperature (LST) and Urban Heat Intensity for Sri Lankan cities.",
    version="1.0.0",
    author="bihanduyevin"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google Earth Engine
@app.on_event("startup")
def initialize_gee():
    try:
        # Use Service Account or default credentials in cloud environments
        ee.Initialize(project="your-gee-project-id") 
    except Exception as e:
        print(f"GEE Initialization Warning: {e}")

# Default coordinates for Sri Lankan urban centers
CITIES = {
    "colombo": {"lat": 6.9271, "lon": 79.8612},
    "kandy": {"lat": 7.2906, "lon": 80.6337},
    "jaffna": {"lat": 9.6615, "lon": 80.0255},
    "galle": {"lat": 6.0535, "lon": 80.2210}
}

def get_lst_image(lon: float, lat: float, start_date: str, end_date: str):
    """Calculates Land Surface Temperature (LST) in Celsius from Landsat 8/9."""
    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(10000) # 10km urban area buffer

    # Filter Landsat 8 Collection 2 Level 2 Surface Temperature
    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
    )

    if collection.size().getInfo() == 0:
        raise ValueError("No low-cloud Landsat scenes found for the selected date range.")

    image = collection.median()

    # Band 10 is Thermal Infrared (ST_B10)
    # Convert DN to Kelvin, then to Celsius: LST = (DN * 0.00341802 + 149.0) - 273.15
    thermal = image.select("ST_B10")
    lst_celsius = thermal.multiply(0.00341802).add(149.0).subtract(273.15).rename("LST")

    return lst_celsius, roi

@app.get("/")
def home():
    return {
        "project": "Sri Lanka Urban Heat Island (UHI) Remote Sensing API",
        "developer": "bihanduyevin",
        "endpoints": ["/api/v1/uhi/city", "/api/v1/uhi/point"]
    }

@app.get("/api/v1/uhi/city")
def get_city_uhi(
    city: str = Query("colombo", description="City name: colombo, kandy, jaffna, galle"),
    start_date: str = Query("2023-01-01", description="Format: YYYY-MM-DD"),
    end_date: str = Query("2023-12-31", description="Format: YYYY-MM-DD")
):
    """Get Mean Land Surface Temperature for major cities in Sri Lanka."""
    city_key = city.lower()
    if city_key not in CITIES:
        raise HTTPException(status_code=400, detail=f"City not supported. Choose from {list(CITIES.keys())}")

    coords = CITIES[city_key]
    try:
        lst_img, roi = get_lst_image(coords["lon"], coords["lat"], start_date, end_date)
        
        # Calculate spatial statistics across the city area
        stats = lst_img.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                reducer2=ee.Reducer.minMax(),
                sharedInputs=True
            ),
            geometry=roi,
            scale=30,
            maxPixels=1e9
        ).getInfo()

        return {
            "city": city,
            "coordinates": coords,
            "period": {"start": start_date, "end": end_date},
            "metrics": {
                "mean_lst_celsius": round(stats.get("LST_mean", 0), 2),
                "max_lst_celsius": round(stats.get("LST_max", 0), 2),
                "min_lst_celsius": round(stats.get("LST_min", 0), 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
