from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
import pandas as pd
from io import BytesIO
from app.database import get_db
from app.models.all_models import Shop, Camera

router = APIRouter(prefix="/api/bulk-insert", tags=["Bulk Insert"])

@router.post("/")
async def bulk_insert(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        raise HTTPException(status_code=400, detail="Only .csv and .xlsx files are supported")
        
    contents = await file.read()
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(contents))
        else:
            df = pd.read_excel(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")
        
    required_cols_v1 = {'shop_name', 'camera_name', 'rtsp_url'}
    required_cols_v2 = {'shop_name', 'display_name', 'rtsp_url'}
    
    col_set = set(df.columns)
    cam_col = None
    if required_cols_v1.issubset(col_set):
        cam_col = 'camera_name'
    elif required_cols_v2.issubset(col_set):
        cam_col = 'display_name'
    else:
        raise HTTPException(status_code=400, detail="File must contain columns: (shop_name, camera_name or display_name, rtsp_url)")
        
    stats = {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}
    
    for index, row in df.iterrows():
        shop_name = str(row['shop_name']).strip()
        camera_name = str(row[cam_col]).strip()
        rtsp_url = str(row['rtsp_url']).strip()
        
        if not shop_name or not camera_name or not rtsp_url or shop_name == 'nan' or camera_name == 'nan':
            stats["skipped"] += 1
            continue
            
        try:
            # Upsert Shop
            shop = db.query(Shop).filter(Shop.name == shop_name).first()
            if not shop:
                shop = Shop(name=shop_name)
                db.add(shop)
                db.commit()
                db.refresh(shop)
                
            # Upsert Camera
            camera = db.query(Camera).filter(Camera.shop_id == shop.id, Camera.name == camera_name).first()
            if not camera:
                camera = Camera(shop_id=shop.id, name=camera_name, rtsp_url=rtsp_url)
                db.add(camera)
                stats["inserted"] += 1
            else:
                if camera.rtsp_url != rtsp_url:
                    camera.rtsp_url = rtsp_url
                    stats["updated"] += 1
                else:
                    stats["skipped"] += 1
            db.commit()
        except Exception as e:
            db.rollback()
            stats["errors"] += 1
            
    return {"message": "Bulk insert processed", "stats": stats}
