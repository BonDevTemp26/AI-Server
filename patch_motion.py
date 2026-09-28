import sys

file_path = "/home/ai-mini-playback/Main Server/cctv-surveillance/backend/api/endpoints/motion.py"
with open(file_path, "r") as f:
    content = f.read()

new_code = """

class TheftCamsRequest(BaseModel):
    cameras: List[int]

_theft_target_cams = []

@router.get("/theft-cams")
def get_theft_cams():
    return {"cameras": _theft_target_cams}

@router.post("/theft-cams")
def set_theft_cams(request: TheftCamsRequest):
    global _theft_target_cams
    _theft_target_cams = request.cameras
    return {"status": "success", "cameras": _theft_target_cams}
"""

if "TheftCamsRequest" not in content:
    content = content.replace("@router.get(\"/active-cameras\")", new_code + "\n@router.get(\"/active-cameras\")")
    with open(file_path, "w") as f:
        f.write(content)
    print("Patched successfully")
else:
    print("Already patched")
