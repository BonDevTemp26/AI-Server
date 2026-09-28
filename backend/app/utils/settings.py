import json
import os

SETTINGS_FILE = "/app/uploads/settings.json"

def get_server_mode():
    if not os.path.exists(SETTINGS_FILE):
        return "standalone"
    try:
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
            return data.get("server_mode", "standalone")
    except Exception:
        return "standalone"

def set_server_mode(mode: str):
    if mode not in ["standalone", "cloud"]:
        raise ValueError("Mode must be standalone or cloud")
        
    data = {}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            pass
            
    data["server_mode"] = mode
    
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=4)
        
    return mode
