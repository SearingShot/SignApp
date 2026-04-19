import struct
import json

with open("src/sign_app/ui/avatar.glb", "rb") as f:
    f.read(12) # skip magic, version, length
    chunk1_len = struct.unpack('<I', f.read(4))[0]
    chunk1_type = f.read(4)
    if chunk1_type == b'JSON':
        data = json.loads(f.read(chunk1_len).decode('utf-8'))
        found = False
        for mesh in data.get("meshes", []):
            if "extras" in mesh and "targetNames" in mesh["extras"]:
                print(f"Mesh {mesh.get('name')} targets: {mesh['extras']['targetNames']}")
                found = True
        if not found:
            print("No targetNames found in any mesh.")
