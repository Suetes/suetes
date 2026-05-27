import pickle, io, struct

# Read the raw pickle opcodes to find stored key strings without needing numpy
with open("suetes/data/scm_Phoenix_cache.pkl", "rb") as f:
    data = f.read()

# Search for known keys as raw bytes
for key in [b"theta_skt", b"theta_surf", b"land_fraction", b"th_v", b"u", b"v", b"w", b"pi", b"rho", b"eta_dot", b"q"]:
    found = key in data
    print(f"  '{key.decode()}': {'FOUND' if found else 'MISSING'}")
