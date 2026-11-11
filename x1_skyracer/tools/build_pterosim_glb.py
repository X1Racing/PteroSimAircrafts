#!/usr/bin/env python3
"""Adapt the X1 SkyRacer web GLB to PteroSim meshes.

Source of geometry (not rewritten here): projectmaximumboost/x1skyracer
revision 35396c71a6f90cea8ef162ffbb69d950aacc64f1, file
X1_SkyRacer_FanTilt/X1_SkyRacer_FanTilt_web.glb.

PteroSim loads glTF as vehicle (x, y, z) = (-z, x, y): X forward, Y right, Z up.
This script bakes the flight pose into that frame, with the nose along glTF -Z,
and turns shells that face inward so a single-sided renderer shows the exterior.
Glass stays double-sided because the canopy is a sheet seen from both sides.
The cockpit screen keeps its 0..1 UVs. Eye_L, Eye_R, Screen_Centre and Stick_Pivot
stay empty nodes.

The October flight-model kit exporter (01e7aee) omitted the interior and is not
the geometry source. Tanks and the engine placeholder stay out of the visual,
as they did in that kit; they are not part of the simulator cockpit.
"""
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

SOURCE_SHA256 = "675e8a3378e750a610d1f5116bf43449d938db87b3e9c9113cc8c37262d9b1ad"
SOURCE_REVISION = "35396c71a6f90cea8ef162ffbb69d950aacc64f1"

SKIP = {"Tank_Transverse", "Tank_Sponson_L", "Tank_Sponson_R", "Engine_Placeholder"}
FANS = {
    "Rotor_FL_Upper": ("x1_fan_fl_upper.glb", "FL upper", "CCW", "lift"),
    "Rotor_FL_Lower": ("x1_fan_fl_lower.glb", "FL lower", "CW", "lift"),
    "Rotor_FR_Upper": ("x1_fan_fr_upper.glb", "FR upper", "CW", "lift"),
    "Rotor_FR_Lower": ("x1_fan_fr_lower.glb", "FR lower", "CCW", "lift"),
    "Rotor_RL_Upper": ("x1_fan_rl_upper.glb", "RL upper", "CW", "lift"),
    "Rotor_RL_Lower": ("x1_fan_rl_lower.glb", "RL lower", "CCW", "lift"),
    "Rotor_RR_Upper": ("x1_fan_rr_upper.glb", "RR upper", "CCW", "lift"),
    "Rotor_RR_Lower": ("x1_fan_rr_lower.glb", "RR lower", "CW", "lift"),
    "Thruster_Fan_L": ("x1_pusher_left.glb", "pusher left", "CCW", "pusher"),
    "Thruster_Fan_R": ("x1_pusher_right.glb", "pusher right", "CW", "pusher"),
}
# Seen from inside the cabin. Everything else is an exterior shell and is wound outward.
INTERIOR = {
    "Cockpit_Tub", "Cockpit_Floor", "Dash", "Dash_Screen", "Screen_Display",
    "Console", "Stick", "Seat_L", "Seat_R", "Footrest_L", "Footrest_R", "Firewall",
}
ANCHORS = ("Eye_L", "Eye_R", "Screen_Centre", "Stick_Pivot")
IN2M = 0.0254
CANT_REAR = np.deg2rad(10.0)
TILT_PUSHER = np.deg2rad(10.0)


def load_glb(path):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SOURCE_SHA256:
        raise SystemExit(f"refusing {path}: sha256 {digest} != {SOURCE_SHA256}")
    off = 12
    gltf = blob = None
    while off < len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        off += 8
        chunk = data[off:off + clen]
        off += clen
        if ctype == 0x4E4F534A:
            gltf = json.loads(chunk)
        elif ctype == 0x004E4942:
            blob = chunk
    return gltf, blob


def accessor(gltf, blob, idx):
    acc = gltf["accessors"][idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    ctype = {5123: np.uint16, 5125: np.uint32, 5126: np.float32}[acc["componentType"]]
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[acc["type"]]
    count = acc["count"]
    return np.frombuffer(blob, dtype=ctype, count=count * ncomp, offset=start).reshape(count, ncomp).copy()


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], float)


def kit_to_ptero(points):
    """Kit glTF (+X left, +Y up, +Z nose) to PteroSim glTF (+X right, +Y up, -Z nose)."""
    out = np.array(points, float, copy=True)
    out[..., 0] *= -1
    out[..., 2] *= -1
    return out


def kit_to_vehicle(point):
    """Kit glTF point to vehicle metres: X forward, Y right, Z up."""
    x, y, z = point
    return np.array([z, -x, y], float)


def vehicle_to_structural_in(vehicle):
    """Vehicle metres to JSBSim structural inches: X aft, Y right, Z up."""
    return np.array([-vehicle[0], vehicle[1], vehicle[2]], float) / IN2M


class GLBWriter:
    def __init__(self, part):
        self.bin = bytearray()
        self.g = {
            "asset": {
                "version": "2.0",
                "generator": "x1_skyracer/tools/build_pterosim_glb.py",
                "extras": {
                    "source_revision": SOURCE_REVISION,
                    "source_sha256": SOURCE_SHA256,
                    "part": part,
                    "frame": "glTF metres, +Y up, nose along -Z. PteroSim loads (x, y, z) as (-z, x, y).",
                },
            },
            "scene": 0,
            "scenes": [{"nodes": []}],
            "nodes": [],
            "meshes": [],
            "materials": [],
            "accessors": [],
            "bufferViews": [],
            "buffers": [],
        }

    def view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin += b"\0"
        off = len(self.bin)
        self.bin += data
        bv = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            bv["target"] = target
        self.g["bufferViews"].append(bv)
        return len(self.g["bufferViews"]) - 1

    def accessor(self, arr, comp, typ, target=None, minmax=False):
        arr = np.ascontiguousarray(arr)
        bv = self.view(arr.tobytes(), target)
        acc = {"bufferView": bv, "componentType": comp, "count": int(arr.shape[0]), "type": typ}
        if minmax:
            acc["min"] = [float(v) for v in (arr.min(0) if arr.ndim > 1 else [arr.min()])]
            acc["max"] = [float(v) for v in (arr.max(0) if arr.ndim > 1 else [arr.max()])]
        self.g["accessors"].append(acc)
        return len(self.g["accessors"]) - 1

    def write(self, path):
        g = {k: v for k, v in self.g.items() if v != []}
        g["buffers"] = [{"byteLength": len(self.bin)}]
        js = json.dumps(g, separators=(",", ":")).encode()
        js += b" " * ((4 - len(js) % 4) % 4)
        while len(self.bin) % 4:
            self.bin += b"\0"
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, total))
            f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
            f.write(struct.pack("<II", len(self.bin), 0x004E4942) + bytes(self.bin))
        return total


def signed_volume(P, T):
    c = P.mean(0)
    a, b, d = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    return float(np.einsum("ij,ij->i", a - c, np.cross(b - c, d - c)).sum() / 6.0)


def face_normals(P, T):
    a, b, d = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    n = np.cross(b - a, d - a)
    length = np.linalg.norm(n, axis=1)
    good = length > 1e-12
    n[good] /= length[good, None]
    return n, good


def facing_fraction(P, T, eye):
    n, good = face_normals(P, T)
    centers = (P[T[:, 0]] + P[T[:, 1]] + P[T[:, 2]]) / 3
    dots = np.einsum("ij,ij->i", n[good], eye - centers[good])
    return float((dots > 0).mean()) if len(dots) else 0.0


def bake_world(gltf, blob):
    nodes = gltf["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    cache = {}

    def world(i):
        if i in cache:
            return cache[i]
        n = nodes[i]
        t = np.array(n.get("translation", [0, 0, 0]), float)
        R = quat_to_mat(n.get("rotation", [0, 0, 0, 1]))
        if i in parent:
            tw, Rw = world(parent[i])
            t, R = tw + Rw @ t, Rw @ R
        cache[i] = (t, R)
        return cache[i]

    parts = []
    for i, n in enumerate(nodes):
        name = n.get("name")
        if "mesh" not in n or name in SKIP:
            continue
        origin, R = world(i)
        prims = []
        for pr in gltf["meshes"][n["mesh"]]["primitives"]:
            pos = accessor(gltf, blob, pr["attributes"]["POSITION"]).astype(np.float64)
            idx = accessor(gltf, blob, pr["indices"]).reshape(-1, 3).astype(np.int64)
            nrm = accessor(gltf, blob, pr["attributes"]["NORMAL"]).astype(np.float64)
            uv = accessor(gltf, blob, pr["attributes"]["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in pr["attributes"] else None
            pw = (R @ pos.T).T + origin
            nw = (R @ nrm.T).T
            nw /= np.maximum(np.linalg.norm(nw, axis=1), 1e-12)[:, None]
            mat = gltf["materials"][pr["material"]]["name"]
            prims.append({"material": mat, "pos": pw, "nrm": nw, "idx": idx, "uv": uv})
        parts.append({"name": name, "origin": origin, "prims": prims})
    anchors = {}
    for n in nodes:
        if n.get("name") in ANCHORS:
            anchors[n["name"]] = np.array(n.get("translation", [0, 0, 0]), float)
    return parts, anchors, gltf


def apply_flip(part):
    for pr in part["prims"]:
        pr["idx"] = pr["idx"][:, ::-1]
        pr["nrm"] = -pr["nrm"]


def eye_for(name, anchors):
    """Left parts are judged from the left eye, right parts from the right eye."""
    if name.endswith("_R"):
        return anchors["Eye_R"]
    if name.endswith("_L"):
        return anchors["Eye_L"]
    return (anchors["Eye_L"] + anchors["Eye_R"]) / 2


def extreme_point(points, axis, sign, lateral_axis=0):
    """Extreme along axis, then the one closest to the centre plane among that band."""
    coord = sign * points[:, axis]
    band = points[coord >= coord.max() - 0.01]
    return band[np.argmin(np.abs(band[:, lateral_axis]))]


def shell_groups(part):
    """Connected face groups across a part's primitives, welded at 0.01 mm."""
    faces = []
    chunks = []
    base = 0
    for pi, pr in enumerate(part["prims"]):
        chunks.append(pr["pos"])
        for li, tri in enumerate(pr["idx"]):
            faces.append((pi, li, tri + base))
        base += len(pr["pos"])
    pos = np.vstack(chunks)
    _, weld = np.unique(np.round(pos, 5), axis=0, return_inverse=True)
    parent = np.arange(int(weld.max()) + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for _, _, tri in faces:
        w = weld[tri]
        parent[find(w[1])] = find(w[0])
        parent[find(w[2])] = find(w[0])
    groups = {}
    for pi, li, tri in faces:
        groups.setdefault(find(weld[tri[0]]), []).append((pi, li))
    return pos, groups


def flip_faces(part, group):
    seen = set()
    for pi, li in group:
        pr = part["prims"][pi]
        a, b, c = (int(v) for v in pr["idx"][li])
        pr["idx"][li] = (a, c, b)
        for v in (a, b, c):
            if (pi, v) not in seen:
                pr["nrm"][v] *= -1
                seen.add((pi, v))


def orient_part(part, anchors):
    P = np.vstack([pr["pos"] for pr in part["prims"]])
    T = []
    base = 0
    for pr in part["prims"]:
        T.append(pr["idx"] + base)
        base += len(pr["pos"])
    T = np.vstack(T)
    vol = signed_volume(P, T)
    reason = "kept"
    flipped = False
    if part["name"] in INTERIOR:
        eye = eye_for(part["name"], anchors)
        before = facing_fraction(P, T, eye)
        apply_flip(part)
        P2 = np.vstack([pr["pos"] for pr in part["prims"]])
        T2 = []
        base = 0
        for pr in part["prims"]:
            T2.append(pr["idx"] + base)
            base += len(pr["pos"])
        after = facing_fraction(P2, np.vstack(T2), eye)
        if after + 1e-6 < before:
            apply_flip(part)  # undo
            reason = f"interior faces the eye ({before:.2f})"
        else:
            flipped = True
            reason = f"interior flipped to face the eye ({before:.2f} -> {after:.2f})"
    else:
        _, groups = shell_groups(part)
        inward = 0
        for group in groups.values():
            # Volume of this shell from the faces it still has.
            tris = []
            pts = []
            for pi, li in group:
                tris.append(part["prims"][pi]["idx"][li])
                pts.append(part["prims"][pi]["pos"])
            # Use the part positions already in each primitive.
            acc = []
            for pi, li in group:
                acc.append(part["prims"][pi]["pos"][part["prims"][pi]["idx"][li]])
            tri = np.stack(acc)
            c = tri.reshape(-1, 3).mean(0)
            shell = float(np.einsum("ij,ij->i", tri[:, 0] - c, np.cross(tri[:, 1] - c, tri[:, 2] - c)).sum() / 6)
            if shell < 0:
                flip_faces(part, group)
                inward += 1
        flipped = inward > 0
        reason = f"exterior volume {vol:.4f}; flipped {inward} inward shell(s)"
    return {"name": part["name"], "volume_before": vol, "flipped": flipped, "reason": reason}


def concat(parts, names=None):
    """Merge primitives by material. Positions stay in kit world metres."""
    groups = {}
    for part in parts:
        if names is not None and part["name"] not in names:
            continue
        for pr in part["prims"]:
            g = groups.setdefault(pr["material"], {"pos": [], "nrm": [], "idx": [], "uv": [], "has_uv": pr["uv"] is not None})
            base = sum(len(p) for p in g["pos"])
            g["pos"].append(pr["pos"])
            g["nrm"].append(pr["nrm"])
            g["idx"].append(pr["idx"] + base)
            if pr["uv"] is not None:
                g["uv"].append(pr["uv"])
            else:
                g["has_uv"] = False
    out = []
    for mat, g in groups.items():
        out.append({
            "material": mat,
            "pos": np.vstack(g["pos"]),
            "nrm": np.vstack(g["nrm"]),
            "idx": np.vstack(g["idx"]),
            "uv": np.vstack(g["uv"]) if g["has_uv"] and g["uv"] else None,
        })
    return out


def add_materials(writer, gltf, needed, livery_png):
    by_name = {m["name"]: m for m in gltf["materials"]}
    ids = {}
    tex_index = None
    if "Livery" in needed:
        writer.g["images"] = [{"bufferView": writer.view(livery_png), "mimeType": "image/png"}]
        writer.g["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
        writer.g["textures"] = [{"sampler": 0, "source": 0}]
        tex_index = 0
    for name in needed:
        src = json.loads(json.dumps(by_name[name]))
        src.pop("doubleSided", None)
        if name == "Glass":
            src["doubleSided"] = True
            src["alphaMode"] = "BLEND"
        if name == "Livery":
            src["pbrMetallicRoughness"]["baseColorTexture"] = {"index": tex_index}
        elif "baseColorTexture" in src.get("pbrMetallicRoughness", {}):
            raise SystemExit(f"unexpected texture on {name}")
        writer.g["materials"].append(src)
        ids[name] = len(writer.g["materials"]) - 1
    return ids


def livery_png(gltf, blob):
    image = gltf["images"][0]
    bv = gltf["bufferViews"][image["bufferView"]]
    start = bv.get("byteOffset", 0)
    return blob[start:start + bv["byteLength"]]


def write_mesh(writer, groups, mat_ids, to_ptero=True, origin=None):
    prims = []
    for g in groups:
        pos = g["pos"] if origin is None else g["pos"] - origin
        nrm = g["nrm"]
        if to_ptero:
            pos = kit_to_ptero(pos)
            nrm = kit_to_ptero(nrm)
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)[:, None]
        attrs = {
            "POSITION": writer.accessor(pos.astype(np.float32), 5126, "VEC3", 34962, True),
            "NORMAL": writer.accessor(nrm.astype(np.float32), 5126, "VEC3", 34962),
        }
        if g["uv"] is not None:
            attrs["TEXCOORD_0"] = writer.accessor(g["uv"].astype(np.float32), 5126, "VEC2", 34962)
        index = g["idx"].reshape(-1).astype(np.uint32)
        attrs_idx = writer.accessor(index, 5125, "SCALAR", 34963)
        prims.append({"attributes": attrs, "indices": attrs_idx, "material": mat_ids[g["material"]]})
    writer.g["meshes"].append({"name": writer.g["asset"]["extras"]["part"], "primitives": prims})
    return 0


def thrust_axis(kind, kit_centroid):
    """Unit thrust direction in the vehicle frame, from the design angles checked against this mesh."""
    if kind == "lift":
        cant = CANT_REAR if kit_centroid[2] < 0 else 0.0  # rear ducts are aft (kit Z < 0)
        return np.array([np.sin(cant), 0.0, np.cos(cant)])
    return np.array([np.cos(TILT_PUSHER), 0.0, np.sin(TILT_PUSHER)])


def spin_axis(thrust, spin):
    # Clockwise seen from the axis tip. CCW lift is viewed from above, so the axis points down.
    # CCW pusher is viewed from behind, so the axis points forward, along thrust.
    if spin == "CCW":
        return -thrust if thrust[2] > 0.5 else thrust
    return thrust if thrust[2] > 0.5 else -thrust


def cushion_point(part):
    """Top of the seat pan: upward-facing Seat faces below the occupant's shoulders."""
    pts = []
    for pr in part["prims"]:
        if pr["material"] != "Seat":
            continue
        n, good = face_normals(pr["pos"], pr["idx"])
        centers = (pr["pos"][pr["idx"][:, 0]] + pr["pos"][pr["idx"][:, 1]] + pr["pos"][pr["idx"][:, 2]]) / 3
        keep = good & (n[:, 1] > 0.6) & (centers[:, 1] < 0.7)
        if keep.any():
            pts.append(centers[keep])
    if not pts:
        return None
    c = np.vstack(pts).mean(0)
    return c


def round_in(structural_m_or_in, already_inches=False):
    value = structural_m_or_in if already_inches else structural_m_or_in
    return float(f"{value:.2f}")


def main(src, out_dir, report_dir):
    gltf, blob = load_glb(src)
    parts, anchors, gltf = bake_world(gltf, blob)
    by = {p["name"]: p for p in parts}
    eye = anchors["Eye_L"]
    decisions = [orient_part(p, anchors) for p in parts if p["name"] not in FANS]
    for p in parts:
        if p["name"] in FANS:
            decisions.append(orient_part(p, anchors))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = livery_png(gltf, blob)
    body_parts = [p for p in parts if p["name"] not in FANS]
    groups = concat(body_parts)
    writer = GLBWriter("x1_body")
    mat_ids = add_materials(writer, gltf, {g["material"] for g in groups}, png)
    write_mesh(writer, groups, mat_ids)
    root = {"name": "x1_body", "mesh": 0, "children": []}
    writer.g["nodes"].append(root)
    writer.g["scenes"][0]["nodes"].append(0)
    anchor_ptero = {}
    for name in ANCHORS:
        ptero = kit_to_ptero(anchors[name])
        anchor_ptero[name] = [round(float(v), 4) for v in ptero]
        writer.g["nodes"].append({"name": name, "translation": [float(v) for v in ptero]})
        root["children"].append(len(writer.g["nodes"]) - 1)
    body_path = out_dir / "x1_body.glb"
    writer.write(body_path)

    fans_meta = []
    for name, (fname, motor, spin, kind) in FANS.items():
        part = by[name]
        groups = concat([part])
        centroid = np.vstack([g["pos"] for g in groups]).mean(0)
        fw = GLBWriter(Path(fname).stem)
        mats = add_materials(fw, gltf, {g["material"] for g in groups}, png)
        write_mesh(fw, groups, mats, origin=centroid)
        fw.g["nodes"].append({"name": Path(fname).stem, "mesh": 0})
        fw.g["scenes"][0]["nodes"].append(0)
        fw.write(out_dir / fname)
        thrust = thrust_axis(kind, centroid)
        axis = spin_axis(thrust, spin)
        origin = part["origin"]
        # Lift hubs are the blade centres. Pusher thrust stays on the duct station; the blades sit ahead of it.
        thrust_point = centroid if kind == "lift" else origin
        vehicle_hub = kit_to_vehicle(centroid)
        vehicle_thrust = kit_to_vehicle(thrust_point)
        fans_meta.append({
            "source": name,
            "file": f"meshes/{fname}",
            "motor": motor,
            "spin_seen_from": ("above" if kind == "lift" else "behind"),
            "spin": spin,
            "thrust_vehicle": [round(float(v), 4) for v in thrust],
            "axis_vehicle": [round(float(v), 4) for v in axis],
            "hub_vehicle_m": [round(float(v), 4) for v in vehicle_hub],
            "thrust_point_vehicle_m": [round(float(v), 4) for v in vehicle_thrust],
            "visual_offset_vehicle_m": [round(float(v), 4) for v in (vehicle_hub - vehicle_thrust)],
            "structural_in": [round_in(v) for v in vehicle_to_structural_in(vehicle_thrust)],
            "jsb_pitch_deg": 80.0 if kind == "lift" and centroid[2] < 0 else (90.0 if kind == "lift" else 10.0),
        })

    seat = cushion_point(by["Seat_L"])
    ballast_vehicle = kit_to_vehicle(np.array([seat[0], seat[1] + 0.28, seat[2]]))
    ballast_in = vehicle_to_structural_in(ballast_vehicle)
    # Existing dry mass and propane, unchanged. Ballast follows the new cushion + 0.28 m.
    masses = [
        (2800.0, np.array([15.25, 0.0, 30.55])),
        (250.0, ballast_in),
        (250.0, np.array([ballast_in[0], -ballast_in[1], ballast_in[2]])),
        (60.9, np.array([-5.91, -33.86, 15.75])),
        (60.9, np.array([-5.91, 33.86, 15.75])),
        (78.1, np.array([-4.13, 0.0, 22.05])),
    ]
    weight = sum(w for w, _ in masses)
    cg_in = sum(w * p for w, p in masses) / weight
    cg_vehicle = np.array([-cg_in[0], cg_in[1], cg_in[2]]) * IN2M
    eye_vehicle = kit_to_vehicle(anchors["Eye_L"])
    eye_from_cg = eye_vehicle - cg_vehicle
    fixture = np.array([cg_in[0], -cg_in[1], -cg_in[2]]) * IN2M

    nose = extreme_point(np.vstack([pr["pos"] for pr in by["Body"]["prims"]]), 2, +1)
    tail = extreme_point(np.vstack([pr["pos"] for pr in by["Body"]["prims"]]), 2, -1)
    canopy_pts = np.vstack([pr["pos"] for pr in by["Canopy"]["prims"]])
    canopy = canopy_pts[canopy_pts[:, 1] >= canopy_pts[:, 1].max() - 0.002].mean(0)
    ducts = {}
    for duct in ("Duct_FL", "Duct_FR", "Duct_RL", "Duct_RR"):
        pts = np.vstack([pr["pos"] for pr in by[duct]["prims"]])
        ducts[duct] = pts[int(pts[:, 1].argmin())]

    def struct(point):
        return [round_in(v) for v in vehicle_to_structural_in(kit_to_vehicle(point))]

    manifest = {
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "winding": decisions,
        "anchors_ptero_m": anchor_ptero,
        "eye_structural_in": [round_in(v) for v in vehicle_to_structural_in(eye_vehicle)],
        "eye_from_cg_vehicle_m": [round(float(v), 4) for v in eye_from_cg],
        "ballast_left_structural_in": [round_in(v) for v in ballast_in],
        "cushion_kit_m": [round(float(v), 4) for v in seat],
        "race_cg_structural_in": [round_in(v) for v in cg_in],
        "race_weight_lb": round(weight, 2),
        "fixture_vehicle_m": [round(float(v), 4) for v in fixture],
        "fans": fans_meta,
        "contacts_structural_in": {
            "NOSE_TIP": struct(nose),
            "TAIL": struct(tail),
            "CANOPY_TOP": struct(canopy),
            **{name: struct(point) for name, point in ducts.items()},
        },
        "screen_note": "Screen_Display UVs are the source 0..1 map (u left to right as the occupants see it, v bottom to top).",
    }
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "geometry_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"body {body_path.stat().st_size} bytes, {len(fans_meta)} fans")
    print("fixture", manifest["fixture_vehicle_m"], "cg_in", manifest["race_cg_structural_in"])
    print("eye", manifest["eye_structural_in"], "from cg", manifest["eye_from_cg_vehicle_m"])
    print("ballast", manifest["ballast_left_structural_in"], "cushion", manifest["cushion_kit_m"])
    for fan in fans_meta:
        print(fan["motor"], "struct", fan["structural_in"], "pitch", fan["jsb_pitch_deg"], "axis", fan["axis_vehicle"], "offset", fan["visual_offset_vehicle_m"])
    print("contacts", json.dumps(manifest["contacts_structural_in"]))
    flipped = [d["name"] for d in decisions if d["flipped"]]
    print("flipped", len(flipped), " ".join(flipped))
    return manifest


if __name__ == "__main__":
    import argparse
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("source_glb")
    parser.add_argument("--out", default=str(here / "meshes"))
    parser.add_argument("--report", default=str(here))
    args = parser.parse_args()
    main(args.source_glb, args.out, args.report)
