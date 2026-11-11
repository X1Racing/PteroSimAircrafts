#!/usr/bin/env python3
"""Asset checks for the X1 SkyRacer meshes. Run from the repository root:

    python3 x1_skyracer/tests/test_visual_assets.py
"""
import json
import struct
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AIR = ROOT / "x1_skyracer"
MANIFEST = json.loads((AIR / "geometry_manifest.json").read_text())


def load_glb(path):
    data = Path(path).read_bytes()
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


def primitives(path):
    gltf, blob = load_glb(path)
    out = []
    for mesh in gltf["meshes"]:
        for pr in mesh["primitives"]:
            pos = accessor(gltf, blob, pr["attributes"]["POSITION"]).astype(np.float64)
            nrm = accessor(gltf, blob, pr["attributes"]["NORMAL"]).astype(np.float64)
            idx = accessor(gltf, blob, pr["indices"]).reshape(-1, 3).astype(np.int64)
            uv = accessor(gltf, blob, pr["attributes"]["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in pr["attributes"] else None
            mat = gltf["materials"][pr["material"]]
            out.append({"pos": pos, "nrm": nrm, "idx": idx, "uv": uv, "mat": mat})
    return gltf, out


def agreement(pr, samples=80000):
    pos, nrm, idx = pr["pos"], pr["nrm"], pr["idx"]
    step = max(1, len(idx) // samples)
    tris = idx[::step]
    a, b, c = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    fn = np.cross(b - a, c - a)
    length = np.linalg.norm(fn, axis=1)
    good = length > 1e-12
    fn = fn[good] / length[good, None]
    vn = nrm[tris[:, 0]] + nrm[tris[:, 1]] + nrm[tris[:, 2]]
    vn = vn[good]
    vl = np.linalg.norm(vn, axis=1)
    vg = vl > 1e-8
    dots = np.einsum("ij,ij->i", fn[vg], vn[vg] / vl[vg, None])
    return float(np.median(dots)), float((dots < 0).mean())


def components_volume(pr):
    pos, idx = pr["pos"], pr["idx"]
    key = np.round(pos, 5)
    _, weld = np.unique(key, axis=0, return_inverse=True)
    parent = np.arange(len(np.unique(weld)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for tri in weld[idx]:
        parent[find(tri[1])] = find(tri[0])
        parent[find(tri[2])] = find(tri[0])
    groups = {}
    for i, tri in enumerate(idx):
        groups.setdefault(find(weld[tri[0]]), []).append(i)
    volumes = []
    for faces in groups.values():
        T = idx[np.array(faces)]
        c = pos[T.reshape(-1)].mean(0)
        a, b, d = pos[T[:, 0]], pos[T[:, 1]], pos[T[:, 2]]
        volumes.append(float(np.einsum("ij,ij->i", a - c, np.cross(b - c, d - c)).sum() / 6))
    return volumes


class SkyRacerAssets(unittest.TestCase):
    def test_provenance(self):
        self.assertEqual(MANIFEST["source_revision"], "35396c71a6f90cea8ef162ffbb69d950aacc64f1")
        self.assertEqual(MANIFEST["source_sha256"], "675e8a3378e750a610d1f5116bf43449d938db87b3e9c9113cc8c37262d9b1ad")

    def test_visual_files_and_winding(self):
        root = ET.parse(AIR / "Visual.xml").getroot()
        for tag in ("fixture", "propeller"):
            for node in root.findall(tag):
                path = AIR / node.get("file")
                self.assertTrue(path.is_file(), path)
                gltf, prims = primitives(path)
                self.assertEqual(gltf["asset"]["version"], "2.0")
                for pr in prims:
                    median, opposite = agreement(pr)
                    self.assertGreater(median, 0.9, node.get("file"))
                    self.assertLess(opposite, 0.01, node.get("file"))
                    self.assertNotIn("doubleSided", pr["mat"]) if pr["mat"]["name"] != "Glass" else self.assertTrue(pr["mat"].get("doubleSided"))

    def test_exterior_shells_face_out_and_cabin_faces_the_eye(self):
        gltf, prims = primitives(AIR / "meshes" / "x1_body.glb")
        by = {}
        for pr in prims:
            by.setdefault(pr["mat"]["name"], pr)
        for name in ("Livery", "Paint_White", "Glass"):
            for vol in components_volume(by[name]):
                if abs(vol) > 0.01:
                    self.assertGreater(vol, 0, name)
        anchors = {n["name"]: np.array(n.get("translation", [0, 0, 0]), float) for n in gltf["nodes"]}
        for name in ("Eye_L", "Eye_R", "Screen_Centre", "Stick_Pivot"):
            self.assertIn(name, anchors)
        tub = by["Cockpit_Tub"]
        centers = tub["pos"][tub["idx"]].mean(1)
        fn = np.cross(tub["pos"][tub["idx"][:, 1]] - tub["pos"][tub["idx"][:, 0]], tub["pos"][tub["idx"][:, 2]] - tub["pos"][tub["idx"][:, 0]])
        fn /= np.maximum(np.linalg.norm(fn, axis=1), 1e-12)[:, None]
        facing = np.einsum("ij,ij->i", fn, anchors["Eye_L"] - centers) > 0
        self.assertGreater(facing.mean(), 0.9)
        screen = by["Screen"]
        self.assertEqual(len(screen["idx"]), 2)
        self.assertTrue(np.min(screen["uv"]) < 0.01 and np.max(screen["uv"]) > 0.99)
        verts = np.unique(np.round(screen["pos"], 5), axis=0)
        # The reclined 0.332 x 0.207 m face spans those two lengths.
        dists = []
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                dists.append(np.linalg.norm(verts[i] - verts[j]))
        lengths = sorted(set(np.round(dists, 3)))
        self.assertAlmostEqual(lengths[0], 0.207, delta=0.004)
        self.assertAlmostEqual(lengths[1], 0.332, delta=0.004)

    def test_propulsion_matches_manifest(self):
        root = ET.parse(AIR / "Propulsion.xml").getroot()
        visual = ET.parse(AIR / "Visual.xml").getroot()
        engines = {e.get("name"): e for e in root.findall("engine")}
        props = {p.get("motor"): p for p in visual.findall("propeller")}
        for fan in MANIFEST["fans"]:
            engine = engines[fan["motor"]]
            loc = engine.find("thruster/location")
            got = [float(loc.find(a).text) for a in ("x", "y", "z")]
            for a, b in zip(got, fan["structural_in"]):
                self.assertAlmostEqual(a, b, delta=0.02)
            axis = [float(v) for v in props[fan["motor"]].get("axis").split()]
            expect = np.array(fan["axis_vehicle"])
            self.assertAlmostEqual(np.dot(axis, expect), 1, delta=0.002)
            # The disk's thin axis, after PteroSim's loader, lies along that shaft.
            _, prims = primitives(AIR / fan["file"])
            pos = np.vstack([pr["pos"] for pr in prims])
            cov = np.cov((pos - pos.mean(0)).T)
            _, vecs = np.linalg.eigh(cov)
            thin = vecs[:, 0]
            loaded = np.array([-thin[2], thin[0], thin[1]])
            self.assertGreater(abs(np.dot(loaded, expect)), 0.98)


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2).result.wasSuccessful() and 0 or 1)
