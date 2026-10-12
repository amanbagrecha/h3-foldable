"""H3 foldable globe: lay H3 cells out on a printable net for a choice of shapes.

Shapes (see shapes.py): H3's own icosahedron plus other polyhedra and globe
gores. Every cell is clipped on the sphere against each panel (face or gore) it
touches, then projected onto that panel: gnomonic for polyhedron faces, so
great-circle cell edges stay straight on paper, sinusoidal for gores. The
halves of a cell split by a fold or seam meet again once the paper is folded.
"""
import hashlib
import json
import math
import os
import sys
import tempfile

import h3
import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
import shapes  # noqa: E402

CACHE_DIR = os.path.join(APP_DIR, ".fused", "cache")
CACHE_VERSION = "v6"

NE_BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
NE_LAND = "ne_110m_land.geojson"
NE_COAST = "ne_110m_coastline.geojson"

MAX_RES = 3
DENSIFY_RAD = math.radians(1.5)  # max arc length between points on curved panels


# ---------------------------------------------------------------- cache utils

def _cache_path(*parts):
    path = os.path.join(CACHE_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _write_atomic(path, data: bytes):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _natural_earth(name):
    path = _cache_path("ne", name)
    cached = _read_json(path)
    if cached is not None:
        return cached
    import requests

    r = requests.get(NE_BASE + name, timeout=30)
    r.raise_for_status()
    _write_atomic(path, r.content)
    return r.json()


# ---------------------------------------------------------------- land mask

RASTER_DEG = 0.1


def _land_raster():
    """Equirectangular 0.1° land mask (uint8), rasterised once from Natural Earth."""
    path = _cache_path("land", f"v1-{NE_LAND}.npy")
    try:
        return np.load(path)
    except (OSError, ValueError):
        pass
    from PIL import Image, ImageDraw

    w, h = int(360 / RASTER_DEG), int(180 / RASTER_DEG)
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)

    def px(ring):
        return [((lng + 180) / RASTER_DEG, (90 - lat) / RASTER_DEG) for lng, lat in ring]

    for feat in _natural_earth(NE_LAND)["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            draw.polygon(px(poly[0]), fill=1)
            for hole in poly[1:]:
                draw.polygon(px(hole), fill=0)
    arr = np.asarray(img, dtype=np.uint8)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-", suffix=".npy")
    os.close(fd)
    np.save(tmp, arr)
    os.replace(tmp, path)
    return arr


def _land_fn(raster):
    def is_land(lat, lng):
        lat = np.asarray(lat, dtype=float)
        lng = np.asarray(lng, dtype=float)
        rows = np.clip(((90 - lat) / RASTER_DEG).astype(int), 0, raster.shape[0] - 1)
        cols = np.clip(((lng + 180) / RASTER_DEG).astype(int), 0, raster.shape[1] - 1)
        return raster[rows, cols] > 0
    return is_land


def _coast_lines():
    for feat in _natural_earth(NE_COAST)["features"]:
        geom = feat["geometry"]
        lines = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
        for line in lines:
            if len(line) >= 2:
                yield np.asarray(line, dtype=float)


# ---------------------------------------------------------------- clipping

def _clip_sphere(P, planes):
    """Clip a spherical polygon (unit-vector ring) to the panel's half-spaces.

    Works on chords: a plane through the origin cuts a chord exactly where it
    cuts the great-circle arc, so normalising the result lands on the sphere.
    """
    poly = list(P)
    for m in planes:
        out = []
        for i, cur in enumerate(poly):
            prev = poly[i - 1]
            dc, dp = cur @ m, prev @ m
            if (dc >= 0) != (dp >= 0):
                out.append(prev + dp / (dp - dc) * (cur - prev))
            if dc >= 0:
                out.append(cur)
        poly = out
        if len(poly) < 3:
            return None
    return shapes.unit(np.array(poly))


def _densify_ring(P):
    out = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        n = max(1, int(math.ceil(math.acos(float(np.clip(a @ b, -1, 1))) / DENSIFY_RAD)))
        out.append(shapes.slerp(a, b, n)[:-1] if n > 1 else a[None])
    return np.concatenate(out)


def _r(a, nd=4):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


# ---------------------------------------------------------------- cells

def _cells(res, panels, is_land, coast_cells):
    ids = [c for base in h3.get_res0_cells() for c in h3.cell_to_children(base, res)]
    pent = set(h3.get_pentagons(res))
    bnds = [h3.cell_to_boundary(c) for c in ids]
    centers_ll = np.array([h3.cell_to_latlng(c) for c in ids])
    C = shapes.xyz(centers_ll[:, 0], centers_ll[:, 1])

    # padded vertex array for the vectorised panel tests
    K = max(len(b) for b in bnds)
    pad = np.array([list(b) + [b[-1]] * (K - len(b)) for b in bnds])
    Pv = shapes.xyz(pad[..., 0], pad[..., 1])  # N x K x 3

    # land share: centre plus points halfway to each vertex
    mid_lat, mid_lng = shapes.latlng((Pv + C[:, None, :]) / 2)
    samples = np.concatenate([is_land(centers_ll[:, 0], centers_ll[:, 1])[:, None], is_land(mid_lat, mid_lng)], axis=1)
    share = samples.mean(axis=1)

    pieces = [[] for _ in ids]
    label = [None] * len(ids)
    for panel in panels:
        D = Pv @ panel.planes.T  # N x K x planes
        touch = ~(D < 0).all(axis=1).any(axis=1)
        inside = (D >= -1e-12).all(axis=(1, 2))
        cin = (C @ panel.planes.T >= 0).all(axis=1)
        for i in np.flatnonzero(cin):
            if label[i] is None:
                label[i] = _r(panel.project(C[i][None])[0])
        for i in np.flatnonzero(touch):
            P = Pv[i, : len(bnds[i])]
            ring = P if inside[i] else _clip_sphere(P, panel.planes)
            if ring is None:
                continue
            if panel.curved:
                ring = _densify_ring(ring)
            pieces[i].append(_r(panel.project(ring)))

    out = []
    for i, cid in enumerate(ids):
        s = share[i]
        cls = 2 if s >= 0.5 else (1 if s > 0 or cid in coast_cells else 0)
        out.append([cid, cls, 1 if cid in pent else 0, pieces[i],
                    np.round(np.array(bnds[i]), 2).tolist(), label[i] or [0, 0]])
    return out


def _distortion(cells, res):
    """Spread of printed-area / true-area across cells, normalised to the median."""
    ratios = []
    for c in cells:
        a = 0.0
        for piece in c[3]:
            p = np.asarray(piece)
            a += 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))
        if a > 0:
            ratios.append(a / h3.cell_area(c[0], unit="km^2"))
    r = np.array(ratios) / np.median(ratios)
    return round(float(np.percentile(r, 1)), 2), round(float(np.percentile(r, 99)), 2)


# ---------------------------------------------------------------- coastline

def _coast(panels):
    net_lines, ll_lines = [], []
    for arr in _coast_lines():
        ll_lines.append(np.round(arr[:, ::-1], 2).tolist())
        P = shapes.xyz(arr[:, 1], arr[:, 0])
        p0, p1 = P[:-1], P[1:]
        for panel in panels:
            keep = np.ones(len(p0), dtype=bool)
            t0 = np.zeros(len(p0))
            t1 = np.ones(len(p0))
            with np.errstate(divide="ignore", invalid="ignore"):
                for m in panel.planes:
                    v0, v1 = p0 @ m, p1 @ m
                    keep &= ~((v0 < 0) & (v1 < 0))
                    t = v0 / (v0 - v1)
                    t0 = np.where(v0 < 0, np.maximum(t0, t), t0)
                    t1 = np.where(v1 < 0, np.minimum(t1, t), t1)
            keep &= t0 < t1
            if not keep.any():
                continue
            cur, last = None, -2
            for i in np.flatnonzero(keep):
                a = p0[i] + t0[i] * (p1[i] - p0[i])
                b = p0[i] + t1[i] * (p1[i] - p0[i])
                if cur is not None and i == last + 1 and t0[i] == 0 and t1[last] == 1:
                    cur.append(b)
                else:
                    if cur:
                        net_lines.append(_r(panel.project(shapes.unit(np.array(cur)))))
                    cur = [a, b]
                last = i
            if cur:
                net_lines.append(_r(panel.project(shapes.unit(np.array(cur)))))
    return net_lines, ll_lines


# ---------------------------------------------------------------- main

def _candidates(shape, is_land):
    path = _cache_path("nets", f"{CACHE_VERSION}-cands-{shape}.json")
    cached = _read_json(path)
    if cached is not None:
        return cached
    cands = shapes.candidates(shape, is_land)
    _write_atomic(path, json.dumps(cands, separators=(",", ":")).encode())
    return cands


def _anchors(shape, cands, current):
    """One clickable dot per net root on the globe; click = best net around it."""
    if shape == "gores":
        return []
    spin = cands[current]["spin"]
    poly = shapes.Poly(shape, spin)
    seen = {}
    for i, c in enumerate(cands):
        if c["spin"] != spin:
            continue
        key = ("v", c["anchor_v"]) if c.get("anchor_v") is not None else ("f", c["root"])
        if key not in seen:
            pt = poly.V[key[1]] if key[0] == "v" else poly.faces[key[1]].center
            lat, lng = shapes.latlng(pt)
            seen[key] = {"ll": [round(float(lat), 3), round(float(lng), 3)], "net": i}
    return list(seen.values())


def main(shape: str = "icosahedron", res: int = 2, net: str = "auto"):
    if shape not in shapes.SHAPES:
        shape = "icosahedron"
    res = max(0, min(MAX_RES, int(res)))
    is_land = _land_fn(_land_raster())
    cands = _candidates(shape, is_land)
    idx = 0 if net in ("", "auto") else int(net) % len(cands)

    key = hashlib.sha1(f"{CACHE_VERSION}|{shape}|{res}|{idx}".encode()).hexdigest()[:16]
    path = _cache_path("nets", f"{key}.json")
    cached = _read_json(path)
    if cached is not None:
        return cached

    cand = cands[idx]
    built = shapes.build(shape, cand)
    panels = built["panels"]

    coast_cells = set()
    for arr in _coast_lines():
        for lng, lat in arr:
            coast_cells.add(h3.latlng_to_cell(float(lat), float(lng), res))

    cells = _cells(res, panels, is_land, coast_cells)
    coast_net, coast_ll = _coast(panels)
    lo, hi = _distortion(cells, res)

    result = {
        "shape": shape,
        "info": shapes.shape_info(shape),
        "res": res,
        "net": idx,
        "net_count": len(cands),
        "stats": {
            "cells": int(h3.get_num_cells(res)),
            "area_km2": float(h3.average_hexagon_area(res, unit="km^2")),
            "edge_km": float(h3.average_hexagon_edge_length(res, unit="km")),
            "cut_land": cand["land"],
            "panels": len(panels),
            "distortion": [lo, hi],
        },
        "panels": [
            {"outline": _r(o), "num": n, "label": _r(np.asarray(o).mean(axis=0))}
            for o, n in zip(built["outlines"], built["numbers"])
        ],
        "folds": [_r(l) for l in built["folds"]],
        "cuts": [_r(l) for l in built["cuts"]],
        "tabs": [_r(t) for t in built["tabs"]],
        "globe": {
            "lines": [
                {"cut": g["cut"], "pts": np.round(np.stack(shapes.latlng(g["pts"]), axis=1), 3).tolist()}
                for g in built["globe"]
            ],
            "anchors": _anchors(shape, cands, idx),
        },
        "cells": cells,
        "coast": coast_net,
        "coast_ll": coast_ll,
    }
    _write_atomic(path, json.dumps(result, separators=(",", ":")).encode())
    return result
