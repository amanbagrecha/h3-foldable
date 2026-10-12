"""Foldable shapes: polyhedra and globe gores, each as panels plus an unfolded net.

A panel is one piece of the sphere (a polyhedron face or a gore) with:
  - planes   inward normals of the great circles bounding it; a point is in the
             panel when it is on the positive side of all of them
  - project  sphere -> net 2D (gnomonic for polyhedron faces, sinusoidal for gores)
  - curved   whether great-circle arcs bend in the projection (need densifying)

Nets for polyhedra come from spanning trees over face adjacency: faces joined
by a tree edge fold, every other edge is cut and gets one glue tab. Each tree is
laid flat, rejected if any faces overlap, rotated to fit a landscape page and
scored on page fit and how much land its cut lines cross.
"""
import math
from collections import deque
from itertools import combinations

import h3
import numpy as np

PAGE_ASPECT = 277 / 166  # A4 landscape printable area (A3 and Letter are close)
TAB_DEPTH = 0.16         # in edge lengths, capped by TAB_MM on the printed page
TAB_MM = 7.0             # tab depth on a 277 mm wide printable area
TAB_ANGLES = (55, 35, 20, 9)  # side slope per end; steep ends squeeze into tight corners
N_GORES = 12


# ---------------------------------------------------------------- sphere math

def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def xyz(lat, lng):
    lat = np.radians(np.asarray(lat, dtype=float))
    lng = np.radians(np.asarray(lng, dtype=float))
    return np.stack([np.cos(lat) * np.cos(lng), np.cos(lat) * np.sin(lng), np.sin(lat)], axis=-1)


def latlng(v):
    v = unit(v)
    return np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1))), np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def slerp(a, b, n):
    """n+1 points along the great-circle arc a -> b (inclusive)."""
    t = np.linspace(0, 1, n + 1)[:, None]
    omega = math.acos(float(np.clip(a @ b, -1, 1)))
    if omega < 1e-12:
        return np.repeat(a[None], n + 1, axis=0)
    return (np.sin((1 - t) * omega) * a + np.sin(t * omega) * b) / math.sin(omega)


def rot2(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _rotz(V, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return V @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T


def _rodrigues(u, target):
    u, target = unit(u), unit(target)
    k = np.cross(u, target)
    s = np.linalg.norm(k)
    if s < 1e-12:
        return np.eye(3)
    k /= s
    c = float(u @ target)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


# ---------------------------------------------------------------- shape catalogue

SHAPES = {
    "icosahedron": {
        "name": "Icosahedron", "tag": "H3 native", "faces": "20 triangles", "spins": [0],
        "blurb": "H3's own shape. H3 draws every cell flat on one of these 20 triangles, and its 12 pentagons "
                 "sit on the 12 corners where five triangles meet. This is the least distorted "
                 "straight-edged net and the only one where the fold lines match the grid's own structure.",
    },
    "dodecahedron": {
        "name": "Dodecahedron", "tag": "12 pentagons", "faces": "12 pentagons", "spins": [0],
        "blurb": "The icosahedron's twin: swap corners for faces and you get 12 pentagons. Here each face is centred "
                 "on one of H3's 12 pentagon cells, so every H3 pentagon sits at the exact middle of a paper pentagon.",
    },
    "truncated-icosahedron": {
        "name": "Soccer ball", "tag": "32 faces", "faces": "12 pentagons + 20 hexagons", "spins": [0],
        "blurb": "Slice the 12 corners off H3's icosahedron and you get a soccer ball, also called a truncated icosahedron. "
                 "The 12 pentagon panels are centred on H3's pentagons and the 20 hexagon panels on H3's 20 faces. "
                 "It has more panels, so it is rounder and less distorted, but there is more to cut.",
    },
    "octahedron": {
        "name": "Octahedron", "tag": "Butterfly", "faces": "8 triangles", "spins": list(range(0, 90, 5)),
        "blurb": "Eight triangles, the shape behind Cahill's 1909 butterfly map and the Cahill–Keyes and "
                 "Waterman maps that followed. It is simple to fold but stretches the grid more than the icosahedron does.",
    },
    "cube": {
        "name": "Cube", "tag": "S2 faces", "faces": "6 squares", "spins": [0],
        "blurb": "Six squares facing the same way as Google's S2 grid: face 0 at 0°N 0°E, face 2 over the North Pole. "
                 "S2 builds its cells on this cube, and H3 builds on the icosahedron. Here you see H3's hexagons "
                 "on S2's cube, with strong stretching toward each square's corners.",
    },
    "tetrahedron": {
        "name": "Tetrahedron", "tag": "4 faces", "faces": "4 triangles", "spins": list(range(0, 120, 5)),
        "blurb": "The fewest flat faces you can wrap a globe in, as in Lee's 1965 tetrahedral map. "
                 "The distortion is extreme: hexagons near the middle of a face print tiny and hexagons near the corners "
                 "print huge. It shows why H3 uses 20 faces.",
    },
    "gores": {
        "name": "Gores", "tag": "Orange peel", "faces": f"{N_GORES} gores", "spins": [i * 2.5 for i in range(12)],
        "blurb": "How printed globes have been made since the 1500s: the sphere is peeled into 12 pointed strips "
                 "(gores), each drawn with a sinusoidal projection around its own centre meridian. The gores bend "
                 "instead of folding, so the finished globe is the roundest of all of these.",
    },
}


def shape_info(shape):
    return {k: v for k, v in SHAPES[shape].items() if k != "spins"}


# ---------------------------------------------------------------- polyhedra

def _h3_icosa_vertices():
    return xyz(*zip(*[h3.cell_to_latlng(p) for p in sorted(h3.get_pentagons(0))]))


def _hull_faces(V):
    """Faces of a convex polyhedron with all vertices on the unit sphere."""
    seen, out = set(), []
    for i, j, k in combinations(range(len(V)), 3):
        nrm = np.cross(V[j] - V[i], V[k] - V[i])
        L = np.linalg.norm(nrm)
        if L < 1e-9:
            continue
        nrm /= L
        s = V @ nrm - V[i] @ nrm
        if s.max() > 1e-6 and s.min() < -1e-6:
            continue
        members = tuple(sorted(np.flatnonzero(np.abs(s) < 1e-6).tolist()))
        if members not in seen:
            seen.add(members)
            out.append(members)
    return out


def _vertices(shape, spin):
    if shape == "icosahedron":
        V = _h3_icosa_vertices()
    elif shape == "dodecahedron":
        I = _h3_icosa_vertices()
        V = unit(np.array([I[list(f)].mean(axis=0) for f in _hull_faces(I)]))
    elif shape == "truncated-icosahedron":
        I = _h3_icosa_vertices()
        pts = []
        for a, b in combinations(range(12), 2):
            if 0.3 < I[a] @ I[b] < 0.6:
                pts += [(2 * I[a] + I[b]) / 3, (I[a] + 2 * I[b]) / 3]
        V = unit(np.array(pts))
    elif shape == "octahedron":
        V = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    elif shape == "cube":
        V = unit(np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float))
    elif shape == "tetrahedron":
        V = unit(np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float))
        V = V @ _rodrigues(V[0], [0, 0, 1]).T  # one corner on the North Pole
    else:
        raise ValueError(f"unknown shape {shape!r}")
    return _rotz(V, spin) if spin else V


class Face:
    def __init__(self, V, members):
        P = V[list(members)]
        o = P.mean(axis=0)
        n = unit(np.cross(P[1] - P[0], P[2] - P[0]))
        if n @ o < 0:
            n = -n
        e1 = unit(P[0] - o)
        e2 = np.cross(n, e1)
        ang = np.arctan2((P - o) @ e2, (P - o) @ e1)
        self.ids = [members[i] for i in np.argsort(ang)]  # counter-clockwise from outside
        self.o, self.n, self.e1, self.e2 = o, n, e1, e2
        self.d = float(n @ o)
        self.center = unit(o)
        self.local = {i: np.array([(V[i] - o) @ e1, (V[i] - o) @ e2]) for i in self.ids}
        planes = []
        for a, b in self.edges():
            m = np.cross(V[a], V[b])
            planes.append(m if m @ o > 0 else -m)
        self.planes = np.array(planes)

    def edges(self):
        return [(self.ids[i], self.ids[(i + 1) % len(self.ids)]) for i in range(len(self.ids))]


class Poly:
    def __init__(self, shape, spin=0):
        self.V = _vertices(shape, spin)
        self.faces = [Face(self.V, m) for m in _hull_faces(self.V)]
        self.edge_faces = {}
        for fi, f in enumerate(self.faces):
            for a, b in f.edges():
                self.edge_faces.setdefault(frozenset((a, b)), []).append(fi)
        self.nbrs = []
        for fi, f in enumerate(self.faces):
            row = []
            for a, b in f.edges():
                key = frozenset((a, b))
                g = [x for x in self.edge_faces[key] if x != fi][0]
                row.append((g, key))
            self.nbrs.append(row)



def _unfold(poly, root, parent):
    """Rigid 2D transform (R, t) per face so tree edges line up in the plane."""
    T = {root: (np.eye(2), np.zeros(2))}
    children = {}
    for c, (p, key) in parent.items():
        children.setdefault(p, []).append((c, key))
    q = deque([root])
    while q:
        p = q.popleft()
        Rp, tp = T[p]
        for c, key in children.get(p, []):
            a, b = sorted(key)
            A = Rp @ poly.faces[p].local[a] + tp
            B = Rp @ poly.faces[p].local[b] + tp
            ac, bc = poly.faces[c].local[a], poly.faces[c].local[b]
            theta = math.atan2(*(B - A)[::-1]) - math.atan2(*(bc - ac)[::-1])
            R = rot2(theta)
            T[c] = (R, A - R @ ac)
            q.append(c)
    return T


def _face_poly(poly, fi, T):
    R, t = T[fi]
    return np.array([R @ poly.faces[fi].local[i] + t for i in poly.faces[fi].ids])


def _convex_overlap(A, B, eps=1e-7):
    for P in (A, B):
        for i in range(len(P)):
            e = P[(i + 1) % len(P)] - P[i]
            L = np.linalg.norm(e)
            if L < 1e-12:
                continue
            ax = np.array([-e[1], e[0]]) / L
            pa, pb = A @ ax, B @ ax
            if pa.max() <= pb.min() + eps or pb.max() <= pa.min() + eps:
                return False
    return True


def _any_overlap(polys):
    lo = np.array([p.min(axis=0) for p in polys])
    hi = np.array([p.max(axis=0) for p in polys])
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if (lo[i] < hi[j]).all() and (lo[j] < hi[i]).all():
                if _convex_overlap(polys[i], polys[j]):
                    return True
    return False


def _trees(poly):
    """Spanning trees from BFS/DFS over every root and neighbour order, de-duplicated."""
    seen, out = set(), []
    maxdeg = max(len(n) for n in poly.nbrs)
    for root in range(len(poly.faces)):
        for off in range(maxdeg):
            def nb(f):
                L = poly.nbrs[f]
                k = off % len(L)
                return L[k:] + L[:k]

            for mode in ("bfs", "dfs"):
                parent = {}
                if mode == "bfs":
                    visited = {root}
                    q = deque([root])
                    while q:
                        f = q.popleft()
                        for g, key in nb(f):
                            if g not in visited:
                                visited.add(g)
                                parent[g] = (f, key)
                                q.append(g)
                else:
                    visited = {root}

                    def visit(f):
                        for g, key in nb(f):
                            if g not in visited:
                                visited.add(g)
                                parent[g] = (f, key)
                                visit(g)

                    visit(root)
                sig = frozenset(k for _, k in parent.values())
                if sig not in seen:
                    seen.add(sig)
                    out.append((root, parent))

    # shortest-path ("radial") unfoldings: every face hangs off the neighbour
    # nearest the root, so the net grows evenly outward and stays compact
    F = len(poly.faces)
    C = np.array([f.center for f in poly.faces])
    extra = []
    for root in range(F):
        dist = np.arccos(np.clip(C @ C[root], -1, 1))
        for pick in (min, max):
            parent = {}
            for g in range(F):
                if g == root:
                    continue
                closer = [(round(float(dist[h]), 9), h, key) for h, key in poly.nbrs[g] if dist[h] < dist[g] - 1e-9]
                best = min(c[0] for c in closer)
                _, h, key = pick((c for c in closer if c[0] == best), key=lambda c: c[1])
                parent[g] = (h, key)
            extra.append((root, parent))

    # seeded random spanning trees widen the search for many-faced shapes
    if F > 12:
        rng = np.random.default_rng(7)
        edges = [(tuple(fs), key) for key, fs in poly.edge_faces.items()]
        for _ in range(300):
            order = rng.permutation(len(edges))
            uf = list(range(F))

            def find(x):
                while uf[x] != x:
                    uf[x] = uf[uf[x]]
                    x = uf[x]
                return x

            adj = {f: [] for f in range(F)}
            for i in order:
                (a, b), key = edges[i]
                ra, rb = find(a), find(b)
                if ra != rb:
                    uf[ra] = rb
                    adj[a].append((b, key))
                    adj[b].append((a, key))
            root = int(rng.integers(F))
            parent, q, visited = {}, deque([root]), {root}
            while q:
                f = q.popleft()
                for g, key in adj[f]:
                    if g not in visited:
                        visited.add(g)
                        parent[g] = (f, key)
                        q.append(g)
            extra.append((root, parent))

    for root, parent in extra:
        sig = frozenset(k for _, k in parent.values())
        if sig not in seen:
            seen.add(sig)
            out.append((root, parent))
    return out


def _icosa_strip_trees(poly):
    """The 60 classic 5-10-5 strip nets (any corner on top, 5 seam positions)."""
    V = poly.V
    adj = (V @ V.T > 0.3) & (V @ V.T < 0.6)
    face_of = {frozenset(f.ids): i for i, f in enumerate(poly.faces)}
    out = []
    for top in range(12):
        ring = [int(j) for j in np.flatnonzero(adj[top])]
        order = [ring[0]]
        while len(order) < 5:
            order.append(next(j for j in ring if adj[order[-1], j] and j not in order))
        bottom = int(np.argmin(V @ V[top]))
        for rot in range(5):
            U = order[rot:] + order[:rot]
            L = [next(j for j in range(12) if j != top and adj[U[i], j] and adj[U[(i + 1) % 5], j]) for i in range(5)]
            seq = []  # zig-zag through the middle band, caps hang off it
            for i in range(5):
                seq.append((frozenset((U[i], U[(i + 1) % 5], L[i])),))
                seq.append((frozenset((L[i], L[(i + 1) % 5], U[(i + 1) % 5])),))
            band = [face_of[s[0]] for s in seq]
            root = band[0]
            parent = {}
            for a, b in zip(band, band[1:]):
                key = frozenset(poly.faces[a].ids) & frozenset(poly.faces[b].ids)
                parent[b] = (a, key)
            for i in range(5):
                cap = face_of[frozenset((top, U[i], U[(i + 1) % 5]))]
                parent[cap] = (band[2 * i], frozenset((U[i], U[(i + 1) % 5])))
                cap = face_of[frozenset((bottom, L[i], L[(i + 1) % 5]))]
                parent[cap] = (band[2 * i + 1], frozenset((L[i], L[(i + 1) % 5])))
            out.append((root, parent, top))
    return out


def _fit_rotation(poly, T):
    """Rotation that fits a landscape page best, preferring north-up among near-ties."""
    pts = np.concatenate([_face_poly(poly, fi, T) for fi in range(len(poly.faces))])
    north = np.array([0.0, 0.0, 1.0])
    dirs, wts = [], []
    for fi, f in enumerate(poly.faces):
        t = north - (north @ f.center) * f.center
        w = np.linalg.norm(t)
        if w < 1e-6:
            continue
        d = T[fi][0] @ np.array([t @ f.e1, t @ f.e2]) / w
        dirs.append(d)
        wts.append(w)
    dirs, wts = np.array(dirs), np.array(wts)

    # outline edges, to prefer rotations that square the net up with the page
    segs = []
    for fi in range(len(poly.faces)):
        P = _face_poly(poly, fi, T)
        segs.append(np.roll(P, -1, axis=0) - P)
    segs = np.concatenate(segs)
    seg_len = np.linalg.norm(segs, axis=1)
    seg_ang = np.degrees(np.arctan2(segs[:, 1], segs[:, 0]))
    angles = set(range(0, 360, 3))
    for a in np.unique(np.round(seg_ang % 90, 2)):
        angles.update(float(-a + k * 90) % 360 for k in range(4))

    best = []
    for deg in sorted(angles):
        R = rot2(math.radians(deg))
        p = pts @ R.T
        w, h = np.ptp(p[:, 0]), np.ptp(p[:, 1])
        scale = min(PAGE_ASPECT / w, 1 / h)
        up = float(((dirs @ R.T)[:, 1] * wts).sum() / wts.sum()) if len(wts) else 0.0
        off = np.abs(((seg_ang + deg) % 90 + 45) % 90 - 45)
        aligned = float(seg_len[off < 1].sum() / seg_len.sum())
        best.append((scale, up + 0.5 * aligned, deg))
    top = max(s for s, _, _ in best)
    scale, _, deg = max((b for b in best if b[0] >= 0.85 * top), key=lambda b: b[1])
    return math.radians(deg), scale


def _edge_land(poly, land_fn):
    out = {}
    for key in poly.edge_faces:
        a, b = sorted(key)
        pts = slerp(poly.V[a], poly.V[b], 48)
        out[key] = float(land_fn(*latlng(pts)).mean())
    return out


def candidates(shape, land_fn):
    """All valid nets for a shape, best first. JSON-serialisable."""
    spins = SHAPES[shape]["spins"]
    if shape == "gores":
        cands = []
        for off in spins:
            land = []
            for i in range(N_GORES):
                lng = off - 180 + i * 360 / N_GORES
                lat = np.linspace(-80, 80, 64)
                land.append(land_fn(lat, np.full_like(lat, (lng + 180) % 360 - 180)).mean())
            cands.append({"spin": off, "land": round(float(np.mean(land)), 4), "scale": 1.0})
        cands.sort(key=lambda c: c["land"])
        return cands

    poly = Poly(shape, 0)
    trees = _icosa_strip_trees(poly) if shape == "icosahedron" else [(r, p, None) for r, p in _trees(poly)]
    nets = []
    for root, parent, anchor_v in trees:
        T = _unfold(poly, root, parent)
        if _any_overlap([_face_poly(poly, fi, T) for fi in range(len(poly.faces))]):
            continue
        angle, scale = _fit_rotation(poly, T)
        nets.append({
            "root": root, "angle": angle, "scale": scale, "anchor_v": anchor_v,
            "parent": {str(c): [p, sorted(k)] for c, (p, k) in parent.items()},
            "folds": [sorted(k) for _, k in parent.values()],
        })
    top_scale = max(n["scale"] for n in nets)
    cands = []
    for spin in spins:
        sp = Poly(shape, spin) if spin else poly
        el = _edge_land(sp, land_fn)
        for n in nets:
            folds = {frozenset(k) for k in n["folds"]}
            cut = [v for k, v in el.items() if k not in folds]
            c = {k: v for k, v in n.items() if k != "folds"}
            c.update(spin=spin, land=round(float(np.mean(cut)), 4))
            cands.append(c)
    # page fit first (in 5 % steps), then least land cut
    cands.sort(key=lambda c: (-math.floor(c["scale"] / top_scale * 20 + 1e-9), c["land"]))
    return cands


# ---------------------------------------------------------------- panels

class GnomonicPanel:
    curved = False

    def __init__(self, face, R, t):
        self.f, self.R, self.t = face, R, t
        self.planes = face.planes

    def project(self, P):
        f = self.f
        Q = P * (f.d / (P @ f.n))[:, None]
        L = np.stack([(Q - f.o) @ f.e1, (Q - f.o) @ f.e2], axis=1)
        return L @ self.R.T + self.t


class GorePanel:
    curved = True

    def __init__(self, lng0, half, tx):
        self.lng0, self.half, self.tx = math.radians(lng0), half, tx
        lL, lR = self.lng0 - half, self.lng0 + half
        self.planes = np.array([[-math.sin(lL), math.cos(lL), 0.0], [math.sin(lR), -math.cos(lR), 0.0]])

    def project(self, P):
        c, s = math.cos(self.lng0), math.sin(self.lng0)
        x = P[:, 0] * c + P[:, 1] * s
        y = -P[:, 0] * s + P[:, 1] * c
        lat = np.arcsin(np.clip(P[:, 2], -1, 1))
        dl = np.arctan2(y, x)
        return np.stack([dl * np.cos(lat) + self.tx, lat], axis=1)


def _tab(A, B, inside, depth, angle_a, angle_b=None):
    """Trapezoid glue flap on edge A-B, on the side away from `inside`.

    angle_a / angle_b are the slopes of the two ends against the edge."""
    angle_b = angle_a if angle_b is None else angle_b
    d = B - A
    L = np.linalg.norm(d)
    u = d / L
    nrm = np.array([-u[1], u[0]])
    if ((A + B) / 2 - inside) @ nrm < 0:
        nrm = -nrm
    ca = 1 / math.tan(math.radians(angle_a))
    cb = 1 / math.tan(math.radians(angle_b))
    depth = min(depth, 0.85 * L / (ca + cb))  # keep a flat top on short edges
    return np.array([A, A + u * depth * ca + nrm * depth, B - u * depth * cb + nrm * depth, B])


def _face_number(shape, face):
    lat, lng = latlng(face.center)
    if shape == "icosahedron":
        return int(h3.get_icosahedron_faces(h3.latlng_to_cell(float(lat), float(lng), 9))[0])
    if shape == "cube":
        c = face.center
        k = int(np.argmax(np.abs(c)))
        return k + (3 if c[k] < 0 else 0)  # S2 face ids
    return None


def build(shape, cand):
    """Panels, outlines, fold/cut lines, tabs and globe overlay for one candidate net."""
    if shape == "gores":
        return _build_gores(cand)
    poly = Poly(shape, cand["spin"])
    parent = {int(c): (p, frozenset(k)) for c, (p, k) in cand["parent"].items()}
    T = _unfold(poly, cand["root"], parent)
    G = rot2(cand["angle"])
    T = {fi: (G @ R, G @ t) for fi, (R, t) in T.items()}
    folds_k = {k for _, k in parent.values()}

    panels, outlines = [], []
    for fi, f in enumerate(poly.faces):
        R, t = T[fi]
        panels.append(GnomonicPanel(f, R, t))
        outlines.append(_face_poly(poly, fi, T))
    numbers = [_face_number(shape, f) for f in poly.faces]
    if any(n is None for n in numbers):
        order = sorted(range(len(outlines)), key=lambda i: (-round(outlines[i].mean(axis=0)[1], 1), outlines[i].mean(axis=0)[0]))
        numbers = [0] * len(order)
        for rank, fi in enumerate(order):
            numbers[fi] = rank + 1

    folds, cuts, tabs = [], [], []
    copies = {}
    for fi, f in enumerate(poly.faces):
        R, t = T[fi]
        for a, b in f.edges():
            key = frozenset((a, b))
            A, B = R @ f.local[a] + t, R @ f.local[b] + t
            if key in folds_k:
                if fi == parent_of(parent, key):
                    folds.append(np.array([A, B]))
            else:
                copies.setdefault(key, []).append((fi, A, B))

    allpts = np.concatenate(outlines)
    extent = max(np.ptp(allpts[:, 0]), np.ptp(allpts[:, 1]) * PAGE_ASPECT)
    edge = min(np.linalg.norm(A - B) for pair in copies.values() for _, A, B in pair)
    depth0 = min(TAB_DEPTH * edge, extent * TAB_MM / 277)
    placed = []
    for key, pair in copies.items():
        choice = None
        ends = sorted(((a, b) for a in TAB_ANGLES for b in TAB_ANGLES), key=lambda e: -min(e) - 0.1 * max(e))
        tries = [(depth0 * k, a, b) for k in (1, 0.6) for a, b in ends]
        for depth, ang_a, ang_b in tries:
            for fi, A, B in pair:
                tab = _tab(A, B, outlines[fi].mean(axis=0), depth, ang_a, ang_b)
                clash = any(_convex_overlap(tab, o) for o in outlines) or any(_convex_overlap(tab, p) for p in placed)
                if not clash:
                    choice = (fi, A, B, tab)
                    break
            if choice:
                break
        for fi, A, B in pair:
            if choice and fi == choice[0]:
                folds.append(np.array([A, B]))
            else:
                cuts.append(np.array([A, B]))
        if choice:
            placed.append(choice[3])
            tabs.append(choice[3])

    globe = []
    for key in poly.edge_faces:
        a, b = sorted(key)
        globe.append({"cut": key not in folds_k, "pts": slerp(poly.V[a], poly.V[b], 40)})

    return {
        "panels": panels, "outlines": outlines, "numbers": numbers,
        "folds": folds, "cuts": cuts, "tabs": tabs, "globe": globe,
        "anchor": poly.V[cand["anchor_v"]] if cand.get("anchor_v") is not None else poly.faces[cand["root"]].center,
        "vertices": poly.V,
    }


def parent_of(parent, key):
    """The child face that owns a tree edge (draw each fold once)."""
    for c, (p, k) in parent.items():
        if k == key:
            return c
    return None


def _build_gores(cand):
    half = math.pi / N_GORES
    width = 2 * half
    lat = np.radians(np.linspace(-90, 90, 181))
    panels, outlines, cuts, tabs = [], [], [], []
    for i in range(N_GORES):
        lng0 = cand["spin"] - 180 + (i + 0.5) * 360 / N_GORES
        tx = (i + 0.5) * width
        panels.append(GorePanel(lng0, half, tx))
        right = np.stack([tx + half * np.cos(lat), lat], axis=1)
        left = np.stack([tx - half * np.cos(lat), lat], axis=1)
        outlines.append(np.concatenate([right, left[::-1]]))
        cuts.append(right)
        cuts.append(left)

    # glue teeth along each gore's right edge, dropped where the neighbour is too close
    for i in range(N_GORES):
        tx = (i + 0.5) * width
        nxt = outlines[(i + 1) % N_GORES] if i + 1 < N_GORES else None
        for hemi in (1, -1):
            edges = np.radians(np.arange(6, 84, 8))
            for a0, a1 in zip(edges, edges[1:]):
                a0, a1 = hemi * a0, hemi * a1
                A = np.array([tx + half * math.cos(a0), a0])
                B = np.array([tx + half * math.cos(a1), a1])
                tab = _tab(A, B, np.array([tx, (a0 + a1) / 2]), 0.28 * np.linalg.norm(B - A), 54)
                if nxt is not None and _convex_overlap(tab, nxt):
                    continue
                tabs.append(tab)

    globe = []
    for i in range(N_GORES):
        lng = cand["spin"] - 180 + i * 360 / N_GORES
        pts = xyz(np.linspace(-90, 90, 91), np.full(91, lng))
        globe.append({"cut": True, "pts": pts})
    return {
        "panels": panels, "outlines": outlines, "numbers": list(range(1, N_GORES + 1)),
        "folds": [], "cuts": cuts, "tabs": tabs, "globe": globe, "anchor": None, "vertices": None,
    }
