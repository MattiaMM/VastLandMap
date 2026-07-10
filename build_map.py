import json
import math
import pathlib

import pymupdf

PDF = "OGA Great Map - 22x22 - v06.pdf"
DPI = 150
Z = DPI / 72.0


def render_page(pdf_path=PDF, dpi=DPI):
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    z = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z))
    png = pix.tobytes("png")
    pathlib.Path("map.png").write_bytes(png)
    return page


def extract_spans(page):
    d = page.get_text("dict")
    out = []
    for b in d["blocks"]:
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            for s in line.get("spans", []):
                txt = s.get("text", "")
                if not txt.strip():
                    continue
                x0, y0, x1, y1 = s["bbox"]
                out.append({
                    "text": txt,
                    "font": s.get("font", ""),
                    "size": float(s.get("size", 0)),
                    "color": int(s.get("color", 0)),
                    "pdf_x": float((x0 + x1) / 2),
                    "pdf_y": float((y0 + y1) / 2),
                    "w": float(x1 - x0),
                    "h": float(y1 - y0),
                })
    return out


def classify_span(span):
    font = span.get("font", "")
    size = span.get("size", 0.0)
    color = span.get("color", 0)

    # drop rules first
    if "Cartographi" in font:
        return None
    if "Bold" in font and 8.5 <= size <= 9.5:
        return None  # HS## hex codes
    if "Bold" in font and 5.0 <= size <= 7.5:
        return None  # institutional short labels
    if "Bold" in font and 12.0 <= size <= 13.0:
        return None  # institutional labels

    # classification (precedence order matters)
    if "AvenirNextCondensed-Demi" in font and size >= 10:
        return "capital"
    if "AvenirNextCondensed-Demi" in font and 6.0 <= size <= 9.0:
        return "landmark"
    if "Garamond-Bold" in font and size >= 18:
        return "land"
    if "Garamond-Bold" in font and 10.5 <= size <= 11.5 and color in (0, 65793):
        return "landmark"
    if ("Garamond-Medium" in font or "Garamond-Italic" in font) and size >= 11:
        return "region"
    if "Garamond-SemiBo" in font and 5.5 <= size <= 6.5 and color == 10799:
        return "constellation"
    if "Garamond-SemiBo" in font and 5.5 <= size <= 6.5:
        return "settlement"
    if "Garamond-SemiBo" in font and 6.5 < size <= 7.0:
        return "settlement"
    if ("Garamond-Medium" in font or "Garamond-Italic" in font) and 8.0 <= size <= 9.0:
        return "settlement"
    if "AvenirNextCondensed-Ital" in font and 5.0 <= size <= 7.5:
        return "province"
    if "AvenirNextCondensed-Medi" in font and 6.0 <= size <= 8.5:
        return "settlement"
    return None


def cluster_fragments(spans, eps_factor=1.8):
    spans = [s for s in spans if s.get("text", "").strip()]
    n = len(spans)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # mutual reachability: both must be within each other's radius
    # (single-link chaining merges whole regions; min() is much tighter)
    for i in range(n):
        ri = spans[i]["size"] * eps_factor
        for j in range(i + 1, n):
            rj = spans[j]["size"] * eps_factor
            dx = spans[i]["pdf_x"] - spans[j]["pdf_x"]
            dy = spans[i]["pdf_y"] - spans[j]["pdf_y"]
            d2 = dx * dx + dy * dy
            if d2 <= min(ri, rj) ** 2:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    labels = []
    for idx_list in groups.values():
        members = [spans[i] for i in idx_list]
        cx = sum(m["pdf_x"] for m in members) / len(members)
        cy = sum(m["pdf_y"] for m in members) / len(members)

        def ang(m):
            return math.atan2(m["pdf_y"] - cy, m["pdf_x"] - cx)

        # reading order via PCA projection (robust for horizontal, vertical, gently curved)
        xs = [m["pdf_x"] - cx for m in members]
        ys = [m["pdf_y"] - cy for m in members]
        sxx = sum(x * x for x in xs)
        syy = sum(y * y for y in ys)
        sxy = sum(xs[i] * ys[i] for i in range(len(xs)))
        denom = sxx - syy
        if abs(denom) < 1e-9 and abs(sxy) < 1e-9:
            vx, vy = 1.0, 0.0
        else:
            theta = 0.5 * math.atan2(2 * sxy, denom)
            vx, vy = math.cos(theta), math.sin(theta)
        # orient so the leftmost member projects smallest (reading L->R)
        leftmost = min(members, key=lambda m: (m["pdf_x"], m["pdf_y"]))
        rightmost = max(members, key=lambda m: (m["pdf_x"], m["pdf_y"]))
        pl = (leftmost["pdf_x"] - cx) * vx + (leftmost["pdf_y"] - cy) * vy
        pr = (rightmost["pdf_x"] - cx) * vx + (rightmost["pdf_y"] - cy) * vy
        if pl > pr:
            vx, vy = -vx, -vy
        members.sort(key=lambda m: (m["pdf_x"] - cx) * vx + (m["pdf_y"] - cy) * vy)
        # Big curved labels (size >= 11) are letter-by-letter: concatenate without
        # separator. Smaller stacked labels are word lines: join with space.
        if any(m.get("size", 0) >= 11 for m in members):
            name = "".join(m["text"] for m in members)
        else:
            name = " ".join(m["text"] for m in members)
        name = " ".join(name.split()) if " " in name else name.strip()
        if len(name) < 2:
            continue
        labels.append({"name": name, "pdf_x": cx, "pdf_y": cy,
                       "category": members[0].get("category")})
    return labels


CAPITAL_TO_LAND = {
    "silent city": "neutral",
    "emerald city": "green",
    "ruins azure": "blue",
    "violet city": "purple",
    "red end": "red",
    "r.l.d. city": "red",
    "orange city": "orange",
    "saffron city": "yellow",
}


def to_png(label, page_h, y_flip):
    py = (page_h - label["pdf_y"]) * Z if y_flip else label["pdf_y"] * Z
    return label["pdf_x"] * Z, py


def build_anchors(capital_labels, page_h, y_flip):
    anchors = {}
    for lbl in capital_labels:
        key = lbl["name"].strip().lower()
        land = CAPITAL_TO_LAND.get(key)
        if land:
            px, py = to_png(lbl, page_h, y_flip)
            anchors[land] = (px, py)
    return anchors


def assign_land(label, anchors):
    if label["category"] == "constellation":
        return "neutral"
    if label["category"] == "land":
        return "neutral"
    best, best_d = "neutral", float("inf")
    for land, (ax, ay) in anchors.items():
        if land == "neutral":
            continue
        dx, dy = label["png_x"] - ax, label["png_y"] - ay
        d = dx * dx + dy * dy
        if d < best_d:
            best_d, best = d, land
    return best


def anchors_lookup(anchors, name):
    land = CAPITAL_TO_LAND.get(name.strip().lower())
    return land if land else "neutral"


if __name__ == "__main__":
    page = render_page()
    page_h = page.rect.height
    raw = extract_spans(page)

    # Task 1 confirmed: PyMuPDF uses top-left origin => y is top-down => no flip
    y_flip = False

    from collections import defaultdict, Counter
    by_cat = defaultdict(list)
    for s in raw:
        c = classify_span(s)
        if c:
            s["category"] = c
            by_cat[c].append(s)

    all_labels = []
    for cat, items in by_cat.items():
        eps = 3.5 if cat in ("land", "region") else 1.8
        for lbl in cluster_fragments(items, eps_factor=eps):
            lbl["category"] = cat
            all_labels.append(lbl)

    capitals = [l for l in all_labels if l["category"] == "capital"]
    anchors = build_anchors(capitals, page_h, y_flip)
    if len([k for k in anchors if k != "neutral"]) < 6:
        raise SystemExit("ERROR: missing capital anchors: " + repr(sorted(anchors)))

    places = []
    seen = set()
    for lbl in all_labels:
        px, py = to_png(lbl, page_h, y_flip)
        name = lbl["name"]
        key = (name.lower(), round(px, 0), round(py, 0))
        if key in seen:
            continue
        seen.add(key)
        land = anchors_lookup(anchors, name) if lbl["category"] == "capital" else assign_land(
            {"png_x": px, "png_y": py, "category": lbl["category"]}, anchors)
        places.append({"id": "p{}".format(len(places)), "name": name,
                       "category": lbl["category"], "land": land,
                       "x": round(px, 1), "y": round(py, 1)})

    out = {"image": {"width": 3638, "height": 3638}, "places": places}
    json.dump(out, open("places.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("places:", len(places))
    print("by category:", dict(Counter(p["category"] for p in places)))
    print("by land:", dict(Counter(p["land"] for p in places)))
    print("anchors:", {k: (round(v[0]), round(v[1])) for k, v in anchors.items()})