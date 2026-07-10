import build_map as bm


def test_extract_spans_returns_dicts_with_required_keys(monkeypatch):
    class FakePage:
        def get_text(self, kind):
            return {"blocks": [
                {"type": 0, "lines": [
                    {"spans": [{"text": "Foo", "font": "X", "size": 6.0,
                                "color": 0, "bbox": (100, 200, 120, 206),
                                "origin": (100, 200)}]}]}]}
    spans = bm.extract_spans(FakePage())
    assert len(spans) == 1
    assert set(spans[0].keys()) == {"text", "font", "size", "color", "pdf_x", "pdf_y", "w", "h"}
    assert spans[0]["text"] == "Foo"


def span(text="X", font="", size=6.0, color=0):
    return {"text": text, "font": font, "size": size, "color": color,
            "pdf_x": 0, "pdf_y": 0, "w": 0, "h": 0}


def test_capital_demi_large():
    assert bm.classify_span(span(font="AvenirNextCondensed-Demi", size=11.0)) == "capital"


def test_land_bold_giant():
    assert bm.classify_span(span(font="CormorantGaramond-Bold", size=24.8)) == "land"


def test_region_medium_serif_large():
    assert bm.classify_span(span(font="CormorantGaramond-Medium", size=11.0)) == "region"
    assert bm.classify_span(span(font="CormorantGaramond-Italic", size=13.8)) == "region"


def test_province_avenir_italic_small():
    assert bm.classify_span(span(font="AvenirNextCondensed-Italic", size=7.0)) == "province"


def test_settlement_avenir_medi():
    assert bm.classify_span(span(font="AvenirNextCondensed-Medi", size=6.0)) == "settlement"


def test_settlement_semibold_serif():
    assert bm.classify_span(span(font="CormorantGaramond-SemiBo", size=6.0, color=0)) == "settlement"


def test_constellation_semibold_teal():
    assert bm.classify_span(span(font="CormorantGaramond-SemiBo", size=6.0, color=10799)) == "constellation"


def test_landmark_bold_11_black():
    assert bm.classify_span(span(font="CormorantGaramond-Bold", size=11.0, color=0)) == "landmark"


def test_drop_cartographic_font():
    assert bm.classify_span(span(font="OurGoldenAge-Cartographi", size=10.0)) is None


def test_drop_hex_codes():
    assert bm.classify_span(span(font="CormorantGaramond-Bold", size=9.0, color=0)) is None


def test_constellation_checked_before_settlement():
    assert bm.classify_span(span(font="CormorantGaramond-SemiBo", size=6.0, color=10799)) == "constellation"


def test_two_adjacent_fragments_merge():
    spans = [
        span(text="OR", font="CormorantGaramond-Bold", size=24.8),
        span(text="ANGE", font="CormorantGaramond-Bold", size=24.8),
    ]
    spans[0]["pdf_x"], spans[0]["pdf_y"] = 100, 100
    spans[1]["pdf_x"], spans[1]["pdf_y"] = 130, 100
    labels = bm.cluster_fragments(spans)
    assert len(labels) == 1
    assert labels[0]["name"] == "ORANGE"
    assert abs(labels[0]["pdf_x"] - 115) < 1


def test_distant_fragments_stay_separate():
    spans = [
        span(text="Alpha", font="CormorantGaramond-Bold", size=24.8),
        span(text="Beta", font="CormorantGaramond-Bold", size=24.8),
    ]
    spans[0]["pdf_x"], spans[0]["pdf_y"] = 100, 100
    spans[1]["pdf_x"], spans[1]["pdf_y"] = 900, 900
    labels = bm.cluster_fragments(spans)
    assert len(labels) == 2
    assert {l["name"] for l in labels} == {"Alpha", "Beta"}


def test_single_span_passes_through():
    spans = [span(text="Threeheads", font="CormorantGaramond-SemiBo", size=6.0, color=0)]
    spans[0]["pdf_x"], spans[0]["pdf_y"] = 100, 100
    labels = bm.cluster_fragments(spans)
    assert len(labels) == 1 and labels[0]["name"] == "Threeheads"


def test_name_strips_and_collapses_whitespace():
    spans = [
        span(text="  Great ", font="AvenirNextCondensed-Medi", size=6.0),
        span(text="Chantlery", font="AvenirNextCondensed-Medi", size=6.0),
    ]
    spans[0]["pdf_x"], spans[0]["pdf_y"] = 100, 100
    spans[1]["pdf_x"], spans[1]["pdf_y"] = 105, 100
    labels = bm.cluster_fragments(spans)
    assert labels[0]["name"] == "Great Chantlery"


def test_assign_land_nearest_capital():
    anchors = {
        "neutral": (1200, 1200),
        "green": (800, 50),
        "blue": (50, 50),
    }
    near_green = {"name": "X", "category": "settlement", "png_x": 810, "png_y": 60}
    near_blue = {"name": "X", "category": "settlement", "png_x": 60, "png_y": 40}
    assert bm.assign_land(near_green, anchors) == "green"
    assert bm.assign_land(near_blue, anchors) == "blue"


def test_assign_land_constellation_neutral_regardless_of_distance():
    anchors = {"green": (800, 50)}
    lbl = {"name": "X", "category": "constellation", "png_x": 801, "png_y": 51}
    assert bm.assign_land(lbl, anchors) == "neutral"
