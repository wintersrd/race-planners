from pathlib import Path

from race_planners.grade import extract_aid_stations, extract_aid_stops_km, parse_gpx


def test_parse_gpx_supports_route_points(tmp_path: Path) -> None:
    gpx_path = tmp_path / "route-only.gpx"
    gpx_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <rte>
    <rtept lat="48.0" lon="-4.0"><ele>5</ele></rtept>
    <rtept lat="48.0" lon="-3.99"><ele>10</ele></rtept>
    <rtept lat="48.0" lon="-3.98"><ele>15</ele></rtept>
  </rte>
</gpx>
""",
        encoding="utf-8",
    )

    trackpoints = parse_gpx(str(gpx_path))

    assert len(trackpoints) == 3
    assert trackpoints[-1].distance_from_start > 0
    assert trackpoints[-1].elevation == 15


def test_parse_gpx_supports_https_gpx_namespace(tmp_path: Path) -> None:
    gpx_path = tmp_path / "https-namespace.gpx"
    gpx_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="https://www.topografix.com/GPX/1/1">
  <trk>
    <trkseg>
      <trkpt lat="48.0" lon="-4.0"><ele>5</ele></trkpt>
      <trkpt lat="48.0" lon="-3.99"><ele>10</ele></trkpt>
      <trkpt lat="48.0" lon="-3.98"><ele>15</ele></trkpt>
    </trkseg>
  </trk>
</gpx>
""",
        encoding="utf-8",
    )

    trackpoints = parse_gpx(str(gpx_path))

    assert len(trackpoints) == 3
    assert trackpoints[-1].distance_from_start > 0
    assert trackpoints[-1].elevation == 15


def test_extract_aid_stops_km_uses_waypoint_types_and_ignores_start_finish(tmp_path: Path) -> None:
    gpx_path = tmp_path / "aid-stops.gpx"
    gpx_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <wpt lat="48.0" lon="-4.0"><type>depart</type></wpt>
  <wpt lat="48.0" lon="-3.99"><type>ravitoliquide</type></wpt>
  <wpt lat="48.0" lon="-3.98"><type>arrivee</type></wpt>
  <trk>
    <trkseg>
      <trkpt lat="48.0" lon="-4.0"><ele>5</ele></trkpt>
      <trkpt lat="48.0" lon="-3.995"><ele>5</ele></trkpt>
      <trkpt lat="48.0" lon="-3.99"><ele>5</ele></trkpt>
      <trkpt lat="48.0" lon="-3.985"><ele>5</ele></trkpt>
      <trkpt lat="48.0" lon="-3.98"><ele>5</ele></trkpt>
    </trkseg>
  </trk>
</gpx>
""",
        encoding="utf-8",
    )

    aid_stops_km = extract_aid_stops_km(str(gpx_path))

    assert len(aid_stops_km) == 1
    assert 0.5 < aid_stops_km[0] < 1.5


def test_extract_aid_stations_returns_typed_metadata(tmp_path: Path) -> None:
    gpx_path = tmp_path / "typed-aid-stops.gpx"
    gpx_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <wpt lat="48.0" lon="-3.99">
    <name>Ravito 1</name>
    <type>ravitoliquide</type>
  </wpt>
  <trk>
    <trkseg>
      <trkpt lat="48.0" lon="-4.0"><ele>5</ele></trkpt>
      <trkpt lat="48.0" lon="-3.99"><ele>5</ele></trkpt>
      <trkpt lat="48.0" lon="-3.98"><ele>5</ele></trkpt>
    </trkseg>
  </trk>
</gpx>
""",
        encoding="utf-8",
    )

    aid_stations = extract_aid_stations(str(gpx_path))

    assert len(aid_stations) == 1
    assert aid_stations[0].label == "Ravito 1"
    assert aid_stations[0].source == "gpx_waypoint"
    assert aid_stations[0].waypoint_type == "ravitoliquide"
