from __future__ import annotations

import io
import json
import pathlib
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout

from lyapunov_shield import WatchdogTimer
from lyapunov_shield.cli import main as cli_main
from lyapunov_shield.svg_atlas import (
    ATLAS_SPECS,
    audit_svg_geometry,
    parse_svg_transform,
    render_svg_atlas,
)


class TestSVGAtlasAndGeometryGate(unittest.TestCase):
    """Cycle 7: Verify all 10 pure-stdlib publication SVGs in `figures/` via
    `xml.etree.ElementTree`, recursive `<g transform>` accumulation, Bezier control-hull
    AABBs, and `<text>` collision heuristics (`OOB == 0`, `overlaps == 0`).
    """

    def test_01_render_and_audit_all_ten_atlas_svgs(self) -> None:
        with WatchdogTimer(20.0):
            repo_root = pathlib.Path(__file__).resolve().parent.parent
            fig_dir = repo_root / "figures"
            report = render_svg_atlas(output_dir=fig_dir)

            self.assertEqual(report["svg_count"], 10)
            self.assertEqual(
                report["total_oob"],
                0,
                f"Expected 0 OOB elements across 10 SVGs, got: {report['files']}",
            )
            self.assertEqual(
                report["total_text_overlaps"],
                0,
                f"Expected 0 text-text overlaps across 10 SVGs, got: {report['files']}",
            )

            expected_names = [spec[0] for spec in ATLAS_SPECS]
            for fname in expected_names:
                svg_path = fig_dir / fname
                self.assertTrue(svg_path.exists(), f"Missing generated SVG: {svg_path}")
                raw = svg_path.read_text(encoding="utf-8")
                # Must parse cleanly as strict XML and have non-trivial vector content
                root = ET.fromstring(raw)
                self.assertTrue(root.tag.endswith("svg"))
                self.assertNotIn("/" + "Users" + "/", raw)

                indep_audit = audit_svg_geometry(svg_path)
                self.assertGreater(indep_audit["element_count"], 15, fname)
                self.assertGreater(indep_audit["text_count"], 6, fname)
                self.assertEqual(
                    indep_audit["oob_count"],
                    0,
                    f"{fname} OOB items: {indep_audit['oob_items']}",
                )
                self.assertEqual(
                    indep_audit["text_overlap_count"],
                    0,
                    f"{fname} text overlaps: {indep_audit['text_overlaps']}",
                )

    def test_02_geometry_audit_catches_oob_transforms_beziers_and_text_overlaps(self) -> None:
        """Red-team the SVG geometry auditor itself to ensure nested transforms, Bezier
        control hulls, primitives, and overlapping texts are deterministically caught.
        """
        # 1) Nested <g transform="translate(...) scale(...)"> pushing a <rect> outside viewBox
        svg_oob_group = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">
          <g transform="translate(150, 20)">
            <g transform="scale(2, 1)">
              <rect x="10" y="10" width="20" height="20" fill="#000"/>
            </g>
          </g>
        </svg>"""
        # Transformed x1 = 150 + 2 * (10 + 20) = 210 > 200!
        aud_g = audit_svg_geometry(svg_oob_group)
        self.assertEqual(aud_g["oob_count"], 1)

        # 2) Cubic & Quadratic Bezier control points protruding outside viewBox even when endpoints stay inside
        svg_oob_bezier = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">
          <path d="M 20 50 C 40 -15, 160 50, 180 50 Q 100 120 20 80" fill="none" stroke="#000" stroke-width="2"/>
        </svg>"""
        aud_b = audit_svg_geometry(svg_oob_bezier)
        self.assertEqual(aud_b["oob_count"], 1)

        # 3) Overlapping <text> boxes
        svg_text_overlap = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 100">
          <text x="50" y="50" font-size="14" text-anchor="start"> Lyap Certificate Alpha </text>
          <text x="95" y="52" font-size="14" text-anchor="start"> Overlapping Label </text>
        </svg>"""
        aud_t = audit_svg_geometry(svg_text_overlap)
        self.assertEqual(aud_t["oob_count"], 0)
        self.assertEqual(aud_t["text_overlap_count"], 1)

        # 4) Transform composition check
        m = parse_svg_transform("translate(10, 20) scale(2, 3)")
        self.assertEqual(m, (2.0, 0.0, 0.0, 3.0, 10.0, 20.0))

    def test_03_cli_render_atlas_subcommand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main(["render-atlas", "--output-dir", tmp])
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["svg_count"], 10)
            self.assertEqual(payload["total_oob"], 0)
            self.assertEqual(payload["total_text_overlaps"], 0)


if __name__ == "__main__":
    unittest.main()
