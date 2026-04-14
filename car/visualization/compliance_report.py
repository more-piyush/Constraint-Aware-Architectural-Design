"""HTML compliance report generation."""

from __future__ import annotations

import base64
from pathlib import Path

from car.models.constraints import SiteConstraints
from car.models.results import InferenceResult


class ComplianceReportGenerator:
    """Generates HTML compliance reports."""

    def generate(
        self,
        result: InferenceResult,
        constraints: SiteConstraints,
        output_path: str | Path = "compliance_report.html",
        network_graph_path: str | Path | None = None,
        floor_plan_path: str | Path | None = None,
    ) -> None:
        """Generate an HTML report with compliance details."""
        map_d = result.map_design
        design = map_d.design
        compliance = map_d.compliance

        # Build violations table rows
        violation_rows = ""
        if compliance.violations:
            for v in compliance.violations:
                severity_color = "#FF4444" if v.severity == "hard" else "#FFA500"
                violation_rows += f"""
                <tr>
                    <td>{v.constraint_name}</td>
                    <td>{v.constraint_type}</td>
                    <td>{v.required_value}</td>
                    <td>{v.actual_value}</td>
                    <td style="color:{severity_color};font-weight:bold">{v.severity.upper()}</td>
                </tr>"""
        else:
            violation_rows = '<tr><td colspan="5" style="text-align:center;color:#228B22">All constraints satisfied</td></tr>'

        # Embed images if available
        network_img = self._embed_image(network_graph_path) if network_graph_path else ""
        floor_plan_img = self._embed_image(floor_plan_path) if floor_plan_path else ""

        # Confidence gauge color
        conf = compliance.confidence_score
        if conf >= 0.8:
            gauge_color = "#228B22"
        elif conf >= 0.5:
            gauge_color = "#FFA500"
        else:
            gauge_color = "#FF4444"

        # Sampled designs summary
        sampled_rows = ""
        for sd in result.sampled_designs[:10]:
            status = "PASS" if sd.compliance.is_compliant else "FAIL"
            status_color = "#228B22" if sd.compliance.is_compliant else "#FF4444"
            sampled_rows += f"""
            <tr>
                <td>#{sd.iteration_id}</td>
                <td>{sd.design.structural_system.value}</td>
                <td>{sd.design.num_floors}</td>
                <td>{sd.design.window_size.value}</td>
                <td>{sd.design.roof_type.value}</td>
                <td>{sd.overall_score:.3f}</td>
                <td style="color:{status_color}">{status}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CAR Compliance Report</title>
<style>
    body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 40px; background: #FAFAFA; color: #333; }}
    h1 {{ color: #2C3E50; border-bottom: 3px solid #3498DB; padding-bottom: 10px; }}
    h2 {{ color: #2C3E50; margin-top: 30px; }}
    .summary-box {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; gap: 30px; align-items: center; }}
    .gauge {{ text-align: center; }}
    .gauge-value {{ font-size: 48px; font-weight: bold; color: {gauge_color}; }}
    .gauge-label {{ font-size: 14px; color: #666; }}
    .status {{ font-size: 24px; font-weight: bold; padding: 10px 20px; border-radius: 5px; }}
    .status.pass {{ background: #E8F5E9; color: #228B22; }}
    .status.fail {{ background: #FFEBEE; color: #FF4444; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 5px; overflow: hidden; }}
    th {{ background: #2C3E50; color: white; padding: 12px 15px; text-align: left; }}
    td {{ padding: 10px 15px; border-bottom: 1px solid #EEE; }}
    tr:hover {{ background: #F5F5F5; }}
    .design-params {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .param {{ background: white; padding: 12px; border-radius: 5px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
    .param-label {{ font-size: 12px; color: #999; text-transform: uppercase; }}
    .param-value {{ font-size: 18px; font-weight: bold; color: #2C3E50; }}
    .images {{ display: flex; gap: 20px; flex-wrap: wrap; }}
    .images img {{ max-width: 100%; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .meta {{ color: #999; font-size: 12px; margin-top: 30px; }}
</style>
</head>
<body>
<h1>CAR Architectural Design Compliance Report</h1>

<div class="summary-box">
    <div class="gauge">
        <div class="gauge-value">{conf:.0%}</div>
        <div class="gauge-label">Confidence Score</div>
    </div>
    <div>
        <div class="status {'pass' if compliance.is_compliant else 'fail'}">
            {'COMPLIANT' if compliance.is_compliant else 'NON-COMPLIANT'}
        </div>
        <p>{compliance.passed_constraints_count}/{compliance.checked_constraints_count} constraints satisfied</p>
    </div>
    <div>
        <div class="gauge-label">Aesthetic Score</div>
        <div style="font-size:24px;font-weight:bold">{map_d.aesthetic_score:.2f}</div>
    </div>
    <div>
        <div class="gauge-label">View Score</div>
        <div style="font-size:24px;font-weight:bold">{map_d.view_score:.2f}</div>
    </div>
    <div>
        <div class="gauge-label">Overall Score</div>
        <div style="font-size:24px;font-weight:bold">{map_d.overall_score:.2f}</div>
    </div>
</div>

<h2>MAP Design Parameters</h2>
<div class="design-params">
    <div class="param"><div class="param-label">Structural System</div>
        <div class="param-value">{design.structural_system.value.replace('_', ' ').title()}</div></div>
    <div class="param"><div class="param-label">Floors</div>
        <div class="param-value">{design.num_floors} ({design.building_height_m:.1f}m)</div></div>
    <div class="param"><div class="param-label">Floor Area</div>
        <div class="param-value">{design.floor_area_sqm:.0f} sqm</div></div>
    <div class="param"><div class="param-label">Wall Type</div>
        <div class="param-value">{design.wall_type.value.replace('_', ' ').title()} ({design.wall_thickness_mm:.0f}mm)</div></div>
    <div class="param"><div class="param-label">Windows</div>
        <div class="param-value">{design.window_size.value.title()}</div></div>
    <div class="param"><div class="param-label">Roof</div>
        <div class="param-value">{design.roof_type.value.replace('_', ' ').title()}</div></div>
    <div class="param"><div class="param-label">Footprint</div>
        <div class="param-value">{design.footprint_width_m:.1f}m x {design.footprint_depth_m:.1f}m</div></div>
    <div class="param"><div class="param-label">Material</div>
        <div class="param-value">{design.primary_material.title()}</div></div>
</div>

<h2>Constraint Violations</h2>
<table>
    <tr><th>Constraint</th><th>Type</th><th>Required</th><th>Actual</th><th>Severity</th></tr>
    {violation_rows}
</table>

<h2>Site Constraints</h2>
<table>
    <tr><th>Parameter</th><th>Value</th></tr>
    <tr><td>Site Area</td><td>{constraints.site_area_sqm:.0f} sqm</td></tr>
    <tr><td>FAR Limit</td><td>{constraints.regulatory.far_limit}</td></tr>
    <tr><td>Height Limit</td><td>{constraints.regulatory.height_limit_m:.1f}m</td></tr>
    <tr><td>Airport Zone</td><td>{'Yes' if constraints.regulatory.is_airport_zone else 'No'}</td></tr>
    <tr><td>Setbacks (F/S/R)</td><td>{constraints.regulatory.setback_front_m}/{constraints.regulatory.setback_side_m}/{constraints.regulatory.setback_rear_m}m</td></tr>
    <tr><td>Seismic Zone</td><td>{constraints.geophysical.seismic_zone.value}</td></tr>
    <tr><td>Solar Azimuth</td><td>{constraints.environmental.solar_azimuth_peak_deg:.0f} deg</td></tr>
    <tr><td>Wind Speed</td><td>{constraints.environmental.prevailing_wind_speed_kmh:.0f} km/h</td></tr>
</table>

{"<h2>Design Alternatives (Top 10)</h2>" if sampled_rows else ""}
{"<table><tr><th>ID</th><th>Structure</th><th>Floors</th><th>Windows</th><th>Roof</th><th>Score</th><th>Status</th></tr>" + sampled_rows + "</table>" if sampled_rows else ""}

<div class="images">
    {f'<div><h2>Bayesian Network</h2>{network_img}</div>' if network_img else ''}
    {f'<div><h2>Floor Plan</h2>{floor_plan_img}</div>' if floor_plan_img else ''}
</div>

<div class="meta">
    <p>Inference method: {result.inference_method} | Elapsed: {result.elapsed_seconds:.2f}s |
       Alternatives generated: {len(result.sampled_designs)}</p>
    <p>Generated by CAR (Constraint-satisfying Architectural design using pRobabilistic graphical models)</p>
</div>
</body>
</html>"""

        Path(output_path).write_text(html, encoding="utf-8")

    def _embed_image(self, path: str | Path) -> str:
        """Embed an image as base64 in an img tag."""
        p = Path(path)
        if not p.exists():
            return ""
        data = p.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        suffix = p.suffix.lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix, "image/png")
        return f'<img src="data:{mime};base64,{b64}" style="max-width:800px">'
