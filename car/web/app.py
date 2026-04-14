"""Flask web interface for the CAR pipeline.

Uses lazy imports for heavy dependencies (pgmpy, numpy, scipy) so the Flask
server can start quickly and serve the input form while those packages load
only when the user submits the form.
"""

from __future__ import annotations

import base64
import io
import logging
import tempfile
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file, abort

logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)
app.config["LAST_RESULT_JSON"] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Render the input form."""
    return render_template("index.html")


@app.route("/preset/<name>")
def preset(name: str):
    """Return JSON for a built-in example preset."""
    from car.examples import residential_low_rise, commercial_high_rise, mixed_use_urban

    examples = {
        "residential": residential_low_rise,
        "commercial": commercial_high_rise,
        "mixed_use": mixed_use_urban,
    }
    if name not in examples:
        abort(404)

    module = examples[name]
    constraints = module.get_constraints()
    intent = module.get_intent()

    data = {
        "site_constraints": constraints.model_dump(),
        "design_intent": intent.model_dump(),
    }
    # Ensure seismic_zone is an int for the form
    data["site_constraints"]["geophysical"]["seismic_zone"] = (
        constraints.geophysical.seismic_zone.value
    )
    return jsonify(data)


@app.route("/run", methods=["POST"])
def run_pipeline():
    """Parse form data, run the inference pipeline, and render results."""
    # --- Lazy imports (heavy dependencies) ---
    from car.inference.pipeline import InferencePipeline, PipelineConfig
    from car.models.constraints import (
        EnvironmentalConstraints,
        GeophysicalConstraints,
        MaterialProperties,
        RegulatoryConstraints,
        SeismicZone,
        SiteConstraints,
        TechnicalConstraints,
    )
    from car.models.design import AestheticFeel, DesignIntent, ViewPriority
    from car.network.builder import NetworkBuilder
    from car.visualization.design_plot import DesignPlotter
    from car.visualization.network_plot import NetworkPlotter

    try:
        # --- Parse Design Intent ---
        design_intent = DesignIntent(
            aesthetic_feel=AestheticFeel(request.form["aesthetic_feel"]),
            view_priority=ViewPriority(request.form["view_priority"]),
            sustainability_priority=ViewPriority(request.form.get("sustainability_priority", "medium")),
            budget_level=request.form.get("budget_level", "medium"),
        )

        # --- Parse Materials (dynamic list, indices may have gaps) ---
        materials: list[MaterialProperties] = []
        for i in range(20):
            key = f"material_name_{i}"
            if key in request.form and request.form[key].strip():
                materials.append(
                    MaterialProperties(
                        name=request.form[f"material_name_{i}"].strip(),
                        youngs_modulus_gpa=float(request.form[f"material_youngs_{i}"]),
                        thermal_mass_kj_per_m3k=float(request.form[f"material_thermal_{i}"]),
                        density_kg_per_m3=float(request.form[f"material_density_{i}"]),
                        cost_per_m3_usd=float(request.form[f"material_cost_{i}"]),
                    )
                )

        if not materials:
            raise ValueError("At least one material is required. Click '+ Add Material' to add one.")

        # --- Parse Site Constraints ---
        site_constraints = SiteConstraints(
            site_area_sqm=float(request.form["site_area_sqm"]),
            regulatory=RegulatoryConstraints(
                far_limit=float(request.form["far_limit"]),
                height_limit_m=float(request.form["height_limit_m"]),
                is_airport_zone="is_airport_zone" in request.form,
                setback_front_m=float(request.form["setback_front_m"]),
                setback_side_m=float(request.form["setback_side_m"]),
                setback_rear_m=float(request.form["setback_rear_m"]),
                min_parking_spaces=int(request.form.get("min_parking_spaces", 0)),
                fire_escape_required="fire_escape_required" in request.form,
            ),
            environmental=EnvironmentalConstraints(
                latitude=float(request.form["latitude"]),
                longitude=float(request.form["longitude"]),
                solar_azimuth_peak_deg=float(request.form["solar_azimuth_peak_deg"]),
                solar_elevation_peak_deg=float(request.form["solar_elevation_peak_deg"]),
                prevailing_wind_direction_deg=float(request.form["prevailing_wind_direction_deg"]),
                prevailing_wind_speed_kmh=float(request.form.get("prevailing_wind_speed_kmh", 15)),
                annual_rainfall_mm=float(request.form.get("annual_rainfall_mm", 800)),
            ),
            geophysical=GeophysicalConstraints(
                seismic_zone=SeismicZone(int(request.form["seismic_zone"])),
                soil_bearing_capacity_kpa=float(request.form.get("soil_bearing_capacity_kpa", 150)),
                water_table_depth_m=float(request.form.get("water_table_depth_m", 5)),
            ),
            technical=TechnicalConstraints(
                available_materials=materials,
                wall_thickness_min_mm=float(request.form.get("wall_thickness_min_mm", 100)),
                wall_thickness_max_mm=float(request.form.get("wall_thickness_max_mm", 500)),
                floor_to_floor_height_min_m=float(request.form.get("floor_to_floor_height_min_m", 2.7)),
                floor_to_floor_height_max_m=float(request.form.get("floor_to_floor_height_max_m", 4.5)),
            ),
        )

        # --- Pipeline Config ---
        method = request.form.get("inference_method", "map")
        pipeline_config = PipelineConfig(
            inference_method=method,
            num_samples=int(request.form.get("num_samples", 500)),
            random_seed=int(request.form.get("random_seed", 42)),
        )

        # --- Run Pipeline ---
        pipeline = InferencePipeline(pipeline_config)
        result = pipeline.run(site_constraints, design_intent)

        # --- Generate Visualizations as base64 ---
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            builder = NetworkBuilder()
            model = builder.build()
            topology = builder.build_topology()

            ng_path = tmp / "network_graph.png"
            fp_path = tmp / "floor_plan.png"
            bs_path = tmp / "building_section.png"

            NetworkPlotter().plot(model, topology, ng_path)
            DesignPlotter().plot_floor_plan(result.map_design.design, fp_path)
            DesignPlotter().plot_building_section(result.map_design.design, bs_path)

            network_b64 = _encode_image(ng_path)
            floorplan_b64 = _encode_image(fp_path)
            section_b64 = _encode_image(bs_path)

            comparison_b64 = None
            if result.sampled_designs:
                comp_path = tmp / "design_comparison.png"
                DesignPlotter().plot_design_comparison(result.sampled_designs, comp_path)
                comparison_b64 = _encode_image(comp_path)

        # Store result JSON for download
        app.config["LAST_RESULT_JSON"] = result.model_dump_json(indent=2)

        return render_template(
            "results.html",
            result=result,
            map_design=result.map_design,
            design=result.map_design.design,
            compliance=result.map_design.compliance,
            network_img=network_b64,
            floorplan_img=floorplan_b64,
            section_img=section_b64,
            comparison_img=comparison_b64,
            site_constraints=site_constraints,
            design_intent=design_intent,
        )

    except Exception as e:
        logger.exception("Pipeline failed")
        return render_template("index.html", error=str(e)), 400


@app.route("/download/<filename>")
def download(filename: str):
    """Serve generated output files."""
    if filename == "result.json" and app.config.get("LAST_RESULT_JSON"):
        buf = io.BytesIO(app.config["LAST_RESULT_JSON"].encode("utf-8"))
        buf.seek(0)
        return send_file(
            buf,
            mimetype="application/json",
            as_attachment=True,
            download_name="result.json",
        )
    abort(404)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_image(path: Path) -> str:
    """Read a PNG file and return a data URI string for embedding in HTML."""
    if not path.exists():
        return ""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(host: str = "127.0.0.1", port: int = 5000, debug: bool = True) -> None:
    """Start the Flask development server."""
    import matplotlib
    matplotlib.use("Agg")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    logger.info(f"Starting CAR web interface at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
