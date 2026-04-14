#!/usr/bin/env python3
"""Quick-start script: Feed input and get results from the CAR pipeline.

This demonstrates the full process:
  1. Define the architect's design intent (PRIMARY driver)
  2. Define site constraints (regulatory, environmental, geophysical, technical)
  3. Run the inference pipeline
  4. Read the results
  5. Generate visualizations
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Define the architect's DESIGN INTENT (drives generation)
# ─────────────────────────────────────────────────────────────────────
from car.models.design import AestheticFeel, DesignIntent, ViewPriority

design_intent = DesignIntent(
    aesthetic_feel=AestheticFeel.MINIMALIST,   # Options: MINIMALIST, INDUSTRIAL, ORGANIC, CLASSICAL
    view_priority=ViewPriority.HIGH,           # Options: LOW, MEDIUM, HIGH
    sustainability_priority=ViewPriority.MEDIUM,
    budget_level="medium",                     # Options: "low", "medium", "high"
)

# ─────────────────────────────────────────────────────────────────────
# STEP 2: Define SITE CONSTRAINTS (filters/shapes the design)
# ─────────────────────────────────────────────────────────────────────
from car.models.constraints import (
    EnvironmentalConstraints,
    GeophysicalConstraints,
    MaterialProperties,
    RegulatoryConstraints,
    SeismicZone,
    SiteConstraints,
    TechnicalConstraints,
)

site_constraints = SiteConstraints(
    site_area_sqm=500.0,

    # --- Regulatory constraints ---
    regulatory=RegulatoryConstraints(
        far_limit=0.6,               # Max Floor Area Ratio
        height_limit_m=10.0,         # Max building height in meters
        is_airport_zone=False,       # Airport approach zone?
        setback_front_m=6.0,         # Required front setback (m)
        setback_side_m=3.0,          # Required side setback (m)
        setback_rear_m=5.0,          # Required rear setback (m)
        min_parking_spaces=2,
        fire_escape_required=False,
    ),

    # --- Environmental constraints ---
    environmental=EnvironmentalConstraints(
        latitude=34.05,
        longitude=-118.24,
        solar_azimuth_peak_deg=180.0,      # Peak sun direction (degrees from north)
        solar_elevation_peak_deg=73.0,     # Peak sun elevation angle
        prevailing_wind_direction_deg=270.0,
        prevailing_wind_speed_kmh=12.0,
        annual_rainfall_mm=380.0,
    ),

    # --- Geophysical constraints ---
    geophysical=GeophysicalConstraints(
        seismic_zone=SeismicZone.ZONE_2,   # 0=none, 5=extreme
        soil_bearing_capacity_kpa=200.0,
        water_table_depth_m=8.0,
    ),

    # --- Technical constraints ---
    technical=TechnicalConstraints(
        available_materials=[
            MaterialProperties(
                name="timber",
                youngs_modulus_gpa=12.0,
                thermal_mass_kj_per_m3k=750.0,
                density_kg_per_m3=500.0,
                cost_per_m3_usd=600.0,
            ),
            MaterialProperties(
                name="steel",
                youngs_modulus_gpa=200.0,
                thermal_mass_kj_per_m3k=3500.0,
                density_kg_per_m3=7850.0,
                cost_per_m3_usd=5000.0,
            ),
        ],
        wall_thickness_min_mm=100.0,
        wall_thickness_max_mm=250.0,
        floor_to_floor_height_min_m=2.7,
        floor_to_floor_height_max_m=3.2,
    ),
)

# ─────────────────────────────────────────────────────────────────────
# STEP 3: Configure and run the inference pipeline
# ─────────────────────────────────────────────────────────────────────
from car.inference.pipeline import InferencePipeline, PipelineConfig

config = PipelineConfig(
    inference_method="map",      # Options: "map" (fastest), "mcmc", "variational"
    num_samples=500,             # Only used for mcmc/variational
    random_seed=42,
)

pipeline = InferencePipeline(config)
result = pipeline.run(site_constraints, design_intent)

# ─────────────────────────────────────────────────────────────────────
# STEP 4: Read the results
# ─────────────────────────────────────────────────────────────────────
map_design = result.map_design

print("=" * 60)
print("  CAR INFERENCE RESULTS")
print("=" * 60)

# Compliance
print(f"\n  COMPLIANCE")
print(f"    Compliant:        {'YES' if map_design.compliance.is_compliant else 'NO'}")
print(f"    Confidence Score: {map_design.compliance.confidence_score:.1%}")
print(f"    Constraints:      {map_design.compliance.passed_constraints_count}"
      f"/{map_design.compliance.checked_constraints_count} passed")

# Scores
print(f"\n  SCORES")
print(f"    Aesthetic:  {map_design.aesthetic_score:.2f}")
print(f"    View:       {map_design.view_score:.2f}")
print(f"    Overall:    {map_design.overall_score:.2f}")

# Design parameters
d = map_design.design
print(f"\n  DESIGN PARAMETERS")
print(f"    Structural System: {d.structural_system.value}")
print(f"    Floors:            {d.num_floors} ({d.building_height_m:.1f}m)")
print(f"    Floor Area:        {d.floor_area_sqm:.0f} sqm per floor")
print(f"    Wall Type:         {d.wall_type.value} ({d.wall_thickness_mm:.0f}mm)")
print(f"    Window Size:       {d.window_size.value}")
print(f"    Roof Type:         {d.roof_type.value}")
print(f"    Footprint:         {d.footprint_width_m:.1f}m x {d.footprint_depth_m:.1f}m")
print(f"    Material:          {d.primary_material}")
print(f"    Window Orient.:    {d.window_orientation_deg:.0f} deg")

# Violations
if map_design.compliance.violations:
    print(f"\n  VIOLATIONS")
    for v in map_design.compliance.violations:
        print(f"    [{v.severity.upper():4s}] {v.constraint_name}: "
              f"{v.actual_value} (required: {v.required_value})")

# Sampled alternatives (if mcmc/variational was used)
if result.sampled_designs:
    print(f"\n  ALTERNATIVES: {len(result.sampled_designs)} designs generated")
    for sd in result.sampled_designs[:5]:
        status = "PASS" if sd.compliance.is_compliant else "FAIL"
        print(f"    #{sd.iteration_id:3d} | {sd.design.structural_system.value:25s} | "
              f"{sd.design.num_floors}F | {sd.design.window_size.value:10s} | "
              f"Score: {sd.overall_score:.3f} | {status}")

print(f"\n  Elapsed: {result.elapsed_seconds:.2f}s")
print(f"  Method:  {result.inference_method}")

# ─────────────────────────────────────────────────────────────────────
# STEP 5: Generate visualizations
# ─────────────────────────────────────────────────────────────────────
output_dir = Path("./output")
output_dir.mkdir(parents=True, exist_ok=True)

from car.network.builder import NetworkBuilder
from car.visualization.network_plot import NetworkPlotter
from car.visualization.design_plot import DesignPlotter
from car.visualization.compliance_report import ComplianceReportGenerator

builder = NetworkBuilder()
model = builder.build()
topology = builder.build_topology()

# Generate all outputs
network_graph = output_dir / "network_graph.png"
floor_plan = output_dir / "floor_plan.png"
section = output_dir / "building_section.png"
report = output_dir / "compliance_report.html"
result_json = output_dir / "result.json"

NetworkPlotter().plot(model, topology, network_graph)
DesignPlotter().plot_floor_plan(d, floor_plan)
DesignPlotter().plot_building_section(d, section)
ComplianceReportGenerator().generate(
    result, site_constraints, report,
    network_graph_path=network_graph,
    floor_plan_path=floor_plan,
)
result_json.write_text(result.model_dump_json(indent=2), encoding="utf-8")

print(f"\n  OUTPUT FILES")
print(f"    {network_graph}    - Bayesian network DAG")
print(f"    {floor_plan}       - Floor plan sketch")
print(f"    {section}  - Building cross-section")
print(f"    {report} - Full HTML compliance report")
print(f"    {result_json}        - Machine-readable JSON")
print("=" * 60)
