"""CLI entry point for CAR using Click."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from car.inference.pipeline import InferencePipeline, PipelineConfig
from car.models.constraints import SiteConstraints
from car.models.design import DesignIntent
from car.network.builder import NetworkBuilder
from car.visualization.compliance_report import ComplianceReportGenerator
from car.visualization.design_plot import DesignPlotter
from car.visualization.network_plot import NetworkPlotter


@click.group()
@click.version_option(version="0.1.0")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool) -> None:
    """CAR: Constraint-satisfying Architectural design using pRobabilistic graphical models."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(name)s | %(message)s")


@cli.command()
@click.option(
    "--config", "-c", type=click.Path(exists=True), required=True,
    help="Path to JSON configuration file with site constraints and design intent.",
)
@click.option(
    "--method", "-m", type=click.Choice(["map", "mcmc", "variational"]),
    default="map", help="Inference method to use.",
)
@click.option(
    "--samples", "-n", type=int, default=500,
    help="Number of design samples to generate (MCMC/VI only).",
)
@click.option(
    "--output-dir", "-o", type=click.Path(), default="./output",
    help="Directory for output files.",
)
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
def run(config: str, method: str, samples: int, output_dir: str, seed: int) -> None:
    """Run the full inference pipeline and generate outputs."""
    config_data = json.loads(Path(config).read_text(encoding="utf-8"))
    site_constraints = SiteConstraints(**config_data["site_constraints"])
    design_intent = DesignIntent(**config_data["design_intent"])

    pipeline_config = PipelineConfig(
        inference_method=method,
        num_samples=samples,
        random_seed=seed,
    )
    pipeline = InferencePipeline(pipeline_config)

    click.echo("Running inference pipeline...")
    result = pipeline.run(site_constraints, design_intent)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate outputs
    click.echo("Generating visualizations...")
    builder = NetworkBuilder()
    model = builder.build()
    topology = builder.build_topology()

    network_graph_path = output_path / "network_graph.png"
    floor_plan_path = output_path / "floor_plan.png"
    section_path = output_path / "building_section.png"
    report_path = output_path / "compliance_report.html"

    NetworkPlotter().plot(model, topology, network_graph_path)
    DesignPlotter().plot_floor_plan(result.map_design.design, floor_plan_path)
    DesignPlotter().plot_building_section(result.map_design.design, section_path)

    if result.sampled_designs:
        comparison_path = output_path / "design_comparison.png"
        DesignPlotter().plot_design_comparison(result.sampled_designs, comparison_path)

    ComplianceReportGenerator().generate(
        result, site_constraints, report_path,
        network_graph_path=network_graph_path,
        floor_plan_path=floor_plan_path,
    )

    # Save result as JSON
    result_json_path = output_path / "result.json"
    result_json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    # Print summary
    click.echo("=" * 60)
    click.echo("RESULTS")
    click.echo("=" * 60)
    md = result.map_design
    click.echo(f"  Confidence Score:  {md.compliance.confidence_score:.1%}")
    click.echo(f"  Compliant:         {'YES' if md.compliance.is_compliant else 'NO'}")
    click.echo(f"  Aesthetic Score:   {md.aesthetic_score:.2f}")
    click.echo(f"  View Score:        {md.view_score:.2f}")
    click.echo(f"  Overall Score:     {md.overall_score:.2f}")
    click.echo(f"  Structure:         {md.design.structural_system.value}")
    click.echo(f"  Floors:            {md.design.num_floors} ({md.design.building_height_m:.1f}m)")
    click.echo(f"  Windows:           {md.design.window_size.value}")
    click.echo(f"  Roof:              {md.design.roof_type.value}")
    click.echo(f"  Alternatives:      {len(result.sampled_designs)}")
    click.echo(f"  Elapsed:           {result.elapsed_seconds:.2f}s")
    click.echo(f"  Output:            {output_path.resolve()}")

    if md.compliance.violations:
        click.echo("\nViolations:")
        for v in md.compliance.violations:
            click.echo(f"  [{v.severity.upper()}] {v.constraint_name}: {v.actual_value} (required: {v.required_value})")


@cli.command()
@click.option(
    "--example", "-e",
    type=click.Choice(["residential", "commercial", "mixed_use"]),
    required=True, help="Example scenario to run.",
)
@click.option(
    "--method", "-m", type=click.Choice(["map", "mcmc", "variational"]),
    default="map", help="Inference method.",
)
@click.option("--output-dir", "-o", type=click.Path(), default="./output")
def example(example: str, method: str, output_dir: str) -> None:
    """Run a built-in example scenario."""
    from car.examples import residential_low_rise, commercial_high_rise, mixed_use_urban

    examples_map = {
        "residential": residential_low_rise,
        "commercial": commercial_high_rise,
        "mixed_use": mixed_use_urban,
    }
    module = examples_map[example]
    module.run(output_dir=output_dir, method=method)


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to.")
@click.option("--port", "-p", type=int, default=5000, help="Port to bind to.")
@click.option("--no-debug", is_flag=True, help="Disable Flask debug mode.")
def web(host: str, port: int, no_debug: bool) -> None:
    """Start the web interface for manual input."""
    from car.web.app import main as start_web

    click.echo(f"Starting CAR web interface at http://{host}:{port}")
    start_web(host=host, port=port, debug=not no_debug)


@cli.command(name="show-network")
def show_network() -> None:
    """Display the Bayesian network structure (nodes and edges)."""
    builder = NetworkBuilder()
    topology = builder.build_topology()

    click.echo(f"\nNodes ({len(topology.nodes)}):")
    click.echo("-" * 80)
    for node in topology.nodes:
        states = ", ".join(node.state_names)
        click.echo(f"  [{node.variable_type:8s}] {node.name:25s} ({node.cardinality} states: {states})")

    click.echo(f"\nEdges ({len(topology.edges)}):")
    click.echo("-" * 80)
    for edge in topology.edges:
        click.echo(f"  {edge.parent:25s} --> {edge.child:25s}  (w={edge.weight:.2f})  {edge.rationale}")

    # Validate
    try:
        model = builder.build()
        click.echo(f"\nModel validation: PASSED")
    except Exception as e:
        click.echo(f"\nModel validation: FAILED ({e})")
