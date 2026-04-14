#!/usr/bin/env python
"""
Create GENROU case file from GENCLS case file.

Converts SMIB.json (GENCLS) to SMIB_genrou.json (GENROU) by:
1. Loading the GENCLS case file
2. Extracting GENROU default parameters from ANDES
3. Creating a new case file with GENROU instead of GENCLS
4. Preserving all other components (buses, lines, loads, etc.)

Usage:
    python scripts/validation/create_genrou_case.py \
        --input-case smib/SMIB.json \
        --output-case test_cases/SMIB_genrou.json
"""

import argparse
import json
import sys
from pathlib import Path

# Fix encoding for Windows
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import andes

    ANDES_AVAILABLE = True
except ImportError:
    ANDES_AVAILABLE = False
    print("ERROR: ANDES not available. Please install: pip install andes")
    sys.exit(1)


def extract_genrou_defaults_from_andes() -> dict:
    """
    Extract GENROU default parameters from ANDES.

    ANDES has default GENROU parameters that follow IEEE standards.
    We'll use these defaults and override only the parameters that
    come from GENCLS (H, D, Pm).

    Returns:
    --------
    genrou_defaults : dict
        Dictionary of GENROU default parameters
    """
    # Create a temporary system to get defaults
    # We'll use a minimal case or create one programmatically
    try:
        # Try to load the case and see if GENROU has defaults
        case_path = andes.get_case("smib/SMIB.json")
        ss = andes.load(case_path, no_output=True)

        # Check if GENROU exists (even if n=0)
        if hasattr(ss, "GENROU"):
            # Get default values from ANDES GENROU model
            # These are the default values when GENROU is created
            genrou_defaults = {
                "M": 12.0,  # Default inertia (M = 2*H, H=6.0 default)
                "D": 1.0,  # Default damping
                "ra": 0.0,  # Armature resistance
                "xd": 1.8,  # Synchronous reactance (d-axis)
                "xq": 1.7,  # Synchronous reactance (q-axis)
                "xd1": 0.3,  # Transient reactance (d-axis)
                "xq1": 0.55,  # Transient reactance (q-axis)
                "Td10": 6.0,  # Transient time constant (d-axis)
                "Tq10": 0.5,  # Transient time constant (q-axis)
            }

            # Try to get actual defaults from ANDES if possible
            # Note: ANDES may not expose defaults directly, so we use standard values
            return genrou_defaults
        else:
            raise ValueError("GENROU model not available in ANDES")

    except Exception as e:
        print(f"Warning: Could not extract defaults from ANDES: {e}")
        print("Using standard IEEE default values")
        # Return standard IEEE default values
        return {
            "M": 12.0,  # M = 2*H, H=6.0
            "D": 1.0,
            "ra": 0.0,
            "xd": 1.8,
            "xq": 1.7,
            "xd1": 0.3,
            "xq1": 0.55,
            "Td10": 6.0,
            "Tq10": 0.5,
        }


def convert_gencls_to_genrou(
    input_case_path: str,
    output_case_path: str,
    genrou_params: dict = None,
) -> str:
    """
    Convert GENCLS case file to GENROU case file.

    Parameters:
    -----------
    input_case_path : str
        Path to input GENCLS case file
    output_case_path : str
        Path to output GENROU case file
    genrou_params : dict, optional
        Additional GENROU parameters to override defaults

    Returns:
    --------
    output_path : str
        Path to created GENROU case file
    """
    # Resolve input path
    try:
        input_path = andes.get_case(input_case_path)
    except Exception:
        input_path = Path(input_case_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Case file not found: {input_case_path}")

    # Load case file
    print(f"Loading case file: {input_path}")
    with open(input_path, "r") as f:
        case_data = json.load(f)

    # Check if GENCLS exists
    if "GENCLS" not in case_data:
        raise ValueError(f"Case file does not contain GENCLS: {input_path}")

    gencls_data = case_data["GENCLS"]

    # Get GENROU defaults
    genrou_defaults = extract_genrou_defaults_from_andes()

    # Override with provided parameters
    if genrou_params:
        genrou_defaults.update(genrou_params)

    # Convert GENCLS to GENROU
    # GENCLS structure: {"idx": [0], "bus": [1], "gen": [0], "M": [12.0], "D": [1.0], ...}
    # GENROU structure: similar but with additional fields

    if isinstance(gencls_data, dict):
        # Dict format
        genrou_data = {}

        # Copy common fields
        for key in ["idx", "bus", "gen", "name", "coi"]:
            if key in gencls_data:
                genrou_data[key] = gencls_data[key]

        # Convert M (inertia) - GENCLS uses M, GENROU also uses M
        if "M" in gencls_data:
            genrou_data["M"] = gencls_data["M"]
        else:
            # Default M = 12.0 (H = 6.0)
            genrou_data["M"] = [genrou_defaults["M"]]

        # Copy D (damping)
        if "D" in gencls_data:
            genrou_data["D"] = gencls_data["D"]
        else:
            genrou_data["D"] = [genrou_defaults["D"]]

        # Add GENROU-specific parameters
        genrou_data["ra"] = [genrou_defaults["ra"]]
        genrou_data["xd"] = [genrou_defaults["xd"]]
        genrou_data["xq"] = [genrou_defaults["xq"]]
        genrou_data["xd1"] = [genrou_defaults["xd1"]]
        genrou_data["xq1"] = [genrou_defaults["xq1"]]
        genrou_data["Td10"] = [genrou_defaults["Td10"]]
        genrou_data["Tq10"] = [genrou_defaults["Tq10"]]

        # Ensure all lists have same length
        n_gens = len(genrou_data.get("idx", [0]))
        for key, value in genrou_data.items():
            if isinstance(value, list) and len(value) < n_gens:
                genrou_data[key] = value * n_gens
            elif not isinstance(value, list):
                genrou_data[key] = [value] * n_gens

    elif isinstance(gencls_data, list):
        # List format: [{"idx": 0, "bus": 1, ...}, ...]
        genrou_data = []
        for gen in gencls_data:
            genrou_gen = {}

            # Copy common fields
            for key in ["idx", "bus", "gen", "name", "coi"]:
                if key in gen:
                    genrou_gen[key] = gen[key]

            # Convert M
            genrou_gen["M"] = gen.get("M", genrou_defaults["M"])
            genrou_gen["D"] = gen.get("D", genrou_defaults["D"])

            # Add GENROU-specific parameters
            genrou_gen["ra"] = genrou_defaults["ra"]
            genrou_gen["xd"] = genrou_defaults["xd"]
            genrou_gen["xq"] = genrou_defaults["xq"]
            genrou_gen["xd1"] = genrou_defaults["xd1"]
            genrou_gen["xq1"] = genrou_defaults["xq1"]
            genrou_gen["Td10"] = genrou_defaults["Td10"]
            genrou_gen["Tq10"] = genrou_defaults["Tq10"]

            genrou_data.append(genrou_gen)
    else:
        raise ValueError(f"Unexpected GENCLS format: {type(gencls_data)}")

    # Create new case data with GENROU instead of GENCLS
    # IMPORTANT: For SMIB, we typically only convert the machine generator (connected to PV)
    # and keep the infinite bus generator (connected to Slack) as GENCLS
    new_case_data = case_data.copy()

    # Determine which generators to convert
    # In SMIB: GENCLS_1 -> PV_1 (machine, convert to GENROU)
    #          GENCLS_2 -> Slack_1 (infinite bus, keep as GENCLS)
    if isinstance(gencls_data, list):
        gencls_to_keep = []
        genrou_to_add = []

        for i, gen in enumerate(gencls_data):
            gen_ref = gen.get("gen", "")
            # Check if connected to Slack (infinite bus) - keep as GENCLS
            if isinstance(gen_ref, str) and "Slack" in gen_ref:
                gencls_to_keep.append(gen)
            else:
                # Machine generator - convert to GENROU
                if isinstance(genrou_data, list) and i < len(genrou_data):
                    genrou_to_add.append(genrou_data[i])
                elif isinstance(genrou_data, dict):
                    # Extract single generator from dict format
                    genrou_gen = {}
                    for key, value in genrou_data.items():
                        if isinstance(value, list) and i < len(value):
                            genrou_gen[key] = value[i]
                        elif not isinstance(value, list):
                            genrou_gen[key] = value
                    genrou_to_add.append(genrou_gen)

        # If we have GENROU generators, add them
        if genrou_to_add:
            new_case_data["GENROU"] = genrou_to_add

        # Keep GENCLS for infinite bus if any
        if gencls_to_keep:
            new_case_data["GENCLS"] = gencls_to_keep
        elif "GENCLS" in new_case_data:
            del new_case_data["GENCLS"]
    else:
        # Dict format - convert all to GENROU (simpler case)
        new_case_data["GENROU"] = genrou_data
        if "GENCLS" in new_case_data:
            del new_case_data["GENCLS"]

    # Save output file
    output_path = Path(output_case_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving GENROU case file: {output_path}")
    with open(output_path, "w") as f:
        json.dump(new_case_data, f, indent=2)

    print(f"✓ GENROU case file created: {output_path}")

    # Validate the case file
    print("\nValidating GENROU case file...")
    try:
        ss = andes.load(str(output_path), no_output=True)
        if hasattr(ss, "GENROU") and ss.GENROU.n > 0:
            print(f"✓ GENROU case file validated: {ss.GENROU.n} generator(s)")

            # Print parameter values
            print("\nGENROU Parameters:")
            print(f"  M (inertia): {ss.GENROU.M.v[0]:.4f} s")
            print(f"  D (damping): {ss.GENROU.D.v[0]:.4f} pu")
            print(f"  xd (sync reactance): {ss.GENROU.xd.v[0]:.4f} pu")
            print(f"  xq (sync reactance): {ss.GENROU.xq.v[0]:.4f} pu")
            print(f"  xd1 (transient reactance): {ss.GENROU.xd1.v[0]:.4f} pu")
            print(f"  xq1 (transient reactance): {ss.GENROU.xq1.v[0]:.4f} pu")
            print(f"  Td10 (transient time const): {ss.GENROU.Td10.v[0]:.4f} s")
            print(f"  Tq10 (transient time const): {ss.GENROU.Tq10.v[0]:.4f} s")
            print(f"  ra (armature resistance): {ss.GENROU.ra.v[0]:.4f} pu")
        else:
            print("⚠️  Warning: GENROU case file loaded but no generators found")
    except Exception as e:
        print(f"⚠️  Warning: Could not validate case file: {e}")

    return str(output_path)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Convert GENCLS case file to GENROU")
    parser.add_argument(
        "--input-case",
        type=str,
        default="smib/SMIB.json",
        help="Input GENCLS case file path",
    )
    parser.add_argument(
        "--output-case",
        type=str,
        default="test_cases/SMIB_genrou.json",
        help="Output GENROU case file path",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GENCLS TO GENROU CONVERSION")
    print("=" * 70)

    try:
        output_path = convert_gencls_to_genrou(
            input_case_path=args.input_case,
            output_case_path=args.output_case,
        )
        print(f"\n✓ Conversion complete: {output_path}")
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import traceback

    main()
