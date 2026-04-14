#!/usr/bin/env python3
"""
Diagnostic script to inspect SMIB case structure.

This script inspects the ANDES SMIB case file to understand its structure
and determine what modifications are needed to properly set Pe initial conditions.
"""

import json
import sys
from pathlib import Path

try:
    import andes

    ANDES_AVAILABLE = True
except ImportError:
    ANDES_AVAILABLE = False
    print("ERROR: ANDES not available. Please install: pip install andes")
    sys.exit(1)


def inspect_smib_case():
    """Inspect the SMIB case file structure."""

    print("=" * 80)
    print("SMIB CASE FILE STRUCTURE INSPECTION")
    print("=" * 80)

    # Load SMIB case
    try:
        case_path = andes.get_case("smib/SMIB.json")
        print(f"\n[OK] Found SMIB case: {case_path}")
    except Exception as e:
        print(f"\n[ERROR] Error getting SMIB case: {e}")
        return None

    # Load and inspect
    try:
        with open(case_path, "r") as f:
            case_data = json.load(f)
        print(f"[OK] Loaded case file successfully")
    except Exception as e:
        print(f"[ERROR] Error loading case file: {e}")
        return None

    # Document structure
    print("\n" + "-" * 80)
    print("CASE FILE COMPONENTS")
    print("-" * 80)
    components = list(case_data.keys())
    print(f"Found {len(components)} components: {', '.join(components)}")

    # Check for StaticGen
    print("\n" + "-" * 80)
    print("STATICGEN CHECK")
    print("-" * 80)
    has_staticgen = "StaticGen" in case_data
    if has_staticgen:
        print("[OK] StaticGen component EXISTS")
        print("\nStaticGen structure:")
        print(json.dumps(case_data["StaticGen"], indent=2))
    else:
        print("[MISSING] StaticGen component MISSING")
        print("\n[ACTION NEEDED] Must add StaticGen component")

    # Check GENCLS structure
    print("\n" + "-" * 80)
    print("GENCLS STRUCTURE")
    print("-" * 80)
    if "GENCLS" in case_data:
        gencls_data = case_data["GENCLS"]
        print(json.dumps(gencls_data, indent=2))

        # Check if GENCLS has 'gen' field (links to StaticGen)
        if isinstance(gencls_data, dict):
            has_gen_link = "gen" in gencls_data
            num_gens = len(gencls_data.get("idx", []))
            buses = gencls_data.get("bus", [])
        elif isinstance(gencls_data, list):
            has_gen_link = any("gen" in g for g in gencls_data)
            num_gens = len(gencls_data)
            buses = [g.get("bus") for g in gencls_data]

        print(f"\nNumber of generators: {num_gens}")
        print(f"Generator buses: {buses}")
        print(f"Has 'gen' link to StaticGen: {has_gen_link}")
    else:
        print("[ERROR] GENCLS component not found!")
        return None

    # Check for power-related fields in GENCLS
    print("\n" + "-" * 80)
    print("POWER SETPOINT FIELDS IN GENCLS")
    print("-" * 80)
    power_fields = ["P0", "Pg", "P", "Pm", "Pgen", "tm0", "tm"]
    found_fields = []
    if isinstance(gencls_data, dict):
        for field in power_fields:
            if field in gencls_data:
                found_fields.append(field)
                print(f"[OK] Found field '{field}': {gencls_data[field]}")
    elif isinstance(gencls_data, list):
        for field in power_fields:
            if any(field in g for g in gencls_data):
                found_fields.append(field)
                values = [g.get(field, "N/A") for g in gencls_data]
                print(f"[OK] Found field '{field}': {values}")

    if not found_fields:
        print("[INFO] No power setpoint fields found in GENCLS")
        print("   (This is expected - power setpoints should be in StaticGen)")

    # Check PQ load
    print("\n" + "-" * 80)
    print("LOAD (PQ) STRUCTURE")
    print("-" * 80)
    if "PQ" in case_data:
        pq_data = case_data["PQ"]
        print(json.dumps(pq_data, indent=2))
    else:
        print("[INFO] PQ component not found")

    # Summary and recommendations
    print("\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)

    if not has_staticgen:
        print("\n[ISSUE] StaticGen component is missing")
        print("\n   RECOMMENDATION:")
        print("   1. Add StaticGen component to case file")
        print("   2. Link GENCLS to StaticGen via 'gen' field")
        print("   3. Set StaticGen.p0 to desired generator power setpoint")
        print(f"\n   Proposed StaticGen structure for {num_gens} generator(s):")
        proposed_staticgen = {
            "idx": list(range(num_gens)),
            "bus": buses,
            "p0": [0.9] * num_gens,
            "q0": [0.0] * num_gens,
            "v0": [1.0] * num_gens,
            "name": [f"Gen_{i}" for i in range(num_gens)],
        }
        print(json.dumps(proposed_staticgen, indent=2))

        print(f"\n   Proposed GENCLS modification:")
        print(f"   Add 'gen' field: {list(range(num_gens))}")
    else:
        print("\n[OK] StaticGen component exists")
        print("[OK] Case file has proper structure")
        print("\n   Can modify StaticGen.p0 to set generator power setpoint")

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)

    return case_data


def main():
    """Main function."""
    case_data = inspect_smib_case()

    if case_data is not None:
        print("\n[OK] Inspection successful")
        return 0
    else:
        print("\n[ERROR] Inspection failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
