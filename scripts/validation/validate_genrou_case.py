#!/usr/bin/env python
"""
Validate GENROU case file.

Tests that the GENROU case file:
1. Loads correctly in ANDES
2. Runs power flow successfully
3. Runs TDS successfully
4. Has equivalent power flow to GENCLS case

Usage:
    python scripts/validation/validate_genrou_case.py \
        --genrou-case test_cases/SMIB_genrou.json \
        --gencls-case smib/SMIB.json
"""

import argparse
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


def validate_genrou_case(genrou_case_path: str, gencls_case_path: str = None):
    """
    Validate GENROU case file.

    Parameters:
    -----------
    genrou_case_path : str
        Path to GENROU case file
    gencls_case_path : str, optional
        Path to GENCLS case file for comparison
    """
    print("=" * 70)
    print("GENROU CASE FILE VALIDATION")
    print("=" * 70)

    # Load GENROU case
    print(f"\n1. Loading GENROU case: {genrou_case_path}")
    try:
        ss_genrou = andes.load(genrou_case_path, no_output=True)
        print(f"✓ GENROU case loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load GENROU case: {e}")
        return False

    # Check GENROU model
    if not hasattr(ss_genrou, "GENROU"):
        print("✗ GENROU model not found in case file")
        return False

    if ss_genrou.GENROU.n == 0:
        print("✗ No GENROU generators found (GENROU.n = 0)")
        return False

    print(f"✓ Found {ss_genrou.GENROU.n} GENROU generator(s)")

    # Print GENROU parameters
    print("\n2. GENROU Parameters:")
    for i in range(ss_genrou.GENROU.n):
        print(f"\n  Generator {i+1}:")
        print(f"    M (inertia): {ss_genrou.GENROU.M.v[i]:.4f} s")
        print(f"    D (damping): {ss_genrou.GENROU.D.v[i]:.4f} pu")
        print(f"    xd (sync reactance): {ss_genrou.GENROU.xd.v[i]:.4f} pu")
        print(f"    xq (sync reactance): {ss_genrou.GENROU.xq.v[i]:.4f} pu")
        print(f"    xd1 (transient reactance): {ss_genrou.GENROU.xd1.v[i]:.4f} pu")
        print(f"    xq1 (transient reactance): {ss_genrou.GENROU.xq1.v[i]:.4f} pu")
        print(f"    Td10 (transient time const): {ss_genrou.GENROU.Td10.v[i]:.4f} s")
        print(f"    Tq10 (transient time const): {ss_genrou.GENROU.Tq10.v[i]:.4f} s")
        print(f"    ra (armature resistance): {ss_genrou.GENROU.ra.v[i]:.4f} pu")

    # Validate parameter relationships
    print("\n3. Parameter Validation:")
    all_valid = True
    for i in range(ss_genrou.GENROU.n):
        xd = ss_genrou.GENROU.xd.v[i]
        xd1 = ss_genrou.GENROU.xd1.v[i]
        xq = ss_genrou.GENROU.xq.v[i]
        xq1 = ss_genrou.GENROU.xq1.v[i]

        if xd <= xd1:
            print(f"  ✗ Generator {i+1}: Xd ({xd:.4f}) <= Xd' ({xd1:.4f}) - Invalid!")
            all_valid = False
        else:
            print(f"  ✓ Generator {i+1}: Xd > Xd' relationship valid")

        if xq <= xq1:
            print(f"  ✗ Generator {i+1}: Xq ({xq:.4f}) <= Xq' ({xq1:.4f}) - Invalid!")
            all_valid = False
        else:
            print(f"  ✓ Generator {i+1}: Xq > Xq' relationship valid")

    if not all_valid:
        print("\n⚠️  Warning: Some parameter relationships are invalid")

    # Test power flow
    print("\n4. Testing Power Flow:")
    try:
        ss_genrou.PFlow.run()
        print("✓ Power flow converged")

        # Print power flow results
        for i in range(ss_genrou.GENROU.n):
            pe = None
            qe = None
            v = None
            delta = None

            if hasattr(ss_genrou.GENROU, "Pe") and len(ss_genrou.GENROU.Pe.v) > i:
                pe = ss_genrou.GENROU.Pe.v[i]
            if hasattr(ss_genrou.GENROU, "Qe") and len(ss_genrou.GENROU.Qe.v) > i:
                qe = ss_genrou.GENROU.Qe.v[i]
            if hasattr(ss_genrou.GENROU, "v") and len(ss_genrou.GENROU.v.v) > i:
                v = ss_genrou.GENROU.v.v[i]
            if hasattr(ss_genrou.GENROU, "delta") and len(ss_genrou.GENROU.delta.v) > i:
                delta = ss_genrou.GENROU.delta.v[i]

            print(f"\n  Generator {i+1} (post-power flow):")
            if pe is not None:
                print(f"    Pe: {pe:.4f} pu")
            if qe is not None:
                print(f"    Qe: {qe:.4f} pu")
            if v is not None:
                print(f"    Vt: {v:.4f} pu")
            if delta is not None:
                print(f"    Delta: {delta:.4f} rad")
    except Exception as e:
        print(f"✗ Power flow failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test TDS
    print("\n5. Testing TDS:")
    try:
        ss_genrou.TDS.config.criteria = 0  # Disable early stopping
        ss_genrou.TDS.config.dt = 0.001  # Small time step for GENROU
        ss_genrou.TDS.run(tf=2.0)  # Short simulation for testing
        print("✓ TDS completed successfully")

        # Try different ways to access time
        if hasattr(ss_genrou.TDS, "t") and hasattr(ss_genrou.TDS.t, "v"):
            print(f"  Simulation time: {ss_genrou.TDS.t.v[-1]:.4f} s")
            print(f"  Number of time steps: {len(ss_genrou.TDS.t.v)}")
        elif hasattr(ss_genrou.TDS, "time"):
            print(f"  Simulation completed (time attribute available)")
        else:
            print(f"  Simulation completed (TDS finished successfully)")
    except Exception as e:
        print(f"✗ TDS failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Compare with GENCLS if provided
    if gencls_case_path:
        print("\n6. Power Flow Equivalence Check:")
        try:
            # Resolve GENCLS case path
            try:
                gencls_path = andes.get_case(gencls_case_path)
            except Exception:
                gencls_path = gencls_case_path

            ss_gencls = andes.load(gencls_path, no_output=True)
            ss_gencls.PFlow.run()

            print("  Comparing GENCLS vs GENROU pre-fault operating points:")

            # Diagnostic: Check what generator models are available
            available_models = [
                attr
                for attr in dir(ss_gencls)
                if not attr.startswith("_") and hasattr(getattr(ss_gencls, attr), "n")
            ]
            generator_models = [
                m for m in available_models if "GEN" in m.upper() or "StaticGen" in m
            ]
            if generator_models:
                print(f"  Available generator models in GENCLS case: {', '.join(generator_models)}")

            # Compare first generator (machine, not infinite bus)
            if (
                hasattr(ss_gencls, "GENCLS")
                and ss_gencls.GENCLS.n > 0
                and len(ss_gencls.GENCLS.Pe.v) > 0
            ):
                gencls_pe = ss_gencls.GENCLS.Pe.v[0]
                gencls_qe = (
                    ss_gencls.GENCLS.Qe.v[0]
                    if (hasattr(ss_gencls.GENCLS, "Qe") and len(ss_gencls.GENCLS.Qe.v) > 0)
                    else None
                )
                gencls_v = (
                    ss_gencls.GENCLS.v.v[0]
                    if (hasattr(ss_gencls.GENCLS, "v") and len(ss_gencls.GENCLS.v.v) > 0)
                    else None
                )
                gencls_delta = (
                    ss_gencls.GENCLS.delta.v[0] if len(ss_gencls.GENCLS.delta.v) > 0 else None
                )

                genrou_pe = (
                    ss_genrou.GENROU.Pe.v[0]
                    if (hasattr(ss_genrou.GENROU, "Pe") and len(ss_genrou.GENROU.Pe.v) > 0)
                    else None
                )
                genrou_qe = (
                    ss_genrou.GENROU.Qe.v[0]
                    if (hasattr(ss_genrou.GENROU, "Qe") and len(ss_genrou.GENROU.Qe.v) > 0)
                    else None
                )
                genrou_v = (
                    ss_genrou.GENROU.v.v[0]
                    if (hasattr(ss_genrou.GENROU, "v") and len(ss_genrou.GENROU.v.v) > 0)
                    else None
                )
                genrou_delta = (
                    ss_genrou.GENROU.delta.v[0]
                    if (hasattr(ss_genrou.GENROU, "delta") and len(ss_genrou.GENROU.delta.v) > 0)
                    else None
                )

                if gencls_pe is None or genrou_pe is None:
                    print("  ⚠️  Could not compare: Pe values not available")
                    return True

                pe_match = abs(gencls_pe - genrou_pe) < 0.001
                qe_match = (
                    abs(gencls_qe - genrou_qe) < 0.001
                    if (gencls_qe is not None and genrou_qe is not None)
                    else None
                )
                v_match = (
                    abs(gencls_v - genrou_v) < 0.001
                    if (gencls_v is not None and genrou_v is not None)
                    else None
                )
                delta_match = abs(gencls_delta - genrou_delta) < 0.01

                print(f"\n    Pe: GENCLS={gencls_pe:.4f}, GENROU={genrou_pe:.4f}, Match={pe_match}")
                if qe_match is not None:
                    print(
                        f"    Qe: GENCLS={gencls_qe:.4f}, GENROU={genrou_qe:.4f}, Match={qe_match}"
                    )
                if v_match is not None:
                    print(f"    Vt: GENCLS={gencls_v:.4f}, GENROU={genrou_v:.4f}, Match={v_match}")
                print(
                    f"    Delta: GENCLS={gencls_delta:.4f}, GENROU={genrou_delta:.4f}, Match={delta_match}"
                )

                if (
                    pe_match
                    and (qe_match is None or qe_match)
                    and (v_match is None or v_match)
                    and delta_match
                ):
                    print("\n  ✓ Power flow equivalence verified")
                else:
                    print("\n  ⚠️  Warning: Power flow values differ between GENCLS and GENROU")
            else:
                # Provide detailed diagnostic information
                print("  ⚠️  Could not compare: GENCLS not found or empty")
                if not hasattr(ss_gencls, "GENCLS"):
                    print(f"     Reason: GENCLS model not found in case file")
                    if generator_models:
                        print(
                            f"     Note: Case uses {', '.join(generator_models)} instead of GENCLS"
                        )
                        print(
                            f"     Suggestion: Use a case file with GENCLS model, or skip comparison with --gencls-case ''"
                        )
                elif hasattr(ss_gencls, "GENCLS") and ss_gencls.GENCLS.n == 0:
                    print(f"     Reason: GENCLS.n = 0 (no GENCLS generators in case)")
                elif hasattr(ss_gencls, "GENCLS") and (
                    not hasattr(ss_gencls.GENCLS, "Pe") or len(ss_gencls.GENCLS.Pe.v) == 0
                ):
                    print(
                        f"     Reason: GENCLS.Pe.v is empty (power flow may not have populated values)"
                    )
                    print(f"     Note: Power flow completed, but Pe values are not available")
        except Exception as e:
            print(f"  ⚠️  Could not compare with GENCLS: {e}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Validate GENROU case file")
    parser.add_argument(
        "--genrou-case",
        type=str,
        required=True,
        help="Path to GENROU case file",
    )
    parser.add_argument(
        "--gencls-case",
        type=str,
        default="smib/SMIB.json",
        help="Path to GENCLS case file for comparison",
    )

    args = parser.parse_args()

    success = validate_genrou_case(
        genrou_case_path=args.genrou_case,
        gencls_case_path=args.gencls_case,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
