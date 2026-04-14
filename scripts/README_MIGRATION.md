# Data Migration to Common Repository

## Quick Start

To migrate existing data files to the new common repository structure:

```bash
# 1. Dry run (see what would be migrated)
python scripts/migrate_to_common_repository.py --dry-run

# 2. Actually migrate
python scripts/migrate_to_common_repository.py

# 3. Verify migration
ls data/common/*.csv
cat data/common/registry.json
```

## What Gets Migrated

The script automatically finds and migrates:
- Files in `data/generated/` (all subdirectories)
- Files in `outputs/experiments/*/data/`
- Files matching: `*parameter_sweep*.csv`, `*trajectory*.csv`, `*parameter_estimation*.csv`

## After Migration

1. **Data is automatically reused**: New experiments will find migrated data via fingerprint matching
2. **Old files remain**: Original files are copied (not moved), so old scripts still work
3. **Update references**: Update any hardcoded paths in your scripts/docs

## Benefits

- ✅ No duplicate data generation
- ✅ Automatic data reuse
- ✅ Full reproducibility tracking
- ✅ Better organization

See `docs/guides/MIGRATION_TO_COMMON_REPOSITORY.md` for detailed information.
