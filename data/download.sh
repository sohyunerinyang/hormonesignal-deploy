#!/usr/bin/env bash
# whh-bench: mcPHASES is DUA-gated (PhysioNet Restricted Health Data License).
# This script cannot auto-download it for you — you must have your own
# credentialed PhysioNet access first. See data/README.md.
#
# Once you have a PhysioNet account with mcPHASES access approved, uncomment
# and run (requires the PhysioNet CLI / wget with your credentials configured):
#
#   wget -r -N -c -np --user <your_physionet_username> --ask-password \
#     https://physionet.org/files/mcphases/1.0.0/ -P data/raw/
#
# Then flatten data/raw/ so the CSVs sit directly in that folder (no nested
# physionet.org/... path), and run:
#   python -m whh_bench.export_results --data-dir data/raw --out results/latest_run.json

echo "mcPHASES requires credentialed PhysioNet access — see data/README.md"
echo "This script will not run automatically. Edit it once you have access."
exit 1
