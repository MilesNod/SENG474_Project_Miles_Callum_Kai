"""Compatibility entry point for the saved-trajectory dataset generator.

Run from the repository root: python src/data/data_generation.py --help
The historical datasetForTestingDefault.npz is retained as an independent example.
"""

from pathlib import Path
import sys

# Kept so the old command still works. Add the ML scripts folder, then use the
# same main/options as fluid-model/src/generate_data.py instead of maintaining two generators.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fluid-model" / "src"))
from generate_data import main


if __name__ == "__main__":
    # Only generate when this file is executed, not when another module imports it.
    main()
