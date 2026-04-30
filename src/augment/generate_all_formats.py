#!/usr/bin/env python3
"""Generate training data in all A-series sizes and DPI values."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from augment.fullpage_generator import generate_full_page_synthetic
from augment.synthetic_generator import generate_test_dataset

paper_sizes = ["A0", "A1", "A2", "A3", "A4"]
dpi_values = [200, 300]
num_samples = 20

base_dir = Path("ground_truth/yolo_training")
base_dir.mkdir(parents=True, exist_ok=True)

for paper_size in paper_sizes:
    for dpi in dpi_values:
        output_dir = base_dir / f"{paper_size}_{dpi}dpi"
        
        print(f"\n{'='*50}")
        print(f"Generating: {paper_size} @ {dpi} DPI")
        print(f"{'='*50}")
        
        metadata = generate_full_page_synthetic(
            output_dir=output_dir,
            num_samples=num_samples,
            dpi_values=[dpi],
            paper_size=paper_size,
            orientation="portrait",
        )
        
        print(f"Generated {len(metadata)} images")

print(f"\n{'='*50}")
print(f"Done! All data in: {base_dir}")
print(f"{'='*50}")
