"""
Phase 7: Dataset Balancing

Balances Common Voice metadata across age groups.
Finds minimum class count and randomly samples equal records from each age group.
Ensures reproducible sampling with fixed random seed.
"""

import json
import logging
import csv
import random
from pathlib import Path
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Random seed for reproducibility
RANDOM_SEED = 42


def load_metadata(metadata_path: str) -> list:
    """
    Load metadata CSV file.
    
    Args:
        metadata_path: Path to metadata CSV
        
    Returns:
        List of dictionaries with metadata rows
    """
    logger.info(f"Loading metadata from: {metadata_path}")
    rows = []
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        logger.info(f"Loaded {len(rows)} rows")
        return rows
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        raise


def get_class_distribution(rows: list, class_col: str = 'age_group') -> dict:
    """
    Get distribution of classes.
    
    Args:
        rows: List of row dictionaries
        class_col: Column name for class
        
    Returns:
        Dictionary with class counts
    """
    dist = defaultdict(int)
    for row in rows:
        class_name = row.get(class_col, '').strip()
        if class_name:
            dist[class_name] += 1
    return {k: int(v) for k, v in sorted(dist.items())}


def balance_dataset(rows: list, class_col: str = 'age_group', 
                   random_state: int = RANDOM_SEED) -> tuple:
    """
    Balance dataset by randomly sampling equal records from each class.
    
    Args:
        rows: List of row dictionaries
        class_col: Column name for class
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (balanced rows list, min class count)
    """
    # Set random seed for reproducibility
    random.seed(random_state)
    
    # Group rows by class
    class_groups = defaultdict(list)
    for row in rows:
        class_name = row.get(class_col, '').strip()
        if class_name:
            class_groups[class_name].append(row)
    
    # Get minimum count
    min_count = min(len(rows) for rows in class_groups.values())
    
    logger.info(f"\nClass counts before balancing:")
    for class_name in sorted(class_groups.keys()):
        count = len(class_groups[class_name])
        logger.info(f"  {class_name}: {count}")
    
    logger.info(f"\nMinimum class count: {min_count}")
    logger.info(f"Balancing dataset by sampling {min_count} records per class...")
    
    # Sample equal records from each class
    balanced_rows = []
    for class_name in sorted(class_groups.keys()):
        class_rows = class_groups[class_name]
        sampled = random.sample(class_rows, min_count)
        balanced_rows.extend(sampled)
        logger.info(f"  Sampled {len(sampled)} records from {class_name}")
    
    return balanced_rows, min_count


def balance_training_dataset():
    """Main balancing function."""
    
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / 'data/processed/commonvoice_metadata.csv'
    output_dir = project_root / 'data/processed'
    output_csv = output_dir / 'balanced_training.csv'
    output_report = output_dir / 'balance_report.json'
    
    logger.info(f"Starting dataset balancing")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Verify input file exists
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ready: {output_dir}")
    
    # Load metadata
    rows = load_metadata(str(input_file))
    
    # Get distribution before balancing
    logger.info(f"\n{'='*60}")
    logger.info(f"CLASS DISTRIBUTION BEFORE BALANCING")
    logger.info(f"{'='*60}")
    dist_before = get_class_distribution(rows, 'age_group')
    total_before = sum(dist_before.values())
    
    for class_name in sorted(dist_before.keys()):
        count = dist_before[class_name]
        pct = (count / total_before) * 100
        logger.info(f"{class_name:15s}: {count:7d} ({pct:5.2f}%)")
    logger.info(f"{'TOTAL':15s}: {total_before:7d} (100.00%)")
    
    # Balance dataset
    logger.info(f"\n{'='*60}")
    logger.info(f"BALANCING DATASET (seed={RANDOM_SEED})")
    logger.info(f"{'='*60}\n")
    balanced_rows, min_count = balance_dataset(rows, 'age_group', RANDOM_SEED)
    
    # Get distribution after balancing
    logger.info(f"\n{'='*60}")
    logger.info(f"CLASS DISTRIBUTION AFTER BALANCING")
    logger.info(f"{'='*60}")
    dist_after = get_class_distribution(balanced_rows, 'age_group')
    total_after = sum(dist_after.values())
    
    for class_name in sorted(dist_after.keys()):
        count = dist_after[class_name]
        pct = (count / total_after) * 100
        logger.info(f"{class_name:15s}: {count:7d} ({pct:5.2f}%)")
    logger.info(f"{'TOTAL':15s}: {total_after:7d} (100.00%)")
    
    # Save balanced dataset
    logger.info(f"\nSaving balanced dataset to: {output_csv}")
    try:
        if balanced_rows:
            # Get fieldnames from first row
            fieldnames = list(balanced_rows[0].keys())
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(balanced_rows)
            logger.info(f"Saved {len(balanced_rows)} rows to CSV")
    except Exception as e:
        logger.error(f"Error writing CSV: {e}")
        raise
    
    # Prepare report
    report = {
        'before_balancing': {
            'total_records': total_before,
            'class_distribution': dist_before
        },
        'after_balancing': {
            'total_records': total_after,
            'samples_per_class': min_count,
            'class_distribution': dist_after
        },
        'balancing_method': 'random_sampling',
        'random_seed': RANDOM_SEED,
        'reduction': {
            'records_removed': total_before - total_after,
            'records_removed_pct': round((total_before - total_after) / total_before * 100, 2)
        }
    }
    
    # Save report
    logger.info(f"\nSaving report to: {output_report}")
    try:
        with open(output_report, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved successfully")
    except Exception as e:
        logger.error(f"Error writing report: {e}")
        raise
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"BALANCING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Records removed: {report['reduction']['records_removed']:,} ({report['reduction']['records_removed_pct']}%)")
    logger.info(f"Final dataset size: {total_after:,} records")
    logger.info(f"Records per class: {min_count:,}")
    logger.info(f"\nOutput files:")
    logger.info(f"  Dataset: {output_csv}")
    logger.info(f"  Report:  {output_report}")
    logger.info(f"{'='*60}\n")
    
    return report


if __name__ == '__main__':
    try:
        report = balance_training_dataset()
    except Exception as e:
        logger.error(f"Balancing failed: {e}")
        exit(1)
