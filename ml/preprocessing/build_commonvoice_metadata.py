"""
Phase 6: Common Voice Metadata Builder

Processes Common Voice validated.tsv file to build metadata with age group mappings.
Filters rows with age labels, maps to age groups, verifies audio files exist.
Outputs metadata CSV and processing report JSON.
"""

import os
import json
import csv
import logging
from pathlib import Path
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Age group mapping
AGE_GROUP_MAP = {
    'teens': 'Teen',
    'twenties': 'Adult',
    'thirties': 'Adult',
    'fourties': 'Adult',
    'fifties': 'Middle_Age',
    'sixties': 'Senior',
    'seventies': 'Senior',
    'eighties': 'Senior',
    'nineties': 'Senior'
}


def build_audio_path(filename: str, base_path: str) -> str:
    """
    Build full audio file path.
    
    Args:
        filename: The filename from path column
        base_path: Base directory containing clips
        
    Returns:
        Full path to audio file
    """
    return os.path.join(base_path, filename)


def file_exists(filepath: str) -> bool:
    """Check if audio file exists."""
    return os.path.isfile(filepath)


def map_age_group(age: str) -> str:
    """
    Map age string to age group category.
    
    Args:
        age: Age string from validated.tsv
        
    Returns:
        Mapped age group or original age if no mapping exists
    """
    age_lower = age.lower().strip()
    return AGE_GROUP_MAP.get(age_lower, age)


def process_commonvoice_metadata():
    """Main processing function."""
    
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / 'data/raw/common_voice/extracted/cv-corpus-25.0-2026-03-09/en/validated.tsv'
    clips_base = project_root / 'data/raw/common_voice/extracted/cv-corpus-25.0-2026-03-09/en/clips'
    output_dir = project_root / 'data/processed'
    output_csv = output_dir / 'commonvoice_metadata.csv'
    output_report = output_dir / 'commonvoice_report.json'
    
    logger.info(f"Starting Common Voice metadata processing")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Clips base directory: {clips_base}")
    logger.info(f"Output directory: {output_dir}")
    
    # Verify input file exists
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ready: {output_dir}")
    
    # Statistics tracking
    stats = {
        'total_rows': 0,
        'age_labeled_rows': 0,
        'existing_audio_rows': 0,
        'missing_audio_rows': 0,
        'age_group_distribution': defaultdict(int),
        'gender_distribution': defaultdict(int)
    }
    
    metadata_rows = []
    missing_files = []
    
    logger.info("Reading validated.tsv file...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 because header is row 1
                stats['total_rows'] += 1
                
                # Filter: Keep only rows with non-empty age
                age = row.get('age', '').strip()
                if not age:
                    continue
                
                stats['age_labeled_rows'] += 1
                
                # Extract fields
                path = row.get('path', '').strip()
                client_id = row.get('client_id', '').strip()
                sentence = row.get('sentence', '').strip()
                gender = row.get('gender', '').strip()
                locale = row.get('locale', '').strip()
                
                # Build full audio path
                audio_path = build_audio_path(path, str(clips_base))
                
                # Check if file exists
                if not file_exists(audio_path):
                    stats['missing_audio_rows'] += 1
                    missing_files.append({
                        'row': row_num,
                        'path': path,
                        'expected_location': audio_path,
                        'client_id': client_id
                    })
                    continue
                
                stats['existing_audio_rows'] += 1
                
                # Map age group
                age_group = map_age_group(age)
                
                # Track distributions
                stats['age_group_distribution'][age_group] += 1
                if gender:
                    stats['gender_distribution'][gender] += 1
                
                # Add to metadata
                metadata_rows.append({
                    'audio_path': audio_path,
                    'file_name': path,
                    'age': age,
                    'age_group': age_group,
                    'gender': gender,
                    'sentence': sentence,
                    'client_id': client_id,
                    'locale': locale
                })
                
                # Progress logging
                if stats['existing_audio_rows'] % 1000 == 0:
                    logger.info(f"Processed {stats['existing_audio_rows']} valid audio files...")
    
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise
    
    logger.info(f"\nFile reading complete. Summary:")
    logger.info(f"  Total rows: {stats['total_rows']}")
    logger.info(f"  Rows with age label: {stats['age_labeled_rows']}")
    logger.info(f"  Rows with existing audio: {stats['existing_audio_rows']}")
    logger.info(f"  Rows with missing audio: {stats['missing_audio_rows']}")
    
    # Save metadata CSV
    logger.info(f"\nSaving metadata to: {output_csv}")
    
    if metadata_rows:
        fieldnames = [
            'audio_path', 'file_name', 'age', 'age_group',
            'gender', 'sentence', 'client_id', 'locale'
        ]
        
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metadata_rows)
            logger.info(f"Saved {len(metadata_rows)} metadata rows to CSV")
        except Exception as e:
            logger.error(f"Error writing CSV: {e}")
            raise
    else:
        logger.warning("No metadata rows to save")
    
    # Prepare report
    report = {
        'total_rows': stats['total_rows'],
        'age_labeled_rows': stats['age_labeled_rows'],
        'existing_audio_rows': stats['existing_audio_rows'],
        'missing_audio_rows': stats['missing_audio_rows'],
        'age_group_distribution': dict(stats['age_group_distribution']),
        'gender_distribution': dict(stats['gender_distribution']),
        'missing_files_count': len(missing_files),
        'missing_files_sample': missing_files[:10]  # First 10 missing files
    }
    
    # Save report JSON
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
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total input rows: {stats['total_rows']}")
    logger.info(f"Rows with age: {stats['age_labeled_rows']}")
    logger.info(f"Valid audio found: {stats['existing_audio_rows']}")
    logger.info(f"Missing audio: {stats['missing_audio_rows']}")
    logger.info(f"\nAge Group Distribution:")
    for age_group in sorted(report['age_group_distribution'].keys()):
        count = report['age_group_distribution'][age_group]
        logger.info(f"  {age_group}: {count}")
    logger.info(f"\nGender Distribution:")
    for gender in sorted(report['gender_distribution'].keys()):
        count = report['gender_distribution'][gender]
        logger.info(f"  {gender}: {count}")
    logger.info(f"\nOutput files:")
    logger.info(f"  Metadata: {output_csv}")
    logger.info(f"  Report: {output_report}")
    logger.info(f"{'='*60}\n")
    
    return report


if __name__ == '__main__':
    try:
        report = process_commonvoice_metadata()
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        exit(1)
