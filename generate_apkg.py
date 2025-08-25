#!/usr/bin/env python3
"""
Convenient script runner for APKG generation from the repository root.
This script automatically handles the correct paths and runs the conversion tool.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the APKG conversion tool from the repository root"""
    
    # Change to tools/scripts directory
    script_dir = Path('tools/scripts')
    if not script_dir.exists():
        print("❌ Error: tools/scripts directory not found!")
        print("📍 Make sure you're running this from the repository root.")
        sys.exit(1)
    
    # Check if the conversion script exists
    conversion_script = script_dir / 'convert_json_to_apkg.py'
    if not conversion_script.exists():
        print(f"❌ Error: {conversion_script} not found!")
        sys.exit(1)
    
    print("🚀 Running APKG generation from repository root...")
    print(f"📁 Script location: {conversion_script}")
    
    # Change to the script directory and run the conversion tool
    try:
        os.chdir(script_dir)
        subprocess.run([sys.executable, 'convert_json_to_apkg.py'], check=True)
        print("✅ APKG generation completed successfully!")
        print("📦 Check data/output/apkg_files/ for generated files.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running conversion script: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)

if __name__ == '__main__':
    main()
