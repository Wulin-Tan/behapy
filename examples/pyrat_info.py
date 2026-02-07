import pandas as pd
import h5py
import os
from pathlib import Path
import subprocess

def get_dir_tree(path, max_depth=3):
    try:
        result = subprocess.run(['tree', '-L', str(max_depth), str(path)], capture_output=True, text=True)
        return result.stdout
    except FileNotFoundError:
        return "tree command not found."

def inspect_csv(file_path):
    try:
        # Try reading first few rows to detect header
        df_peek = pd.read_csv(file_path, header=None, nrows=5)
        
        # Check if it looks like DLC multi-animal or single animal
        is_dlc = False
        header_type = "Flat"
        bodyparts = []
        
        if any("scorer" in str(val).lower() for val in df_peek.iloc[0].values):
            is_dlc = True
            # Peek further for headers
            if any("individuals" in str(val).lower() for val in df_peek.iloc[1].values):
                header_type = "DLC Multi-Animal (4 levels)"
                header_rows = [0, 1, 2, 3]
            else:
                header_type = "DLC Single Animal (3 levels)"
                header_rows = [0, 1, 2]
            
            df = pd.read_csv(file_path, header=header_rows, index_col=0)
            if header_type == "DLC Multi-Animal (4 levels)":
                bodyparts = list(df.columns.get_level_values(2).unique())
            else:
                bodyparts = list(df.columns.get_level_values(1).unique())
        else:
            df = pd.read_csv(file_path, nrows=10)
            bodyparts = list(df.columns)

        return f"""
#### CSV Inspection: {file_path.name}
- **Header Type**: {header_type}
- **Detected as DLC**: {is_dlc}
- **Shape**: {df.shape} (showing partial rows)
- **Bodyparts**: {', '.join(map(str, bodyparts[:10]))}{'...' if len(bodyparts) > 10 else ''}
- **First few rows**:
```text
{df.head(3).to_string()}
```
"""
    except Exception as e:
        return f"Error inspecting CSV: {e}"

def inspect_h5(file_path):
    try:
        report = [f"#### H5 Inspection: {file_path.name}"]
        with h5py.File(file_path, 'r') as f:
            report.append(f"- **Top-level keys**: {list(f.keys())}")
            
            def get_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    return f"{name} (Dataset, {obj.shape}, {obj.dtype})"
                return f"{name} (Group)"

            structure = []
            f.visititems(lambda name, obj: structure.append("  " * name.count('/') + get_structure(name, obj)))
            
            report.append("- **Structure Tree (truncated)**:")
            report.append("```text\n" + "\n".join(structure[:20]) + ("\n..." if len(structure) > 20 else "") + "\n```")
            
            # Sample data from first dataset
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data = f[key][()]
                    report.append(f"- **Sample Data ({key})**:")
                    report.append(f"```text\n{str(data)[:200]}...\n```")
                    break
        return "\n".join(report)
    except Exception as e:
        return f"Error inspecting H5: {e}"

def main():
    pyrat_dir = Path('reference/PyRAT_dataset')
    output_file = Path('docs/pyrat_dataset_info.md')
    
    if not pyrat_dir.exists():
        print(f"Error: {pyrat_dir} does not exist.")
        return

    all_files = list(pyrat_dir.rglob('*'))
    data_extensions = ['.h5', '.csv', '.hdf5', '.slp']
    data_files = [f for f in all_files if f.suffix.lower() in data_extensions]
    doc_files = [f for f in all_files if f.name.upper().startswith('README') or f.suffix.lower() in ['.txt', '.md']]

    report = []
    report.append("# PyRAT Dataset Information Report\n")
    
    # 1. File Summary
    report.append("## 1. File Summary")
    total_size = sum(f.stat().st_size for f in all_files if f.is_file())
    report.append(f"- **Total Files**: {len([f for f in all_files if f.is_file()])}")
    report.append(f"- **Data Files**: {len(data_files)}")
    report.append(f"- **Total Size**: {total_size / (1024*1024):.2f} MB")
    
    # 2. Data Files Table
    report.append("\n## 2. Data Files")
    report.append("| File Name | Relative Path | Size (KB) | Extension |")
    report.append("| --- | --- | --- | --- |")
    for f in sorted(data_files):
        report.append(f"| {f.name} | {f.relative_to(pyrat_dir)} | {f.stat().st_size / 1024:.1f} | {f.suffix} |")

    # 3. Inspect First Data File
    report.append("\n## 3. First Data File Inspection")
    if data_files:
        first_file = sorted(data_files)[0]
        if first_file.suffix.lower() == '.csv':
            report.append(inspect_csv(first_file))
        elif first_file.suffix.lower() in ['.h5', '.hdf5']:
            report.append(inspect_h5(first_file))
        else:
            report.append(f"Inspection for {first_file.suffix} not implemented yet.")
    else:
        report.append("No data files found for inspection.")

    # 4. Documentation
    report.append("\n## 4. Documentation Found")
    if doc_files:
        for doc in doc_files:
            report.append(f"- {doc.relative_to(pyrat_dir)}")
    else:
        report.append("No documentation files found.")

    # 5. Directory Tree
    report.append("\n## 5. Directory Tree (Max Depth 3)")
    report.append("```text\n" + get_dir_tree(pyrat_dir) + "\n```")

    # 6. Manual Notes
    report.append("\n## 6. Manual Notes")
    report.append("- **Animal Type**: Unknown (Needs verification)")
    report.append("- **Behavior**: Unknown")
    report.append("- **Tracking Software**: Detected from headers (see Section 3)")
    report.append("- **FPS**: Unknown")
    report.append("- **Recording Duration**: Unknown")
    report.append("- **Known Issues**: None noted yet")

    with open(output_file, 'w') as f:
        f.write("\n".join(report))
    
    print(f"Report generated at {output_file}")

if __name__ == "__main__":
    main()
