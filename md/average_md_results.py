#!/usr/bin/env python3
"""
MD Benchmark Results Aggregator
Aggregates individual MD run outputs across multiple runs and thread counts
Handles line-wrapped output format and extracts energy + force metrics

Same format as matMul: only algorithm names (Native, Kahan, Neumaier), no Serial/Parallel prefix
Only Avg values for force metrics (not Max/Min)

Usage:
  python3 average_md_results.py <test_name> <threads|serial>
  
Examples:
  python3 average_md_results.py md_3d_300_500 4
  python3 average_md_results.py md_2d_500_500 serial
  python3 average_md_results.py md_3d_200_200 8
"""

import sys
import re
import glob
from collections import defaultdict

def process_md_files(files, mode):
    """
    Process MD output files and aggregate results.
    For serial mode: aggregates all 15 runs (5×4threads + 5×8threads + 5×16threads)
    For parallel mode: aggregates 5 runs for specific thread count
    
    Reads from run files which have TWO TABLES: energy table and force table
    
    Returns:
        dict: Aggregated data by algorithm (Native, Kahan, Neumaier)
    """
    data = defaultdict(lambda: {
        'abs_err': [],
        'rel_err': [],
        'time': [],
        'max_abs_force': [],
        'max_rel_force': [],
        'vec_abs_error': [],
        'vec_rel_error': [],
        'net_force_drift': [],
    })
    
    for filename in files:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Fix line wrapping: join lines that start with whitespace followed by exponent
        content = re.sub(r'\n\s+([e\-0-9])', r' \1', content)
        
        # Split by tables
        lines = content.split('\n')
        in_energy_table = False
        in_force_table = False
        
        for line in lines:
            # Detect table headers
            if 'energy table' in line.lower():
                in_energy_table = True
                in_force_table = False
                continue
            if 'force table' in line.lower():
                in_energy_table = False
                in_force_table = True
                continue
            
            # Skip headers and separators
            if any(skip in line for skip in ['Variant', '---', 'Error', 'Drift']):
                continue
            
            if not line.strip():
                continue
            
            # Parse line - split by |
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 2:
                continue
            
            algorithm_full = parts[0]
            
            # Extract algorithm name (remove Serial/Parallel prefix)
            if algorithm_full.startswith("Serial "):
                algorithm = algorithm_full[7:]  # Remove "Serial "
            elif algorithm_full.startswith("Parallel "):
                algorithm = algorithm_full[9:]  # Remove "Parallel "
            else:
                algorithm = algorithm_full
            
            # Filter by mode
            if mode == "serial" and not algorithm_full.startswith("Serial"):
                continue
            if mode == "parallel" and not algorithm_full.startswith("Parallel"):
                continue
            
            # Parse energy table (columns: AbsErr, RelErr, Time)
            if in_energy_table and len(parts) >= 4:
                try:
                    abs_err = float(parts[1])
                    rel_err = float(parts[2])
                    time_str = parts[3].rstrip('s')
                    time_val = float(time_str)
                    
                    data[algorithm]['abs_err'].append(abs_err)
                    data[algorithm]['rel_err'].append(rel_err)
                    data[algorithm]['time'].append(time_val)
                except (ValueError, IndexError):
                    pass
            
            # Parse force table (columns: MaxAbsForceDiff, MaxRelForceDiff, VecAbsError, VecRelError, NetForceDrift)
            if in_force_table and len(parts) >= 6:
                try:
                    max_abs_force = float(parts[1])
                    max_rel_force = float(parts[2])
                    vec_abs_error = float(parts[3])
                    vec_rel_error = float(parts[4])
                    net_force_drift = float(parts[5])
                    
                    data[algorithm]['max_abs_force'].append(max_abs_force)
                    data[algorithm]['max_rel_force'].append(max_rel_force)
                    data[algorithm]['vec_abs_error'].append(vec_abs_error)
                    data[algorithm]['vec_rel_error'].append(vec_rel_error)
                    data[algorithm]['net_force_drift'].append(net_force_drift)
                except (ValueError, IndexError):
                    pass
    
    return data

def print_averages(data):
    """Print aggregated results in matMul-style table format."""
    print(f"{'Algoritam':<18} | {'AbsErr':<12} | {'RelErr':<12} | "
          f"{'Time':<12} | {'MaxAbsForce':<12} | {'MaxRelForce':<12} | {'VecAbsError':<12} | "
          f"{'VecRelError':<12} | {'NetForceDrift':<12}")
    print("-" * 130)
    
    # Sorted algorithms: Native, Kahan, Neumaier (same as matMul)
    algorithms_order = ['Native', 'Kahan', 'Neumaier']
    
    for algorithm in algorithms_order:
        if algorithm not in data:
            continue
        
        info = data[algorithm]
        
        # Check if we have data
        if not info['abs_err']:
            continue
        
        # Calculate averages
        avg_abs = sum(info['abs_err']) / len(info['abs_err'])
        avg_rel = sum(info['rel_err']) / len(info['rel_err'])
        avg_time = sum(info['time']) / len(info['time'])
        
        avg_max_abs_force = sum(info['max_abs_force']) / len(info['max_abs_force']) if info['max_abs_force'] else 0.0
        avg_max_rel_force = sum(info['max_rel_force']) / len(info['max_rel_force']) if info['max_rel_force'] else 0.0
        avg_vec_abs_error = sum(info['vec_abs_error']) / len(info['vec_abs_error']) if info['vec_abs_error'] else 0.0
        avg_vec_rel_error = sum(info['vec_rel_error']) / len(info['vec_rel_error']) if info['vec_rel_error'] else 0.0
        avg_net_force_drift = sum(info['net_force_drift']) / len(info['net_force_drift']) if info['net_force_drift'] else 0.0
        
        print(f"{algorithm:<18} | {avg_abs:<12.6e} | {avg_rel:<12.6e} | "
              f"{avg_time:<12.6e} | {avg_max_abs_force:<12.6e} | {avg_max_rel_force:<12.6e} | "
              f"{avg_vec_abs_error:<12.6e} | {avg_vec_rel_error:<12.6e} | "
              f"{avg_net_force_drift:<12.6e}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 average_md_results.py <test_name> <threads|serial>", file=sys.stderr)
        print("Examples:", file=sys.stderr)
        print("  python3 average_md_results.py md_3d_300_500 4", file=sys.stderr)
        print("  python3 average_md_results.py md_2d_500_500 serial", file=sys.stderr)
        sys.exit(1)
    
    test_name = sys.argv[1]  # e.g., "md_3d_300_500"
    threads_arg = sys.argv[2]  # e.g., "4", "8", "16", "serial"
    
    mode = "serial" if threads_arg == "serial" else "parallel"
    
    # Find matching files
    if threads_arg == "serial":
        pattern = f"{test_name}_threads*_run*.out"
    else:
        pattern = f"{test_name}_threads{threads_arg}_run*.out"
    
    files = sorted(glob.glob(pattern))
    
    if files:
        data = process_md_files(files, mode)
        print_averages(data)
    else:
        print(f"No files matching {pattern}", file=sys.stderr)
        sys.exit(1)
