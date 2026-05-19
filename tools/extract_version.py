#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

import argparse
import re
import sys

def get_quadriga_lib_version(hpp_path=None):
    """
    Reads the version string directly from quadriga_lib.hpp.
    Parses: #define QUADRIGA_LIB_VERSION_STR "x.y.z"
    """
    search_paths = [hpp_path] if hpp_path else [
        'include/quadriga_lib.hpp',
    ]

    for path in search_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    m = re.search(r'#define\s+QUADRIGA_LIB_VERSION_STR\s+"([^"]+)"', line)
                    if m:
                        return m.group(1)
            print(f"Error: QUADRIGA_LIB_VERSION_STR not found in '{path}'.", file=sys.stderr)
            sys.exit(1)
        except FileNotFoundError:
            continue

    print("Error: quadriga_lib.hpp not found in any expected path.", file=sys.stderr)
    sys.exit(1)

def update_html_first_line(html_path, heading):
    """
    Replaces the first line of the HTML file at html_path with the given heading.
    """
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: file '{html_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error reading '{html_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # Format the new first line
    new_first_line = f'<span class="qd-heading"><b>{heading}</b></span>\n'

    if not lines:
        # If file is empty, just write the new heading as the only line
        lines = [new_first_line]
    else:
        # Replace the first line
        lines[0] = new_first_line

    # Write back the modified content
    try:
        with open(html_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    except IOError as e:
        print(f"Error writing to '{html_path}': {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Update the first line of an HTML file with a heading that includes the quadriga-lib version."
    )
    parser.add_argument(
        'html_file',
        help="Path to the HTML file to modify."
    )
    parser.add_argument(
        'headline',
        help="Base headline string (e.g. 'MALAB / Octave API Documentation for Quadriga-Lib')."
    )
    parser.add_argument(
        '--version-path',
        help="Path to the quadriga-lib version executable. If not provided, searches common paths.",
        default=None
    )

    args = parser.parse_args()

    # 1. Get version from 'quadriga-lib-version'
    version = get_quadriga_lib_version(args.version_path)

    # 2. Append version to the provided headline
    full_heading = f"{args.headline} v{version}"

    # 3. Update the first line of the HTML file
    update_html_first_line(args.html_file, full_heading)

if __name__ == '__main__':
    main()