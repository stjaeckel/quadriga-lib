#!/usr/bin/env python3
import argparse
import subprocess
import sys

def get_quadriga_lib_version():
    """
    Calls the 'quadriga-lib-version' command and returns its output (version string).
    """
    try:
        # Run the command and capture its output
        output = subprocess.check_output(['build/quadriga-lib-version'], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"Error: failed to run 'quadriga-lib-version': {e.output.decode().strip()}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'quadriga-lib-version' command not found in PATH.", file=sys.stderr)
        sys.exit(1)

    # Decode and strip any trailing newline/whitespace
    version = output.decode('utf-8').strip()
    return version

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
    new_first_line = f"<big><b>{heading}</b></big>\n"

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

    args = parser.parse_args()

    # 1. Get version from 'quadriga-lib-version'
    version = get_quadriga_lib_version()

    # 2. Append version to the provided headline
    full_heading = f"{args.headline} v{version}"

    # 3. Update the first line of the HTML file
    update_html_first_line(args.html_file, full_heading)

if __name__ == '__main__':
    main()
