#!/bin/bash
# =============================================================================
# debian/stamp-changelog.sh
#
# Single source of truth: include/quadriga_lib.hpp
#   #define QUADRIGA_LIB_VERSION v0_10_9   →   0.10.9
#
# This script rewrites debian/changelog with the correct version and
# distro codename. Run it once before dpkg-buildpackage:
#
#   debian/stamp-changelog.sh          # auto-detect distro codename
#   debian/stamp-changelog.sh noble    # force a specific codename
# =============================================================================
set -euo pipefail

HEADER="include/quadriga_lib.hpp"
CHANGELOG="debian/changelog"

# --- Extract version ---
if [ ! -f "$HEADER" ]; then
    echo "Error: $HEADER not found. Run from the repo root." >&2
    exit 1
fi

RAW=$(grep '#define QUADRIGA_LIB_VERSION' "$HEADER" \
      | sed 's/.*VERSION v//; s/_/./g; s/[[:space:]]*$//')

if [ -z "$RAW" ]; then
    echo "Error: could not parse QUADRIGA_LIB_VERSION from $HEADER" >&2
    exit 1
fi

VERSION="${RAW}"
DEB_VERSION="${VERSION}-1"

# --- Distro codename ---
if [ -n "${1:-}" ]; then
    CODENAME="$1"
elif command -v lsb_release &>/dev/null; then
    CODENAME=$(lsb_release -cs)
else
    CODENAME="noble"
fi

# --- Timestamp ---
TIMESTAMP=$(date -R)

# --- Write changelog ---
cat > "$CHANGELOG" <<EOF
quadriga-lib (${DEB_VERSION}) ${CODENAME}; urgency=medium

  * Packaged from upstream version ${VERSION}.
  * Static library, headers, MATLAB MEX, Octave MEX, Python module.
  * HTML API documentation included.
  * AVX2 acceleration enabled; CUDA excluded from .deb.

 -- Stephan Jaeckel <info@quadriga-lib.org>  ${TIMESTAMP}
EOF

echo "Stamped debian/changelog: quadriga-lib ${DEB_VERSION} for ${CODENAME}"
