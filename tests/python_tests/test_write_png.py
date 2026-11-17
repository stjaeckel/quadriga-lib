# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

import sys
import os
import pathlib
import tempfile
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing your package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib

try:
    from PIL import Image
except ImportError:  # the decoding checks need Pillow
    Image = None

class test_case(unittest.TestCase):

    def test(self):

        N = 50                 # size along one edge
        max_val = 10.0          # value at the upper‑right corner

        # Create a left‑to‑right gradient (columns vary, rows repeat)
        gradient = np.tile(np.linspace(0.0, max_val, N, endpoint=False), (25, 1))
        gradient[0,:] = 0.0
        gradient[-1,:] = 10.0

        # File name to write
        out_file = "test.png"

        # quadriga_lib.tools.write_png(fn, data, colormap, min_val, max_val, log_transform)
        quadriga_lib.tools.write_png(
            out_file,
            gradient,
            "jet",          # any valid Matplotlib‑style colormap string works
            0.0,                # min_val  (matches gradient.min())
            max_val,            # max_val  (matches gradient.max())
            False               # log_transform
        )

        # Check that the file was created
        self.assertTrue(os.path.exists(out_file))

        # Clean up
        os.remove(out_file)

    @unittest.skipUnless(Image is not None, "Pillow is not installed")
    def test_degenerate_range_nan_and_errors(self):
        out_file = "test_png_range.png"
        first, last = (0, 0, 131), (128, 0, 0)  # first and last colour of "jet"

        def pixels(data, *args):
            quadriga_lib.tools.write_png(out_file, data, "jet", *args)
            with Image.open(out_file) as im:
                px = np.asarray(im.convert("RGB"))
            os.remove(out_file)
            return px

        # Constant data, a single value and all-NaN data give the first colour everywhere
        for data in (np.full((4, 3), 2.5), np.array([[7.0]]), np.full((4, 3), np.nan)):
            px = pixels(data)
            self.assertEqual(px.shape, data.shape + (3,))
            npt.assert_array_equal(px.reshape(-1, 3), np.tile(first, (data.size, 1)))

        # NaN pixels get the first colour, +-Inf are clipped and do not change the auto range
        px = pixels(np.array([[0.0, 1.0, np.inf, -np.inf, np.nan]]))
        npt.assert_array_equal(px[0], [first, last, last, first, first])

        # Inverted user range and empty data raise
        with self.assertRaises(ValueError):
            quadriga_lib.tools.write_png(out_file, np.ones((2, 2)), "jet", 2.0, 1.0)
        with self.assertRaises(ValueError):
            quadriga_lib.tools.write_png(out_file, np.zeros((0, 3)))
        with self.assertRaises(ValueError):  # infinite limits
            quadriga_lib.tools.write_png(out_file, np.ones((2, 2)), "jet", -np.inf)
        self.assertFalse(os.path.exists(out_file))

    def test_path_none_and_flag_types(self):
        data = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 6.0]])
        with tempfile.TemporaryDirectory() as tmp:
            # fn as str and as pathlib.Path; log_transform None selects the default (False)
            fn_str = os.path.join(tmp, "a.png")
            fn_path = pathlib.Path(tmp) / "b.png"
            quadriga_lib.tools.write_png(fn_str, data)
            quadriga_lib.tools.write_png(fn_path, data, log_transform=None)
            with open(fn_str, "rb") as fa, open(fn_path, "rb") as fb:
                self.assertEqual(fa.read(), fb.read())

            with self.assertRaises(TypeError) as ctx:
                quadriga_lib.tools.write_png(fn_str, data, log_transform=2.5)
            self.assertIn("'log_transform'", str(ctx.exception))

            # data is mandatory
            with self.assertRaises(TypeError):
                quadriga_lib.tools.write_png(fn_str)
            with self.assertRaises(TypeError):
                quadriga_lib.tools.write_png(fn_str, None)
