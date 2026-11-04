# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.
"""MATLAB / Octave help text (.m file) from the first /*!MD block of a MEX source file.

Usage: python3 extract_matlab_comments.py <mex source .cpp> <output .m>
Standard library only; runs on Python 3.10 and on Windows.
"""

import sys
import re


def plain_links(line):
    """[[name]] and .[[name]] become the name, [text](url) becomes 'text (url)'."""
    line = re.sub(r"\[\[([^\[\]]+)\]\]", r"\1", line)
    return re.sub(r"\[([^\]]*)\]\(([^)\s]*)\)", r"\1 (\2)", line)


def markdown_converter(content_md):
    lines = content_md.split('\n')
    content_mat = []

    for ind, line in enumerate(lines):
        line = plain_links(line)
        # Check if line starts with a heading syntax (e.g., #, ##, ###, etc.)
        if ind < 2:
            content_mat.append('% ' + line)
        elif line.startswith('```'):
            line = line.replace('`', '')
        elif line.startswith('#'):
            line = line.replace('#', '')
            line = line.replace('<br>', '')
            content_mat.append('%' + line)
        elif line.lstrip().startswith('|'):
            line = line.replace('*', ' ')
            line = line.replace('`', ' ')
            content_mat.append('%   ' + line)
        else:
            line = line.replace('<br>', '')
            line = line.replace('*', '')
            line = line.replace('`', '')
            line = re.sub(r'<a href=[^>]*>([^<]*)<\/a>', r'\1', line)
            content_mat.append('%   ' + line)
   
    return '\n'.join(content_mat)

def extract_matlab_comments(mex_filename, m_filename):
    with open(mex_filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find comment block
    start_tag = '/*!MD'
    end_tag = 'MD!*/'
    
    start_index = content.find(start_tag)
    end_index = content.find(end_tag)
    
    if start_index == -1 or end_index == -1:
        return

    # Extract MATLAB comments
    md_comments = content[start_index + len(start_tag):end_index].strip()
    matlab_comments = markdown_converter(md_comments)

    disclaimer = """
%
%
% quadriga-lib c++/MEX Utility library for radio channel modelling and simulations
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% All rights reserved.
%
% e-mail: info@quadriga-lib.org
%
% Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
% in compliance with the License. You may obtain a copy of the License at
% http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software distributed under the License
% is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
% or implied. See the License for the specific language governing permissions and limitations under
% the License.
    """
    matlab_comments = matlab_comments + disclaimer

    # Write to .m file
    with open(m_filename, 'w', encoding='utf-8') as file:
        file.write(matlab_comments)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 extract_matlab_comments.py <mex source .cpp> <output .m>")
        sys.exit(1)

    mex_file = sys.argv[1]
    m_file = sys.argv[2]
    extract_matlab_comments(mex_file, m_file)
