import sys
import re

def markdown_converter(content_md):
    lines = content_md.split('\n')
    content_mat = []

    for line in lines:
        # Check if line starts with a heading syntax (e.g., #, ##, ###, etc.)
        if line.startswith('```'):
            line = line.replace('`', '')
        elif line.startswith('#'):
            line = line.replace('#', '')
            content_mat.append('%' + line)
        else:
            line = line.replace('<br>', '')
            line = line.replace('*', '')
            line = line.replace('`', '')
            line = re.sub(r'<a href=[^>]*>([^<]*)<\/a>', r'\1', line)
            content_mat.append('%    ' + line)
   
    return '\n'.join(content_mat)

def extract_matlab_comments(mex_filename, m_filename):
    with open(mex_filename, 'r') as file:
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
% Copyright (C) 2022-2024 Stephan Jaeckel (https://sjc-wireless.com)
% All rights reserved.
%
% e-mail: info@sjc-wireless.com
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
    with open(m_filename, 'w') as file:
        file.write(matlab_comments)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_comments.py <path_to_your_mex_file.cpp> <output.m>")
        sys.exit(1)

    mex_file = sys.argv[1]
    m_file = sys.argv[2]
    extract_matlab_comments(mex_file, m_file)
