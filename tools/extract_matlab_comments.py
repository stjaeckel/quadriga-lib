import sys

def extract_matlab_comments(mex_filename, m_filename):
    with open(mex_filename, 'r') as file:
        content = file.read()
    
    # Find MATLAB-specific comment block
    start_tag = '/*!MATLAB'
    end_tag = 'MATLAB!*/'
    
    start_index = content.find(start_tag)
    end_index = content.find(end_tag)
    
    if start_index == -1 or end_index == -1:
        return

    # Extract MATLAB comments
    matlab_comments = content[start_index + len(start_tag):end_index].strip()

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
