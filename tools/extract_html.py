import os
import re
import sys
import hashlib
import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate HTML from files with optional preamble."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to the output HTML file."
    )
    parser.add_argument(
        "-p", "--preamble",
        required=True,
        help="Optional path to preamble HTML file."
    )
    parser.add_argument(
        "-d", "--directory",
        default="",
        help="Optional path to folder containing source files."
    )
    parser.add_argument("-c", action="store_true", help="Compact format")
    return parser


def extract_sections(file_content):
    sections = re.findall(r'\/\*!SECTION\n(.*?)\nSECTION!\*\/', file_content, re.DOTALL)
    return sections


def extract_md_sections(file_content):
    md_sections = re.findall(r'\/\*!MD\n(.*?)\nMD!\*\/', file_content, re.DOTALL)
    return md_sections


def extract_sections_desc(file_content):
    md_sections = re.findall(r'\/\*!SECTION_DESC\n(.*?)\SECTION_DESC!\*\/', file_content, re.DOTALL)
    return md_sections


def escape_code(match):
    code = match.group(1)
    code = re.sub(r'&', '&amp;', code)
    code = re.sub(r'<', '&lt;', code)
    code = re.sub(r'>', '&gt;', code)
    return f'<code>{code}</code>'


def format_text(text):
     # Replace `code` with <code>code</code>
    text = re.sub(r'`(.*?)`', escape_code, text)

    # Pointers
    text = re.sub(r'\*\*\*\*(.*?)\*\*', r'**&#42;&#42;\1**', text)

    # Replace **bold** with <b>bold</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    
    # Replace *italic* with <i>italic</i>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
      
    # Encode [[]] as internal links
    text = re.sub(r'\[\[(.*?)\]\]', r'<a href="#\1">\1</a>', text)

    return text


def format_nested_lists(text):
    lines = text.split('\n')
    formatted_text = ""
    inside_list = False

    for i in range(len(lines)):
        line = lines[i]
        if line.startswith("- "):  # Start of a new list item
            if not inside_list:
                inside_list = True
                formatted_text += "<ul>"
            else:
                formatted_text += "</li>"
            formatted_text += '<li style="margin-bottom: 0.4em;">' + line[2:] + '\n'
        elif (line.startswith("  ") or line.startswith("  ")) and inside_list:  # Continuation of the current list item
            formatted_text += " " + line + "\n"
        else:  # End of the list or some other content
            if inside_list:
                inside_list = False
                formatted_text += "</li></ul>\n"
            formatted_text += line + "\n"

        if inside_list and i == len(lines) - 1:  # If the last line was part of a list
            formatted_text += "</li></ul>\n"

    return formatted_text


def format_tables(text):
    lines = text.split('\n')
    html_content = ""
    in_table = False
    table_lines = []

    def process_table(table_lines):
        if not table_lines:
            return ""
        html_content = '<table style="border-collapse: separate; border-spacing: 20px 0;">\n'
        # Identify alignment line if present
        alignment_line = None
        if len(table_lines) > 1 and all(c in "|-: " for c in table_lines[1]):
            alignment_line = table_lines.pop(1)
        # Process table content
        for i, line in enumerate(table_lines):
            cols = line.strip('|').split('|')
            html_content += "<tr>\n"
            for col in cols:
                if i == 0 and alignment_line is not None:
                    # If this is the first line, and an alignment line was found, use <th>
                    html_content += f"  <th>{col.strip()}</th>\n"
                else:
                    html_content += f"  <td>{col.strip()}</td>\n"
            html_content += "</tr>\n"
        html_content += "</table><br>\n"

        return html_content

    in_pre = False
    for line in lines:
        # Track <pre> blocks and skip table parsing inside them
        if "<pre>" in line:
            in_pre = True
        if "</pre>" in line:
            in_pre = False
            html_content += line + '\n'
            continue
        if in_pre:
            html_content += line + '\n'
            continue
        # Check if we are entering a table
        if "|" in line and not in_table:
            in_table = True
            table_lines.append(line)
        # Check if we are inside a table
        elif in_table:
            if "|" not in line:
                in_table = False
                # Convert table to HTML
                if table_lines:
                    html_content += process_table(table_lines)
                table_lines = []
                html_content += line + '\n'
            else:
                table_lines.append(line)
        # If we are not in a table, just add the line to the content
        else:
            html_content += line + '\n'

    # In case the file ends with a table
    if in_table and table_lines:
        html_content += process_table(table_lines)

    return html_content


def generate_html(folder_name, html_output_file, html_preamble, compact):

    # Read the content of the file 'tools/html_parts/html_head.html.part'
    with open("tools/html_parts/html_head.html.part", "r") as head_file:
        head_content = head_file.read()

    # Search and replace the specified line
    filename_html = os.path.basename(html_output_file)
    head_content = head_content.replace(
        f'<a class="quadriga-lib_menu" href="{filename_html}">',
        f'<a class="quadriga-lib_menu_selected" href="{filename_html}">'
    )
    html_content = head_content

    if len(html_preamble) > 0:
        with open(html_preamble, "r") as preamble_file:
            html_content += preamble_file.read()

    # Parse source files and create documentation
    if len(folder_name) > 0:

        # Append the content to the beginning of the html_content
        html_content += "\n<b>Overview</b>\n<ul>\n"

        files = os.listdir(folder_name)
        section_dict = {}
        section_desc_dict = {}
        sections = []

        for file_name in sorted(files):
            if file_name.endswith('.cpp') or file_name.endswith('.md'):
                with open(os.path.join(folder_name, file_name), 'r') as f:
                    content = f.read()
                    if '/*!SECTION' not in content:
                        continue

                    file_sections = extract_sections(content)
                    file_md_sections = extract_md_sections(content)
                    file_sections_desc = extract_sections_desc(content)

                    for section in file_sections:
                        if section not in section_dict:
                            section_dict[section] = []
                            section_desc_dict[section] = []
                        
                        for md_section in file_md_sections:
                            lines = md_section.split('\n')
                            function_name = lines[0].replace("# ", "").lower()
                            if "<++>" in function_name:
                                add_space = 1;
                            else:
                                add_space = 0;
                            function_name = function_name.replace("<++>", "")
                            if function_name not in [func[0] for func in section_dict[section]]:
                                section_dict[section].append((function_name, md_section, add_space))

                        for sections_desc in file_sections_desc:
                            section_desc_dict[section].append(sections_desc)


        section_dict = dict(sorted(section_dict.items()))
        section_desc_dict = dict(sorted(section_desc_dict.items()))
        sections = list(section_dict.keys())
        
        # Section list
        for idx, section in enumerate(sections):
            h = hashlib.shake_256(section.encode("utf-8"))
            html_content += f'<li><a href="#{h.hexdigest(4)}">{section}</a></li>\n'

        html_content += "</ul>\n<br>\n"

        # Print functions per section
        if not compact:
            for idx, section in enumerate(sections):
                html_content += f'<b>{section}</b>\n<ul>\n<table style="border-collapse: separate; border-spacing: 20px 0;">\n<tbody>\n'

                for function_name, md_section, add_space in section_dict[section]:
                    lines = md_section.split('\n')
                    description = lines[1]
                    padding = ""
                    if add_space:
                        padding = ' style="padding-bottom: 10px;"'
                    html_content += f'<tr><td{padding}><a href="#{function_name}">{function_name}</a></td><td{padding}>{description}</td></tr>\n'

                html_content += '</tbody>\n</table>\n</ul>\n<br>\n'

        for idx, section in enumerate(sections):
            h = hashlib.shake_256(section.encode("utf-8"))
            if compact:
                html_content += f'<hr class="greyline">\n<br>\n'
            else:
                html_content += f'<div class="pagebreak"></div>\n<hr class="greyline">\n<hr class="greyline">\n<br>\n<br>\n'
            html_content += f'<a name="{h.hexdigest(4)}"></a>\n'
            html_content += f'<font size=+1><b>{section}</b></font>\n<br>\n'
            if not compact:
                html_content += f'<br>\n'

            for sections_desc in section_desc_dict[section]:
                lines = sections_desc.split('\n')

                section_content = ""
                i = 0
                while i < len(lines):
                    if lines[i].strip() == "```":  # Start of code block
                        section_content += "<pre>\n"
                        i += 1
                        while i < len(lines) and lines[i].strip() != "```":  # End of code block
                            ln = lines[i]
                            ln = ln.replace('&','&amp;')
                            ln = ln.replace("<","&lt;")
                            ln = ln.replace(">","&gt;")
                            section_content += ln + "\n"
                            i += 1
                        section_content += "</pre>\n"
                        i += 1  # Move past the ending backticks
                    else:
                        ln = format_text(lines[i])
                        section_content += ln + " \n"
                        i += 1

                # Apply text formatting
                section_content = format_nested_lists(section_content)
                section_content = format_tables(section_content)
                
                html_content += f'{section_content.strip()}\n'


            for function_name, md_section, add_space in section_dict[section]:
                lines = md_section.split('\n')

                short_description = ""
                if len(lines) > 1:
                    short_description = lines[1]
               
                if len(short_description) > 0:
                    html_content += f'<div class="pagebreak"></div><div class="noprint"><hr class="greyline"><br></div>\n<a name="{function_name}"></a>\n<b>{function_name}</b> - {short_description}\n<ul>\n'
                else:
                    html_content += f'<div class="pagebreak"></div><div class="noprint"><hr class="greyline"><br></div>\n<a name="{function_name}"></a>\n<b>{function_name}</b>\n<ul>\n'

                i = 2  # Start after the function name and short description
                while i < len(lines):
                    line = lines[i]
                    if line.startswith("## "):  # Detected a section
                        section_name = line.replace("## ", "")
                        i += 1
                        section_content = ""
                        while i < len(lines) and not lines[i].startswith("## "):
                            if lines[i].strip() == "```":  # Start of code block
                                section_content += "<pre>\n"
                                i += 1
                                while i < len(lines) and lines[i].strip() != "```":  # End of code block
                                    ln = lines[i]
                                    ln = ln.replace('&','&amp;')
                                    ln = ln.replace("<","&lt;")
                                    ln = ln.replace(">","&gt;")
                                    section_content += ln + "\n"
                                    i += 1
                                section_content += "</pre>\n"
                                i += 1  # Move past the ending backticks
                            else:
                                ln = format_text(lines[i])
                                section_content += ln + " \n"
                                i += 1

                        # Apply text formatting
                        section_content = format_nested_lists(section_content)
                        section_content = format_tables(section_content)
                        #section_content = format_text(section_content)

                        html_content += f'<li style="margin-bottom: 1.5em;">\n<b><i>{section_name}</i></b><br>{section_content.strip()}\n</li>\n'
                    else:
                        i += 1

                html_content += '</ul>\n'

    # Read the content of the file 'tools/html_parts/html_foot.html.part'
    with open("tools/html_parts/html_foot.html.part", "r") as foot_file:
        foot_content = foot_file.read()

    html_content = html_content + foot_content

    return html_content


def main():
    parser = get_parser()
    args = parser.parse_args()

    html_output = generate_html(args.directory, args.output, args.preamble, args.c)

    with open(args.output, "w") as f:
        f.write(html_output)

    print(f"HTML output saved to {args.output}")


if __name__ == "__main__":
    main()

