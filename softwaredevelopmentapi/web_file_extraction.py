import os
import docx

class CodeSeparator:
    def __init__(self, docx_path, directory):
        self.docx_path = docx_path
        self.directory = directory
        self.text = self.read_docx()
        self.html_code = ""
        self.css_code = ""
        self.js_code = ""

    def read_docx(self):
        doc = docx.Document(self.docx_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)

    def separate_code(self):
        # Finding the HTML code
        html_start = self.text.find("<!DOCTYPE html>")
        css_start = self.text.find("/* style.css */")
        js_start = self.text.find("// script.js")

        if html_start != -1 and css_start != -1 and js_start != -1:
            self.html_code = self.text[html_start:css_start].strip()
            self.css_code = self.text[css_start:js_start].strip()
            self.js_code = self.text[js_start:].strip()

    def save_to_files(self):
        os.makedirs(self.directory, exist_ok=True)

        html_path = os.path.join(self.directory, "index.html")
        css_path = os.path.join(self.directory, "style.css")
        js_path = os.path.join(self.directory, "script.js")

        with open(html_path, 'w') as file:
            file.write(self.html_code)
        
        with open(css_path, 'w') as file:
            file.write(self.css_code)
        
        with open(js_path, 'w') as file:
            file.write(self.js_code)