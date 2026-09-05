import zipfile
import xml.etree.ElementTree as ET
import sys

def get_docx_text(path):
    try:
        with zipfile.ZipFile(path) as docx:
            tree = ET.parse(docx.open('word/document.xml'))
            root = tree.getroot()
            wpn = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
            paragraphs = []
            for p in root.iter(f'{{{wpn}}}p'):
                texts = [node.text for node in p.iter(f'{{{wpn}}}t') if node.text]
                if texts:
                    paragraphs.append(''.join(texts))
            return '\n'.join(paragraphs)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(get_docx_text(sys.argv[1]))
    else:
        print("Please provide a file path")
