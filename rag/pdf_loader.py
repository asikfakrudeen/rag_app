import os
import fitz


def load_pdf(pdf_path):
    """
    Opens a PDF file, extracts text from each page, and returns a list of dictionaries
    containing the text, page number, and source file name.
    
    Example:
        Input: load_pdf("contract.pdf")
        Output: [
            {"text": "Amendment 1: Termination...", "page": 1, "source": "contract.pdf"},
            {"text": "Section 2: Renewals...", "page": 2, "source": "contract.pdf"}
        ]
    """

    # Open the PDF document using PyMuPDF (fitz)
    document = fitz.open(pdf_path)

    pages = []

    # Iterate through every page in the document
    for page_number, page in enumerate(document):

        # Extract all visible text from the current page
        text = page.get_text()

        # Check if the text on the page is not empty (ignoring whitespace)
        if text.strip():

            # Save the text along with its metadata (page number and filename)
            pages.append({
                "text": text,
                "page": page_number + 1,  # Adding 1 since enumeration starts at 0
                "source": os.path.basename(pdf_path)
            })

    # Close the document to free up system resources
    document.close()

    return pages
