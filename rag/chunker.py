def chunk_text(text, chunk_size, overlap):
    """
    Splits a large text into smaller chunks of a specified size (in words).
    The 'overlap' parameter ensures that consecutive chunks share some words,
    preventing mid-sentence breaks from losing necessary context.

    Example:
        Input: chunk_text("Hello world this is a test", chunk_size=3, overlap=1)
        Output: ["Hello world this", "this is a", "a test"]
    """

    # Split the incoming string into a list of words
    words = text.split()

    chunks = []
    start = 0

    # Iterate until we reach the end of the word list
    while start < len(words):

        # Calculate the end index for the current chunk
        end = start + chunk_size

        # Join the sliced words back into a single string chunk
        chunk = " ".join(words[start:end])

        # If the chunk isn't just empty space, add it to our chunk list
        if chunk.strip():
            chunks.append(chunk)

        # Move the start pointer forward, applying the overlap to the next chunk
        start += chunk_size - overlap

    return chunks


def create_chunks(pages, chunk_size, overlap):
    """
    Takes a list of page dictionaries (from the pdf_loader) and iterates over them,
    chunking the text for each page while keeping track of metadata (page number, source).

    Example:
        Input: 
            pages = [{"text": "Hello world this is a test", "page": 1, "source": "doc.pdf"}]
            create_chunks(pages, chunk_size=3, overlap=1)
            
        Output:
            [
                {"id": "doc.pdf_1_0", "text": "Hello world this", "page": 1, "source": "doc.pdf"},
                {"id": "doc.pdf_1_1", "text": "this is a", "page": 1, "source": "doc.pdf"},
                {"id": "doc.pdf_1_2", "text": "a test", "page": 1, "source": "doc.pdf"}
            ]
    """

    all_chunks = []

    # Loop through each page in the loaded document
    for page in pages:

        # Chunk the text found on this specific page
        chunks = chunk_text(
            page["text"],
            chunk_size,
            overlap
        )

        # Add the newly created chunks to the master list, preserving their metadata
        for i, chunk in enumerate(chunks):

            all_chunks.append({
                "id": f"{page['source']}_{page['page']}_{i}", # Create a unique ID for ChromaDB
                "text": chunk,
                "page": page["page"],
                "source": page["source"]
            })

    return all_chunks
