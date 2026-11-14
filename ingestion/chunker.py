def chunk_text(
        text,
        chunk_size=160,
        overlap=30
):
    """
    Split text into overlapping word-based chunks.

    chunk_size = number of words per chunk
    overlap    = repeated words between neighboring chunks
    """

    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunks.append(
            " ".join(chunk_words)
        )

        if end >= len(words):
            break

        start = end - overlap

    return chunks


if __name__ == "__main__":

    sample = " ".join(
        [f"word{i}" for i in range(500)]
    )

    chunks = chunk_text(sample)

    print("Number of chunks:", len(chunks))

    for i, chunk in enumerate(chunks):
        print(
            f"Chunk {i}: "
            f"{len(chunk.split())} words"
        )