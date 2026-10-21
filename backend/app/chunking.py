
def split_text_into_chunks(
    text: str,
    chunk_size: int = 200,
    overlap: int = 40
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    words = text.split()
    chunks = []
    current_words = []
    current_length = 0

    for word in words:
        added_length = len(word) if not current_words else len(word) + 1

        if current_words and current_length + added_length > chunk_size:
            chunks.append(" ".join(current_words))

            # Keep whole words from the end for overlap
            overlap_words = []
            overlap_length = 0

            for previous_word in reversed(current_words):
                extra = len(previous_word) if not overlap_words else len(previous_word) + 1

                if overlap_length + extra > overlap:
                    break

                overlap_words.insert(0, previous_word)
                overlap_length += extra

            current_words = overlap_words
            current_length = len(" ".join(current_words))

        added_length = len(word) if not current_words else len(word) + 1
        current_words.append(word)
        current_length += added_length

    if current_words:
        final_chunk = " ".join(current_words)
        if not chunks or final_chunk != chunks[-1]:
            chunks.append(final_chunk)

    return chunks
