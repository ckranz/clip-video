"""Phonetic encoding algorithms for fuzzy matching.

Implements Soundex and Metaphone algorithms for phonetic similarity matching,
plus Levenshtein distance for string edit distance calculations.
"""

from __future__ import annotations


def soundex(word: str) -> str:
    """Generate Soundex code for a word.

    Soundex encodes consonants by sound, ignoring vowels (except at start),
    to group words that sound similar together.

    Args:
        word: Word to encode

    Returns:
        Four-character Soundex code (e.g., "K162" for "Kubernetes")
    """
    if not word:
        return "0000"

    word = word.upper()

    # Remove non-alphabetic characters
    word = "".join(c for c in word if c.isalpha())

    if not word:
        return "0000"

    # Soundex encoding map
    # Letters that sound similar are grouped together
    encoding_map = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
        # A, E, I, O, U, H, W, Y are ignored (mapped to "")
    }

    # Keep first letter
    result = word[0]

    # Encode remaining letters
    prev_code = encoding_map.get(word[0], "")

    for char in word[1:]:
        code = encoding_map.get(char, "")
        if code and code != prev_code:
            result += code
            prev_code = code
        elif not code:
            # Vowels/H/W/Y act as separators but don't get a code
            prev_code = ""

    # Pad with zeros or truncate to 4 characters
    result = (result + "000")[:4]

    return result


def metaphone(word: str) -> str:
    """Generate Metaphone code for a word.

    Metaphone is a more advanced phonetic algorithm than Soundex,
    handling English pronunciation rules better.

    Args:
        word: Word to encode

    Returns:
        Metaphone code string
    """
    if not word:
        return ""

    word = word.upper()

    # Remove non-alphabetic characters
    word = "".join(c for c in word if c.isalpha())

    if not word:
        return ""

    # Vowels for reference
    vowels = "AEIOU"

    result = []
    i = 0
    length = len(word)

    # Handle special starting patterns
    if word.startswith(("KN", "GN", "PN", "AE", "WR")):
        word = word[1:]
        length -= 1
    elif word.startswith("WH"):
        word = "W" + word[2:]
        length -= 1
    elif word.startswith("X"):
        word = "S" + word[1:]

    while i < length:
        char = word[i]
        next_char = word[i + 1] if i + 1 < length else ""
        next_next = word[i + 2] if i + 2 < length else ""

        # Skip duplicate adjacent letters (except C)
        if char == next_char and char != "C":
            i += 1
            continue

        if char in vowels:
            # Only add vowel if it's at the start
            if i == 0:
                result.append(char)
            i += 1

        elif char == "B":
            # Drop B if after M at end of word
            if not (i == length - 1 and i > 0 and word[i - 1] == "M"):
                result.append("B")
            i += 1

        elif char == "C":
            if next_char == "H":
                result.append("X")
                i += 2
            elif next_char in "IEY":
                result.append("S")
                i += 1
            else:
                result.append("K")
                i += 1

        elif char == "D":
            if next_char == "G" and next_next in "IEY":
                result.append("J")
                i += 2
            else:
                result.append("T")
                i += 1

        elif char == "F":
            result.append("F")
            i += 1

        elif char == "G":
            if next_char == "H":
                if i + 2 < length and word[i + 2] not in vowels:
                    i += 2
                    continue
                result.append("F")
                i += 2
            elif next_char == "N":
                i += 1
                continue
            elif next_char in "IEY":
                result.append("J")
                i += 1
            else:
                result.append("K")
                i += 1

        elif char == "H":
            # H is silent unless preceded/followed by vowel
            if i > 0 and word[i - 1] not in vowels:
                i += 1
                continue
            if next_char in vowels:
                result.append("H")
            i += 1

        elif char == "J":
            result.append("J")
            i += 1

        elif char == "K":
            # Drop K if after C
            if i > 0 and word[i - 1] == "C":
                i += 1
                continue
            result.append("K")
            i += 1

        elif char == "L":
            result.append("L")
            i += 1

        elif char == "M":
            result.append("M")
            i += 1

        elif char == "N":
            result.append("N")
            i += 1

        elif char == "P":
            if next_char == "H":
                result.append("F")
                i += 2
            else:
                result.append("P")
                i += 1

        elif char == "Q":
            result.append("K")
            i += 1

        elif char == "R":
            result.append("R")
            i += 1

        elif char == "S":
            if next_char == "H":
                result.append("X")
                i += 2
            elif next_char == "I" and next_next in "OA":
                result.append("X")
                i += 1
            else:
                result.append("S")
                i += 1

        elif char == "T":
            if next_char == "H":
                result.append("0")  # Use 0 for TH sound
                i += 2
            elif next_char == "I" and next_next in "OA":
                result.append("X")
                i += 1
            else:
                result.append("T")
                i += 1

        elif char == "V":
            result.append("F")
            i += 1

        elif char == "W":
            if next_char in vowels:
                result.append("W")
            i += 1

        elif char == "X":
            result.append("KS")
            i += 1

        elif char == "Y":
            if next_char in vowels:
                result.append("Y")
            i += 1

        elif char == "Z":
            result.append("S")
            i += 1

        else:
            i += 1

    return "".join(result)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein (edit) distance between two strings.

    The Levenshtein distance is the minimum number of single-character
    edits (insertions, deletions, substitutions) required to change
    one string into the other.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Number of edits needed
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    # Use two rows for space optimization
    previous_row = list(range(len(s2) + 1))
    current_row = [0] * (len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row[0] = i + 1

        for j, c2 in enumerate(s2):
            # Cost of insertion, deletion, substitution
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)

            current_row[j + 1] = min(insertions, deletions, substitutions)

        previous_row, current_row = current_row, previous_row

    return previous_row[len(s2)]


def phonetic_similarity(word1: str, word2: str) -> float:
    """Calculate phonetic similarity between two words.

    Uses both Soundex and Metaphone codes, plus Levenshtein distance
    on the phonetic codes to produce a similarity score.

    Args:
        word1: First word
        word2: Second word

    Returns:
        Similarity score from 0.0 (different) to 1.0 (identical)
    """
    if not word1 or not word2:
        return 0.0

    word1_lower = word1.lower()
    word2_lower = word2.lower()

    # Exact match
    if word1_lower == word2_lower:
        return 1.0

    # Soundex comparison
    soundex1 = soundex(word1)
    soundex2 = soundex(word2)
    soundex_match = 1.0 if soundex1 == soundex2 else 0.0

    # Metaphone comparison
    meta1 = metaphone(word1)
    meta2 = metaphone(word2)

    if meta1 and meta2:
        meta_distance = levenshtein_distance(meta1, meta2)
        max_meta_len = max(len(meta1), len(meta2))
        meta_similarity = 1.0 - (meta_distance / max_meta_len) if max_meta_len > 0 else 0.0
    else:
        meta_similarity = 0.0

    # Raw Levenshtein on original words
    raw_distance = levenshtein_distance(word1_lower, word2_lower)
    max_len = max(len(word1_lower), len(word2_lower))
    raw_similarity = 1.0 - (raw_distance / max_len) if max_len > 0 else 0.0

    # Weighted combination
    # Metaphone is most important (handles pronunciation)
    # Soundex provides coarse grouping
    # Raw similarity catches minor spelling variations
    score = (meta_similarity * 0.5) + (soundex_match * 0.3) + (raw_similarity * 0.2)

    return min(1.0, max(0.0, score))


def phrase_phonetic_similarity(phrase1: str, phrase2: str) -> float:
    """Calculate phonetic similarity between two phrases.

    Compares phrases word by word and averages the similarity scores.

    Args:
        phrase1: First phrase
        phrase2: Second phrase

    Returns:
        Average similarity score from 0.0 to 1.0
    """
    if not phrase1 or not phrase2:
        return 0.0

    words1 = phrase1.lower().split()
    words2 = phrase2.lower().split()

    # Exact match
    if words1 == words2:
        return 1.0

    # If different number of words, lower the score
    len_diff = abs(len(words1) - len(words2))
    len_penalty = 1.0 - (len_diff * 0.1)  # 10% penalty per word difference
    len_penalty = max(0.5, len_penalty)  # Don't go below 50%

    # Match words greedily
    if len(words1) <= len(words2):
        shorter, longer = words1, words2
    else:
        shorter, longer = words2, words1

    total_similarity = 0.0
    used_indices = set()

    for word1 in shorter:
        best_score = 0.0
        best_idx = -1

        for idx, word2 in enumerate(longer):
            if idx in used_indices:
                continue
            score = phonetic_similarity(word1, word2)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0:
            used_indices.add(best_idx)
            total_similarity += best_score

    # Average over the shorter phrase length
    avg_similarity = total_similarity / len(shorter) if shorter else 0.0

    return avg_similarity * len_penalty
