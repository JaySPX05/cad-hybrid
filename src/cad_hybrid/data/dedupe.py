"""Find near-duplicate ECG images.

Small ECG image datasets often contain the same ECG photographed or scanned more than
once. If copies land in both train and test sets, accuracy looks better than it is.
We group near-duplicates so the splitter keeps each group on one side.
"""

from __future__ import annotations

from collections.abc import Sequence


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two integer hashes."""
    return bin(a ^ b).count("1")


def group_near_duplicates(hashes: Sequence[int], max_distance: int = 4) -> list[int]:
    """Assign a group id to each hash; hashes within ``max_distance`` bits share a group.

    Uses union-find over all pairs (fine for a few thousand images).
    """
    parent = list(range(len(hashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hamming(hashes[i], hashes[j]) <= max_distance:
                parent[find(j)] = find(i)

    roots = [find(i) for i in range(len(hashes))]
    relabel = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return [relabel[r] for r in roots]


def image_phash(path: str) -> int:
    """Perceptual hash of an image as an integer (requires ``imagehash`` and ``Pillow``)."""
    import imagehash
    from PIL import Image

    with Image.open(path) as img:
        return int(str(imagehash.phash(img)), 16)
