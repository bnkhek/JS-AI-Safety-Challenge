import re
from collections import Counter
from typing import List, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

BOILERPLATE_PATTERNS = [r"<\|im_start\|>\s*\w+\s*",  
    r"<\|im_end\|>",              
    r"<\|endoftext\|>",
    r"<\|pad\|>",
    r"<s>|</s>",
    r"\[INST\]|\[/INST\]",
    r"<<SYS>>|<</SYS>>",
    r"^(System|User|Assistant)\s*:\s*", 
    r"^You are a helpful assistant\.\s*",]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE | re.MULTILINE)


def clean_and_deduplicate_outputs(texts: List[str]) -> List[str]:
    """
    Strip boilerplate text (chat-template markers, system prompts, special
    tokens etc) from each leaked output, then deduplicate.
    """
    seen = set()
    cleaned = []
    for text in texts:
        stripped = BOILERPLATE_RE.sub("", text).strip()
        # Collapse multiple consecutive whitespace / newlines
        stripped = re.sub(r"\s+", " ", stripped)
        if stripped and stripped not in seen:
            seen.add(stripped)
            cleaned.append(stripped)
    return cleaned


def build_tfidf_matrix(unique_strings: List[str],
    ngram_range: Tuple[int, int] = (4, 6),) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Compute a TF-IDF matrix where each row corresponds to one unique string u
    and each column corresponds to a character n-gram.
    """
    vectorizer = TfidfVectorizer(ngram_range=ngram_range,)
    matrix = vectorizer.fit_transform(unique_strings).toarray().astype(np.float32)
    return matrix, vectorizer


def cluster_strings(matrix: np.ndarray, eps: float = 0.3, min_samples: int = 2) -> np.ndarray:
    """
    Cluster the TF-IDF vectors using DBSCAN with cosine distance.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
    return db.fit_predict(matrix)


def get_largest_cluster(labels: np.ndarray, 
    strings: List[str]) -> List[str]:
    """Return all strings found in the largest cluster.
    """
    # Find the largest cluster
    # DBSCAN outputs non negative integers for cluster membership
    # -1 corresponds to a noisy data point
    # So ignore any label that is negative
    label_counts = Counter([label for label in labels if label >= 0])
    largest_label = label_counts.most_common()[0][0]
    largest_cluster_members = []
    for string, label in zip(strings, labels):
        if label == largest_label:
            largest_cluster_members.append(string)
    return largest_cluster_members


def extract_frequent_ngrams(unique_strings: List[str], labels: np.ndarray,
    ngram_sizes: Tuple[int, ...] = (4, 5, 6), p:float = 1 / 3,
) -> List[str]:
    """
    From the largest DBSCAN cluster, collect every character n-gram
    that appears in at least p% of the cluster's member 
    strings (paper: p = 33%).
    """
    largest_cluster_members = get_largest_cluster(labels, unique_strings)
    n_members       = len(largest_cluster_members)
    threshold       = p * n_members

    # Count how many cluster members contain each character n-gram
    ngram_counts = Counter()
    seen_in_cluster = set({})
    for member in largest_cluster_members:
        for n in ngram_sizes:
            for i in range(len(member) - n + 1):
                gram = member[i : i + n]
                ngram_counts[gram] += 1
                if gram not in seen_in_cluster:
                    seen_in_cluster.add(gram)

    frequent_ngrams = []
    for n_gram, counts in ngram_counts.items():
        if counts >= threshold:
            frequent_ngrams.append(n_gram)
    return frequent_ngrams


def stitch_ngrams(
    ngrams: List[str],
    max_passes: int = 20,
    min_overlap: int = 3) -> List[str]:
    """
    Iteratively merge pairs of strings that share an overlapping suffix/prefix,
    building up longer motif strings.
    Stops when a full pass over all strings produces no new merges or when it hits a max pass limit.
    As such it's technically not exhaustive as I couldn't quite figure it out that way.
    Strings that are substrings of longer retained strings are then dropped.
    """
    # Note to self, can probably optimize with a lookup table
    # But this function shouldn't be the bottleneck anytime soon?
    if min_overlap < 3:
        print("WARNING: Setting min_overlap below 3 is likely to lead to spurious results.")

    seqs = list(dict.fromkeys(ngrams)) 
    for _ in range(max_passes):
        changed = False
        i = 0
        while i < len(seqs):
            best_j = -1
            best_result = seqs[i]
            for j in range(len(seqs)):
                if j == i:
                    continue
                a, b = seqs[i], seqs[j]
                max_overlap = min(len(a), len(b)) - 1
                for k in range(max_overlap, min_overlap - 1, -1):
                    if a[-k:] == b[:k]:
                        candidate = a + b[k:]
                        if len(candidate) > len(best_result):
                            best_result = candidate
                            best_j = j
                        break  # longest overlap for this (i, j) pair found

            if best_j != -1:
                # Replace seqs[i] with the merged string, remove seqs[best_j].
                # Don't advance i so the merged string can merge further.
                seqs[i] = best_result
                seqs.pop(best_j)
                changed = True
            else:
                i += 1

        if not changed:
            break

    # Drop any string that is a substring of a longer retained string,
    # then deduplicate exact copies.
    result = [s for s in seqs if not any(s != t and s in t for t in seqs)]
    return sorted(dict.fromkeys(result), key=len, reverse=True)


def discover_motifs(
    leaked_outputs: List[str],
    ngram_sizes: Tuple[int, ...] = (4, 5, 6),
    dbscan_eps: float = 0.3,
    dbscan_min_samples: int = 2,
    presence_threshold: float = 0.33,
    min_motif_length: int = 6,
    common_substring_min_length: int = 20,
    common_substring_threshold: float = 0.75,
    top_m: int = 50) -> List[str]:
    """
    Perform the motif discovery step in the paper.
    The common substring parameters are here if I implement them in the future.
    I have no idea what they actually do based on what the paper says. It's very vague...
    """
    print(f"Motif discovery - {len(leaked_outputs)} raw outputs.")

    unique_strings = clean_and_deduplicate_outputs(leaked_outputs)
    print(f"After cleaning and deduplication: {len(unique_strings)} unique strings.")
    if len(unique_strings) < dbscan_min_samples:
        print("Too few unique strings to cluster.")
        return []

    matrix, _ = build_tfidf_matrix(unique_strings, ngram_range=(min(ngram_sizes), max(ngram_sizes)))
    print(f"TF-IDF matrix: {matrix.shape[0]} strings x {matrix.shape[1]} features")
    labels = cluster_strings(matrix, eps=dbscan_eps, min_samples=dbscan_min_samples)
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"DBSCAN produced {n_clusters} clusters, {n_noise} noise points")
    if n_clusters == 0:
        print("No clusters found - try increasing eps or reducing min_samples.")
        return []

    frequent = extract_frequent_ngrams(
        unique_strings, labels, ngram_sizes, p=presence_threshold)
    if not frequent:
        return []

    stitched = stitch_ngrams(frequent)
    if not stitched:
        return []
    motifs = [m for m in stitched if len(m) >= min_motif_length]
    print(f"{len(motifs)} motifs after stitching and length filter.")
    print(f"(min_len = {min_motif_length})")
    return motifs