import random
from collections import Counter

import nltk
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from nltk.corpus import reuters, gutenberg

nltk.download("reuters")
nltk.download("gutenberg")
nltk.download("averaged_perceptron_tagger")


def plot_word_counts(
    words: list[str],
    count1: np.ndarray,
    count2: np.ndarray,
    label1: str,
    label2: str,
    title: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Utility function to plot word counts for comparison."""
    width_px = 1920
    height_px = 1080

    dpi = 120

    fig_width = width_px / dpi
    fig_height = height_px / dpi
    figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    x = np.arange(len(words))  # the label locations

    ax.bar(
        x,
        count1,
        label=label1,
        color="turquoise",
    )
    ax.bar(
        x,
        count2,
        label=label2,
        color="orange",
        alpha=0.6,
    )

    ax.set_xlabel("Verbs", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=90)
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)

    fig.tight_layout()

    return fig, ax


def get_verbs(words):
    """Filter verbs from a list of words using POS tagging."""

    def is_verb(pos):
        return pos.startswith("VB")

    tagged_words = nltk.pos_tag(words)
    verbs = [word for word, pos in tagged_words if is_verb(pos) and len(word) > 1]
    return verbs


def sample_words(words, num_samples):
    """Sample a fixed number of words."""
    return random.sample(words, min(num_samples, len(words)))


def analyze_text_corpus(corpus1, corpus2, top_n=50, num_samples=10000):
    """Analyze verb distributions in two text corpora and return top N common verbs."""
    words1 = [word.lower() for word in corpus1.words() if word.isalpha()]
    words2 = [word.lower() for word in corpus2 if word.isalpha()]

    # Sample words to ensure equal number of tokens
    sampled_words1 = sample_words(words1, num_samples)
    sampled_words2 = sample_words(words2, num_samples)

    verbs1 = get_verbs(sampled_words1)
    verbs2 = get_verbs(sampled_words2)

    counter1 = Counter(verbs1)
    counter2 = Counter(verbs2)
    common_verbs = set(counter1.keys()).intersection(set(counter2.keys()))

    common_counts1 = {verb: counter1[verb] for verb in common_verbs}
    common_counts2 = {verb: counter2[verb] for verb in common_verbs}
    # Get top N common verbs
    top_common_verbs = sorted(
        common_counts1.keys(),
        key=lambda x: common_counts1[x] + common_counts2[x],
        reverse=True,
    )[:top_n]
    count1 = [common_counts1[verb] for verb in top_common_verbs]
    count2 = [common_counts2[verb] for verb in top_common_verbs]

    return count1, count2, top_common_verbs


def main():
    # Analyze Text Data
    text_corpus1 = reuters
    text_corpus2_files = [
        "austen-emma.txt",
        "austen-persuasion.txt",
        "austen-sense.txt",
        "chesterton-ball.txt",
        "chesterton-brown.txt",
        "chesterton-thursday.txt",
        "edgeworth-parents.txt",
        "melville-moby_dick.txt",
        "shakespeare-caesar.txt",
        "shakespeare-hamlet.txt",
        "shakespeare-macbeth.txt",
    ]

    text_corpus2 = [word for fileid in text_corpus2_files for word in gutenberg.words(fileid)]

    text_data1, text_data2, common_verbs = analyze_text_corpus(
        text_corpus1,
        text_corpus2,
        top_n=50,
        num_samples=10000,
    )

    # Plot verb distribution comparison
    fig, ax = plot_word_counts(
        words=common_verbs,
        count1=text_data1,
        count2=text_data2,
        label1="News stories (Formal)",
        label2="Narrative Texts (Informal)",
        title="Top 50 Common Verb Count in Formal and Informal Texts",
    )
    st.pyplot(fig)


if __name__ == "__main__":
    main()
