import gensim
from gensim.models import Word2Vec

# Sample training data
sentences = [
    ["I", "love", "natural", "language", "processing"],
    ["Word2Vec", "is", "a", "target", "tool"],
    ["Machine", "learning", "is", "fun"]
]

# Train Skip-gram model (sg=1)
skipgram_model = Word2Vec(sentences, vector_size=100, window=2, min_count=1, sg=1)

# Train CBOW model (sg=0)
cbow_model = Word2Vec(sentences, vector_size=100, window=2, min_count=1, sg=0)

# Choose a word for vector and similarity checks
word = "Machine"

# Get word vectors
cbow_vector = cbow_model.wv[word]
skipgram_vector = skipgram_model.wv[word]

# Print word vectors
print(f"CBOW Vector for '{word}':\n", cbow_vector)
print(f"\nSkip-gram Vector for '{word}':\n", skipgram_vector)

# Find similar words
cbow_similar = cbow_model.wv.most_similar(word, topn=2)
skipgram_similar = skipgram_model.wv.most_similar(word, topn=3)

print(f"\nWords similar to '{word}' in CBOW model:", cbow_similar)
print(f"Words similar to '{word}' in Skip-gram model:", skipgram_similar)

# Example: Vector for another word and similarity check
print("\nVector for 'language':", skipgram_model.wv['language'])
print("Most similar to 'processing':", skipgram_model.wv.most_similar('processing', topn=1))
print("Most similar to 'fun':", skipgram_model.wv.most_similar('fun', topn=1))
