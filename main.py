import os
import httpx
import numpy as np

WCL_EMBEDDING_URL = "http://ai.thewcl.com:6502/embedding_vector"
WCL_API_KEY = "stu-Mahdi-cea6af21c6be379adb17d8170a188934"

whatmode = input("Do you want to compare sentences or paraphrased sentences? (c/p): ").lower()

def get_embedding(text: str) -> list[float]:
    headers = {
        "Authorization": f"Bearer {WCL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {"text": text}
    response = httpx.post(WCL_EMBEDDING_URL, headers=headers, json=payload)

    response.raise_for_status()

    return response.json()["embedding"]

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def user_input_paraphrase_mode():
    n = int(input("How many paraphrased sentence pairs would you like to compare? "))

    sentence_pairs = []
    for i in range(n):
        print(f"\nPair {i+1}:")
        s1 = input("  Original sentence: ").title()
        s2 = input("  Paraphrased sentence: ").title()
        sentence_pairs.append((s1, s2))

    print("\nFetching embeddings...\n")
    unique_sentences = set(s for pair in sentence_pairs for s in pair)
    embeddings = {s: get_embedding(s) for s in unique_sentences}

    results = []
    for s1, s2 in sentence_pairs:
        sim = cosine_similarity(embeddings[s1], embeddings[s2])
        results.append((sim, s1, s2))

    print("Paraphrased Sentence Similarities:\n")
    for score, s1, s2 in results:
        print(f"Similarity: {score:.4f}")
        print(f" • \"{s1}\"")
        print(f" • \"{s2}\"\n")

def main():
    prompt1 = input("Enter first sentence: ").title()
    prompt2 = input("Enter second sentence: ").title()
    prompt3 = input("Enter third sentence: ").title()
    prompt4 = input("Enter fourth sentence: ").title()
    
    embedding1 = get_embedding(prompt1)
    embedding2 = get_embedding(prompt2)
    embedding3 = get_embedding(prompt3)
    embedding4 = get_embedding(prompt4)
    
    similarity = cosine_similarity(embedding1, embedding2)
    similarity2 = cosine_similarity(embedding1, embedding3)
    similarity3 = cosine_similarity(embedding1, embedding4)
    similarity4 = cosine_similarity(embedding2, embedding3)
    similarity5 = cosine_similarity(embedding2, embedding4)
    similarity6 = cosine_similarity(embedding3, embedding4)

    if similarity > similarity2:
        print(f"The sentences with the closest similarity are: {prompt1} and {prompt2}")
        print(f"Cosine Similarity Score: {similarity:.4f}")
    elif similarity > similarity3:
        print(f"The sentences with the closest similarity are: {prompt1} and {prompt3}")
        print(f"Cosine Similarity Score: {similarity2:.4f}")
    elif similarity > similarity4:
        print(f"The sentences with the closest similarity are: {prompt1} and {prompt4}")
        print(f"Cosine Similarity Score: {similarity3:.4f}")
    elif similarity > similarity5:
        print(f"The sentences with the closest similarity are: {prompt2} and {prompt3}")
        print(f"Cosine Similarity Score: {similarity4:.4f}")
    elif similarity > similarity6:
        print(f"The sentences with the closest similarity are: {prompt2} and {prompt4}")
        print(f"Cosine Similarity Score: {similarity5:.4f}")
    else:
        print(f"The sentences with the closest similarity are: {prompt3} and {prompt4}")
        print(f"Cosine Similarity Score: {similarity6:.4f}")
    
if __name__ == "__main__":
    if whatmode == "c":
        main()
    elif whatmode == "p":
        user_input_paraphrase_mode()
    else:
        print("Invalid mode. Please enter 'c' for compare or 'p' for paraphrase.")
 