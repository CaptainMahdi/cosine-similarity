import os
import httpx
import numpy as np

WCL_EMBEDDING_URL = "http://ai.thewcl.com:6502/embedding_vector"
WCL_API_KEY = "stu-Mahdi-cea6af21c6be379adb17d8170a188934"

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

def main():
    prompt1 = input("Enter first sentence: ")
    prompt2 = input("Enter second sentence: ")
    embedding1 = get_embedding(prompt1)
    embedding2 = get_embedding(prompt2)
    similarity = cosine_similarity(embedding1, embedding2)

    print(f"Cosine Similarity Score: {similarity:.4f}")
    
if __name__ == "__main__":
    main()
 