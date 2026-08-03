from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        context_blocks = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source = metadata.get("source_url") or metadata.get("source") or result.get("id")
            context_blocks.append(
                f"[Đoạn {index} | nguồn: {source}]\n{result['content']}"
            )
        context = "\n\n".join(context_blocks) or "Không tìm thấy ngữ cảnh phù hợp."
        prompt = (
            "Bạn là trợ lý hỏi đáp dựa trên cơ sở tri thức. Chỉ trả lời bằng "
            "thông tin trong NGỮ CẢNH. Nếu ngữ cảnh không đủ, hãy nói rõ rằng "
            "không đủ thông tin; không tự suy đoán.\n\n"
            f"NGỮ CẢNH:\n{context}\n\n"
            f"CÂU HỎI: {question}\n\n"
            "TRẢ LỜI (ngắn gọn và nêu nguồn/đoạn đã dùng khi có thể):"
        )
        return self.llm_fn(prompt)
