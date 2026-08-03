# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Diệp Đức Lai

**Mã sinh viên:** 2A202601784

**Nhóm:** Nguyễn Minh Công — Nguyễn Văn Sang — Diệp Đức Lai
**Ngày:** 03/08/2026

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Bài tập 1.1)

**Độ tương tự cosine cao nghĩa là gì?**

Hai vector embedding có cosine similarity cao khi chúng hướng gần giống nhau, tức hai đoạn văn bản thường biểu đạt chủ đề hoặc ý nghĩa gần nhau dù không nhất thiết dùng cùng từ ngữ.

**Ví dụ có độ tương tự cao:**

- Câu A: “Khách hàng được gửi yêu cầu hoàn trả trong vòng 15 ngày.”
- Câu B: “Người mua có thể yêu cầu trả hàng trong thời hạn mười lăm ngày.”
- Lý do: hai câu cùng nói về chủ thể mua hàng, hành động hoàn trả và cùng một thời hạn.

**Ví dụ có độ tương tự thấp:**

- Câu A: “TikTok Shop thu phí hoa hồng từ người bán.”
- Câu B: “Dự báo thời tiết Hà Nội ngày mai có mưa.”
- Lý do: hai câu thuộc hai miền nghĩa hoàn toàn khác nhau: chính sách thương mại điện tử và thời tiết.

**Tại sao ưu tiên cosine hơn Euclidean distance?**

Cosine tập trung vào hướng của vector — mô hình ý nghĩa tương đối — và ít bị ảnh hưởng bởi độ lớn vector. Euclidean distance phụ thuộc trực tiếp vào độ lớn nên hai embedding cùng hướng nhưng có norm khác nhau vẫn có thể bị coi là xa; điều này ít phù hợp hơn khi so sánh ngữ nghĩa văn bản.

### Bài toán Chunking (Bài tập 1.2)

Với `length=10.000`, `chunk_size=500`, `overlap=50`:

```text
step = 500 - 50 = 450
chunk_count = ceil((10.000 - 50) / 450)
            = ceil(9.950 / 450)
            = ceil(22,111...)
            = 23 chunks
```

Khi tăng overlap lên 100:

```text
step = 500 - 100 = 400
chunk_count = ceil((10.000 - 100) / 400)
            = ceil(9.900 / 400)
            = ceil(24,75)
            = 25 chunks
```

Số chunk tăng từ 23 lên 25 vì mỗi cửa sổ chỉ tiến thêm 400 thay vì 450 ký tự. Overlap lớn hơn giúp giữ ngữ cảnh nằm ở ranh giới hai chunk, nhưng làm tăng số vector, thời gian embedding và chi phí lưu trữ/tìm kiếm.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ

**`SentenceChunker.chunk`**

Tôi dùng regex `(?<=[.!?])(?:[ \t]+|\n+)` để tách sau dấu kết thúc câu nhưng vẫn giữ dấu câu trong nội dung. Các câu được `strip`, bỏ phần rỗng và ghép tuần tự theo `max_sentences_per_chunk`; văn bản rỗng trả về danh sách rỗng.

**`RecursiveChunker.chunk` / `_split`**

Thuật toán thử lần lượt `\n\n`, `\n`, `. `, khoảng trắng rồi ký tự. Mỗi mức giữ lại separator, tham lam ghép các đơn vị nếu chưa quá `chunk_size`; phần quá dài được đưa xuống separator kế tiếp. Base case là đoạn đã đủ ngắn hoặc hết separator, khi đó cắt cứng theo kích thước để luôn kết thúc.

**`compute_similarity` và comparator**

Cosine được tính bằng dot product chia cho tích hai norm L2 và trả `0.0` nếu có vector zero. Comparator chạy đủ Fixed/Sentence/Recursive, trả số chunk, độ dài trung bình và nội dung chunk để có thể kiểm tra cả định lượng lẫn tính mạch lạc.

### Lớp `EmbeddingStore`

**`add_documents` + `search`**

Mỗi `Document` được chuẩn hóa thành record gồm ID, content, metadata, embedding và storage ID không trùng. Store luôn giữ bản in-memory để hành vi nhất quán; nếu ChromaDB có sẵn thì đồng thời mirror dữ liệu sang collection. Search embedding câu hỏi một lần, tính dot product với từng record, sắp giảm dần và cắt `top_k`.

**`search_with_filter` + `delete_document`**

Metadata được chuẩn hóa về kiểu scalar; filter được áp dụng **trước** similarity search để giảm nhiễu. Xóa tìm tất cả record có `metadata['doc_id']` trùng ID tài liệu, xóa toàn bộ chunk tương ứng ở memory và Chroma, trả `True` chỉ khi thực sự có dữ liệu bị xóa.

### `KnowledgeBaseAgent.answer`

Agent truy xuất top-k chunk, đánh số từng đoạn và đưa `source_url` vào context. Prompt yêu cầu chỉ dùng thông tin trong context, nói rõ khi thiếu dữ liệu và không suy đoán; sau đó truyền prompt cho `llm_fn` được inject. Thiết kế này tách retrieval khỏi model sinh, dễ unit test và thay backend LLM.

### Chiến lược cá nhân cho corpus TikTok Shop

Tôi chọn `RecursiveChunker(chunk_size=700)` vì corpus chính sách có nhiều đoạn, dòng, câu và bảng với độ dài khác nhau. Strategy ưu tiên tách theo đoạn (`\n\n`), dòng (`\n`), câu (`. `), từ rồi mới đến ký tự, nhờ đó tránh cắt cứng giữa điều kiện và ngoại lệ tốt hơn fixed-size. Đánh đổi là chunk đứng riêng có thể mất heading cha, nhưng trên benchmark khóa của nhóm cấu hình này tạo 140 chunk và đưa đủ năm gold evidence lên rank 1.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

### Kết quả kiểm thử

Lệnh chạy:

```bash
python3 -m pytest tests/ -v
```

Kết quả:

```text
collected 42 items

TestProjectStructure                         2 passed
TestClassBasedInterfaces                    2 passed
TestFixedSizeChunker                        7 passed
TestSentenceChunker                         4 passed
TestRecursiveChunker                        4 passed
TestEmbeddingStore                          8 passed
TestKnowledgeBaseAgent                      2 passed
TestComputeSimilarity                       4 passed
TestCompareChunkingStrategies               3 passed
TestEmbeddingStoreSearchWithFilter          3 passed
TestEmbeddingStoreDeleteDocument            3 passed

============================== 42 passed ==============================
```

**Số lượng bài test vượt qua:** **42 / 42**

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Backend thực nghiệm: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, vector 384 chiều, normalize embedding.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|---:|---|---|---|---:|:---:|
| 1 | Người mua có thể yêu cầu trả hàng trong vòng 15 ngày. | Khách hàng được gửi yêu cầu hoàn trả trong thời hạn mười lăm ngày. | Cao | 0,8153 | Có |
| 2 | Người bán không được điều hướng khách hàng ra ngoài TikTok Shop. | Nhà bán hàng bị cấm gửi liên kết để người mua giao dịch ngoài nền tảng. | Cao | 0,6871 | Có |
| 3 | COD là hình thức thanh toán khi nhận hàng. | Khách hàng trả tiền cho đơn hàng vào lúc bưu kiện được giao. | Cao | 0,5384 | Có, nhưng thấp hơn dự kiến |
| 4 | TikTok Shop thu phí hoa hồng từ người bán. | Dự báo thời tiết Hà Nội ngày mai có mưa. | Thấp | -0,0082 | Có |
| 5 | Điểm AHR do vi phạm được tự động xóa sau 90 ngày. | Người mua gửi trả sản phẩm bị lỗi cho đơn vị vận chuyển. | Thấp | 0,2528 | Có |

**Kết quả bất ngờ nhất:** cặp 3 chỉ đạt 0,5384 dù “COD” chính là “thanh toán khi nhận hàng”. Nguyên nhân có thể là một câu dùng từ viết tắt và định nghĩa khái niệm, câu kia mô tả hành động thực tế; embedding nhận ra quan hệ nhưng không coi chúng gần như hai câu paraphrase trực tiếp. Điều này cho thấy nên viết query có cả thuật ngữ và cách diễn đạt tự nhiên.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

### Cấu hình

- Corpus chung: 8 chính sách TikTok Shop Việt Nam trong `data/tiktok_shop_policies/`.
- Benchmark chung: `benchmarks/tiktok_shop_team_v1.json`, đúng 5 query và gold answer đã khóa.
- Embedder chính thức của bảng so sánh nhóm: OpenAI `text-embedding-3-small`, không dùng mock.
- Chiến lược cá nhân: `RecursiveChunker(chunk_size=700)`.
- Tổng số chunk: 140; `top_k=3`.
- Metadata filter: Q1 dùng `customer_role=buyer`; Q3–Q5 dùng `customer_role=seller`; Q2 không filter.
- Tôi chạy kiểm chứng độc lập bằng local multilingual embedder trên đúng corpus/query/strategy; kết quả cũng đạt Hit@3 100%, MRR 1,00 và được lưu tại `results/lai-recursive-700-local-validation.json`. Điểm số cột dưới là của lần local validation, còn rank chính thức lấy từ bảng so sánh nhóm dùng OpenAI.

| # | Câu hỏi chung của nhóm | Top-1 gold evidence với Recursive 700 | Rank nhóm | Score local | Câu trả lời Agent được kiểm chứng từ context |
|---:|---|---|---:|---:|---|
| 1 | Người mua được gửi yêu cầu trả hàng hoặc hoàn tiền trong bao lâu sau khi đơn được giao? | `tiktok-buyer-return-eligibility#0`: chứa cả mốc thời gian và trạng thái giao hàng | 1 | 0,606454 | Trong vòng mười lăm (15) ngày dương lịch sau khi trạng thái đơn được cập nhật thành “Đã giao hàng”. |
| 2 | Điều gì xảy ra nếu khách hàng không nhận đơn COD ba lần trong khoảng thời gian 60 ngày? | `tiktok-cod-policy#6`: điều kiện tạm tắt COD | 1 | 0,757578 | TikTok Shop có quyền tạm tắt COD trong sáu mươi (60) ngày. |
| 3 | Quy trình và thời hạn để người bán kháng nghị một hành động thực thi như thế nào? | `tiktok-seller-performance-policy#22`: mục quy trình kháng nghị | 1 | 0,754937 | Tối đa hai lần: lần đầu trong 30 ngày từ thông báo; lần hai trong 15 ngày từ khi lần đầu bị từ chối. |
| 4 | Muốn đăng bán mỹ phẩm thuộc nhóm hạn chế thì cần chuẩn bị và thể hiện thông tin, tài liệu gì? | `tiktok-restricted-products#1`: thông báo mỹ phẩm và nhãn bao bì | 1 | 0,757284 | Cần thông báo sản phẩm mỹ phẩm và ảnh nhãn có tên, chức năng, hướng dẫn, thành phần, xuất xứ, khối lượng, hạn dùng và cảnh báo. |
| 5 | Điểm AHR bị trừ tự động xóa sau bao lâu, và ngoại lệ nào khiến quy tắc không áp dụng? | `tiktok-seller-performance-policy#3`: quy tắc xóa điểm AHR | 1 | 0,732360 | Sau mỗi 90 ngày, trừ khi cửa hàng đã bị nền tảng đóng, tức AHR bằng 0. |

**Bao nhiêu câu có chunk liên quan trong top-3?** **5 / 5**.

**Kết quả chính thức trong bảng nhóm:** Hit@3 **100%**, MRR **1,00**, tương đương **10/10** retrieval — cả năm gold chunk ở rank 1. Rank/evidence lấy từ báo cáo nhóm; similarity score được ghi riêng từ lần kiểm chứng local để không tự dựng điểm OpenAI bị thiếu khỏi artifact gốc.

### So sánh với baseline

| Thành viên | Chiến lược | Số chunk | Hit@3 | MRR |
|---|---|---:|---:|---:|
| Nguyễn Minh Công | Heading-aware 700 | 187 | 80% | 0,70 |
| Nguyễn Văn Sang | Fixed-size 700 | 129 | 80% | 0,67 |
| **Diệp Đức Lai** | **Recursive 700** | **140** | **100%** | **1,00** |

### Failure analysis và bài học

Câu 4 là failure case rõ nhất của bảng nhóm: Heading-aware lấy đúng tài liệu sản phẩm hạn chế nhưng top-1 là yêu cầu cho thực phẩm/đồ uống, còn gold chunk mỹ phẩm vắng khỏi top-3. Filter `customer_role=seller` chỉ giới hạn đối tượng chứ chưa phân biệt ngành hàng vì schema chưa có `product_category`. Recursive 700 giữ hai required terms của phần mỹ phẩm trong cùng chunk và đưa chunk đó lên rank 1. Cải thiện tiếp theo là thêm `product_category=cosmetics`, dùng hybrid BM25 + vector hoặc reranker.

Điều quan trọng nhất tôi học được là strategy phức tạp hơn không mặc nhiên tốt hơn. Heading-aware giữ breadcrumb rất rõ nhưng tạo thêm 47 chunk so với Recursive và vẫn miss Q4; Recursive cân bằng tốt hơn giữa mạch lạc và kích thước trên năm query hiện tại. Tuy vậy benchmark còn nhỏ, vì thế kết luận chỉ áp dụng cho corpus này và cần được kiểm tra lại với nhiều query hơn.

---

## Tự đánh giá

| Tiêu chí | Điểm tự đánh giá |
|---|---:|
| Khởi động | 5 / 5 |
| Hướng tiếp cận | 10 / 10 |
| Hoàn thiện code | 30 / 30 |
| Dự đoán độ tương tự | 5 / 5 |
| Kết quả truy xuất | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
