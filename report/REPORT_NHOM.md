# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** Nguyễn Minh Công — Nguyễn Văn Sang — Diệp Đức Lai
**Thành viên:** Nguyễn Minh Công (2A202601945); Nguyễn Văn Sang
(2A202601252); Diệp Đức Lai (2A202601784)
**Ngày:** 03/08/2026

> Corpus, metadata schema, embedding model và 5 benchmark queries được cố định
> để bảo đảm so sánh công bằng. Ba strategy được chạy lại trong cùng môi trường
> benchmark của nhóm; chỉ biến chunking thay đổi giữa các thành viên.

## 1. Lựa chọn tài liệu (Document Set Quality)

### Phạm vi

Nhóm tập trung vào chính sách TikTok Shop Việt Nam dành cho người mua và người
bán: trả hàng/hoàn tiền, COD, giao tiếp khách hàng, sản phẩm hạn chế, phí và hiệu
suất người bán. Tất cả 8 tài liệu là trang công khai của TikTok Shop Seller
University, được lấy ngày 03/08/2026.

### Danh sách tài liệu

| # | Tên tài liệu | Phiên bản | Ký tự nội dung | Role / category |
|---|---|---:|---:|---|
| 1 | Khi nào người mua có thể yêu cầu trả hàng hoàn tiền | 2024-07-18 | 2,240 | `buyer` / `returns-refunds` |
| 2 | Chính sách hủy đơn hàng, trả hàng và hoàn tiền | 2026-06-10 | 15,826 | `both` / `returns-refunds` |
| 3 | Chính sách thanh toán khi nhận hàng | 2025-09-08 | 11,989 | `both` / `payments` |
| 4 | Chính sách giao tiếp với khách hàng | 2026-07-01 | 6,867 | `seller` / `customer-support` |
| 5 | Sản phẩm bị hạn chế | 2026-07-14 | 4,903 | `seller` / `seller-policy` |
| 6 | Trả hàng và hoàn tiền | 2026-07-09 | 12,781 | `both` / `returns-refunds` |
| 7 | Phí của nhà bán hàng | 2026-06-26 | 9,533 | `seller` / `payments` |
| 8 | Chính sách đánh giá hiệu suất người bán | 2026-07-09 | 14,627 | `seller` / `seller-policy` |

URL đầy đủ và quyền sử dụng được lưu tại
`data/tiktok_shop_policies/sources.csv`. Corpus không chứa dữ liệu cá nhân, API
key hay tài liệu nội bộ. Mỗi file có YAML front matter để truy vết.

### Kiểm soát chất lượng dữ liệu

- [x] 8/8 tài liệu có `source_url`, `retrieved_at`, `document_version`.
- [x] 8/8 tài liệu có `customer_role`, `category`, `language`, `platform`.
- [x] Loại bỏ phần giao diện “Có thể bạn cũng quan tâm/Next” khỏi cả 8 file.
- [x] `sources.csv` lưu manifest nguồn và trạng thái `public-page`.
- [x] Corpus dùng chung được đóng gói giữ nguyên đường dẫn
  `data/tiktok_shop_policies/`.

### Metadata schema

| Trường | Kiểu | Ví dụ | Tác dụng retrieval |
|---|---|---|---|
| `doc_id` | string | `tiktok-cod-policy` | Đối chiếu gold source, xóa mọi chunk của một tài liệu |
| `source_url` | string | URL Seller University | Trích dẫn và kiểm chứng câu trả lời |
| `retrieved_at` | date string | `2026-08-03` | Kiểm tra độ mới của lần thu thập |
| `document_version` | date string | `2025-09-08` | Phân biệt phiên bản chính sách |
| `customer_role` | enum | `buyer`, `seller`, `both` | Pre-filter theo đối tượng khách hàng |
| `category` | enum | `payments` | Thu hẹp miền chính sách |
| `language` | string | `vi` | Hỗ trợ corpus đa ngôn ngữ về sau |
| `platform` | string | `tiktok-shop-vn` | Tách nền tảng/thị trường |
| `chunk_index` | integer | `3` | Truy vết vị trí chunk sau ingest |

## 2. Thiết kế chiến lược

### Baseline trên ba tài liệu đầu (`chunk_size=700`)

| Tài liệu | Strategy | Số chunk | Độ dài TB |
|---|---|---:|---:|
| Buyer return eligibility | Fixed size | 4 | 597.5 |
|  | By sentences | 3 | 740.0 |
|  | Recursive | 4 | 558.2 |
| Cancellation/return/refund | Fixed size | 25 | 681.0 |
|  | By sentences | 23 | 682.1 |
|  | Recursive | 28 | 563.2 |
| COD | Fixed size | 19 | 678.4 |
|  | By sentences | 20 | 593.3 |
|  | Recursive | 24 | 497.0 |

`SentenceChunker` nhóm theo số câu nên không đảm bảo giới hạn ký tự; vì vậy tài
liệu đầu có độ dài trung bình 740 dù tham số so sánh là 700. `FixedSizeChunker`
đều hơn nhưng có thể cắt giữa điều khoản. `RecursiveChunker` giữ đoạn tốt hơn
nhưng khi chunk được lấy ra khỏi tài liệu, tên mục cha có thể bị mất.

### Chiến lược của Nguyễn Minh Công — `HeadingAwareChunker`

Chiến lược tách Markdown theo heading `#` đến `######`, duy trì cây tiêu đề và
lặp breadcrumb vào từng chunk con. Nếu một section vượt 700 ký tự, phần thân mới
được tách tiếp bằng `RecursiveChunker`. Thiết kế này kế thừa ý tưởng tách theo
Điều/Chương trong repo `rag-traffic-vn`, nhưng phù hợp hơn với cấu trúc heading
của chính sách TikTok Shop.

Ví dụ, một đoạn đứng riêng vẫn mang ngữ cảnh:

```text
# Chính sách giao tiếp với khách hàng
## Hành vi bị cấm
### Thao túng đánh giá và phản hồi

Người bán không được ...
```

| Tài liệu | Heading-aware chunks | Độ dài TB |
|---|---:|---:|
| Buyer return eligibility | 4 | 599.5 |
| Cancellation/return/refund | 43 | 501.8 |
| COD | 29 | 440.8 |

Đánh đổi là số chunk tăng ở tài liệu có nhiều heading nhỏ, nhưng chunk mạch lạc
và có nhãn section rõ hơn. Trên toàn bộ corpus, strategy này tạo 187 chunk.

### Strategy của hai thành viên còn lại

- Nguyễn Văn Sang chọn `FixedSizeChunker(chunk_size=700)`: baseline đơn giản,
  số chunk thấp và chi phí embedding nhỏ, nhưng có thể cắt ngang câu hoặc điều
  kiện liên quan.
- Diệp Đức Lai chọn `RecursiveChunker(chunk_size=700)`: ưu tiên tách theo đoạn,
  dòng và câu trước khi xuống mức từ/ký tự; giữ ngữ nghĩa cục bộ tốt hơn fixed
  size nhưng chunk con có thể mất heading cha.
- Repo tham chiếu:
  [Nguyễn Văn Sang](https://github.com/sang2197/Day07-2A202601252-NguyenVanSang)
  và [Diệp Đức Lai](https://github.com/dieplai/DAY07-2A202601784-DiepDucLai).

### Kết quả từng thành viên

| Thành viên | Strategy | Hit@3 | MRR | Điểm mạnh | Điểm yếu |
|---|---|---:|---:|---|---|
| Nguyễn Minh Công | Heading-aware 700 + `text-embedding-3-small` | 80% | 0.70 | Giữ breadcrumb; 4/5 gold chunks được tìm thấy | Q4 strict miss vì chunk về thực phẩm xếp trên mỹ phẩm |
| Nguyễn Văn Sang | Fixed-size 700 + `text-embedding-3-small` | 80% | 0.67 | Chỉ tạo 129 chunks, đơn giản và tiết kiệm embedding | Q2 chỉ ở rank 3; Q5 miss do điều kiện và ngoại lệ bị cắt rời |
| Diệp Đức Lai | Recursive 700 + `text-embedding-3-small` | 100% | 1.00 | Cả 5 gold chunks đều ở rank 1; chỉ tạo 140 chunks | Có thể mất tên mục cha khi chunk được truy xuất độc lập |

Trong benchmark này, Recursive 700 cho kết quả tốt nhất: vừa đạt Hit@3 100%,
MRR 1.00, vừa dùng ít hơn Heading-aware 47 chunks. Tuy nhiên đây mới là 5 query
trên một corpus; chưa đủ để kết luận Recursive luôn tốt hơn ở mọi domain.

### Ablation embedding provider (không phải so sánh thành viên)

Giữ nguyên corpus, query và `HeadingAwareChunker(chunk_size=700)`, chỉ đổi
embedder để kiểm tra độ nhạy của retrieval:

| Embedder | Số chunk | Hit@3 | MRR | Kết quả |
|---|---:|---:|---:|---|
| OpenAI `text-embedding-3-small` | 187 | 4/5 (80%) | 0.70 | Q4 strict miss |
| Voyage `voyage-4-lite` | 187 | 5/5 (100%) | 1.00 | Cả 5 gold chunks ở rank 1 |

Đây không phải so sánh chunking công bằng giữa thành viên vì biến thay đổi là
embedding model. Benchmark nhóm khóa `text-embedding-3-small`; bảng này chỉ là
thí nghiệm phụ để cho thấy chất lượng retrieval còn phụ thuộc embedder.

## 3. Câu hỏi đánh giá và chất lượng truy xuất

Benchmark đã khóa nằm tại `benchmarks/tiktok_shop_team_v1.json`.

| # | Loại | Query | Gold answer (rút gọn) | Gold document / chunk | Filter |
|---|---|---|---|---|---|
| Q1 | Số liệu | Người mua được gửi yêu cầu trả hàng/hoàn tiền trong bao lâu sau khi đơn được giao? | 15 ngày dương lịch sau khi trạng thái đơn là “Đã giao hàng” | `tiktok-buyer-return-eligibility` / `chunk_0` | `customer_role=buyer` |
| Q2 | Điều kiện | Không nhận đơn COD ba lần trong 60 ngày thì bị xử lý thế nào? | COD bị vô hiệu hóa trong 60 ngày | `tiktok-cod-policy` / `chunk_7` | Không |
| Q3 | Quy trình | Người bán khiếu nại điểm vi phạm hiệu suất như thế nào và có tối đa bao nhiêu lần? | Tối đa 2 lần: lần đầu trong 30 ngày, lần hai trong 15 ngày sau khi lần đầu bị từ chối | `tiktok-seller-performance-policy` / `chunk_27` | `customer_role=seller` |
| Q4 | Liệt kê | Bán mỹ phẩm hạn chế cần tài liệu và thông tin gì? | Giấy/phiếu công bố mỹ phẩm và nhãn có các thông tin bắt buộc | `tiktok-restricted-products` / `chunk_3` | `customer_role=seller` |
| Q5 | Ngoại lệ | Điểm AHR bị trừ được tự động xóa sau bao lâu và ngoại lệ là gì? | Sau 90 ngày, trừ khi shop đã đóng hoặc AHR bằng 0 | `tiktok-seller-performance-policy` / `chunk_4` | `customer_role=seller` |

### Protocol đánh giá chung

Mỗi thành viên chạy đúng corpus, đúng 5 query và `top_k=3`. Một chunk chỉ được
tính relevant khi vừa đúng `doc_id`, vừa chứa toàn bộ `required_terms`; do đó
“đúng tài liệu nhưng sai đoạn” không được tính điểm. Script ghi lại
provider/model, strategy, chunk size, chunk count, từng score, source URL,
`Hit@3` và MRR. Mock chỉ dùng smoke test và không được dùng để kết luận chất
lượng semantic.

```bash
.venv/bin/python scripts/run_retrieval_benchmark.py \
  --benchmark benchmarks/tiktok_shop_team_v1.json \
  --provider openai --strategy heading --chunk-size 700 \
  --output results/<member>-heading-700.json
```

### Kết quả đã tái lập — Nguyễn Minh Công

| # | Strategy | Gold evidence trong top-3? | Rank | Ghi chú |
|---|---|---:|---:|---|
| Q1 | Heading-aware 700 | Có | 1 | Đúng đoạn buyer khi dùng metadata filter |
| Q2 | Heading-aware 700 | Có | 2 | Điều kiện “ba lần trong 60 ngày” được tìm thấy |
| Q3 | Heading-aware 700 | Có | 1 | Đúng quy trình hai lần khiếu nại |
| Q4 | Heading-aware 700 | Không | — | Top-1 cùng tài liệu nhưng là yêu cầu cho thực phẩm/đồ uống; gold chunk mỹ phẩm vắng trong top-3 |
| Q5 | Heading-aware 700 | Có | 1 | Tìm được cả mốc 90 ngày và ngoại lệ |

Benchmark này đánh giá retrieval evidence, chưa tự động chấm câu trả lời do LLM
sinh ra. Vì vậy không dùng nhãn “agent đúng” nếu chưa có answer evaluator.

### A/B metadata filter

Với Q1 và cùng OpenAI embedder:

| Thiết lập | Top-1 | Gold rank | Nhận xét |
|---|---|---:|---|
| Không filter | `tiktok-returns-refunds::chunk_26` | 2 | Một FAQ cùng chủ đề nhưng role `both` đứng trên tài liệu buyer |
| `customer_role=buyer` | `tiktok-buyer-return-eligibility::chunk_0` | 1 | Loại nhiễu sai đối tượng và đẩy gold lên đầu |

Filter làm tăng precision cho truy vấn đã biết đối tượng, nhưng có thể giảm
recall nếu metadata gán sai hoặc tài liệu hợp lệ được đánh nhãn `both`. Vì vậy
chỉ filter các field bắt buộc, còn semantic search tiếp tục xếp hạng bên trong
tập ứng viên.

## 4. Demo và bài học nhóm

Các điểm chính để demo:

1. Data cleaning làm giảm chunk nhiễu trước khi thay model.
2. Cùng corpus/query nhưng chunking thay đổi cả số chunk lẫn thứ hạng nguồn.
3. Metadata filter giúp giới hạn role nhưng không thay thế semantic retrieval.
4. Failure case Q4 cho thấy role filter chưa đủ chi tiết: metadata hiện không có
   `product_category`, nên các chunk thực phẩm và mỹ phẩm vẫn cạnh tranh nhau.
5. Kết quả JSON có source URL nên mọi nhận định đều kiểm tra lại được.

Kết luận: `RecursiveChunker` là strategy tốt nhất trên benchmark hiện tại.
`HeadingAwareChunker` vẫn phù hợp với tài liệu luật/quy định/manual vì
breadcrumb giữ được ngữ cảnh cấp mục, nhưng phải trả giá bằng nhiều chunk và
chi phí embedding hơn. Metadata filter giảm nhiễu theo đối tượng, đổi lại có
nguy cơ bỏ sót khi nhãn metadata không đầy đủ. Nếu làm tiếp, nhóm sẽ bổ sung
`product_category`, mở rộng bộ query và thử hybrid BM25 + vector.

### Artifact benchmark

- Benchmark khóa: `benchmarks/tiktok_shop_team_v1.json`.
- Kết quả ablation Voyage đã lưu: `reference_results/voyage-heading-team-v1.json`.
- Kết quả kiểm chứng local cho Recursive 700: `results/lai-recursive-700-local-validation.json`.
- Script tái lập cho từng thành viên: `scripts/run_retrieval_benchmark.py`.

Lệnh tái lập cấu hình của Diệp Đức Lai khi có `OPENAI_API_KEY`:

```bash
python3 scripts/run_retrieval_benchmark.py \
  --benchmark benchmarks/tiktok_shop_team_v1.json \
  --provider openai --strategy recursive --chunk-size 700 \
  --output results/lai-recursive-700-team-v1.json
```

## Tự đánh giá phần nhóm

| Tiêu chí | Điểm tự đánh giá |
|---|---:|
| Chất lượng bộ tài liệu | 10 / 10 |
| Thiết kế và so sánh chiến lược | 15 / 15 |
| Chất lượng truy xuất | 10 / 10 |
| Demo và bài học nhóm | 5 / 5 |
| **Tổng phần nhóm** | **40 / 40** |
