# Đặc tả kỹ thuật: Chỉ số Error Reduction Rate (ERR) cho bài toán chuẩn hóa văn bản tiếng Việt

Tài liệu này xác định đặc tả toán học, thuật toán căn chỉnh token (alignment), thuật toán xác định các sự kiện sửa lỗi chuẩn (canonical correction events), định nghĩa ma trận nhầm lẫn (TP/FP/FN, Precision/Recall) và bộ test vectors tổng hợp để làm cơ sở cho **Issue #7** triển khai độc lập module tính ERR cho dự án `vietnorm-qlora`.

---

## 1. Nguồn trích dẫn và bối cảnh học thuật

### 1.1 Nguồn chính thức của bài báo ViLexNorm
- **Tên bài báo:** *ViLexNorm: A Lexical Normalization Corpus for Vietnamese Social Media Text*
- **Tác giả:** Thanh-Nhi Nguyen, Thanh-Phong Le, Kiet Van Nguyen
- **Hội nghị:** *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)*, trang 1421–1437, St. Julian's, Malta, 17–22/03/2024.
- **ACL Anthology URL:** [https://aclanthology.org/2024.eacl-long.85/](https://aclanthology.org/2024.eacl-long.85/)
- **PDF URL:** [https://aclanthology.org/2024.eacl-long.85.pdf](https://aclanthology.org/2024.eacl-long.85.pdf)
- **DOI chính thức:** [10.18653/v1/2024.eacl-long.85](https://doi.org/10.18653/v1/2024.eacl-long.85)

### 1.2 Nguồn gốc của chỉ số ERR trong Lexical Normalization
Chỉ số ERR trong bài báo ViLexNorm được kế thừa từ các công trình nền tảng của Rob van der Goot:
- **van der Goot (2019):** *MoNoise: A Multi-lingual and Easy-to-use Lexical Normalization Tool*. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: System Demonstrations*, trang 201–206, Florence, Italy. Association for Computational Linguistics. URL: [https://aclanthology.org/P19-3032/](https://aclanthology.org/P19-3032/), DOI: [10.18653/v1/P19-3032](https://doi.org/10.18653/v1/P19-3032).
- **van der Goot et al. (2021):** *MultiLexNorm: A Shared Task on Multilingual Lexical Normalization*. In *Proceedings of the Seventh Workshop on Noisy User-generated Text (W-NUT 2021)*, trang 493–509, Online. Association for Computational Linguistics. URL: [https://aclanthology.org/2021.wnut-1.55/](https://aclanthology.org/2021.wnut-1.55/), DOI: [10.18653/v1/2021.wnut-1.55](https://doi.org/10.18653/v1/2021.wnut-1.55).

---

## 2. Phân định ranh giới: Bằng chứng bài báo và Phần còn mơ hồ

Để tuân thủ nghiêm ngặt quy tắc trong `AGENTS.md` ("*không gọi metric là paper-compatible cho tới khi phương pháp đó được kiểm chứng bằng test vectors và nguồn chính thức*"), bảng dưới đây phân định rõ:
1. **Paper-grounded facts**: Những điều bài báo ViLexNorm (Mục 4.3, trang 1425) khẳng định rõ ràng.
2. **Ambiguities**: Những chi tiết bài báo không công bố hoặc không cung cấp mã nguồn evaluator.
3. **Quy ước đề xuất (Proposed conventions)**: Những giả định kỹ thuật do đội ngũ đề xuất cho Issue #7. **Các mục này phải được Maintainer phê duyệt trước khi gọi là tương thích.**

| Khía cạnh | Khẳng định trong bài báo ViLexNorm (Mục 4.3) | Điểm mơ hồ (Ambiguity) | Quy ước đề xuất cho Issue #7 | Trạng thái phê duyệt |
| :--- | :--- | :--- | :--- | :--- |
| **Công thức ERR** | $\text{ERR} = \frac{\text{Accuracy}_{\text{system}} - \text{Accuracy}_{\text{baseline}}}{1 - \text{Accuracy}_{\text{baseline}}}$ | Mẫu số bằng 0 khi baseline đạt 100% accuracy (trong bài báo, tập dữ liệu thu thập đã lọc câu có từ cần chuẩn hóa) | Gán `0.0` (kèm cờ `zero_denominator_clean`) khi cả 2 bằng 1.0; gán `null` (kèm cờ `zero_denominator_degraded`) khi hệ thống làm hỏng câu sạch | `[Quy ước đề xuất]` |
| **Baseline** | Leave-As-Is (LAI): giữ nguyên văn bản gốc, $\text{ERR}_{\text{LAI}} = 0.0$ | Không có | Triển khai hàm `identity_baseline(text)` | Đã xác nhận từ bài báo |
| **Negative ERR** | ERR có thể nhận giá trị âm khi model kém hơn LAI (ví dụ LSTM đạt $-4.3781$) | Biên độ chặn dưới của negative ERR | Không chặn dưới nhân tạo (giữ nguyên giá trị âm thực tế) | Đã xác nhận từ bài báo |
| **Alignment 1-n, n-1** | *"with 1-n and n-1, we use Levenshtein distance to compute token accuracy"* | Không đưa ra mã nguồn căn chỉnh, ma trận chi phí hay cách break ties | Khoảng cách Levenshtein ở cấp độ token: $\text{Acc} = \max(0, 1 - \frac{\text{dist}}{\|G\|})$ | `[Quy ước đề xuất]` |
| **Tokenization** | Khảo sát cả syllable-level (BARTpho syllable) và word-level (BARTpho word) | Quy tắc chuẩn hóa chuỗi và tách từ cụ thể | Mặc định **NFKC + whitespace + str.split** | `[Quy ước đề xuất]` |
| **Phân biệt hoa/thường** | Không nói rõ case-sensitive hay uncased | Đánh giá phân biệt hoa/thường hay đưa về chữ thường | Mặc định **Case-sensitive** | `[Quy ước đề xuất]` |
| **Mẫu số Accuracy** | Token accuracy | Dùng độ dài Gold hay độ dài Alignment khi có tokens chèn thêm (insertion) | Chuẩn hóa theo độ dài Gold: $\text{Acc} = \max(0, 1 - \frac{D(P, G)}{\|G\|})$, phạt tự nhiên insertion và deletion qua $D$ | `[Quy ước đề xuất]` |
| **Precision / Recall** | Báo cáo Precision và Recall để phân biệt FP và FN | Không định nghĩa công thức toán cho TP, FP, FN ở mức token alignment | Đo lường độc lập trên **Canonical Correction Events** (không dùng để tính ERR) | `[Quy ước đề xuất]` |
| **Cấp độ tính toán** | Báo cáo kết quả tổng trên toàn tập kiểm thử | Micro-average (gộp toàn bộ corpus) hay Macro-average (trung bình cộng từng câu) | Mặc định **Micro-average (Corpus-level)** | `[Quy ước đề xuất]` |

---

## 3. Nguyên lý và Định dạng dữ liệu đầu vào (Input & Tokenization)

### 3.1 Cấp độ biểu diễn ngôn ngữ tiếng Việt
Trong văn bản mạng xã hội tiếng Việt, các đơn vị âm tiết (syllable, phân tách bằng khoảng trắng) là đơn vị tự nhiên nhất để chuẩn hóa lỗi chính tả, teencode và viết tắt:
- Ví dụ: `"mik ko bt lm sao"` $\rightarrow$ 5 âm tiết: `["mik", "ko", "bt", "lm", "sao"]`.
- Gold: `"mình không biết làm sao"` $\rightarrow$ 5 âm tiết: `["mình", "không", "biết", "làm", "sao"]`.

> [!NOTE]
> **Quy ước đề xuất về Tokenization:**
> 1. Chuẩn hóa chuỗi bằng Unicode NFKC để thống nhất mã ký tự dựng sẵn.
> 2. Cắt khoảng trắng đầu cuối (`strip()`) và tách chuỗi theo khoảng trắng: `unicodedata.normalize('NFKC', text).strip().split()`.
> 3. Mặc định giữ nguyên chữ hoa/thường (**Case-sensitive**) `[Quy ước đề xuất]`.

---

## 4. Mô hình Leave-As-Is (LAI) Baseline

Leave-As-Is (LAI) là baseline không tham số (null model):
$$\text{LAI}(x) = x$$
Hệ thống LAI giữ nguyên vẹn chuỗi đầu vào $x$.
- Khoảng cách token Levenshtein so với Gold reference $G$ là $D(S, G)$.
- Độ chính xác của baseline là:
  $$\text{Accuracy}_{\text{baseline}} = \max\left(0.0, 1.0 - \frac{D(S, G)}{|G|}\right)$$
- Khi đánh giá LAI bằng chính công thức ERR:
  $$\text{ERR}_{\text{LAI}} = \frac{\text{Accuracy}_{\text{baseline}} - \text{Accuracy}_{\text{baseline}}}{1 - \text{Accuracy}_{\text{baseline}}} = \frac{0}{1 - \text{Accuracy}_{\text{baseline}}} = 0.0$$
Do đó, $\text{ERR}_{\text{LAI}}$ luôn luôn bằng **$0.0$** (hoặc $0\%$) khi câu có lỗi cần chuẩn hóa.

---

## 5. Công thức toán học của Error Reduction Rate (ERR)

### 5.1 Công thức chuẩn từ bài báo
Theo Mục 4.3 của bài báo ViLexNorm, ERR được định nghĩa duy nhất thông qua Token Accuracy:
$$\text{ERR} = \frac{\text{Accuracy}_{\text{system}} - \text{Accuracy}_{\text{baseline}}}{1 - \text{Accuracy}_{\text{baseline}}}$$

Trong đó:
- $\text{Accuracy}_{\text{system}}$: Độ chính xác token của hệ thống so với chuỗi chuẩn hóa vàng (Gold reference).
- $\text{Accuracy}_{\text{baseline}}$: Độ chính xác token của văn bản gốc (Source) so với chuỗi chuẩn hóa vàng.
- $1 - \text{Accuracy}_{\text{baseline}}$: Tỷ lệ lỗi trong văn bản gốc cần được khắc phục.

> [!IMPORTANT]
> **Lưu ý phương pháp luận:** Dự án **không áp dụng** công thức $\text{ERR} = \frac{\text{TP} - \text{FP}}{\text{TP} + \text{FN}}$ vì bài báo ViLexNorm không đưa ra công thức này cho bài toán seq2seq, và mối liên hệ đại số đó chỉ đúng trong các bài toán phân loại token 1-1 cố định không đổi độ dài. ERR trong `vietnorm-qlora` được tính nghiêm ngặt từ Token Accuracy.

### 5.2 Các miền giá trị của ERR
1. **$\text{ERR} = 1.0$ (100%):** Hệ thống hoàn hảo, khắc phục toàn bộ lỗi của văn bản gốc ($\text{Accuracy}_{\text{system}} = 1.0$).
2. **$0 < \text{ERR} < 1.0$:** Hệ thống cải thiện văn bản gốc, giảm được một phần lỗi.
3. **$\text{ERR} = 0.0$:** Hệ thống không mang lại giá trị gia tăng so với việc không làm gì cả (bằng điểm LAI).
4. **$\text{ERR} < 0.0$ (Negative ERR):** Hệ thống làm hỏng văn bản gốc, tạo ra nhiều lỗi sai hơn số lỗi nó sửa được ($\text{Accuracy}_{\text{system}} < \text{Accuracy}_{\text{baseline}}$). Bảng 3 trong bài báo ViLexNorm ghi nhận kiến trúc LSTM đạt $\text{ERR} = -4.3781$.

### 5.3 Cấp độ tập dữ liệu (Corpus-level / Micro) vs Cấp độ câu (Sentence-level / Macro)
- **Corpus-level (Micro-average) `[Quy ước đề xuất]`:**
  Tính tổng khoảng cách và tổng độ dài token tham chiếu trên toàn bộ tập kiểm thử:
  $$\text{Accuracy}_{\text{system}}^{\text{corpus}} = \max\left(0, 1 - \frac{\sum_i D(P_i, G_i)}{\sum_i |G_i|}\right), \quad \text{Accuracy}_{\text{baseline}}^{\text{corpus}} = \max\left(0, 1 - \frac{\sum_i D(S_i, G_i)}{\sum_i |G_i|}\right)$$
  Sau đó áp dụng công thức ERR một lần duy nhất cho toàn tập. Đây là cách tiếp cận chuẩn giúp phản ánh đúng trọng số của câu dài và giảm đáng kể vấn đề mẫu số 0 ở cấp câu (dù không tránh hoàn toàn nếu toàn bộ tập kiểm thử là các câu sạch không có lỗi).
- **Sentence-level (Macro-average):**
  Tính ERR cho từng câu rồi lấy trung bình cộng.

### 5.4 Hiện tượng Mẫu số bằng 0 ($1 - \text{Accuracy}_{\text{baseline}} = 0$)
Trong bài báo ViLexNorm (Mục 3), tập dữ liệu được xây dựng bằng cách thu thập các bình luận mạng xã hội có chứa từ bất chuẩn cần chuẩn hóa; vì vậy, trong phân chia benchmark chuẩn của bài báo, các câu đều có lỗi và không xuất hiện trường hợp mẫu số bằng 0.

Tuy nhiên, trong các bộ kiểm thử phần mềm, pipeline sản xuất hoặc dữ liệu tổng quát, hiện tượng văn bản đầu vào sạch hoàn toàn có thể xảy ra. Để đảm bảo code không crash, dự án đề xuất quy ước như sau:
- **Kịch bản A (Câu sạch và Model giữ sạch):** $\text{Accuracy}_{\text{baseline}} = 1.0$ và $\text{Accuracy}_{\text{system}} = 1.0$.
  - Gán $\text{ERR} = 0.0$ kèm cờ trạng thái `err_status: "zero_denominator_clean"`.
- **Kịch bản B (Câu sạch nhưng Model làm hỏng - Over-normalization):** $\text{Accuracy}_{\text{baseline}} = 1.0$ và $\text{Accuracy}_{\text{system}} < 1.0$.
  - Tử số âm, mẫu số 0. Trả về $\text{ERR} = \text{null}$ kèm cờ trạng thái `err_status: "zero_denominator_degraded"`.

---

## 6. Thuật toán Căn chỉnh Token (Token Alignment) và Token Accuracy

### 6.1 Thuật toán Levenshtein ở cấp độ Token (Quy ước đề xuất)
Để tính độ chính xác token một cách tất định (deterministic), ta sử dụng khoảng cách Levenshtein kinh điển (Wagner-Fischer) trên danh sách các token:
- Cho dãy token $A = [a_1, \dots, a_m]$ và $B = [b_1, \dots, b_n]$:
  - $\text{cost}(\text{match}) = 0$ nếu $a_i = b_j$.
  - $\text{cost}(\text{substitution}) = 1$ nếu $a_i \neq b_j$.
  - $\text{cost}(\text{insertion}) = 1$.
  - $\text{cost}(\text{deletion}) = 1$.
- Khoảng cách $D(A, B) = \text{token\_levenshtein\_distance}(A, B)$ là tổng số thao tác nhỏ nhất để biến $A$ thành $B$.

### 6.2 Công thức Token Accuracy tất định có phạt Insertion
$$\text{Accuracy}(A, B) = \max\left(0.0, 1.0 - \frac{D(A, B)}{|B|}\right)$$
Trong đó $|B|$ là số lượng token của chuỗi tham chiếu Gold.
- **Tính tất định:** Chỉ phụ thuộc vào khoảng cách sửa đổi Levenshtein tối thiểu và độ dài Gold, không phụ thuộc vào đường đi traceback cụ thể.
- **Xử lý 1-n:** Khi 1 từ nguồn tách thành $n$ từ đích (ví dụ `đky` $\rightarrow$ `đăng ký`), $D(S, G) = 2$ (1 thay thế + 1 chèn). Nếu mô hình dự đoán đúng cả 2 từ, $D(P, G) = 0 \implies \text{Accuracy} = 1.0$.
- **Xử lý n-1:** Khi $n$ từ nguồn gộp thành 1 từ đích (ví dụ `r ất` $\rightarrow$ `rất`), $D(S, G) = 2$ (1 thay thế + 1 xóa token thừa). Nếu mô hình gộp đúng, $D(P, G) = 0 \implies \text{Accuracy} = 1.0$.
- **Xử lý Insertion (Chèn thừa):** Nếu mô hình sinh thêm các token thừa, mỗi token thừa làm tăng $D(P, G)$ lên 1 đơn vị, phạt trực tiếp độ chính xác của mô hình theo tỷ lệ $\frac{1}{|G|}$.

---

## 7. Canonical Correction Events và Precision / Recall

### 7.1 Phân tách độc lập giữa Token Accuracy và Precision / Recall
Precision và Recall không dùng để tính toán ERR mà đóng vai trò là các chỉ số chẩn đoán (diagnostic metrics) nhằm đánh giá chất lượng của các can thiệp sửa đổi.

### 7.2 Thuật toán trích xuất Canonical Correction Events (Quy ước đề xuất)
Để xác định các sự kiện sửa lỗi một cách hình thức và tự động:
1. **Căn chỉnh Wagner-Fischer với Tie-break cố định:**
   Khi truy vết (traceback) từ $(m, n)$ về $(0, 0)$, nếu có nhiều thao tác cho cùng chi phí tối ưu, thứ tự ưu tiên cố định là:
   $$\text{Match} \rightarrow \text{Substitution} \rightarrow \text{Deletion} \rightarrow \text{Insertion}$$
2. **Gom cụm Non-match liên tiếp tối đại (Maximal Contiguous Non-Match Blocks):**
   Gộp tất cả các thao tác liên tiếp không phải là `match` (`sub`, `del`, `ins`) thành một khối sửa đổi duy nhất.
   Mỗi khối xác định một **Edit Event** gồm:
   - `source_span`: khoảng nửa mở `[start, end)` (theo quy ước Python slicing `source_tokens = Source[start:end]`; `start == end` biểu diễn vị trí chèn giữa token `start - 1` và `start`).
   - `source_tokens`: danh sách token trong Source bị thay thế (rỗng nếu là thao tác chèn thuần túy `[start, start)`).
   - `replacement_tokens`: danh sách token thay thế tương ứng.
3. **Hai tập sự kiện:**
   - **Target Events ($\mathcal{E}_{\text{target}}$):** Trích xuất từ cặp (Source, Gold).
   - **Predicted Events ($\mathcal{E}_{\text{pred}}$):** Trích xuất từ cặp (Source, Prediction).

### 7.3 Khớp sự kiện và Ma trận nhầm lẫn
- **True Positive (TP):** Một sự kiện dự đoán $e_p \in \mathcal{E}_{\text{pred}}$ khớp chính xác với một sự kiện vàng $e_t \in \mathcal{E}_{\text{target}}$:
  $$e_p.\text{source\_span} == e_t.\text{source\_span} \quad \text{và} \quad e_p.\text{replacement\_tokens} == e_t.\text{replacement\_tokens}$$
- **False Positive (FP):**
  - Một sự kiện dự đoán $e_p$ can thiệp vào vị trí không có trong $\mathcal{E}_{\text{target}}$ (Over-normalization).
  - HOẶC $e_p$ can thiệp vào một span lỗi nhưng sinh ra `replacement_tokens` không khớp Gold (sửa sai).
- **False Negative (FN):**
  - Một sự kiện vàng $e_t \in \mathcal{E}_{\text{target}}$ không có sự kiện dự đoán nào khớp (Under-normalization / bỏ sót).
- **True Negative (TN):** Số lượng token sạch trong Source không bị bất kỳ sự kiện nào can thiệp.

### 7.4 Công thức tính và Xử lý ngoại lệ Mẫu số 0
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- Khi $\text{TP} + \text{FP} = 0$ (mô hình không thực hiện bất kỳ can thiệp nào):
  - `precision = null` kèm trạng thái `precision_status: "undefined_no_predictions"`.
- Khi $\text{TP} + \text{FN} = 0$ (văn bản không có sự kiện lỗi nào cần sửa):
  - `recall = null` kèm trạng thái `recall_status: "undefined_no_targets"`.

---

## 8. Bộ Test Vectors Tổng hợp Đầy đủ (9 Vectors)

> [!CAUTION]
> Toàn bộ 9 test vectors dưới đây được biên soạn **hoàn toàn tổng hợp (synthetic)** độc lập, **không sử dụng bất kỳ câu nào từ tập dữ liệu ViLexNorm**.

---

### Vector 1: Baseline Leave-As-Is (LAI)
- **Mục tiêu:** Kiểm chứng $\text{ERR} = 0.0$ khi hệ thống không can thiệp.
- **Source ($S$):** `"hnay đi hok vui"` (4 tokens)
- **Gold ($G$):** `"hôm nay đi học vui"` (5 tokens)
- **Prediction ($P$):** `"hnay đi hok vui"` (4 tokens)
- **Khoảng cách token Levenshtein:** $D(S, G) = 3, D(P, G) = 3, |G| = 5$.
- **Độ chính xác:** $\text{Accuracy}_{\text{baseline}} = 0.40, \text{Accuracy}_{\text{system}} = 0.40 \implies \mathbf{\text{ERR} = 0.0}$.
- **Events:**
  - $\mathcal{E}_{\text{target}}$: Span `[0, 1)` (`hnay` $\rightarrow$ `hôm nay`), Span `[2, 3)` (`hok` $\rightarrow$ `học`). (2 events)
  - $\mathcal{E}_{\text{pred}}$: rỗng (0 events).
  - $\text{TP} = 0, \text{FP} = 0, \text{FN} = 2, \text{TN} = 2$.
  - $\text{Precision} = \text{null}$ (`undefined_no_predictions`), $\text{Recall} = 0.0$.

---

### Vector 2: Positive ERR (Hệ thống sửa đúng một phần)
- **Mục tiêu:** Kiểm chứng $0 < \text{ERR} < 1.0$ khi hệ thống sửa thành công lỗi 1-n mà không gây over-normalization.
- **Source ($S$):** `"hnay đi hok vui"` (4 tokens)
- **Gold ($G$):** `"hôm nay đi học vui"` (5 tokens)
- **Prediction ($P$):** `"hôm nay đi hok vui"` (5 tokens)
- **Khoảng cách token Levenshtein:** $D(S, G) = 3, D(P, G) = 1, |G| = 5$.
- **Độ chính xác:** $\text{Accuracy}_{\text{baseline}} = 0.40, \text{Accuracy}_{\text{system}} = 0.80$.
- **Chỉ số ERR:** $\text{ERR} = \frac{0.80 - 0.40}{1.0 - 0.40} = \frac{0.40}{0.60} \approx \mathbf{0.666667}$ (66.67%).
- **Events:**
  - $\mathcal{E}_{\text{target}}$: 2 events.
  - $\mathcal{E}_{\text{pred}}$: Span `[0, 1)` (`hnay` $\rightarrow$ `hôm nay`). (1 event)
  - Khớp 1 event $\implies \text{TP} = 1, \text{FP} = 0, \text{FN} = 1, \text{TN} = 2$.
  - $\text{Precision} = 1.0, \text{Recall} = 0.50$.

---

### Vector 3: Negative ERR (Hệ thống sửa bừa / Suy thoái chất lượng)
- **Mục tiêu:** Kiểm chứng $\text{ERR} < 0.0$ khi mô hình can thiệp sai vào các từ vốn dĩ đã đúng.
- **Source ($S$):** `"hnay đi hok vui"` (4 tokens)
- **Gold ($G$):** `"hôm nay đi học vui"` (5 tokens)
- **Prediction ($P$):** `"hnay chạy hok buồn"` (4 tokens)
- **Khoảng cách token Levenshtein:** $D(S, G) = 3, D(P, G) = 5, |G| = 5$.
- **Độ chính xác:** $\text{Accuracy}_{\text{baseline}} = 0.40, \text{Accuracy}_{\text{system}} = 0.0$.
- **Chỉ số ERR:** $\text{ERR} = \frac{0.00 - 0.40}{1.0 - 0.40} \approx \mathbf{-0.666667}$ (-66.67%).
- **Events:**
  - $\mathcal{E}_{\text{target}}$: 2 events.
  - $\mathcal{E}_{\text{pred}}$: Span `[1, 2)` (`đi` $\rightarrow$ `chạy`), Span `[3, 4)` (`vui` $\rightarrow$ `buồn`). (2 events)
  - Không khớp event nào $\implies \text{TP} = 0, \text{FP} = 2, \text{FN} = 2, \text{TN} = 0$.
  - $\text{Precision} = 0.0, \text{Recall} = 0.0$.

---

### Vector 4: Căn chỉnh 1-n (Tách từ hoàn hảo)
- **Mục tiêu:** Kiểm chứng trường hợp 1 token viết tắt tách thành 2 token chuẩn hóa và đạt $\text{ERR} = 1.0$.
- **Source ($S$):** `"tôi đky môn này"` (4 tokens)
- **Gold ($G$):** `"tôi đăng ký môn này"` (5 tokens)
- **Prediction ($P$):** `"tôi đăng ký môn này"` (5 tokens)
- **Khoảng cách token Levenshtein:** $D(S, G) = 2, D(P, G) = 0, |G| = 5$.
- **Độ chính xác:** $\text{Accuracy}_{\text{baseline}} = 0.60, \text{Accuracy}_{\text{system}} = 1.0 \implies \mathbf{\text{ERR} = 1.0}$.
- **Events:**
  - $\mathcal{E}_{\text{target}}$: Span `[1, 2)` (`đky` $\rightarrow$ `đăng ký`). (1 event)
  - $\mathcal{E}_{\text{pred}}$: Span `[1, 2)` (`đky` $\rightarrow$ `đăng ký`). (1 event)
  - $\text{TP} = 1, \text{FP} = 0, \text{FN} = 0, \text{TN} = 3$.
  - $\text{Precision} = 1.0, \text{Recall} = 1.0$.

---

### Vector 5: Căn chỉnh n-1 (Gộp từ hoàn hảo)
- **Mục tiêu:** Kiểm chứng trường hợp $n$ tokens gõ tách rời được gộp thành 1 token chuẩn và đạt $\text{ERR} = 1.0$.
- **Source ($S$):** `"tôi thấy r ất vui"` (5 tokens: `r`, `ất` bị tách)
- **Gold ($G$):** `"tôi thấy rất vui"` (4 tokens)
- **Prediction ($P$):** `"tôi thấy rất vui"` (4 tokens)
- **Khoảng cách token Levenshtein:** $D(S, G) = 2$ (1 thay thế + 1 xóa token thừa trong baseline), $D(P, G) = 0, |G| = 4$.
- **Độ chính xác:** $\text{Accuracy}_{\text{baseline}} = 0.50, \text{Accuracy}_{\text{system}} = 1.0 \implies \mathbf{\text{ERR} = 1.0}$.
- **Events:**
  - $\mathcal{E}_{\text{target}}$: Span `[2, 4)` (`r ất` $\rightarrow$ `rất`). (1 event)
  - $\mathcal{E}_{\text{pred}}$: Span `[2, 4)` (`r ất` $\rightarrow$ `rất`). (1 event)
  - $\text{TP} = 1, \text{FP} = 0, \text{FN} = 0, \text{TN} = 3$.
  - $\text{Precision} = 1.0, \text{Recall} = 1.0$.

---

### Vector 6: Hiện tượng Chèn (Insert) và Bỏ sót (Delete) có phạt Insertion
- **Mục tiêu:** Kiểm chứng việc phạt trực tiếp các token chèn thừa trong dự đoán.
- **Source ($S$):** `"hôm nay trời đẹp"` (4 tokens)
- **Gold ($G$):** `"hôm nay trời rất đẹp"` (5 tokens)
- **Prediction ($P$):** `"hôm nay trời thực sự đẹp"` (6 tokens)
- **Khoảng cách token Levenshtein:** $D(S, G) = 1, D(P, G) = 2, |G| = 5$.
- **Độ chính xác:** $\text{Accuracy}_{\text{baseline}} = 0.80, \text{Accuracy}_{\text{system}} = 0.60$.
- **Chỉ số ERR:** $\text{ERR} = \frac{0.60 - 0.80}{1.00 - 0.80} = \mathbf{-1.0}$ (-100%).
- **Events:**
  - $\mathcal{E}_{\text{target}}$: Span `[3, 3)` (chèn `rất`). (1 event)
  - $\mathcal{E}_{\text{pred}}$: Span `[3, 3)` (chèn `thực sự`). (1 event)
  - Nội dung thay thế khác nhau $\implies \text{TP} = 0, \text{FP} = 1, \text{FN} = 1, \text{TN} = 4$.
  - $\text{Precision} = 0.0, \text{Recall} = 0.0$.

---

### Vector 7: Câu không đổi (Unchanged / Clean sentence)
- **Mục tiêu:** Stress test trường hợp câu nguồn vốn đã chuẩn 100%.
- **Source ($S$):** `"thời tiết hôm nay rất đẹp"` (6 tokens)
- **Gold ($G$):** `"thời tiết hôm nay rất đẹp"` (6 tokens)
- **Prediction ($P$):** `"thời tiết hôm nay rất đẹp"` (6 tokens)
- **Khoảng cách và Độ chính xác:** $D(S, G) = 0, D(P, G) = 0 \implies \text{Accuracy}_{\text{base}} = 1.0, \text{Accuracy}_{\text{sys}} = 1.0$.
- **Chỉ số ERR:** $\mathbf{\text{ERR} = 0.0}$ (`zero_denominator_clean`).
- **Events:**
  - $\mathcal{E}_{\text{target}} = \emptyset, \mathcal{E}_{\text{pred}} = \emptyset$.
  - $\text{TP} = 0, \text{FP} = 0, \text{FN} = 0, \text{TN} = 6$.
  - $\text{Precision} = \text{null}$ (`undefined_no_predictions`), $\text{Recall} = \text{null}$ (`undefined_no_targets`).

---

### Vector 8: Denominator Zero - Giữ nguyên câu sạch
- **Mục tiêu:** Stress test kiểm chứng độc lập việc xử lý chia cho 0 khi baseline = 1.0 và system = 1.0.
- **Source ($S$):** `"học sinh chăm chỉ"` (4 tokens)
- **Gold ($G$):** `"học sinh chăm chỉ"` (4 tokens)
- **Prediction ($P$):** `"học sinh chăm chỉ"` (4 tokens)
- **Kết quả:** $\text{Accuracy}_{\text{base}} = 1.0, \text{Accuracy}_{\text{sys}} = 1.0 \implies \mathbf{\text{ERR} = 0.0}$ (`zero_denominator_clean`).
- **Events:** $\text{Precision} = \text{null}, \text{Recall} = \text{null}$.

---

### Vector 9: Denominator Zero - Làm hỏng câu sạch (Hai substitutions liền nhau)
- **Mục tiêu:** Stress test kiểm chứng việc gộp hai thao tác thay thế liên tiếp thành một sự kiện FP duy nhất.
- **Source ($S$):** `"học sinh chăm chỉ"` (4 tokens)
- **Gold ($G$):** `"học sinh chăm chỉ"` (4 tokens)
- **Prediction ($P$):** `"học sinh lười biếng"` (4 tokens)
- **Khoảng cách và Độ chính xác:** $D(S, G) = 0, D(P, G) = 2 \implies \text{Accuracy}_{\text{base}} = 1.0, \text{Accuracy}_{\text{sys}} = 0.50$.
- **Chỉ số ERR:** Tử số âm, mẫu số $0 \implies \mathbf{\text{ERR} = \text{null}}$ (`zero_denominator_degraded`).
- **Events:**
  - $\mathcal{E}_{\text{target}} = \emptyset$.
  - $\mathcal{E}_{\text{pred}}$: Hai thay thế liền kề `chăm` $\rightarrow$ `lười` và `chỉ` $\rightarrow$ `biếng` trên span `[2, 4)` gộp thành **1 FP event duy nhất**: `["chăm", "chỉ"]` $\rightarrow$ `["lười", "biếng"]`.
  - $\text{TP} = 0, \mathbf{\text{FP} = 1}, \text{FN} = 0, \text{TN} = 2$.
  - $\text{Precision} = 0 / 1 = 0.0, \text{Recall} = \text{null}$ (`undefined_no_targets`).

---

## 9. Hướng dẫn dành cho Implementer (Issue #7)

Khi thực hiện Issue #7, lập trình viên cần tuân thủ các bước sau:
1. **Đọc file JSON Test Vectors:** Sử dụng file `tests/fixtures/err_test_vectors.json` làm fixture chuẩn trong pytest.
2. **Sử dụng Tokenizer chuẩn:**
   ```python
   import unicodedata

   def tokenize_canonical(text: str) -> list[str]:
       return unicodedata.normalize("NFKC", text).strip().split()
   ```
3. **Triển khai thuật toán tất định:**
   ```python
   def token_levenshtein_distance(a: list[str], b: list[str]) -> int:
       ...

   def token_accuracy(seq: list[str], gold: list[str]) -> float:
       if not gold:
           return 1.0 if not seq else 0.0
       dist = token_levenshtein_distance(seq, gold)
       return max(0.0, 1.0 - dist / len(gold))

   def compute_err(acc_system: float, acc_baseline: float) -> float | None:
       if acc_baseline >= 1.0:
           return 0.0 if acc_system >= 1.0 else None
       return (acc_system - acc_baseline) / (1.0 - acc_baseline)
   ```
4. **Cung cấp API đánh giá toàn tập (Micro-average):**
   ```python
   def evaluate_corpus_err(records: list[dict[str, str]]) -> dict[str, float | None]:
       ...
   ```

---

## 10. Danh mục cần Maintainer phê duyệt (Checklist for Approval)

Trước khi gắn nhãn `paper-compatible` cho module triển khai ở Issue #7, Maintainer cần xem xét và duyệt các điểm sau:
- [ ] **Quy ước 1:** Chấp thuận quy tắc tách từ và chuẩn hóa chuỗi: `unicodedata.normalize('NFKC', text).strip().split()`, chế độ phân biệt hoa thường (**case-sensitive**).
- [ ] **Quy ước 2:** Chấp thuận công thức Token Accuracy tất định có phạt insertion: $\text{Acc} = \max(0, 1 - \frac{D(A, B)}{|B|})$ dựa trên khoảng cách Levenshtein ở cấp âm tiết (syllable).
- [ ] **Quy ước 3:** Chấp thuận thuật toán xác định Canonical Correction Events bằng Wagner-Fischer tie-break cố định và gom cụm maximal contiguous non-match blocks.
- [ ] **Quy ước 4:** Chấp thuận quy tắc trả về `null` cho Precision khi $\text{TP} + \text{FP} = 0$, và `null` cho Recall khi $\text{TP} + \text{FN} = 0$.
- [ ] **Quy ước 5:** Chấp thuận quy ước xử lý mẫu số bằng 0 của ERR: gán $0.0$ cho câu sạch được giữ sạch, và gán `null` kèm cờ `zero_denominator_degraded` cho câu sạch bị sửa hỏng.
- [ ] **Quy ước 6:** Chấp thuận mặc định chế độ tính **Micro-average** (Corpus-level) khi báo cáo chỉ số benchmark của mô hình.
