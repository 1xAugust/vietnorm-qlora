# VietNorm-QLoRA 🇻🇳

**Bộ công cụ nghiên cứu mã nguồn mở cho các thí nghiệm fine-tuning một LLM nhỏ bằng QLoRA để chuẩn hóa văn bản tiếng Việt không trang trọng.**

**Trạng thái: research starter / pre-training.** Repository hiện cung cấp bộ đánh giá chạy được trên CPU, identity baseline, script chuẩn bị dữ liệu có seed cấu hình được và một công thức QLoRA thử nghiệm cho `Qwen/Qwen2.5-1.5B-Instruct`. Công thức GPU **chưa được xác minh end-to-end**. Repository chưa phát hành trained adapter, benchmark của model hoặc tuyên bố cải thiện chất lượng.

## Chức năng hướng tới

Input: `mik ko bt lm sao để đky môn học nx`  
Target: `Mình không biết làm sao để đăng ký môn học nữa.`

Cặp trên chỉ dùng để minh họa, **không phải annotation chính thức của ViLexNorm**. Mục tiêu là sửa viết tắt, teencode và lỗi chính tả trong khi giữ nguyên ý nghĩa, tên riêng và phần văn bản đã đúng.

## Vì sao có dự án này?

- Xây dựng pipeline dễ sử dụng cho bài toán chuẩn hóa văn bản tiếng Việt trên mạng xã hội.
- Hỗ trợ thí nghiệm QLoRA có thể tái lập với model mở cỡ nhỏ.
- Đánh giá minh bạch bằng identity baseline, exact match và character error rate (CER).
- Error Reduction Rate (ERR) tương thích với bài báo, demo tương tác và adapter tải xuống mới nằm trong roadmap, chưa được phát hành.

## Bắt đầu nhanh: không cần GPU hoặc dataset

Yêu cầu **Python 3.10+**.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m ruff check .
python -m pytest -q
python -m vietnorm_qlora baseline "mik ko bt lm sao"
python -m vietnorm_qlora evaluate --file data/examples.jsonl
```

Lệnh `baseline` cố ý **không sửa** input; đây là đối chứng leave-as-is. File `data/examples.jsonl` chỉ chứa dữ liệu tổng hợp để kiểm thử code, không phải benchmark hoặc kết quả nghiên cứu.

## Chuẩn bị dữ liệu nghiên cứu ViLexNorm

ViLexNorm là corpus của bên thứ ba theo giấy phép **CC BY-NC-SA 4.0**, không thuộc giấy phép MIT của repository này. Không có hàng dữ liệu ViLexNorm nào được đóng gói trong repository. Phải kiểm tra quyền sử dụng, attribution và revision trước khi tải, phân phối lại hoặc triển khai. Xem [DATA.md](DATA.md).

```bash
python -m pip install -e '.[train]'
python scripts/prepare_data.py \
  --revision "<full ViLexNorm commit SHA>" \
  --output-dir data/processed \
  --max-train 3000
```

Thay placeholder bằng commit SHA đầy đủ của revision ViLexNorm đã duyệt cho thí nghiệm. Script
ghi `manifest.json` cạnh `train.jsonl`, `dev.jsonl` và `test.jsonl`; manifest chứa provenance,
schema đầu vào và số hàng trước/sau subset, nhưng không chứa nội dung từng hàng của corpus.

Script tải `visolex/ViLexNorm` từ Hugging Face và lọc trường `type` để giữ membership `train`, `dev` và `test`. Các file prompt/completion sinh ra nằm trong thư mục được Git bỏ qua. Script không dùng test examples để training.

## Kiểm tra duplicate và data leakage

Sau khi chuẩn bị các split, chạy audit bắt buộc để phát hiện duplicate nội bộ và data leakage chéo split trước khi huấn luyện hoặc benchmark:

```bash
python scripts/audit_data.py --data-dir data/processed --update-manifest
# hoặc dùng CLI tiện ích:
python -m vietnorm_qlora audit --data-dir data/processed
```

Pipeline audit kiểm tra deterministic:
- Trùng lặp nội bộ từng split: trùng cặp đầy đủ `(original, normalized)`, cùng `original` nhưng khác `normalized` (`inconsistent_target`), và near-duplicate;
- Trùng lặp chéo split giữa `train-dev`, `train-test`, `dev-test`: exact `original`, exact `normalized`, exact full pair, `inconsistent_target`, và near-duplicate input;
- Near-duplicate Unicode tiếng Việt: chuẩn hóa Unicode NFC, case-folding, gộp khoảng trắng thừa và tính normalized Levenshtein similarity với threshold mặc định `0.85` (cấu hình được qua `--near-threshold`);
- An toàn dữ liệu: báo cáo tổng hợp chỉ xuất số lượng count, mặc định không để lộ raw text của dataset;
- Exit behavior: trả exit code `0` khi sạch (PASS) và exit code khác `0` (`1`) khi phát hiện leakage (FAIL).

## Huấn luyện QLoRA (GPU / thử nghiệm)

Chỉ chạy trên CUDA GPU phù hợp, ví dụ Colab T4 đã được cấp phát, sau khi kiểm tra môi trường và ghi lại đầy đủ dependency versions. Không chạy công thức này trên máy chỉ có CPU hoặc máy có 8 GB RAM.

```bash
python scripts/train_qlora.py --train data/processed/train.jsonl --validation data/processed/dev.jsonl --out outputs/qwen15b-qlora
```

Script hiện dùng 4-bit NF4 BitsAndBytes, PEFT LoRA rank 16 và TRL `SFTTrainer`. API của Transformers, TRL hoặc PEFT có thể thay đổi; công thức chưa được xác minh end-to-end. Không thay đổi hyperparameters hoặc base model mà không ghi nhận đó là một biến thí nghiệm riêng.

## Đánh giá (evaluation)

```bash
# JSONL held-out gồm các trường original, normalized và prediction
python -m vietnorm_qlora evaluate --file data/examples.jsonl
```

Các metric đã có: sentence exact match, CER, change rate và count. Các metric này **không thay thế ERR token-aligned của bài báo ViLexNorm**; không được công bố kết quả paper-comparable trước khi ERR được triển khai và kiểm thử đúng phương pháp.

## Trạng thái và đóng góp

Tiến độ nghiên cứu được quản lý duy nhất trong [ROADMAP.md](ROADMAP.md). Quy trình đóng góp nằm tại [CONTRIBUTING.md](CONTRIBUTING.md); quy tắc dành cho coding agent nằm tại [AGENTS.md](AGENTS.md).

Nguồn dataset và bài báo gốc:

- https://huggingface.co/datasets/visolex/ViLexNorm
- https://github.com/ngxtnhi/ViLexNorm
- Nguyen et al. (2024), EACL: https://aclanthology.org/2024.eacl-long.85/

**Code license:** MIT, xem [LICENSE](LICENSE). **Dataset license:** CC BY-NC-SA 4.0. Model, checkpoint và adapter có thể chịu thêm điều kiện giấy phép; việc code là mã nguồn mở không làm cho training data hoặc model artifact tự động được phép sử dụng thương mại.
