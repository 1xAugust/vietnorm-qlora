# Nguồn gốc và quy tắc sử dụng dữ liệu

## Nguồn ViLexNorm

Dataset nguồn là **ViLexNorm**, corpus cho bài toán chuẩn hóa từ vựng tiếng Việt:

- Repository của tác giả: https://github.com/ngxtnhi/ViLexNorm
- Bản mirror trên Hugging Face: https://huggingface.co/datasets/visolex/ViLexNorm
- Bài báo: https://aclanthology.org/2024.eacl-long.85/

ViLexNorm được phát hành theo **CC BY-NC-SA 4.0**. Dataset **không thuộc giấy phép MIT** chỉ vì code của dự án dùng MIT. Không sao chép toàn bộ corpus vào repository công khai và không mặc định coi dataset là phù hợp cho mục đích thương mại. Trước khi phân phối lại hoặc triển khai, phải kiểm tra attribution, provenance, quyền sử dụng và các điều kiện áp dụng. Model được training từ corpus có thể cần đánh giá pháp lý và chính sách riêng.

## Revision và cấu trúc split

Bản Hugging Face hợp nhất dữ liệu vào một split vật lý và dùng trường `type` để biểu diễn membership `train`, `dev` và `test`. Metadata hiển thị ở upstream từng có khác biệt về tên split và số hàng; vì vậy mỗi thí nghiệm phải ghi lại:

- dataset identifier và revision/commit chính xác;
- fingerprint hoặc checksum nếu công cụ cung cấp;
- các giá trị duy nhất của trường `type` và số hàng theo từng giá trị;
- số lượng hàng sau khi lọc, loại trùng hoặc lấy subset;
- command và seed đã dùng để tạo dữ liệu đầu vào.

Không thay đổi schema, ý nghĩa trường hoặc membership của split mà không có Issue, đánh giá tác động và cập nhật tài liệu. Output mặc định của `scripts/prepare_data.py` nằm tại `data/processed/` và không được commit.

## Phòng tránh data leakage

- Chỉ dùng `train` để fit parameters; dùng `dev` để chọn checkpoint, prompt hoặc hyperparameters.
- Không dùng `test` để điều chỉnh prompt, chọn checkpoint, quyết định stopping hoặc sửa pipeline dựa trên từng mẫu.
- Chỉ chạy đánh giá cuối trên `test` sau khi cấu hình đã được cố định.
- Kiểm tra exact duplicate và near-duplicate giữa các split trước khi công bố kết quả; ghi lại phương pháp và số lượng phát hiện.
- Không đưa nội dung dataset có giấy phép hạn chế hoặc dữ liệu cá nhân vào Issue, log, fixture, prompt mẫu hay Pull Request.

## Fixture trong repository

`data/examples.jsonl` là dữ liệu tổng hợp nguyên gốc, chỉ dùng để kiểm thử CLI và metric. File này không phải benchmark dataset và kết quả trên file không được trình bày như kết quả nghiên cứu.

Mọi dữ liệu mới được đề xuất đưa vào repository phải có provenance, license, mục đích sử dụng và xác nhận không chứa thông tin cá nhân hoặc nội dung không được phép công bố.
