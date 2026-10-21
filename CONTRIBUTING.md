# Hướng dẫn đóng góp

Cảm ơn bạn đã đóng góp cho công cụ NLP tiếng Việt. Với thay đổi không hiển nhiên, hãy mở Issue trước để thống nhất vấn đề, phạm vi và tiêu chí hoàn thành.

## Thiết lập môi trường

Yêu cầu Python 3.10 trở lên.

```bash
python -m venv .venv
python -m pip install -e '.[dev]'
python -m ruff check .
python -m pytest -q
```

Trên Windows PowerShell, kích hoạt môi trường bằng `.venv\Scripts\Activate.ps1`. Không cần GPU để chạy bộ test hiện tại.

## Issue, branch và commit

1. Mỗi thay đổi đáng kể phải gắn với một Issue có mô tả tái hiện được hoặc tiêu chí chấp nhận rõ ràng.
2. Tạo branch có phạm vi hẹp, ví dụ `feat/issue-12-err`, `fix/issue-18-unicode` hoặc `docs/issue-21-data-license`.
3. Không làm việc trực tiếp trên default branch và không trộn refactor không liên quan vào cùng Pull Request.
4. Commit phải nhỏ, độc lập và dùng tiền tố dễ hiểu như `feat:`, `fix:`, `test:`, `docs:`, `chore:` hoặc `ci:`.
5. Không dùng force push trên branch dùng chung; không commit dataset có giới hạn, secrets, checkpoint hoặc model weights.

## Chất lượng thay đổi

- Code Python phải tương thích Python 3.10+, dùng UTF-8, type hints cho interface mới và test cho hành vi mới hoặc lỗi được sửa.
- Chạy `python -m ruff check .` và `python -m pytest -q` trước khi mở Pull Request.
- Không thay đổi base model, training arguments, prompt, dataset split hoặc metric rồi gộp kết quả với thí nghiệm cũ như thể cùng một cấu hình.
- Với số liệu nghiên cứu, cung cấp command, seed, dataset revision/fingerprint, dependency versions, GPU/VRAM, raw evaluation outputs và giới hạn của kết quả. Không suy diễn hoặc tạo benchmark không có bằng chứng.

Chi tiết về dữ liệu nằm trong [DATA.md](DATA.md). Quy tắc vận hành đầy đủ cho coding agent và Definition of Done nằm trong [AGENTS.md](AGENTS.md).

## Pull Request

Pull Request cần mô tả vấn đề, cách giải quyết, Issue liên quan, bằng chứng kiểm thử và ảnh hưởng tới dataset/model/license/GPU. Điền checklist trong template và ghi rõ mọi bước chưa thể kiểm tra. Maintainer có thể yêu cầu tách PR nếu phạm vi quá rộng.

**Issue phù hợp cho người mới:** cải thiện ví dụ normalization; bổ sung Unicode tests; thêm config validation; viết hướng dẫn Windows/Colab; triển khai ERR bám theo bài báo với citation và test vectors.
