# Tóm tắt

Mô tả vấn đề, cách giải quyết và phạm vi thay đổi.

Closes #

## Kiểm tra đã thực hiện

Liệt kê command và kết quả thực tế. Ghi rõ bước chưa thể chạy.

- [ ] `python -m ruff check .`
- [ ] `python -m pytest -q`
- [ ] Chỉ dùng CPU trong CI; không tải model hoặc chạy fine-tuning ngoài kế hoạch được duyệt.

## Dataset, model và reproducibility

- [ ] Không thay đổi dataset schema/split, base model hoặc training arguments; hoặc thay đổi đã được mô tả và phê duyệt trong Issue.
- [ ] Không có data leakage, dữ liệu cá nhân, secrets, checkpoint hoặc model weights trong diff.
- [ ] Thay đổi liên quan thí nghiệm có command, seed, dataset revision/fingerprint, dependency versions, GPU/VRAM và raw outputs phù hợp.

## Checklist hoàn thành

- [ ] PR có phạm vi hẹp và liên kết Issue.
- [ ] Tài liệu và config liên quan đã được cập nhật.
- [ ] Tôi đã tự review diff và nêu rõ mọi giới hạn hoặc rủi ro còn lại.
