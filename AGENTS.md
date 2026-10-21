# Quy tắc dành cho coding agent

File này là nguồn quy tắc vận hành chung cho Codex, Claude và các coding agent khác trong repository `vietnorm-qlora`. Khi có xung đột, yêu cầu trực tiếp của maintainer được ưu tiên; agent phải báo rõ xung đột trước khi hành động.

## 1. Đọc repository trước khi sửa

1. Kiểm kê toàn bộ cây thư mục và đọc `README.md`, `CONTRIBUTING.md`, `DATA.md`, `ROADMAP.md`, `pyproject.toml`, file liên quan trong `.github/`, code và test thuộc phạm vi Issue.
2. Nếu có Git metadata, kiểm tra branch và working tree; bảo toàn mọi thay đổi có sẵn của người dùng. Không dùng `git reset`, force push hoặc xóa file để làm sạch workspace.
3. Xác định Issue, acceptance criteria, file dự kiến thay đổi và lệnh kiểm tra trước khi chỉnh sửa.
4. Không tự mở rộng phạm vi sang refactor, đổi model, đổi training arguments, đổi prompt, đổi schema hoặc split dữ liệu.

Nguồn thông tin chính: `README.md` cho onboarding; `ROADMAP.md` cho tiến độ; `DATA.md` cho dữ liệu và license; `CONTRIBUTING.md` cho contributor workflow; file này cho ràng buộc của agent.

## 2. Issue, branch, commit và Pull Request

- Thay đổi không hiển nhiên phải gắn với một Issue có phạm vi và tiêu chí hoàn thành.
- Dùng branch ngắn gọn như `feat/issue-<id>-<slug>`, `fix/issue-<id>-<slug>` hoặc `docs/issue-<id>-<slug>`; không làm việc trực tiếp trên default branch nếu chưa được maintainer cho phép.
- Commit phải atomic và dùng tiền tố `feat:`, `fix:`, `test:`, `docs:`, `chore:` hoặc `ci:`. Agent không được tự commit, push, tạo release hoặc merge nếu người dùng chưa yêu cầu rõ.
- Pull Request phải liên kết Issue, giải thích thay đổi, liệt kê kiểm tra đã chạy và nêu tác động tới dataset, model, license, GPU hoặc reproducibility.
- Không force push branch dùng chung, không che giấu test fail và không trộn thay đổi ngoài phạm vi.

## 3. Python và kiểm thử

- Hỗ trợ Python 3.10+, mã nguồn UTF-8, ưu tiên standard library và dependency hiện có.
- Tuân theo PEP 8; dùng tên rõ nghĩa, `pathlib` cho đường dẫn và type hints cho interface mới. Giữ hàm nhỏ, trách nhiệm đơn nhất và tránh abstraction chưa cần thiết.
- Mỗi hành vi mới hoặc bug fix phải có test tái hiện; không sửa test chỉ để làm cho implementation sai được pass.
- Chạy tối thiểu `python -m ruff check .` và `python -m pytest -q`. Nếu không chạy được, báo chính xác command, lỗi và phần chưa được xác minh.
- Không tự động sửa hàng loạt hoặc thay đổi logic ngoài Issue. Nếu Ruff phát hiện lỗi cũ, báo riêng thay vì dùng `--fix` trên toàn repository.

## 4. Dataset và data leakage

- Tuân thủ `DATA.md`; không commit ViLexNorm, dữ liệu cá nhân, dữ liệu không rõ license hoặc output đã xử lý từ corpus có giới hạn.
- Pin và ghi dataset identifier, revision/fingerprint, số hàng mỗi split, seed và command chuẩn bị dữ liệu.
- Giữ nguyên membership `train`/`dev`/`test`. Fit trên `train`, chọn prompt/checkpoint/hyperparameters trên `dev`, chỉ đánh giá cuối trên `test` sau khi cấu hình đã khóa.
- Kiểm tra duplicate và near-duplicate giữa các split. Không xem từng lỗi test để điều chỉnh hệ thống rồi tiếp tục gọi đó là held-out evaluation.
- Fixture public phải là dữ liệu tổng hợp hoặc có quyền phân phối được chứng minh.

## 5. LoRA/QLoRA, khả năng tái lập và đánh giá

- Không chạy fine-tuning, tải base model hoặc khởi động job GPU nếu chưa có yêu cầu rõ, ngân sách tài nguyên và output directory được duyệt.
- Base model và mọi LoRA/QLoRA hyperparameter là biến nghiên cứu. Mỗi thay đổi phải có Issue, lý do, cấu hình riêng và không được gộp kết quả với run cũ.
- Mỗi run phải ghi command, commit/revision code nếu có, dataset revision/fingerprint, seed, prompt, package versions, CUDA/driver, GPU/VRAM, trainable parameters, wall time, peak VRAM, checkpoint selection và raw predictions/metrics.
- Giữ generation parameters cố định khi so sánh. Báo cáo identity và untuned baseline cùng điều kiện; nêu rõ số mẫu, split và uncertainty/limitations.
- Không gọi metric hiện tại là ERR tương thích bài báo cho tới khi phương pháp đó được triển khai, dẫn nguồn và kiểm thử bằng test vectors.

## 6. GPU, VRAM và job huấn luyện

- Trước job được phép, kiểm tra `nvidia-smi`, CUDA compatibility, VRAM còn trống, dung lượng đĩa và tiến trình GPU đang chạy.
- Không chạy đồng thời nhiều job trên cùng GPU nếu chưa được điều phối. Dùng đường dẫn output riêng và không ghi đè checkpoint của run khác.
- Bắt đầu bằng smoke test có giới hạn khi kế hoạch thí nghiệm cho phép. Nếu OOM hoặc incompatibility xảy ra, dừng và ghi log đã khử secrets; không âm thầm đổi batch size, precision, sequence length hoặc hyperparameters.
- CI luôn dùng CPU và không tải model/dataset lớn.

## 7. Secrets và artifact của model

- Token/API key chỉ được truyền qua biến môi trường hoặc secret store. Không hard-code, in ra terminal, đưa vào log, Issue, notebook hoặc config được commit.
- Khử token, đường dẫn riêng tư và nội dung dữ liệu khỏi log trước khi chia sẻ. Kiểm tra diff trước khi hoàn tất.
- Không commit `.env`, cache, checkpoint, model weights, adapter, W&B logs hoặc generated predictions có dữ liệu hạn chế. Tôn trọng `.gitignore`; nếu một artifact cần phát hành, phải có license, checksum, model card và phê duyệt riêng.

## 8. Definition of Done cho mỗi Issue

Một Issue chỉ hoàn thành khi:

- acceptance criteria được đáp ứng và không có thay đổi ngoài phạm vi;
- test phù hợp đã được thêm/cập nhật, Ruff và pytest đã chạy hoặc blocker được ghi rõ;
- tài liệu, CLI help và config liên quan nhất quán với hành vi thực tế;
- không có secret, dataset bị hạn chế, checkpoint hoặc model artifact trong diff;
- thay đổi nghiên cứu có đủ provenance và metadata reproducibility;
- diff cuối đã được tự review, Pull Request mô tả rủi ro và mọi bước chưa xác minh;
- không tuyên bố benchmark, compatibility hoặc test thành công nếu chưa có bằng chứng chạy thực tế.
