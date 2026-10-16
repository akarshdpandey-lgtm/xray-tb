# TBScan AI FastAPI backend

This backend loads `tb_densenet121_best.pth` and exposes the API expected by the TBScan AI frontend.

## API
- `GET /health`
- `POST /predict` with multipart form field named `file`
- Interactive docs: `/docs`

## Model details from the supplied README/checkpoint
- DenseNet-121
- RGB input resized to 224x224
- Normalization mean: `[0.485, 0.456, 0.406]`
- Normalization std: `[0.229, 0.224, 0.225]`
- Output index 0 = `no_tb`, index 1 = `tb`

## Deploy on Render
1. Upload the contents of this folder to a new GitHub repository.
2. In Render, create a new Web Service from that repository.
3. Render should detect `render.yaml`; use the configured build/start commands.
4. After deployment, open `https://YOUR-SERVICE.onrender.com/health` and confirm `model_loaded: true`.
5. In the TBScan AI website's API Settings, set the base URL to `https://YOUR-SERVICE.onrender.com` (no `/predict` suffix), then save/connect.
6. Test with an appropriate sample image and check Render logs if errors occur.

## Important
- The checkpoint is included in this package. Do not commit private datasets or patient-identifiable images.
- Model scores are not calibrated clinical probabilities and must not be presented as a diagnosis. This is an experimental research prototype, not a replacement for clinical assessment.
- Confirm dataset/model licensing and deployment requirements before public release.
