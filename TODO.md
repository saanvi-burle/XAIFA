# Fix Upload 500 Errors - TODO

## Issues Found
1. Model Upload: Wrong field name (`file` instead of `model_file`), missing required fields
2. Dataset Upload: Wrong field name (`file` instead of `dataset_file`), missing `dataset_format`, labels sent as text instead of file
3. Proxy port mismatch (8002 vs 8000)
4. CORS port mismatch (5173 vs 3000)

## Steps

- [x] Fix `frontend/src/components/Upload.tsx` - Add all required form fields, fix field names, convert labels to file blob
- [x] Fix `frontend/vite.config.ts` - Change proxy target to port 8000
- [x] Fix `backend/app/core/config.py` - Add port 3000 to CORS origins
- [ ] Test upload functionality (run backend + frontend)
