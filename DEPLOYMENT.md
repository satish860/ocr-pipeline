# Deployment Guide

## Railway (Recommended - Easiest)

Railway auto-deploys from your GitHub repository.

### Steps:

1. **Go to Railway**: https://railway.app

2. **Sign up with GitHub**

3. **New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose `ocr-pipeline` repository
   - Railway auto-detects the Dockerfile ✅

4. **Add Environment Variable**:
   - Go to your service → Variables
   - Add: `OPENROUTER_API_KEY` = `your_key_here`

5. **Deploy**:
   - Railway automatically builds and deploys
   - You'll get a URL like: `https://ocr-pipeline-production.up.railway.app`

6. **Test**:
   ```bash
   curl https://your-app.up.railway.app/
   ```

### Cost:
- Free tier: $5 credit/month
- After: ~$5-10/month depending on usage

---

## Alternative: Render.com

1. Go to https://render.com
2. New → Web Service
3. Connect your GitHub repo
4. Set environment: `OPENROUTER_API_KEY`
5. Deploy!

---

## Local Testing

Before deploying, test locally:

```bash
# Install dependencies
uv sync

# Run API
uv run uvicorn src.api.main:app --reload

# Test endpoint
curl -X POST http://localhost:8000/ocr \
  -F "file=@path/to/image.png"
```

---

## Later Optimization (Global Edge)

Once working, consider:
- **Fly.io**: Global edge deployment (30+ regions)
- **Cloudflare Workers**: When Containers becomes GA
- **AWS CloudFront + Lambda**: Enterprise option
