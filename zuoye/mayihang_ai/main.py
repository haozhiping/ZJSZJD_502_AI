import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# 引入我们刚才写的业务逻辑
from rag_service import process_and_store_pdf, chat_with_rag, get_db_stats

# 初始化 FastAPI 应用
app = FastAPI(title="企业知识大脑 RAG 接口服务")

# 确保文件上传目录存在
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# 定义对话接口的请求体格式
class ChatRequest(BaseModel):
    message: str


# ----------------- 路由定义 -----------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    负责把 templates/index.html 发送给用户的浏览器，让用户看到界面。
    """
    template_path = os.path.join("templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="前端模板 index.html 丢失")

    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/docs")
async def api_get_docs_stats():
    """
    供前端查询当前向量数据库中有多少个知识切片，用于展示在左侧边栏。
    """
    try:
        return get_db_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def api_upload_pdf(file: UploadFile = File(...)):
    """
    接收用户拖拽上传的 PDF 文件。它先利用 shutil 将文件保存在本地的 uploads/ 目录，然后呼叫 rag_service.py 去处理这个文件。
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="系统当前仅支持 PDF 格式的文件！")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # 保存上传的文件到 uploads 文件夹
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 处理 PDF (提取文本、向量化、存入 Chroma)
        chunks_count = process_and_store_pdf(file_path)

        return {
            "filename": file.filename,
            "chunks_count": chunks_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")


@app.post("/api/chat")
async def api_chat(request: ChatRequest):
    """
    接收用户的提问，呼叫 rag_service.py 去寻找答案，并将最终的 AI 回答和引用的原文片段组装成 JSON 返回给前端。
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")

    try:
        answer, sources = chat_with_rag(request.message)
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail="大模型生成或检索过程中发生异常，请检查后台日志。")


# ----------------- 启动入口 -----------------
if __name__ == "__main__":
    print("🚀 正在启动企业知识大脑后台服务...")
    print("访问地址: http://127.0.0.1:8000")
    # 启动 uvicorn 服务器
    uvicorn.run(app="main:app", host="0.0.0.0", port=8000, reload=True)