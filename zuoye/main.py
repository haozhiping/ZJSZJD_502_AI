import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# 引入封装好的 RAG 服务
from rag_service import RAGService

app = FastAPI(title="502班AI协作作业 - Phase 1")

# 初始化 RAG 服务
# 本地 ChromaDB 数据保存在 "./chroma_data_zuoye"
rag_service = RAGService(db_path="./chroma_data_zuoye", collection_name="enterprise_brain")

# 确保上传文件夹存在
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# 定义对话请求的数据模型
class ChatRequest(BaseModel):
    message: str


# ==========================================
# 路由 1：主页，提供精美的 Web 页面
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def read_index():
    """
    读取并返回 templates/index.html 页面。
    浏览器访问 http://127.0.0.1:8000 即可看到界面。
    """
    index_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found under templates/")
    
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return html_content


# ==========================================
# 路由 2：文档上传与向量化接口
# ==========================================
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    TODO: 实现文档上传与解析向量化逻辑
    步骤指引：
    1. 校验文件后缀是否为 .pdf
    2. 将 file.file 读取到的文件流写入到本地 ./uploads/ 目录下
    3. 调用 rag_service.process_and_add_pdf(文件本地路径) 提取文本段落并添加到 ChromaDB
    4. 返回成功信息，如 {"status": "success", "filename": file.filename, "chunks_count": 写入段落数}
    """
    # 1. 验证文件类型
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="目前仅支持上传 PDF 格式的文件！")

    # 2. 拼接保存的文件路径
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # TODO: 补全文件保存逻辑
        # 提示：使用 with open(file_path, "wb") as buffer: 写入 file.file.read()，或者使用 shutil.copyfileobj
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # TODO: 调用 RAG 服务的解析与入库方法
        count = rag_service.process_and_add_pdf(file_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_count": count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")


# ==========================================
# 路由 3：智能问答接口
# ==========================================
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    TODO: 实现基于知识库的智能问答逻辑
    步骤指引：
    1. 获取 request.message 用户发送的问题
    2. 调用 rag_service.search_and_qa(question) 进行数据库检索和大模型生成回答
    3. 返回统一格式，如 {"answer": "回答内容", "sources": [{"content": "...", "distance": 0.1}, ...]}
    """
    user_query = request.message
    if not user_query.strip():
        raise HTTPException(status_code=400, detail="提问内容不能为空！")
        
    try:
        # TODO: 调用 RAG 服务的检索问答逻辑
        result = rag_service.search_and_qa(user_query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答服务出错: {str(e)}")


# ==========================================
# 路由 4：获取知识库状态统计接口
# ==========================================
@app.get("/api/docs")
async def get_docs_stats():
    """
    TODO: 获取当前向量数据库的状态（包含总段落切片数等）
    步骤指引：
    1. 调用 rag_service.get_db_stats() 获取状态
    2. 返回给前端展示
    """
    try:
        # TODO: 调用 RAG 服务获取统计数据并返回
        stats = rag_service.get_db_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


if __name__ == "__main__":
    # 启动 FastAPI 服务，监听本地 8000 端口
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
