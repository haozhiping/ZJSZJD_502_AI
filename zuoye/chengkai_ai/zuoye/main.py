import os
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from rag_service import RAGService, VectorDBConnector, qwen_ef

# ==================== 应用初始化 ====================
app = FastAPI(
    title="企业知识大脑 - RAG 系统",
    description="基于 ChromaDB 和通义千问的智能文档问答系统",
    version="1.0.0"
)

# 静态文件和模板配置
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
templates_dir = os.path.join(BASE_DIR, "templates")
uploads_dir = r"/chengkai_ai\uploads"

app.mount("/static", StaticFiles(directory=templates_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# RAG 服务实例（全局单例）
rag_service = RAGService(
    db_path="./chroma_data",
    collection_name="my_knowledge",
    model_name="qwen-plus"
)


# ==================== 数据模型 ====================
class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str


class ChatResponse(BaseModel):
    """聊天响应模型"""
    answer: str
    sources: list
    query: str


class UploadResponse(BaseModel):
    """上传响应模型"""
    filename: str
    chunks_count: int
    total_paragraphs: int


class StatsResponse(BaseModel):
    """统计信息响应模型"""
    collection_name: str
    total_chunks: int
    db_path: str


# ==================== 路由定义 ====================
@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    上传 PDF 文档并自动向量化入库

    Args:
        file: 上传的 PDF 文件

    Returns:
        文件名、切片数量等信息
    """
    try:
        # 验证文件类型
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

        # 保存临时文件到指定上传目录
        uploads_dir = r"/chengkai_ai\uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        temp_file_path = os.path.join(uploads_dir, file.filename)

        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # 处理 PDF 并入库
        result = rag_service.upload_and_process_pdf(temp_file_path, model_family="Qwen")

        # 删除文件（临时存入，保留则可注释此行）
        os.remove(temp_file_path)

        return UploadResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    智能问答接口

    Args:
        request: 包含用户消息的请求对象

    Returns:
        AI 回答和引用来源
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="消息不能为空")

        result = rag_service.query(request.message, top_n=2, template_name="product")
        return ChatResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def index():
    """首页 - 加载前端页面"""
    try:
        html_file = os.path.join(templates_dir, "index.html")
        if not os.path.exists(html_file):
            return HTMLResponse(content=f"<h1>错误</h1><p>找不到页面文件: {html_file}</p>", status_code=404)

        with open(html_file, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        return HTMLResponse(content=f"<h1>错误</h1><p>{str(e)}</p>", status_code=500)


@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "service": "RAG API"}


@app.get("/api/docs", response_model=StatsResponse)
async def get_document_stats():
    """
    获取知识库统计信息

    Returns:
        集合名称、切片总数等统计信息
    """
    try:
        stats = rag_service.db_connector.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


# ==================== 启动入口 ====================
if __name__ == "__main__":
    # 确保 API Key 已配置
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️  警告: 环境变量 DASHSCOPE_API_KEY 未设置")
        print("请在运行前设置: set DASHSCOPE_API_KEY=your_api_key")

    print("🚀 启动企业知识大脑 RAG 系统...")
    print("📖 API 文档: http://localhost:8000/docs")
    print("💬 聊天界面: http://localhost:8000")

    uvicorn.run(
        app="main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
