from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import os
import uuid
import re
import time
import asyncio
from typing import List, Dict, Optional
import PyPDF2
import docx
import aiofiles
# main.py-ல் இதை சேர்:

app = FastAPI(title="AI Document Q&A System", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 STATIC FOLDER MOUNT
app.mount("/static", StaticFiles(directory="static"), name="static")

# 🔥 ROOT URL → index.html
@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

# ==================== AI Models Import ====================
# main.py-ல் இந்த lines check பண்ணுங்கள்
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("⚠️ AI models not available - using keyword search")

try:
    from transformers import pipeline # type: ignore
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers pipeline loaded")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available")

# ==================== App Setup ====================


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Configuration ====================
UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Memory storage with auto-cleanup
documents_db = {}  # file_id -> document data
user_to_file = {}  # user_id -> file_id
file_upload_time = {}  # file_id -> timestamp

# Models (loaded on demand to save memory)
MODELS = {}

def get_embedding_model():
    """Lazy load embedding model"""
    if "embedding" not in MODELS and SENTENCE_TRANSFORMER_AVAILABLE:
        MODELS["embedding"] = SentenceTransformer("all-MiniLM-L6-v2") # type: ignore
    return MODELS.get("embedding")

def get_qa_model():
    """Lazy load QA model"""
    if "qa" not in MODELS and TRANSFORMERS_AVAILABLE:
        MODELS["qa"] = pipeline( # type: ignore
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=-1  # CPU
        )
    return MODELS.get("qa")

# ==================== Utility Functions ====================
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF"""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += f"[Page {len(reader.pages)}] {page_text}\n"
            return text
    except Exception as e:
        print(f"PDF Error: {e}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX"""
    try:
        doc = docx.Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"DOCX Error: {e}")
        return ""

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            print(f"TXT Error: {e}")
            return ""

def clean_text(text: str) -> str:
    """Clean text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?()-]', ' ', text)
    return text.strip()

def split_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks for embedding"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        if current_size + len(word) + 1 <= chunk_size:
            current_chunk.append(word)
            current_size += len(word) + 1
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks if chunks else [text[:chunk_size]]

def create_embeddings(chunks: List[str]):
    """Create embeddings for chunks"""
    model = get_embedding_model()
    if model and chunks:
        return model.encode(chunks, convert_to_tensor=True)
    return None

def find_relevant_chunks(question: str, chunks: List[str], embeddings) -> List[str]:
    """Find relevant chunks using semantic search"""
    if not SENTENCE_TRANSFORMER_AVAILABLE or embeddings is None:
        # Fallback: keyword matching
        question_words = set(question.lower().split())
        relevant = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            if len(question_words & chunk_words) > 0:
                relevant.append(chunk)
        return relevant[:3]
    
    try:
        model = get_embedding_model()
        question_embedding = model.encode(question, convert_to_tensor=True) # type: ignore
        scores = util.cos_sim(question_embedding, embeddings) # type: ignore
        
        # Get top 3 chunks
        top_k = min(3, len(scores[0]))
        top_indices = scores[0].argsort(descending=True)[:top_k]
        
        relevant = []
        for idx in top_indices:
            if scores[0][idx] > 0.3:  # Threshold
                relevant.append(chunks[idx])
        
        return relevant if relevant else [chunks[i] for i in top_indices[:2] if i < len(chunks)]
    except:
        return chunks[:2]

def extract_answer_ai(question: str, context: str) -> Optional[str]:
    """Extract answer using AI model"""
    if not TRANSFORMERS_AVAILABLE or not context:
        return None
    
    try:
        qa_model = get_qa_model()
        result = qa_model({
            'question': question,
            'context': context[:4000]
        }) # type: ignore
        
        if result['score'] > 0.1:
            answer = result['answer'].strip()
            if answer and len(answer) > 5:
                return answer
    except Exception as e:
        print(f"QA Error: {e}")
    
    return None

def get_best_answer(question: str, chunks: List[str], text: str) -> str:
    """Get best answer from document"""
    # Try AI first
    if TRANSFORMERS_AVAILABLE:
        context = " ".join(chunks[:3])
        answer = extract_answer_ai(question, context)
        if answer:
            return answer
    
    # Fallback to semantic or keyword search
    if SENTENCE_TRANSFORMER_AVAILABLE:
        # Use embedding search
        model = get_embedding_model()
        if model:
            chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
            question_embedding = model.encode(question, convert_to_tensor=True)
            scores = util.cos_sim(question_embedding, chunk_embeddings) # type: ignore
            
            best_idx = scores.argmax().item()
            if scores[0][best_idx] > 0.2:
                return chunks[best_idx][:500]
    
    # Final fallback: keyword matching
    question_lower = question.lower()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    best_sentence = ""
    best_score = 0
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(1 for word in question_lower.split() if word in sentence_lower)
        if score > best_score:
            best_score = score
            best_sentence = sentence
    
    if best_sentence and best_score > 0:
        return best_sentence[:500]
    
    return text[:300] if text else "No relevant information found."

# ==================== File Management ====================
async def delete_old_file(user_id: str):
    """Delete old file when user uploads new one"""
    if user_id in user_to_file:
        old_file_id = user_to_file[user_id]
        if old_file_id in documents_db:
            del documents_db[old_file_id]
        if old_file_id in file_upload_time:
            del file_upload_time[old_file_id]
        
        # Delete from disk
        for filename in os.listdir(UPLOAD_FOLDER):
            if old_file_id in filename:
                os.remove(os.path.join(UPLOAD_FOLDER, filename))
        
        print(f"🗑️ Deleted old file for user {user_id}")

async def auto_cleanup():
    """Auto delete files older than 24 hours"""
    while True:
        await asyncio.sleep(3600)  # Check every hour
        current_time = time.time()
        to_delete = []
        
        for file_id, upload_time in file_upload_time.items():
            if current_time - upload_time > 24 * 3600:  # 24 hours
                to_delete.append(file_id)
        
        for file_id in to_delete:
            if file_id in documents_db:
                del documents_db[file_id]
            if file_id in file_upload_time:
                del file_upload_time[file_id]
            
            # Remove from user mapping
            for user_id, f_id in list(user_to_file.items()):
                if f_id == file_id:
                    del user_to_file[user_id]
            
            # Delete from disk
            for filename in os.listdir(UPLOAD_FOLDER):
                if file_id in filename:
                    os.remove(os.path.join(UPLOAD_FOLDER, filename))
            
            print(f"⏰ Auto-deleted: {file_id}")

# ==================== API Endpoints ====================


@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form("user_001")
):
    """Upload and process document"""
    try:
        # Delete old file first
        await delete_old_file(user_id)
        
        # Validate file
        allowed_ext = {'pdf', 'docx', 'txt', 'doc'}
        filename = file.filename or "document"
        file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
        
        if file_ext not in allowed_ext:
            raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(allowed_ext)}")
        
        # Generate unique ID
        file_id = f"{uuid.uuid4().hex[:8]}_{user_id}.{file_ext}"
        file_path = os.path.join(UPLOAD_FOLDER, file_id)
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Extract text
        text = ""
        if file_ext == 'pdf':
            text = extract_text_from_pdf(file_path)
        elif file_ext in ['docx', 'doc']:
            text = extract_text_from_docx(file_path)
        else:
            text = extract_text_from_txt(file_path)
        
        if not text or len(text.strip()) < 10:
            raise HTTPException(400, "Could not extract text from document")
        
        # Process text
        text = clean_text(text)
        chunks = split_into_chunks(text, 300)
        embeddings = create_embeddings(chunks)
        
        # Store in database
        documents_db[file_id] = {
            'text': text,
            'chunks': chunks,
            'embeddings': embeddings,
            'filename': filename,
            'user_id': user_id,
            'file_path': file_path,
            'upload_time': time.time(),
            'size': len(text),
            'chunk_count': len(chunks)
        }
        
        # Update user mapping
        user_to_file[user_id] = file_id
        file_upload_time[file_id] = time.time()
        
        return JSONResponse({
            'success': True,
            'message': 'Document uploaded successfully',
            'file_id': file_id,
            'filename': filename,
            'preview': text[:200] + "..." if len(text) > 200 else text,
            'size': len(text),
            'chunks': len(chunks),
            'user_id': user_id,
            'note': 'File will auto-delete after 24 hours'
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.post("/ask")
async def ask_question(
    file_id: str = Form(...),
    question: str = Form(...),
    user_id: Optional[str] = Form(None)
):
    """Ask question about document"""
    try:
        if file_id not in documents_db:
            raise HTTPException(404, "Document not found")
        
        doc = documents_db[file_id]
        
        # Verify user (optional)
        if user_id and doc['user_id'] != user_id:
            raise HTTPException(403, "Access denied")
        
        # Check file age
        file_age = time.time() - doc['upload_time']
        if file_age > 24 * 3600:
            print(f"⚠️ File {file_id} is {file_age/3600:.1f} hours old")
        
        # Get answer
        answer = get_best_answer(
            question, 
            doc['chunks'], 
            doc['text']
        )
        
        # Clean answer
        answer = answer.strip()
        if len(answer) > 500:
            sentences = answer.split('.')
            if len(sentences) > 2:
                answer = '.'.join(sentences[:2]) + '.'
            else:
                answer = answer[:497] + "..."
        
        return JSONResponse({
            'success': True,
            'answer': answer,
            'question': question,
            'file_id': file_id,
            'filename': doc['filename'],
            'answer_length': len(answer),
            'file_age_hours': round(file_age / 3600, 1),
            'ai_used': SENTENCE_TRANSFORMER_AVAILABLE or TRANSFORMERS_AVAILABLE
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@app.get("/user/{user_id}")
async def get_user_document(user_id: str):
    """Get user's current document"""
    if user_id in user_to_file:
        file_id = user_to_file[user_id]
        if file_id in documents_db:
            doc = documents_db[file_id]
            file_age = time.time() - doc['upload_time']
            
            return JSONResponse({
                'user_id': user_id,
                'file_id': file_id,
                'filename': doc['filename'],
                'uploaded_ago': round(file_age / 3600, 2),
                'hours_remaining': max(0, 24 - file_age / 3600),
                'size': doc['size'],
                'chunks': doc['chunk_count']
            })
    
    return JSONResponse({
        'user_id': user_id,
        'file': None,
        'message': 'No active document'
    })

@app.delete("/delete")
async def delete_document(
    user_id: str = Form(...),
    file_id: Optional[str] = Form(None)
):
    """Delete document"""
    try:
        if file_id:
            # Delete specific file
            if file_id in documents_db:
                doc = documents_db[file_id]
                if doc['user_id'] != user_id:
                    raise HTTPException(403, "Access denied")
                
                # Delete
                del documents_db[file_id]
                if file_id in file_upload_time:
                    del file_upload_time[file_id]
                
                # Remove from user mapping
                if user_id in user_to_file and user_to_file[user_id] == file_id:
                    del user_to_file[user_id]
                
                # Delete file
                if os.path.exists(doc['file_path']):
                    os.remove(doc['file_path'])
                
                return JSONResponse({'success': True, 'message': f'Deleted {file_id}'})
            else:
                raise HTTPException(404, "File not found")
        else:
            # Delete user's file
            if user_id in user_to_file:
                file_id = user_to_file[user_id]
                if file_id in documents_db:
                    doc = documents_db[file_id]
                    
                    # Delete
                    del documents_db[file_id]
                    if file_id in file_upload_time:
                        del file_upload_time[file_id]
                    del user_to_file[user_id]
                    
                    # Delete file
                    if os.path.exists(doc['file_path']):
                        os.remove(doc['file_path'])
                    
                    return JSONResponse({'success': True, 'message': f'Deleted user {user_id}\'s file'})
            
            return JSONResponse({'success': True, 'message': 'No file to delete'})
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Delete failed: {str(e)}")

@app.get("/health")
async def health_check():
    """System health"""
    return JSONResponse({
        'status': 'healthy',
        'documents': len(documents_db),
        'users': len(user_to_file),
        'ai_models': {
            'sentence_transformers': SENTENCE_TRANSFORMER_AVAILABLE,
            'transformers': TRANSFORMERS_AVAILABLE
        },
        'auto_cleanup': 'active',
        'upload_folder': UPLOAD_FOLDER,
        'timestamp': time.time()
    })

# ==================== Startup ====================
@app.on_event("startup")
async def startup():
    """Start background tasks"""
    asyncio.create_task(auto_cleanup())
    print("🚀 AI Document Q&A System Started")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🤖 AI Models: SentenceTransformer={SENTENCE_TRANSFORMER_AVAILABLE}, Transformers={TRANSFORMERS_AVAILABLE}")

# ==================== Local Testing ====================
if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
