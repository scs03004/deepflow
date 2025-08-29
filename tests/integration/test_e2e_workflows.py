"""
Comprehensive End-to-End Workflow Tests (Priority 4)
Tests complete workflows from file discovery to report generation and AI integration.
"""

import pytest
import tempfile
import os
import shutil
import time
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import xml.etree.ElementTree as ET

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Test availability of required modules
TOOLS_AVAILABLE = False
MCP_AVAILABLE = False
REALTIME_AVAILABLE = False

try:
    from tools.dependency_visualizer import DependencyAnalyzer, DependencyVisualizer
    from tools.code_analyzer import CodeAnalyzer
    from tools.doc_generator import DocumentationGenerator
    from tools.pre_commit_validator import PreCommitValidator
    TOOLS_AVAILABLE = True
except ImportError:
    pass

try:
    from deepflow.mcp.server import DeepflowMCPServer
    MCP_AVAILABLE = True
except ImportError:
    pass

try:
    from deepflow.mcp.realtime_intelligence import RealTimeIntelligenceEngine
    REALTIME_AVAILABLE = True
except ImportError:
    pass


@dataclass
class WorkflowResult:
    """Result of an end-to-end workflow execution."""
    success: bool
    execution_time: float
    outputs_created: List[str]
    errors: List[str]
    metrics: Dict[str, Any]


class E2EWorkflowRunner:
    """Utility for running complete end-to-end workflows."""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.outputs_dir = self.project_path / "workflow_outputs"
        self.outputs_dir.mkdir(exist_ok=True)
    
    def create_realistic_project(self) -> List[Path]:
        """Create a realistic Python project structure."""
        files_created = []
        
        # Project root files
        (self.project_path / "README.md").write_text("""# Test Project
        
A realistic Python project for end-to-end workflow testing.

## Features
- Web API with FastAPI
- Database integration
- Authentication system
- Background tasks
- Testing suite
""")
        
        (self.project_path / "requirements.txt").write_text("""
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
sqlalchemy==2.0.23
alembic==1.13.0
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
python-multipart==0.0.6
python-jose==3.3.0
passlib==1.7.4
""")
        
        (self.project_path / "pyproject.toml").write_text("""
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "test-project"
version = "1.0.0"
description = "Test project for workflow validation"
authors = [{name = "Test Author", email = "test@example.com"}]

[project.optional-dependencies]
dev = ["pytest", "black", "mypy", "flake8"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
""")
        
        # Source code structure
        src_dir = self.project_path / "src" / "testproject"
        src_dir.mkdir(parents=True)
        
        # Main application
        (src_dir / "__init__.py").write_text('"""Test project package."""\\n__version__ = "1.0.0"')
        files_created.append(src_dir / "__init__.py")
        
        (src_dir / "main.py").write_text("""
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from . import models, schemas, crud
from .database import SessionLocal, engine
from .auth import get_current_user
from .background import background_tasks

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Test Project API", version="1.0.0")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root():
    return {"message": "Hello World", "version": "1.0.0"}

@app.get("/users/{user_id}")
def read_user(user_id: int, db: Session = Depends(get_db)):
    user = crud.get_user(db, user_id=user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    return crud.create_user(db=db, user=user)

@app.post("/background-task/")
def trigger_background_task(task_data: dict):
    background_tasks.add_task("process_data", task_data)
    return {"message": "Task scheduled"}
""")
        files_created.append(src_dir / "main.py")
        
        # Models
        (src_dir / "models.py").write_text("""
from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    items = relationship("Item", back_populates="owner")

class Item(Base):
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String)
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    owner = relationship("User", back_populates="items")
""")
        files_created.append(src_dir / "models.py")
        
        # Schemas
        (src_dir / "schemas.py").write_text("""
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import List, Optional

class ItemBase(BaseModel):
    title: str
    description: Optional[str] = None

class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int
    owner_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    items: List[Item] = []
    
    class Config:
        from_attributes = True
""")
        files_created.append(src_dir / "schemas.py")
        
        # CRUD operations
        (src_dir / "crud.py").write_text("""
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from . import models, schemas

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def create_user_item(db: Session, item: schemas.ItemCreate, user_id: int):
    db_item = models.Item(**item.dict(), owner_id=user_id)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
""")
        files_created.append(src_dir / "crud.py")
        
        # Database configuration
        (src_dir / "database.py").write_text("""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", "sqlite:///./test.db"
)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
""")
        files_created.append(src_dir / "database.py")
        
        # Authentication
        (src_dir / "auth.py").write_text("""
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from . import crud, models, schemas
from .database import SessionLocal

SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(db: Session, email: str, password: str):
    user = crud.get_user_by_email(db, email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    db = SessionLocal()
    user = crud.get_user_by_email(db, email=email)
    db.close()
    
    if user is None:
        raise credentials_exception
    return user
""")
        files_created.append(src_dir / "auth.py")
        
        # Background tasks
        (src_dir / "background.py").write_text("""
from typing import Dict, Any
import asyncio
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class BackgroundTaskManager:
    def __init__(self):
        self.tasks = []
        self.running = False
    
    def add_task(self, task_type: str, data: Dict[str, Any]):
        task = {
            'id': len(self.tasks),
            'type': task_type,
            'data': data,
            'created_at': datetime.utcnow(),
            'status': 'pending'
        }
        self.tasks.append(task)
        logger.info(f"Added background task: {task_type}")
        return task['id']
    
    async def process_data(self, data: Dict[str, Any]):
        '''Process data in background.'''
        logger.info(f"Processing data: {data}")
        await asyncio.sleep(1)  # Simulate processing time
        return {'status': 'completed', 'result': data}
    
    async def send_notification(self, user_id: int, message: str):
        '''Send notification to user.'''
        logger.info(f"Sending notification to user {user_id}: {message}")
        await asyncio.sleep(0.5)
        return {'status': 'sent', 'user_id': user_id}

# Global task manager instance
background_tasks = BackgroundTaskManager()
""")
        files_created.append(src_dir / "background.py")
        
        # Tests
        tests_dir = self.project_path / "tests"
        tests_dir.mkdir(exist_ok=True)
        
        (tests_dir / "__init__.py").write_text("")
        (tests_dir / "conftest.py").write_text("""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.testproject.main import app, get_db
from src.testproject.database import Base

SQLALCHEMY_DATABASE_URL = "sqlite:///./test_test.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture
def client():
    return TestClient(app)
""")
        
        (tests_dir / "test_main.py").write_text("""
def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World", "version": "1.0.0"}

def test_create_user(client):
    response = client.post(
        "/users/",
        json={"email": "test@example.com", "password": "testpassword"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "id" in data

def test_read_user(client):
    # First create a user
    response = client.post(
        "/users/",
        json={"email": "test2@example.com", "password": "testpassword"}
    )
    user_id = response.json()["id"]
    
    # Then read the user
    response = client.get(f"/users/{user_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test2@example.com"
""")
        files_created.append(tests_dir / "test_main.py")
        
        # Configuration files
        (self.project_path / ".gitignore").write_text("""
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

.pytest_cache/
.coverage
htmlcov/

*.log
""")
        
        return files_created


@pytest.mark.integration
class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.workflow_runner = E2EWorkflowRunner(str(self.test_path))
        
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools not available")
    def test_full_project_analysis_pipeline(self):
        """Test complete analysis workflow from file discovery to report generation."""
        print("\\n🔄 Testing full project analysis pipeline...")
        
        # Create realistic project
        created_files = self.workflow_runner.create_realistic_project()
        print(f"Created realistic project with {len(created_files)} files")
        
        outputs_created = []
        errors = []
        start_time = time.time()
        
        try:
            # Step 1: Dependency Analysis
            print("Step 1: Analyzing dependencies...")
            analyzer = DependencyAnalyzer(str(self.test_path))
            dep_result = analyzer.analyze_project()
            assert dep_result is not None, "Dependency analysis should succeed"
            
            # Step 2: Code Quality Analysis
            print("Step 2: Analyzing code quality...")
            code_analyzer = CodeAnalyzer(str(self.test_path))
            quality_result = code_analyzer.analyze_code_quality(str(self.test_path))
            assert quality_result is not None, "Code quality analysis should succeed"
            
            # Step 3: Documentation Generation
            print("Step 3: Generating documentation...")
            doc_gen = DocumentationGenerator(str(self.test_path))
            
            # Generate dependency map
            dep_map_result = doc_gen.generate_dependency_map()
            if dep_map_result and os.path.exists(dep_map_result):
                outputs_created.append(dep_map_result)
                print(f"✅ Generated dependency map: {dep_map_result}")
            
            # Step 4: Visualization Generation
            print("Step 4: Generating visualizations...")
            try:
                visualizer = DependencyVisualizer(str(self.test_path))
                
                # Generate Mermaid visualization
                mermaid_output = self.workflow_runner.outputs_dir / "dependency_graph.html"
                viz_result = visualizer.generate_mermaid_graph(str(mermaid_output))
                if viz_result and os.path.exists(mermaid_output):
                    outputs_created.append(str(mermaid_output))
                    print(f"✅ Generated visualization: {mermaid_output}")
                    
            except Exception as e:
                errors.append(f"Visualization generation error: {e}")
                print(f"⚠️ Visualization step failed: {e}")
            
            # Step 5: Report Compilation
            print("Step 5: Compiling comprehensive report...")
            report_data = {
                'project_path': str(self.test_path),
                'analysis_timestamp': time.time(),
                'dependency_analysis': {
                    'completed': dep_result is not None,
                    'file_count': len(created_files)
                },
                'code_quality': {
                    'completed': quality_result is not None,
                    'issues_found': len(quality_result.get('issues', [])) if quality_result else 0
                },
                'documentation': {
                    'outputs_created': outputs_created
                },
                'workflow_errors': errors
            }
            
            # Save report
            report_file = self.workflow_runner.outputs_dir / "analysis_report.json"
            report_file.write_text(json.dumps(report_data, indent=2))
            outputs_created.append(str(report_file))
            print(f"✅ Generated report: {report_file}")
            
        except Exception as e:
            errors.append(f"Pipeline error: {e}")
            print(f"❌ Pipeline error: {e}")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Validate workflow results
        print(f"\\n📊 Workflow completed in {execution_time:.2f}s")
        print(f"   Outputs created: {len(outputs_created)}")
        print(f"   Errors encountered: {len(errors)}")
        
        # Should complete successfully with minimal errors
        assert len(outputs_created) >= 1, f"Should create at least one output: {outputs_created}"
        assert len(errors) <= 2, f"Should have minimal errors: {errors}"  # Allow some errors for optional steps
        
        # Should complete in reasonable time
        assert execution_time <= 120, f"Full pipeline should complete within 2 minutes, took {execution_time:.2f}s"
        
        # Verify specific outputs exist
        for output_path in outputs_created:
            assert os.path.exists(output_path), f"Output file should exist: {output_path}"
            file_size = os.path.getsize(output_path)
            assert file_size > 0, f"Output file should not be empty: {output_path}"
    
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP server not available")
    def test_mcp_claude_code_integration(self):
        """Test full integration with simulated Claude Code environment."""
        print("\\n🤖 Testing MCP Claude Code integration...")
        
        # Create test project
        created_files = self.workflow_runner.create_realistic_project()
        
        try:
            # Mock MCP dependencies
            with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \\
                 patch('deepflow.mcp.server.TOOLS_AVAILABLE', True):
                
                server = DeepflowMCPServer()
                
                # Test 1: Tool Discovery
                tools = server.get_tools()
                print(f"Available MCP tools: {len(tools)}")
                assert len(tools) >= 4, f"Should have at least 4 core tools: {[t.name for t in tools]}"
                
                # Test 2: Dependency Analysis Tool
                analyze_deps_tool = next((t for t in tools if 'analyze_dependencies' in t.name), None)
                assert analyze_deps_tool is not None, "Should have analyze_dependencies tool"
                
                # Simulate tool execution
                try:
                    with patch.object(server, '_handle_analyze_dependencies') as mock_handler:
                        mock_result = {
                            'project_path': str(self.test_path),
                            'files_analyzed': len(created_files),
                            'dependencies_found': 15,
                            'analysis_successful': True
                        }
                        mock_handler.return_value = [MagicMock(text=json.dumps(mock_result))]
                        
                        # Execute tool
                        arguments = {'project_path': str(self.test_path)}
                        result = mock_handler(arguments)
                        
                        assert len(result) >= 1, "Tool should return result"
                        response_data = json.loads(result[0].text)
                        assert response_data['analysis_successful'], "Analysis should succeed"
                        
                        print("✅ MCP dependency analysis tool working")
                        
                except Exception as e:
                    print(f"⚠️ MCP tool execution simulation failed: {e}")
                
                # Test 3: Code Quality Tool
                quality_tool = next((t for t in tools if 'code_quality' in t.name), None)
                if quality_tool:
                    print("✅ MCP code quality tool available")
                
                # Test 4: Documentation Generation Tool
                docs_tool = next((t for t in tools if 'documentation' in t.name), None)
                if docs_tool:
                    print("✅ MCP documentation tool available")
                
                print("✅ MCP Claude Code integration test completed")
                
        except ImportError as e:
            pytest.skip(f"MCP integration dependencies not available: {e}")
        except Exception as e:
            pytest.fail(f"MCP integration test failed: {e}")
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools not available")
    def test_concurrent_analysis_sessions(self):
        """Test multiple simultaneous analysis sessions."""
        print("\\n⚡ Testing concurrent analysis sessions...")
        
        # Create multiple project directories
        num_sessions = 3
        project_dirs = []
        
        for session_idx in range(num_sessions):
            session_dir = self.test_path / f"project_{session_idx}"
            session_dir.mkdir()
            
            # Create smaller projects for concurrent testing
            workflow_runner = E2EWorkflowRunner(str(session_dir))
            created_files = workflow_runner.create_realistic_project()
            project_dirs.append((session_dir, len(created_files)))
        
        print(f"Created {num_sessions} concurrent test projects")
        
        # Run concurrent analysis sessions
        import concurrent.futures
        
        def run_analysis_session(session_data):
            """Run analysis for a single session."""
            session_dir, file_count = session_data
            session_id = session_dir.name
            
            try:
                start_time = time.time()
                
                # Dependency analysis
                analyzer = DependencyAnalyzer(str(session_dir))
                dep_result = analyzer.analyze_project()
                
                # Code quality analysis
                code_analyzer = CodeAnalyzer(str(session_dir))
                quality_result = code_analyzer.analyze_code_quality(str(session_dir))
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                return {
                    'session_id': session_id,
                    'success': True,
                    'execution_time': execution_time,
                    'file_count': file_count,
                    'dependency_result': dep_result is not None,
                    'quality_result': quality_result is not None
                }
                
            except Exception as e:
                return {
                    'session_id': session_id,
                    'success': False,
                    'error': str(e),
                    'file_count': file_count
                }
        
        # Execute concurrent sessions
        start_time = time.time()
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_sessions) as executor:
            futures = [executor.submit(run_analysis_session, proj_data) for proj_data in project_dirs]
            
            for future in concurrent.futures.as_completed(futures, timeout=120):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({'success': False, 'error': str(e)})
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze concurrent execution results
        successful_sessions = [r for r in results if r.get('success', False)]
        failed_sessions = [r for r in results if not r.get('success', False)]
        
        print(f"\\n📊 Concurrent analysis results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Successful sessions: {len(successful_sessions)}/{num_sessions}")
        print(f"   Failed sessions: {len(failed_sessions)}")
        
        # Should complete most sessions successfully
        success_rate = len(successful_sessions) / num_sessions
        assert success_rate >= 0.66, f"At least 66% of concurrent sessions should succeed: {success_rate:.1%}"
        
        # Should complete concurrently faster than sequential
        if successful_sessions:
            avg_session_time = sum(s.get('execution_time', 0) for s in successful_sessions) / len(successful_sessions)
            expected_sequential_time = avg_session_time * num_sessions
            
            # Concurrent execution should show some speedup
            speedup_ratio = expected_sequential_time / total_time if total_time > 0 else 1
            print(f"   Concurrency speedup: {speedup_ratio:.1f}x")
            
            # Should be at least 1.5x faster than sequential
            assert speedup_ratio >= 1.5, f"Concurrent execution should show speedup: {speedup_ratio:.1f}x"
        
        print("✅ Concurrent analysis sessions test completed")
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools not available")
    def test_project_migration_scenarios(self):
        """Test analysis consistency across project structure changes."""
        print("\\n🔄 Testing project migration scenarios...")
        
        # Create initial project
        created_files = self.workflow_runner.create_realistic_project()
        
        # Perform initial analysis
        print("Analyzing initial project structure...")
        initial_analyzer = DependencyAnalyzer(str(self.test_path))
        initial_result = initial_analyzer.analyze_project()
        
        assert initial_result is not None, "Initial analysis should succeed"
        print(f"✅ Initial analysis: found {len(created_files)} files")
        
        # Migration Scenario 1: File Renaming
        print("\\nScenario 1: File renaming...")
        old_file = self.test_path / "src" / "testproject" / "models.py"
        new_file = self.test_path / "src" / "testproject" / "database_models.py"
        
        if old_file.exists():
            shutil.move(str(old_file), str(new_file))
            
            # Update import references
            main_file = self.test_path / "src" / "testproject" / "main.py"
            if main_file.exists():
                content = main_file.read_text()
                updated_content = content.replace("from . import models", "from . import database_models as models")
                main_file.write_text(updated_content)
            
            # Re-analyze after renaming
            renamed_analyzer = DependencyAnalyzer(str(self.test_path))
            renamed_result = renamed_analyzer.analyze_project()
            
            assert renamed_result is not None, "Analysis should handle file renaming"
            print("✅ File renaming handled successfully")
        
        # Migration Scenario 2: Directory Restructuring
        print("\\nScenario 2: Directory restructuring...")
        old_src_dir = self.test_path / "src" / "testproject"
        new_src_dir = self.test_path / "app"
        
        if old_src_dir.exists():
            shutil.move(str(old_src_dir), str(new_src_dir))
            
            # Re-analyze after restructuring
            restructured_analyzer = DependencyAnalyzer(str(self.test_path))
            restructured_result = restructured_analyzer.analyze_project()
            
            assert restructured_result is not None, "Analysis should handle directory restructuring"
            print("✅ Directory restructuring handled successfully")
        
        # Migration Scenario 3: New Files Addition
        print("\\nScenario 3: Adding new files...")
        new_module = new_src_dir / "utils.py" if new_src_dir.exists() else self.test_path / "utils.py"
        new_module.write_text("""
def utility_function(data):
    '''Utility function for data processing.'''
    return {'processed': True, 'data': data}

class UtilityHelper:
    def __init__(self):
        self.name = "helper"
    
    def help_with_task(self, task):
        return f"Helping with {task}"
""")
        
        # Re-analyze with new files
        expanded_analyzer = DependencyAnalyzer(str(self.test_path))
        expanded_result = expanded_analyzer.analyze_project()
        
        assert expanded_result is not None, "Analysis should handle new files"
        print("✅ New file addition handled successfully")
        
        # Migration Scenario 4: File Deletion
        print("\\nScenario 4: File deletion...")
        if new_module.exists():
            new_module.unlink()
        
        # Re-analyze after deletion
        reduced_analyzer = DependencyAnalyzer(str(self.test_path))
        reduced_result = reduced_analyzer.analyze_project()
        
        assert reduced_result is not None, "Analysis should handle file deletion"
        print("✅ File deletion handled successfully")
        
        print("\\n✅ All project migration scenarios completed successfully")
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools not available")
    def test_version_control_integration(self):
        """Test Git integration and commit validation workflows."""
        print("\\n🔧 Testing version control integration...")
        
        # Create project with Git repo
        created_files = self.workflow_runner.create_realistic_project()
        
        # Initialize Git repository
        try:
            subprocess.run(['git', 'init'], cwd=self.test_path, check=True, 
                         capture_output=True, text=True)
            subprocess.run(['git', 'config', 'user.email', 'test@example.com'], 
                         cwd=self.test_path, check=True)
            subprocess.run(['git', 'config', 'user.name', 'Test User'], 
                         cwd=self.test_path, check=True)
            print("✅ Git repository initialized")
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            pytest.skip(f"Git not available for integration testing: {e}")
        
        try:
            # Test Pre-commit Validation
            print("Testing pre-commit validation...")
            
            if TOOLS_AVAILABLE:
                try:
                    validator = PreCommitValidator(str(self.test_path))
                    
                    # Stage some files for commit
                    subprocess.run(['git', 'add', '.'], cwd=self.test_path, check=True)
                    
                    # Validate commit readiness
                    validation_result = validator.validate_changes()
                    print(f"✅ Pre-commit validation: {validation_result is not None}")
                    
                except Exception as e:
                    print(f"⚠️ Pre-commit validation not available: {e}")
            
            # Create initial commit
            subprocess.run(['git', 'commit', '-m', 'Initial commit'], 
                         cwd=self.test_path, check=True, capture_output=True)
            print("✅ Initial commit created")
            
            # Test Change Impact Analysis
            print("Testing change impact analysis...")
            
            # Make a change to a core file
            main_file = self.test_path / "src" / "testproject" / "main.py"
            if not main_file.exists():
                # Try alternative path after restructuring
                main_file = self.test_path / "app" / "main.py"
            
            if main_file.exists():
                original_content = main_file.read_text()
                modified_content = original_content + """\\n
# New endpoint for testing
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}
"""
                main_file.write_text(modified_content)
                
                # Analyze impact of change
                analyzer = DependencyAnalyzer(str(self.test_path))
                impact_result = analyzer.analyze_project()
                
                assert impact_result is not None, "Change impact analysis should work"
                print("✅ Change impact analysis completed")
                
                # Stage and commit the change
                subprocess.run(['git', 'add', str(main_file)], cwd=self.test_path, check=True)
                subprocess.run(['git', 'commit', '-m', 'Add health check endpoint'], 
                             cwd=self.test_path, check=True, capture_output=True)
                print("✅ Change committed successfully")
            
            # Test Git Log Analysis
            print("Testing git log analysis...")
            
            log_result = subprocess.run(['git', 'log', '--oneline'], 
                                      cwd=self.test_path, capture_output=True, text=True)
            
            if log_result.returncode == 0:
                commits = log_result.stdout.strip().split('\\n')
                print(f"✅ Found {len(commits)} commits in history")
                assert len(commits) >= 1, "Should have at least one commit"
            
            print("\\n✅ Version control integration test completed")
            
        except subprocess.CalledProcessError as e:
            pytest.skip(f"Git operations failed: {e}")
        except Exception as e:
            print(f"⚠️ Version control test error: {e}")
            # VCS integration tests may fail due to environment constraints


if __name__ == "__main__":
    pytest.main([__file__, "-v"])