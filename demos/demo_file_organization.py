#!/usr/bin/env python3
"""
File Organization Feature Demo
Demonstrates how deepflow handles messy AI-generated project structures.

This demo shows the complete file organization workflow:
1. AI generates messy code with files scattered everywhere
2. deepflow analyzes the project structure  
3. deepflow provides intelligent reorganization recommendations
4. deepflow safely applies changes with backups
"""

import tempfile
import shutil
from pathlib import Path
import sys

# Add the project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from deepflow.smart_refactoring_engine import SmartRefactoringEngine
    SMART_REFACTORING_AVAILABLE = True
except ImportError:
    SMART_REFACTORING_AVAILABLE = False
    print("Smart Refactoring Engine not available!")


def create_messy_ai_project(project_dir: Path):
    """Create a typical messy AI-generated project structure."""
    print("[DEMO] Creating messy AI-generated project structure...")
    
    # Simulate AI generating files with no organization - all in root
    messy_files = {
        # Test files scattered in root
        'test_authentication.py': '''
import unittest
from unittest.mock import Mock
from authentication import AuthManager

class TestAuthentication(unittest.TestCase):
    def setUp(self):
        self.auth_manager = AuthManager()
    
    def test_login_success(self):
        result = self.auth_manager.login("user@test.com", "password123")
        self.assertTrue(result['success'])
    
    def test_login_failure(self):
        result = self.auth_manager.login("bad@test.com", "wrongpass")
        self.assertFalse(result['success'])

if __name__ == '__main__':
    unittest.main()
''',
        
        'user_model_test.py': '''
import pytest
from user_model import User, UserProfile

def test_user_creation():
    user = User("john@example.com", "John Doe")
    assert user.email == "john@example.com"
    assert user.name == "John Doe"

def test_user_profile():
    profile = UserProfile(user_id=1, bio="Software Developer")
    assert profile.user_id == 1
    assert profile.bio == "Software Developer"
''',
        
        'integration_test.py': '''
import requests
import json
from flask import Flask

def test_api_endpoints():
    """Test all API endpoints work correctly."""
    base_url = "http://localhost:5000"
    
    # Test user registration
    response = requests.post(f"{base_url}/api/register", json={
        "email": "test@example.com",
        "password": "testpass123"
    })
    assert response.status_code == 201
''',
        
        # Model files in root
        'user_model.py': '''
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import sqlalchemy as sa
from sqlalchemy.orm import declarative_base

Base = declarative_base()

@dataclass
class User(Base):
    __tablename__ = 'users'
    
    id: int = sa.Column(sa.Integer, primary_key=True)
    email: str = sa.Column(sa.String(255), unique=True, nullable=False)
    name: str = sa.Column(sa.String(255), nullable=False)
    created_at: datetime = sa.Column(sa.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'created_at': self.created_at.isoformat()
        }

@dataclass 
class UserProfile(Base):
    __tablename__ = 'user_profiles'
    
    id: int = sa.Column(sa.Integer, primary_key=True)
    user_id: int = sa.Column(sa.Integer, sa.ForeignKey('users.id'))
    bio: Optional[str] = sa.Column(sa.Text)
    avatar_url: Optional[str] = sa.Column(sa.String(512))
''',
        
        'product_model.py': '''
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional
import sqlalchemy as sa
from sqlalchemy.orm import relationship
from user_model import Base

@dataclass
class Product(Base):
    __tablename__ = 'products'
    
    id: int = sa.Column(sa.Integer, primary_key=True)
    name: str = sa.Column(sa.String(255), nullable=False)
    description: Optional[str] = sa.Column(sa.Text)
    price: Decimal = sa.Column(sa.Numeric(10, 2), nullable=False)
    category_id: int = sa.Column(sa.Integer, sa.ForeignKey('categories.id'))
    
    category = relationship("Category", back_populates="products")

class Category(Base):
    __tablename__ = 'categories'
    
    id: int = sa.Column(sa.Integer, primary_key=True)
    name: str = sa.Column(sa.String(100), nullable=False)
    products = relationship("Product", back_populates="category")
''',
        
        # View files in root
        'user_view.py': '''
from flask import render_template, request, jsonify, session
from user_model import User, UserProfile
from authentication import require_auth

def render_user_profile(user_id):
    """Render user profile page."""
    user = User.query.get_or_404(user_id)
    profile = UserProfile.query.filter_by(user_id=user_id).first()
    
    return render_template('user_profile.html', 
                          user=user, 
                          profile=profile)

@require_auth
def edit_user_profile():
    """Handle user profile editing."""
    if request.method == 'POST':
        user_id = session['user_id']
        profile_data = request.get_json()
        
        # Update profile logic here
        return jsonify({'success': True, 'message': 'Profile updated'})
    
    return render_template('edit_profile.html')
''',
        
        'product_view.py': '''
from flask import render_template, request, jsonify
from product_model import Product, Category
from sqlalchemy.orm import sessionmaker

def render_product_list():
    """Render product listing page with filtering."""
    category_id = request.args.get('category')
    search_term = request.args.get('search', '')
    
    query = Product.query
    
    if category_id:
        query = query.filter(Product.category_id == category_id)
    
    if search_term:
        query = query.filter(Product.name.contains(search_term))
    
    products = query.all()
    categories = Category.query.all()
    
    return render_template('products.html', 
                          products=products,
                          categories=categories)

def render_product_detail(product_id):
    """Render individual product detail page."""
    product = Product.query.get_or_404(product_id)
    return render_template('product_detail.html', product=product)
''',
        
        # Controller files in root  
        'authentication_controller.py': '''
from flask import Blueprint, request, jsonify, session
from werkzeug.security import check_password_hash, generate_password_hash
from user_model import User
import jwt
import datetime

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/api/register', methods=['POST'])
def register():
    """User registration endpoint."""
    data = request.get_json()
    
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Email and password required'}), 400
    
    existing_user = User.query.filter_by(email=data['email']).first()
    if existing_user:
        return jsonify({'error': 'Email already registered'}), 409
    
    hashed_password = generate_password_hash(data['password'])
    new_user = User(email=data['email'], 
                   name=data.get('name', ''),
                   password_hash=hashed_password)
    
    # Save to database logic here
    return jsonify({'message': 'User registered successfully'}), 201

@auth_bp.route('/api/login', methods=['POST'])
def login():
    """User login endpoint."""
    data = request.get_json()
    
    user = User.query.filter_by(email=data.get('email')).first()
    
    if user and check_password_hash(user.password_hash, data.get('password')):
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, 'secret_key')
        
        return jsonify({'token': token, 'user': user.to_dict()})
    
    return jsonify({'error': 'Invalid credentials'}), 401
''',
        
        'product_api_controller.py': '''
from flask import Blueprint, request, jsonify
from product_model import Product, Category
from authentication import require_auth
import json

api_bp = Blueprint('api', __name__)

@api_bp.route('/api/products', methods=['GET'])
def get_products():
    """Get products with filtering and pagination."""
    page = request.args.get('page', 1, type=int)
    category = request.args.get('category')
    search = request.args.get('search', '')
    
    query = Product.query
    
    if category:
        query = query.filter(Product.category_id == category)
    
    if search:
        query = query.filter(Product.name.contains(search))
    
    products = query.paginate(page=page, per_page=20, error_out=False)
    
    return jsonify({
        'products': [p.to_dict() for p in products.items],
        'total': products.total,
        'pages': products.pages,
        'current_page': page
    })

@api_bp.route('/api/products', methods=['POST'])
@require_auth
def create_product():
    """Create new product."""
    data = request.get_json()
    
    product = Product(
        name=data['name'],
        description=data.get('description'),
        price=data['price'],
        category_id=data['category_id']
    )
    
    # Save product logic here
    return jsonify(product.to_dict()), 201
''',
        
        # Utility files scattered around
        'string_utils.py': '''
import re
from typing import Optional

def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')

def truncate_text(text: str, max_length: int, suffix: str = '...') -> str:
    """Truncate text to specified length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format currency amount."""
    symbols = {'USD': '$', 'EUR': '€', 'GBP': '£'}
    symbol = symbols.get(currency, currency)
    return f"{symbol}{amount:.2f}"
''',
        
        'email_helper.py': '''
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional
import os

class EmailService:
    def __init__(self):
        self.smtp_host = os.getenv('SMTP_HOST', 'localhost')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = os.getenv('SMTP_USER')
        self.smtp_pass = os.getenv('SMTP_PASS')
    
    def send_email(self, to_emails: List[str], subject: str, 
                   body: str, html_body: Optional[str] = None):
        """Send email to recipients."""
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.smtp_user
        msg['To'] = ', '.join(to_emails)
        
        text_part = MIMEText(body, 'plain')
        msg.attach(text_part)
        
        if html_body:
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)
        
        # SMTP sending logic here
        return True

def send_welcome_email(user_email: str, user_name: str):
    """Send welcome email to new users."""
    email_service = EmailService()
    
    subject = "Welcome to Our Platform!"
    body = f"Hi {user_name},\\n\\nWelcome to our platform!"
    
    return email_service.send_email([user_email], subject, body)
''',
        
        'file_upload_helper.py': '''
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from PIL import Image
import uuid

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \\
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file, upload_folder: str, filename: str = None):
    """Save uploaded file to disk."""
    if file and allowed_file(file.filename):
        if not filename:
            filename = secure_filename(file.filename)
        
        # Add unique identifier to prevent conflicts
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
        
        filepath = os.path.join(upload_folder, unique_filename)
        file.save(filepath)
        return unique_filename
    
    return None

def resize_image(image_path: str, max_width: int = 800, max_height: int = 600):
    """Resize image while maintaining aspect ratio."""
    with Image.open(image_path) as img:
        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        img.save(image_path, optimize=True, quality=85)
''',
        
        # Config files in root
        'config_development.py': '''
import os

class DevelopmentConfig:
    DEBUG = True
    TESTING = False
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.getenv('DEV_DATABASE_URL', 'sqlite:///dev_app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Security
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'jwt-dev-secret')
    
    # Email configuration
    MAIL_SERVER = os.getenv('MAIL_SERVER', 'localhost')
    MAIL_PORT = int(os.getenv('MAIL_PORT', '587'))
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
    
    # File upload
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
''',
        
        'settings_production.py': '''
import os

class ProductionConfig:
    DEBUG = False
    TESTING = False
    
    # Database - Use PostgreSQL in production
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
    if SQLALCHEMY_DATABASE_URI and SQLALCHEMY_DATABASE_URI.startswith('postgres://'):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace('postgres://', 'postgresql://', 1)
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }
    
    # Security - Must be set via environment variables
    SECRET_KEY = os.getenv('SECRET_KEY')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
    
    if not SECRET_KEY or not JWT_SECRET_KEY:
        raise ValueError("SECRET_KEY and JWT_SECRET_KEY must be set in production")
    
    # CORS settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '').split(',')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
''',
        
        # Script files in root
        'data_migration_script.py': '''
#!/usr/bin/env python3
"""
Data migration script for moving from legacy system.
Run this script to migrate user data and product catalog.
"""

import sys
import json
from pathlib import Path
from user_model import User, UserProfile
from product_model import Product, Category

def migrate_users_from_json(json_file_path: str):
    """Migrate users from legacy JSON format."""
    print(f"Migrating users from {json_file_path}...")
    
    with open(json_file_path, 'r') as f:
        legacy_users = json.load(f)
    
    migrated_count = 0
    
    for legacy_user in legacy_users:
        # Create new User object
        user = User(
            email=legacy_user['email_address'],  # Different field name in legacy
            name=f"{legacy_user['first_name']} {legacy_user['last_name']}",
        )
        
        # Create profile if bio exists
        if legacy_user.get('biography'):
            profile = UserProfile(
                user_id=user.id,
                bio=legacy_user['biography']
            )
        
        migrated_count += 1
        print(f"Migrated user: {user.email}")
    
    print(f"Migration completed. {migrated_count} users migrated.")

def migrate_products_from_csv(csv_file_path: str):
    """Migrate products from CSV format."""
    import csv
    
    print(f"Migrating products from {csv_file_path}...")
    
    with open(csv_file_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            product = Product(
                name=row['product_name'],
                description=row['description'],
                price=float(row['price']),
                category_id=int(row['category_id'])
            )
            
            print(f"Migrated product: {product.name}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python data_migration_script.py <users_json_file> [products_csv_file]")
        sys.exit(1)
    
    users_file = sys.argv[1]
    migrate_users_from_json(users_file)
    
    if len(sys.argv) > 2:
        products_file = sys.argv[2]
        migrate_products_from_csv(products_file)
    
    print("All migrations completed successfully!")
''',
        
        'database_setup_script.py': '''
#!/usr/bin/env python3
"""
Database setup and initialization script.
Creates tables and sets up initial data.
"""

from user_model import Base, User, UserProfile
from product_model import Product, Category
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

def create_database_tables(database_url: str):
    """Create all database tables."""
    print("Creating database tables...")
    
    engine = sa.create_engine(database_url)
    Base.metadata.create_all(engine)
    
    print("Database tables created successfully!")

def seed_initial_data(database_url: str):
    """Seed database with initial data."""
    print("Seeding initial data...")
    
    engine = sa.create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Create default categories
    categories = [
        Category(name="Electronics"),
        Category(name="Books"),
        Category(name="Clothing"),
        Category(name="Home & Garden"),
        Category(name="Sports & Outdoors")
    ]
    
    for category in categories:
        session.add(category)
    
    # Create admin user
    admin_user = User(
        email="admin@example.com",
        name="System Administrator"
    )
    session.add(admin_user)
    
    session.commit()
    print("Initial data seeded successfully!")

if __name__ == '__main__':
    import os
    
    database_url = os.getenv('DATABASE_URL', 'sqlite:///app.db')
    
    create_database_tables(database_url)
    seed_initial_data(database_url)
    
    print("Database setup completed!")
''',
        
        # Files with inconsistent naming patterns
        'UserServiceClass.py': '''  # PascalCase - inconsistent!
class UserService:
    """Service class for user-related operations."""
    
    def __init__(self, db_session):
        self.db = db_session
    
    def get_user_by_email(self, email: str):
        """Get user by email address."""
        return self.db.query(User).filter(User.email == email).first()
    
    def create_user_profile(self, user_id: int, profile_data: dict):
        """Create user profile."""
        profile = UserProfile(user_id=user_id, **profile_data)
        self.db.add(profile)
        self.db.commit()
        return profile
    
    def update_user_info(self, user_id: int, updates: dict):
        """Update user information."""
        user = self.db.query(User).get(user_id)
        if user:
            for key, value in updates.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            self.db.commit()
        return user
''',
        
        'api-response-handler.py': '''  # kebab-case - inconsistent!
from flask import jsonify
from typing import Any, Dict, Optional

def success_response(data: Any, message: str = "Success", status_code: int = 200):
    """Create standardized success response."""
    return jsonify({
        'success': True,
        'message': message,
        'data': data
    }), status_code

def error_response(message: str, error_code: str = None, status_code: int = 400):
    """Create standardized error response."""
    response_data = {
        'success': False,
        'message': message
    }
    
    if error_code:
        response_data['error_code'] = error_code
    
    return jsonify(response_data), status_code

def validation_error_response(validation_errors: Dict[str, str]):
    """Create response for validation errors."""
    return jsonify({
        'success': False,
        'message': 'Validation failed',
        'validation_errors': validation_errors
    }), 422
''',
        
        # Files that should stay in root
        'main.py': '''
#!/usr/bin/env python3
"""
Main application entry point.
This file should stay in the root directory.
"""

from flask import Flask
from authentication_controller import auth_bp
from product_api_controller import api_bp

def create_app():
    """Create and configure Flask application."""
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
''',
        
        'app.py': '''
"""
Alternative application entry point.
This file should also stay in the root directory.
"""

from main import create_app

app = create_app()

if __name__ == '__main__':
    app.run()
'''
    }
    
    # Create all the messy files
    for filename, content in messy_files.items():
        file_path = project_dir / filename
        file_path.write_text(content.strip(), encoding='utf-8')
    
    print(f"[DEMO] Created {len(messy_files)} files in a messy structure")
    print("       Files are scattered in root with inconsistent naming patterns")
    print("       This is typical of AI-generated projects without organization")
    return messy_files


def demonstrate_file_organization():
    """Demonstrate the complete file organization workflow."""
    if not SMART_REFACTORING_AVAILABLE:
        print("ERROR: Smart Refactoring Engine not available!")
        return
    
    # Create temporary directory for demo
    demo_dir = Path(tempfile.mkdtemp(prefix="deepflow_file_org_demo_"))
    print(f"[DEMO] Working in temporary directory: {demo_dir}")
    
    try:
        # Step 1: Create messy AI-generated project
        messy_files = create_messy_ai_project(demo_dir)
        print()
        
        # Step 2: Analyze the messy structure
        print("=" * 60)
        print("STEP 2: ANALYZING MESSY PROJECT STRUCTURE")
        print("=" * 60)
        
        engine = SmartRefactoringEngine(demo_dir)
        analysis = engine.analyze_file_organization(check_patterns=True)
        
        print(f"[ANALYSIS] Project Structure Score: {analysis.project_structure_score:.1f}/100")
        print(f"[ANALYSIS] Total files analyzed: {analysis.current_structure['total_files']}")
        print(f"[ANALYSIS] Root directory files: {analysis.current_structure.get('depth_distribution', {}).get(0, 0)}")
        print()
        
        # Show root clutter issues
        if analysis.root_clutter_files:
            print(f"[ISSUES] Root Clutter Detected: {len(analysis.root_clutter_files)} files")
            print("Files that should be moved to subdirectories:")
            for clutter in analysis.root_clutter_files[:10]:  # Show first 10
                print(f"  • {clutter['file_name']} -> {clutter['suggested_directory']}/")
                print(f"    Reason: {clutter['reason']} (confidence: {clutter['confidence']:.0%})")
            if len(analysis.root_clutter_files) > 10:
                print(f"  ... and {len(analysis.root_clutter_files) - 10} more files")
            print()
        
        # Show naming inconsistencies
        if analysis.naming_inconsistencies:
            print(f"[ISSUES] Naming Inconsistencies: {len(analysis.naming_inconsistencies)} files")
            print("Files with inconsistent naming patterns:")
            patterns = {}
            for issue in analysis.naming_inconsistencies:
                patterns[issue['expected_pattern']] = patterns.get(issue['expected_pattern'], 0) + 1
            
            if patterns:
                dominant_pattern = max(patterns, key=patterns.get)
                print(f"  Project mainly uses: {dominant_pattern}")
                
            for issue in analysis.naming_inconsistencies[:5]:  # Show first 5
                print(f"  • {issue['file_path']} ({issue['current_pattern']}) -> {issue['suggested_name']}")
            if len(analysis.naming_inconsistencies) > 5:
                print(f"  ... and {len(analysis.naming_inconsistencies) - 5} more files")
            print()
        
        # Show recommended directory structure
        if analysis.suggested_directories:
            print("[RECOMMENDATIONS] Suggested Directory Structure:")
            for suggestion in analysis.suggested_directories:
                priority_tag = "[HIGH]" if suggestion['priority'] == 'high' else "[MED]"
                print(f"  {priority_tag} {suggestion['directory_name']}/")
                print(f"    • Purpose: {suggestion['description']}")
                print(f"    • Files to organize: {suggestion['file_count']}")
            print()
        
        # Show high-level recommendations
        if analysis.organization_recommendations:
            print("[RECOMMENDATIONS] Organization Actions:")
            for rec in analysis.organization_recommendations:
                priority_tag = "[HIGH]" if rec['priority'] == 'high' else "[MED]" if rec['priority'] == 'medium' else "[LOW]"
                print(f"  {priority_tag} {rec['action']}")
                print(f"    • {rec['description']}")
                print(f"    • Affects {rec['affected_files']} files")
            print()
        
        # Step 3: Show dry-run organization
        print("=" * 60)
        print("STEP 3: DRY-RUN FILE ORGANIZATION")
        print("=" * 60)
        
        dry_run_results = engine.organize_files(analysis, dry_run=True, backup=True)
        
        print(f"[DRY RUN] Changes Preview:")
        print(f"  • Directories to create: {len(dry_run_results['directories_created'])}")
        print(f"  • Files to move: {len(dry_run_results['files_moved'])}")
        print(f"  • Files to rename: {len(dry_run_results['files_renamed'])}")
        print(f"  • Total changes: {dry_run_results['changes_applied']}")
        print()
        
        if dry_run_results['directories_created']:
            print("[DRY RUN] Directories to create:")
            for directory in dry_run_results['directories_created']:
                print(f"  • {directory}")
            print()
        
        if dry_run_results['files_moved']:
            print("[DRY RUN] Files to move (high confidence only):")
            for move in dry_run_results['files_moved'][:10]:  # Show first 10
                print(f"  • {move['from']} -> {move['to']}")
            if len(dry_run_results['files_moved']) > 10:
                print(f"  ... and {len(dry_run_results['files_moved']) - 10} more moves")
            print()
        
        if dry_run_results['files_renamed']:
            print("[DRY RUN] Files to rename (very high confidence only):")
            for rename in dry_run_results['files_renamed']:
                print(f"  • {rename['from']} -> {rename['to']}")
            print()
        
        # Step 4: Apply actual organization (safely)
        print("=" * 60)
        print("STEP 4: APPLYING FILE ORGANIZATION (WITH BACKUPS)")
        print("=" * 60)
        
        actual_results = engine.organize_files(analysis, dry_run=False, backup=True)
        
        if actual_results['success']:
            print(f"[SUCCESS] File organization completed!")
            print(f"  • Directories created: {len([d for d in actual_results['directories_created'] if not d.startswith('[DRY RUN]')])}")
            print(f"  • Files moved: {len([f for f in actual_results['files_moved'] if not f['from'].startswith('[DRY RUN]')])}")
            print(f"  • Files renamed: {len([f for f in actual_results['files_renamed'] if not f['from'].startswith('[DRY RUN]')])}")
            print(f"  • Total changes applied: {actual_results['changes_applied']}")
            print()
            
            if actual_results['directories_created']:
                print("[APPLIED] Directories created:")
                for directory in actual_results['directories_created']:
                    if not directory.startswith('[DRY RUN]'):
                        print(f"  • {directory}/")
                        # Show what files are in this directory
                        dir_path = demo_dir / directory
                        if dir_path.exists():
                            files_in_dir = list(dir_path.glob('*.py'))
                            if files_in_dir:
                                print(f"    Files: {', '.join(f.name for f in files_in_dir[:3])}")
                                if len(files_in_dir) > 3:
                                    print(f"           ... and {len(files_in_dir) - 3} more")
                print()
        else:
            print(f"[ERROR] File organization failed!")
            if actual_results['errors']:
                for error in actual_results['errors']:
                    print(f"  • {error}")
            print()
        
        # Step 5: Analyze improved structure
        print("=" * 60)
        print("STEP 5: ANALYZING IMPROVED PROJECT STRUCTURE")
        print("=" * 60)
        
        # Re-analyze the project after organization
        improved_analysis = engine.analyze_file_organization(check_patterns=True)
        
        print(f"[IMPROVEMENT] Structure Score: {analysis.project_structure_score:.1f} -> {improved_analysis.project_structure_score:.1f}")
        print(f"[IMPROVEMENT] Root clutter: {len(analysis.root_clutter_files)} -> {len(improved_analysis.root_clutter_files)} files")
        print(f"[IMPROVEMENT] Naming issues: {len(analysis.naming_inconsistencies)} -> {len(improved_analysis.naming_inconsistencies)} files")
        print()
        
        # Show final directory structure
        print("[FINAL] Project Directory Structure:")
        for item in sorted(demo_dir.rglob('*')):
            if item.is_file() and item.suffix == '.py':
                relative_path = item.relative_to(demo_dir)
                depth = len(relative_path.parts) - 1
                indent = "  " * depth
                
                if depth == 0:
                    print(f"{indent}[FILE] {item.name} (root)")
                else:
                    parent_dir = relative_path.parts[0]
                    print(f"{indent}[FILE] {item.name}")
        print()
        
        # Step 6: Demonstrate AI workflow integration
        print("=" * 60)
        print("STEP 6: AI WORKFLOW INTEGRATION DEMO")
        print("=" * 60)
        
        print("[AI WORKFLOW] How to use in Claude Code:")
        print("1. analyze_file_organization project_path='.'")
        print("   -> Analyze current project structure and get recommendations")
        print()
        print("2. organize_files dry_run=true project_path='.'")
        print("   -> Preview what changes would be made (safe preview)")
        print()
        print("3. organize_files apply_changes=true backup=true project_path='.'")
        print("   -> Apply the organization changes with backups")
        print()
        
        print("[AI BENEFITS] This feature helps with:")
        print("• AI generates messy code -> deepflow organizes it cleanly")
        print("• Scattered files -> intelligent directory structure")
        print("• Inconsistent naming -> standardized patterns")
        print("• Root clutter -> proper file organization")
        print("• Manual organization -> automated with safety checks")
        print()
        
        # Success summary
        print("=" * 60)
        print("[SUCCESS] FILE ORGANIZATION DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("Key achievements:")
        score_improvement = improved_analysis.project_structure_score - analysis.project_structure_score
        print(f"[OK] Structure score improved by {score_improvement:.1f} points")
        print(f"[OK] Organized {len(analysis.root_clutter_files)} files from root clutter")
        print(f"[OK] Applied {actual_results['changes_applied']} organization changes")
        print(f"[OK] Created {len([d for d in actual_results['directories_created'] if not d.startswith('[DRY RUN]')])} directories")
        print(f"[OK] All changes applied safely with backup protection")
        print()
        
        print("This demonstrates how deepflow solves AI coding file organization problems!")
        print("Files scattered by AI are now properly organized with intelligent recommendations.")
    
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary directory
        if demo_dir.exists():
            shutil.rmtree(demo_dir)
            print(f"[CLEANUP] Removed temporary demo directory: {demo_dir}")


if __name__ == '__main__':
    print("=" * 70)
    print("[FILE ORG]  DEEPFLOW FILE ORGANIZATION FEATURE DEMO")
    print("   Solving AI-Generated Project Structure Problems")
    print("=" * 70)
    print()
    
    demonstrate_file_organization()