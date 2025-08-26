#!/usr/bin/env python3
"""
Test Suite for Priority 3: AI Session Intelligence Features
===========================================================

This test suite comprehensively tests Priority 3 features:
- Session Continuity Tracking  
- Change Impact Analysis
- Pattern Learning
- Multi-file Coordination
- Session Journaling

Coverage includes unit tests, integration tests, and MCP protocol tests.
"""

import pytest
import asyncio
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# Import Priority 3 classes and functions
try:
    from deepflow.mcp.realtime_intelligence import (
        RealTimeIntelligenceEngine,
        SessionContext,
        ChangeImpactAnalysis,
        PatternLearningData,
        MultiFileCoordination,
        SessionJournalEntry
    )
    from deepflow.mcp.server import DeepflowMCPServer
    PRIORITY3_AVAILABLE = True
except ImportError:
    PRIORITY3_AVAILABLE = False

# Skip all tests if Priority 3 dependencies are not available
pytestmark = pytest.mark.skipif(not PRIORITY3_AVAILABLE, reason="Priority 3 dependencies not available")

class TestSessionContinuityTracking:
    """Test session continuity tracking functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create real-time intelligence engine for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = RealTimeIntelligenceEngine(temp_dir, ai_awareness=True)
            yield engine
            # Cleanup
            if engine._current_session:
                engine.end_ai_session()
    
    def test_start_ai_session(self, engine):
        """Test starting a new AI session."""
        session_name = "Test Feature Implementation"
        session_description = "Implementing user authentication feature"
        session_tags = {"feature", "auth"}
        
        session_id = engine.start_ai_session(
            session_name=session_name,
            session_description=session_description,
            session_tags=session_tags
        )
        
        assert session_id is not None
        assert session_id.startswith("session_")
        assert engine._current_session is not None
        assert engine._current_session.session_name == session_name
        assert engine._current_session.session_description == session_description
        assert engine._current_session.session_tags == session_tags
        assert engine._stats['sessions_tracked'] == 1
        assert len(engine._session_journal) == 1  # Session start journal entry
    
    def test_end_ai_session(self, engine):
        """Test ending an AI session."""
        # Start a session first
        session_id = engine.start_ai_session("Test Session", "Testing session end")
        
        # Add some mock data to the session
        engine._current_session.files_modified.add("test_file.py")
        engine._current_session.changes_made.append({"test": "change"})
        
        achievements = ["Implemented feature X", "Fixed bug Y"]
        completed_session = engine.end_ai_session(achievements=achievements)
        
        assert completed_session is not None
        assert completed_session.session_id == session_id
        assert completed_session.end_time is not None
        assert completed_session.goals_achieved == achievements
        assert engine._current_session is None
        assert len(engine._session_history) == 1
        assert len(engine._session_journal) == 2  # Start + end entries
    
    def test_get_session_context(self, engine):
        """Test getting current session context."""
        # No active session
        context = engine.get_session_context()
        assert context is None
        
        # With active session
        session_id = engine.start_ai_session("Context Test", "Testing context retrieval")
        context = engine.get_session_context()
        
        assert context is not None
        assert context.session_id == session_id
        assert context.session_name == "Context Test"
    
    def test_restore_session_context(self, engine):
        """Test restoring a previous session context."""
        # Create and end a session
        original_session_id = engine.start_ai_session("Original Session", "Original description")
        engine._current_session.files_modified.add("original_file.py")
        engine._current_session.patterns_learned["test_pattern"] = {"example": "data"}
        completed_session = engine.end_ai_session(["Original goal"])
        
        # Restore the session
        success = engine.restore_session_context(original_session_id)
        
        assert success is True
        assert engine._current_session is not None
        assert engine._current_session.session_name == f"Restored: Original Session"
        assert "original_file.py" in engine._current_session.files_modified
        assert "test_pattern" in engine._current_session.patterns_learned
        assert engine._stats['session_context_restorations'] == 1
    
    def test_restore_nonexistent_session(self, engine):
        """Test attempting to restore a non-existent session."""
        success = engine.restore_session_context("nonexistent_session_123")
        assert success is False
        assert engine._current_session is None

class TestChangeImpactAnalysis:
    """Test change impact analysis functionality."""
    
    @pytest.fixture
    def engine_with_mock_graph(self):
        """Create engine with mocked dependency graph."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = RealTimeIntelligenceEngine(temp_dir, ai_awareness=True)
            
            # Mock dependency graph
            mock_graph = Mock()
            mock_graph.nodes = {
                "module_a.py": Mock(imports=["module_b.py"]),
                "module_b.py": Mock(imports=["utils.py"]),
                "test_module_a.py": Mock(imports=["module_a.py"])
            }
            engine._dependency_graph = mock_graph
            
            # Create test files
            test_dir = Path(temp_dir)
            (test_dir / "test_module_a.py").write_text("# Test file")
            (test_dir / "README.md").write_text("# Documentation")
            
            yield engine
    
    @pytest.mark.asyncio
    async def test_analyze_change_impact_basic(self, engine_with_mock_graph):
        """Test basic change impact analysis."""
        file_path = "module_b.py"
        change_type = "modification"
        
        impact_analysis = await engine_with_mock_graph.analyze_change_impact(
            file_path=file_path,
            change_type=change_type
        )
        
        assert impact_analysis.change_id is not None
        assert impact_analysis.affected_file == file_path
        assert impact_analysis.change_type == change_type
        assert impact_analysis.risk_assessment in ['low', 'medium', 'high', 'critical']
        assert 0 <= impact_analysis.impact_score <= 1.0
        assert isinstance(impact_analysis.ripple_effects, list)
        assert isinstance(impact_analysis.mitigation_suggestions, list)
        assert engine_with_mock_graph._stats['impact_analyses_performed'] == 1
    
    @pytest.mark.asyncio
    async def test_analyze_change_impact_with_dependencies(self, engine_with_mock_graph):
        """Test impact analysis with dependency impacts."""
        file_path = "utils.py"  # This is imported by module_b.py
        
        impact_analysis = await engine_with_mock_graph.analyze_change_impact(
            file_path=file_path,
            change_type="modification"
        )
        
        # Should find module_b.py as a dependency impact
        assert len(impact_analysis.dependency_impacts) > 0
        assert "module_b.py" in impact_analysis.dependency_impacts
        assert any(effect['impact_type'] == 'dependency' for effect in impact_analysis.ripple_effects)
    
    @pytest.mark.asyncio
    async def test_analyze_change_impact_with_tests(self, engine_with_mock_graph):
        """Test impact analysis detecting test files."""
        file_path = "module_a.py"
        
        impact_analysis = await engine_with_mock_graph.analyze_change_impact(
            file_path=file_path,
            change_type="modification"
        )
        
        # Should detect test_module_a.py as a test impact
        assert len(impact_analysis.test_impacts) > 0
        assert any("test_module_a.py" in test_file for test_file in impact_analysis.test_impacts)
        assert any(effect['impact_type'] == 'testing' for effect in impact_analysis.ripple_effects)
    
    @pytest.mark.asyncio
    async def test_analyze_change_impact_with_session(self, engine_with_mock_graph):
        """Test impact analysis with active session."""
        # Start a session
        session_id = engine_with_mock_graph.start_ai_session("Impact Test", "Testing impact analysis")
        
        impact_analysis = await engine_with_mock_graph.analyze_change_impact(
            file_path="test_file.py",
            change_type="addition"
        )
        
        # Should update session context
        current_session = engine_with_mock_graph.get_session_context()
        assert "test_file.py" in current_session.files_modified
        assert len(current_session.changes_made) == 1
        assert current_session.changes_made[0]['file_path'] == "test_file.py"
        assert current_session.changes_made[0]['change_type'] == "addition"

class TestPatternLearning:
    """Test pattern learning functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create engine for pattern learning tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = RealTimeIntelligenceEngine(temp_dir, ai_awareness=True)
            yield engine
    
    def test_learn_pattern_basic(self, engine):
        """Test basic pattern learning."""
        pattern_type = "naming_convention"
        pattern_data = {"style": "snake_case", "examples": ["user_id", "file_path"]}
        learned_from_files = ["module_a.py", "module_b.py"]
        
        learned_pattern = engine.learn_pattern(
            pattern_type=pattern_type,
            pattern_data=pattern_data,
            learned_from_files=learned_from_files
        )
        
        assert learned_pattern.pattern_id is not None
        assert learned_pattern.pattern_type == pattern_type
        assert learned_pattern.learned_from_files == learned_from_files
        assert learned_pattern.confidence_score > 0
        assert learned_pattern.usage_frequency == 1
        assert learned_pattern.project_specificity > 0
        assert engine._stats['patterns_learned'] == 1
        assert len(engine._pattern_learning_data) == 1
        assert pattern_type in engine._learned_patterns
    
    def test_learn_pattern_with_session(self, engine):
        """Test pattern learning with active session."""
        session_id = engine.start_ai_session("Pattern Learning Test")
        
        pattern_type = "import_style"
        pattern_data = {"style": "direct_imports", "example": "from module import function"}
        
        learned_pattern = engine.learn_pattern(
            pattern_type=pattern_type,
            pattern_data=pattern_data,
            learned_from_files=["test.py"]
        )
        
        # Should update session patterns
        current_session = engine.get_session_context()
        assert learned_pattern.pattern_id in current_session.patterns_learned
        assert current_session.patterns_learned[learned_pattern.pattern_id] == pattern_data
    
    def test_multiple_patterns_same_type(self, engine):
        """Test learning multiple patterns of the same type."""
        pattern_type = "function_naming"
        
        # Learn first pattern
        pattern1 = engine.learn_pattern(
            pattern_type=pattern_type,
            pattern_data={"style": "verb_noun", "example": "get_user"},
            learned_from_files=["file1.py"]
        )
        
        # Learn second pattern
        pattern2 = engine.learn_pattern(
            pattern_type=pattern_type,
            pattern_data={"style": "action_object", "example": "create_account"},
            learned_from_files=["file2.py"]
        )
        
        assert len(engine._learned_patterns[pattern_type]) == 2
        assert pattern1.pattern_id != pattern2.pattern_id

class TestMultiFileCoordination:
    """Test multi-file coordination functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create engine for coordination tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = RealTimeIntelligenceEngine(temp_dir, ai_awareness=True)
            yield engine
    
    def test_start_multi_file_coordination(self, engine):
        """Test starting multi-file coordination."""
        coordination_type = "refactoring"
        related_files = {"module_a.py", "module_b.py", "utils.py"}
        context = "Refactoring authentication system"
        
        coordination_id = engine.start_multi_file_coordination(
            coordination_type=coordination_type,
            related_files=related_files,
            context=context
        )
        
        assert coordination_id is not None
        assert coordination_id.startswith("coord_refactoring_")
        assert engine._stats['multi_file_coordinations_managed'] == 1
        assert len(engine._multi_file_coordinations) == 1
        
        coordination = engine._multi_file_coordinations[0]
        assert coordination.coordination_type == coordination_type
        assert coordination.related_files == related_files
        assert coordination.coordination_context == context
        assert all(status is False for status in coordination.completion_status.values())
    
    def test_update_file_coordination(self, engine):
        """Test updating file coordination progress."""
        # Start coordination
        related_files = {"file_a.py", "file_b.py"}
        coordination_id = engine.start_multi_file_coordination(
            coordination_type="feature_development",
            related_files=related_files,
            context="Adding new feature"
        )
        
        # Update progress for file_a.py
        change_details = {"completed": True, "changes": "Added new function"}
        success = engine.update_file_coordination(
            coordination_id=coordination_id,
            file_path="file_a.py",
            change_details=change_details
        )
        
        assert success is True
        
        coordination = engine._multi_file_coordinations[0]
        assert len(coordination.change_sequence) == 1
        assert coordination.change_sequence[0]['file_path'] == "file_a.py"
        assert coordination.change_sequence[0]['change_details'] == change_details
        assert coordination.completion_status["file_a.py"] is True
        assert coordination.completion_status["file_b.py"] is False
    
    def test_update_nonexistent_coordination(self, engine):
        """Test updating non-existent coordination."""
        success = engine.update_file_coordination(
            coordination_id="nonexistent_coord_123",
            file_path="test.py",
            change_details={"test": "data"}
        )
        assert success is False

class TestSessionJournaling:
    """Test session journaling functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create engine for journaling tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = RealTimeIntelligenceEngine(temp_dir, ai_awareness=True)
            yield engine
    
    def test_journal_entry_creation(self, engine):
        """Test creating journal entries."""
        entry_id = engine._add_journal_entry(
            entry_type="change",
            entry_title="Modified authentication logic",
            entry_description="Updated login validation",
            affected_files=["auth.py", "user.py"],
            ai_context="Improving security",
            outcome="Successfully updated validation"
        )
        
        assert entry_id is not None
        assert entry_id.startswith("journal_change_")
        assert len(engine._session_journal) == 1
        assert engine._stats['journal_entries_created'] == 1
        
        entry = engine._session_journal[0]
        assert entry.entry_type == "change"
        assert entry.entry_title == "Modified authentication logic"
        assert entry.affected_files == ["auth.py", "user.py"]
        assert entry.ai_context == "Improving security"
        assert entry.outcome == "Successfully updated validation"
    
    def test_session_lifecycle_journaling(self, engine):
        """Test that session lifecycle events are journaled."""
        # Start session - should create journal entry
        session_id = engine.start_ai_session("Journal Test", "Testing journaling")
        assert len(engine._session_journal) == 1
        assert engine._session_journal[0].entry_type == "session_start"
        
        # End session - should create another journal entry
        engine.end_ai_session(["Test completed"])
        assert len(engine._session_journal) == 2
        assert engine._session_journal[1].entry_type == "session_end"
    
    def test_pattern_learning_journaling(self, engine):
        """Test that pattern learning is journaled."""
        engine.start_ai_session("Pattern Test")
        
        engine.learn_pattern(
            pattern_type="test_pattern",
            pattern_data={"example": "data"},
            learned_from_files=["test.py"]
        )
        
        # Should have session_start + pattern_learned entries
        assert len(engine._session_journal) == 2
        assert engine._session_journal[1].entry_type == "pattern_learned"
        assert "Learned test_pattern Pattern" in engine._session_journal[1].entry_title

class TestSessionIntelligence:
    """Test comprehensive session intelligence functionality."""
    
    @pytest.fixture
    def engine_with_data(self):
        """Create engine with sample session data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = RealTimeIntelligenceEngine(temp_dir, ai_awareness=True)
            
            # Create sample session
            session_id = engine.start_ai_session("Intelligence Test", "Testing intelligence features")
            engine._current_session.files_modified.add("test_file.py")
            engine._current_session.changes_made.append({"test": "change"})
            engine._current_session.ai_interactions = 5
            
            # Add some sample data
            engine._change_impact_analyses.append(
                ChangeImpactAnalysis(
                    change_id="test_change",
                    affected_file="test.py",
                    change_type="modification",
                    ripple_effects=[],
                    dependency_impacts=[],
                    test_impacts=[],
                    documentation_impacts=[],
                    risk_assessment="low",
                    impact_score=0.2,
                    mitigation_suggestions=[]
                )
            )
            
            engine._pattern_learning_data.append(
                PatternLearningData(
                    pattern_id="test_pattern",
                    pattern_type="naming",
                    pattern_description="Test pattern",
                    learned_from_files=["test.py"],
                    confidence_score=0.8,
                    usage_frequency=3,
                    pattern_examples=[{"example": "data"}],
                    project_specificity=0.7
                )
            )
            
            yield engine
    
    def test_get_session_intelligence(self, engine_with_data):
        """Test getting comprehensive session intelligence."""
        intelligence = engine_with_data.get_session_intelligence(limit=10)
        
        assert 'current_session' in intelligence
        assert 'session_history' in intelligence
        assert 'impact_analyses' in intelligence
        assert 'learned_patterns' in intelligence
        assert 'multi_file_coordinations' in intelligence
        assert 'journal_entries' in intelligence
        
        # Validate current session data
        current = intelligence['current_session']
        assert current is not None
        assert current['session_name'] == "Intelligence Test"
        assert current['files_modified'] == ["test_file.py"]
        assert current['changes_made'] == 1
        assert current['ai_interactions'] == 5
        
        # Validate impact analyses
        assert len(intelligence['impact_analyses']) == 1
        impact = intelligence['impact_analyses'][0]
        assert impact['change_id'] == "test_change"
        assert impact['risk_assessment'] == "low"
        
        # Validate learned patterns
        assert len(intelligence['learned_patterns']) == 1
        pattern = intelligence['learned_patterns'][0]
        assert pattern['pattern_id'] == "test_pattern"
        assert pattern['confidence_score'] == 0.8
    
    def test_session_history_tracking(self, engine_with_data):
        """Test session history tracking."""
        # End current session and start another
        engine_with_data.end_ai_session(["Goal 1", "Goal 2"])
        engine_with_data.start_ai_session("Second Session", "Another test session")
        
        intelligence = engine_with_data.get_session_intelligence()
        
        assert len(intelligence['session_history']) == 1
        historical_session = intelligence['session_history'][0]
        assert historical_session['session_name'] == "Intelligence Test"
        assert historical_session['goals_achieved'] == 2
        assert historical_session['duration'] is not None

class TestMCPIntegration:
    """Test MCP server integration for Priority 3 features."""
    
    @pytest.fixture
    async def mcp_server(self):
        """Create MCP server for testing."""
        server = DeepflowMCPServer()
        await server.start()
        yield server
        await server.stop()
    
    @pytest.mark.asyncio
    async def test_start_ai_session_mcp(self, mcp_server):
        """Test start_ai_session MCP tool."""
        # First start real-time monitoring
        await mcp_server._handle_start_realtime_monitoring({"project_path": "."})
        
        # Test start AI session
        result = await mcp_server._handle_start_ai_session({
            "session_name": "MCP Test Session",
            "session_description": "Testing MCP integration",
            "session_tags": ["test", "mcp"]
        })
        
        assert len(result) == 1
        response_text = result[0].text
        response_data = json.loads(response_text)
        
        assert response_data['status'] == 'started'
        assert response_data['session_name'] == "MCP Test Session"
        assert response_data['session_id'] is not None
    
    @pytest.mark.asyncio
    async def test_session_context_mcp(self, mcp_server):
        """Test get_session_context MCP tool."""
        # Start monitoring and session
        await mcp_server._handle_start_realtime_monitoring({"project_path": "."})
        await mcp_server._handle_start_ai_session({
            "session_name": "Context Test",
            "session_description": "Testing context retrieval"
        })
        
        # Get session context
        result = await mcp_server._handle_get_session_context({})
        
        assert len(result) == 1
        response_data = json.loads(result[0].text)
        
        assert response_data['session_name'] == "Context Test"
        assert response_data['session_description'] == "Testing context retrieval"
        assert 'duration_seconds' in response_data
    
    @pytest.mark.asyncio
    async def test_change_impact_analysis_mcp(self, mcp_server):
        """Test analyze_change_impact MCP tool."""
        # Start monitoring
        await mcp_server._handle_start_realtime_monitoring({"project_path": "."})
        
        # Analyze change impact
        result = await mcp_server._handle_analyze_change_impact({
            "file_path": "test_file.py",
            "change_type": "modification",
            "change_details": {"description": "Updated function"}
        })
        
        assert len(result) == 1
        response_data = json.loads(result[0].text)
        
        assert response_data['affected_file'] == "test_file.py"
        assert response_data['change_type'] == "modification"
        assert response_data['risk_assessment'] in ['low', 'medium', 'high', 'critical']
        assert 'impact_score' in response_data
        assert 'mitigation_suggestions' in response_data
    
    @pytest.mark.asyncio
    async def test_session_intelligence_mcp(self, mcp_server):
        """Test get_session_intelligence MCP tool."""
        # Start monitoring and create some data
        await mcp_server._handle_start_realtime_monitoring({"project_path": "."})
        await mcp_server._handle_start_ai_session({"session_name": "Intelligence Test"})
        await mcp_server._handle_analyze_change_impact({
            "file_path": "example.py",
            "change_type": "addition"
        })
        
        # Get session intelligence
        result = await mcp_server._handle_get_session_intelligence({"limit": 20})
        
        assert len(result) == 1
        response_data = json.loads(result[0].text)
        
        assert 'current_session' in response_data
        assert 'session_history' in response_data
        assert 'impact_analyses' in response_data
        assert 'learned_patterns' in response_data
        assert 'multi_file_coordinations' in response_data
        assert 'journal_entries' in response_data
        
        # Should have current session
        assert response_data['current_session'] is not None
        assert response_data['current_session']['session_name'] == "Intelligence Test"
        
        # Should have impact analysis
        assert len(response_data['impact_analyses']) >= 1
    
    @pytest.mark.asyncio
    async def test_mcp_tools_without_monitoring(self, mcp_server):
        """Test that Priority 3 tools require real-time monitoring."""
        # Try to start session without monitoring
        result = await mcp_server._handle_start_ai_session({"session_name": "Test"})
        
        assert len(result) == 1
        assert "Real-time monitoring not available" in result[0].text

class TestPerformanceAndStress:
    """Test performance and stress scenarios for Priority 3 features."""
    
    @pytest.fixture
    def engine(self):
        """Create engine for performance tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = RealTimeIntelligenceEngine(temp_dir, ai_awareness=True)
            yield engine
    
    def test_multiple_sessions_performance(self, engine):
        """Test handling multiple sessions efficiently."""
        session_ids = []
        
        # Create and end 10 sessions quickly
        for i in range(10):
            session_id = engine.start_ai_session(f"Session {i}", f"Description {i}")
            session_ids.append(session_id)
            
            # Add some data to each session
            engine._current_session.files_modified.add(f"file_{i}.py")
            engine._current_session.changes_made.append({"change": i})
            
            engine.end_ai_session([f"Goal {i}"])
        
        # Verify all sessions are in history
        assert len(engine._session_history) == 10
        assert engine._stats['sessions_tracked'] == 10
        
        # Verify session IDs are unique
        assert len(set(session_ids)) == 10
    
    @pytest.mark.asyncio
    async def test_large_impact_analysis(self, engine):
        """Test impact analysis with large numbers of dependencies."""
        # Mock a large dependency graph
        mock_graph = Mock()
        mock_nodes = {}
        
        # Create 100 mock files with dependencies
        for i in range(100):
            file_name = f"module_{i}.py"
            mock_nodes[file_name] = Mock(imports=[f"module_{j}.py" for j in range(min(i+1, 10))])
        
        mock_graph.nodes = mock_nodes
        engine._dependency_graph = mock_graph
        
        # Analyze impact on a central file
        impact_analysis = await engine.analyze_change_impact("module_5.py", "modification")
        
        # Should handle large dependency tree efficiently
        assert impact_analysis is not None
        assert len(impact_analysis.dependency_impacts) > 0
        assert impact_analysis.impact_score >= 0
    
    def test_pattern_learning_memory_efficiency(self, engine):
        """Test that pattern learning doesn't consume excessive memory."""
        # Learn many patterns
        for i in range(100):
            engine.learn_pattern(
                pattern_type=f"type_{i % 5}",  # Reuse some types
                pattern_data={"pattern": f"pattern_{i}", "data": list(range(10))},
                learned_from_files=[f"file_{i}.py"]
            )
        
        # Verify patterns are stored efficiently
        assert len(engine._pattern_learning_data) == 100
        assert engine._stats['patterns_learned'] == 100
        
        # Check pattern type grouping
        assert len(engine._learned_patterns) <= 7  # 5 types + initial types
    
    def test_journal_entry_management(self, engine):
        """Test journal entry management with many entries."""
        engine.start_ai_session("Journal Stress Test")
        
        # Create many journal entries
        for i in range(200):
            engine._add_journal_entry(
                entry_type="test",
                entry_title=f"Entry {i}",
                entry_description=f"Test entry number {i}",
                affected_files=[f"file_{i}.py"],
                ai_context=f"Context {i}",
                outcome=f"Outcome {i}"
            )
        
        # Verify all entries are stored
        assert len(engine._session_journal) == 201  # 200 + session start
        assert engine._stats['journal_entries_created'] == 201

if __name__ == "__main__":
    pytest.main([__file__, "-v"])