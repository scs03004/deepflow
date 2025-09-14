# Design Documents

This folder contains all design documents for Deepflow, organized by lifecycle stage.

## 📁 Folder Structure

### 📝 **Drafts** (`/drafts/`)
**Status:** Draft  
**Purpose:** New design documents awaiting review and approval, plus templates and management tools

### 🚧 **In Progress** (`/in-progress/`)
**Status:** Approved & Implementation Started  
**Purpose:** Design documents currently being implemented

**Process:** Move here when implementation begins

### ✅ **Completed** (`/completed/`)
**Status:** Successfully Implemented  
**Purpose:** Design documents for successfully delivered features

**Process:** Move here when implementation is complete and deployed

### ❌ **Cancelled** (`/cancelled/`)
**Status:** Decided Not to Implement  
**Purpose:** Design documents for features that were decided against

**Process:** Move here when a feature is officially cancelled or deprioritized indefinitely

### 🗄️ **Archived** (`/archived/`)
**Status:** Outdated or Superseded  
**Purpose:** Old design documents that have been replaced by newer versions

**Process:** Move here when designs become outdated or are replaced by updated versions

## 🔄 Lifecycle Management

### Status Transitions
```
Drafts → In Progress → Completed
  ↓         ↓           ↓
  ↓    → Cancelled ←    ↓
  ↓                     ↓
  → → → Archived ← ← ← ←
```

### Moving Documents
When moving documents between folders:

1. **Update Status Field** in document header
2. **Complete Post-Implementation Section** if moving to completed
3. **Update file references** in related documentation
4. **Commit the move** with clear message

### Commands for Moving Files
```bash
# Move to in-progress when starting implementation
mv drafts/FEATURE_NAME_DESIGN.md in-progress/

# Move to completed when implementation finished
mv in-progress/FEATURE_NAME_DESIGN.md completed/

# Move to cancelled if decided against
mv drafts/FEATURE_NAME_DESIGN.md cancelled/

# Move old designs to archived
mv completed/OLD_FEATURE_DESIGN.md archived/
```

## 📋 Template Usage

Use `drafts/_TEMPLATE.md` as the base for all new design documents. The template includes:

- Standard structure and sections
- Post-implementation tracking
- Risk assessment framework
- Testing strategy template

## 🔍 Finding Documents

### By Status
- **Planning new features:** Check `drafts/` folder for draft designs
- **Checking implementation progress:** Look in `in-progress/` folder
- **Understanding existing features:** Check `completed/` folder
- **Learning from decisions:** Review `cancelled/` folder for context

### By Feature Area
All design documents follow naming convention: `FEATURE_NAME_DESIGN.md`

Examples:
- `UNICODE_CLEANUP_SYSTEM_DESIGN.md`
- `MCP_ENHANCEMENT_DESIGN.md`
- `WORKFLOW_ORCHESTRATION_DESIGN.md`

## 📈 Metrics & Tracking

Track these metrics for design document effectiveness:
- Time from draft to implementation start
- Implementation success rate vs design complexity
- Design document reuse and reference frequency
- Post-implementation lessons learned trends

---

**Last Updated:** January 2025  
**Process Version:** 2.0