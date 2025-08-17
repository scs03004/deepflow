# 🔄 Project Change Impact Checklist Template

Use this checklist when making changes to ensure all downstream files are updated.

## 📋 **Before Making Changes**

### 1. **Identify Change Type**
- [ ] **Core Functionality** (API, models, core modules)
- [ ] **Configuration** (config files, environment)
- [ ] **Web Interface** (HTML, CSS, JS)
- [ ] **Database Schema** (models, migrations)
- [ ] **Documentation** (README, guides)
- [ ] **Scripts/Tools** (setup, utilities)
- [ ] **Dependencies** (requirements, packages)

### 2. **Search for References**
Run these commands to find all references:
```bash
# Find file references
grep -r "filename" . --exclude-dir=.git

# Find import references  
grep -r "from.*module" . --include="*.py"

# Find configuration references
grep -r "CONFIG_VAR" . --include="*.py" --include="*.md"

# Find API endpoint references
grep -r "/api/endpoint" . --include="*.js" --include="*.html" --include="*.md"
```

## 🎯 **Change-Specific Checklists**

### **Core Functionality Changes**

When modifying main entry points, API routes, or core modules:

#### Required Updates:
- [ ] **Tests**: Update relevant test files for modified functionality
- [ ] **API Documentation**: Update API documentation
- [ ] **Integration Tests**: Verify integration tests still pass
- [ ] **Configuration**: Check if config files need updates
- [ ] **Docker**: Verify containerization compatibility

#### Verify:
- [ ] All tests still pass
- [ ] API endpoints work correctly
- [ ] Application starts without errors

### **Configuration Changes**

When modifying config files or environment variables:

#### Required Updates:
- [ ] **Documentation**: Update setup instructions
- [ ] **Docker Files**: Update environment variables
- [ ] **Setup Scripts**: Update installation scripts
- [ ] **Deployment Guide**: Update deployment documentation
- [ ] **Tests**: Update configuration tests

#### Verify:
- [ ] Application starts with new config
- [ ] Environment detection works on all platforms
- [ ] Deployment still functional

### **Database Schema Changes**

When modifying database models:

#### Required Updates:
- [ ] **Tests**: Update model tests
- [ ] **API Tests**: Update API tests for model changes
- [ ] **Migration Logic**: Add database migration if needed
- [ ] **Documentation**: Update model relationships in docs
- [ ] **Sample Data**: Update any seed data scripts

#### Verify:
- [ ] Database creates without errors
- [ ] All model relationships work
- [ ] API endpoints handle new schema
- [ ] Existing data migrates properly

### **Web Interface Changes**

When modifying frontend files:

#### Required Updates:
- [ ] **Tests**: Update frontend tests
- [ ] **API Compatibility**: Ensure frontend matches API changes
- [ ] **Mobile Testing**: Test responsive design
- [ ] **Theme Compatibility**: Verify all themes work
- [ ] **Documentation**: Update UI documentation

#### Verify:
- [ ] All themes load properly
- [ ] Responsive design maintained
- [ ] Browser console shows no errors
- [ ] Accessibility standards maintained

### **Documentation Changes**

When modifying documentation:

#### Required Updates:
- [ ] **Cross-References**: Update internal links
- [ ] **Code Examples**: Verify all code snippets work
- [ ] **File Paths**: Check all referenced paths exist
- [ ] **Version Info**: Update version numbers if applicable
- [ ] **Screenshots**: Update if UI changed

#### Verify:
- [ ] All links work (no 404s)
- [ ] Installation instructions accurate
- [ ] Examples execute successfully
- [ ] Markdown renders properly

### **Script/Tool Changes**

When modifying scripts or tools:

#### Required Updates:
- [ ] **Documentation**: Update relevant documentation
- [ ] **README**: Update installation instructions
- [ ] **Error Handling**: Ensure robust error messages
- [ ] **Cross-Platform**: Test on different platforms
- [ ] **Dependencies**: Update if new packages needed

#### Verify:
- [ ] Scripts execute without errors
- [ ] Error messages are helpful
- [ ] All platforms supported
- [ ] Documentation matches actual behavior

## 🧪 **Testing Strategy by Risk Level**

### **High-Risk Changes** (Full Test Suite)
- Configuration changes
- Database schema modifications
- Core API changes
- System architecture updates

**Required Testing:**
```bash
# Full test suite
pytest

# Integration testing
# Add project-specific integration tests

# Manual verification
# Add project-specific manual verification steps
```

### **Medium-Risk Changes** (Targeted Testing)
- Web interface updates
- New features
- Tool modifications

**Required Testing:**
```bash
# Relevant test categories
pytest tests/test_relevant_module.py

# Functional verification
# Add project-specific functional tests
```

### **Low-Risk Changes** (Minimal Testing)
- Documentation updates
- Theme changes
- Static content

**Required Testing:**
```bash
# Basic functionality
pytest tests/test_basic.py

# Visual verification
# Add project-specific visual checks
```

## 📝 **Session Log Requirements**

After completing changes, create session log:

**Required Content:**
- [ ] **Date and duration**
- [ ] **Files modified** (with line counts)
- [ ] **Tests updated/added**
- [ ] **Documentation updated**
- [ ] **Verification steps completed**
- [ ] **Any breaking changes noted**
- [ ] **Future opportunities identified**

**Template:**
```markdown
# Session Log: [Description]

**Date**: [YYYY-MM-DD]
**Duration**: [time]
**Objective**: [brief description]

## ✅ Files Modified
- `path/to/file.py` - [description] (+X/-Y lines)
- `tests/test_file.py` - [description] (+X/-Y lines)

## ✅ Tests Updated
- [test descriptions]

## ✅ Documentation Updated  
- [doc update descriptions]

## ✅ Verification Completed
- [verification steps]

## 🔮 Future Opportunities
- [potential improvements]
```

## 🚨 **Emergency Rollback Plan**

If changes break something:

1. **Check Version Control**: `git status`
2. **Review Recent Changes**: `git diff`
3. **Run Specific Tests**: Run tests for affected components
4. **Rollback if Needed**: `git checkout -- [file]`
5. **Document Issue**: Add to session log

## 💡 **Best Practices**

1. **Make Small Changes**: Easier to track and test
2. **Update Tests First**: TDD approach prevents regressions
3. **Use Search Tools**: Always search for references before changes
4. **Test Cross-Platform**: Verify compatibility across platforms
5. **Document Everything**: Update docs as you go, not after

This checklist ensures systematic change management and helps maintain project integrity across all modifications.