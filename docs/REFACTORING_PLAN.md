# StarCoder + Apple Metal Orchard Integration Plan

## Objective
1. Initialize `/starcoder` as a public Git repository
2. Add `apple-metal-orchard` as a submodule
3. Replace hardcoded paths with environment variables (--repo flag)
4. Clean up unnecessary documentation files
5. Fix Metal device detection issue
6. Full regression testing

## Timeline

### Phase 1: Git Initialization & Repository Setup
- [ ] Initialize `/starcoder` as git repository
- [ ] Create appropriate .gitignore
- [ ] Add `apple-metal-orchard` as submodule at `./metal-backend`
- [ ] Rename `/starcoder` to `/git-starcoder` OR keep as `/starcoder` with proper namespace
- [ ] Create initial commit

### Phase 2: Path Refactoring
Identify all hardcoded paths and replace with `os.environ.get("REPO_ROOT")`:

**Files to update:**
- `device_backend.py` - Any hardcoded paths
- `model_trainer_metal_cuda.py` - Hardcoded dylib paths
- `orchard_bridge.py` (in submodule) - Path to libflash_attn.dylib
- `config.yaml` - Any absolute paths
- Test files - Fixture paths

**Changes:**
```python
# Before
dylib_path = "/Users/ianreitsma/projects/Apple-Metal-Orchard/experimental/kernel_lib/flashattn/libflash_attn.dylib"

# After
repo_root = os.environ.get("REPO_ROOT", os.path.dirname(__file__))
dylib_path = os.path.join(repo_root, "metal-backend/experimental/kernel_lib/flashattn/libflash_attn.dylib")
```

### Phase 3: Documentation Cleanup

**Files to remove (excessive docs):**
- AUDIT_AND_OPTIMIZATION.md
- COMMANDS.md
- COMPLETION_REPORT.txt
- DELIVERABLES.txt
- DELIVERY_*.txt
- ERRORS_FIXED_SESSION2.md
- FIXES_AND_IMPROVEMENTS.md
- FIXES_APPLIED.md
- FULL_COVERAGE_*.md
- IMPLEMENTATION_*.md
- INDEX_METAL_CUDA.md
- LONG_CONTEXT_*.md
- METAL_CUDA_INTEGRATION_SUMMARY.md
- MODEL_UPGRADE_GUIDE.md
- NEW-FILES-INDEX.md
- NEXT_STEPS.md
- OPTIMIZATION*.md
- PROJECT_SUMMARY.txt
- QUICK_START*.md
- README_*.md
- RUST_*.md
- SYSTEM_*.md
- TOP_0_01_PERCENT_GUIDE.md
- VERIFICATION_*.md
- deployment-summary.md
- file-manifest.md
- git-pipeline-guide.md

**Keep:**
- README.md (main)
- START_HERE.md (or rename to QUICKSTART.md)
- HARDWARE-OPTIMIZED.md
- config.yaml
- INSTALL.sh

### Phase 4: Fix Metal Device Detection

**Root Cause:** `_metal_available()` checks `torch.backends.mps.is_available()` but torch may not be fully initialized at import time.

**Solution:**
1. Lazy-initialize torch checks
2. Move metal availability check to `setup()` method
3. Implement fallback retry mechanism

**File:** `device_backend.py`

```python
# Add to DeviceBackend.__init__
self._torch_initialized = False

# Add method
def _ensure_torch_initialized(self):
    """Ensure torch is fully loaded before checking backends."""
    global torch
    if torch is None:
        try:
            import torch as torch_module
            torch = torch_module
            self._torch_initialized = True
        except ImportError:
            return False
    return True

# Update _metal_available
@staticmethod
def _metal_available() -> bool:
    """Check if Metal (Apple GPU) is available."""
    if torch is None:
        try:
            import torch as torch_module
            return torch_module.backends.mps.is_available() and torch_module.backends.mps.is_built()
        except Exception as e:
            logger.debug(f"Metal check failed: {e}")
            return False
    try:
        result = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        logger.debug(f"Metal available: {result}")
        return result
    except Exception as e:
        logger.debug(f"Metal check failed: {e}")
        return False
```

### Phase 5: Testing

**Test suite to run:**
1. `python -m pytest test_metal_cuda_integration.py -v`
2. `python -m pytest test_mps_smoke_training.py -v`
3. `python -m pytest test_trainer_one_step.py -v`
4. Manual verification: `python device_backend.py`

**Expected outcomes:**
- Device should be detected as 'mps' (not 'cpu')
- Metal FlashAttention should load
- Training step should execute on Metal
- VRAM should be properly estimated

## Implementation Steps

### 1. Git Setup
```bash
cd /Users/ianreitsma/projects/starcoder
git init
git config user.name "Ian Reitsma"
git config user.email "your-email@example.com"
git remote add origin <your-remote-url>

# Add submodule
git submodule add https://github.com/your-org/apple-metal-orchard.git metal-backend
git submodule update --init --recursive
```

### 2. Environment Variable Setup
Create wrapper script:
```bash
#!/bin/bash
# run_starcoder.sh
export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$@"
```

### 3. Path Updates (see files list above)

### 4. Run Full Test Suite

### 5. Commit & Push

## Key Changes Needed

### device_backend.py - Metal Detection Fix

The issue is torch isn't being properly checked at runtime. The fix involves:
1. Ensuring torch is imported BEFORE checking MPS availability
2. Adding logging to debug the detection flow
3. Implementing retry logic if first check fails

### Critical: Submodule Structure

```
/Users/ianreitsma/projects/starcoder/
├── metal-backend/                     (submodule: apple-metal-orchard)
│   ├── experimental/
│   │   └── kernel_lib/
│   │       └── flashattn/
│   │           └── libflash_attn.dylib
│   ├── docs/
│   └── ...other files...
├── device_backend.py
├── model_trainer_metal_cuda.py
├── test_metal_cuda_integration.py
└── ... other project files ...
```

## Risk Mitigation

1. **Before removing docs:** Back up to /data/docs_backup
2. **Before path updates:** Create git branch
3. **Before running tests:** Verify REPO_ROOT is set
4. **Submodule issues:** Use --recursive on clone/pull

