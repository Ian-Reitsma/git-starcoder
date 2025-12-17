#!/usr/bin/env python3
import pathlib
import py_compile
import sys

def fix_dataset_builder():
    p = pathlib.Path("dataset_builder_long_context.py")
    content = p.read_text()
    # Replace line 38
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'meta Dict[str' in line and 'field(default_factory' in line:
            lines[i] = '    meta: Dict[str, Any] = field(default_factory=dict)'
            print(f"Fixed line {i+1} in dataset_builder_long_context.py")
            break
    p.write_text('\n'.join(lines))

def fix_scheduler():
    p = pathlib.Path("training/long_context_scheduler.py")
    content = p.read_text()
    # Replace the should_include_sequence signature
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'sequence_meta Dict[str' in line and 'should_include_sequence' in line:
            lines[i] = '    def should_include_sequence(self, epoch: int, sequence_meta: Dict[str, Any]) -> bool:'
            print(f"Fixed line {i+1} in training/long_context_scheduler.py")
            break
    p.write_text('\n'.join(lines))

if __name__ == '__main__':
    print("Applying syntax fixes...")
    fix_dataset_builder()
    fix_scheduler()
    
    print("\nCompiling...")
    try:
        py_compile.compile("dataset_builder_long_context.py", doraise=True)
        py_compile.compile("training/long_context_scheduler.py", doraise=True)
        py_compile.compile("prepare_long_context.py", doraise=True)
        print("✓ All modules compiled successfully")
        sys.exit(0)
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        sys.exit(1)
