#!/usr/bin/env python3
"""Developer entry point for common tasks.

Commands:
  - verify-kernels [ptx_dir] [exe_path]
  - audit-connectivity

Defaults:
  ptx_dir = logs/ptx
  exe_path = build/slime.exe (if exists)
"""

import argparse
import sys
from core import Shell, Paths

def cmd_verify_kernels(ptx_dir: str = None, exe_path: str = None) -> int:
    from pathlib import Path
    ptx = Paths.ptx_dir() if ptx_dir is None else Path(ptx_dir)
    exe = Paths.build_artifact('slime.exe') if exe_path is None else Path(exe_path)

    script = Paths.script('verify_kernels.py')
    args = [sys.executable, str(script), str(ptx)]
    if exe:
        args.append(str(exe))

    result = Shell.run(args, description="Verify kernels in build artifacts")
    return result.value if result.is_ok() else 1


def cmd_audit_connectivity(ptx_dir: str = None, exe_path: str = None, build: bool = False, allow_source_only: bool = False) -> int:
    script = Paths.script('audit_connectivity.py')
    args = [sys.executable, str(script)]
    if ptx_dir:
        args += ["--ptx-dir", ptx_dir]
    if exe_path:
        args += ["--exe", exe_path]
    if build:
        args.append("--build")
    if allow_source_only:
        args.append("--allow-source-only")

    result = Shell.run(args, description="Audit MNIST connectivity")
    return result.value if result.is_ok() else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('verify-kernels', help='Verify expected kernels in build artifacts')
    p.add_argument('--ptx-dir', default=None)
    p.add_argument('--exe', default=None)

    p2 = sub.add_parser('audit-connectivity', help='Audit MNIST-CA connectivity in hybrid lifecycle (requires compiled artifacts)')
    p2.add_argument('--ptx-dir', default=None, help='PTX/preprocessed dir (default: logs/ptx)')
    p2.add_argument('--exe', default=None, help='Executable path (default: build/slime.exe if exists)')
    p2.add_argument('--build', action='store_true', help='Run build/compile.bat before auditing')
    p2.add_argument('--allow-source-only', action='store_true', help='Allow source-only audit without compiled artifacts')

    args = parser.parse_args(argv)

    if args.cmd == 'verify-kernels':
        return cmd_verify_kernels(args.ptx_dir, args.exe)
    elif args.cmd == 'audit-connectivity':
        return cmd_audit_connectivity(args.ptx_dir, args.exe, args.build, args.allow_source_only)
    else:
        print(f"Unknown command: {args.cmd}")
        return 2


if __name__ == '__main__':
    sys.exit(main())
