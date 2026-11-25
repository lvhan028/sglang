"""Record actual worker identity and loaded SGLang source; fail on source leakage."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import sys

_recorded = False


def record(force=False):
    global _recorded
    if (_recorded and not force) or not os.environ.get('SCBENCH_PROVENANCE_DIR'):
        return
    root = Path(os.environ['SCBENCH_SOURCE_ROOT']).resolve()
    modules = {name: str(Path(module.__file__).resolve())
               for name, module in tuple(sys.modules.items())
               if (name == 'sglang' or name.startswith('sglang.')) and getattr(module, '__file__', None)}
    leaked = {name: path for name, path in modules.items() if not Path(path).is_relative_to(root)}
    data = {'pid': os.getpid(), 'uid': os.getuid(), 'euid': os.geteuid(),
            'role': os.environ['SCBENCH_ROLE'], 'root': str(root), 'modules': modules,
            'leaked': leaked, 'versions': {}}
    for package in ['torch', 'triton', 'flashinfer-python', 'sgl-kernel', 'mooncake-transfer-engine']:
        try:
            data['versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            data['versions'][package] = None
    out = Path(os.environ['SCBENCH_PROVENANCE_DIR'])
    out.mkdir(exist_ok=True)
    (out / f'{data["role"]}-{os.getpid()}.json').write_text(json.dumps(data, indent=2) + '\n')
    if os.geteuid() != 0 or leaked:
        raise RuntimeError(f'Source admission failed: euid={os.geteuid()}, leaked={leaked}')
    _recorded = True
