# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules
ROOT = Path(SPECPATH).resolve()
datas=[]; binaries=[]; hiddenimports=[]
for pkg in ['numpy','pandas','matplotlib','scipy','sklearn','joblib','openpyxl','PIL','h5py','pykrige','tensorflow','keras']:
    try:
        d,b,h=collect_all(pkg); datas+=d; binaries+=b; hiddenimports+=h
    except Exception as exc: print('[WARN]',pkg,exc)
try:
    d,b,h=collect_all('torch'); datas+=d; binaries+=b; hiddenimports+=h
except Exception as exc: print('[WARN] torch',exc)
hiddenimports += collect_submodules('core') + collect_submodules('backend') + collect_submodules('adapters')
for src,dst in [('config.json','.'),('README_CN.md','.'),('backend','backend'),('adapters','adapters'),('teacher_models','teacher_models'),('pretrained_models','pretrained_models'),('external_models','external_models'),('models','models'),('wing_geometry','wing_geometry')]:
    p=ROOT/src
    if p.exists(): datas.append((str(p),dst))
a=Analysis(['main.py'],pathex=[str(ROOT)],binaries=binaries,datas=datas,hiddenimports=hiddenimports,hookspath=[],hooksconfig={},runtime_hooks=[],excludes=['torch.testing._internal','matplotlib.tests','numpy.tests','pandas.tests','scipy.tests','sklearn.tests'],noarchive=False,optimize=0)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='WingStrainMonitor',debug=False,bootloader_ignore_signals=False,strip=False,upx=False,console=False,disable_windowed_traceback=False,argv_emulation=False,target_arch=None,codesign_identity=None,entitlements_file=None)
coll=COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,upx_exclude=[],name='WingStrainMonitor')
