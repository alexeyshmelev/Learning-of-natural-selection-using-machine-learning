# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['app.py'],
             pathex=['C:\\HSE\\EPISTASIS\\biopack_v2'],
             binaries=[],
             datas=[('rnn_v_1_0_0.pth', '.'), ('transformer_adm_v_1_0_0.pth', '.'), ('transformer_force_v_1_0_0.pth', '.'), ('transformer_gen_v_1_0_0.pth', '.'), ('transformer_inf_v_1_0_0.pth', '.'), ('transformer_tof_v_1_0_0.pth', '.'), ('BioPack.png', '.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
for d in a.datas:
    if '_C.cp39-win_amd64.pyd' in d[0]:
        a.datas.remove(d)
        break
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='app',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False )
