# U55C Full-FPGA LLM Project - Start Here

This bundle contains the **U55C lab manual**, plus the two earlier starter packages and a new cluster-specific preparation scaffold.

1. Read `U55C_Full_FPGA_LLM_KV_Lab_Manual.pdf` or the editable `.md` version.
2. **Do not redo Labs 0-3**. Your completed work stays as-is. If you already have the original `kv_fpga_project` directory, do not overwrite it with the old starter zip.
3. Extract `Full_FPGA_Tiny_Transformer_Starter.zip` into the existing project root (it adds `full_fpga/`). The old Full-FPGA manual introduced this code; the new U55C manual continues to use it.
4. Extract `U55C_Extension_Starter.zip` into the same project root (it adds `u55c_extension/`). You may rename the directory to `u55c/` after updating paths in your scripts.
5. From `u55c_extension/` run `python3 scripts/cache_layout.py --mode report` and `python3 -m unittest discover -s tests -v`.
6. Complete Labs 4 and 5 now, before your U55C access arrives. Only after receiving credentials should you collect the installed platform information and build Vitis kernels.

The ZIP does **not** contain an already synthesized U55C transformer, a cluster allocation mechanism, or a completed HBM/XRT driver. Those are team implementation tasks. Do not assume that any `connectivity.example.cfg` or example hardware command is ready-to-run on the actual cluster until the platform and shell are verified.
