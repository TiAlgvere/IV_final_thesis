# Publication Pipeline Environment

Recommended layout:

- `IV_final_thesis/.venv` is the main thesis Python environment.
- `gmesh-mcp/.venv` is only for the Gmsh MCP server.
- `.venv-1` should not be used unless it contains unique packages that are not already available in the main thesis environment.
- ElmerFEM is external native software and is not installed through `pip`.

Windows PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

Check the active Python executable:

```powershell
python -c "import sys; print(sys.executable)"
```

If ElmerFEM is installed outside `PATH`, set `ELMER_HOME` to the Elmer installation root, for example:

```powershell
$env:ELMER_HOME = "C:\ElmerFEM-gui-nompi-Windows-AMD64"
```

That is the actual local Windows installation used for this thesis workspace.

