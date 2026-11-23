@echo off
set "UV_CACHE_DIR=C:\Users\vanmu\copperbelt-mineral-prospectivity\.runtime_cache\uv"
set "UV_TOOL_DIR=C:\Users\vanmu\copperbelt-mineral-prospectivity\.runtime_cache\uv-tools"
set "UV_OFFLINE=1"
if not exist "C:\Users\vanmu\copperbelt-mineral-prospectivity\.runtime_cache\qgis_mcp" mkdir "C:\Users\vanmu\copperbelt-mineral-prospectivity\.runtime_cache\qgis_mcp"
set "QGIS_MCP_LOG_FILE=C:\Users\vanmu\copperbelt-mineral-prospectivity\.runtime_cache\qgis_mcp\server.log"
uvx --from https://github.com/nkarasiak/qgis-mcp/archive/refs/tags/v0.15.1.zip qgis-mcp-server
