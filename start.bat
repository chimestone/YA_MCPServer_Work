@echo off
chcp 65001 >nul
echo ========================================
echo    MCP 分类器服务 - 快速启动
echo ========================================
echo.

echo [1/3] 检查依赖...
python -c "import sklearn, pandas, jieba" 2>nul
if errorlevel 1 (
    echo 依赖缺失，正在安装...
    pip install -r requirements.txt
) else (
    echo ✓ 依赖已安装
)

echo.
echo [2/3] 运行测试...
python quick_test.py
if errorlevel 1 (
    echo ✗ 测试失败，请检查错误信息
    pause
    exit /b 1
)

echo.
echo [3/3] 启动MCP服务...
echo.
echo 服务配置:
echo   - 模式: SSE (HTTP)
echo   - 地址: http://0.0.0.0:12345
echo   - 日志: logs/
echo.
echo 按 Ctrl+C 停止服务
echo ========================================
echo.

python server.py
