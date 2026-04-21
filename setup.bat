@echo off
REM Quick Start Setup Script for Windows
REM Parkinson's Detection System

echo ==========================================
echo Parkinson's Detection System Setup
echo ==========================================
echo.

REM Check Python
echo Checking Python version...
python --version
echo.

REM Create virtual environment
set /p create_venv="Create virtual environment? (y/n): "
if /i "%create_venv%"=="y" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
    echo Activate it with: venv\Scripts\activate
    echo.
)

REM Install dependencies
set /p install_deps="Install required packages? (y/n): "
if /i "%install_deps%"=="y" (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo Dependencies installed.
    echo.
)

REM Create directories
echo Creating directory structure...
mkdir data\healthy\spiral 2>nul
mkdir data\healthy\wave 2>nul
mkdir data\healthy\handwriting 2>nul
mkdir data\parkinsons\spiral 2>nul
mkdir data\parkinsons\wave 2>nul
mkdir data\parkinsons\handwriting 2>nul
mkdir results 2>nul
mkdir patient_records 2>nul
echo Directories created.
echo.

REM Create __init__.py files
echo Creating Python package files...
type nul > models\__init__.py
type nul > utils\__init__.py
type nul > app\__init__.py
echo Package files created.
echo.

echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Place your dataset images in the data\ directories
echo 2. Train the model: python train.py --epochs 20
echo 3. Launch the app: python app\gradio_app.py
echo.
echo For detailed instructions, see README.md
echo.
pause
