@echo off
REM === Edit this one line for each new video ===
set VIDEO=C:\Users\abdul\Downloads\Video Project 7.mp4

REM === Rarely need to touch these ===
set OUTPUT_DIR=C:\Users\abdul\OneDrive\Desktop\VS-Code\projects\Local-Ai-shorts-generator-out-of-local-video\output
set NUM_CLIPS=3
set VISION_MODEL=qwen2.5vl:7b
set CODE_RECT=0,170,1920,610
set WEBCAM_RECT=1390,780,530,300

REM Derive the transcript json path from the video's filename (whisper names
REM its output <basename>.json in --output_dir), so you never re-type it.
for %%F in ("%VIDEO%") do set BASENAME=%%~nF
set TRANSCRIPT=%OUTPUT_DIR%\%BASENAME%.json

echo === Step 1/2: Transcribing with whisper ===
whisper "%VIDEO%" --model medium --language English --output_format json --word_timestamps True --output_dir "%OUTPUT_DIR%"
if errorlevel 1 (
    echo Whisper failed - stopping before make_short.py.
    pause
    exit /b 1
)

echo === Step 2/2: Running make_short.py ===
python make_short.py --video "%VIDEO%" --transcript "%TRANSCRIPT%" --num-clips %NUM_CLIPS% --vision-model %VISION_MODEL% --code-rect "%CODE_RECT%" --webcam-rect "%WEBCAM_RECT%"

pause