#!/bin/bash
echo "Installing necessary Arch Linux native dependencies..."
sudo pacman -S --needed tesseract tesseract-data-eng poppler ffmpeg base-devel cmake inotify-tools
echo "Done."
