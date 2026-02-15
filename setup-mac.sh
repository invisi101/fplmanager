#!/bin/bash
# Creates Gaffer.app in /Applications that launches FPL Gaffer Brain

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="/Applications/Gaffer.app"

echo "Setting up Gaffer.app..."
echo "Project directory: $PROJECT_DIR"

# Create venv if it doesn't exist
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$PROJECT_DIR/.venv"
    echo "Installing dependencies..."
    "$PROJECT_DIR/.venv/bin/pip3" install -r "$PROJECT_DIR/requirements.txt"
else
    echo "Virtual environment already exists."
fi

# Build the .app bundle
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

# Copy icon if it exists
if [ -f "$PROJECT_DIR/Gaffer.icns" ]; then
    cp "$PROJECT_DIR/Gaffer.icns" "$APP_DIR/Contents/Resources/Gaffer.icns"
fi

# Info.plist
cat > "$APP_DIR/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleExecutable</key>
	<string>Gaffer</string>
	<key>CFBundleName</key>
	<string>Gaffer</string>
	<key>CFBundleIdentifier</key>
	<string>com.fplmanager.gaffer</string>
	<key>CFBundleVersion</key>
	<string>1.0</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleIconFile</key>
	<string>Gaffer</string>
	<key>LSUIElement</key>
	<true/>
</dict>
</plist>
PLIST

# Launcher script with the user's actual project path
cat > "$APP_DIR/Contents/MacOS/Gaffer" << LAUNCHER
#!/bin/bash
PROJECT_DIR="$PROJECT_DIR"
cd "\$PROJECT_DIR"
source .venv/bin/activate

# Kill any existing instance on port 9875
lsof -ti:9875 | xargs kill -9 2>/dev/null

# Start Flask and open browser once ready
python3 -m src.app &
SERVER_PID=\$!

# Wait for server to be ready (up to 30s)
for i in \$(seq 1 60); do
    if curl -s http://127.0.0.1:9875 >/dev/null 2>&1; then
        open http://127.0.0.1:9875
        break
    fi
    sleep 0.5
done

# Keep running until server exits
wait \$SERVER_PID
LAUNCHER

chmod +x "$APP_DIR/Contents/MacOS/Gaffer"

echo ""
echo "Gaffer.app installed to /Applications"
echo "You can now launch it from Spotlight, Launchpad, or your Applications folder."
