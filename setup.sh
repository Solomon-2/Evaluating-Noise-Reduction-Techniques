#!/bin/bash

# Ensure required env vars are set
if [ -z "$HACKATIME_API_KEY" ] || [ -z "$HACKATIME_API_URL" ]; then
    echo "ERROR: Please set HACKATIME_API_KEY and HACKATIME_API_URL environment variables."
    exit 1
fi

# Set config path
CONFIG_PATH="$HOME/.wakatime.cfg"

# Write config file
cat > "$CONFIG_PATH" <<EOF
[settings]
api_url = $HACKATIME_API_URL
api_key = $HACKATIME_API_KEY
heartbeat_rate_limit_seconds = 30
EOF

echo "✅ Config file created at $CONFIG_PATH"

# Read values back (just for confirmation)
api_url=$(grep "api_url" "$CONFIG_PATH" | cut -d '=' -f2 | xargs)
api_key=$(grep "api_key" "$CONFIG_PATH" | cut -d '=' -f2 | xargs)
heartbeat_rate=$(grep "heartbeat_rate_limit_seconds" "$CONFIG_PATH" | cut -d '=' -f2 | xargs)

echo "🔗 API URL: $api_url"
echo "🔑 API Key: ${api_key:0:4}...${api_key: -4}"

# Send test heartbeat
echo "📡 Sending test heartbeat..."
time_now=$(date +%s)

json=$(cat <<EOF
[{
    "type": "file",
    "time": $time_now,
    "entity": "test.txt",
    "language": "Text"
}]
EOF
)

response=$(curl -s -X POST "$api_url/users/current/heartbeats" \
    -H "Authorization: Bearer $api_key" \
    -H "Content-Type: application/json" \
    -d "$json")

if [[ "$response" == *"entity"* ]]; then
    echo -e "\n✅ Test heartbeat sent successfully!"

    echo -e "\n\033[36m  _   _            _         _   _                "
    echo " | | | | __ _  ___| | ____ _| |_(_)_ __ ___   ___ "
    echo " | |_| |/ _\` |/ __| |/ / _\` | __| | '_ \` _ \ / _ \\"
    echo " |  _  | (_| | (__|   < (_| | |_| | | | | | |  __/"
    echo " |_| |_|\__,_|\___|_|\_\__,_|\__|_|_| |_| |_|\___|"
    echo -e "                \033[32mReady to track!\033[0m"
    echo -e "           \033[33mhttps://hackatime.hackclub.com\033[0m"
else
    echo -e "\n❌ Failed to send heartbeat."
    echo "Response:"
    echo "$response"
fi
