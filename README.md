# Discord Hate Speech Detection Bot

A Discord bot that uses machine learning to detect and moderate hate speech in server messages. The bot monitors messages in real-time, identifies potentially harmful content, and takes configurable actions such as flagging, deleting messages, or temporarily muting users.

<img width="655" alt="Screen Shot 2025-05-16 at 1 04 53 PM" src="https://github.com/user-attachments/assets/aea3031a-8b2d-41cf-a1ea-4c559b4dd825" />


## Features

- Real-time hate speech detection using a pre-trained machine learning model
- Configurable actions: flag, delete, or mute
- Server-specific settings with channel and role whitelisting
- Moderator notification system
- Detection statistics and analysis tools
- Admin commands for configuration and moderation

## Setup & Installation

### Prerequisites

- Python 3.9 or higher
- A Discord account and a registered application/bot
- pip (Python package manager)

### Environment Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/discord-hate-speech-bot.git
   cd discord-hate-speech-bot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the project root directory with your Discord bot token:
   ```
   DISCORD_TOKEN=your_discord_bot_token
   ```

### Creating a Discord Bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application" and give it a name
3. Navigate to the "Bot" tab and click "Add Bot"
4. Under "Privileged Gateway Intents", enable "Message Content Intent"
5. Copy the bot token and add it to your `.env` file
6. Generate an invite link from the OAuth2 URL Generator:
   - Select "bot" scope
   - Select permissions: "Read Messages/View Channels", "Send Messages", "Manage Messages", "Moderate Members"
7. Use the generated URL to invite the bot to your server

## Running the Bot

1. Make sure your virtual environment is activated
2. Run the bot:
   ```bash
   python bot.py
   ```

3. The bot will connect to Discord and start monitoring messages in servers where it has been added

## Bot Commands

- `!analyze <text>` - Analyze text for hate speech (Moderator only)
- `!stats` - Show detection statistics (Moderator only)
- `!reset_stats` - Reset detection statistics (Admin only)
- `!config` - View current server configuration (Admin only)
- `!config <setting> <value>` - Update server configuration (Admin only)

### Configuration Options

- `enabled` - Enable/disable the bot (true/false)
- `threshold` - Detection sensitivity (0.0-1.0)
- `action` - Action to take when hate speech is detected (flag/delete/mute)
- `mod_channel` - Channel ID for moderator notifications
- `whitelist_channels` - Channels to ignore
- `whitelist_roles` - Roles that bypass detection

## Running the Tests

The project includes a comprehensive test suite to ensure everything works correctly:

1. Make sure your virtual environment is activated
2. Install test dependencies:
   ```bash
   pip install pytest pytest-asyncio
   ```

3. Run all tests:
   ```bash
   pytest -xvs test_bot.py
   ```

4. Run specific test categories:
   ```bash
   # Test just the detector
   pytest -xvs tests/test_bot.py::TestHateSpeechDetector
   
   # Test server configuration
   pytest -xvs tests/test_bot.py::TestServerConfig
   
   # Test bot events
   pytest -xvs tests/test_bot.py::TestBotEvents
   
   # Test bot commands
   pytest -xvs tests/test_bot.py::TestBotCommands
   ```

5. Run a specific test:
   ```bash
   pytest -xvs test_bot.py::TestHateSpeechDetector::test_predict_hate_speech
   ```

## Model Training & Customization

The default bot uses a pre-trained model, but you can fine-tune or replace it:

1. Collect training data (labeled examples of hate speech and non-hate speech)
2. Fine-tune the model using the Hugging Face Transformers library
3. Replace the model path in the `HateSpeechDetector` class

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the pre-trained models
- [discord.py](https://github.com/Rapptz/discord.py) for the Discord API wrapper
