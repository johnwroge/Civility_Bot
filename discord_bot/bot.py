# Discord Hate Speech Detection Bot
# This implementation uses discord.py and a pre-trained ML model for hate speech detection

import os
import discord_bot.bot as bot
import discord
from discord.ext import commands
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import logging
import datetime
import json

# ML libraries
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_logs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("hate_speech_bot")

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')

# Initialize Discord bot with necessary intents
intents = discord.Intents.default()
intents.message_content = True  # Needed to read message content
bot = commands.Bot(command_prefix='!', intents=intents)

# Initialize the hate speech detection model
class HateSpeechDetector:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_roberta")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"Hate speech model loaded on {self.device}")
        
        # Statistics
        self.detection_count = 0
        self.total_messages = 0
        self.last_reset = datetime.datetime.now()
        
    async def predict(self, text):
        """
        Predict if a message contains hate speech.
        Returns probability score (0-1) where higher values indicate hate speech.
        """
        self.total_messages += 1
        
        # Handle empty messages
        if not text or text.isspace():
            return 0.0
            
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            hate_score = scores[0][1].item()  # Probability of hate speech class
        
        # Count detection if above threshold
        if hate_score > 0.7:
            self.detection_count += 1
            
        return hate_score
        
    def get_stats(self):
        """Return current detection statistics"""
        return {
            "total_messages": self.total_messages,
            "detected_hate_speech": self.detection_count,
            "detection_rate": (self.detection_count / max(1, self.total_messages)) * 100,
            "running_since": self.last_reset.isoformat()
        }
        
    def reset_stats(self):
        """Reset the detection statistics"""
        self.detection_count = 0
        self.total_messages = 0
        self.last_reset = datetime.datetime.now()

# Create the detector instance
detector = HateSpeechDetector()

# Dictionary to store server configuration
server_config = {}

# Helper functions
def load_server_config():
    """Load server configurations from disk"""
    global server_config
    try:
        with open('server_config.json', 'r') as f:
            server_config = json.load(f)
    except FileNotFoundError:
        server_config = {}
        save_server_config()
    logger.info(f"Loaded config for {len(server_config)} servers")

def save_server_config():
    """Save server configurations to disk"""
    with open('server_config.json', 'w') as f:
        json.dump(server_config, f)
    logger.info("Server configurations saved")

# Configure default settings for a server
def get_server_settings(guild_id):
    """Get settings for a specific server or create defaults"""
    guild_id = str(guild_id)  # Convert to string for JSON
    if guild_id not in server_config:
        server_config[guild_id] = {
            "enabled": True,
            "threshold": 0.7,  # Default threshold for flagging hate speech
            "action": "flag",  # Options: flag, delete, mute
            "mod_channel": None,  # Channel ID for moderator notifications
            "whitelist_channels": [],  # Channels to ignore
            "whitelist_roles": []  # Roles that bypass detection
        }
        save_server_config()
    return server_config[guild_id]

# Bot events
@bot.event
async def on_ready():
    """Called when the bot is ready and connected to Discord"""
    load_server_config()
    logger.info(f'{bot.user.name} has connected to Discord!')
    await bot.change_presence(activity=bot.Activity(
        type=bot.ActivityType.watching, 
        name="for hate speech | !help"
    ))

@bot.event
async def on_message(message):
    """Called for every message the bot can see"""
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return
        
    # Get server settings
    if message.guild:
        settings = get_server_settings(message.guild.id)
        
        # Check if bot is enabled for this server and channel is not whitelisted
        if (not settings["enabled"] or 
            str(message.channel.id) in settings["whitelist_channels"]):
            await bot.process_commands(message)
            return
            
        # Check if user has a whitelisted role
        if any(str(role.id) in settings["whitelist_roles"] for role in message.author.roles):
            await bot.process_commands(message)
            return
            
        # Analyze the message content
        hate_score = await detector.predict(message.content)
        
        # If above threshold, take action
        if hate_score > settings["threshold"]:
            logger.info(f"Detected hate speech ({hate_score:.2f}): {message.content}")
            
            # Take appropriate action based on settings
            if settings["action"] == "delete":
                try:
                    await message.delete()
                    await message.author.send(
                        f"Your message was removed for potentially containing hate speech. "
                        f"Please review the server rules."
                    )
                except bot.errors.Forbidden:
                    logger.warning(f"Could not delete message in {message.guild.name}")
                    
            elif settings["action"] == "mute":
                try:
                    # Try to mute the user (timeout)
                    timeout_duration = datetime.timedelta(minutes=10)
                    await message.author.timeout_for(timeout_duration)
                    await message.channel.send(
                        f"{message.author.mention} has been temporarily muted for using hate speech."
                    )
                except bot.errors.Forbidden:
                    logger.warning(f"Could not mute user in {message.guild.name}")
            
            else:  # Default action: flag
                await message.add_reaction("⚠️")
            
            # Notify moderators if a mod channel is set
            if settings["mod_channel"]:
                mod_channel = bot.get_channel(int(settings["mod_channel"]))
                if mod_channel:
                    embed = bot.Embed(
                        title="Hate Speech Detected",
                        description=f"**User:** {message.author.mention}\n"
                                   f"**Channel:** {message.channel.mention}\n"
                                   f"**Content:** {message.content}\n"
                                   f"**Score:** {hate_score:.2f}",
                        color=bot.Color.red()
                    )
                    embed.set_footer(text=f"Message ID: {message.id}")
                    await mod_channel.send(embed=embed)
    
    # Process commands in the message
    await bot.process_commands(message)

# Bot commands
@bot.command(name='stats')
@commands.has_permissions(manage_messages=True)
async def stats(ctx):
    """Show detection statistics"""
    stats_data = detector.get_stats()
    embed = bot.Embed(
        title="Hate Speech Detection Stats",
        color=bot.Color.blue()
    )
    embed.add_field(name="Total Messages Analyzed", value=stats_data["total_messages"])
    embed.add_field(name="Detected Hate Speech", value=stats_data["detected_hate_speech"])
    embed.add_field(name="Detection Rate", value=f"{stats_data['detection_rate']:.2f}%")
    embed.add_field(name="Running Since", value=stats_data["running_since"])
    await ctx.send(embed=embed)

@bot.command(name='reset_stats')
@commands.has_permissions(administrator=True)
async def reset_stats(ctx):
    """Reset detection statistics"""
    detector.reset_stats()
    await ctx.send("Detection statistics have been reset.")

@bot.command(name='config')
@commands.has_permissions(administrator=True)
async def config(ctx, setting=None, value=None):
    """
    Configure bot settings for this server
    Usage: !config [setting] [value]
    Settings: enabled, threshold, action, mod_channel
    """
    settings = get_server_settings(ctx.guild.id)
    
    # List current settings if no parameters
    if setting is None:
        embed = bot.Embed(
            title="Server Configuration",
            description="Current settings for this server:",
            color=bot.Color.blue()
        )
        for key, val in settings.items():
            if key == "whitelist_channels" or key == "whitelist_roles":
                if val:
                    embed.add_field(name=key, value=", ".join(str(x) for x in val))
                else:
                    embed.add_field(name=key, value="None")
            else:
                embed.add_field(name=key, value=str(val))
        await ctx.send(embed=embed)
        return
        
    # Update a specific setting
    setting = setting.lower()
    if setting not in settings:
        await ctx.send(f"Unknown setting: {setting}")
        return
        
    # Handle different setting types
    if setting == "enabled":
        if value.lower() in ("true", "yes", "on", "1"):
            settings["enabled"] = True
        elif value.lower() in ("false", "no", "off", "0"):
            settings["enabled"] = False
        else:
            await ctx.send("Value must be true/false")
            return
            
    elif setting == "threshold":
        try:
            threshold = float(value)
            if 0 <= threshold <= 1:
                settings["threshold"] = threshold
            else:
                await ctx.send("Threshold must be between 0 and 1")
                return
        except ValueError:
            await ctx.send("Threshold must be a number between 0 and 1")
            return
            
    elif setting == "action":
        if value.lower() in ("flag", "delete", "mute"):
            settings["action"] = value.lower()
        else:
            await ctx.send("Action must be 'flag', 'delete', or 'mute'")
            return
            
    elif setting == "mod_channel":
        if value.lower() == "none":
            settings["mod_channel"] = None
        else:
            # Try to get channel ID from mention or direct ID
            if value.startswith("<#") and value.endswith(">"):
                channel_id = value[2:-1]
            else:
                channel_id = value
                
            # Verify channel exists
            try:
                channel = await bot.fetch_channel(int(channel_id))
                if channel.guild.id != ctx.guild.id:
                    await ctx.send("That channel doesn't belong to this server")
                    return
                settings["mod_channel"] = channel_id
            except (bot.errors.NotFound, ValueError):
                await ctx.send("Invalid channel")
                return
                
    elif setting == "whitelist_channels" or setting == "whitelist_roles":
        # Handle adding/removing from lists
        action, item = value.split(" ", 1) if " " in value else (value, None)
        
        if action.lower() == "add" and item:
            # Extract ID from mention or use direct ID
            if item.startswith("<#") or item.startswith("<@&"):
                item_id = item[2:-1]
            else:
                item_id = item
                
            try:
                # Verify item exists
                if setting == "whitelist_channels":
                    await bot.fetch_channel(int(item_id))
                else:  # whitelist_roles
                    # Just try to parse as int, can't easily verify role exists
                    int(item_id)
                    
                if item_id not in settings[setting]:
                    settings[setting].append(item_id)
            except (bot.errors.NotFound, ValueError):
                await ctx.send(f"Invalid {setting[10:-1]}")
                return
                
        elif action.lower() == "remove" and item:
            if item.startswith("<#") or item.startswith("<@&"):
                item_id = item[2:-1]
            else:
                item_id = item
                
            if item_id in settings[setting]:
                settings[setting].remove(item_id)
            else:
                await ctx.send(f"That {setting[10:-1]} is not in the whitelist")
                return
                
        elif action.lower() == "clear":
            settings[setting] = []
            
        else:
            await ctx.send("Usage: !config whitelist_channels add/remove/clear [channel]")
            return
    
    # Save changes
    save_server_config()
    await ctx.send(f"Updated {setting} to {settings[setting]}")

@bot.command(name='analyze')
@commands.has_permissions(manage_messages=True)
async def analyze(ctx, *, text):
    """Analyze text for hate speech"""
    score = await detector.predict(text)
    settings = get_server_settings(ctx.guild.id)
    threshold = settings["threshold"]
    
    if score > threshold:
        color = bot.Color.red()
        result = "Detected hate speech"
    else:
        color = bot.Color.green()
        result = "No hate speech detected"
        
    embed = bot.Embed(
        title="Text Analysis",
        description=f"**Text:** {text}\n**Score:** {score:.4f}\n**Result:** {result}",
        color=color
    )
    await ctx.send(embed=embed)

# Error handling
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.errors.MissingPermissions):
        await ctx.send("You don't have permission to use this command.")
    elif isinstance(error, commands.errors.CommandNotFound):
        return  # Ignore command not found errors
    else:
        logger.error(f"Command error: {str(error)}")
        await ctx.send(f"An error occurred: {str(error)}")

# Start the bot
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)