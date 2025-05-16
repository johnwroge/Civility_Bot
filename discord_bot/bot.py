# Discord Hate Speech Detection Bot
# This implementation uses discord.py and a pre-trained ML model for hate speech detection

import os
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
        self.tokenizer = AutoTokenizer.from_pretrained("tomh/toxigen_roberta")
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
            scores = torch.sigmoid(outputs.logits)
            hate_score = scores[0][1].item()  # Assuming class 1 is "toxic"

        
        # Count detection if above threshold
        if hate_score > 0.7:
            self.detection_count += 1
        print("John W", text, hate_score, text.isspace())
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
    # The classification threshold determines how confidently the model must predict hate speech
    # before triggering a moderation action. While the original ToxiGen paper used a threshold of 0.7 
    # to select only the most toxic samples for training (discarding ~2/3 of generated examples), 
    # this value is too strict for real-time moderation, where subtle but harmful content should still be flagged.
    #
    # Recommended moderation threshold:
    # - 0.3 to 0.4: good balance between catching explicit and implicit hate
    # - >0.7: extremely confident, can be used for harsher actions (e.g. delete/mute)
    #
    # We use a lower threshold (default: 0.3) to increase recall and catch more potentially harmful messages.

    if guild_id not in server_config:
        server_config[guild_id] = {
            "enabled": True,
            "threshold": 0.3,  # Default threshold for flagging hate speech
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
    logger.info(f"Bot is running. Logged in as {bot.user.name} ({bot.user.id})")

    await bot.change_presence(activity=discord.Activity(
        type=discord.ActivityType.watching, 
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
            logger.info(f"Detected hate speech")
            logger.info(f"User: {message.author} | Channel: {message.channel} | Score: {hate_score:.2f} | Message: {message.content}")
            await message.channel.send(f"⚠️ Hate speech detected in: {message.content}")

        if hate_score > 0.3 and hate_score <= settings["threshold"]:
            logger.info(f"Near miss (score={hate_score:.2f}): {message.content}")
  
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
                try: 
                    await message.add_reaction("⚠️")
                    await message.reply(
                        "**⚠️ This message may violate our server's hate speech policy.**\n"
                        "Please be respectful. Continued violations may result in moderation actions "
                        "including message removal or temporary mutes.\n"
                        "_If you believe this was a mistake, contact a moderator._"
                    )       
                    await message.delete()
                except discord.Forbidden:
                    logger.warning(f"Could not delete message from {message.author} in {message.guild.name}")
                except Exception as e:
                    logger.error(f"Failed to flag/delete message: {e}")
        
            # Notify moderators if a mod channel is set
            if settings["mod_channel"]:
                mod_channel = bot.get_channel(int(settings["mod_channel"]))
                if mod_channel:
                    embed = discord.Embed(
                        title="Hate Speech Detected",
                        description=f"**User:** {message.author.mention}\n"
                                   f"**Channel:** {message.channel.mention}\n"
                                   f"**Content:** {message.content}\n"
                                   f"**Score:** {hate_score:.2f}",
                        color=discord.Color.red()
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
    embed = discord.Embed(
        title="Hate Speech Detection Stats",
        color=discord.Color.blue()
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
        embed = discord.Embed(
            title="Server Configuration",
            description="Current settings for this server:",
            color=discord.Color.blue()
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
        color = discord.Color.red()
        result = "Detected hate speech"
    else:
        color = discord.Color.green()
        result = "No hate speech detected"
        
    embed = discord.Embed(
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