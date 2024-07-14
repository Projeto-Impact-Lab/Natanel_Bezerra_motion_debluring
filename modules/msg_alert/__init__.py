import telebot
import json
from config import Config

class MsgAlert(Config):

    api_key = None
    bot = None
    myId = None

    def __init__(self,) -> None:

        self.load_config()
        self.bot = telebot.TeleBot(self.api_key)


    def load_config(self):
        try:
            with open(self.config_telegram, "r") as f:
                config_data = json.load(f)
                self.api_key = config_data.get("api_key")
                self.myId = config_data.get("myId")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            exit(1)

    def send_msg(self, msg="Hello", image_path=None):
        try:
            if image_path:
                # Upload the image
                with open(image_path, 'rb') as photo:
                    self.bot.send_photo(self.myId, photo, caption=msg)
            else:
                # Send a regular text message
                self.bot.send_message(self.myId, msg)
            return True
        except Exception as e:
            print(f"Error sending message: {e}")
            return False