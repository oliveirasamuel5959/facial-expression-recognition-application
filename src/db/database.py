import mysql.connector
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
from core.logging import setup_logging

setup_logging()
load_dotenv(override=True)

class EmotionDatabase:
    def __init__(self):
        self.connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_DATABASE")
        )
        self.cursor = self.connection.cursor()

    def add_emotion(self, emotion: str, acc: str) -> None:
        """Insert a new emotion record into the emotions table."""
        query = """
        INSERT INTO emotions (emotion, timestamp)
        VALUES (%s, %s)
        """
        values = (emotion, datetime.now())
        self.cursor.execute(query, values)
        self.connection.commit()
        logging.info(f"Emotion '{emotion}' with {acc} inserted successfully.")

    def close(self):
        """Close cursor and database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("[🔒] Database connection closed.")

# if __name__ == "__main__":
#     db = EmotionDatabase()
#     db.add_emotion("happy")
#     db.close()
