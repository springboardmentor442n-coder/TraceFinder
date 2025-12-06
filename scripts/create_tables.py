import sys
import os

# Add project root to path so we can import Backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Backend import database, models

def create_tables():
    print("Creating tables...")
    try:
        models.Base.metadata.create_all(bind=database.engine)
        print("✅ Tables created successfully.")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")

if __name__ == "__main__":
    create_tables()
