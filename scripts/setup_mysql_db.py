import sqlalchemy
from sqlalchemy import create_engine, text

# Connect to default 'mysql' database to create the new db
# User: root, Password: (empty), Host: localhost, Port: 3306
DATABASE_URL = "mysql+pymysql://root:@localhost:3306/mysql"

def create_database():
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            # Create database if not exists
            conn.execute(text("CREATE DATABASE IF NOT EXISTS document_founder"))
            print("✅ Database 'document_founder' created or already exists.")
    except Exception as e:
        print(f"❌ Error creating database: {e}")

if __name__ == "__main__":
    create_database()
