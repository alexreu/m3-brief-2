from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

database_url = os.getenv('DATABASE_URL')

if database_url is None:
    raise ValueError("DATABASE_URL environment variable is not set.")

engine = create_engine(database_url)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
