import sqlite3
import os
from werkzeug.security import generate_password_hash

# Database initialization script

def init_db():
    # Create database directory if it doesn't exist
    os.makedirs('instance', exist_ok=True)
    
    # Connect to SQLite database (will create it if it doesn't exist)
    conn = sqlite3.connect('instance/facesnap.sqlite')
    cursor = conn.cursor()
    
    # Read schema file
    with open('schema.sql', 'r') as f:
        schema = f.read()
    
    # Execute schema commands
    cursor.executescript(schema)
    
    # Check if admin user exists
    cursor.execute("SELECT COUNT(*) FROM admins")
    admin_count = cursor.fetchone()[0]
    
    # Create default admin user if none exists
    if admin_count == 0:
        default_username = 'admin'
        default_password = 'admin123'  # This should be changed after first login
        password_hash = generate_password_hash(default_password)
        
        cursor.execute(
            "INSERT INTO admins (username, password_hash, email) VALUES (?, ?, ?)",
            (default_username, password_hash, 'admin@facesnap.local')
        )
        print(f"Created default admin user: {default_username} (password: {default_password})")
        print("IMPORTANT: Please change this password after first login!")
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print("Database initialized successfully!")

if __name__ == "__main__":
    init_db()