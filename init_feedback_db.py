import pymysql
import logging

MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'cas_db',
    'charset': 'utf8mb4',
    'autocommit': True
}

def init_db():
    try:
        conn = pymysql.connect(**MYSQL_CONFIG)
        with conn.cursor() as cursor:
            # Create feedback_forms table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_forms (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    structure_json TEXT NOT NULL,
                    status ENUM('draft', 'published') DEFAULT 'draft',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create feedback_responses table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_responses (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    form_id INT,
                    user_email VARCHAR(255),
                    response_json TEXT NOT NULL,
                    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (form_id) REFERENCES feedback_forms(id) ON DELETE CASCADE
                )
            """)
            print("Feedback tables initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    init_db()
