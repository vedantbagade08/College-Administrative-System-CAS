import pymysql
import time
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config from app.py
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'cas_db',
    'charset': 'utf8mb4',
    'autocommit': True
}

def test_connect_no_db():
    logger.info("Testing connection WITHOUT database...")
    config = MYSQL_CONFIG.copy()
    config.pop('database', None)
    try:
        conn = pymysql.connect(**config)
        logger.info("SUCCESS: Connected to MySQL server.")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"FAIL: Could not connect to MySQL server: {e}")
        return False

def test_connect_with_db():
    logger.info("Testing connection WITH database...")
    try:
        conn = pymysql.connect(**MYSQL_CONFIG)
        logger.info("SUCCESS: Connected to 'cas_db'.")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"FAIL: Could not connect to 'cas_db': {e}")
        return False

def worker_connect(ident):
    try:
        conn = pymysql.connect(**MYSQL_CONFIG)
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        # logger.info(f"Thread {ident}: Success")
    except Exception as e:
        logger.error(f"Thread {ident}: FAIL - {e}")

def test_concurrency():
    logger.info("Testing CONCURRENT connections (20 threads)...")
    threads = []
    for i in range(20):
        t = threading.Thread(target=worker_connect, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    logger.info("Concurrent test finished.")

if __name__ == "__main__":
    if test_connect_no_db():
        if test_connect_with_db():
            test_concurrency()
        else:
            logger.warning("Skipping concurrency test because DB connection failed.")
    else:
        logger.error("Skipping remaining tests because Server connection failed.")
