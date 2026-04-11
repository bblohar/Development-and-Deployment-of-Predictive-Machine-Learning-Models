import mysql.connector
from faker import Faker
import random

fake = Faker('en_IN')

# --- RAILWAY CLOUD CREDENTIALS ---
db_config = {
    "host": "metro.proxy.rlwy.net",
    "user": "root",
    "password": "drcPdFLoorZOmRsOwUDKMTNkSTtfZenz",
    "database": "kalavati_db",
    "port": 34345
}

def seed_cloud_database_v2():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 1. DROP AND RECREATE TABLE WITH EXACT WORKBENCH COLUMNS
        print("Cleaning up old table structure...")
        cursor.execute("DROP TABLE IF EXISTS kalavati_advanced_bms_data")
        
        cursor.execute("""
            CREATE TABLE kalavati_advanced_bms_data (
                CustomerID INT PRIMARY KEY,
                Customer_Name VARCHAR(255),
                Location VARCHAR(100),
                Account_Age_Days INT,
                Monthly_Fee_INR INT,
                Industry VARCHAR(100),
                Feature_Usage_Score INT,
                Total_Users INT,
                Support_Tickets INT,
                Payment_Delay_Days INT,
                Last_Login_Days INT,
                Avg_Resolution_Time_Hrs FLOAT,
                NPS_Score INT,
                Is_Churn INT
            )
        """)
        
        # 2. GENERATE 1000 RECORDS WITH STRATEGIC CORRELATION
        cities = ['Mumbai', 'Pune', 'Shirpur', 'Ahmedabad', 'Surat', 'Nagpur', 'Nashik', 'Dhule', 'Indore']
        industries = ['Logistics', 'Healthcare', 'Retail', 'Finance', 'Tech']
        data = []

        print("Generating 1000 records matching your Workbench features...")

        for i in range(1304, 2304): # Starting from 1304 as seen in your image
            name = fake.name()
            location = random.choice(cities)
            industry = random.choice(industries)
            age_days = random.randint(30, 1500)
            fee = random.randint(5000, 50000)
            users = random.randint(1, 50)
            
            # --- THE ML LOGIC (Strategic Seeding) ---
            is_churn = random.random() < 0.25  # 25% Churn Rate
            
            if is_churn:
                usage = random.randint(5, 45)
                tickets = random.randint(8, 25)
                delay = random.randint(15, 30)
                last_login = random.randint(20, 60)
                resolution = round(random.uniform(40.0, 100.0), 2) # Slower resolution
                nps = random.randint(1, 4)
                churn = 1
            else:
                usage = random.randint(60, 100)
                tickets = random.randint(0, 5)
                delay = random.randint(0, 7)
                last_login = random.randint(0, 5)
                resolution = round(random.uniform(2.0, 20.0), 2) # Faster resolution
                nps = random.randint(7, 10)
                churn = 0

            data.append((i, name, location, age_days, fee, industry, usage, users, tickets, delay, last_login, resolution, nps, churn))

        # 3. BULK INSERT
        sql = """INSERT INTO kalavati_advanced_bms_data 
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        
        cursor.executemany(sql, data)
        conn.commit()
        
        print(f"✅ Successfully seeded {len(data)} records to Railway Cloud!")
        conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    seed_cloud_database_v2()