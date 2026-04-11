import mysql.connector
from faker import Faker
import random

# Initialize Faker with Indian locale for realistic names/cities
fake = Faker('en_IN')

# Connect to your MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Spyrob@2909",
    database="kalavati_db"
)
cursor = conn.cursor()

# 1. Fetch all existing CustomerIDs
cursor.execute("SELECT CustomerID FROM kalavati_advanced_bms_data")
ids = [row[0] for row in cursor.fetchall()]

print(f"Updating {len(ids)} records...")

# 2. Update each record with a random Name and Location
# Common Indian cities for your location feature
cities = ['Mumbai', 'Pune', 'Shirpur', 'Ahmedabad', 'Surat', 'Nagpur', 'Nashik', 'Dhule', 'Indore']

for cid in ids:
    name = fake.name()
    loc = random.choice(cities)
    
    sql = "UPDATE kalavati_advanced_bms_data SET Customer_Name = %s, Location = %s WHERE CustomerID = %s"
    cursor.execute(sql, (name, loc, cid))

conn.commit()
print("✅ 1000+ Records successfully updated with Names and Locations!")
conn.close()