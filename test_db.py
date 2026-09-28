import os; from sqlalchemy import create_engine; engine = create_engine("postgresql://postgres:postgres@127.0.0.1:5432/aimonitor"); conn = engine.connect(); print("success"); conn.close()
