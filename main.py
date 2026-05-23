import pandas as pd
from data_profiling import ProfileReport

base_path = r'D:\Kaggle\F1 Pit stop predictions\Data'

df = pd.read_csv(base_path + r'\train.csv')

profile = ProfileReport(df, title="Profiling Report")

# Save report
profile.to_file("profiling_report.html")