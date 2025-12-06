import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Load CSV file from the same folder
csv_file = Path(__file__).parent / "s=0.1.csv"
df = pd.read_csv(csv_file)

# Reshape into long format for seaborn
df_long = df.melt(var_name="Category", value_name="Value")

# Create the boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x="Category", y="Value", data=df_long, showfliers=False)

# Set y-axis limits from 0 to 3*10^-6
plt.ylim(0, 3e-6)

# Labels and title
plt.xlabel("Initial Tumor Size")
plt.ylabel("Quasi-Resistant Fraction")
plt.title("Quasi-Resistant Cell Fraction at Tumor Detection (s=0.1)")

plt.tight_layout()
plt.show()