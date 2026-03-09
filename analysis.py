import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



sns.set_style("whitegrid")



df = pd.read_csv("data/Global YouTube Statistics.csv", encoding="latin1")



df["category"] = df["category"].fillna("Unknown")
df["Country"] = df["Country"].fillna("Unknown")
df["created_year"] = df["created_year"].fillna(df["created_year"].median())

df = df[df["subscribers"] > 0]

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistics:")
print(df.describe())

df["engagement_rate"] = df["video views"] / df["subscribers"]

df["channel_age"] = 2023 - df["created_year"]

df["log_subscribers"] = np.log1p(df["subscribers"])



top_channels = df.sort_values(by="subscribers", ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(data=top_channels, x="subscribers", y="Youtuber")
plt.title("Top 10 YouTube Channels by Subscribers")
plt.xlabel("Subscribers")
plt.ylabel("Channel")
plt.show()



category_counts = df["category"].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=category_counts.values, y=category_counts.index)
plt.title("Most Popular YouTube Channel Categories")
plt.xlabel("Number of Channels")
plt.ylabel("Category")
plt.show()



country_counts = df["Country"].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=country_counts.values, y=country_counts.index)
plt.title("Top Countries by Number of YouTube Channels")
plt.xlabel("Number of Channels")
plt.ylabel("Country")
plt.show()



top_engagement = df.sort_values(by="engagement_rate", ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(data=top_engagement, x="engagement_rate", y="Youtuber")
plt.title("Top Channels by Engagement Rate")
plt.xlabel("Engagement Rate")
plt.ylabel("Channel")
plt.show()



corr = df.corr(numeric_only=True)

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()



plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="uploads", y="video views")
plt.title("Uploads vs Video Views")
plt.xlabel("Number of Uploads")
plt.ylabel("Video Views")
plt.show()



print("\nTop 5 Channels by Video Views:")
print(df.sort_values(by="video views", ascending=False).head(5))


plt.figure(figsize=(8,6))
sns.histplot(df["engagement_rate"], bins=50)
plt.title("Distribution of Engagement Rate")
plt.xlabel("Engagement Rate")
plt.show()



plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="subscribers", y="video views")
plt.title("Subscribers vs Video Views")
plt.xlabel("Subscribers")
plt.ylabel("Video Views")
plt.show()



category_engagement = df.groupby("category")["engagement_rate"].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=category_engagement.values, y=category_engagement.index)
plt.title("Average Engagement Rate by Category")
plt.xlabel("Average Engagement")
plt.ylabel("Category")
plt.show()



plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="channel_age", y="subscribers")
plt.title("Channel Age vs Subscribers")
plt.xlabel("Channel Age (Years)")
plt.ylabel("Subscribers")
plt.show()



top_earnings = df.sort_values(by="highest_yearly_earnings", ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(data=top_earnings, x="highest_yearly_earnings", y="Youtuber")
plt.title("Top YouTube Channels by Estimated Yearly Earnings")
plt.xlabel("Yearly Earnings")
plt.ylabel("Channel")
plt.show()



plt.figure(figsize=(8,6))
sns.histplot(df["log_subscribers"], bins=40)
plt.title("Log Distribution of Subscribers")
plt.xlabel("Log Subscribers")
plt.show()



top_views = df.sort_values(by="video views", ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(data=top_views, x="video views", y="Youtuber")
plt.title("Top Channels by Video Views")
plt.xlabel("Video Views")
plt.ylabel("Channel")
plt.show()

df.to_csv("youtube_cleaned.csv", index=False)