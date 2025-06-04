import pandas as pd

# Load the CSV files into dataframes
keypoints_df = pd.read_csv("image_keypoints_test.csv")
metrics_df = pd.read_csv("image_metrics_test.csv")

# Remove file extensions from the 'Image Name' column
keypoints_df['Image Name'] = keypoints_df['Image Name'].str.replace(r'\.\w+$', '', regex=True)
metrics_df['Image Name'] = metrics_df['Image Name'].str.replace(r'\.\w+$', '', regex=True)

# Set 'Image Name' as the index for both dataframes
keypoints_df.set_index("Image Name", inplace=True)
metrics_df.set_index("Image Name", inplace=True)

# Merge the dataframes on the index (image_name)
merged_df = keypoints_df.merge(metrics_df, left_index=True, right_index=True)

# Save the combined test dataframe to a CSV file without the 'offensive' column
merged_df.to_csv("final_test_image_data1.csv", index=True)

print("Test dataset saved as 'final_test_image_data1.csv'")
