import pandas as pd

# Load the CSV files into dataframes for offensive images
keypoints_df_o = pd.read_csv("image_keypoints_o.csv")
metrics_df_o = pd.read_csv("image_metrics_o.csv")

# Remove file extensions from the 'Image Name' column
keypoints_df_o['Image Name'] = keypoints_df_o['Image Name'].str.replace(r'\.\w+$', '', regex=True)
metrics_df_o['Image Name'] = metrics_df_o['Image Name'].str.replace(r'\.\w+$', '', regex=True)

# Set 'Image Name' as the index for both dataframes
keypoints_df_o.set_index("Image Name", inplace=True)
metrics_df_o.set_index("Image Name", inplace=True)

# Merge the dataframes on the index (image_name)
merged_df_o = keypoints_df_o.merge(metrics_df_o, left_index=True, right_index=True)


# Load the CSV files into dataframes for nonoffensive images
keypoints_df_no = pd.read_csv("image_keypoints_no.csv")
metrics_df_no = pd.read_csv("image_metrics_no.csv")

# Remove file extensions from the 'Image Name' column
keypoints_df_no['Image Name'] = keypoints_df_no['Image Name'].str.replace(r'\.\w+$', '', regex=True)
metrics_df_no['Image Name'] = metrics_df_no['Image Name'].str.replace(r'\.\w+$', '', regex=True)

# Set 'Image Name' as the index for both dataframes
keypoints_df_no.set_index("Image Name", inplace=True)
metrics_df_no.set_index("Image Name", inplace=True)

# Merge the dataframes on the index (image_name)
merged_df_no = keypoints_df_no.merge(metrics_df_no, left_index=True, right_index=True)

# Add the 'offensive' column to the offensive images dataframe
merged_df_o['offensive'] = 'Yes'

# Add the 'offensive' column to the non-offensive images dataframe
merged_df_no['offensive'] = 'No'

# # Display the first few rows to verify
# print("Offensive DataFrame:")
# print(merged_df_o.head())

# print("\nNon-Offensive DataFrame:")
# print(merged_df_no.head())
# Check if column names are the same
columns_match = list(merged_df_o.columns) == list(merged_df_no.columns)

if columns_match:
    print("Column names are the same. Proceeding to merge the dataframes.")
else:
    print("Column names differ. Ensure consistency before merging.")
    print("Columns in merged_df_o:", merged_df_o.columns)
    print("Columns in merged_df_no:", merged_df_no.columns)

# Merge the dataframes if columns match
if columns_match:
    combined_df = pd.concat([merged_df_o, merged_df_no])
    
    # Display the combined dataframe
    print("Merged DataFrame:")
    print(combined_df.head())

    # Save the combined dataframe to a CSV file
    combined_df.to_csv("final_combined_image_data.csv", index=True)
else:
    print("Cannot merge dataframes as column names are different. Resolve discrepancies.")