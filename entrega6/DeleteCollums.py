import csv

filename = 'entrega6/data/drought.forecasting_dataset'

# Open the Excel file and create a CSV reader
with open(f'{filename}.csv', 'r') as f:
    reader = csv.reader(f)
    
    # Create a new CSV file to store the modified data
    with open(f'{filename}_DROP.csv', 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        
        # Loop through the rows in the original file
        for row in reader:
            # Keep the "Date" and "QV2M" columns
            values = [row[0], row[7]]
            
            # Write the "Date" and "QV2M" values to the new CSV file
            writer.writerow(values)

print("Modified file created successfully!")