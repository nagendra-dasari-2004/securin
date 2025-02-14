import csv

def print_technique_id(technique_id):
    print(f"TechniqueID: {technique_id}")

# Reload the merged dataset to ensure correct processing
df_merged = pd.read_csv(merged_file_path)

# Print each row with respective labels
for index, row in df_merged.iterrows():
    if row["Type"] == "Tactic":
        print(f"Index: {index}, Tactic ID: {row['ID']}, Tactic Name: {row['Name']}, Tactic Description: {row['Description']}")
    else:
        print(f"Index: {index}, Technique ID: {row['ID']}, Technique Name: {row['Name']}, Technique Description: {row['Description']}")


def print_technique_name(technique_name):
    print(f"TechniqueName: {technique_name}")

def print_technique_description(technique_description):
    print(f"TechniqueDescription: {technique_description}")

def print_all(technique_id, technique_name, technique_description):
    print_technique_id(technique_id)
    print_technique_name(technique_name)
    print_technique_description(technique_description)

def print_tactic_id(tactic_id):
    print(f"TacticID: {tactic_id}")

def print_tactic_name(tactic_name):
    print(f"TacticName: {tactic_name}")

def print_tactic_description(tactic_description):
    print(f"TacticDescription: {tactic_description}")

def print_all_tactics(tactic_id, tactic_name, tactic_description):
    print_tactic_id(tactic_id)
    print_tactic_name(tactic_name)
    print_tactic_description(tactic_description)

def list_techniques(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        techniques_started = False
        
        for row in reader:
            if row and row[0] == "TechniqueID":  # Check for the start of the Techniques section
                techniques_started = True
                continue  # Skip the header row
            
            if techniques_started:
                technique_id = row[0]
                technique_name = row[1]
                technique_description = row[2]
                print_all(technique_id, technique_name, technique_description)

def list_tactics(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        tactics_started = False
        
        for row in reader:
            if row and row[0] == "TacticID":  # Check for the start of the Tactics section
                tactics_started = True
                continue  # Skip the header row
            
            if tactics_started and row:  # Ensure the row is not empty
                tactic_id = row[0]
                tactic_name = row[1]
                tactic_description = row[2]
                print_all_tactics(tactic_id, tactic_name, tactic_description)

# Call the function with the path to the data.csv file
list_techniques('techniques.csv')
list_tactics('tactics.csv')
