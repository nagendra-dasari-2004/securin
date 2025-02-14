from mitreattack.stix20 import MitreAttackData


def main():
    mitre_attack_data = MitreAttackData(r"enterprise-attack.json")

    tactics = mitre_attack_data.get_tactics(remove_revoked_deprecated=True)

    print(f"Retrieved {len(tactics)} ATT&CK tactics.")


def mainattack():
    mitre_attack_data = MitreAttackData("enterprise-attack.json")

    # Retrieve tactics
    tactics = mitre_attack_data.get_tactics(remove_revoked_deprecated=True)

    # Print the number of tactics retrieved
    print(f"Retrieved {len(tactics)} ATT&CK tactics:\n")

    # Loop through each tactic and print its name and description
    for tactic in tactics:
        print(f"- {tactic['name']} ({tactic['external_references'][0]['external_id']})")
        print(f"  Description: {tactic.get('description', 'No description available.')}")
        print()


if __name__ == "__main__":
    mainattack()
    main()
    
