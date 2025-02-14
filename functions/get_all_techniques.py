from mitreattack.stix20 import MitreAttackData


def main():
    mitre_attack_data = MitreAttackData("enterprise-attack.json")

    techniques = mitre_attack_data.get_techniques(remove_revoked_deprecated=True)

    print(f"Retrieved {len(techniques)} ATT&CK techniques.")

def mainattack():
    mitre_attack_data = MitreAttackData("enterprise-attack.json")

    # Retrieve all techniques
    techniques = mitre_attack_data.get_techniques(remove_revoked_deprecated=True)

    # Print all techniques
    print("All ATT&CK Techniques:\n")
    for technique in techniques:
        print(f"- {technique['name']} ({mitre_attack_data.get_attack_id(technique['id'])})")



if __name__ == "__main__":
    main()
    mainattack()
