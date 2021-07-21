import boto3
client = boto3.client(service_name='comprehendmedical', region_name='us-east-2')



def identify_Sysmptoms(sentence):
    symptoms = []
    symptom = ''
    site = ''
    result = client.detect_entities(Text=sentence )
    entities = result['Entities'];
    for entity in entities:
        if entity['Category'] == 'MEDICAL_CONDITION':
            symptom = entity['Text']

            if "Attributes" in entity:
                for att in entity['Attributes']:
                    if 'RelationshipType' in att:
                        if att['RelationshipType'] == "SYSTEM_ORGAN_SITE":
                            site = att['Text'] + ' '+symptom
                            symptoms.append(site)
                            site = ''
            else:
                symptoms.append(symptom)
                

    print(symptoms)


            

        


def main():
    value = input("Please enter a string:\n")
    if value != '':
        identify_Sysmptoms(value)

if __name__ == "__main__":
    main()