import json

def convert_txt_to_json(input_filepath, output_filepath):
    sentiment_dict = {}

    try:
        with open(input_filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()

            for line in lines[1:]:
                if not line.strip():
                    continue
                
                columns = line.strip('\n').split('\t')
                
                if len(columns) >= 5:
                    try:
                        pos_score = float(columns[2].strip())
                        neg_score = float(columns[3].strip())
                        
                        synset_terms = columns[4].strip().split()
                        for term in synset_terms:
                            sentiment_dict[term] = {
                                'posScore': pos_score,
                                'negScore': neg_score
                            }
                    except ValueError:
                        print(f"Error: Invalid number format in line: {line.strip()}")
                        continue

        with open(output_filepath, 'w', encoding='utf-8') as json_file:
            json.dump(sentiment_dict, json_file, ensure_ascii=False, indent=4)
            
        print(f"Data saved at: {output_filepath}")

    except FileNotFoundError:
        print(f"Error: Not found file {input_filepath}")
    except Exception as e:
        print(f"Error: {e}")

input_file = 'Sentiwordnet/VietSentiWordnet_ver1.0.txt'  
output_file = 'Sentiwordnet/vietnamese_sentiwordnet.json'

convert_txt_to_json(input_file, output_file)