
import json
import pandas as pd
import sys

def load_json_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]
    
def extract_response(resp):
    if isinstance(resp, dict):
        return resp.get("text", None)
    return None

def merge_reviews_with_metadata(reviews_path, metadata_path, output_csv_path):
    # Cargar los archivos JSONL
    reviews = load_json_lines(reviews_path)
    metadata = load_json_lines(metadata_path)

    # Convertir a DataFrame
    reviews_df = pd.DataFrame(reviews)
    metadata_df = pd.DataFrame(metadata).drop_duplicates(subset=['gmap_id'])

    reviews_df["response"] = reviews_df["resp"].apply(extract_response)

    #reviews_df.to_csv('reviews_df.csv', index=False, encoding='utf-8')

    # Hacer el merge por gmap_id
    merged_df = reviews_df.merge(metadata_df, left_on='gmap_id', right_on='gmap_id', how='inner', suffixes=('', '_metadata'))
    

    # Seleccionar y renombrar columnas
    final_df = merged_df[['name_metadata', 'text', 'response', 'rating', 'avg_rating', 'num_of_reviews']]
    final_df.columns = ['business_name', 'review', 'response', 'rating', 'avg_rating', 'num_of_reviews']


    for col in ['review', 'response']:
        final_df[col] = final_df[col].astype(str).str.replace('\n', '\\n')


    # Guardar como CSV
    final_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Archivo guardado como: {output_csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python merge_reviews_with_metadata.py <reviews.json> <metadata.json> <output.csv>")
        sys.exit(1)

    reviews_path = sys.argv[1]
    metadata_path = sys.argv[2]
    output_csv_path = sys.argv[3]

    merge_reviews_with_metadata(reviews_path, metadata_path, output_csv_path)
