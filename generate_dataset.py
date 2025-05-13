
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

def get_food_category_set():
    food_categories = [
        "restaurant", "american restaurant", "pizza restaurant", "breakfast restaurant", "fast food restaurant",
        "cafe", "takeout restaurant", "sandwich shop", "bar", "bar & grill", "coffee shop", "ice cream shop",
        "bakery", "caterer", "italian restaurant", "chinese restaurant", "deli", "bagel shop", "hamburger restaurant",
        "donut shop", "mexican restaurant", "family restaurant", "lunch restaurant", "delivery restaurant",
        "asian restaurant", "diner", "seafood restaurant", "fine dining restaurant", "snack bar", "wine bar",
        "brunch restaurant", "vegetarian restaurant", "taco restaurant", "chicken wings restaurant",
        "barbecue restaurant", "espresso bar", "pub", "cocktail bar", "beer store", "gourmet grocery store",
        "brewery", "brewpub", "steak house", "salad shop", "sushi restaurant", "vegan restaurant",
        "new american restaurant", "thai restaurant", "indian restaurant", "chocolate shop", "cheese shop",
        "dessert shop", "gastropub", "bistro", "juice shop", "frozen yogurt shop", "buffet restaurant",
        "organic restaurant", "soup restaurant", "ramen restaurant", "hot dog restaurant", "tapas restaurant",
        "middle eastern restaurant", "french restaurant", "mediterranean restaurant", "burrito restaurant",
        "latin american restaurant", "vietnamese restaurant", "pancake restaurant", "chicken restaurant",
        "pho restaurant", "greek restaurant", "gluten-free restaurant"
    ]
    return set(cat.lower() for cat in food_categories)

def is_food_business(category_list, food_categories):
    if not category_list:
        return False
    return any(cat.lower() in food_categories for cat in category_list if isinstance(cat, str))

def filter_food_businesses(metadata_df):
    food_categories = get_food_category_set()
    return metadata_df[metadata_df['category'].apply(lambda cats: is_food_business(cats, food_categories))]

def merge_reviews_with_metadata(reviews_path, metadata_path, output_csv_path):
    reviews = load_json_lines(reviews_path)
    metadata = load_json_lines(metadata_path)

    reviews_df = pd.DataFrame(reviews)
    metadata_df = pd.DataFrame(metadata).drop_duplicates(subset=['gmap_id'])
    metadata_df = filter_food_businesses(metadata_df)

    # Filter reviews
    reviews_df["response"] = reviews_df["resp"].apply(extract_response)
    reviews_df = reviews_df[reviews_df['text'].notna() & (reviews_df['text'].str.strip() != "")]
    review_counts = reviews_df.groupby('gmap_id').size().reset_index(name='review_count')
    valid_gmap_ids = review_counts[review_counts['review_count'] >= 30]['gmap_id']

    reviews_df = reviews_df[reviews_df['gmap_id'].isin(valid_gmap_ids)]
    metadata_df = metadata_df[metadata_df['gmap_id'].isin(valid_gmap_ids)]

    # Final merge
    merged_df = reviews_df.merge(metadata_df, on='gmap_id', how='inner', suffixes=('', '_metadata'))

    final_df = merged_df[['name_metadata', 'text', 'response', 'rating', 'avg_rating', 'num_of_reviews']]
    final_df.columns = ['business_name', 'review', 'response', 'rating', 'avg_rating', 'num_of_reviews']

    for col in ['review', 'response']:
        final_df[col] = final_df[col].astype(str).str.replace('\n', '\\n').str.replace('\r', '')

    final_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Archivo guardado como: {output_csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python generate_dataset.py <reviews.json> <metadata.json> <output.csv>")
        sys.exit(1)

    reviews_path = sys.argv[1]
    metadata_path = sys.argv[2]
    output_csv_path = sys.argv[3]

    merge_reviews_with_metadata(reviews_path, metadata_path, output_csv_path)
