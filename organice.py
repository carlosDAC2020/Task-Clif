import json
from collections import OrderedDict

# Step 1: Read the JSON file
with open("general_words_representation.json", "r") as file:
    data = json.load(file)

# Step 2: Sort outer dict and each inner dict by keys
sorted_data = {
    outer_key: dict(sorted(inner_dict.items(), key=lambda item: item[1], reverse=True))
    for outer_key, inner_dict in data.items()
}

# # Step 3: Save the sorted data to a new file
# with open("general_words_representation1.json", "w") as file:
#     json.dump(sorted_data, file, indent=4)

# %%


import json
from collections import Counter

with open("general_words_representation.json", "r") as file:
    data = json.load(file)


def top_words_from_keys(dict_general_words, keys, n_top=3, exclude_words=None):
    # Load data
    data = dict_general_words

    total_counter = Counter()

    for key in keys:
        if key in data:
            inner_dict = data[key]
            # Exclude specific words if requested
            if exclude_words:
                inner_dict = {k: v for k, v in inner_dict.items() if k not in exclude_words}
            total_counter.update(inner_dict)
        else:
            print(f"Warning: Key '{key}' not found in the data.")

    # Get the top N words (no frequencies)
    list_obtenida=[word for word, _ in total_counter.most_common(n_top)]
    list_obtenida.extend(keys)
    # return [word for word, _ in total_counter.most_common(n_top)]
    return list_obtenida

# Example usage
top_words = top_words_from_keys(data, keys=["hirschsprung", "africa"], n_top=3)
print(top_words)

# %%
